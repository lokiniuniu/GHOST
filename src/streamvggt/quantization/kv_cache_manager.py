"""
QVGInfiniteVGGTKVCacheManager - QVG Paper Section 4 + InfiniteVGGT Compatibility

End-to-end quantised KV Cache manager that:
  1. Splits each layer's KV into anchor tokens (first-frame) and candidate tokens.
  2. Applies QVG-Pro (T=4, B=16, INT4) to anchor tokens  – permanent, high-fidelity.
  3. Applies QVG standard  (T=1, B=64, INT2/4) to candidate tokens – rolling, high-compression.
  4. Stores all parameters (quantised ints, scales, centroids, assignments) on GPU.
  5. On read, dequantises and returns BF16 tensors compatible with FlashAttention.

Diversity-aware Rolling Memory compatibility (InfiniteVGGT Sec 3):
  - Anchor tokens (num_anchor_tokens first along seq dim) → QVG-Pro, never evicted.
  - Candidate tokens (remaining) → QVG standard, subject to TopK eviction.
  - Quantisation happens AFTER eviction (caller evicts first, then calls update_layer_cache).
  - Budget allocation logic in Aggregator._calculate_dynamic_budgets is untouched.

Memory composition target (QVG Fig 7a):
  quantised values ≥ 65%, overhead ≤ 35% of total bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .progressive_residual_quantization import (
    ProgressiveResidualQuantization,
    PRQQuantizedCache,
)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class QVGConfig:
    """
    Unified configuration for QVG KV Cache quantisation.

    Quick presets:
        QVGConfig.standard_int2()   → 6.94×–7.05× compression
        QVGConfig.standard_int4()   → 3.72×–3.75× compression
        QVGConfig.pro_int2()        → 4.97×–5.20× compression
        QVGConfig.pro_int4()        → 3.05×–3.15× compression
    """
    enabled: bool = True

    # ── Candidate token quantisation (QVG standard)
    candidate_bits: int = 2          # INT2 for maximum compression
    candidate_stages: int = 1        # T=1
    candidate_group_size: int = 64   # B=64

    # ── Anchor token quantisation (QVG-Pro, first-frame tokens)
    anchor_bits: int = 4             # INT4 for precision
    anchor_stages: int = 2           # T=2 [Optimization] Reduced from 4 for faster encode
    anchor_group_size: int = 16      # B=16

    # ── Common
    num_centroids: int = 256         # fixed C=256 (Sec 5.1), uint8-compatible
    kmeans_max_iters: int = 5        # [Optimization] Reduced from 10 for faster encode
    pre_rope_key: bool = True       # Pre-RoPE key caching (Sec 5.1)

    # ── Chunk-based quantisation (encode/decode every chunk_size frames instead of every frame)
    chunk_size: int = 16            # [Optimization] Increased from 8 to reduce encode frequency

    # ── Anchor full precision (skip quantisation for anchor tokens, keep BF16)
    anchor_full_precision: bool = True

    # ── Special tokens (camera + register) full precision – always BF16, never quantised
    num_special_tokens: int = 5  # 1 camera + 4 register (patch_start_idx)

    @classmethod
    def standard_int2(cls) -> "QVGConfig":
        return cls(candidate_bits=2, candidate_stages=1, candidate_group_size=64,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16)

    @classmethod
    def standard_int4(cls) -> "QVGConfig":
        return cls(candidate_bits=4, candidate_stages=1, candidate_group_size=64,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16)

    @classmethod
    def pro_int2(cls) -> "QVGConfig":
        return cls(candidate_bits=2, candidate_stages=4, candidate_group_size=16,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16)

    @classmethod
    def pro_int4(cls) -> "QVGConfig":
        return cls(candidate_bits=4, candidate_stages=4, candidate_group_size=16,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16)

    @classmethod
    def pro_int4_k512(cls) -> "QVGConfig":
        return cls(candidate_bits=4, candidate_stages=4, candidate_group_size=16,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16,
                   num_centroids=512)

    @classmethod
    def pro_int4_k1024(cls) -> "QVGConfig":
        return cls(candidate_bits=4, candidate_stages=4, candidate_group_size=16,
                   anchor_bits=4,   anchor_stages=2,   anchor_group_size=16,
                   num_centroids=1024)


# ─── Per-layer quantised cache state ─────────────────────────────────────────

@dataclass
class _LayerCacheState:
    """Internal state for one transformer layer."""
    # [Special tokens] Camera + register KV – always BF16, never quantised
    special_anchor_k_bf16: Optional[torch.Tensor] = None  # [B, H, min(5,anc), D]
    special_anchor_v_bf16: Optional[torch.Tensor] = None
    special_cand_k_bf16: Optional[torch.Tensor] = None   # [B, H, min(5,cnd), D]
    special_cand_v_bf16: Optional[torch.Tensor] = None

    # Anchor tokens (QVG-Pro): patch tokens of first frame
    anchor_k_qcache: Optional[PRQQuantizedCache] = None
    anchor_v_qcache: Optional[PRQQuantizedCache] = None
    # [anchor_full_precision] BF16 buffer when anchor patch is stored in full precision
    anchor_k_bf16_buffer: Optional[torch.Tensor] = None  # [B, H, anc-5, D]
    anchor_v_bf16_buffer: Optional[torch.Tensor] = None

    # Candidate tokens (QVG standard): rolling, subject to eviction (patch tokens only)
    cand_k_qcache: Optional[PRQQuantizedCache] = None
    cand_v_qcache: Optional[PRQQuantizedCache] = None

    # [Chunk mode, Delta buffer] BF16 buffer for NEW tokens only (since last chunk boundary)
    cand_k_delta_buffer: Optional[torch.Tensor] = None  # [B, H, N_delta, D]
    cand_v_delta_buffer: Optional[torch.Tensor] = None
    cand_len_at_last_boundary: int = 0  # candidate seq length at last encode

    # [First chunk only] Full BF16 buffer before first encode (no quantized cache yet)
    k_bf16_buffer: Optional[torch.Tensor] = None  # [B, H, N, D]
    v_bf16_buffer: Optional[torch.Tensor] = None

    # Streaming centroid warm-start buffers (one per PRQ stage, per K/V split)
    anchor_k_prev_centroids: Optional[List[Optional[torch.Tensor]]] = None
    anchor_v_prev_centroids: Optional[List[Optional[torch.Tensor]]] = None
    cand_k_prev_centroids:   Optional[List[Optional[torch.Tensor]]] = None
    cand_v_prev_centroids:   Optional[List[Optional[torch.Tensor]]] = None

    # How many tokens are anchors in this layer
    num_anchor_tokens: int = 0

    def nbytes(self) -> int:
        total = 0
        for qc in (self.anchor_k_qcache, self.anchor_v_qcache,
                   self.cand_k_qcache,   self.cand_v_qcache):
            if qc is not None:
                total += qc.nbytes()
        for buf in (
            self.special_anchor_k_bf16, self.special_anchor_v_bf16,
            self.special_cand_k_bf16, self.special_cand_v_bf16,
            self.anchor_k_bf16_buffer, self.anchor_v_bf16_buffer,
        ):
            if buf is not None:
                total += buf.numel() * buf.element_size()
        return total


# ─── Main cache manager ───────────────────────────────────────────────────────

class QVGInfiniteVGGTKVCacheManager:
    """
    QVG KV Cache Manager for InfiniteVGGT streaming 3D reconstruction.

    Usage inside Aggregator.forward (use_cache=True path):

        manager = QVGInfiniteVGGTKVCacheManager(config, num_layers)

        # Inside the global-attention loop for layer `layer_idx`:
        past_kv = manager.get_past_key_values_for_layer(layer_idx, device, dtype)
        ... run Block.forward(use_cache=True, past_key_values=past_kv) ...
        # → returns new_kv = (k_full, v_full) after eviction
        manager.update_layer_cache(layer_idx, new_kv[0], new_kv[1], num_anchor)

        # At end of inference, report memory:
        print(manager.total_nbytes() / 1024**3, "GB")
    """

    def __init__(
        self,
        config: QVGConfig,
        num_layers: int,
        num_special_tokens: int = 5,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_special_tokens = num_special_tokens or config.num_special_tokens

        # Build per-configuration PRQ instances (reused across layers/heads)
        self._prq_anchor = ProgressiveResidualQuantization(
            num_stages=config.anchor_stages,
            group_size=config.anchor_group_size,
            bits=config.anchor_bits,
            num_centroids=config.num_centroids,
            kmeans_max_iters=config.kmeans_max_iters,
        )
        self._prq_cand = ProgressiveResidualQuantization(
            num_stages=config.candidate_stages,
            group_size=config.candidate_group_size,
            bits=config.candidate_bits,
            num_centroids=config.num_centroids,
            kmeans_max_iters=config.kmeans_max_iters,
        )

        # Per-layer state
        self._states: List[_LayerCacheState] = [
            _LayerCacheState() for _ in range(num_layers)
        ]

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_past_key_values_for_layer(
        self,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        timing_dict: Optional[dict] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dequantise and return (k, v) for layer_idx, or None if empty.

        [Chunk mode, Delta buffer] past = concat(decode(anchor), decode(cand_quantized), delta_buffer).
        If no quantized cache (first chunk), return None.

        Called BEFORE the attention Block forward pass each frame.
        FlashAttention compatible: returns plain BF16 tensors.

        Returns:
            (k, v) each [B, H, N_past, D]  or  None
        """
        state = self._states[layer_idx]
        # [First chunk] Full buffer when no quantized cache yet
        if state.k_bf16_buffer is not None:
            return (
                state.k_bf16_buffer.to(device, dtype),
                state.v_bf16_buffer.to(device, dtype),
            )
        n_spec = self.num_special_tokens
        has_anchor = (
            state.special_anchor_k_bf16 is not None
            or state.anchor_k_qcache is not None
            or state.anchor_k_bf16_buffer is not None
        )
        has_cand = (
            state.special_cand_k_bf16 is not None
            or state.cand_k_qcache is not None
            or state.cand_k_delta_buffer is not None
        )
        if not has_anchor and not has_cand:
            return None

        k_parts, v_parts = [], []

        # Special anchor (camera + register) – always BF16
        if state.special_anchor_k_bf16 is not None:
            k_parts.append(state.special_anchor_k_bf16.to(device, dtype))
            v_parts.append(state.special_anchor_v_bf16.to(device, dtype))

        # Anchor patch tokens
        if state.anchor_k_bf16_buffer is not None:
            k_parts.append(state.anchor_k_bf16_buffer.to(device, dtype))
            v_parts.append(state.anchor_v_bf16_buffer.to(device, dtype))
        elif state.anchor_k_qcache is not None:
            k_parts.append(self._prq_anchor.decode(state.anchor_k_qcache, timing_dict=timing_dict).to(device, dtype))
            v_parts.append(self._prq_anchor.decode(state.anchor_v_qcache, timing_dict=timing_dict).to(device, dtype))

        # Special candidate (camera + register) – always BF16
        if state.special_cand_k_bf16 is not None:
            k_parts.append(state.special_cand_k_bf16.to(device, dtype))
            v_parts.append(state.special_cand_v_bf16.to(device, dtype))

        # Candidate patch tokens
        if state.cand_k_qcache is not None:
            k_parts.append(self._prq_cand.decode(state.cand_k_qcache, timing_dict=timing_dict).to(device, dtype))
            v_parts.append(self._prq_cand.decode(state.cand_v_qcache, timing_dict=timing_dict).to(device, dtype))

        if state.cand_k_delta_buffer is not None:
            k_parts.append(state.cand_k_delta_buffer.to(device, dtype))
            v_parts.append(state.cand_v_delta_buffer.to(device, dtype))

        k = torch.cat(k_parts, dim=2) if len(k_parts) > 1 else k_parts[0]
        v = torch.cat(v_parts, dim=2) if len(v_parts) > 1 else v_parts[0]
        return (k, v)

    def update_layer_cache(
        self,
        layer_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        num_anchor_tokens: int,
        frame_idx: int = 0,
        timing_dict: Optional[dict] = None,
    ) -> None:
        """
        Quantise and store (k_full, v_full) after eviction for layer_idx.

        k_full / v_full: [B, H, N_after_eviction, D]  BF16
        num_anchor_tokens: first `num_anchor_tokens` seq positions are anchors.
        frame_idx: current frame index (for chunk boundary detection).

        [Chunk mode] Only encode at chunk boundaries (frame_idx+1) % chunk_size == 0.
        Within chunk, store in BF16 buffer to avoid encode overhead.

        Called AFTER the attention Block forward (which already did TopK eviction).

        Anchor split  → QVG-Pro  (T=4, B=16, INT4)
        Candidate split → QVG standard (T=1, B=64, INT2/INT4)
        """
        state = self._states[layer_idx]
        state.num_anchor_tokens = num_anchor_tokens

        chunk_size = self.config.chunk_size
        at_chunk_boundary = chunk_size <= 1 or (frame_idx + 1) % chunk_size == 0

        if not at_chunk_boundary:
            # Within chunk: store only new tokens in delta buffer (or full buffer if first chunk)
            B, H, N, D = k_full.shape
            anc = min(num_anchor_tokens, N)
            cnd = N - anc

            has_quantized = state.anchor_k_qcache is not None or state.cand_k_qcache is not None
            if has_quantized:
                delta_len = state.cand_k_delta_buffer.shape[2] if state.cand_k_delta_buffer is not None else 0
                n_new = cnd - (state.cand_len_at_last_boundary + delta_len)
                if n_new > 0:
                    new_k = k_full[:, :, anc + state.cand_len_at_last_boundary + delta_len :, :].contiguous().detach().clone()
                    new_v = v_full[:, :, anc + state.cand_len_at_last_boundary + delta_len :, :].contiguous().detach().clone()
                    if state.cand_k_delta_buffer is not None:
                        state.cand_k_delta_buffer = torch.cat([state.cand_k_delta_buffer, new_k], dim=2)
                        state.cand_v_delta_buffer = torch.cat([state.cand_v_delta_buffer, new_v], dim=2)
                    else:
                        state.cand_k_delta_buffer = new_k
                        state.cand_v_delta_buffer = new_v
            else:
                # First chunk: store full KV until first encode
                state.k_bf16_buffer = k_full.detach().clone()
                state.v_bf16_buffer = v_full.detach().clone()
            return

        # At chunk boundary: encode and store, clear buffers
        B, H, N, D = k_full.shape
        anc = min(num_anchor_tokens, N)
        cnd = N - anc
        n_spec = self.num_special_tokens

        # ── Special tokens (camera + register): always BF16, never quantised ──
        n_spec_anc = min(n_spec, anc)
        n_spec_cand = min(n_spec, cnd)
        if n_spec_anc > 0:
            state.special_anchor_k_bf16 = k_full[:, :, :n_spec_anc, :].contiguous().detach().clone()
            state.special_anchor_v_bf16 = v_full[:, :, :n_spec_anc, :].contiguous().detach().clone()
        else:
            state.special_anchor_k_bf16 = None
            state.special_anchor_v_bf16 = None
        if n_spec_cand > 0:
            state.special_cand_k_bf16 = k_full[:, :, anc : anc + n_spec_cand, :].contiguous().detach().clone()
            state.special_cand_v_bf16 = v_full[:, :, anc : anc + n_spec_cand, :].contiguous().detach().clone()
        else:
            state.special_cand_k_bf16 = None
            state.special_cand_v_bf16 = None

        # ── Anchor patch segment: [B, H, n_spec_anc:anc, D] ─────────────────
        anc_patch_len = anc - n_spec_anc
        if anc_patch_len > 0:
            ak = k_full[:, :, n_spec_anc:anc, :].contiguous()
            av = v_full[:, :, n_spec_anc:anc, :].contiguous()
            if self.config.anchor_full_precision:
                state.anchor_k_bf16_buffer = ak.detach().clone()
                state.anchor_v_bf16_buffer = av.detach().clone()
                state.anchor_k_qcache = None
                state.anchor_v_qcache = None
            else:
                state.anchor_k_qcache, new_anc_k_cents = self._prq_anchor.encode(
                    ak, state.anchor_k_prev_centroids, timing_dict=timing_dict
                )
                state.anchor_v_qcache, new_anc_v_cents = self._prq_anchor.encode(
                    av, state.anchor_v_prev_centroids, timing_dict=timing_dict
                )
                state.anchor_k_prev_centroids = new_anc_k_cents
                state.anchor_v_prev_centroids = new_anc_v_cents
                state.anchor_k_bf16_buffer = None
                state.anchor_v_bf16_buffer = None
        else:
            state.anchor_k_qcache = None
            state.anchor_v_qcache = None
            state.anchor_k_bf16_buffer = None
            state.anchor_v_bf16_buffer = None

        # ── Candidate patch segment: [B, H, anc+n_spec_cand:, D] ────────────
        cand_patch_len = cnd - n_spec_cand
        if cand_patch_len > 0:
            ck = k_full[:, :, anc + n_spec_cand :, :].contiguous()
            cv = v_full[:, :, anc + n_spec_cand :, :].contiguous()
            state.cand_k_qcache, new_cnd_k_cents = self._prq_cand.encode(
                ck, state.cand_k_prev_centroids, timing_dict=timing_dict
            )
            state.cand_v_qcache, new_cnd_v_cents = self._prq_cand.encode(
                cv, state.cand_v_prev_centroids, timing_dict=timing_dict
            )
            state.cand_k_prev_centroids = new_cnd_k_cents
            state.cand_v_prev_centroids = new_cnd_v_cents
        else:
            state.cand_k_qcache = None
            state.cand_v_qcache = None

        # Clear buffers and set candidate length for next chunk
        state.cand_len_at_last_boundary = cnd
        state.cand_k_delta_buffer = None
        state.cand_v_delta_buffer = None
        state.k_bf16_buffer = None
        state.v_bf16_buffer = None

    def reset(self) -> None:
        """Clear all cached states (call between sequences)."""
        self._states = [_LayerCacheState() for _ in range(self.num_layers)]

    def total_nbytes(self) -> int:
        """Total GPU bytes across all layers (quantised representation)."""
        return sum(s.nbytes() for s in self._states)

    def bf16_buffer_nbytes(self) -> int:
        """Total GPU bytes of BF16 buffers (full buffer + delta buffer + anchor + special tokens)."""
        total = 0
        for s in self._states:
            if s.k_bf16_buffer is not None:
                total += s.k_bf16_buffer.numel() * s.k_bf16_buffer.element_size()
            if s.v_bf16_buffer is not None:
                total += s.v_bf16_buffer.numel() * s.v_bf16_buffer.element_size()
            if s.cand_k_delta_buffer is not None:
                total += s.cand_k_delta_buffer.numel() * s.cand_k_delta_buffer.element_size()
            if s.cand_v_delta_buffer is not None:
                total += s.cand_v_delta_buffer.numel() * s.cand_v_delta_buffer.element_size()
            if s.special_anchor_k_bf16 is not None:
                total += s.special_anchor_k_bf16.numel() * s.special_anchor_k_bf16.element_size()
            if s.special_anchor_v_bf16 is not None:
                total += s.special_anchor_v_bf16.numel() * s.special_anchor_v_bf16.element_size()
            if s.special_cand_k_bf16 is not None:
                total += s.special_cand_k_bf16.numel() * s.special_cand_k_bf16.element_size()
            if s.special_cand_v_bf16 is not None:
                total += s.special_cand_v_bf16.numel() * s.special_cand_v_bf16.element_size()
            if s.anchor_k_bf16_buffer is not None:
                total += s.anchor_k_bf16_buffer.numel() * s.anchor_k_bf16_buffer.element_size()
            if s.anchor_v_bf16_buffer is not None:
                total += s.anchor_v_bf16_buffer.numel() * s.anchor_v_bf16_buffer.element_size()
        return total

    def memory_breakdown(self) -> dict:
        """Return detailed memory breakdown for debugging."""
        quantized = self.total_nbytes()
        bf16_buf = self.bf16_buffer_nbytes()
        return {
            "qvg_quantized_bytes": quantized,
            "qvg_bf16_buffer_bytes": bf16_buf,
            "qvg_total_bytes": quantized + bf16_buf,
        }

    def compression_ratio_vs_bf16(self, original_nbytes: int) -> float:
        """
        Compute achieved compression ratio vs BF16 baseline.

        Args:
            original_nbytes: how many bytes the BF16 KV tensors would have used.
        """
        if original_nbytes == 0:
            return 0.0
        return original_nbytes / max(self.total_nbytes(), 1)

    def layer_state_summary(self, layer_idx: int) -> dict:
        """Debug helper: returns byte counts for a single layer."""
        state = self._states[layer_idx]
        return {
            "anchor_bytes": (
                (state.anchor_k_qcache.nbytes() if state.anchor_k_qcache else 0) +
                (state.anchor_v_qcache.nbytes() if state.anchor_v_qcache else 0)
            ),
            "candidate_bytes": (
                (state.cand_k_qcache.nbytes() if state.cand_k_qcache else 0) +
                (state.cand_v_qcache.nbytes() if state.cand_v_qcache else 0)
            ),
            "total_bytes": state.nbytes(),
            "num_anchor_tokens": state.num_anchor_tokens,
        }
