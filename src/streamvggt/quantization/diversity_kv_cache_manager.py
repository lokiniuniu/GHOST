"""
Diversity-Aware KV Cache Manager for InfiniteVGGT
=================================================
Integrates diversity_aware_quantization with the aggregator.
First frame full precision; subsequent frames: K channel-wise, V token-wise,
with top 10-20% high-diversity tokens exempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .diversity_aware_quantization import (
    DiversityQuantConfig,
    quantize_diversity_aware,
    dequantize_diversity_aware,
)


@dataclass
class _DiversityLayerState:
    """Per-layer cache state."""
    data: Optional[Dict] = None

    def nbytes(self) -> int:
        if self.data is None:
            return 0
        total = 0
        for key in ("anchor_k", "anchor_v", "exempt_k_vals", "exempt_v_vals"):
            t = self.data.get(key)
            if t is not None:
                total += t.numel() * t.element_size()
        for key in ("exempt_indices",):
            t = self.data.get(key)
            if t is not None:
                total += t.numel() * t.element_size()
        for key in ("cand_quant_k", "cand_quant_v"):
            t = self.data.get(key)
            if t is not None:
                total += t.numel() * t.element_size()
        for key in ("cand_scale_k", "cand_scale_v", "cand_zero_point_k", "cand_zero_point_v"):
            t = self.data.get(key)
            if t is not None:
                total += t.numel() * t.element_size()
        return total


class DiversityKVCacheManager:
    """
    KV Cache Manager with diversity-aware quantization.
    - First frame (anchor): full precision
    - Subsequent frames: K channel-wise, V token-wise quant
    - Top diversity_exempt_ratio tokens: full precision
    """

    def __init__(
        self,
        config: DiversityQuantConfig,
        num_layers: int,
    ):
        self.config = config
        self.num_layers = num_layers
        self._states: List[_DiversityLayerState] = [
            _DiversityLayerState() for _ in range(num_layers)
        ]

    def get_past_key_values_for_layer(
        self,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Dequantize and return (k, v) for attention. Dequant on CPU if offloaded to avoid GPU peak."""
        state = self._states[layer_idx]
        if state.data is None:
            return None
        k, v = dequantize_diversity_aware(state.data)
        return k.to(device, dtype=dtype), v.to(device, dtype=dtype)

    def update_layer_cache(
        self,
        layer_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        num_anchor_tokens: int,
        **kwargs,
    ) -> None:
        """Quantize and store KV after eviction."""
        if not self.config.enabled:
            data = {
                "anchor_k": k_full.detach().clone(),
                "anchor_v": v_full.detach().clone(),
                "cand_quant_k": None,
                "cand_scale_k": None,
                "cand_zero_point_k": None,
                "cand_quant_v": None,
                "cand_scale_v": None,
                "exempt_indices": None,
                "exempt_k_vals": None,
                "exempt_v_vals": None,
            }
        else:
            data = quantize_diversity_aware(
                k_full, v_full, num_anchor_tokens, self.config
            )
        if self.config.offload_to_cpu:
            for k, v in data.items():
                if v is not None and isinstance(v, torch.Tensor):
                    data[k] = v.cpu()
        self._states[layer_idx].data = data

    def total_nbytes(self) -> int:
        return sum(s.nbytes() for s in self._states)

    def memory_breakdown(self) -> Dict[str, float]:
        """For debug_mem compatibility."""
        total = self.total_nbytes()
        return {
            "qvg_quantized_bytes": float(total),
            "qvg_bf16_buffer_bytes": 0.0,
        }

    def reset(self) -> None:
        """Clear cache for new sequence."""
        for s in self._states:
            s.data = None
