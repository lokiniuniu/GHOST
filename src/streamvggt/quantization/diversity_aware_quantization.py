"""
Diversity-Aware KV Cache Quantization for InfiniteVGGT
======================================================
- First frame: full precision (no quantization)
- Subsequent frames: K and V quantized separately
  - Key: channel-wise ASYMMETRIC quantization (preserves channel diversity)
  - Value: token-wise SYMMETRIC quantization (more robust)
- High-diversity tokens (Top 10-20% by Diversity Score) exempt from quantization
- Supports int2, int4, int8 quantization
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


def _compute_diversity_scores(k: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token diversity score from Key (InfiniteVGGT style).
    Lower cosine similarity to mean = higher diversity.
    Returns scores [B, H, N] where higher = more diverse.
    """
    k_norm = F.normalize(k.float(), p=2, dim=-1)
    mean_k = k_norm.mean(dim=2, keepdim=True)
    cos_sim = (k_norm * mean_k).sum(dim=-1)
    diversity = -cos_sim
    return diversity


def _select_exempt_indices(diversity: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top ratio tokens by diversity. Returns (indices [B,H,n_exempt], mask [B,H,N])."""
    B, H, N = diversity.shape
    n_exempt = max(1, int(N * ratio))
    n_exempt = min(n_exempt, N)
    _, idx = torch.topk(diversity, k=n_exempt, dim=-1)
    mask = torch.zeros(B, H, N, dtype=torch.bool, device=diversity.device)
    mask.scatter_(2, idx, True)
    return idx, mask


def _quantize_symmetric_channel_wise(
    x: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Channel-wise symmetric quantization for Key.
    x: [B, H, N, D], scale per channel (last dim).
    Returns (x_int, scale) where scale [D].
    """
    max_val = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=(0, 1, 2), keepdim=True).clamp(min=1e-8)
    x_scaled = x / scale
    x_int = torch.clamp(torch.round(x_scaled), -max_val - 1, max_val).to(torch.int32)
    return x_int, scale.squeeze()


def _quantize_asymmetric_channel_wise_masked(
    x: torch.Tensor, mask: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Channel-wise ASYMMETRIC quantization for Key.
    mask=True means exempt (skip). Returns (x_int, scale, zero_point).
    scale = (max - min) / (qmax - qmin), zero_point = qmin - round(min / scale)
    """
    # Mask exempt with inf/-inf so they don't affect min/max
    x_for_min = x.masked_fill(mask.unsqueeze(-1), float("inf"))
    x_for_max = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
    x_min = x_for_min.amin(dim=(0, 1, 2), keepdim=True)
    x_max = x_for_max.amax(dim=(0, 1, 2), keepdim=True)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    range_val = (qmax - qmin)
    x_range = (x_max - x_min).squeeze()
    # Fallback to symmetric when range too small (constant channel)
    x_quant = x.masked_fill(mask.unsqueeze(-1), 0.0)
    use_sym = x_range < 1e-7
    scale = torch.where(use_sym, x_quant.abs().amax(dim=(0, 1, 2)).squeeze().clamp(min=1e-8), x_range.clamp(min=1e-8) / range_val)
    zero_point = torch.where(
        use_sym,
        torch.zeros(scale.shape, device=scale.device, dtype=torch.int32),
        (qmin - torch.round(x_min.squeeze() / scale)).clamp(qmin, qmax).to(torch.int32),
    )
    x_scaled = x / scale
    x_int = torch.clamp(torch.round(x_scaled) + zero_point, qmin, qmax).to(torch.int32)
    x_int = torch.where(mask.unsqueeze(-1), torch.zeros_like(x_int), x_int)
    return x_int, scale, zero_point


def _quantize_symmetric_token_wise_masked(
    x: torch.Tensor, mask: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-wise quant for non-exempt tokens. mask=True means exempt."""
    x_quant = x.masked_fill(mask.unsqueeze(-1), 0.0)
    scale = x_quant.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    max_val = 2 ** (bits - 1) - 1
    x_scaled = x_quant / scale
    x_int = torch.clamp(torch.round(x_scaled), -max_val - 1, max_val).to(torch.int32)
    return x_int, scale.squeeze(-1)


def _quantize_asymmetric_token_wise_masked(
    x: torch.Tensor, mask: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Token-wise ASYMMETRIC quantization for Value.
    mask=True means exempt. Returns (x_int, scale, zero_point).
    scale [B,H,N], zero_point [B,H,N].
    """
    x_for_min = x.masked_fill(mask.unsqueeze(-1), float("inf"))
    x_for_max = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
    x_min = x_for_min.amin(dim=-1, keepdim=True)
    x_max = x_for_max.amax(dim=-1, keepdim=True)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    range_val = (qmax - qmin)
    x_range = (x_max - x_min).squeeze(-1).clamp(min=1e-8)
    scale = x_range / range_val
    zero_point = (qmin - torch.round(x_min.squeeze(-1) / scale)).clamp(qmin, qmax).to(torch.int32)
    x_scaled = x / scale.unsqueeze(-1)
    x_int = torch.clamp(torch.round(x_scaled) + zero_point.unsqueeze(-1), qmin, qmax).to(torch.int32)
    x_int = torch.where(mask.unsqueeze(-1), torch.zeros_like(x_int), x_int)
    return x_int, scale, zero_point


def _dequantize_channel_wise(
    x_int: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric: x = x_int * scale. Asymmetric: x = (x_int - zero_point) * scale."""
    if zero_point is not None:
        return ((x_int.float() - zero_point.float()) * scale).to(dtype)
    return (x_int.float() * scale).to(dtype)


def _dequantize_token_wise(
    x_int: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric: x = x_int * scale. Asymmetric: x = (x_int - zero_point) * scale."""
    if zero_point is not None:
        return ((x_int.float() - zero_point.unsqueeze(-1).float()) * scale.unsqueeze(-1)).to(dtype)
    return (x_int.float() * scale.unsqueeze(-1)).to(dtype)


def _pack_int4(x_int: torch.Tensor) -> torch.Tensor:
    """Pack int4 values [B,H,N,D] into int32 [B,H,N,D//8]. Values in [-8,7]."""
    x_uint = (x_int + 8).clamp(0, 15).to(torch.uint8)
    B, H, N, D = x_uint.shape
    assert D % 8 == 0, "D must be divisible by 8 for int4 packing"
    x_uint = x_uint.reshape(B, H, N, D // 8, 8)
    packed = (x_uint[..., 0].int() +
              x_uint[..., 1].int() * 16 +
              x_uint[..., 2].int() * 256 +
              x_uint[..., 3].int() * 4096 +
              x_uint[..., 4].int() * 65536 +
              x_uint[..., 5].int() * 1048576 +
              x_uint[..., 6].int() * 16777216 +
              x_uint[..., 7].int() * 268435456)
    return packed.to(torch.int32)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 [B,H,N,D//8] to int4 [B,H,N,D]."""
    B, H, N, D8 = packed.shape
    D = D8 * 8
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    vals = ((packed.unsqueeze(-1).long() >> shifts) & 15) - 8
    return vals.reshape(B, H, N, D).to(torch.int32)


def _pack_int2(x_int: torch.Tensor) -> torch.Tensor:
    """Pack int2 values [B,H,N,D] into int32 [B,H,N,D//16]. Values in [-2,1]."""
    x_uint = (x_int + 2).clamp(0, 3).to(torch.uint8)
    B, H, N, D = x_uint.shape
    assert D % 16 == 0, "D must be divisible by 16 for int2 packing"
    x_uint = x_uint.reshape(B, H, N, D // 16, 16)
    shifts = torch.tensor([4 ** i for i in range(16)], device=x_uint.device, dtype=torch.int64)
    packed = (x_uint.int().to(x_uint.device) * shifts).sum(dim=-1)
    return packed.to(torch.int32)


def _unpack_int2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 [B,H,N,D//16] to int2 [B,H,N,D]."""
    B, H, N, D16 = packed.shape
    D = D16 * 16
    shifts = (torch.arange(16, device=packed.device, dtype=torch.int64) * 2)
    vals = ((packed.unsqueeze(-1).long() >> shifts) & 3) - 2
    return vals.reshape(B, H, N, D).to(torch.int32)


@dataclass
class DiversityQuantConfig:
    """Configuration for diversity-aware KV quantization."""
    enabled: bool = True
    bits: int = 4
    diversity_exempt_ratio: float = 0.15
    chunk_size: int = 1
    offload_to_cpu: bool = False  # False=fast (GPU dequant), True=low GPU mem (CPU dequant+transfer)
    value_asymmetric: bool = False  # Value: asymmetric (True) or symmetric (False)

    @classmethod
    def int2(cls, exempt_ratio: float = 0.15, offload_to_cpu: bool = False, value_asymmetric: bool = False) -> "DiversityQuantConfig":
        return cls(bits=2, diversity_exempt_ratio=exempt_ratio, offload_to_cpu=offload_to_cpu, value_asymmetric=value_asymmetric)

    @classmethod
    def int4(cls, exempt_ratio: float = 0.15, offload_to_cpu: bool = False, value_asymmetric: bool = False) -> "DiversityQuantConfig":
        return cls(bits=4, diversity_exempt_ratio=exempt_ratio, offload_to_cpu=offload_to_cpu, value_asymmetric=value_asymmetric)

    @classmethod
    def int8(cls, exempt_ratio: float = 0.15, offload_to_cpu: bool = False, value_asymmetric: bool = False) -> "DiversityQuantConfig":
        return cls(bits=8, diversity_exempt_ratio=exempt_ratio, offload_to_cpu=offload_to_cpu, value_asymmetric=value_asymmetric)


def quantize_diversity_aware(
    k: torch.Tensor,
    v: torch.Tensor,
    num_anchor_tokens: int,
    config: DiversityQuantConfig,
) -> Dict[str, Any]:
    """
    Quantize KV cache with diversity-aware exemption.

    Args:
        k, v: [B, H, N, D]
        num_anchor_tokens: first frame tokens (full precision)
        config: DiversityQuantConfig

    Returns:
        dict with anchor_k, anchor_v (full), and for candidates:
        exempt_mask, quant_k, scale_k, quant_v, scale_v, exempt_k, exempt_v
    """
    B, H, N, D = k.shape
    anc = min(num_anchor_tokens, N)
    cand = N - anc

    anchor_k = k[:, :, :anc, :].detach().clone()
    anchor_v = v[:, :, :anc, :].detach().clone()

    if cand <= 0:
        return {
            "anchor_k": anchor_k,
            "anchor_v": anchor_v,
            "cand_quant_k": None,
            "cand_scale_k": None,
            "cand_zero_point_k": None,
            "cand_quant_v": None,
            "cand_scale_v": None,
            "exempt_indices": None,
            "exempt_k_vals": None,
            "exempt_v_vals": None,
            "bits": config.bits,
        }

    cand_k = k[:, :, anc:, :]
    cand_v = v[:, :, anc:, :]

    diversity = _compute_diversity_scores(cand_k)
    exempt_indices, exempt_mask = _select_exempt_indices(diversity, config.diversity_exempt_ratio)
    n_exempt = exempt_indices.shape[2]

    # Sparse exempt storage: only store exempt token values [B,H,n_exempt,D], not full [B,H,N,D]
    exempt_k_vals = torch.gather(
        cand_k, 2, exempt_indices.unsqueeze(-1).expand(-1, -1, -1, cand_k.shape[-1])
    ).contiguous()
    exempt_v_vals = torch.gather(
        cand_v, 2, exempt_indices.unsqueeze(-1).expand(-1, -1, -1, cand_v.shape[-1])
    ).contiguous()

    quant_mask = ~exempt_mask
    n_quant = quant_mask.sum().item()
    if n_quant <= 0:
        return {
            "anchor_k": anchor_k,
            "anchor_v": anchor_v,
            "cand_quant_k": None,
            "cand_scale_k": None,
            "cand_zero_point_k": None,
            "cand_quant_v": None,
            "cand_scale_v": None,
            "cand_zero_point_v": None,
            "exempt_indices": exempt_indices,
            "exempt_k_vals": exempt_k_vals,
            "exempt_v_vals": exempt_v_vals,
            "bits": config.bits,
        }

    # Key: asymmetric channel-wise; Value: symmetric or asymmetric token-wise
    quant_k, scale_k, zero_point_k = _quantize_asymmetric_channel_wise_masked(
        cand_k, exempt_mask, config.bits
    )
    if config.value_asymmetric:
        quant_v, scale_v, zero_point_v = _quantize_asymmetric_token_wise_masked(
            cand_v, exempt_mask, config.bits
        )
    else:
        quant_v, scale_v = _quantize_symmetric_token_wise_masked(
            cand_v, exempt_mask, config.bits
        )
        zero_point_v = None

    # Compact storage: int8->int8, int4->packed int32 (8x), int2->packed int32 (16x)
    if config.bits == 8:
        quant_k = quant_k.clamp(-128, 127).to(torch.int8)
        quant_v = quant_v.clamp(-128, 127).to(torch.int8)
    elif config.bits == 4:
        quant_k = _pack_int4(quant_k)
        quant_v = _pack_int4(quant_v)
    elif config.bits == 2:
        quant_k = _pack_int2(quant_k)
        quant_v = _pack_int2(quant_v)

    out: Dict[str, Any] = {
        "anchor_k": anchor_k,
        "anchor_v": anchor_v,
        "cand_quant_k": quant_k,
        "cand_scale_k": scale_k,
        "cand_zero_point_k": zero_point_k,
        "cand_quant_v": quant_v,
        "cand_scale_v": scale_v,
        "exempt_indices": exempt_indices,
        "exempt_k_vals": exempt_k_vals,
        "exempt_v_vals": exempt_v_vals,
        "exempt_mask": exempt_mask,
        "bits": config.bits,
    }
    if zero_point_v is not None:
        out["cand_zero_point_v"] = zero_point_v
    return out


def dequantize_diversity_aware(data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct k, v from quantized storage."""
    anchor_k = data["anchor_k"]
    anchor_v = data["anchor_v"]
    B, H, anc, D = anchor_k.shape
    exempt_indices = data.get("exempt_indices")
    exempt_k_vals = data.get("exempt_k_vals")
    exempt_v_vals = data.get("exempt_v_vals")

    if data["cand_quant_k"] is None:
        if exempt_k_vals is not None:
            N_cand = exempt_indices.shape[2]
            dtype = anchor_k.dtype
            cand_k = torch.zeros(B, H, N_cand, D, dtype=dtype, device=exempt_k_vals.device)
            cand_v = torch.zeros(B, H, N_cand, D, dtype=dtype, device=exempt_v_vals.device)
            exempt_k_vals = exempt_k_vals.to(dtype=dtype)
            exempt_v_vals = exempt_v_vals.to(dtype=dtype)
            idx_exp = exempt_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            cand_k.scatter_(2, idx_exp, exempt_k_vals)
            cand_v.scatter_(2, idx_exp, exempt_v_vals)
            k = torch.cat([anchor_k, cand_k], dim=2)
            v = torch.cat([anchor_v, cand_v], dim=2)
        else:
            k, v = anchor_k, anchor_v
        return k, v

    quant_k = data["cand_quant_k"]
    scale_k = data["cand_scale_k"]
    zero_point_k = data.get("cand_zero_point_k")
    quant_v = data["cand_quant_v"]
    scale_v = data["cand_scale_v"]
    exempt_mask = data["exempt_mask"]
    bits = data.get("bits", 8)
    N_cand = quant_k.shape[2]

    # Unpack if bit-packed (int4->8x, int2->16x)
    if bits == 4:
        quant_k = _unpack_int4(quant_k)
        quant_v = _unpack_int4(quant_v)
    elif bits == 2:
        quant_k = _unpack_int2(quant_k)
        quant_v = _unpack_int2(quant_v)

    dtype = anchor_k.dtype
    if quant_k.dtype == torch.int8:
        quant_k = quant_k.float()
        quant_v = quant_v.float()
    # Key: asymmetric (zero_point); Value: symmetric or asymmetric
    zero_point_v = data.get("cand_zero_point_v")
    dequant_k = _dequantize_channel_wise(quant_k, scale_k, dtype, zero_point=zero_point_k)
    dequant_v = _dequantize_token_wise(quant_v, scale_v, dtype, zero_point=zero_point_v)

    # In-place scatter exempt values (avoid clone)
    idx_exp = exempt_indices.unsqueeze(-1).expand(-1, -1, -1, D)
    exempt_k_vals = exempt_k_vals.to(dtype=dtype)
    exempt_v_vals = exempt_v_vals.to(dtype=dtype)
    dequant_k.scatter_(2, idx_exp, exempt_k_vals)
    dequant_v.scatter_(2, idx_exp, exempt_v_vals)

    k = torch.cat([anchor_k, dequant_k], dim=2)
    v = torch.cat([anchor_v, dequant_v], dim=2)
    return k, v
