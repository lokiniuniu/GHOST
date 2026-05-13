"""
Triton Fused Dequantisation Kernel - QVG Paper Section 4.3

Implements a single-kernel fused "multi-stage dequant + centroid add-back" pass.
Intermediate results are kept in registers, avoiding repeated global-memory reads.

Falls back to PyTorch eager when Triton is unavailable or on CPU.
Compatible with FlashAttention (does not instantiate attention weight matrices).

Key design (QVG Sec 4.3):
  - For each row (token) i:
      r = int_to_float(x_int[i]) * scale[group(i)]   # dequant residual
      for t = T-1 .. 0:
          r = r + centroids[t][assignments[t][i]]     # add centroid back
  - All centroid lookups and accumulation stay in registers.
"""

from __future__ import annotations

import torch
from typing import List, Optional

# Try to import Triton
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ─── Triton kernel (single-stage, vectorised over tokens) ────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_dequant_add_centroids_kernel(
        x_int_ptr,       # [N, D] int8
        scales_ptr,      # [G, D] float32
        # Stage 0 centroids and assignments (pre-loaded)
        cent0_ptr,       # [C, D] float32
        asgn0_ptr,       # [N]    int32
        # Stage 1 (optional, set ptr=0 to skip)
        cent1_ptr,
        asgn1_ptr,
        # Stage 2 (optional)
        cent2_ptr,
        asgn2_ptr,
        # Stage 3 (optional)
        cent3_ptr,
        asgn3_ptr,
        out_ptr,         # [N, D] bfloat16
        N: tl.constexpr,
        D: tl.constexpr,
        G: tl.constexpr,
        group_size: tl.constexpr,
        T: tl.constexpr,   # number of PRQ stages (1..4)
        BLOCK_D: tl.constexpr,
    ):
        """
        Each program handles one token row i across all D dimensions.
        Registers hold the running accumulator to avoid global-memory round-trips.
        """
        row = tl.program_id(0)
        if row >= N:
            return

        group = row // group_size
        col_offsets = tl.arange(0, BLOCK_D)
        mask = col_offsets < D

        # Load int8 quantised value and dequantise in-register
        x_int = tl.load(x_int_ptr + row * D + col_offsets, mask=mask, other=0).to(tl.float32)
        scale = tl.load(scales_ptr + group * D + col_offsets, mask=mask, other=1.0).to(tl.float32)
        acc = x_int * scale   # dequantised residual R̂^(T)

        # Stage T-1 .. 0: add centroid back (backward iteration)
        # Stage 3
        if T >= 4:
            a3 = tl.load(asgn3_ptr + row).to(tl.int32)
            c3 = tl.load(cent3_ptr + a3 * D + col_offsets, mask=mask, other=0.0).to(tl.float32)
            acc = acc + c3
        # Stage 2
        if T >= 3:
            a2 = tl.load(asgn2_ptr + row).to(tl.int32)
            c2 = tl.load(cent2_ptr + a2 * D + col_offsets, mask=mask, other=0.0).to(tl.float32)
            acc = acc + c2
        # Stage 1
        if T >= 2:
            a1 = tl.load(asgn1_ptr + row).to(tl.int32)
            c1 = tl.load(cent1_ptr + a1 * D + col_offsets, mask=mask, other=0.0).to(tl.float32)
            acc = acc + c1
        # Stage 0 (always present)
        a0 = tl.load(asgn0_ptr + row).to(tl.int32)
        c0 = tl.load(cent0_ptr + a0 * D + col_offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + c0

        # Write output as BF16
        tl.store(out_ptr + row * D + col_offsets, acc.to(tl.bfloat16), mask=mask)


def _get_null_ptr():
    """Return a zeroed placeholder tensor pointer for Triton null-ptr sentinel."""
    return torch.zeros(1, device="cuda", dtype=torch.float32)


def fused_dequant_add_centroids_triton(
    x_int: torch.Tensor,          # [N, D] int8
    scales: torch.Tensor,         # [G, D] fp8/bf16/f32
    centroids_list: List[torch.Tensor],    # T × [C, D] bf16
    assignments_list: List[torch.Tensor],  # T × [N] uint8
    orig_N: int,
    group_size: int,
) -> torch.Tensor:
    """
    Fused Triton kernel: multi-stage dequant + centroid add-back.

    Falls back to PyTorch eager if Triton unavailable or tensors on CPU.
    """
    if not _TRITON_AVAILABLE or not x_int.is_cuda:
        return _eager_dequant_add_centroids(
            x_int, scales, centroids_list, assignments_list, orig_N, group_size
        )

    T = len(centroids_list)
    assert 1 <= T <= 4, "Triton kernel supports T=1..4 stages"

    N, D = x_int.shape
    G = math.ceil(orig_N / group_size)
    BLOCK_D = triton.next_power_of_2(D)

    out = torch.empty((N, D), device=x_int.device, dtype=torch.bfloat16)

    # Prepare centroid/assignment pointers (pad with nulls for unused stages)
    def _prep_cent(t):
        return centroids_list[t].float().contiguous() if t < T else _get_null_ptr().reshape(1, 1)

    def _prep_asgn(t):
        return assignments_list[t].int().contiguous() if t < T else torch.zeros(1, device=x_int.device, dtype=torch.int32)

    _fused_dequant_add_centroids_kernel[(N,)](
        x_int.contiguous(),
        scales.float().contiguous(),
        _prep_cent(0), _prep_asgn(0),
        _prep_cent(1), _prep_asgn(1),
        _prep_cent(2), _prep_asgn(2),
        _prep_cent(3), _prep_asgn(3),
        out,
        N=N, D=D, G=G,
        group_size=group_size,
        T=T,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return out[:orig_N]


# ─── Pure-PyTorch eager fallback ─────────────────────────────────────────────

def _eager_dequant_add_centroids(
    x_int: torch.Tensor,
    scales: torch.Tensor,
    centroids_list: List[torch.Tensor],
    assignments_list: List[torch.Tensor],
    orig_N: int,
    group_size: int,
) -> torch.Tensor:
    """
    PyTorch eager implementation of fused dequant + centroid add-back.
    Used when Triton is unavailable or as a correctness reference.
    """
    import math as _math

    N, d = x_int.shape
    G = _math.ceil(orig_N / group_size)
    pad = G * group_size - orig_N

    xf = x_int.float()
    sf = scales.float()

    if pad > 0:
        import torch.nn.functional as F
        xf = F.pad(xf, (0, 0, 0, pad))

    xf = xf.reshape(G, group_size, d)
    xf = xf * sf.unsqueeze(1)
    xf = xf.reshape(G * group_size, d)
    if pad > 0:
        xf = xf[:orig_N]

    # Backward centroid add-back
    for t in range(len(centroids_list) - 1, -1, -1):
        c = centroids_list[t].float()
        a = assignments_list[t].long()
        xf = xf + c[a]

    return xf.to(torch.bfloat16)


# ─── Public dispatch ──────────────────────────────────────────────────────────

def fused_dequant(
    x_int: torch.Tensor,
    scales: torch.Tensor,
    centroids_list: List[torch.Tensor],
    assignments_list: List[torch.Tensor],
    orig_N: int,
    group_size: int,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Public dispatch: use Triton kernel when available and requested,
    otherwise fall back to PyTorch eager.

    Args:
        x_int:            [N, D] int8 quantised residual
        scales:           [G, D] scale factors
        centroids_list:   T × [C, D] centroid tensors (stage 0..T-1)
        assignments_list: T × [N] assignment vectors  (stage 0..T-1)
        orig_N:           unpadded token count
        group_size:       B (64 or 16)
        use_triton:       toggle Triton backend

    Returns:
        [orig_N, D] BF16 reconstructed tensor
    """
    if use_triton and _TRITON_AVAILABLE and x_int.is_cuda:
        return fused_dequant_add_centroids_triton(
            x_int, scales, centroids_list, assignments_list, orig_N, group_size
        )
    return _eager_dequant_add_centroids(
        x_int, scales, centroids_list, assignments_list, orig_N, group_size
    )


# Expose availability flag
TRITON_AVAILABLE = _TRITON_AVAILABLE

import math


# ─── Triton fused encode (symmetric per-group quantize) ───────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_quantize_group_kernel(
        x_ptr,
        out_int_ptr,
        scales_ptr,
        group_size: tl.constexpr,
        D: tl.constexpr,
        max_val: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per group: reduce max, compute scale, quantize."""
        g = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_D)
        mask = col_offsets < D

        # Load group data [group_size, D]
        max_vals = tl.zeros([D], dtype=tl.float32)
        for row in range(group_size):
            off = (g * group_size + row) * D + col_offsets
            vals = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            max_vals = tl.maximum(max_vals, tl.abs(vals))

        # Scale = max / max_val, clamp to avoid /0
        scale = max_vals / max_val
        scale = tl.where(scale < 1e-8, 1e-8, scale)

        # Store scales
        tl.store(scales_ptr + g * D + col_offsets, scale, mask=mask)

        # Quantize each row in group (round = sign * floor(|x| + 0.5), tl.math.round not in Triton)
        for row in range(group_size):
            off = (g * group_size + row) * D + col_offsets
            vals = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            s = vals / scale
            x_q = tl.where(s >= 0, tl.math.floor(s + 0.5), -tl.math.floor(-s + 0.5))
            x_q = tl.minimum(tl.maximum(x_q, -max_val), max_val)
            tl.store(out_int_ptr + off, x_q.to(tl.int8), mask=mask)


def fused_symmetric_per_group_quantize(
    x: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple:
    """
    Triton-fused symmetric per-group quantize.
    Returns (x_int8_flat, scales) for bit-packing by caller.
    """
    if not _TRITON_AVAILABLE or not x.is_cuda:
        return None
    N, d = x.shape
    G = math.ceil(N / group_size)
    pad = G * group_size - N
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    x = x.contiguous().float()
    max_val = 2 ** (bits - 1) - 1
    BLOCK_D = triton.next_power_of_2(d)
    out_int = torch.empty((G * group_size, d), device=x.device, dtype=torch.int8)
    scales = torch.empty((G, d), device=x.device, dtype=torch.float32)
    _fused_quantize_group_kernel[(G,)](
        x, out_int, scales,
        group_size=group_size, D=d, max_val=max_val, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out_int, scales
