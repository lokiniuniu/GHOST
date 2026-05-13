"""
Progressive Residual Quantization (PRQ) - QVG Paper Section 4.2

Inspired by video codec multi-scale encoding:
  - T stages of SAS smoothing reduce residual amplitude iteratively
  - Only the final-stage residual is low-bit quantised
  - Dequantisation reverses via backward iteration over stored centroids

QVG Paper Table 1 compression ratios (BF16 baseline):
  QVG  (T=1, B=64,  INT2): 6.94× – 7.05×
  QVG  (T=1, B=64,  INT4): 3.72× – 3.75×
  QVG-Pro (T=4, B=16, INT2): 4.97× – 5.20×
  QVG-Pro (T=4, B=16, INT4): 3.05× – 3.15×

Quantisation formula (QVG Sec 3.2 – symmetric per-group integer):
  S_X  = max(|X_BF16|) / (2^{b-1} - 1)
  X_INT = round(X_BF16 / S_X)
  X̂    = S_X * X_INT
  error ≤ S_X / 2
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .semantic_aware_smoothing import SemanticAwareSmoothing

# Import Triton fused decode/encode dispatch (falls back to eager automatically)
try:
    from .triton_kernels import fused_dequant as _fused_dequant
    _USE_FUSED_DECODE = True
except ImportError:
    _fused_dequant = None
    _USE_FUSED_DECODE = False

try:
    from .triton_kernels import fused_symmetric_per_group_quantize as _fused_quantize
except ImportError:
    _fused_quantize = None


# ─── Per-group symmetric integer quantisation ────────────────────────────────

def _pack_int2(x_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack four INT2 values (range [-1,1]) from int8 into one uint8.
    Stores values offset by 1 to map {-1,0,1} → {0,1,2} (2-bit unsigned).

    Layout (little-endian within each byte):
        byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
    where v0..v3 ∈ {0,1,2,3} = original + 1.

    Args:
        x_int8: [N] int8 with values in [-1, 1]
    Returns:
        [ceil(N/4)] uint8 packed tensor
    """
    N = x_int8.numel()
    pad = (4 - N % 4) % 4
    x = x_int8.reshape(-1)
    if pad:
        x = F.pad(x.float(), (0, pad)).to(torch.int8)
    # Offset to unsigned: {-1,0,1} → {0,1,2}
    x_u = (x.int() + 1).to(torch.uint8)          # [N']
    x_u = x_u.reshape(-1, 4)                      # [N'//4, 4]
    packed = (x_u[:, 0]
              | (x_u[:, 1] << 2)
              | (x_u[:, 2] << 4)
              | (x_u[:, 3] << 6))
    return packed.to(torch.uint8)


def _unpack_int2(packed: torch.Tensor, orig_N: int) -> torch.Tensor:
    """Reverse of _pack_int2. Returns [orig_N] int8."""
    p = packed.to(torch.int32)
    v0 = ((p >> 0) & 0x3).to(torch.int8) - 1
    v1 = ((p >> 2) & 0x3).to(torch.int8) - 1
    v2 = ((p >> 4) & 0x3).to(torch.int8) - 1
    v3 = ((p >> 6) & 0x3).to(torch.int8) - 1
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)
    return unpacked[:orig_N]


def _pack_int4(x_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack two INT4 values (range [-7,7]) from int8 into one uint8.
    Stores values offset by 8 to map [-7,7] → [1,15] (4-bit unsigned, 0 unused).

    Args:
        x_int8: [N] int8 with values in [-7, 7]
    Returns:
        [ceil(N/2)] uint8 packed tensor
    """
    N = x_int8.numel()
    pad = N % 2
    x = x_int8.reshape(-1)
    if pad:
        x = torch.cat([x, x.new_zeros(1)])
    x_u = (x.int() + 8).to(torch.uint8)           # offset to [1,15]
    x_u = x_u.reshape(-1, 2)
    packed = x_u[:, 0] | (x_u[:, 1] << 4)
    return packed.to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, orig_N: int) -> torch.Tensor:
    """Reverse of _pack_int4. Returns [orig_N] int8."""
    p = packed.to(torch.int32)
    lo = ((p >> 0) & 0xF).to(torch.int8) - 8
    hi = ((p >> 4) & 0xF).to(torch.int8) - 8
    unpacked = torch.stack([lo, hi], dim=-1).reshape(-1)
    return unpacked[:orig_N]


def _symmetric_per_group_quantize(
    x: torch.Tensor,
    bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    QVG Sec 3.2: symmetric per-group integer quantisation with bit-packing.

    [Optimization] Uses Triton fused kernel when available (CUDA + Triton).

    S_X  = max(|X|) / (2^{b-1} - 1)    (per group)
    X_INT = clamp(round(X / S_X), -(2^{b-1}-1), 2^{b-1}-1)

    Bit-packing achieves physical storage density:
      INT2: 4 values/byte  → 8× smaller than FP32, 4× smaller than int8
      INT4: 2 values/byte  → 4× smaller than FP32, 2× smaller than int8

    Scale factors stored in FP8 E4M3 when available, else BF16 (QVG Sec 5.1).

    Args:
        x:          [N, d]    BF16/FP16 input
        bits:       2 or 4
        group_size: B=64 (QVG) or B=16 (QVG-Pro)

    Returns:
        x_packed: [ceil(N*d / packing_factor)] uint8 – bit-packed quantised values
        scales:   [G, d]  scale factors, G = ceil(N / group_size)
    """
    N, d = x.shape
    G = math.ceil(N / group_size)
    pad = G * group_size - N

    # [Optimization] Triton fused path when available
    if _fused_quantize is not None and x.is_cuda:
        result = _fused_quantize(x, bits, group_size)
        if result is not None:
            out_int, scales = result
            if pad > 0:
                out_int = out_int[:N]
            x_int8 = out_int.reshape(-1)
            if bits == 2:
                x_packed = _pack_int2(x_int8)
            else:
                x_packed = _pack_int4(x_int8)
            scales = _try_fp8_scales(scales)
            return x_packed, scales

    # PyTorch fallback
    max_val = 2 ** (bits - 1) - 1          # 1 for INT2, 7 for INT4
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))       # pad along first dim

    x_grouped = x.reshape(G, group_size, d).float()
    scales = x_grouped.abs().amax(dim=1) / max_val   # [G, d]
    scales = scales.clamp(min=1e-8)                  # avoid /0

    x_q = torch.round(x_grouped / scales.unsqueeze(1)).clamp(-max_val, max_val)
    x_q = x_q.reshape(G * group_size, d)
    if pad > 0:
        x_q = x_q[:N]    # remove padding rows

    x_int8 = x_q.to(torch.int8).reshape(-1)

    if bits == 2:
        x_packed = _pack_int2(x_int8)
    else:  # bits == 4
        x_packed = _pack_int4(x_int8)

    scales = _try_fp8_scales(scales)

    return x_packed, scales


def _try_fp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Store scale factors in FP8 E4M3 if the runtime supports it (QVG Sec 5.1)."""
    try:
        fp8_type = torch.float8_e4m3fn
        return scales.to(fp8_type)
    except (AttributeError, RuntimeError):
        return scales.to(torch.bfloat16)


def _symmetric_per_group_dequantize(
    x_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    orig_N: int,
    bits: int,
    d: int,
) -> torch.Tensor:
    """
    Reverse of _symmetric_per_group_quantize.

    X̂ = S_X * X_INT   (QVG Sec 3.2)

    Args:
        x_packed:   uint8 bit-packed tensor (INT2 or INT4 encoding)
        scales:     [G, d]   scale factors (FP8 or BF16)
        group_size: B
        orig_N:     original unpadded N
        bits:       2 or 4
        d:          feature dimension

    Returns:
        [orig_N, d]  BF16 reconstructed tensor
    """
    G = math.ceil(orig_N / group_size)
    pad = G * group_size - orig_N
    scales_f = scales.float()   # upcast from FP8/BF16

    # Unpack bits → int8
    total_elem = orig_N * d
    if bits == 2:
        x_int8 = _unpack_int2(x_packed, total_elem)
    else:
        x_int8 = _unpack_int4(x_packed, total_elem)

    x_f = x_int8.float().reshape(orig_N, d)

    if pad > 0:
        x_f = F.pad(x_f, (0, 0, 0, pad))

    x_grouped = x_f.reshape(G, group_size, d)
    dequant = x_grouped * scales_f.unsqueeze(1)     # broadcast per-group scale
    dequant = dequant.reshape(G * group_size, d)
    if pad > 0:
        dequant = dequant[:orig_N]
    return dequant.to(torch.bfloat16)


# ─── PRQ data container ───────────────────────────────────────────────────────

class PRQQuantizedCache:
    """
    Holds all parameters required to dequantise a single KV cache tensor.

    Memory layout (see QVG paper Fig 7a):
      - x_int:       low-bit quantised residual   (≥65% of total bytes)
      - scales:      per-group scale factors, FP8  (small)
      - centroids:   T centroid matrices           (small)
      - assignments: T uint8 assignment vectors    (small)
      - orig_shape:  metadata
    """

    __slots__ = (
        "x_packed", "scales", "group_size", "bits", "feat_dim",
        "centroids_list", "assignments_list",
        "orig_N", "orig_shape",
    )

    def __init__(
        self,
        x_packed: torch.Tensor,       # uint8 bit-packed quantised residual
        scales: torch.Tensor,
        group_size: int,
        bits: int,
        feat_dim: int,                # d (feature dimension)
        centroids_list: List[torch.Tensor],
        assignments_list: List[torch.Tensor],
        orig_N: int,
        orig_shape: torch.Size,
    ):
        self.x_packed = x_packed
        self.scales = scales
        self.group_size = group_size
        self.bits = bits
        self.feat_dim = feat_dim
        self.centroids_list = centroids_list
        self.assignments_list = assignments_list
        self.orig_N = orig_N
        self.orig_shape = orig_shape

    def nbytes(self) -> int:
        """Estimate total GPU bytes consumed by this quantised cache entry."""
        n = self.x_packed.numel() * self.x_packed.element_size()
        n += self.scales.numel() * self.scales.element_size()
        for c in self.centroids_list:
            n += c.numel() * c.element_size()
        for a in self.assignments_list:
            n += a.numel() * a.element_size()
        return n


# ─── Progressive Residual Quantisation ────────────────────────────────────────

class ProgressiveResidualQuantization:
    """
    QVG Paper Section 4.2 – Progressive Residual Quantisation.

    Two preset configurations (QVG Sec 5.1):
      - QVG standard  (high compression): T=1, B=64,  works on candidate tokens
      - QVG-Pro       (high accuracy):    T=4, B=16,  works on anchor tokens

    Workflow:
      Encode:
        R^(0) = X
        for t in 1..T:
            R^(t), C^(t), π^(t) = SAS(R^(t-1), prev_centroids)
        X_INT, S_X = quantise(R^(T), bits, B)
        store: X_INT, S_X, {C^(t)}, {π^(t)}

      Decode (QVG Sec 4.2 Step 4 – backward iteration):
        R̂^(T) = dequantise(X_INT, S_X)
        for t in T..1:
            R̂^(t-1) = R̂^(t) + C^(t)[π^(t)]   ≡ SAS.reconstruct(...)
        return R̂^(0)
    """

    # QVG-standard preset  – candidate tokens (InfiniteVGGT rolling memory)
    PRESET_STANDARD = dict(num_stages=1, group_size=64, bits=2)
    # QVG-Pro preset       – anchor tokens   (first-frame, never evicted)
    PRESET_PRO      = dict(num_stages=4, group_size=16, bits=4)

    def __init__(
        self,
        num_stages: int = 1,
        group_size: int = 64,
        bits: int = 2,
        num_centroids: int = 256,           # fixed, QVG Sec 5.1
        kmeans_max_iters: int = 10,   # [Optimization 2] Reduced from 20 for speed
    ):
        assert bits in (2, 4), "Only INT2 and INT4 are supported"
        assert num_stages >= 1
        self.T = num_stages
        self.B = group_size
        self.bits = bits
        self.sas = SemanticAwareSmoothing(
            num_centroids=num_centroids,
            kmeans_max_iters=kmeans_max_iters,
        )

    # ── Encoding ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        prev_centroids_list: Optional[List[Optional[torch.Tensor]]] = None,
        timing_dict: Optional[dict] = None,
    ) -> Tuple["PRQQuantizedCache", List[torch.Tensor]]:
        """
        Quantise a KV cache tensor via T stages of SAS + final low-bit quantisation.

        Args:
            x:  [B_batch, H, N, D]  or  [N, D]  – BF16/FP16 KV tensor
                Caller should reshape to [N_flat, D] before calling when
                operating per-head (the CacheManager handles reshaping).
            prev_centroids_list:  List[Optional[Tensor]] of length T,
                                  warm-start centroids per stage (streaming accel.)
            timing_dict:  Optional dict to accumulate encode_sas, encode_quantize time

        Returns:
            qcache:              PRQQuantizedCache object
            new_centroids_list:  List[Tensor] (len T) for next chunk's warm-start
        """
        import time
        orig_shape = x.shape
        if x.ndim > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x
        orig_N = x_2d.shape[0]

        centroids_list: List[torch.Tensor] = []
        assignments_list: List[torch.Tensor] = []
        new_centroids_list: List[torch.Tensor] = []

        residual = x_2d
        if timing_dict is not None and x_2d.is_cuda:
            torch.cuda.synchronize()
        t_sas0 = time.perf_counter() if timing_dict is not None else 0
        for t in range(self.T):
            prev_c = None
            if prev_centroids_list is not None and t < len(prev_centroids_list):
                prev_c = prev_centroids_list[t]
            res, centroids, assignments = self.sas(residual, prev_centroids=prev_c, timing_dict=timing_dict)
            centroids_list.append(centroids)
            assignments_list.append(assignments)
            new_centroids_list.append(centroids)
            # Only intermediate residuals are passed forward;
            # final residual is what gets quantised (QVG Sec 4.2 Step 2)
            residual = res
        if timing_dict is not None and x_2d.is_cuda:
            torch.cuda.synchronize()
        if timing_dict is not None:
            timing_dict["encode_sas"] = timing_dict.get("encode_sas", 0.0) + time.perf_counter() - t_sas0

        # Quantise only the last-stage residual R^(T)
        d = residual.shape[-1]
        if timing_dict is not None and residual.is_cuda:
            torch.cuda.synchronize()
        t_quant0 = time.perf_counter() if timing_dict is not None else 0
        x_packed, scales = _symmetric_per_group_quantize(residual, self.bits, self.B)
        if timing_dict is not None and residual.is_cuda:
            torch.cuda.synchronize()
        if timing_dict is not None:
            timing_dict["encode_quantize"] = timing_dict.get("encode_quantize", 0.0) + time.perf_counter() - t_quant0

        qcache = PRQQuantizedCache(
            x_packed=x_packed,
            scales=scales,
            group_size=self.B,
            bits=self.bits,
            feat_dim=d,
            centroids_list=centroids_list,
            assignments_list=assignments_list,
            orig_N=orig_N,
            orig_shape=orig_shape,
        )
        return qcache, new_centroids_list

    # ── Decoding ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def decode(self, qcache: "PRQQuantizedCache", use_triton: bool = True, timing_dict: Optional[dict] = None) -> torch.Tensor:
        """
        Reconstruct BF16 KV tensor via backward iteration (QVG Sec 4.2 Step 4).

        Uses the Triton fused dequant kernel (QVG Sec 4.3) when available:
          - Keeps intermediate results in registers
          - Avoids repeated global-memory round-trips for centroid add-back
          - Falls back to PyTorch eager automatically

        Returns:
            [orig_shape] BF16 tensor
        """
        import time
        orig_N, d = qcache.orig_N, qcache.feat_dim
        total_elem = orig_N * d

        # Unpack bit-packed quantized values to int8 [orig_N, d]
        if timing_dict is not None and qcache.x_packed.is_cuda:
            torch.cuda.synchronize()
        t_unpack0 = time.perf_counter() if timing_dict is not None else 0
        if qcache.bits == 2:
            x_int_flat = _unpack_int2(qcache.x_packed, total_elem)
        else:
            x_int_flat = _unpack_int4(qcache.x_packed, total_elem)
        x_int = x_int_flat.reshape(orig_N, d).to(qcache.x_packed.device)
        if timing_dict is not None and qcache.x_packed.is_cuda:
            torch.cuda.synchronize()
        if timing_dict is not None:
            timing_dict["decode_unpack"] = timing_dict.get("decode_unpack", 0.0) + time.perf_counter() - t_unpack0

        # [Optimization 1] Triton fused path: dequant + centroid add-back in one kernel
        if timing_dict is not None and qcache.x_packed.is_cuda:
            torch.cuda.synchronize()
        t_dequant0 = time.perf_counter() if timing_dict is not None else 0
        if _USE_FUSED_DECODE and use_triton and qcache.x_packed.is_cuda:
            x_hat_2d = _fused_dequant(
                x_int, qcache.scales,
                qcache.centroids_list, qcache.assignments_list,
                orig_N, qcache.group_size,
                use_triton=True,
            )
            if timing_dict is not None and qcache.x_packed.is_cuda:
                torch.cuda.synchronize()
            if timing_dict is not None:
                timing_dict["decode_dequant_triton"] = timing_dict.get("decode_dequant_triton", 0.0) + time.perf_counter() - t_dequant0
        else:
            # Eager fallback: dequant then centroid add-back
            t_sym0 = time.perf_counter() if timing_dict is not None else 0
            x_hat_2d = _symmetric_per_group_dequantize(
                qcache.x_packed, qcache.scales,
                qcache.group_size, orig_N,
                qcache.bits, d,
            )
            if timing_dict is not None and qcache.x_packed.is_cuda:
                torch.cuda.synchronize()
            if timing_dict is not None:
                timing_dict["decode_dequant_sym"] = timing_dict.get("decode_dequant_sym", 0.0) + time.perf_counter() - t_sym0
            t_cent0 = time.perf_counter() if timing_dict is not None else 0
            for t in range(self.T - 1, -1, -1):
                centroids   = qcache.centroids_list[t].float()
                assignments = qcache.assignments_list[t].long()
                x_hat_2d = x_hat_2d + centroids[assignments].to(x_hat_2d.dtype)
            if timing_dict is not None and qcache.x_packed.is_cuda:
                torch.cuda.synchronize()
            if timing_dict is not None:
                timing_dict["decode_centroid_add"] = timing_dict.get("decode_centroid_add", 0.0) + time.perf_counter() - t_cent0

        return x_hat_2d.reshape(qcache.orig_shape)

    # ── Convenience factory methods ───────────────────────────────────────────

    @classmethod
    def make_standard(cls) -> "ProgressiveResidualQuantization":
        """QVG standard: T=1, B=64, INT2 – for candidate/rolling tokens."""
        return cls(**cls.PRESET_STANDARD)

    @classmethod
    def make_pro(cls) -> "ProgressiveResidualQuantization":
        """QVG-Pro: T=4, B=16, INT4 – for anchor tokens (first frame)."""
        return cls(**cls.PRESET_PRO)

    @classmethod
    def make_standard_int4(cls) -> "ProgressiveResidualQuantization":
        """QVG standard with INT4 for slightly higher quality candidates."""
        return cls(num_stages=1, group_size=64, bits=4)
