"""
Semantic-Aware Smoothing (SAS) - QVG Paper Section 4.1

Exploits spatiotemporal redundancy in video KV Cache via grouped k-means
centroid subtraction, producing low-amplitude residuals that are far more
quantization-friendly.

Key results (QVG paper Fig 6):
  - Key Cache quantization error reduction: ~6.9x
  - Value Cache quantization error reduction: ~2.6x
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# [Optimization] Optional Flash-KMeans for faster k-means (pip install flash-kmeans)
try:
    from flash_kmeans import batch_kmeans_Euclid
    _FLASH_KMEANS_AVAILABLE = True
except ImportError:
    _FLASH_KMEANS_AVAILABLE = False

# Disable torch.compile for k-means (CUDAGraph incompatible with iterative loop)
try:
    _disable_compile = torch.compiler.disable
except AttributeError:
    def _disable_compile(fn):
        return fn


# ─── k-means helpers ────────────────────────────────────────────────────────

@_disable_compile
def _kmeans_step_impl(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single E-step + M-step of Lloyd's algorithm.

    Args:
        x:          [N, d]  token features
        centroids:  [C, d]  current centroid estimates

    Returns:
        new_centroids: [C, d]
        assignments:   [N] int64 - index into [0..C-1]
    """
    # E-step: nearest centroid via L2 distance  (chunked to save memory)
    # dist[i,c] = ||x[i] - centroids[c]||^2
    #           = ||x[i]||^2 + ||c||^2 - 2 x[i]·c
    x_sq = (x * x).sum(-1, keepdim=True)        # [N,1]
    c_sq = (centroids * centroids).sum(-1)       # [C]
    dot  = x @ centroids.T                       # [N,C]
    dist = x_sq + c_sq.unsqueeze(0) - 2.0 * dot # [N,C]
    assignments = dist.argmin(dim=-1)            # [N]

    # M-step: recompute centroids as group means
    C, d = centroids.shape
    new_centroids = torch.zeros_like(centroids)
    counts = torch.zeros(C, device=x.device, dtype=x.dtype)
    new_centroids.scatter_add_(0, assignments.unsqueeze(-1).expand(-1, d), x)
    counts.scatter_add_(0, assignments, torch.ones(assignments.shape[0], device=x.device, dtype=x.dtype))
    # Avoid division by zero for empty clusters – retain old centroid
    mask = counts > 0
    new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(-1)
    new_centroids[~mask] = centroids[~mask]

    return new_centroids, assignments


# [Optimization 4] torch.compile disabled: reduce-overhead/CUDAGraphs incompatible
# with iterative k-means loop (output overwritten by subsequent run).
_kmeans_step = _kmeans_step_impl


@_disable_compile
def _kmeans(
    x: torch.Tensor,
    num_centroids: int,
    max_iters: int = 20,
    init_centroids: Optional[torch.Tensor] = None,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mini Lloyd's k-means on GPU.

    [Optimization] Uses Flash-KMeans (Triton) when available for 2-5x speedup.
    Falls back to PyTorch Lloyd when Flash-KMeans not installed or init_centroids
    required (Flash-KMeans does not support warm-start).

    Args:
        x:               [N, d]
        num_centroids:   C (fixed at 256, QVG Sec 5.1)
        max_iters:       Lloyd iteration cap
        init_centroids:  [C, d] or None – warm-start (only used in PyTorch fallback)
        tol:             centroid shift convergence threshold

    Returns:
        centroids:    [C, d]  final centroids (BF16/FP16 same as x)
        assignments:  [N]     uint8-compatible indices in [0..C-1]
    """
    N, d = x.shape
    C = min(num_centroids, N)   # guard against tiny chunks

    # [Optimization] Use Flash-KMeans when available and no warm-start needed
    if _FLASH_KMEANS_AVAILABLE and x.is_cuda and init_centroids is None:
        # Flash-KMeans expects (batch, N, d); we have (N, d)
        x_batch = x.unsqueeze(0).float()
        if x_batch.shape[1] >= C:
            cluster_ids, centers, _ = batch_kmeans_Euclid(
                x_batch, n_clusters=C, tol=tol, verbose=False
            )
            centroids = centers[0].to(x.dtype)
            assignments = cluster_ids[0]
            return centroids, assignments

    # PyTorch fallback (supports warm-start)
    if init_centroids is not None and init_centroids.shape[0] == C:
        centroids = init_centroids.clone().to(x.device, x.dtype)
    else:
        idx = torch.randperm(N, device=x.device)[:C]
        centroids = x[idx].clone()

    for _ in range(max_iters):
        new_centroids, assignments = _kmeans_step(x, centroids)
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, assignments


# ─── Semantic-Aware Smoothing ────────────────────────────────────────────────

class SemanticAwareSmoothing:
    """
    QVG Paper Section 4.1 – Semantic-Aware Smoothing.

    For a chunk KV tensor X ∈ ℝ^{N×d}:
      1. k-means cluster tokens along the sequence axis → C=256 groups
      2. Subtract each group's centroid  → residual R = X - centroid[π]
      3. Cache centroids for the next chunk (streaming acceleration, Sec 4.3)

    Usage (per attention head, per layer):
        sas = SemanticAwareSmoothing(num_centroids=256)
        residual, centroids, assignments = sas(chunk_kv, prev_centroids)
        # store centroids for next call
    """

    def __init__(
        self,
        num_centroids: int = 256,   # QVG Sec 5.1 fixed value
        kmeans_max_iters: int = 10,  # [Optimization 2] Reduced from 20
        kmeans_tol: float = 5e-4,    # [Optimization 4] Relaxed for faster convergence
    ):
        self.C = num_centroids
        self.max_iters = kmeans_max_iters
        self.tol = kmeans_tol

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        prev_centroids: Optional[torch.Tensor] = None,
        timing_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               [N, d]  BF16/FP16 KV chunk (one head's tokens)
            prev_centroids:  [C, d]  or None – warm-start for streaming (Sec 4.3)
            timing_dict:     Optional dict for encode_kmeans, encode_centroid_sub

        Returns:
            residual:    [N, d]     R = X - centroid[π],  same dtype as x
            centroids:   [C, d]     k-means centroids,   same dtype as x
            assignments: [N]        uint8-compatible group indices
        """
        import time
        orig_dtype = x.dtype
        xf = x.float()
        cf = prev_centroids.float() if prev_centroids is not None else None

        if timing_dict is not None and x.is_cuda:
            torch.cuda.synchronize()
        t_kmeans0 = time.perf_counter() if timing_dict is not None else 0
        centroids, assignments = _kmeans(
            xf, self.C,
            max_iters=self.max_iters,
            init_centroids=cf,
            tol=self.tol,
        )
        if timing_dict is not None and x.is_cuda:
            torch.cuda.synchronize()
        if timing_dict is not None:
            timing_dict["encode_kmeans"] = timing_dict.get("encode_kmeans", 0.0) + time.perf_counter() - t_kmeans0

        if timing_dict is not None and x.is_cuda:
            torch.cuda.synchronize()
        t_sub0 = time.perf_counter() if timing_dict is not None else 0
        residual = xf - centroids[assignments]
        if timing_dict is not None and x.is_cuda:
            torch.cuda.synchronize()
        if timing_dict is not None:
            timing_dict["encode_centroid_sub"] = timing_dict.get("encode_centroid_sub", 0.0) + time.perf_counter() - t_sub0

        # uint8 for C<=256, int16 for C>256 (e.g. k=512, 1024)
        asgn_dtype = torch.uint8 if self.C <= 256 else torch.int16
        return (
            residual.to(orig_dtype),
            centroids.to(orig_dtype),
            assignments.to(asgn_dtype),
        )

    @torch.no_grad()
    def reconstruct(
        self,
        residual: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert the centroid subtraction:  X̂_{G_i} = R_i + C_{π_i}

        Args:
            residual:    [N, d]
            centroids:   [C, d]
            assignments: [N]  (uint8)

        Returns:
            [N, d]  reconstructed KV tensor
        """
        return residual + centroids[assignments.long()]
