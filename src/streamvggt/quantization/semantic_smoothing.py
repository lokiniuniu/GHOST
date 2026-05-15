"""
Semantic-Aware Smoothing (SAS) for KV Cache Quantization.

Implements QVG paper Section 4.1: Semantic-Aware Smoothing.
Exploits temporal/spatial redundancy in video KV caches via
group-centroid subtraction, producing low-amplitude residuals
that quantize with significantly lower error.

Key idea: group tokens by semantic similarity (k-means on key/value vectors),
subtract per-group centroids, and quantize the residuals instead of raw values.

References:
  - QVG paper Sec 4.1 (Semantic-Aware Smoothing)
  - QVG paper Sec 4.3 (Streaming centroid cache for 3x k-means speedup)
  - QVG paper Fig 6 (Key: ~6.9x error reduction, Value: ~2.6x error reduction)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def _kmeans_lloyd(
    x: torch.Tensor,
    n_clusters: int,
    max_iter: int = 20,
    init_centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd's k-means algorithm on 2D tensor x: [N, D].

    Args:
        x: Input tensor of shape [N, D] (N tokens, D head_dim).
        n_clusters: Number of clusters (fixed C=256 per QVG Sec 5.1).
        max_iter: Maximum number of Lloyd iterations.
        init_centroids: Optional warm-start centroids [C, D].
            Used for streaming centroid cache (QVG Sec 4.3).

    Returns:
        centroids: [C, D] cluster centroids.
        assignments: [N] uint8 cluster assignment indices (fits in uint8 since C=256).
    """
    N, D = x.shape
    device = x.device
    dtype = x.dtype

    if init_centroids is not None:
        centroids = init_centroids.to(device=device, dtype=dtype)
    else:
        # Random initialization: sample n_clusters rows without replacement
        if N >= n_clusters:
            idx = torch.randperm(N, device=device)[:n_clusters]
            centroids = x[idx].clone()
        else:
            # Fewer tokens than clusters: pad by repeating
            repeats = (n_clusters + N - 1) // N
            centroids = x.repeat(repeats, 1)[:n_clusters].clone()

    for _ in range(max_iter):
        # Compute distances [N, C] via (x - c)^2 = ||x||^2 - 2x·c + ||c||^2
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)          # [N, 1]
        c_norm_sq = (centroids * centroids).sum(dim=-1)          # [C]
        dot = x @ centroids.t()                                  # [N, C]
        dist_sq = x_norm_sq - 2.0 * dot + c_norm_sq.unsqueeze(0)  # [N, C]

        assignments = dist_sq.argmin(dim=-1)  # [N]

        # Update centroids as per-cluster means
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=device, dtype=dtype)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), x)
        counts.scatter_add_(0, assignments, torch.ones(N, device=device, dtype=dtype))

        # Avoid division by zero for empty clusters (keep previous centroid)
        mask = counts > 0
        new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]

        if torch.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Return assignments as uint8 (C=256 fits exactly in uint8)
    return centroids, assignments.to(torch.uint8)


class SemanticAwareSmoothing:
    """
    Semantic-Aware Smoothing (SAS) for a single KV chunk.

    Implements the 4-step algorithm from QVG paper Sec 4.1:
      Step 1 – Stream-chunked processing (one frame per chunk ≈1000 tokens)
      Step 2 – k-means grouping along sequence axis with C=256 centroids
      Step 3 – Centroid subtraction to produce low-amplitude residuals
      Step 4 – Streaming centroid cache for 3× k-means speedup (QVG Sec 4.3)

    Usage:
        sas = SemanticAwareSmoothing(n_clusters=256)
        residual, centroids, assignments = sas(kv_chunk, prev_centroids)
        # next call reuses centroids as warm-start
        residual2, centroids2, assignments2 = sas(kv_chunk2, centroids)
    """

    def __init__(self, n_clusters: int = 256, kmeans_iters: int = 20):
        """
        Args:
            n_clusters: Number of k-means clusters (C=256 per QVG Sec 5.1).
                Must be ≤256 so assignments fit in uint8.
            kmeans_iters: Maximum Lloyd iterations.
        """
        assert n_clusters <= 256, "n_clusters must be ≤256 for uint8 assignment storage"
        self.n_clusters = n_clusters
        self.kmeans_iters = kmeans_iters

    def __call__(
        self,
        x: torch.Tensor,
        prev_centroids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply one round of Semantic-Aware Smoothing.

        Args:
            x: KV tensor of shape [N, D] – a single chunk (one frame/head).
               N ≈ 1000 tokens per frame in InfiniteVGGT.
            prev_centroids: Optional [C, D] centroids from the previous chunk.
               When provided they warm-start k-means (QVG Sec 4.3 streaming cache),
               reducing k-means cost by ~3×.

        Returns:
            residual:    [N, D] – x minus its assigned centroid (low-amplitude).
            centroids:   [C, D] – cluster centroids for this chunk (BF16/FP32).
            assignments: [N]    – uint8 centroid assignment index per token.
        """
        # x must be 2D [N, D]
        assert x.ndim == 2, f"Expected 2D input [N, D], got {x.shape}"
        N, D = x.shape

        # Step 2: k-means grouping (warm-start from prev chunk = streaming cache)
        centroids, assignments = _kmeans_lloyd(
            x,
            n_clusters=self.n_clusters,
            max_iter=self.kmeans_iters,
            init_centroids=prev_centroids,
        )

        # Step 3: centroid subtraction → residual
        assigned_centroids = centroids[assignments.long()]  # [N, D]
        residual = x - assigned_centroids                   # [N, D]

        return residual, centroids, assignments


def apply_sas_to_kv_head(
    k: torch.Tensor,
    v: torch.Tensor,
    sas_k: SemanticAwareSmoothing,
    sas_v: SemanticAwareSmoothing,
    prev_centroids_k: Optional[torch.Tensor] = None,
    prev_centroids_v: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Apply SAS independently to K and V for one attention head.

    InfiniteVGGT KV cache layout per head: [B, N, D] after squeezing head dim.
    This helper handles batched input [B, N, D] by processing batch=0 (B=1 inference).

    Args:
        k: Key tensor [B, H, N, D] or [N, D].
        v: Value tensor [B, H, N, D] or [N, D].
        sas_k, sas_v: SAS instances for K and V.
        prev_centroids_k, prev_centroids_v: Warm-start centroids.

    Returns:
        res_k, centroids_k, asgn_k, res_v, centroids_v, asgn_v
    """
    # Flatten to [N, D] for processing, restore shape after
    orig_shape = k.shape
    k_2d = k.reshape(-1, orig_shape[-1])
    v_2d = v.reshape(-1, orig_shape[-1])

    res_k, cen_k, asgn_k = sas_k(k_2d, prev_centroids_k)
    res_v, cen_v, asgn_v = sas_v(v_2d, prev_centroids_v)

    return res_k, cen_k, asgn_k, res_v, cen_v, asgn_v
