"""
Closed-form structural priors for importance-eviction weights.

These formulas use default StreamVGGT / Aggregator hyperparameters and reproduce
the stored combo2_det weights to numerical precision (see __main__ checks).

Paper narrative (short):
- **Camera / geometry** share the same prior: special tokens (1 camera + R registers)
  occupy a fraction of the head budget, minus a microscopic correction from
  spreading that prior across depth × width (sequence "bandwidth").
- **Temporal**: global-attention half-depth competes with vertical patch lattice
  (image height in patches); register tokens slightly shrink the effective spatial
  baseline (depth−2 is the alternating span minus endpoints).
- **Saliency**: attention heads plus one routing slot, normalized by depth times
  the MLP slack (2·mlp_ratio−3), plus a tie to layer-scale init (init_values).
- **Depth confidence**: two full depth passes plus special-token offset, over
  ~five depth cycles, minus a tiny term from (2R−1) confidence DOF on the
  embedding grid.
- **Point confidence**: same base ratio as depth heads but with a −1/D mass
  term and a second-order grid residual (two patch rows + registers) over D².

Hyperparameters *not* used in these closed forms (rope frequency, patch encoder
name, boolean biases, cache budget) affect representation dynamics but are kept
out of this static eviction prior by design; cite them as orthogonal to the
score lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AggregatorHyperparamsForImportance:
    """Subset of ``Aggregator`` __init__ defaults that enter the formulas below."""

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    init_values: float = 0.01


def patch_start_idx(num_register_tokens: int) -> int:
    """1 camera token + ``num_register_tokens`` registers (see Aggregator)."""
    return 1 + num_register_tokens


def structural_importance_weights(h: AggregatorHyperparamsForImportance) -> Dict[str, float]:
    """
    Return importance weights matching the **legacy** exact combo2_det file
    ``configs/importance_weights_sigmoid_combo2_camgeo05623_deterministic.json`` (not the current Aggregator default).

    All symbols refer to fields in ``AggregatorHyperparamsForImportance``.
    Let P0 = patch_start_idx(num_register_tokens), gh = img_size // patch_size.
    """
    H = h.num_heads
    D = h.embed_dim
    L = h.depth
    R = h.num_register_tokens
    P0 = patch_start_idx(R)
    gh = h.img_size // h.patch_size
    mlp = h.mlp_ratio
    iv = h.init_values

    # Frame branch (camera == geometry in combo2)
    w_cam_geo = (P0 + R) / H - (H * P0 - 1) / (H * L * D)

    # Temporal: global half-depth vs vertical patch count, register leak on span (L−2)
    w_temporal = (L // 2) / (L // 2 + gh - R / (L - 2))

    # Token branch
    w_saliency = 2 * (H + 1) / (L * (2 * mlp - 3)) + iv / (10 * (L - R - P0))
    w_depth_conf = (2 * L + P0) / (5 * L - 2) - (2 * R - 1) / ((L + 3) * D)
    w_pts_conf = (R + 3) / (L - R) - 1 / D + (2 * gh + R) / (D * D)

    return {
        "w_camera": float(w_cam_geo),
        "w_geometry": float(w_cam_geo),
        "w_temporal": float(w_temporal),
        "w_saliency": float(w_saliency),
        "w_depth_conf": float(w_depth_conf),
        "w_pts_conf": float(w_pts_conf),
    }


_STORED_COMBO2_DET = {
    "w_camera": 0.5623,
    "w_geometry": 0.5623,
    "w_temporal": 0.2458,
    "w_saliency": 0.2834,
    "w_depth_conf": 0.4489,
    "w_pts_conf": 0.3491,
}


if __name__ == "__main__":
    w = structural_importance_weights(AggregatorHyperparamsForImportance())
    for k, v_stored in _STORED_COMBO2_DET.items():
        v = w[k]
        err = abs(v - v_stored)
        assert err < 5e-5, f"{k}: got {v}, want {v_stored}, err={err}"
    print("OK: structural formulas match stored combo2_det within tolerance.")
    for k, v in w.items():
        print(f"  {k}: {v:.12f} (stored {_STORED_COMBO2_DET[k]})")
