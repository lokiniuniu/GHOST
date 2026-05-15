"""
Importance-based eviction for KV cache pruning.

Uses multi-modal importance scores:
- Frame-level: camera pose change, depth gradient variance, temporal proximity
- Token-level: patch token spatial gradient (visual saliency), depth_conf, pts3d_conf

Supports incremental computation with cache to avoid O(num_frames^2) redundant work.
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F


def _sigmoid_norm(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid normalization used before weighted sum."""
    return torch.sigmoid(x)


def _pose_change_score(pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pose change between two frames (absT_quaR_FoV encoding).
    Returns scalar score: higher = more change = more important.
    - Translation: L2 norm of T difference
    - Rotation: 1 - |quat_dot| (angle between quaternions)
    """
    T_a, quat_a = pose_a[..., :3], pose_a[..., 3:7]
    T_b, quat_b = pose_b[..., :3], pose_b[..., 3:7]
    trans_diff = (T_a - T_b).norm(dim=-1)
    quat_a = F.normalize(quat_a, p=2, dim=-1)
    quat_b = F.normalize(quat_b, p=2, dim=-1)
    quat_dot = (quat_a * quat_b).sum(dim=-1).abs().clamp(0, 1)
    rot_diff = 1.0 - quat_dot
    return trans_diff + rot_diff


def compute_frame_importance(
    frame_metadata: List[Dict[str, torch.Tensor]],
    current_frame_idx: int,
    w_camera: float = 0.3,
    w_geometry: float = 0.2,
    w_temporal: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute frame-level importance for frames 0..current_frame_idx.

    Args:
        frame_metadata: List of dicts, each with keys: camera_pose [B,9], depth [B,H,W,1],
            depth_conf [B,H,W], pts3d_conf [B,H,W]. Length = current_frame_idx (past frames only).
            For current frame we use defaults.
        current_frame_idx: Index of current frame (0-based).
        w_camera, w_geometry, w_temporal: Weights for weighted average.
        device: Output device.

    Returns:
        frame_importance: [num_frames] in [0, 1], higher = more important.
    """
    num_frames = current_frame_idx + 1
    if device is None and frame_metadata:
        d = next((m["camera_pose"] for m in frame_metadata if "camera_pose" in m), None)
        device = d.device if d is not None else torch.device("cpu")
    dtype = torch.float32

    camera_scores = torch.zeros(num_frames, device=device, dtype=dtype)
    geometry_scores = torch.zeros(num_frames, device=device, dtype=dtype)
    temporal_scores = torch.zeros(num_frames, device=device, dtype=dtype)

    for t in range(num_frames):
        if t == current_frame_idx:
            temporal_scores[t] = 1.0
            camera_scores[t] = 0.5
            geometry_scores[t] = 0.5
            continue

        meta = frame_metadata[t] if t < len(frame_metadata) else {}
        pose = meta.get("camera_pose")
        depth = meta.get("depth")
        if pose is not None and device is not None:
            pose = pose.to(device)
        if depth is not None and device is not None:
            depth = depth.to(device)

        if pose is not None and t > 0 and (t - 1) < len(frame_metadata):
            prev_meta = frame_metadata[t - 1]
            prev_pose = prev_meta.get("camera_pose")
            if prev_pose is not None and device is not None:
                prev_pose = prev_pose.to(device)
            if prev_pose is not None:
                change = _pose_change_score(
                    prev_pose.float().mean(dim=0, keepdim=True),
                    pose.float().mean(dim=0, keepdim=True),
                )
                camera_scores[t] = change.squeeze().item()
            else:
                camera_scores[t] = 0.5
        elif pose is not None and t == 0:
            camera_scores[t] = 0.5
        else:
            camera_scores[t] = 0.5

        if depth is not None:
            d = depth.squeeze()
            if d.dim() == 3:
                d = d[0]
            gx = (d[:, 1:] - d[:, :-1]).float()
            gy = (d[1:, :] - d[:-1, :]).float()
            gx = F.pad(gx.unsqueeze(0), (0, 1), mode="replicate").squeeze(0)
            gy = F.pad(gy.unsqueeze(0), (0, 0, 0, 1), mode="replicate").squeeze(0)
            mag = (gx ** 2 + gy ** 2).sqrt()
            geometry_scores[t] = mag.var().item()
        else:
            geometry_scores[t] = 0.5

        temporal_scores[t] = (t + 1) / num_frames

    camera_scores = _sigmoid_norm(camera_scores)
    geometry_scores = _sigmoid_norm(geometry_scores)
    temporal_scores = _sigmoid_norm(temporal_scores)

    frame_importance = torch.zeros(num_frames, device=device, dtype=dtype)
    for t in range(num_frames):
        frame_importance[t] = (
            w_camera * camera_scores[t]
            + w_geometry * geometry_scores[t]
            + w_temporal * temporal_scores[t]
        )
    frame_importance = frame_importance / (frame_importance.max() + 1e-8)
    return frame_importance


def compute_patch_saliency(patch_tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """
    Compute spatial gradient magnitude of patch tokens (visual saliency).
    Higher = edges/corners = more important.

    Args:
        patch_tokens: [B*S, N, C] where N = patch_h * patch_w
        patch_h, patch_w: Spatial dimensions.

    Returns:
        saliency: [patch_h, patch_w] (mean over batch)
    """
    B, N, C = patch_tokens.shape
    x = patch_tokens.reshape(B, patch_h, patch_w, C)
    gx = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum(dim=-1)
    gy = (x[:, 1:, :, :] - x[:, :-1, :, :]).pow(2).sum(dim=-1)
    gx = F.pad(gx, (0, 1), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    mag = (gx + gy).sqrt()
    return mag.mean(dim=0)


def pool_to_patch(
    conf: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """Average pool [B, H, W] to [B, patch_h, patch_w]."""
    if conf.dim() == 2:
        conf = conf.unsqueeze(0)
    return F.adaptive_avg_pool2d(conf, (patch_h, patch_w))


def compute_token_importance(
    patch_tokens: Optional[torch.Tensor],
    frame_metadata: List[Dict[str, torch.Tensor]],
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    patch_start_idx: int,
    w_saliency: float = 0.2,
    w_depth_conf: float = 0.45,
    w_pts_conf: float = 0.35,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute token-level importance for patch tokens across all frames.
    Returns [num_frames, n_patch] where n_patch = patch_h * patch_w.
    """
    n_patch = patch_h * patch_w
    num_frames = current_frame_idx + 1
    if device is None:
        device = patch_tokens.device if patch_tokens is not None else torch.device("cpu")
    dtype = torch.float32

    token_importance = torch.zeros(num_frames, n_patch, device=device, dtype=dtype)

    for t in range(num_frames):
        saliency = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
        depth_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
        pts_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5

        if t == current_frame_idx and patch_tokens is not None:
            sal = compute_patch_saliency(patch_tokens, patch_h, patch_w)
            saliency = sal.flatten()

        if t < len(frame_metadata):
            meta = frame_metadata[t]
            if "depth_conf" in meta:
                dc = meta["depth_conf"]
                if device is not None:
                    dc = dc.to(device)
                if dc.dim() == 3:
                    dc = dc[0]
                dc = pool_to_patch(dc.unsqueeze(0)
                    if dc.dim() == 2 else dc, patch_h, patch_w)
                depth_conf = dc.flatten()
            if "pts3d_conf" in meta or "conf" in meta:
                pc = meta.get("pts3d_conf") or meta.get("conf")
                if pc is not None:
                    if device is not None:
                        pc = pc.to(device)
                    if pc.dim() == 3:
                        pc = pc[0]
                    pc = pool_to_patch(pc.unsqueeze(0)
                        if pc.dim() == 2 else pc, patch_h, patch_w)
                    pts_conf = pc.flatten()

        token_importance[t] = (
            w_saliency * _sigmoid_norm(saliency)
            + w_depth_conf * _sigmoid_norm(depth_conf)
            + w_pts_conf * _sigmoid_norm(pts_conf)
        )

    token_importance = token_importance / (token_importance.max() + 1e-8)
    return token_importance


def compute_combined_importance(
    frame_importance: torch.Tensor,
    token_importance: torch.Tensor,
    num_frames: int,
    n_patch: int,
    w_frame: float = 0.5,
    w_token: float = 0.5,
    patch_start_idx: int = 0,
    special_token_boost: float = 0.3,
    special_token_tiebreak_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Combine frame and token importance for all tokens.

    When patch_start_idx > 0, special tokens (camera + register) get:
      score = frame_importance[t] + special_token_boost + eps * rank_id
    where rank_id is deterministic within a frame (camera, reg0, reg1, ...).
    This deterministically breaks ties without introducing randomness.

    Returns [num_frames * tokens_per_frame] where tokens_per_frame = patch_start_idx + n_patch
    when patch_start_idx > 0, else [num_frames * n_patch].
    Order per frame: [camera, register x4, patch_0, ..., patch_{n_patch-1}].
    """
    tokens_per_frame = (patch_start_idx + n_patch) if patch_start_idx > 0 else n_patch
    combined = torch.zeros(
        num_frames * tokens_per_frame,
        device=frame_importance.device,
        dtype=frame_importance.dtype,
    )
    for t in range(num_frames):
        base = t * tokens_per_frame
        if patch_start_idx > 0:
            # Special tokens: frame_importance[t] + boost + deterministic tiebreak
            special_base = frame_importance[t].item() + special_token_boost
            rank_id = torch.arange(
                patch_start_idx, device=frame_importance.device, dtype=frame_importance.dtype
            )
            combined[base : base + patch_start_idx] = (
                special_base + special_token_tiebreak_eps * rank_id
            )
            # Patch tokens: frame + token importance
            patch_start = base + patch_start_idx
            patch_end = base + tokens_per_frame
            combined[patch_start:patch_end] = (
                w_frame * frame_importance[t]
                + w_token * token_importance[t]
            )
        else:
            combined[base : base + n_patch] = (
                w_frame * frame_importance[t]
                + w_token * token_importance[t]
            )
    return combined / (combined.max() + 1e-8)


def _compute_camera_score_for_frame(
    frame_metadata: List[Dict[str, torch.Tensor]],
    t: int,
    device: Optional[torch.device] = None,
) -> float:
    """Compute raw camera (pose change) score for a single frame t."""
    if t == 0:
        return 0.5
    meta = frame_metadata[t] if t < len(frame_metadata) else {}
    prev_meta = frame_metadata[t - 1] if (t - 1) < len(frame_metadata) else {}
    pose = meta.get("camera_pose")
    prev_pose = prev_meta.get("camera_pose")
    if pose is None or prev_pose is None:
        return 0.5
    if device is not None:
        pose = pose.to(device)
        prev_pose = prev_pose.to(device)
    change = _pose_change_score(
        prev_pose.float().mean(dim=0, keepdim=True),
        pose.float().mean(dim=0, keepdim=True),
    )
    return change.squeeze().item()


def _compute_geometry_score_for_frame(
    frame_metadata: List[Dict[str, torch.Tensor]],
    t: int,
    device: Optional[torch.device] = None,
) -> float:
    """Compute raw geometry (depth gradient variance) score for a single frame t."""
    meta = frame_metadata[t] if t < len(frame_metadata) else {}
    depth = meta.get("depth")
    if depth is None:
        return 0.5
    if device is not None:
        depth = depth.to(device)
    d = depth.squeeze()
    if d.dim() == 3:
        d = d[0]
    gx = (d[:, 1:] - d[:, :-1]).float()
    gy = (d[1:, :] - d[:-1, :]).float()
    gx = F.pad(gx.unsqueeze(0), (0, 1), mode="replicate").squeeze(0)
    gy = F.pad(gy.unsqueeze(0), (0, 0, 0, 1), mode="replicate").squeeze(0)
    mag = (gx ** 2 + gy ** 2).sqrt()
    return mag.var().item()


def _get_raw_token_components_for_frame(
    patch_tokens: Optional[torch.Tensor],
    frame_metadata: List[Dict[str, torch.Tensor]],
    t: int,
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get raw saliency, depth_conf, pts_conf for a frame (before weighting). Returns (saliency, depth_conf, pts_conf)."""
    n_patch = patch_h * patch_w
    if device is None:
        device = patch_tokens.device if patch_tokens is not None else torch.device("cpu")
    dtype = torch.float32

    saliency = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
    depth_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
    pts_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5

    if t == current_frame_idx and patch_tokens is not None:
        sal = compute_patch_saliency(patch_tokens, patch_h, patch_w)
        saliency = sal.flatten().clone()
        # Return BEFORE /max for raw saliency

    if t < len(frame_metadata):
        meta = frame_metadata[t]
        if "depth_conf" in meta:
            dc = meta["depth_conf"]
            if device is not None:
                dc = dc.to(device)
            if dc.dim() == 3:
                dc = dc[0]
            dc = pool_to_patch(dc.unsqueeze(0) if dc.dim() == 2 else dc, patch_h, patch_w)
            depth_conf = dc.flatten()
        if "pts3d_conf" in meta or "conf" in meta:
            pc = meta.get("pts3d_conf") or meta.get("conf")
            if pc is not None:
                if device is not None:
                    pc = pc.to(device)
                if pc.dim() == 3:
                    pc = pc[0]
                pc = pool_to_patch(pc.unsqueeze(0) if pc.dim() == 2 else pc, patch_h, patch_w)
                pts_conf = pc.flatten()

    return saliency, depth_conf, pts_conf


def _compute_token_importance_for_frame(
    patch_tokens: Optional[torch.Tensor],
    frame_metadata: List[Dict[str, torch.Tensor]],
    t: int,
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    w_saliency: float = 0.2,
    w_depth_conf: float = 0.45,
    w_pts_conf: float = 0.35,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute token importance for a single frame t. Returns [n_patch]."""
    n_patch = patch_h * patch_w
    if device is None:
        device = patch_tokens.device if patch_tokens is not None else torch.device("cpu")
    dtype = torch.float32

    saliency = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
    depth_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5
    pts_conf = torch.ones(n_patch, device=device, dtype=dtype) * 0.5

    if t == current_frame_idx and patch_tokens is not None:
        sal = compute_patch_saliency(patch_tokens, patch_h, patch_w)
        saliency = sal.flatten()

    if t < len(frame_metadata):
        meta = frame_metadata[t]
        if "depth_conf" in meta:
            dc = meta["depth_conf"]
            if device is not None:
                dc = dc.to(device)
            if dc.dim() == 3:
                dc = dc[0]
            dc = pool_to_patch(dc.unsqueeze(0) if dc.dim() == 2 else dc, patch_h, patch_w)
            depth_conf = dc.flatten()
        if "pts3d_conf" in meta or "conf" in meta:
            pc = meta.get("pts3d_conf") or meta.get("conf")
            if pc is not None:
                if device is not None:
                    pc = pc.to(device)
                if pc.dim() == 3:
                    pc = pc[0]
                pc = pool_to_patch(pc.unsqueeze(0) if pc.dim() == 2 else pc, patch_h, patch_w)
                pts_conf = pc.flatten()

    return (
        w_saliency * _sigmoid_norm(saliency)
        + w_depth_conf * _sigmoid_norm(depth_conf)
        + w_pts_conf * _sigmoid_norm(pts_conf)
    )


def compute_importance_incremental(
    frame_metadata: List[Dict[str, torch.Tensor]],
    patch_tokens: Optional[torch.Tensor],
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    patch_start_idx: int,
    importance_cache: Optional[Dict[str, Any]] = None,
    cached_importance_scores: Optional[torch.Tensor] = None,
    w_camera: float = 0.3,
    w_geometry: float = 0.2,
    w_temporal: float = 0.5,
    w_saliency: float = 0.2,
    w_depth_conf: float = 0.45,
    w_pts_conf: float = 0.35,
    w_frame: float = 0.5,
    w_token: float = 0.5,
    special_token_boost: float = 0.3,
    special_token_tiebreak_eps: float = 1e-6,
    special_token_noise_scale: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute importance scores using cache for token rows (one append per new frame).
    Camera/geometry raw scores are rebuilt each step from the full ``frame_metadata`` so
    each index t sees metadata[t] (incremental append previously left them at 0.5).
    When cached_importance_scores is provided (from prior eviction), compute only for current
    frame and concat, so importance_scores always matches num_candidates.

    Returns:
        importance_scores: [num_candidate_tokens] for candidate frames (sliced for eviction)
        cache: Updated cache for next frame
    """
    n_patch = patch_h * patch_w
    num_frames = current_frame_idx + 1
    tokens_per_frame = (patch_start_idx + n_patch) if patch_start_idx > 0 else n_patch
    num_anchor_frames = 1

    # Backward compatibility: old configs may still pass special_token_noise_scale.
    # We map it to deterministic tiebreak magnitude (no randomness).
    if special_token_tiebreak_eps == 1e-6 and special_token_noise_scale > 0:
        special_token_tiebreak_eps = float(special_token_noise_scale)

    if cached_importance_scores is not None:
        new_frame_importance = compute_importance_for_current_frame_only(
            patch_tokens, frame_metadata, current_frame_idx,
            patch_h, patch_w, patch_start_idx,
            w_frame=w_frame, w_token=w_token,
            special_token_boost=special_token_boost,
            special_token_tiebreak_eps=special_token_tiebreak_eps,
            special_token_noise_scale=special_token_noise_scale,
            device=device,
        )
        importance_scores = torch.cat([
            cached_importance_scores.to(new_frame_importance.device).to(new_frame_importance.dtype),
            new_frame_importance,
        ], dim=0)
        if importance_cache is not None:
            importance_cache["num_frames"] = num_frames
        return importance_scores, importance_cache if importance_cache is not None else {}

    if device is None and frame_metadata:
        d = next((m.get("camera_pose") for m in frame_metadata if "camera_pose" in m), None)
        device = d.device if d is not None else torch.device("cpu")
    dtype = torch.float32

    if importance_cache is None or importance_cache.get("num_frames", 0) == 0:
        cache = {
            "camera_scores_raw": [],
            "geometry_scores_raw": [],
            "token_importance_list": [],
            "last_meta_len_updated": 0,
            "n_patch": n_patch,
            "patch_start_idx": patch_start_idx,
        }
    else:
        cache = importance_cache

    camera_raw = cache["camera_scores_raw"]
    geometry_raw = cache["geometry_scores_raw"]
    token_list = cache["token_importance_list"]

    if len(token_list) < num_frames:
        for t in range(len(token_list), num_frames):
            tok_t = _compute_token_importance_for_frame(
                patch_tokens, frame_metadata, t, current_frame_idx,
                patch_h, patch_w, w_saliency, w_depth_conf, w_pts_conf, device
            )
            token_list.append(tok_t)

    # Delay-one-frame incremental update:
    # At step i, frame_metadata has indices [0..i-1], so only those can be computed from real metadata.
    if len(camera_raw) < num_frames:
        camera_raw.extend([0.5] * (num_frames - len(camera_raw)))
    if len(geometry_raw) < num_frames:
        geometry_raw.extend([0.5] * (num_frames - len(geometry_raw)))

    last_meta_len_updated = int(cache.get("last_meta_len_updated", 0))
    available_meta_len = min(len(frame_metadata), num_frames)
    if available_meta_len > last_meta_len_updated:
        for t in range(last_meta_len_updated, available_meta_len):
            camera_raw[t] = _compute_camera_score_for_frame(frame_metadata, t, device)
            geometry_raw[t] = _compute_geometry_score_for_frame(frame_metadata, t, device)
        cache["last_meta_len_updated"] = available_meta_len

    camera_scores = torch.tensor(camera_raw, device=device, dtype=dtype)
    geometry_scores = torch.tensor(geometry_raw, device=device, dtype=dtype)
    camera_scores = _sigmoid_norm(camera_scores)
    geometry_scores = _sigmoid_norm(geometry_scores)

    temporal_scores = torch.tensor(
        [(t + 1) / num_frames for t in range(num_frames)],
        device=device,
        dtype=dtype,
    )
    temporal_scores = _sigmoid_norm(temporal_scores)

    frame_importance = (
        w_camera * camera_scores
        + w_geometry * geometry_scores
        + w_temporal * temporal_scores
    )
    frame_importance = frame_importance / (frame_importance.max() + 1e-8)

    token_importance = torch.stack(token_list, dim=0)
    token_importance = token_importance / (token_importance.max() + 1e-8)

    importance_scores = compute_combined_importance(
        frame_importance,
        token_importance,
        num_frames,
        n_patch,
        w_frame=w_frame,
        w_token=w_token,
        patch_start_idx=patch_start_idx,
        special_token_boost=special_token_boost,
        special_token_tiebreak_eps=special_token_tiebreak_eps,
    )

    num_candidate_frames = max(0, num_frames - num_anchor_frames)
    if num_candidate_frames > 0:
        start = num_anchor_frames * tokens_per_frame
        end = num_frames * tokens_per_frame
        importance_scores = importance_scores[start:end]
    else:
        importance_scores = None

    cache["num_frames"] = num_frames

    # [Profile] Print when num_frames >= _profile_min_frames (default 300)
    _pmf = int(importance_cache.get("_profile_min_frames", 300)) if importance_cache else 300
    if (importance_cache is not None and importance_cache.get("_profile_raw")
            and not importance_cache.get("_profile_printed") and num_frames >= _pmf):
        _print_importance_raw_stats(
            num_frames, temporal_scores,
            frame_metadata, patch_tokens, current_frame_idx, patch_h, patch_w, device
        )
        importance_cache["_profile_printed"] = True

    return importance_scores, cache


def _print_importance_raw_stats(
    num_frames: int,
    temporal_scores: torch.Tensor,
    frame_metadata: List[Dict],
    patch_tokens: Optional[torch.Tensor],
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    device: Optional[torch.device],
) -> None:
    """Print raw stats, sigmoid values, and suggested coefficients to match current weights.

    Camera/geometry are recomputed from the final ``frame_metadata`` (incremental cache
    rows were computed before metadata[t] existed and were not representative).
    """
    import numpy as np

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    # Align with token metrics: use full metadata available at profile time (not stale cache).
    cam_arr = np.array(
        [_compute_camera_score_for_frame(frame_metadata, t, device) for t in range(num_frames)]
    )
    geo_arr = np.array(
        [_compute_geometry_score_for_frame(frame_metadata, t, device) for t in range(num_frames)]
    )
    temp_arr = temporal_scores.cpu().numpy()

    sal_means, depth_means, pts_means = [], [], []
    sal_maxs = []
    for t in range(num_frames):
        sal, dc, pc = _get_raw_token_components_for_frame(
            patch_tokens, frame_metadata, t, current_frame_idx, patch_h, patch_w, device
        )
        sal_c = sal.cpu().numpy()
        dc_c = dc.cpu().numpy()
        pc_c = pc.cpu().numpy()
        sal_means.append(sal_c.mean())
        sal_maxs.append(sal_c.max())
        depth_means.append(dc_c.mean())
        pts_means.append(pc_c.mean())

    sal_arr = np.array(sal_means)
    depth_arr = np.array(depth_means)
    pts_arr = np.array(pts_means)

    # Current weights (default: improved_default_cgt_15)
    w_cam, w_geo, w_temp = 0.3, 0.2, 0.5
    w_sal, w_depth, w_pts = 0.2, 0.45, 0.35

    # Normalized (current): each is scaled by max to [0,1], then weighted
    # For sigmoid path: we want w_new * mean(sigmoid(raw)) ≈ w_old * mean(normalized)
    cam_norm = cam_arr / (cam_arr.max() + 1e-8)
    geo_norm = geo_arr / (geo_arr.max() + 1e-8)
    sal_norm = sal_arr / (np.array(sal_maxs) + 1e-8)
    depth_norm = np.clip(depth_arr, 0, 1)
    pts_norm = np.clip(pts_arr, 0, 1)
    temp_norm = temp_arr

    def stats(name, raw, norm):
        r_mean, r_std, r_min, r_max = raw.mean(), raw.std(), raw.min(), raw.max()
        sig = sigmoid(raw)
        s_mean = sig.mean()
        n_mean = norm.mean()
        return r_mean, r_std, r_min, r_max, s_mean, n_mean

    metrics = [
        ("camera", cam_arr, cam_norm),
        ("geometry", geo_arr, geo_norm),
        ("temporal", temp_arr, temp_norm),
        ("saliency", sal_arr, sal_norm),
        ("depth_conf", depth_arr, depth_norm),
        ("pts_conf", pts_arr, pts_norm),
    ]
    weights = {"camera": w_cam, "geometry": w_geo, "temporal": w_temp, "saliency": w_sal, "depth_conf": w_depth, "pts_conf": w_pts}

    print("\n" + "=" * 80)
    print("[Importance Profile] Raw stats (per frame mean), sigmoid, normalized. Suggested coef for sigmoid path.")
    print("=" * 80)
    print(f"{'metric':<12} {'raw_mean':>10} {'raw_std':>10} {'raw_min':>10} {'raw_max':>10} {'sigmoid_mean':>12} {'norm_mean':>10} {'w_current':>10}")
    print("-" * 80)

    coef_suggest = {}
    for name, raw, norm in metrics:
        r_mean, r_std, r_min, r_max, s_mean, n_mean = stats(name, raw, norm)
        w_cur = weights[name]
        print(f"{name:<12} {r_mean:>10.4f} {r_std:>10.4f} {r_min:>10.4f} {r_max:>10.4f} {s_mean:>12.4f} {n_mean:>10.4f} {w_cur:>10.2f}")

        if s_mean > 1e-6:
            coef_suggest[name] = (w_cur * n_mean) / s_mean
        else:
            coef_suggest[name] = w_cur

    print("-" * 80)
    print("If using sigmoid(raw) with uniform base w_base, suggested coefficient per metric:")
    print("  contrib_current = w * norm_mean,  contrib_sigmoid = coef * sigmoid_mean")
    print("  To match: coef = w_current * norm_mean / sigmoid_mean")
    for name in coef_suggest:
        w_cur = weights[name]
        print(f"  {name}: coef = {coef_suggest[name]:.4f}  (vs w_current={w_cur:.2f}, factor {coef_suggest[name]/w_cur:.4f}x)")
    print("=" * 80 + "\n")


def compute_importance_for_current_frame_only(
    patch_tokens: Optional[torch.Tensor],
    frame_metadata: List[Dict[str, torch.Tensor]],
    current_frame_idx: int,
    patch_h: int,
    patch_w: int,
    patch_start_idx: int,
    w_frame: float = 0.5,
    w_token: float = 0.5,
    special_token_boost: float = 0.3,
    special_token_tiebreak_eps: float = 1e-6,
    special_token_noise_scale: float = 0.0,
    w_saliency: float = 0.2,
    w_depth_conf: float = 0.45,
    w_pts_conf: float = 0.35,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute importance for only the current frame. Returns [tokens_per_frame].
    Used when we have cached_importance_scores from prior eviction.
    """
    n_patch = patch_h * patch_w
    tokens_per_frame = (patch_start_idx + n_patch) if patch_start_idx > 0 else n_patch
    num_frames = 1

    if device is None:
        device = patch_tokens.device if patch_tokens is not None else torch.device("cpu")
    dtype = torch.float32

    frame_importance = torch.tensor([1.0], device=device, dtype=dtype)
    token_importance = _compute_token_importance_for_frame(
        patch_tokens, frame_metadata, current_frame_idx, current_frame_idx,
        patch_h, patch_w, w_saliency, w_depth_conf, w_pts_conf, device
    ).unsqueeze(0)

    # Backward compatibility: old configs may still pass special_token_noise_scale.
    # We map it to deterministic tiebreak magnitude (no randomness).
    if special_token_tiebreak_eps == 1e-6 and special_token_noise_scale > 0:
        special_token_tiebreak_eps = float(special_token_noise_scale)

    combined = compute_combined_importance(
        frame_importance,
        token_importance,
        num_frames,
        n_patch,
        w_frame=w_frame,
        w_token=w_token,
        patch_start_idx=patch_start_idx,
        special_token_boost=special_token_boost,
        special_token_tiebreak_eps=special_token_tiebreak_eps,
    )
    return combined
