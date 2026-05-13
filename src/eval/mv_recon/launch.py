import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import random
import torch
import argparse
import numpy as np
try:
    import open3d as o3d
except Exception:
    o3d = None
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument(
        "--scenes_root",
        type=str,
        default=None,
        help="Path to 7scenes dataset root directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="7scenes",
        choices=["7scenes", "nrgbd", "long3d", "all"],
        help="Evaluation dataset selection. Default: 7scenes.",
    )
    parser.add_argument(
        "--nrgbd_root",
        type=str,
        default=None,
        help="Path to NRGBD dataset root directory.",
    )
    parser.add_argument(
        "--long3d_root",
        type=str,
        default=None,
        help="Path to Long3D dataset root directory.",
    )
    parser.add_argument(
        "--long3d_extract_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Long3D: auto-extract images.7z when images are missing.",
    )
    parser.add_argument(
        "--long3d_max_points",
        type=int,
        default=1200000,
        help="For Long3D eval: max sampled points for pred/gt point clouds.",
    )
    parser.add_argument(
        "--long3d_chamfer_max_dist",
        type=float,
        default=1.0,
        help="For Long3D eval: Chamfer distance clipping upper bound.",
    )
    parser.add_argument(
        "--long3d_chunk_size",
        type=int,
        default=256,
        help="For Long3D: frames per inference chunk. <=0 disables chunking.",
    )
    parser.add_argument(
        "--eviction_mode",
        type=str,
        default="importance",
        choices=["importance"],
        help="KV cache eviction strategy. Default: importance.",
    )
    parser.add_argument(
        "--budget_proportions_path",
        type=str,
        default=None,
        help="Path to precomputed KV budget proportions JSON (from compute_kv_budget_strategies.py)",
    )
    parser.add_argument(
        "--budget_strategy",
        type=str,
        default=None,
        choices=["cosine", "fisher", "gradient", "hessian", "pruning"],
        help="Stratification strategy for layer budget: cosine, fisher, gradient, hessian, pruning. Uses configs/kv_budget_proportions_{strategy}.json",
    )
    parser.add_argument(
        "--use_cosine_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use cosine-similarity-based layer budget (configs/kv_budget_proportions.json). "
            "Default: enabled. Use --no-use_cosine_budget to disable. "
            "Overridden by --budget_strategy if set. Deprecated alternative: --budget_strategy cosine."
        ),
    )
    parser.add_argument(
        "--use_importance_in_attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When using importance-based eviction modes, multiply key vectors by importance before attention "
            "(K' = K * w). Use --softmax_importance_before_k for w = softmax(imp)*n on candidates. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--softmax_importance_before_k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --use_importance_in_attn: on candidate keys, w = softmax(importance) * n_candidates; "
            "anchor keys use w=1. Default: disabled (use clamped raw importance)."
        ),
    )
    parser.add_argument(
        "--debug_importance_in_attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug log for importance-in-attention. Prints per-frame/per-layer whether "
            "K reweighting was applied and basic importance stats."
        ),
    )
    parser.add_argument(
        "--importance_weights_path",
        type=str,
        default=None,
        help=(
            "JSON path for importance weights (w_camera, w_geometry, w_temporal, w_saliency, w_depth_conf, "
            "w_pts_conf, w_frame, w_token; optional special_token_boost, special_token_tiebreak_eps, "
            "special_token_noise_scale[legacy->deterministic tiebreak]). "
            "Used when eviction_mode is importance. "
            "Omit this and --importance_preset to use default weights "
            "(configs/importance_weights_default.json)."
        ),
    )
    parser.add_argument(
        "--importance_preset",
        type=int,
        default=None,
        help=(
            "Index of preset in configs/importance_weights_presets.json. "
            "Overrides importance_weights_path. Omit both preset and path to use default near01_01."
        ),
    )
    parser.add_argument(
        "--test_id",
        type=str,
        default=None,
        help="Limit 7scenes to single scene (e.g. chess, fire). Useful for quick subset eval.",
    )
    parser.add_argument(
        "--seq_id",
        type=str,
        default=None,
        help=(
            "7scenes only: keep only this sequence id after TestSplit normalization "
            "(e.g. seq-14). Requires --test_id to be the scene name (e.g. redkitchen)."
        ),
    )
    parser.add_argument(
        "--profile_importance_raw",
        action="store_true",
        help="Print raw importance stats and sigmoid coefficients (use with small subset).",
    )
    parser.add_argument(
        "--profile_min_frames",
        type=int,
        default=300,
        help="With --profile_importance_raw, print stats after this many frames (default 300).",
    )
    parser.add_argument(
        "--eval_repeat",
        type=int,
        default=1,
        help="Run the full evaluation this many times in one process. When >1, writes to output_dir/run_0/, run_1/, ...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for torch/numpy/random; enables cudnn deterministic mode for more reproducible runs.",
    )
    parser.add_argument(
        "--speed_only",
        action="store_true",
        help="Only run model inference timing; skip reconstruction metrics and point-cloud evaluation.",
    )
    parser.add_argument(
        "--total_budget",
        type=int,
        default=1200000,
        help="Total KV token budget passed to StreamVGGT (scales per-layer budgets from proportions). Default: 1200000.",
    )
    parser.add_argument(
        "--kv_share_method",
        type=str,
        default="none",
        choices=["none", "layer_group", "coarse", "delta"],
        help="Experimental KV-sharing method across adjacent layers.",
    )
    parser.add_argument("--kv_share_group_size", type=int, default=2, help="Layer grouping size for layer_group.")
    parser.add_argument("--kv_share_heads_ratio", type=float, default=0.5, help="Head sharing ratio for layer_group.")
    parser.add_argument("--kv_share_token_ratio", type=float, default=0.5, help="Token sharing ratio for layer_group.")
    parser.add_argument("--kv_coarse_start_layer", type=int, default=12, help="Start layer index for coarse sharing.")
    parser.add_argument("--kv_coarse_stride", type=int, default=4, help="Far-memory stride for coarse sharing.")
    parser.add_argument("--kv_coarse_near_frames", type=int, default=4, help="Near full-res frames for coarse sharing.")
    parser.add_argument("--kv_delta_start_layer", type=int, default=1, help="Start layer index for delta sharing.")
    parser.add_argument("--kv_delta_keep_ratio", type=float, default=0.5, help="Token keep ratio for delta sharing.")
    parser.add_argument(
        "--kv_target_budget",
        type=int,
        default=None,
        help=(
            "Optional KV compression target budget used by kv_share_method. "
            "When set, total_budget still controls layer budget allocation, and kv sharing "
            "additionally compresses historical KV toward kv_target_budget / total_budget."
        ),
    )
    parser.add_argument(
        "--frame_sparse_mode",
        type=str,
        default="none",
        choices=["none", "static_window", "dynamic_topk", "sparsity", "sparse_vggt"],
        help=(
            "Frame-attention sparsity mode. "
            "none: dense frame attention; static_window: local patch window + global special tokens; "
            "dynamic_topk: static_window plus top-k patch queries that can see all patch keys; "
            "sparsity: grid K/V subsampling + diagonal preservation + mean-fill; "
            "sparse_vggt: Sparse-VGGT-style pooled attention selects sparse patch blocks."
        ),
    )
    parser.add_argument(
        "--frame_sparse_window",
        type=int,
        default=7,
        help="Odd local window size for frame_sparse_mode static_window/dynamic_topk.",
    )
    parser.add_argument(
        "--frame_sparse_start_layer",
        type=int,
        default=0,
        help="Start layer index (inclusive) for frame-attention sparsity.",
    )
    parser.add_argument(
        "--frame_sparse_apply_every",
        type=int,
        default=1,
        help="Apply frame-attention sparsity every N frame-attention layers.",
    )
    parser.add_argument(
        "--frame_sparse_topk_ratio",
        type=float,
        default=0.1,
        help="For dynamic_topk only: ratio of patch queries promoted to global visibility.",
    )
    parser.add_argument(
        "--frame_sparse_stride_h",
        type=int,
        default=2,
        help="For sparsity mode: grid subsampling stride along patch height.",
    )
    parser.add_argument(
        "--frame_sparse_stride_w",
        type=int,
        default=2,
        help="For sparsity mode: grid subsampling stride along patch width.",
    )
    parser.add_argument(
        "--frame_sparse_preserve_diagonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For sparsity mode: preserve diagonal self-attention terms.",
    )
    parser.add_argument(
        "--frame_sparse_use_mean_fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For sparsity mode: add one mean K/V token for dropped columns.",
    )
    parser.add_argument(
        "--frame_sparse_debug_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For sparsity mode: print per-layer sparse K/V stats.",
    )
    parser.add_argument(
        "--frame_sparse_debug_print_every",
        type=int,
        default=1,
        help="For sparsity mode with debug stats: print every N layers.",
    )
    parser.add_argument(
        "--frame_sparse_svggt_sparse_ratio",
        type=float,
        default=0.75,
        help=(
            "For frame_sparse_mode sparse_vggt: sparse ratio in [0,1], "
            "keep ratio is approximately (1 - sparse_ratio)."
        ),
    )
    parser.add_argument(
        "--frame_sparse_svggt_cdf_threshold",
        type=float,
        default=None,
        help=(
            "For frame_sparse_mode sparse_vggt: optional CDF threshold in [0,1] for block selection. "
            "Can be combined with sparse_ratio."
        ),
    )
    parser.add_argument(
        "--frame_sparse_svggt_topk_blocks",
        type=int,
        default=None,
        help=(
            "For frame_sparse_mode sparse_vggt: optional fixed top-k key blocks per query block. "
            "When provided, combined with sparse_ratio/cdf by taking the larger keep count."
        ),
    )
    parser.add_argument(
        "--frame_sparse_svggt_pool_mode",
        type=str,
        default="avg",
        choices=["avg", "max"],
        help="For frame_sparse_mode sparse_vggt: pooling mode for proxy attention.",
    )
    parser.add_argument(
        "--frame_sparse_svggt_ks_q",
        type=int,
        default=128,
        help="For frame_sparse_mode sparse_vggt: query pooling kernel size.",
    )
    parser.add_argument(
        "--frame_sparse_svggt_ks_k",
        type=int,
        default=64,
        help="For frame_sparse_mode sparse_vggt: key pooling kernel size.",
    )
    parser.add_argument(
        "--frame_sparse_svggt_use_sparge_kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For frame_sparse_mode sparse_vggt: use SpargeAttn CUDA kernel when available. "
            "Falls back to dense-mask implementation if unavailable."
        ),
    )
    parser.add_argument(
        "--use_flex_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch FlexAttention for masked frame-attention path.",
    )
    parser.add_argument(
        "--flex_block_size",
        type=int,
        default=128,
        help="Block size used for FlexAttention block mask construction.",
    )
    parser.add_argument(
        "--flex_compile_mode",
        type=str,
        default="fullgraph",
        choices=["none", "default", "reduce-overhead", "fullgraph"],
        help="Compile mode for FlexAttention callable.",
    )
    return parser


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_chamfer_distance(points_pred, points_gt, max_dist=1.0):
    max_eval_points = 100000
    if points_pred.shape[0] > max_eval_points:
        idx = np.random.choice(points_pred.shape[0], max_eval_points, replace=False)
        points_pred = points_pred[idx]
    if points_gt.shape[0] > max_eval_points:
        idx = np.random.choice(points_gt.shape[0], max_eval_points, replace=False)
        points_gt = points_gt[idx]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
    pcd_pred = pcd_pred.voxel_down_sample(0.05)
    pcd_gt = pcd_gt.voxel_down_sample(0.05)

    d1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    d2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    d1 = np.clip(d1, 0, max_dist)
    d2 = np.clip(d2, 0, max_dist)
    return float(np.mean(d1) + np.mean(d2))


def main(args):
    _set_eval_seed(args.seed)
    print(f"[launch] seed={args.seed}, cudnn_deterministic=True", flush=True)
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, Long3D
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
        # resolution = (518, 336)
    else:
        raise NotImplementedError
    scenes_root = args.scenes_root
    if scenes_root is None:
        # Default: 7scenes folder in project root (alongside src/)
        # __file__ is in src/eval/mv_recon/, go up to project root
        project_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
        base_7scenes = osp.join(project_root, "7scenes")
        # Resolve symlink and try nested 7scenes/7scenes (e.g. data/7scenes)
        base_7scenes = osp.realpath(base_7scenes) if osp.exists(base_7scenes) else base_7scenes
        nested = osp.join(base_7scenes, "7scenes")
        scenes_root = nested if osp.isdir(nested) else base_7scenes
    project_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    nrgbd_root = args.nrgbd_root
    if nrgbd_root is None:
        nrgbd_root = osp.join(project_root, "datasets", "nrgbd")
    long3d_root = args.long3d_root
    if long3d_root is None:
        long3d_root = osp.join(project_root, "datasets", "Long3D")

    datasets_all = {}
    if args.dataset in ("7scenes", "all"):
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT=scenes_root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=2,
            max_frames=args.max_frames,
            test_id=args.test_id,
            seq_id=args.seq_id,
        )
    if args.dataset in ("nrgbd", "all"):
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT=nrgbd_root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=2,
            max_frames=args.max_frames,
            test_id=args.test_id,
        )
    if args.dataset in ("long3d", "all"):
        datasets_all["Long3D"] = Long3D(
            split="test",
            ROOT=long3d_root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=1,
            max_frames=args.max_frames,
            test_id=args.test_id,
            extract_missing=args.long3d_extract_missing,
        )

    # Resolve budget_proportions_path when --use_cosine_budget or --budget_strategy
    budget_proportions_path = args.budget_proportions_path
    project_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    if args.budget_strategy is not None:
        budget_proportions_path = osp.join(project_root, "configs", f"kv_budget_proportions_{args.budget_strategy}.json")
        if not osp.isfile(budget_proportions_path):
            raise FileNotFoundError(
                f"Budget config not found: {budget_proportions_path}. "
                f"Run: python eval/mv_recon/compute_kv_budget_strategies.py --strategy {args.budget_strategy} --weights ... --scenes_root ..."
            )
        print(f"[launch] budget_strategy={args.budget_strategy}, path={budget_proportions_path}", flush=True)
    elif args.use_cosine_budget:
        budget_proportions_path = osp.join(project_root, "configs", "kv_budget_proportions.json")
        if not osp.isfile(budget_proportions_path):
            budget_proportions_path = osp.join(project_root, "configs", "kv_budget_proportions_cosine.json")
        if not osp.isfile(budget_proportions_path):
            raise FileNotFoundError(
                f"Cosine budget config not found: {budget_proportions_path}. "
                "Run compute_kv_budget_strategies.py --strategy cosine first."
            )
        print(f"[launch] use_cosine_budget=True, path={budget_proportions_path}", flush=True)

    # Resolve importance_weights for importance-based eviction modes
    importance_weights = None
    core_weight_keys = (
        "w_camera",
        "w_geometry",
        "w_temporal",
        "w_saliency",
        "w_depth_conf",
        "w_pts_conf",
    )
    if args.eviction_mode in ("importance",):
        project_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
        presets_path = osp.join(project_root, "configs", "importance_weights_presets.json")
        if args.importance_preset is not None:
            if not osp.isfile(presets_path):
                raise FileNotFoundError(f"Importance presets not found: {presets_path}")
            with open(presets_path) as f:
                presets = json.load(f)
            idx = args.importance_preset
            if idx < 0 or idx >= len(presets):
                raise ValueError(f"importance_preset must be 0..{len(presets)-1}, got {idx}")
            importance_weights = presets[idx]
            print(f"[launch] importance_preset={idx}, weights={importance_weights}", flush=True)
        elif args.importance_weights_path:
            with open(args.importance_weights_path) as f:
                importance_weights = json.load(f)
            print(f"[launch] importance_weights from {args.importance_weights_path}", flush=True)
        else:
            default_importance_json = osp.join(
                project_root,
                "configs",
                "importance_weights_default.json",
            )
            if not osp.isfile(default_importance_json):
                raise FileNotFoundError(
                    f"Default importance weights not found: {default_importance_json}. "
                    "Pass --importance_weights_path explicitly."
                )
            with open(default_importance_json) as f:
                importance_weights = json.load(f)
            print(
                f"[launch] importance_weights: default from {default_importance_json}",
                flush=True,
            )
        if importance_weights is not None:
            print("[launch] active importance weights:", flush=True)
            for key in core_weight_keys:
                if key in importance_weights:
                    print(f"{key}: {float(importance_weights[key]):.4f}", flush=True)

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    target_ratio = None
    if args.kv_target_budget is not None:
        if args.total_budget <= 0:
            raise ValueError("--total_budget must be > 0 when using --kv_target_budget")
        target_ratio = float(args.kv_target_budget) / float(args.total_budget)
        if target_ratio <= 0:
            raise ValueError("--kv_target_budget must be > 0")
        target_ratio = min(1.0, target_ratio)

    print(
        f"[launch] eviction_mode={args.eviction_mode}, max_frames={args.max_frames}, "
        f"total_budget={args.total_budget}, kv_share_method={args.kv_share_method}, "
        f"kv_target_budget={args.kv_target_budget}",
        flush=True,
    )
    kv_share_cfg = {
        "method": args.kv_share_method,
        "group_size": args.kv_share_group_size,
        "share_heads_ratio": args.kv_share_heads_ratio,
        "share_token_ratio": args.kv_share_token_ratio,
        "coarse_start_layer": args.kv_coarse_start_layer,
        "coarse_stride": args.kv_coarse_stride,
        "coarse_near_frames": args.kv_coarse_near_frames,
        "delta_start_layer": args.kv_delta_start_layer,
        "delta_keep_ratio": args.kv_delta_keep_ratio,
        "target_ratio": target_ratio,
    }
    if model_name == "StreamVGGT":
        # from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = StreamVGGT(total_budget=args.total_budget)
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to(device)
    elif model_name == "VGGT":
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = VGGT()
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to(device)

    else:
        raise NotImplementedError
    del ckpt
    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_repeat < 1:
        raise ValueError("--eval_repeat must be >= 1")

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for run_idx in range(args.eval_repeat):
            run_root = (
                args.output_dir
                if args.eval_repeat <= 1
                else osp.join(args.output_dir, f"run_{run_idx}")
            )
            if args.eval_repeat > 1:
                os.makedirs(run_root, exist_ok=True)
                print(
                    f"[launch] eval_repeat {run_idx + 1}/{args.eval_repeat}, run_root={run_root}",
                    flush=True,
                )
            for name_data, dataset in datasets_all.items():
                save_path = osp.join(run_root, name_data)
                os.makedirs(save_path, exist_ok=True)
                log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")
    
                acc_all = 0
                acc_all_med = 0
                comp_all = 0
                comp_all_med = 0
                nc1_all = 0
                nc1_all_med = 0
                nc2_all = 0
                nc2_all_med = 0
    
                fps_all = []
                time_all = []
    
                with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                    for data_idx in tqdm(idxs):
                        # Reset peak memory stats at the start of each sequence
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        batch = default_collate([dataset[data_idx]])
                        ignore_keys = set(
                            [
                                "depthmap",
                                "dataset",
                                "label",
                                "instance",
                                "idx",
                                "true_shape",
                                "rng",
                            ]
                        )
                        if name_data == "Long3D":
                            # Long3D evaluation does not use per-pixel GT tensors in model forward.
                            # Keep them on CPU to avoid OOM on long sequences.
                            ignore_keys.update(
                                {
                                    "pts3d",
                                    "valid_mask",
                                    "ray_map",
                                    "camera_intrinsics",
                                    "camera_pose",
                                }
                            )
                        long3d_chunk_gpu = (
                            name_data == "Long3D"
                            and args.long3d_chunk_size is not None
                            and int(args.long3d_chunk_size) > 0
                            and len(batch) > int(args.long3d_chunk_size)
                        )
                        ignore_device = set(ignore_keys)
                        if long3d_chunk_gpu:
                            ignore_device.add("img")
                        for view in batch:
                            for name in view.keys():  # pseudo_focal
                                if name in ignore_device:
                                    continue
                                if isinstance(view[name], tuple) or isinstance(
                                    view[name], list
                                ):
                                    view[name] = [
                                        x.to(device, non_blocking=True) for x in view[name]
                                    ]
                                else:
                                    view[name] = view[name].to(device, non_blocking=True)
    
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []
                        in_camera1 = None  
    
                        if model_name == "stream3r" or "VGGT":
                            revisit = args.revisit
                            update = not args.freeze
                            if revisit > 1:
                                # repeat input for 'revisit' times
                                new_views = []
                                for r in range(revisit):
                                    for i in range(len(batch)):
                                        new_view = deepcopy(batch[i])
                                        new_view["idx"] = [
                                            (r * len(batch) + i)
                                            for _ in range(len(batch[i]["idx"]))
                                        ]
                                        new_view["instance"] = [
                                            str(r * len(batch) + i)
                                            for _ in range(len(batch[i]["instance"]))
                                        ]
                                        if r > 0:
                                            if not update:
                                                new_view["update"] = torch.zeros_like(
                                                    batch[i]["update"]
                                                ).bool()
                                        new_views.append(new_view)
                                batch = new_views
                            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                            with torch.cuda.amp.autocast(dtype=dtype):
                                if isinstance(batch, dict) and "img" in batch:
                                    batch["img"] = (batch["img"] + 1.0) / 2.0
                                elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                                    for view in batch:
                                        view["img"] = (view["img"] + 1.0) / 2.0
    
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            infer_t0 = time.perf_counter()
                            with torch.cuda.amp.autocast(dtype=dtype):
                                with torch.no_grad():
                                    frame_sparse_cfg = {
                                        "mode": args.frame_sparse_mode,
                                        "window_size": args.frame_sparse_window,
                                        "start_layer": args.frame_sparse_start_layer,
                                        "apply_every": args.frame_sparse_apply_every,
                                        "topk_ratio": args.frame_sparse_topk_ratio,
                                        "stride_h": args.frame_sparse_stride_h,
                                        "stride_w": args.frame_sparse_stride_w,
                                        "preserve_diagonal": args.frame_sparse_preserve_diagonal,
                                        "use_mean_fill": args.frame_sparse_use_mean_fill,
                                        "debug_sparse_stats": args.frame_sparse_debug_stats,
                                        "debug_print_every": args.frame_sparse_debug_print_every,
                                        "svggt_sparse_ratio": args.frame_sparse_svggt_sparse_ratio,
                                        "svggt_cdf_threshold": args.frame_sparse_svggt_cdf_threshold,
                                        "svggt_topk_blocks": args.frame_sparse_svggt_topk_blocks,
                                        "svggt_pool_mode": args.frame_sparse_svggt_pool_mode,
                                        "svggt_ks_q": args.frame_sparse_svggt_ks_q,
                                        "svggt_ks_k": args.frame_sparse_svggt_ks_k,
                                        "svggt_use_sparge_kernel": args.frame_sparse_svggt_use_sparge_kernel,
                                    }
                                    if name_data == "Long3D" and args.long3d_chunk_size is not None and args.long3d_chunk_size > 0 and len(batch) > args.long3d_chunk_size:
                                        chunk_size = int(args.long3d_chunk_size)
                                        all_preds = []
                                        all_views = []
                                        peak_kv_cache_bytes = 0
                                        seq_state = None
                                        for s in range(0, len(batch), chunk_size):
                                            e = min(len(batch), s + chunk_size)
                                            chunk_views_gpu = []
                                            for view in batch[s:e]:
                                                v = dict(view)
                                                for name in v.keys():
                                                    if name in ignore_keys:
                                                        continue
                                                    if isinstance(v[name], tuple) or isinstance(
                                                        v[name], list
                                                    ):
                                                        v[name] = [
                                                            x.to(device, non_blocking=True)
                                                            for x in v[name]
                                                        ]
                                                    elif isinstance(v[name], torch.Tensor):
                                                        v[name] = v[name].to(
                                                            device, non_blocking=True
                                                        )
                                                chunk_views_gpu.append(v)
                                            chunk_results = model.inference(
                                                chunk_views_gpu,
                                                eviction_mode=args.eviction_mode,
                                                budget_proportions_path=budget_proportions_path,
                                                importance_weights=importance_weights,
                                                use_importance_in_attn=args.use_importance_in_attn,
                                                softmax_importance_before_k=args.softmax_importance_before_k,
                                                debug_importance_in_attn=args.debug_importance_in_attn,
                                                profile_importance_raw=args.profile_importance_raw,
                                                profile_min_frames=args.profile_min_frames
                                                if args.profile_importance_raw
                                                else None,
                                                kv_share_cfg=kv_share_cfg,
                                                frame_sparse_cfg=frame_sparse_cfg,
                                                use_flex_attention=args.use_flex_attention,
                                                flex_block_size=args.flex_block_size,
                                                flex_compile_mode=args.flex_compile_mode,
                                                sequence_state=seq_state,
                                                return_sequence_state=True,
                                            )
                                            seq_state = chunk_results.sequence_state
                                            if chunk_results.ress is not None:
                                                all_preds.extend(chunk_results.ress)
                                            if chunk_results.views is not None:
                                                all_views.extend(chunk_results.views)
                                            peak_kv_cache_bytes = max(
                                                peak_kv_cache_bytes,
                                                getattr(chunk_results, "kv_cache_mem_bytes", None) or 0,
                                            )
                                            del chunk_views_gpu
                                            torch.cuda.empty_cache()
                                        class _ChunkResults:
                                            pass
                                        results = _ChunkResults()
                                        results.ress = all_preds
                                        results.views = all_views
                                        results.kv_cache_mem_bytes = peak_kv_cache_bytes
                                    else:
                                        results = model.inference(
                                            batch,
                                            eviction_mode=args.eviction_mode,
                                            budget_proportions_path=budget_proportions_path,
                                            importance_weights=importance_weights,
                                            use_importance_in_attn=args.use_importance_in_attn,
                                            softmax_importance_before_k=args.softmax_importance_before_k,
                                            debug_importance_in_attn=args.debug_importance_in_attn,
                                            profile_importance_raw=args.profile_importance_raw,
                                            profile_min_frames=args.profile_min_frames
                                            if args.profile_importance_raw
                                            else None,
                                            kv_share_cfg=kv_share_cfg,
                                            frame_sparse_cfg=frame_sparse_cfg,
                                            use_flex_attention=args.use_flex_attention,
                                            flex_block_size=args.flex_block_size,
                                            flex_compile_mode=args.flex_compile_mode,
                                        )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            infer_sec = time.perf_counter() - infer_t0
    
                            preds, batch = results.ress, results.views
                            kv_cache_mem_bytes = getattr(results, "kv_cache_mem_bytes", None) or 0 

                            if args.use_proj:
                                pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
                                depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
                                depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
                                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                                                    batch[0]["img"].shape[-2:])

                                if "DTU" in name_data:
                                    depth_map = depth_map * 1000.0
                                    extrinsic[..., :3, 3] *= 1000.0

                                point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                                                extrinsic.squeeze(0),
                                                                                                intrinsic.squeeze(0))
                            valid_length = len(preds) // args.revisit
                            if args.revisit > 1:
                                preds = preds[-valid_length:]
                                batch = batch[-valid_length:]

                            if args.speed_only:
                                scene_id = batch[-1]["label"][0].rsplit("/", 1)[0] if len(batch) > 0 else f"{name_data}_{data_idx}"
                                num_frames = max(len(preds), 1)
                                ms_per_frame = infer_sec * 1000.0 / num_frames
                                fps = num_frames / max(infer_sec, 1e-9)
                                kv_cache_mem_gb = kv_cache_mem_bytes / (1024**3)
                                speed_line = (
                                    f"TimingOnly: Idx={scene_id}, frames={num_frames}, "
                                    f"infer_sec={infer_sec:.6f}, ms_per_frame={ms_per_frame:.3f}, fps={fps:.3f}, "
                                    f"KVCache={kv_cache_mem_gb:.3f} GB"
                                )
                                print(speed_line, flush=True)
                                with open(log_file, "a") as f:
                                    f.write(speed_line + "\n")
                                torch.cuda.empty_cache()
                                continue
                        if name_data == "Long3D":
                            if o3d is None:
                                raise ImportError("open3d is required for Long3D evaluation.")

                            scene_id = (
                                batch[-1]["label"][0].split("/", 1)[0]
                                if len(batch) > 0
                                else f"Long3D_{data_idx}"
                            )
                            gt_pcd_path = osp.join(long3d_root, scene_id, "dense_cloud_map.pcd")
                            if not osp.isfile(gt_pcd_path):
                                skip_line = f"[Long3D] missing GT pcd: {gt_pcd_path}, skip {scene_id}"
                                print(skip_line, flush=True)
                                with open(log_file, "a") as f:
                                    f.write(skip_line + "\n")
                                torch.cuda.empty_cache()
                                continue

                            pred_chunks = []
                            for pred in preds:
                                pts_pred = pred["pts3d_in_other_view"].cpu().numpy()[0]
                                conf_pred = pred["conf"].cpu().numpy()[0]
                                valid_pred = np.isfinite(pts_pred).all(axis=-1) & np.isfinite(conf_pred)
                                if args.conf_thresh > 0:
                                    valid_pred &= conf_pred > args.conf_thresh
                                if np.any(valid_pred):
                                    pred_chunks.append(pts_pred[valid_pred].astype(np.float32))

                            if len(pred_chunks) == 0:
                                skip_line = f"[Long3D] no valid predicted points for {scene_id}, skip"
                                print(skip_line, flush=True)
                                with open(log_file, "a") as f:
                                    f.write(skip_line + "\n")
                                torch.cuda.empty_cache()
                                continue

                            pred_pts = np.concatenate(pred_chunks, axis=0)
                            if pred_pts.shape[0] > args.long3d_max_points:
                                sel = np.random.choice(pred_pts.shape[0], args.long3d_max_points, replace=False)
                                pred_pts = pred_pts[sel]

                            pcd_gt = o3d.io.read_point_cloud(gt_pcd_path)
                            gt_pts = np.asarray(pcd_gt.points, dtype=np.float32)
                            if gt_pts.shape[0] == 0:
                                skip_line = f"[Long3D] empty GT pcd for {scene_id}, skip"
                                print(skip_line, flush=True)
                                with open(log_file, "a") as f:
                                    f.write(skip_line + "\n")
                                torch.cuda.empty_cache()
                                continue
                            if gt_pts.shape[0] > args.long3d_max_points:
                                sel = np.random.choice(gt_pts.shape[0], args.long3d_max_points, replace=False)
                                gt_pts = gt_pts[sel]

                            pcd_pred = o3d.geometry.PointCloud()
                            pcd_pred.points = o3d.utility.Vector3dVector(pred_pts)
                            pcd_gt_eval = o3d.geometry.PointCloud()
                            pcd_gt_eval.points = o3d.utility.Vector3dVector(gt_pts)

                            # Align prediction to GT with ICP before metric computation.
                            trans_init = np.eye(4)
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                pcd_pred,
                                pcd_gt_eval,
                                0.1,
                                trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            )
                            pcd_pred = pcd_pred.transform(reg_p2p.transformation)

                            pcd_pred.estimate_normals()
                            pred_normals = np.asarray(pcd_pred.normals)
                            pcd_gt_eval.estimate_normals()
                            gt_normals = np.asarray(pcd_gt_eval.normals)

                            acc, acc_med, nc1, nc1_med = accuracy(
                                pcd_gt_eval.points, pcd_pred.points, gt_normals, pred_normals
                            )
                            comp, comp_med, nc2, nc2_med = completion(
                                pcd_gt_eval.points, pcd_pred.points, gt_normals, pred_normals
                            )
                            pred_pts_aligned = np.asarray(pcd_pred.points, dtype=np.float32)
                            cd_value = _compute_chamfer_distance(
                                pred_pts_aligned, gt_pts, max_dist=args.long3d_chamfer_max_dist
                            )

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                            else:
                                max_mem_gb = 0.0
                            kv_cache_mem_gb = kv_cache_mem_bytes / (1024**3)
                            log_line = (
                                f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                                f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                            )
                            cd_line = f"CD: Idx: {scene_id}, CD: {cd_value}"
                            mem_line = f"MaxMem: {max_mem_gb:.3f} GB | KVCache: {kv_cache_mem_gb:.3f} GB"
                            print(log_line, flush=True)
                            print(cd_line, flush=True)
                            print(mem_line, flush=True)
                            with open(log_file, "a") as f:
                                f.write(log_line + "\n")
                                f.write(cd_line + "\n")
                                f.write(mem_line + "\n")

                            acc_all += acc
                            comp_all += comp
                            nc1_all += nc1
                            nc2_all += nc2
                            acc_all_med += acc_med
                            comp_all_med += comp_med
                            nc1_all_med += nc1_med
                            nc2_all_med += nc2_med

                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        if o3d is None:
                            raise ImportError("open3d is required for full metric evaluation (non --speed_only mode).")

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            if args.use_proj:
                                pts = point_map_by_unprojection[j]
                                conf = depth_conf[0, j].cpu().data.numpy()
                            else:
                                pts = pred_pts[j].cpu().numpy()[0]
                                conf = preds[j]["conf"].cpu().data.numpy()[0]

                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]

                            # Align predicted 3D points to the ground truth
                            # pts = geotrf(in_camera1, pts)
                            # pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append(image[None, ...])
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])
    
                        images_all = np.concatenate(images_all, axis=0)
                        pts_all = np.concatenate(pts_all, axis=0)
                        pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                        masks_all = np.concatenate(masks_all, axis=0)
    
                        scene_id = view["label"][0].rsplit("/", 1)[0]
    
                        save_params = {}
    
                        save_params["images_all"] = images_all
                        save_params["pts_all"] = pts_all
                        save_params["pts_gt_all"] = pts_gt_all
                        save_params["masks_all"] = masks_all
    
                        np.save(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                            save_params,
                        )
    
                        if "DTU" in name_data:
                            threshold = 100
                        else:
                            threshold = 0.1
    
                        pts_all_masked = pts_all[masks_all > 0]
                        pts_gt_all_masked = pts_gt_all[masks_all > 0]
                        images_all_masked = images_all[masks_all > 0]
    
                        mask = np.isfinite(pts_all_masked)  
                        pts_all_masked = pts_all_masked[mask]
    
                        mask_gt = np.isfinite(pts_gt_all_masked)
                        pts_gt_all_masked = pts_gt_all_masked[mask]
    
                        if args.use_proj:
                            def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
                                assert src.shape == dst.shape
                                N, dim = src.shape
    
                                mu_src = src.mean(axis=0)
                                mu_dst = dst.mean(axis=0)
                                src_c = src - mu_src
                                dst_c = dst - mu_dst
    
                                Sigma = dst_c.T @ src_c / N  # (3,3)
    
                                U, D, Vt = np.linalg.svd(Sigma) 
    
                                S = np.eye(dim)
                                if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                                    S[-1, -1] = -1
    
                                R = U @ S @ Vt
    
                                if with_scale:
                                    var_src = (src_c ** 2).sum() / N
                                    s = (D * S.diagonal()).sum() / var_src
                                else:
                                    s = 1.0
    
                                t = mu_dst - s * R @ mu_src
    
                                return s, R, t
    
                            pts_all_masked = pts_all_masked.reshape(-1, 3)
                            pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                            s, R, t = umeyama_alignment(pts_all_masked, pts_gt_all_masked, with_scale=True)
                            pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                            pts_all_masked = pts_all_aligned
    
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            pts_all_masked.reshape(-1, 3)
                        )
                        pcd.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                            ),
                            pcd,
                        )
    
                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(
                            pts_gt_all_masked.reshape(-1, 3)
                        )
                        pcd_gt.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        o3d.io.write_point_cloud(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                            pcd_gt,
                        )
    
                        trans_init = np.eye(4)
    
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd,
                            pcd_gt,
                            threshold,
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )
    
                        transformation = reg_p2p.transformation
    
                        pcd = pcd.transform(transformation)
    
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-mask_align.ply"
                            ),
                            pcd,
                        )
    
                        pcd.estimate_normals()
                        pcd_gt.estimate_normals()
    
                        gt_normal = np.asarray(pcd_gt.normals)
                        pred_normal = np.asarray(pcd.normals)
    
                        acc, acc_med, nc1, nc1_med = accuracy(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        comp, comp_med, nc2, nc2_med = completion(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
    
                        # Get peak GPU memory (GB) for this sequence
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                        else:
                            max_mem_gb = 0.0
    
                        kv_cache_mem_gb = kv_cache_mem_bytes / (1024**3)
                        log_line = (
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                            f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                        )
                        mem_line = f"MaxMem: {max_mem_gb:.3f} GB | KVCache: {kv_cache_mem_gb:.3f} GB"
                        print(log_line)
                        print(mem_line)
                        with open(log_file, "a") as f:
                            f.write(log_line + "\n")
                            f.write(mem_line + "\n")
    
                        acc_all += acc
                        comp_all += comp
                        nc1_all += nc1
                        nc2_all += nc2
    
                        acc_all_med += acc_med
                        comp_all_med += comp_med
                        nc1_all_med += nc1_med
                        nc2_all_med += nc2_med
    
                        # release cuda memory
                        torch.cuda.empty_cache()
    
                accelerator.wait_for_everyone()
                # Get depth from pcd and run TSDFusion
                if accelerator.is_main_process:
                    to_write = ""
                    # Copy the error log from each process to the main error log
                    for i in range(8):
                        if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                            break
                        with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                            to_write += f_sub.read()
    
                    with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                        log_data = to_write
                        metrics = defaultdict(list)
                        for line in log_data.strip().split("\n"):
                            match = regex.match(line)
                            if match:
                                data = match.groupdict()
                                # Exclude 'scene_id' from metrics as it's an identifier
                                for key, value in data.items():
                                    if key != "scene_id":
                                        metrics[key].append(float(value))
                                metrics["nc"].append(
                                    (float(data["nc1"]) + float(data["nc2"])) / 2
                                )
                                metrics["nc_med"].append(
                                    (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                                )
                            match_cd = regex_cd.match(line)
                            if match_cd:
                                data_cd = match_cd.groupdict()
                                metrics["cd"].append(float(data_cd["cd"]))
                        mean_metrics = {
                            metric: sum(values) / len(values)
                            for metric, values in metrics.items()
                        }
    
                        c_name = "mean"
                        print_str = f"{c_name.ljust(20)}: "
                        for m_name in mean_metrics:
                            print_num = np.mean(mean_metrics[m_name])
                            print_str = print_str + f"{m_name}: {print_num:.3f} | "
                        print_str = print_str + "\n"
                        f.write(to_write + print_str)



from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

pattern_cd = r"""
    CD:\s*Idx:\s*(?P<scene_id>[^,]+),\s*
    CD:\s*(?P<cd>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)
regex_cd = re.compile(pattern_cd, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)