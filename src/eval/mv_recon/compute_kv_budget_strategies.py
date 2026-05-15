#!/usr/bin/env python3
"""
离线计算 KV Cache 每层 token 预算分配 - 支持多种分层策略

策略:
1. cosine (默认): 基于输入输出余弦相似度，相似度越高越冗余
2. fisher: Fisher 信息矩阵，聚合层内参数 Fisher 值
3. gradient: 梯度范数，层参数梯度越大越重要
4. hessian: 海森对角近似，反映参数敏感性
5. pruning: 剪枝/层冗余，移除层后性能下降幅度

用法:
    cd src
    python eval/mv_recon/compute_kv_budget_strategies.py \\
        --strategy fisher \\
        --weights ../ckpt/checkpoints.pth \\
        --scenes_root /path/to/7scenes \\
        --output_path ../configs/kv_budget_proportions_fisher.json
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def get_args_parser():
    p = argparse.ArgumentParser("Compute KV budget from layer importance strategies", add_help=False)
    p.add_argument("--strategy", type=str, default="cosine",
                   choices=["cosine", "fisher", "gradient", "hessian", "pruning"],
                   help="Stratification strategy: cosine, fisher, gradient, hessian, pruning")
    p.add_argument("--weights", type=str, default="", required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--scenes_root", type=str, default=None)
    p.add_argument("--output_path", type=str, default=None,
                   help="Output JSON path. Default: configs/kv_budget_proportions_{strategy}.json")
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--total_budget", type=int, default=1200000)
    p.add_argument("--sample_sequences", type=str, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature for converting importance to proportions")
    p.add_argument("--num_frames_for_grad", type=int, default=2,
                   help="Number of frames for Fisher/Gradient/Hessian (forward without cache). Reduced from 5 to avoid OOM.")
    p.add_argument("--pruning_num_frames", type=int, default=15,
                   help="Frames per sequence for pruning strategy. Reduced from 60 to avoid OOM.")
    p.add_argument("--use_gradient_checkpoint", action="store_true",
                   help="Use gradient checkpointing for Fisher/Gradient/Hessian to reduce memory")
    return p


def _broadcast_mask_for_indexing(mask, tensor):
    """Broadcast mask (H,W) or (B,H,W) to match tensor (B,H,W,C) for boolean indexing.
    Returns flat mask for use with tensor.reshape(-1, 3)[mask] -> (N, 3)."""
    m = mask
    # Add leading dim if 2D
    while m.dim() < tensor.dim() - 1:
        m = m.unsqueeze(0)
    # Flatten spatial dims: (B,H,W) -> (B*H*W,), tensor (B,H,W,C) -> (B*H*W,C)
    m = m.reshape(-1)
    return m


def load_7scenes_sequence(scenes_root, scene_seq, max_frames, resolution=(518, 392)):
    """加载单条 7Scenes 序列"""
    import os.path as osp
    from eval.mv_recon.data import SevenScenes

    candidates = [
        (osp.join(scenes_root, scene_seq), scenes_root),
        (osp.join(scenes_root, "7scenes", scene_seq), osp.join(scenes_root, "7scenes")),
    ]
    data_path = None
    resolved_root = scenes_root
    for p, root in candidates:
        if osp.isdir(p):
            data_path = p
            resolved_root = root
            break
    if data_path is None:
        raise FileNotFoundError(f"7Scenes path not found: {scene_seq}")

    num_files = len([n for n in os.listdir(data_path) if "color" in n])
    if num_files == 0:
        raise FileNotFoundError(f"No color images in {scene_seq}")
    img_idxs = [f"{i:06d}" for i in range(min(num_files, max_frames))]
    tuple_list = [f"{scene_seq} " + " ".join(img_idxs)]

    dataset = SevenScenes(
        split="test",
        ROOT=resolved_root,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=1,
        max_frames=max_frames,
        tuple_list=tuple_list,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No data for {scene_seq}")

    batch = dataset[0]
    frames = []
    for v in batch:
        frame = {
            "img": v["img"],
            "valid_mask": v["valid_mask"],
            "camera_pose": v["camera_pose"],
            "depthmap": v["depthmap"],
            "pts3d": v["pts3d"],
            "label": v.get("label", scene_seq),
        }
        frames.append(frame)
    return frames


def get_all_7scenes_sequences(scenes_root, sample_sequences=None):
    """获取所有 7scenes 测试序列（从 TestSplit.txt 自动读取）"""
    import os.path as osp
    if sample_sequences is not None:
        return sample_sequences
    base = osp.join(scenes_root, "7scenes") if osp.isdir(osp.join(scenes_root, "7scenes")) else scenes_root
    seqs = []
    for scene in sorted(os.listdir(base)):
        split_file = osp.join(base, scene, "TestSplit.txt")
        if not osp.isfile(split_file):
            continue
        with open(split_file) as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line:
                    continue
                num_part = "".join(filter(str.isdigit, line))
                seq_id = f"seq-{num_part.zfill(2)}"
                seqs.append(f"{scene}/{seq_id}")
    return seqs


# =============================================================================
# Strategy: Cosine
# =============================================================================

def compute_cosine_similarity(x_in: torch.Tensor, x_out: torch.Tensor) -> float:
    x_in_flat = x_in.reshape(-1, x_in.shape[-1]).float()
    x_out_flat = x_out.reshape(-1, x_out.shape[-1]).float()
    cos_sim = F.cosine_similarity(x_in_flat.unsqueeze(0), x_out_flat.unsqueeze(0), dim=1).mean()
    return cos_sim.item()


# =============================================================================
# Strategy: Fisher / Gradient / Hessian
# =============================================================================

def compute_loss_from_views(model, views, criterion, device):
    """Compute 3D regression loss from views (batch)."""
    if criterion is None:
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.cuda.amp.autocast(enabled=False):
        out = model.forward(views, use_cache=False)
    preds = []
    for s in range(len(views)):
        res = {
            "pts3d_in_other_view": out.ress[s]["pts3d_in_other_view"],
            "conf": out.ress[s]["conf"],
            "camera_pose": out.ress[s]["camera_pose"],
        }
        preds.append(res)

    gt_pts, pred_pts, gt_factor, pr_factor, masks, _ = criterion.get_all_pts3d_t(views, preds)
    total_loss = 0.0
    count = 0
    for i in range(len(gt_pts)):
        m = masks[i]
        if m.sum() > 0:
            m = _broadcast_mask_for_indexing(m, pred_pts[i])
            l = criterion.criterion(pred_pts[i].reshape(-1, 3)[m], gt_pts[i].reshape(-1, 3)[m])
            total_loss = total_loss + l.mean()
            count += 1
    if count > 0:
        total_loss = total_loss / count
    return total_loss


def run_fisher_gradient_hessian(
    model, frames, device, num_layers, strategy, num_frames, criterion,
    use_gradient_checkpoint=False,
):
    """
    Fisher: E[(dL/dθ)²]
    Gradient: ||dL/dθ|| per layer
    Hessian: diagonal approx = grad² (similar to Fisher)
    """
    layer_fisher = np.zeros(num_layers)
    layer_count = np.zeros(num_layers)

    for seq_idx, scene_seq in enumerate(tqdm(frames.keys(), desc="Sequences")):
        frame_list = frames[scene_seq]
        n_use = min(num_frames, len(frame_list))
        if n_use < 2:
            continue

        views = []
        for i in range(n_use):
            f = frame_list[i]
            img = f["img"]
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.array(img)).float()
            img = img.to(device)
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.min() >= -1.1 and img.max() <= 1.1:
                img = (img + 1.0) / 2.0
            view = {
                "img": img,
                "valid_mask": f["valid_mask"].to(device) if isinstance(f["valid_mask"], torch.Tensor) else torch.from_numpy(f["valid_mask"]).to(device),
                "camera_pose": f["camera_pose"].to(device) if isinstance(f["camera_pose"], torch.Tensor) else torch.from_numpy(f["camera_pose"]).float().unsqueeze(0).to(device),
                "pts3d": f["pts3d"].to(device) if isinstance(f["pts3d"], torch.Tensor) else torch.from_numpy(f["pts3d"]).float().unsqueeze(0).to(device),
            }
            if view["camera_pose"].dim() == 2:
                view["camera_pose"] = view["camera_pose"].unsqueeze(0)
            if view["pts3d"].dim() == 3:
                view["pts3d"] = view["pts3d"].unsqueeze(0)
            views.append(view)

        model.train()
        if use_gradient_checkpoint:
            model.aggregator.gradient_checkpointing = True
        model.zero_grad()
        loss = compute_loss_from_views(model, views, criterion, device)
        if not torch.isfinite(loss) or loss.item() == 0:
            continue
        loss.backward()

        for layer_idx in range(num_layers):
            block = model.aggregator.global_blocks[layer_idx]
            grad_sq_sum = 0.0
            grad_norm_sum = 0.0
            for p in block.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    grad_sq_sum += (g ** 2).sum().item()
                    grad_norm_sum += g.norm().item()
            if strategy == "fisher" or strategy == "hessian":
                layer_fisher[layer_idx] += grad_sq_sum
            else:
                layer_fisher[layer_idx] += grad_norm_sum
            layer_count[layer_idx] += 1

        if use_gradient_checkpoint:
            model.aggregator.gradient_checkpointing = False
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with np.errstate(divide="ignore", invalid="ignore"):
        importance = np.where(layer_count > 0, layer_fisher / np.maximum(layer_count, 1), 1e-6)
    importance = np.clip(importance, 1e-6, None)
    return importance, {}


# =============================================================================
# Strategy: Pruning
# =============================================================================

def run_pruning_strategy(model, frames, device, num_layers, num_frames, criterion):
    """
    剪枝/层冗余: 移除层后性能下降越多，层越重要
    """
    baseline_losses = []
    layer_skip_losses = defaultdict(list)

    for scene_seq, frame_list in tqdm(frames.items(), desc="Pruning sequences"):
        n_use = min(num_frames, len(frame_list))
        if n_use < 2:
            continue

        views = []
        for i in range(n_use):
            f = frame_list[i]
            img = f["img"]
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.array(img)).float()
            img = img.to(device)
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.min() >= -1.1 and img.max() <= 1.1:
                img = (img + 1.0) / 2.0
            view = {
                "img": img,
                "valid_mask": f["valid_mask"].to(device) if isinstance(f["valid_mask"], torch.Tensor) else torch.from_numpy(f["valid_mask"]).to(device),
                "camera_pose": f["camera_pose"].to(device) if isinstance(f["camera_pose"], torch.Tensor) else torch.from_numpy(f["camera_pose"]).float().unsqueeze(0).to(device),
                "pts3d": f["pts3d"].to(device) if isinstance(f["pts3d"], torch.Tensor) else torch.from_numpy(f["pts3d"]).float().unsqueeze(0).to(device),
            }
            if view["camera_pose"].dim() == 2:
                view["camera_pose"] = view["camera_pose"].unsqueeze(0)
            if view["pts3d"].dim() == 3:
                view["pts3d"] = view["pts3d"].unsqueeze(0)
            views.append(view)

        with torch.no_grad():
            out_baseline = model.forward(views, use_cache=False)
        preds_baseline = [{"pts3d_in_other_view": out_baseline.ress[s]["pts3d_in_other_view"],
                          "conf": out_baseline.ress[s]["conf"],
                          "camera_pose": out_baseline.ress[s]["camera_pose"]} for s in range(len(views))]
        gt_pts, pred_pts, _, _, masks, _ = criterion.get_all_pts3d_t(views, preds_baseline)
        loss_baseline = 0.0
        cnt = 0
        for i in range(len(gt_pts)):
            if masks[i].sum() > 0:
                m = _broadcast_mask_for_indexing(masks[i], pred_pts[i])
                loss_baseline += criterion.criterion(
                    pred_pts[i].reshape(-1, 3)[m], gt_pts[i].reshape(-1, 3)[m]
                ).mean().item()
                cnt += 1
        if cnt > 0:
            loss_baseline /= cnt
            baseline_losses.append(loss_baseline)

        for layer_idx in range(num_layers):
            block = model.aggregator.global_blocks[layer_idx]
            original_forward = block.forward

            def _identity_forward(self, x, *args, **kwargs):
                return x
            block.forward = _identity_forward.__get__(block, type(block))
            try:
                with torch.no_grad():
                    out_skip = model.forward(views, use_cache=False)
                preds_skip = [{"pts3d_in_other_view": out_skip.ress[s]["pts3d_in_other_view"],
                              "conf": out_skip.ress[s]["conf"],
                              "camera_pose": out_skip.ress[s]["camera_pose"]} for s in range(len(views))]
                gt_pts, pred_pts, _, _, masks, _ = criterion.get_all_pts3d_t(views, preds_skip)
                loss_skip = 0.0
                cnt = 0
                for i in range(len(gt_pts)):
                    if masks[i].sum() > 0:
                        m = _broadcast_mask_for_indexing(masks[i], pred_pts[i])
                        loss_skip += criterion.criterion(
                            pred_pts[i].reshape(-1, 3)[m], gt_pts[i].reshape(-1, 3)[m]
                        ).mean().item()
                        cnt += 1
                if cnt > 0:
                    loss_skip /= cnt
                    layer_skip_losses[layer_idx].append(loss_skip)
            finally:
                block.forward = original_forward

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_mean = np.mean(baseline_losses) if baseline_losses else 1.0
    importance = np.zeros(num_layers)
    for layer_idx in range(num_layers):
        losses = layer_skip_losses[layer_idx]
        if losses:
            skip_mean = np.mean(losses)
            importance[layer_idx] = max(0, skip_mean - baseline_mean)
        else:
            importance[layer_idx] = 1e-6
    importance = np.clip(importance, 1e-6, None)
    return importance, {"baseline_loss": baseline_mean}


# =============================================================================
# Main
# =============================================================================

def importance_to_budgets(importance, total_budget, temperature):
    """Convert importance scores to per-layer budgets."""
    importance = np.clip(importance.astype(np.float32), 1e-6, None)
    scaled = importance / temperature
    proportions = torch.softmax(torch.from_numpy(scaled), dim=0).numpy()
    budgets = (proportions * total_budget).astype(np.int32)
    diff = total_budget - budgets.sum()
    if diff != 0:
        budgets[np.argmax(proportions)] += diff
    return proportions, budgets


def main():
    args = get_args_parser().parse_args()

    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.weights)

    from streamvggt.models.streamvggt import StreamVGGT
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = StreamVGGT(total_budget=args.total_budget)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to(device)

    num_layers = model.aggregator.depth
    num_heads = model.aggregator.global_blocks[0].attn.num_heads

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.scenes_root is None:
        base_7scenes = os.path.join(project_root, "7scenes")
        nested = os.path.join(base_7scenes, "7scenes")
        scenes_root = nested if os.path.isdir(nested) else base_7scenes
    else:
        scenes_root = args.scenes_root

    sequences = get_all_7scenes_sequences(scenes_root, args.sample_sequences)
    print(f"Strategy: {args.strategy}, {len(sequences)} sequences, max_frames={args.max_frames}")

    if args.output_path is None:
        args.output_path = os.path.join(project_root, "configs", f"kv_budget_proportions_{args.strategy}.json")

    if args.strategy == "cosine":
        from eval.mv_recon.compute_kv_budget_from_cosine_sim import run_inference_collect_cosine_sim
        num_heads = model.aggregator.global_blocks[0].attn.num_heads
        head_dim = model.aggregator.global_blocks[0].attn.head_dim
        cos_sim_sum = np.zeros(num_layers)
        cos_sim_count = np.zeros(num_layers)
        extra = {}
        for scene_seq in tqdm(sequences, desc="Cosine"):
            try:
                frames = load_7scenes_sequence(scenes_root, scene_seq, args.max_frames)
            except FileNotFoundError:
                continue
            s_sum, s_count = run_inference_collect_cosine_sim(
                model, frames, device, num_layers, num_heads, head_dim
            )
            cos_sim_sum += s_sum
            cos_sim_count += s_count
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_sim_mean = np.where(cos_sim_count > 0, cos_sim_sum / cos_sim_count, 0.5)
        importance = 1.0 - cos_sim_mean
        importance = np.clip(importance, 1e-6, 1.0)
        extra = {"cos_sim_per_layer": cos_sim_mean.tolist()}
    else:
        frames_dict = {}
        for scene_seq in tqdm(sequences, desc="Loading"):
            try:
                frames = load_7scenes_sequence(scenes_root, scene_seq, args.max_frames)
                frames_dict[scene_seq] = frames
            except FileNotFoundError:
                continue

        criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

        if args.strategy in ("fisher", "gradient", "hessian"):
            importance, extra = run_fisher_gradient_hessian(
                model, frames_dict, device, num_layers, args.strategy,
                args.num_frames_for_grad, criterion,
                use_gradient_checkpoint=args.use_gradient_checkpoint,
            )
        else:
            importance, extra = run_pruning_strategy(
                model, frames_dict, device, num_layers,
                args.pruning_num_frames, criterion
            )

    proportions, budgets = importance_to_budgets(importance, args.total_budget, args.temperature)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "strategy": args.strategy,
        "total_budget": args.total_budget,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "importance_per_layer": importance.tolist(),
        "proportions": proportions.tolist(),
        "budgets_per_layer": budgets.tolist(),
        "temperature": args.temperature,
    }
    for k, v in extra.items():
        if k not in result and isinstance(v, (list, float, int, str)):
            result[k] = v
        elif k == "cos_sim_per_layer":
            result[k] = v

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {args.output_path}")
    print("Layer | Importance | Proportion | Budget")
    print("-" * 45)
    for i in range(num_layers):
        print(f"  {i:2d}  | {importance[i]:.6f}   | {proportions[i]:.4f}     | {budgets[i]:6d}")
    print("-" * 45)
    print(f"Sum budgets: {budgets.sum()}")


if __name__ == "__main__":
    main()
