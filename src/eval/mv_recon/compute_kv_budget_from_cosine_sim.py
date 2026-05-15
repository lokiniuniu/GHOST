#!/usr/bin/env python3
"""
离线计算 KV Cache 每层每头的 token 预算分配

基于 improved_importance 模式：利用每层每个头输入与输出之间的余弦相似度
来代表该层的重要性。相似度越高说明该层越冗余，应分配更少的 token 预算。

用法:
    cd src
    python eval/mv_recon/compute_kv_budget_from_cosine_sim.py \\
        --weights ../ckpt/checkpoints.pth \\
        --scenes_root /path/to/7scenes \\
        --output_path ../configs/kv_budget_proportions.json \\
        --max_frames 60 \\
        --total_budget 1200000
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
    p = argparse.ArgumentParser("Compute KV budget from layer/head cosine similarity", add_help=False)
    p.add_argument("--weights", type=str, default="", required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--scenes_root", type=str, default=None)
    p.add_argument("--output_path", type=str, default="configs/kv_budget_proportions.json")
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--total_budget", type=int, default=1200000)
    p.add_argument("--sample_sequences", type=str, nargs="+", default=None,
                   help="If set, only use these sequences; else use all 7scenes test sequences")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature for converting (1-cos_sim) to proportions")
    p.add_argument("--per_head", action="store_true",
                   help="Compute per-head similarity and average to layer (requires attention mod)")
    return p


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
    """获取所有 7scenes 测试序列"""
    import os.path as osp
    base = osp.join(scenes_root, "7scenes") if osp.isdir(osp.join(scenes_root, "7scenes")) else scenes_root
    if sample_sequences is not None:
        return sample_sequences
    # 默认使用 kv_cache_pruning_analysis 中的 scene_list
    return [
        "stairs/seq-06", "stairs/seq-02", "pumpkin/seq-06", "chess/seq-01",
        "heads/seq-02", "fire/seq-02", "office/seq-03", "pumpkin/seq-03",
        "redkitchen/seq-07", "chess/seq-02", "office/seq-01", "redkitchen/seq-01",
        "fire/seq-01",
    ]


def compute_cosine_similarity(x_in: torch.Tensor, x_out: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    计算 x_in 与 x_out 的余弦相似度。
    x_in, x_out: [B, N, C]
    返回: scalar (mean over B*N)
    """
    x_in_flat = x_in.reshape(-1, x_in.shape[-1]).float()
    x_out_flat = x_out.reshape(-1, x_out.shape[-1]).float()
    cos_sim = F.cosine_similarity(x_in_flat.unsqueeze(0), x_out_flat.unsqueeze(0), dim=1).mean()
    return cos_sim.item()


def run_inference_collect_cosine_sim(
    model, frames, device, num_layers, num_heads, head_dim,
):
    """
    逐帧推理，收集每层每头的输入-输出余弦相似度。
    相似度越高 = 越冗余 = 越不重要。
    """
    # 注册 hook 捕获每层 block 的输入和输出
    layer_inputs = {}
    layer_outputs = {}
    handles = []

    def make_hook(layer_idx):
        def pre_hook(module, input):
            layer_inputs[layer_idx] = input[0].detach().clone()

        def post_hook(module, input, output):
            # output: (tokens, new_kv, scores) or (tokens, new_kv, scores, attn_w)
            layer_outputs[layer_idx] = output[0].detach().clone()

        return pre_hook, post_hook

    for layer_idx in range(num_layers):
        block = model.aggregator.global_blocks[layer_idx]
        pre_h, post_h = make_hook(layer_idx)
        handles.append(block.register_forward_pre_hook(pre_h))
        handles.append(block.register_forward_hook(post_h))

    # 重置 cache 状态
    past_key_values = [None] * num_layers
    for block in model.aggregator.global_blocks:
        if hasattr(block.attn, "_reset_cache_state"):
            block.attn._reset_cache_state()

    # 累积每层的 cos_sim 和 count
    cos_sim_sum = np.zeros(num_layers)
    cos_sim_count = np.zeros(num_layers)

    past_key_values_camera = [None] * model.camera_head.trunk_depth
    importance_cache = {}
    all_ress = []

    with torch.no_grad():
        for i, frame in enumerate(tqdm(frames, desc="Frames", leave=False)):
            images = frame["img"]
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(np.array(images)).float()
            images = images.to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)  # B, C, H, W
            images = images.unsqueeze(0)  # B, S, C, H, W
            if images.min() >= -1.1 and images.max() <= 1.1:
                images = (images + 1.0) / 2.0

            frame_metadata = [
                {"camera_pose": r["camera_pose"], "depth": r["depth"],
                 "depth_conf": r["depth_conf"], "conf": r.get("conf")}
                for r in all_ress
            ] if all_ress else []

            # 调用 aggregator（与 inference 中一致）
            agg_out = model.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
                total_budget=model.total_budget,
                qvg_manager=None,
                kv_mode="anchor1_only",
                eviction_mode="importance",
                frame_metadata=frame_metadata,
                importance_cache=importance_cache if importance_cache is not None else {},
            )
            aggregated_tokens, patch_start_idx, past_key_values = agg_out[:3]

            # 计算每层 cos_sim
            for layer_idx in range(num_layers):
                if layer_idx in layer_inputs and layer_idx in layer_outputs:
                    x_in = layer_inputs[layer_idx]
                    x_out = layer_outputs[layer_idx]
                    cos_sim = compute_cosine_similarity(x_in, x_out)
                    cos_sim_sum[layer_idx] += cos_sim
                    cos_sim_count[layer_idx] += 1

            # 清理
            layer_inputs.clear()
            layer_outputs.clear()

            # 运行 heads 获取 predictions，用于下一帧的 frame_metadata
            with torch.cuda.amp.autocast(enabled=False):
                pose_enc, past_key_values_camera = model.camera_head(
                    aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True
                )
                depth, depth_conf = model.depth_head(
                    aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                )
                pts3d, pts3d_conf = model.point_head(
                    aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                )
            res = {
                "camera_pose": pose_enc[-1][:, 0, :],
                "depth": depth[:, 0],
                "depth_conf": depth_conf[:, 0],
                "conf": pts3d_conf[:, 0],
            }
            all_ress.append(res)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    return cos_sim_sum, cos_sim_count


def main():
    args = get_args_parser().parse_args()

    # add_ckpt_path for dust3r
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.weights)

    from streamvggt.models.streamvggt import StreamVGGT

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = StreamVGGT(total_budget=args.total_budget)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to(device)

    num_layers = model.aggregator.depth
    num_heads = model.aggregator.global_blocks[0].attn.num_heads
    head_dim = model.aggregator.global_blocks[0].attn.head_dim

    # 解析 scenes_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.scenes_root is None:
        base_7scenes = os.path.join(project_root, "7scenes")
        nested = os.path.join(base_7scenes, "7scenes")
        scenes_root = nested if os.path.isdir(nested) else base_7scenes
    else:
        scenes_root = args.scenes_root

    sequences = get_all_7scenes_sequences(scenes_root, args.sample_sequences)
    print(f"Computing cosine similarity on {len(sequences)} sequences, max_frames={args.max_frames}")

    cos_sim_sum = np.zeros(num_layers)
    cos_sim_count = np.zeros(num_layers)

    for scene_seq in tqdm(sequences, desc="Sequences"):
        try:
            frames = load_7scenes_sequence(scenes_root, scene_seq, args.max_frames)
        except FileNotFoundError as e:
            print(f"  Skip {scene_seq}: {e}")
            continue

        s_sum, s_count = run_inference_collect_cosine_sim(
            model, frames, device, num_layers, num_heads, head_dim
        )
        cos_sim_sum += s_sum
        cos_sim_count += s_count

    # 平均
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_sim_mean = np.where(cos_sim_count > 0, cos_sim_sum / cos_sim_count, 0.5)

    # 重要性 = 1 - cos_sim（相似度越高越冗余，重要性越低）
    importance = 1.0 - cos_sim_mean
    importance = np.clip(importance, 1e-6, 1.0)

    # 转为预算比例（与 _calculate_dynamic_budgets 一致：diversity_scores = 1 - scores）
    scaled = importance / args.temperature
    proportions = torch.softmax(torch.from_numpy(scaled.astype(np.float32)), dim=0).numpy()

    # 计算每层预算
    budgets = (proportions * args.total_budget).astype(np.int32)
    # 确保总和等于 total_budget
    diff = args.total_budget - budgets.sum()
    if diff != 0:
        budgets[np.argmax(proportions)] += diff

    # 保存
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "total_budget": args.total_budget,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "cos_sim_per_layer": cos_sim_mean.tolist(),
        "importance_per_layer": importance.tolist(),
        "proportions": proportions.tolist(),
        "budgets_per_layer": budgets.tolist(),
        "temperature": args.temperature,
    }
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {args.output_path}")
    print("Layer | CosSim | Importance | Proportion | Budget")
    print("-" * 55)
    for i in range(num_layers):
        print(f"  {i:2d}  | {cos_sim_mean[i]:.4f} | {importance[i]:.4f}     | {proportions[i]:.4f}     | {budgets[i]:6d}")
    print("-" * 55)
    print(f"Sum budgets: {budgets.sum()}")


if __name__ == "__main__":
    main()
