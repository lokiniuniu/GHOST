#!/bin/bash
# 离线计算 KV Cache 每层预算分配（基于输入输出余弦相似度）
#
# 用法:
#   bash scripts/run_compute_kv_budget.sh [ckpt_path] [7scenes_root] [output_json]
#
# 示例:
#   bash scripts/run_compute_kv_budget.sh ckpt/checkpoints.pth 7scenes configs/kv_budget_proportions.json

CKPT="${1:-ckpt/checkpoints.pth}"
SCENES="${2:-7scenes}"
OUTPUT="${3:-configs/kv_budget_proportions.json}"

cd "$(dirname "$0")/.."
cd src

python eval/mv_recon/compute_kv_budget_from_cosine_sim.py \
    --weights "../${CKPT}" \
    --scenes_root "../${SCENES}" \
    --output_path "../${OUTPUT}" \
    --max_frames 60 \
    --total_budget 1200000
