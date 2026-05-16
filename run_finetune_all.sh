#!/usr/bin/env bash
# Run per-dimension LoRA fine-tuning of Qwen2.5-VL-3B for all 8 dimensions,
# sequentially. Adapters are written to
# models/qwen2.5-vl-3b-mv-lora/<dim>/final/.
#
# Prereqs:
#   - data/finetune-0515/<dim>/{train,val,test}.jsonl produced by
#     scripts/05_prepare_finetune_data.py
#
# Usage:
#   bash run_finetune_all.sh                          # all 8 dims, all GPUs
#   bash run_finetune_all.sh --gpu 4                  # single GPU
#   bash run_finetune_all.sh --gpu 2,3                # multi GPU
#   DIMS="q3 q6" bash run_finetune_all.sh --gpu 1    # subset of dims
#   EPOCHS=2 bash run_finetune_all.sh --gpu 0         # override epochs
#   bash run_finetune_all.sh --gpu 0 --wandb          # with W&B logging

set -euo pipefail

# ── Parse script-level args ──
GPU=""
WANDB_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="${2:-}"
            shift 2
            ;;
        --wandb)
            WANDB_FLAG="--wandb"
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [--gpu <id>] [--wandb]"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

DIMS="${DIMS:-q1 q2 q3 q4 q5 q6 q7 q8}"
DATA_ROOT="${DATA_ROOT:-data}"
FT_ROOT="${FT_ROOT:-data/finetune-0515}"
OUT_ROOT="${OUT_ROOT:-models/qwen2.5-vl-3b-mv-lora}"
BASE_MODEL="${BASE_MODEL:-/umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEV_BATCH="${PER_DEV_BATCH:-1}"
EVAL_STEPS="${EVAL_STEPS:-0}"
SAVE_STEPS="${SAVE_STEPS:-0}"

# Source W&B env if available and --wandb requested
if [[ -n "$WANDB_FLAG" && -f "/workspace/tingting/.wandb/env.sh" ]]; then
    source /workspace/tingting/.wandb/env.sh
fi

mkdir -p "$OUT_ROOT"

echo "================================================="
echo "Per-dim LoRA fine-tune"
echo "  base model : $BASE_MODEL"
echo "  ft data    : $FT_ROOT"
echo "  out root   : $OUT_ROOT"
echo "  dims       : $DIMS"
echo "  gpu        : ${GPU:-all}"
echo "  epochs=$EPOCHS  lr=$LR  rank=$RANK  alpha=$ALPHA"
echo "  grad_accum=$GRAD_ACCUM  per_dev_batch=$PER_DEV_BATCH"
echo "  eval_steps=$EVAL_STEPS  save_steps=$SAVE_STEPS"
echo "  wandb      : ${WANDB_FLAG:-off}"
echo "================================================="

GPU_ARG=""
if [[ -n "$GPU" ]]; then
    GPU_ARG="--gpu $GPU"
fi

for dim in $DIMS; do
    train_jsonl="$FT_ROOT/$dim/train.jsonl"
    val_jsonl="$FT_ROOT/$dim/val.jsonl"
    out_dir="$OUT_ROOT/$dim"

    if [[ ! -f "$train_jsonl" ]]; then
        echo "ERROR: $train_jsonl not found — run scripts/05_prepare_finetune_data.py first"
        exit 1
    fi

    echo
    echo ">>> [$dim] training → $out_dir"
    $PY scripts/06_finetune_qwen.py \
        --dim "$dim" \
        --train_jsonl "$train_jsonl" \
        --val_jsonl   "$val_jsonl" \
        --data_root   "$DATA_ROOT" \
        --base_model  "$BASE_MODEL" \
        --output_dir  "$out_dir" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --lora_rank "$RANK" \
        --lora_alpha "$ALPHA" \
        --grad_accum "$GRAD_ACCUM" \
        --per_device_batch_size "$PER_DEV_BATCH" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        $GPU_ARG \
        $WANDB_FLAG
    echo ">>> [$dim] done."
done

echo
echo "All LoRA adapters saved under $OUT_ROOT/<dim>/final/"
