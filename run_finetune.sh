#!/usr/bin/env bash
# Fine-tune one or more Q dimensions sequentially.
#
# Usage:
#   bash run_finetune.sh --gpu 6 --dim q8              # single dim
#   bash run_finetune.sh --gpu 4 --dim q1 q2 q3        # multiple dims
#   bash run_finetune.sh --gpu 4                        # all dims (q1-q8)
#   bash run_finetune.sh --gpu 3 --dim all
# train all 
#   bash run_finetune.sh --gpu 4 --dim q1 --epochs 5   # override epochs
#   bash run_finetune.sh --gpu 2,3 --dim q6 q7 q8      # multi-GPU

set -euo pipefail

# ── Defaults ──
GPU=""
DIMS=()
EPOCHS=3
BATCH=1
GRAD_ACCUM=4
LR=1e-4
EVAL_STEPS=200
SAVE_STEPS=2000
BASE_MODEL="/workspace/tingting/models/Qwen2.5-VL-3B-Instruct"
DATA_ROOT="data"
FT_ROOT="data/finetune-0515-fix"
OUT_ROOT="models/0515/qwen2.5-vl-3b-mv-lora"
WANDB_PROJECT="3dbench-finetune-all-dim"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2"; shift 2 ;;
        --dim)        shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do DIMS+=("$1"); shift; done ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --eval_steps) EVAL_STEPS="$2"; shift 2 ;;
        --save_steps) SAVE_STEPS="$2"; shift 2 ;;
        --base_model) BASE_MODEL="$2"; shift 2 ;;
        --ft_root)    FT_ROOT="$2"; shift 2 ;;
        --out_root)   OUT_ROOT="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 --gpu <id> [--dim q1 q2 ...] [--epochs N] [--batch N] [--lr F]"
            exit 1 ;;
    esac
done

# Default to all dims if none specified
if [[ ${#DIMS[@]} -eq 0 ]]; then
    DIMS=(q1 q2 q3 q4 q5 q6 q7 q8)
fi

if [[ -z "$GPU" ]]; then
    echo "ERROR: --gpu is required (e.g. --gpu 0 or --gpu 2,3)"
    exit 1
fi

# ── W&B env (source for API key, then restore our project name) ──
_SAVE_PROJECT="$WANDB_PROJECT"
if [[ -f /workspace/tingting/.wandb/env.sh ]]; then
    source /workspace/tingting/.wandb/env.sh
fi
export WANDB_PROJECT="$_SAVE_PROJECT"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "================================================="
echo "Fine-tune Qwen2.5-VL-3B LoRA"
echo "  dims       : ${DIMS[*]}"
echo "  gpu        : $GPU"
echo "  epochs=$EPOCHS  batch=$BATCH  grad_accum=$GRAD_ACCUM  lr=$LR"
echo "  eval_steps=$EVAL_STEPS  save_steps=$SAVE_STEPS"
echo "  ft_root    : $FT_ROOT"
echo "  out_root   : $OUT_ROOT"
echo "================================================="

for dim in "${DIMS[@]}"; do
    train_jsonl="$FT_ROOT/$dim/train.jsonl"
    val_jsonl="$FT_ROOT/$dim/val.jsonl"
    out_dir="$OUT_ROOT/$dim"

    if [[ ! -f "$train_jsonl" ]]; then
        echo "ERROR: $train_jsonl not found"
        exit 1
    fi

    # all-in-one prompt is much longer → need larger max_seq_len
    if [[ "$dim" == "all" ]]; then
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-40960}"
    else
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
    fi

    echo
    echo ">>> [$dim] training → $out_dir  (max_seq_len=$MAX_SEQ_LEN)"
    python scripts/06_finetune_qwen.py \
        --dim "$dim" \
        --train_jsonl "$train_jsonl" \
        --val_jsonl   "$val_jsonl" \
        --data_root   "$DATA_ROOT" \
        --base_model  "$BASE_MODEL" \
        --output_dir  "$out_dir" \
        --epochs "$EPOCHS" \
        --per_device_batch_size "$BATCH" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --gpu "$GPU" \
        --wandb --wandb_project "$WANDB_PROJECT"
    echo ">>> [$dim] done."
done

echo
echo "All done. Adapters: $OUT_ROOT/<dim>/final/"
