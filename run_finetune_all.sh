#!/usr/bin/env bash
# Run per-dimension LoRA fine-tuning of Qwen2.5-VL-3B for all 6 dimensions,
# sequentially, on a single GPU. Adapters are written to
# models/qwen2.5-vl-3b-mv-lora/<dim>/final/.
#
# Prereqs:
#   - data/gt-q6-mv/<suite>/manifest.json exists for all 4 suites
#   - data/finetune-mv/<dim>/{train,val,test}.jsonl produced by
#     scripts/05_prepare_finetune_data.py
#
# Usage:
#   bash run_finetune_all.sh                     # all 6 dims, GPU 0
#   CUDA_VISIBLE_DEVICES=1 bash run_finetune_all.sh   # pick a different GPU
#   DIMS="q3 q6"  bash run_finetune_all.sh       # subset of dims
#   EPOCHS=2  bash run_finetune_all.sh           # override epochs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the local conda env if not already active. Falls back to the
# venv-style absolute python path so this script works under both `bash` and
# the user's daily shell.
PY="${PY:-/umd-datapool/tingting/envs/vlmbench/bin/python}"

DIMS="${DIMS:-q1 q2 q3 q4 q5 q6}"
DATA_ROOT="${DATA_ROOT:-data}"
FT_ROOT="${FT_ROOT:-data/finetune-mv}"
OUT_ROOT="${OUT_ROOT:-models/qwen2.5-vl-3b-mv-lora}"
BASE_MODEL="${BASE_MODEL:-/umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEV_BATCH="${PER_DEV_BATCH:-1}"

mkdir -p "$OUT_ROOT"

echo "================================================="
echo "Per-dim LoRA fine-tune"
echo "  base model : $BASE_MODEL"
echo "  ft data    : $FT_ROOT"
echo "  out root   : $OUT_ROOT"
echo "  dims       : $DIMS"
echo "  epochs=$EPOCHS  lr=$LR  rank=$RANK  alpha=$ALPHA"
echo "  grad_accum=$GRAD_ACCUM  per_dev_batch=$PER_DEV_BATCH"
echo "================================================="

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
    "$PY" scripts/06_finetune_qwen.py \
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
        --per_device_batch_size "$PER_DEV_BATCH"
    echo ">>> [$dim] done."
done

echo
echo "All 6 LoRA adapters saved under $OUT_ROOT/<dim>/final/"
