#!/usr/bin/env bash
# Run VLM evaluation — all config written here, just call:
#   bash run_eval.sh --gpu 4
#
# To switch eval target, edit the "CONFIG" section below.

set -euo pipefail

# =====================================================================
# CONFIG — edit these to control what to eval
# =====================================================================

# Data
MANIFEST="data/gt-demo-libero-all-suite-train-fix/manifest.json"

# Filtering (leave empty to use all)
SUITES=""                        # e.g. "libero_spatial libero_object"
TASK_IDS="9"                     # e.g. "0 1 2" — val=8, test=9
DEMO_INDICES=""                  # e.g. "0 1"
MAX_SAMPLES=""                   # e.g. "100"

# ── Pick ONE mode: baseline OR lora OR lora_multi ──

# Mode A: Zero-shot baseline (uncomment to use)
# EVAL_MODE="baseline"
# MODELS="qwen2.5-vl-3b"

# Mode B: Single LoRA checkpoint (uncomment to use)
# EVAL_MODE="lora"
# LORA="models/0515/qwen2.5-vl-3b-mv-lora/q3/checkpoint-4000"

# Mode C: All per-dim LoRA at a given checkpoint (default)
EVAL_MODE="lora_multi"
LORA_ROOT="models/0515/qwen2.5-vl-3b-mv-lora"
CHECKPOINT="checkpoint-4000"     # or "final"
DIMS="q1 q2 q3 q4 q5 q6"       # which dims to eval

# =====================================================================
# END CONFIG — usually no need to edit below
# =====================================================================

GPU=""
OUT_DIR=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --resume)  RESUME="--resume"; shift ;;
        *)         echo "Usage: $0 --gpu <id> [--out_dir PATH] [--resume]"; exit 1 ;;
    esac
done

if [[ -z "$GPU" ]]; then
    echo "ERROR: --gpu is required"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Build filter args
FILTER_ARGS=""
[[ -n "$SUITES" ]]       && FILTER_ARGS="$FILTER_ARGS --suites $SUITES"
[[ -n "$TASK_IDS" ]]     && FILTER_ARGS="$FILTER_ARGS --task_ids $TASK_IDS"
[[ -n "$DEMO_INDICES" ]] && FILTER_ARGS="$FILTER_ARGS --demo_indices $DEMO_INDICES"
[[ -n "$MAX_SAMPLES" ]]  && FILTER_ARGS="$FILTER_ARGS --max_samples $MAX_SAMPLES"

# ── Baseline ──
if [[ "$EVAL_MODE" == "baseline" ]]; then
    model_tag=$(echo $MODELS | tr ' ' '+')
    [[ -z "$OUT_DIR" ]] && OUT_DIR="rollout/${model_tag}_${TIMESTAMP}"

    echo "================================================="
    echo "Zero-shot eval"
    echo "  gpu      : $GPU"
    echo "  models   : $MODELS"
    echo "  manifest : $MANIFEST"
    echo "  filters  :$FILTER_ARGS"
    echo "  out_dir  : $OUT_DIR"
    echo "================================================="

    CUDA_VISIBLE_DEVICES="$GPU" python scripts/02_run_vlm_eval.py \
        --manifest "$MANIFEST" \
        --out_dir "$OUT_DIR" \
        --models $MODELS \
        --device cuda \
        $FILTER_ARGS $RESUME

    echo "Done. Results: $OUT_DIR/"
fi

# ── Single LoRA ──
if [[ "$EVAL_MODE" == "lora" ]]; then
    _base="$(basename "$LORA")"
    _parent="$(basename "$(dirname "$LORA")")"
    if [[ "$_base" == "final" || "$_base" == checkpoint-* ]]; then
        _dim="$_parent"; _ckpt="$_base"
    else
        _dim="$_base"; _ckpt="final"
    fi

    [[ -z "$OUT_DIR" ]] && OUT_DIR="rollout/lora-${_dim}_${_ckpt}_${TIMESTAMP}"

    echo "================================================="
    echo "LoRA eval (single)"
    echo "  gpu      : $GPU"
    echo "  lora     : $LORA"
    echo "  dim      : $_dim"
    echo "  ckpt     : $_ckpt"
    echo "  manifest : $MANIFEST"
    echo "  filters  :$FILTER_ARGS"
    echo "  out_dir  : $OUT_DIR"
    echo "================================================="

    CUDA_VISIBLE_DEVICES="$GPU" python scripts/02_run_vlm_eval.py \
        --manifest "$MANIFEST" \
        --out_dir "$OUT_DIR" \
        --models qwen2.5-vl-3b-mv-lora \
        --lora_dir "$LORA" \
        --lora_dim "$_dim" \
        --device cuda \
        $FILTER_ARGS $RESUME

    echo "Done. Results: $OUT_DIR/"
fi

# ── Multi-dim LoRA ──
if [[ "$EVAL_MODE" == "lora_multi" ]]; then
    dims_tag=$(echo $DIMS | tr ' ' '+')
    [[ -z "$OUT_DIR" ]] && OUT_DIR="rollout/lora-${dims_tag}_${CHECKPOINT}_${TIMESTAMP}"

    echo "================================================="
    echo "LoRA eval (multi-dim)"
    echo "  gpu        : $GPU"
    echo "  lora_root  : $LORA_ROOT"
    echo "  checkpoint : $CHECKPOINT"
    echo "  dims       : $DIMS"
    echo "  manifest   : $MANIFEST"
    echo "  filters    :$FILTER_ARGS"
    echo "  out_dir    : $OUT_DIR"
    echo "================================================="

    for dim in $DIMS; do
        lora_dir="$LORA_ROOT/$dim/$CHECKPOINT"
        if [[ ! -d "$lora_dir" ]]; then
            echo "WARNING: $lora_dir not found, skipping $dim"
            continue
        fi

        echo
        echo ">>> [$dim] LoRA eval: $lora_dir"
        CUDA_VISIBLE_DEVICES="$GPU" python scripts/02_run_vlm_eval.py \
            --manifest "$MANIFEST" \
            --out_dir "$OUT_DIR" \
            --models qwen2.5-vl-3b-mv-lora \
            --lora_dir "$lora_dir" \
            --lora_dim "$dim" \
            --device cuda \
            $FILTER_ARGS $RESUME
        echo ">>> [$dim] done."
    done

    echo
    echo "All done. Results: $OUT_DIR/"
fi
