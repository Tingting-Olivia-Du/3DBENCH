#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# VLA Ablation Evaluation Launcher
#
# Usage:
#   # Evaluate a single experiment on all LIBERO suites
#   bash scripts/run_vla_ablation_eval.sh e0_full
#
#   # Evaluate on a specific suite only
#   bash scripts/run_vla_ablation_eval.sh e0_full libero_spatial
#
#   # Evaluate all completed experiments
#   bash scripts/run_vla_ablation_eval.sh all
#
#   # Specify GPU
#   CUDA_DEVICE=0 bash scripts/run_vla_ablation_eval.sh e0_full
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"               # 3DBENCH/
VLM4VLA_ROOT="$(dirname "$PROJECT_ROOT")/VLM4VLA"     # VLM4VLA/
CONFIG_DIR="$PROJECT_ROOT/configs/vla_ablation"
RESULTS_DIR="$PROJECT_ROOT/results/vla_ablation"

# Use the vlmbench env interpreter (has lightning/flash-attn/etc.), not system python.
PYTHON="${PYTHON:-$(dirname "$PROJECT_ROOT")/envs/vlmbench/bin/python}"

# ── Eval config ────────────────────────────────────────────────────────
CUDA_DEVICE=${CUDA_DEVICE:-0}
EXECUTE_STEP=${EXECUTE_STEP:-1}
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

mkdir -p "$RESULTS_DIR"

find_checkpoint() {
    # Find the best/latest checkpoint for an experiment.
    # Search the whole experiment dir so it works whether ckpts live under
    # <exp>/checkpoints/.../ (deep) or directly under <exp>/ (flat).
    local exp_name="$1"
    local ckpt_root="$VLM4VLA_ROOT/runs/vla_ablation/${exp_name}"

    if [ ! -d "$ckpt_root" ]; then
        echo ""
        return
    fi

    # Latest by step number in 'step=N.ckpt'; fall back to lexical sort.
    local latest
    latest=$(find "$ckpt_root" -name "*.ckpt" -type f 2>/dev/null \
             | sed -E 's/.*step=([0-9]+)\.ckpt/\1\t&/' \
             | sort -n | tail -1 | cut -f2-)
    echo "$latest"
}

run_eval() {
    local exp_name="$1"
    local suite="$2"
    local config_path="$CONFIG_DIR/${exp_name}.json"

    local ckpt_path
    ckpt_path=$(find_checkpoint "$exp_name")

    if [ -z "$ckpt_path" ]; then
        echo "[SKIP] $exp_name: no checkpoint found"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  Evaluating: $exp_name on $suite"
    echo "  Checkpoint: $ckpt_path"
    echo "  GPU:        $CUDA_DEVICE"
    echo "================================================================"

    local cmd="cd $VLM4VLA_ROOT && PYTHONPATH=$VLM4VLA_ROOT CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $PYTHON eval/libero/run_libero_eval.py \
        --ckpt_path $ckpt_path \
        --config_path $config_path \
        --execute_step $EXECUTE_STEP \
        --task_suite_name $suite \
        --center_crop True"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd" 2>&1 | tee "$RESULTS_DIR/${exp_name}_${suite}.log"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────
EXP_TARGET="${1:-}"
SUITE_TARGET="${2:-}"

if [ -z "$EXP_TARGET" ]; then
    echo "Usage: bash scripts/run_vla_ablation_eval.sh <experiment|all> [suite]"
    echo ""
    echo "Suites: ${TASK_SUITES[*]}"
    exit 1
fi

if [ "$EXP_TARGET" = "all" ]; then
    # Evaluate all experiments that have checkpoints
    for config_file in "$CONFIG_DIR"/e*_*.json; do
        exp_name=$(basename "$config_file" .json)
        if [ -n "$SUITE_TARGET" ]; then
            run_eval "$exp_name" "$SUITE_TARGET"
        else
            for suite in "${TASK_SUITES[@]}"; do
                run_eval "$exp_name" "$suite"
            done
        fi
    done
else
    if [ -n "$SUITE_TARGET" ]; then
        run_eval "$EXP_TARGET" "$SUITE_TARGET"
    else
        for suite in "${TASK_SUITES[@]}"; do
            run_eval "$EXP_TARGET" "$suite"
        done
    fi
fi

echo ""
echo "Evaluation complete. Logs saved to: $RESULTS_DIR/"
