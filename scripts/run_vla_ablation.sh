#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# VLA Ablation Training Launcher
#
# Usage:
#   # Run a single experiment
#   bash scripts/run_vla_ablation.sh e0_full
#
#   # Run all experiments (priority order)
#   bash scripts/run_vla_ablation.sh all
#
#   # Run only full-finetune experiments
#   bash scripts/run_vla_ablation.sh full
#
#   # Run only head-only experiments
#   bash scripts/run_vla_ablation.sh head
#
#   # Dry run (print commands only)
#   DRY_RUN=1 bash scripts/run_vla_ablation.sh all
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"               # 3DBENCH/
VLM4VLA_ROOT="$(dirname "$PROJECT_ROOT")/VLM4VLA"     # VLM4VLA/
CONFIG_DIR="$PROJECT_ROOT/configs/vla_ablation"

# ── GPU config ─────────────────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_PORT=${MASTER_PORT:-6042}

# ── Environment ────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

# ── Priority order for experiments ─────────────────────────────────────
PRIORITY_ORDER=(
    e8_head e8_full
    e1_head e1_full
    e3_head e3_full
    e9_head e9_full
    e2_head e2_full
    e7_head e7_full
    e6_head e6_full
    e4_head e4_full
    e5_head e5_full
    e0_head e0_full
)

run_experiment() {
    local exp_name="$1"
    local config_path="$CONFIG_DIR/${exp_name}.json"

    if [ ! -f "$config_path" ]; then
        echo "[ERROR] Config not found: $config_path"
        return 1
    fi

    echo ""
    echo "================================================================"
    echo "  Running: $exp_name"
    echo "  Config:  $config_path"
    echo "  GPUs:    $GPUS_PER_NODE x $NUM_NODES nodes"
    echo "================================================================"

    local cmd="cd $VLM4VLA_ROOT && torchrun \
        --nnodes $NUM_NODES \
        --node_rank 0 \
        --nproc_per_node $GPUS_PER_NODE \
        --master_addr 127.0.0.1 \
        --master_port $MASTER_PORT \
        main.py \
        $config_path \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NUM_NODES"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    echo "Usage: bash scripts/run_vla_ablation.sh <experiment|all|full|head>"
    echo ""
    echo "Experiments: ${PRIORITY_ORDER[*]}"
    exit 1
fi

case "$TARGET" in
    all)
        for exp in "${PRIORITY_ORDER[@]}"; do
            run_experiment "$exp"
        done
        ;;
    full)
        for exp in "${PRIORITY_ORDER[@]}"; do
            [[ "$exp" == *_full ]] && run_experiment "$exp"
        done
        ;;
    head)
        for exp in "${PRIORITY_ORDER[@]}"; do
            [[ "$exp" == *_head ]] && run_experiment "$exp"
        done
        ;;
    *)
        run_experiment "$TARGET"
        ;;
esac

echo ""
echo "Done."
