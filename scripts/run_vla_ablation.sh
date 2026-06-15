#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# VLA Ablation Training Launcher
#
# Usage:
#   # Run a single experiment
#   bash scripts/run_vla_ablation.sh e0_full
## 只用第 0 号卡跑单个实验
# CUDA_VISIBLE_DEVICES=2 MASTER_PORT=6046 bash scripts/run_vla_ablation.sh e4_full
# CUDA_VISIBLE_DEVICES=6 MASTER_PORT=6044 bash scripts/run_vla_ablation.sh e1_full
# # 用第 2、3 号卡(2 张),需要同时把 GPUS_PER_NODE 改成 2
# CUDA_VISIBLE_DEVICES=2,3 GPUS_PER_NODE=2 bash scripts/run_vla_ablation.sh e0_full
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
# Override with CONFIG_DIR=... to use a different config set
# (e.g. CONFIG_DIR=$PROJECT_ROOT/configs/vla_ablation_fm for the FMDecoder variants)
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/vla_ablation}"

# ── GPU config ─────────────────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_PORT=${MASTER_PORT:-6042}

# ── Action-loss mode ───────────────────────────────────────────────────
# Empty = use each config's act_head.loss_type (default split_bce). Set to
# "l1_unified" (openvla-oft parity: single equal-weight L1 over all 7 dims,
# gripper as raw {-1,+1} float regression) or "split_bce" to override.
#   LOSS_TYPE=l1_unified bash scripts/run_vla_ablation.sh e0_full
LOSS_TYPE=${LOSS_TYPE:-}

# ── Environment ────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
# wandb project for this training batch (flip + 0.65 norm). main.py reads
# WANDB_PROJECT (config "wandb_project" overrides it). Override on the CLI:
#   WANDB_PROJECT=my_proj bash scripts/run_vla_ablation.sh e1_full
export WANDB_PROJECT=${WANDB_PROJECT:-vla_ablation_fmdecoder}

# ── Priority order for experiments ─────────────────────────────────────
PRIORITY_ORDER=(
    e8_head e8_full
    e1_head e1_full
    e3_head e3_full
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

    if [ -n "$LOSS_TYPE" ]; then
        cmd="$cmd --loss_type $LOSS_TYPE"
    fi

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
