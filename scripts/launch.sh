#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# launch.sh — hang ONE VLA ablation training on a chosen GPU (background)
#
# Usage:
#   bash scripts/launch.sh <exp> <gpu> [port]
#
#   <exp>   experiment name, e.g. e0_head, e8_full  (configs/vla_ablation/<exp>.json)
#   <gpu>   GPU index to pin (CUDA_VISIBLE_DEVICES), e.g. 4
#   [port]  torchrun master_port; default 6100+gpu (so each GPU gets a unique port)
#
# Examples:
#   bash scripts/launch.sh e0_full 4          # original Qwen backbone, full finetune, GPU4
#   bash scripts/launch.sh e8_head 1 6101     # explicit port
#
# Logs:  .tmp/vla_runs/<exp>_gpu<gpu>.log   (tail -f to watch)
# Stop:  pkill -f "master_port <port>"
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP="${1:-}"
GPU="${2:-}"
if [ -z "$EXP" ] || [ -z "$GPU" ]; then
    echo "Usage: bash scripts/launch.sh <exp> <gpu> [port]"
    echo "  e.g. bash scripts/launch.sh e0_full 4"
    exit 1
fi
PORT="${3:-$((6100 + GPU))}"

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                 # 3DBENCH/
WORKSPACE="$(dirname "$PROJECT_ROOT")"                  # /workspace/tingting
VLM4VLA_ROOT="$WORKSPACE/VLM4VLA"
VENV_BIN="$WORKSPACE/envs/vlmbench/bin"                 # py3.12 training env
CONFIG="$PROJECT_ROOT/configs/vla_ablation/${EXP}.json"
LOG_DIR="$WORKSPACE/.tmp/vla_runs"
LOG="$LOG_DIR/${EXP}_gpu${GPU}.log"

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi
if [ ! -x "$VENV_BIN/torchrun" ]; then
    echo "[ERROR] vlmbench torchrun not found at $VENV_BIN/torchrun"
    exit 1
fi
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  exp:    $EXP"
echo "  config: $CONFIG"
echo "  GPU:    $GPU      master_port: $PORT"
echo "  log:    $LOG"
echo "================================================================"

cd "$VLM4VLA_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
PATH="$VENV_BIN:$PATH" \
WANDB_MODE="${WANDB_MODE:-offline}" \
PYTHONUNBUFFERED=1 \
nohup torchrun \
    --nnodes 1 --node_rank 0 --nproc_per_node 1 \
    --master_addr 127.0.0.1 --master_port "$PORT" \
    main.py "$CONFIG" \
    --gpus 1 --num_nodes 1 \
    > "$LOG" 2>&1 &

PID=$!
echo "launched pid $PID  ->  tail -f $LOG"
echo "stop with:  pkill -f \"master_port $PORT\""
