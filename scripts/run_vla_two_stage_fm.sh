#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# Evo-1-style two-stage Flow-Matching VLA training (RLDS dualcam baseline)
#
# Usage:
#   bash scripts/run_vla_two_stage_fm.sh                 # run stage1 then stage2
#   STAGE=stage1 bash scripts/run_vla_two_stage_fm.sh    # only stage1
#   STAGE1_CKPT=/path/to.ckpt STAGE=stage2 bash scripts/run_vla_two_stage_fm.sh  # skip stage1
#   DRY_RUN=1 bash scripts/run_vla_two_stage_fm.sh       # print commands only
#   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 bash scripts/run_vla_two_stage_fm.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"               # 3DBENCH/
VLM4VLA_ROOT="$(dirname "$PROJECT_ROOT")/VLM4VLA"     # VLM4VLA/
CONFIG_DIR="$PROJECT_ROOT/configs/vla_ablation_rlds"

STAGE1_CFG="$CONFIG_DIR/fm_dualcam_stage1.json"
STAGE2_CFG="$CONFIG_DIR/fm_dualcam_stage2.json"
STAGE1_CKPT_DIR="$VLM4VLA_ROOT/runs/vla_two_stage_fm/stage1/checkpoints"

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_PORT=${MASTER_PORT:-6044}
STAGE=${STAGE:-all}
LOSS_TYPE=${LOSS_TYPE:-l1_unified}

export TMPDIR="${TMPDIR:-/workspace/tingting/.tmp}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export WANDB_PROJECT=${WANDB_PROJECT:-vla_two_stage_fm}

run_main() {
    local config_path="$1"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[DRY RUN] cd %s && torchrun --nnodes %s --node_rank 0 --nproc_per_node %s --master_addr 127.0.0.1 --master_port %s main.py %s --gpus %s --num_nodes %s --loss_type %s\n' \
            "$VLM4VLA_ROOT" "$NUM_NODES" "$GPUS_PER_NODE" "$MASTER_PORT" \
            "$config_path" "$GPUS_PER_NODE" "$NUM_NODES" "$LOSS_TYPE"
    else
        cd "$VLM4VLA_ROOT" && torchrun \
            --nnodes "$NUM_NODES" --node_rank 0 --nproc_per_node "$GPUS_PER_NODE" \
            --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
            main.py "$config_path" --gpus "$GPUS_PER_NODE" --num_nodes "$NUM_NODES" --loss_type "$LOSS_TYPE"
    fi
}

latest_ckpt() {
    ls -t "$STAGE1_CKPT_DIR"/*.ckpt 2>/dev/null | head -n 1
}

run_stage1() {
    echo "================ STAGE 1 (freeze VLM, train FMDecoder) ================"
    run_main "$STAGE1_CFG"
}

run_stage2() {
    local ckpt="${STAGE1_CKPT:-}"
    if [ -z "$ckpt" ]; then
        ckpt="$(latest_ckpt || true)"
    fi
    if [ "${DRY_RUN:-0}" != "1" ] && [ -z "$ckpt" ]; then
        echo "[ERROR] No stage1 checkpoint found in $STAGE1_CKPT_DIR (set STAGE1_CKPT=...)"
        exit 1
    fi
    echo "================ STAGE 2 (full finetune, resume_pretrain) ============="
    echo "  stage1 ckpt: ${ckpt:-<dry-run-unknown>}"

    # Generate a temp stage2 config with model_load_path filled in (do not mutate the original).
    local tmp_cfg="$TMPDIR/fm_dualcam_stage2.filled.$$.json"
    trap 'rm -f "$tmp_cfg"' RETURN
    if [ "${DRY_RUN:-0}" = "1" ] && [ -z "$ckpt" ]; then
        ckpt="DRY_RUN_PLACEHOLDER.ckpt"
    fi
    # No jq in this env; inject model_load_path via the conda env python.
    CKPT="$ckpt" SRC="$STAGE2_CFG" DST="$tmp_cfg" \
      conda run -p /workspace/tingting/envs/vlmbench-rlds python3 -c "
import json, os
c = json.load(open(os.environ['SRC']))
c['model_load_path'] = os.environ['CKPT']
json.dump(c, open(os.environ['DST'], 'w'), indent=4, ensure_ascii=False)
"
    echo "  filled config: $tmp_cfg"
    run_main "$tmp_cfg"
}

case "$STAGE" in
    stage1) run_stage1 ;;
    stage2) run_stage2 ;;
    all)    run_stage1; run_stage2 ;;
    *) echo "Unknown STAGE=$STAGE (use stage1|stage2|all)"; exit 1 ;;
esac

echo ""
echo "Done."
