#!/usr/bin/env bash
# Stage-1 smoke: full-variant 5K convs x 100 steps on GPUs 4-7.
# Gotchas encoded here (memory-backed): explicit main_process_port (never 0 /
# 29500), checkpoints on the big rootfs, wandb offline, default HF cache
# (fast tokenizer + Qwen3-VL-4B live in /root/.cache/huggingface).
set -euo pipefail
cd /workspace/tingting/starVLA

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# user wandb: entity/project/API key/dirs (tingtingdu06-uw-madison / 3dvlm)
source /workspace/tingting/.wandb/env.sh
export WANDB_MODE=${WANDB_MODE:-online}
ACCEL_CONFIG=${ACCEL_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
# HF_HOME stays at /root/angli/hf_cache (profile): Qwen3-VL-4B lives there;
# the FAST tokenizer resolves via playground/Pretrained_models/fast symlink.

PYBIN=/workspace/ghsun/miniconda3/envs/starVLA/bin
PORT=$($PYBIN/python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "main_process_port=$PORT"

RUN_ID=${RUN_ID:-stage1_smoke_$(date +%m%d_%H%M)}

$PYBIN/accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes ${NPROC:-4} \
  --main_process_port "$PORT" \
  starVLA/training/train_starvlm.py \
  --config_yaml /workspace/tingting/3dvla-stage0/train/stage1.yaml \
  --run_id "$RUN_ID" \
  "$@" \
  2>&1 | tee /workspace/tingting/3dvla-checkpoints/"$RUN_ID".log
