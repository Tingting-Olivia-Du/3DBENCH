#!/bin/bash
# Probe a stage1_none checkpoint with the standard dual-domain battery.
# Usage: probe_none_ckpt.sh <tag> <ckpt> <gpu_m1> <gpu_lib>
#   e.g. probe_none_ckpt.sh none60k .../steps_60000_model.safetensors 5 7
# Launches a detached tmux session probe_<tag>; scoring is done by the caller's
# watcher (score_bench.py --models stage1 --suffix _<tag>[_libero]).
set -eu
TAG=$1; CKPT=$2; GPU_M1=$3; GPU_LIB=$4
PY=/workspace/ghsun/miniconda3/envs/starVLA/bin/python
EV=/workspace/tingting/3dvla-stage0/scripts/eval_bench.py
LOG=/workspace/tingting/3dvla-checkpoints
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

tmux new-session -d -s "probe_$TAG" -n m1 \
  "HF_HOME=/root/angli/hf_cache $PY $EV --model stage1 --ckpt $CKPT \
   --suffix _$TAG --device cuda:$GPU_M1 2>&1 | tee $LOG/probe_${TAG}_m1.log"
tmux new-window -t "probe_$TAG" -n libero \
  "HF_HOME=/root/angli/hf_cache $PY $EV --model stage1 --ckpt $CKPT \
   --bench /workspace/tingting/3dvla-data/benchmark/libero_bench.jsonl \
   --suffix _${TAG}_libero --device cuda:$GPU_LIB 2>&1 | tee $LOG/probe_${TAG}_libero.log"
echo "launched tmux probe_$TAG (m1 on cuda:$GPU_M1, libero on cuda:$GPU_LIB)"
