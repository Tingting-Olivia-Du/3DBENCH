#!/bin/bash
# Scale-out ETL driver: every-other chunk (0,2,4,...,244) = 123 chunks ~= 122K episodes.
# Resumable: etl_chunk.py skips chunks whose manifest exists.
export HF_HOME=/root/angli/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace/tingting/3dvla-stage0
for c in $(seq 0 2 244); do
  # disk guard: stop if less than 40GB free
  avail=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
  if [ "$avail" -lt 40 ]; then
    echo "[driver] STOP: only ${avail}GB free"; exit 1
  fi
  python3 scripts/etl_chunk.py --chunk "$c" --workers 16 2>&1 | tail -2
done
echo "[driver] ALL DONE"
