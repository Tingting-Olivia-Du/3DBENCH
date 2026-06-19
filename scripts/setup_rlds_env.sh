#!/bin/bash
# Build the RLDS-capable training env (separate from vlmbench, which has
# numpy 2 / py3.12 and cannot host tensorflow 2.15).
set -euo pipefail

CONDA=/workspace/ghsun/miniconda3/condabin/conda
ENV_PREFIX=/workspace/tingting/envs/vlmbench-rlds
OPENVLA_ROOT=/workspace/tingting/VLM4VLA/openvla
VLM4VLA_ROOT=/workspace/tingting/VLM4VLA

# 1) Base env: python 3.11 (tf 2.15 has no 3.12 wheels).
$CONDA create -y -p "$ENV_PREFIX" python=3.11

PY="$CONDA run -p $ENV_PREFIX python"
PIP="$CONDA run -p $ENV_PREFIX pip"

# 2) torch matched to vlmbench (cu126 build).
$PIP install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126

# 3) TF/RLDS stack (these force numpy<2).
$PIP install "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" \
             "tensorflow_graphics==2021.12.3" "numpy<2"

# 4) dlimp (moojink fork; --no-deps so it doesn't drag conflicting pins).
$PIP install --no-deps "git+https://github.com/moojink/dlimp_openvla"

# 5) Training stack matched to vlmbench.
$PIP install "transformers==5.8.0" "lightning==2.6.5"

# 6) prismatic (openvla) + vlm4vla, editable, no-deps to preserve pins above.
$PIP install --no-deps -e "$OPENVLA_ROOT"
$PIP install --no-deps -e "$VLM4VLA_ROOT"

# 7) Transitive deps required by prismatic / vlm4vla (appended after first run).
$PIP install draccus einops rich wandb h5py pillow

# 8) torchvision matched to torch 2.7.1+cu126 (must pin explicitly to avoid
#    pip upgrading torch to latest).
$PIP install "torch==2.7.1" "torchvision==0.22.1" \
             --index-url https://download.pytorch.org/whl/cu126

# 9) openvla transitive deps (json-numpy, jsonlines, sentencepiece).
$PIP install json-numpy jsonlines "sentencepiece==0.1.99"

# 10) Fix protobuf / tensorflow-metadata conflict: tf 2.15 uses protobuf<5,
#     but tensorflow-metadata 1.21.0 requires protobuf>=5 (runtime_version).
#     Pin to 1.15.0 which is compatible with protobuf 4.x.
$PIP install "tensorflow-metadata==1.15.0" "protobuf==4.25.9"

# 11) tensorboardX (optional but listed in vlm4vla requirements).
$PIP install tensorboardX

echo "[setup_rlds_env] done. Prefix: $ENV_PREFIX"
