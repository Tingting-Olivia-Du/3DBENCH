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

# 12) timm — required by prismatic (vision backbone registry); missing from
#     prismatic's install_requires when installed with --no-deps (step 6).
#     Without this, `import prismatic` raises ModuleNotFoundError: No module named 'timm'.
$PIP install "timm==1.0.27"

# 13) omegaconf — imported by vlm4vla.data.calvin_dataset, which is pulled in
#     by vlm4vla/data/__init__.py. Without it, `from vlm4vla.data import
#     OpenVLADataset` (and main.py) raise ModuleNotFoundError: No module named
#     'omegaconf'.
$PIP install "omegaconf==2.3.1"

# 14) vlm4vla backbone/runtime deps pulled in by main.py's import chain
#     (model builders, video/image utils). Missing from the --no-deps installs
#     above; surfaced when launching real training via run_vla_ablation.sh.
$PIP install \
    "qwen-vl-utils==0.0.14" \
    "diffusers==0.38.0" \
    "open_clip_torch==2.20.0" \
    "flamingo-pytorch==0.1.2" \
    "hydra-core==1.3.3" \
    "decord==0.6.0" \
    "accelerate==1.14.0"

# 15) flash-attn 2.8.3 — vlm_builder.py hardcodes attn_implementation=
#     "flash_attention_2". MUST match the working vlmbench (HDF5) env's
#     flash-attn 2.8.3 so the RLDS run is a true parity reference. The source
#     build fails here; use the official prebuilt wheel for cp311 / torch2.7 /
#     cu12. ABI variant = abiTRUE because torch._C._GLIBCXX_USE_CXX11_ABI is True.
$PIP install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# 16) LIBERO simulator — ONLY needed for closed-loop eval (run_libero_eval.py:
#     `from libero.libero import benchmark`), NOT for RLDS training. The local
#     LIBERO repo is missing its top-level package __init__.py, so find_packages()
#     returns nothing and the editable install is an empty 5KB shell. Create the
#     (empty) __init__.py first, then install --no-deps (its requirements.txt
#     pins numpy 1.22 / transformers 4.21 which would wreck this env).
LIBERO_ROOT=/workspace/tingting/LIBERO
touch "$LIBERO_ROOT/libero/__init__.py"
$PIP install --no-deps -e "$LIBERO_ROOT"

# 17) LIBERO runtime sim deps, versions matched to the working vlmbench env.
#     CAUTION: robosuite/robomimic/numba pull numpy>=2 and a newer opencv, which
#     break the TF 2.15 stack (needs numpy<2). Re-pin numpy 1.26.4 + opencv
#     4.6.0.66 AFTER installing them so the RLDS data path keeps working.
$PIP install \
    "robosuite==1.4.0" "robomimic==0.2.0" "bddl==1.0.1" "mujoco==3.8.0" \
    "easydict" "egl_probe" "glfw==2.10.0" "thop==0.1.1-2209072238" "cloudpickle"
$PIP install "numpy==1.26.4" "opencv-python==4.6.0.66"

# 18) Extra eval-only deps imported transitively by eval/libero/* (pose math,
#     LIBERO env base classes). Surfaced when launching run_libero_eval.py.
$PIP install "transforms3d==0.4.2" "future==1.0.0" "gym==0.26.2"

echo "[setup_rlds_env] done. Prefix: $ENV_PREFIX"
