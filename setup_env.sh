#!/usr/bin/env bash
# ============================================================
# setup_env.sh  ―  Create and populate the vlmbench conda env
#
# Usage:
#   bash setup_env.sh [--env-name NAME] [--env-prefix PATH] [--cuda 12.1]
#
# Defaults:
#   --env-name    vlmbench
#   --env-prefix  /umd-datapool/tingting/envs/vlmbench
#   --cuda        12.1
#
# The env is placed on /umd-datapool (large partition) to avoid
# the 100%-full /data issue.
# ============================================================

set -euo pipefail

# ---------- defaults ----------------------------------------
ENV_NAME="vlmbench"
ENV_PREFIX="/umd-datapool/tingting/envs/vlmbench"
CUDA_VER="12.8"      # server driver supports CUDA 12.8
PYTHON_VER="3.12"    # lerobot requires >=3.12

# Repo roots (assumed to be siblings of 3DBENCH)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"          # /umd-datapool/tingting
LIBERO_DIR="${REPO_ROOT}/LIBERO"
LEROBOT_DIR="${SCRIPT_DIR}/lerobot"
BENCH_DIR="$SCRIPT_DIR"

# ---------- argument parsing --------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --env-name)   ENV_NAME="$2";   shift 2 ;;
    --env-prefix) ENV_PREFIX="$2"; shift 2 ;;
    --cuda)       CUDA_VER="$2";   shift 2 ;;
    --python)     PYTHON_VER="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------- helpers -----------------------------------------
log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
ok()   { echo -e "\033[1;32m  ✓\033[0m $*"; }
warn() { echo -e "\033[1;33m  !\033[0m $*"; }

# ---------- sanity checks -----------------------------------
log "Pre-flight checks"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Make sure Miniconda/Anaconda is on PATH."
  exit 1
fi
ok "conda found: $(conda --version)"

if [[ ! -d "$LIBERO_DIR" ]]; then
  warn "LIBERO repo not found at $LIBERO_DIR — LIBERO install will be skipped."
  SKIP_LIBERO=1
else
  ok "LIBERO repo: $LIBERO_DIR"
  SKIP_LIBERO=0
fi

if [[ ! -d "$LEROBOT_DIR" ]]; then
  warn "LeRobot repo not found at $LEROBOT_DIR — LeRobot install will be skipped."
  SKIP_LEROBOT=1
else
  ok "LeRobot repo: $LEROBOT_DIR"
  SKIP_LEROBOT=0
fi

# ---------- create / reuse env ------------------------------
log "Setting up conda env: $ENV_PREFIX  (Python $PYTHON_VER)"

mkdir -p "$(dirname "$ENV_PREFIX")"

if conda env list | grep -qE "(^| )${ENV_PREFIX}( |$)"; then
  warn "Env already exists at $ENV_PREFIX — reusing it."
else
  conda create -y --prefix "$ENV_PREFIX" python="$PYTHON_VER"
  ok "Env created."
fi

# Activate
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
ok "Activated: $(which python)  [$(python --version)]"

# Redirect pip cache to the large partition (avoids /data filling up)
PIP_CACHE="${REPO_ROOT}/.pip-cache"
mkdir -p "$PIP_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE"
ok "pip cache → $PIP_CACHE"

# ---------- 1. PyTorch (with CUDA) --------------------------
log "Step 1/5 — PyTorch (CUDA ${CUDA_VER})"

# Pin to the highest versions allowed by lerobot: torch>=2.7,<2.11 / torchvision>=0.22,<0.26
# cu128 wheels: torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0
TORCH_INDEX="https://download.pytorch.org/whl/cu$(echo "$CUDA_VER" | tr -d '.')"
pip install --cache-dir "$PIP_CACHE" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0" \
    --index-url "$TORCH_INDEX"
ok "PyTorch 2.10.0+cu128 installed."

# ---------- 2. LeRobot + LIBERO (via lerobot[libero]) --------
# lerobot's "libero" extra installs hf-libero, which is the HuggingFace-maintained
# LIBERO fork and pulls in all required deps:
#   robosuite==1.4.0, robomimic==0.2.0, bddl==1.0.1, mujoco>=3.0.0,
#   gymnasium, hydra-core, easydict, einops, opencv-python, etc.
log "Step 2/5 — LeRobot + LIBERO (lerobot[libero])"

if [[ "$SKIP_LEROBOT" -eq 0 ]]; then
  pip install --cache-dir "$PIP_CACHE" \
      -e "${LEROBOT_DIR}[libero]"
  ok "LeRobot + hf-libero installed from $LEROBOT_DIR"
else
  # Fall back: install hf-libero directly (still brings all LIBERO deps)
  warn "LeRobot repo not found — installing hf-libero directly from PyPI."
  pip install --cache-dir "$PIP_CACHE" "hf-libero>=0.1.3"
  ok "hf-libero installed."
fi

# If a local LIBERO repo also exists, install it on top so local edits take effect.
if [[ "$SKIP_LIBERO" -eq 0 ]]; then
  log "  Also installing local LIBERO repo (takes precedence over hf-libero)"
  pip install --cache-dir "$PIP_CACHE" -e "$LIBERO_DIR" --no-deps
  ok "Local LIBERO installed from $LIBERO_DIR"
fi

# ---------- 4. 3DBench core requirements --------------------
log "Step 4/5 — 3DBench core requirements"

pip install --cache-dir "$PIP_CACHE" \
    "numpy>=1.24" \
    "Pillow>=10.0" \
    "tqdm>=4.66" \
    "pyyaml>=6.0" \
    "scikit-learn>=1.3" \
    "scipy>=1.11"
ok "Core deps installed."

# ---------- 5. VLM inference libraries ----------------------
log "Step 5/5 — VLM inference libraries"

pip install --cache-dir "$PIP_CACHE" \
    "transformers>=4.45" \
    "accelerate>=0.30" \
    "qwen-vl-utils>=0.0.8" \
    sentencepiece \
    decord

# timm is required by InternVL2; install explicitly with cache redirect
pip install --cache-dir "$PIP_CACHE" timm
ok "VLM deps installed (transformers, accelerate, qwen-vl-utils, timm)."

# ---------- smoke test --------------------------------------
log "Smoke test"

python - <<'PYEOF'
import importlib, sys

checks = [
    ("torch",        lambda m: f"v{m.__version__}, CUDA={m.cuda.is_available()}"),
    ("numpy",        lambda m: f"v{m.__version__}"),
    ("PIL",          lambda m: f"v{m.__version__}"),
    ("transformers", lambda m: f"v{m.__version__}"),
    ("accelerate",   lambda m: f"v{m.__version__}"),
    ("timm",         lambda m: f"v{m.__version__}"),
    ("sklearn",      lambda m: f"v{m.__version__}"),
    ("yaml",         lambda m: "ok"),
    ("tqdm",         lambda m: "ok"),
]

ok, fail = [], []
for name, info in checks:
    try:
        m = importlib.import_module(name)
        ok.append(f"  ✓  {name:20s} {info(m)}")
    except Exception as e:
        fail.append(f"  ✗  {name:20s} {e}")

print("\n".join(ok))
if fail:
    print("\nFAILED:")
    print("\n".join(fail))
    sys.exit(1)
else:
    print("\nAll checks passed.")
PYEOF

# ---------- optional: register env in Jupyter ---------------
if command -v jupyter &>/dev/null; then
  pip install --cache-dir "$PIP_CACHE" ipykernel -q
  python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
  ok "Jupyter kernel registered as '$ENV_NAME'."
fi

# ---------- done --------------------------------------------
echo ""
echo "=================================================="
echo " Environment ready."
echo "   Activate:  conda activate $ENV_PREFIX"
echo "   Run bench: bash $BENCH_DIR/run_benchmark.sh"
echo "=================================================="
