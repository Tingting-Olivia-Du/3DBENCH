#!/bin/bash
# Download prebuilt modified_libero_rlds TFDS datasets (openvla LIBERO).
# These already contain a `wrist_image` stream and are stored UPRIGHT.
#
# SOURCE (verified 2026-06-19):
#   Primary: A pre-downloaded copy already exists on this machine at:
#     /workspace/ghsun/sf/densf/openvla-SF/modified_libero_rlds/
#   This script symlinks the required suite into DEST to avoid a redundant
#   multi-GB download.  If the local copy is absent (e.g., on a fresh machine),
#   the fallback is to download from HuggingFace:
#     Dataset ID: openvla/modified_libero_rlds
#     HF mirror:  https://hf-mirror.com  (set HF_ENDPOINT)
#     Clone cmd:  git clone git@hf.co:datasets/openvla/modified_libero_rlds
#     OR via CLI: huggingface-cli download openvla/modified_libero_rlds \
#                   --repo-type dataset --include "<suite>/*" --local-dir "$DEST"
#   The openvla/modified_libero_rlds repo is documented in:
#     /workspace/tingting/VLM4VLA/openvla/README.md
#     /workspace/tingting/openvla-oft/LIBERO.md
set -euo pipefail

DEST=/workspace/tingting/LIBERO_rlds
mkdir -p "$DEST"

CONDA=/workspace/ghsun/miniconda3/condabin/conda
ENV=/workspace/tingting/envs/vlmbench-rlds

# Only e0 needs libero_10 to start; pass "all" to fetch the 4 suites.
SUITES="${1:-libero_10_no_noops}"
if [ "$SUITES" = "all" ]; then
  SUITES="libero_10_no_noops libero_goal_no_noops libero_object_no_noops libero_spatial_no_noops"
fi

# Local machine path where the dataset already lives (avoids re-download).
LOCAL_SRC=/workspace/ghsun/sf/densf/openvla-SF/modified_libero_rlds

for s in $SUITES; do
  if [ -d "$DEST/$s" ]; then
    echo "[skip] $s already exists at $DEST/$s"
  elif [ -d "$LOCAL_SRC/$s" ]; then
    echo "[symlink] $s from local copy -> $DEST/$s"
    ln -s "$LOCAL_SRC/$s" "$DEST/$s"
  else
    echo "[download] $s from HuggingFace -> $DEST/$s"
    # Mirror via hf-mirror (matches main.py HF_ENDPOINT convention).
    export HF_ENDPOINT="https://hf-mirror.com"
    $CONDA run -p "$ENV" pip install -q "huggingface_hub[cli]"
    $CONDA run -p "$ENV" huggingface-cli download \
        openvla/modified_libero_rlds --repo-type dataset \
        --include "$s/*" --local-dir "$DEST"
  fi
done

echo "[download] done -> $DEST"
ls "$DEST"

# ---- Verify TFDS loads it and has wrist_image ----
echo ""
echo "[verify] Running TFDS check for libero_10_no_noops ..."
$CONDA run -p "$ENV" python3 - <<'PYEOF'
import tensorflow_datasets as tfds
b = tfds.builder("libero_10_no_noops", data_dir="/workspace/tingting/LIBERO_rlds")
print("splits:", {k: v.num_examples for k, v in b.info.splits.items()})
feat = b.info.features["steps"]["observation"]
print("obs keys:", list(feat.keys()))
assert "image" in feat and "wrist_image" in feat, "FATAL: missing wrist_image stream"
print("TFDS OK with wrist_image")
PYEOF
