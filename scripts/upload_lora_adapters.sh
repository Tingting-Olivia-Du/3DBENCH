#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# upload_lora_adapters.sh — upload the per-dimension LoRA adapters to HF.
#
# For each subdir (q1..q8, all) under the LoRA root, picks:
#   - "final" if it exists, else
#   - the highest-numbered "checkpoint-N"
# and uploads that WHOLE folder to <repo>/<dim>/ on Hugging Face.
#
# Usage:
#   bash scripts/upload_lora_adapters.sh <HF_USER> [REPO_NAME] [--private|--public]
#
# Examples:
#   bash scripts/upload_lora_adapters.sh TingtingDu
#   bash scripts/upload_lora_adapters.sh TingtingDu qwen25vl-mv-lora-perdim --public
#
# Requires: hf CLI logged in with WRITE token (hf auth whoami).
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

HF_USER="${1:-}"
if [ -z "$HF_USER" ]; then
    echo "Usage: bash scripts/upload_lora_adapters.sh <HF_USER> [REPO_NAME] [--private|--public]"
    exit 1
fi
REPO_NAME="qwen25vl-mv-lora-perdim"
VISIBILITY="--private"
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --private) VISIBILITY="--private" ;;
        --public)  VISIBILITY="" ;;
        *)         REPO_NAME="$1" ;;
    esac
    shift
done
REPO="$HF_USER/$REPO_NAME"

HF="/workspace/tingting/envs/vlmbench/bin/hf"
LORA_ROOT="/workspace/tingting/3DBENCH/models/0515/qwen2.5-vl-3b-mv-lora"

if [ ! -x "$HF" ]; then echo "[ERROR] hf CLI not found at $HF"; exit 1; fi
if [ ! -d "$LORA_ROOT" ]; then echo "[ERROR] LoRA root not found: $LORA_ROOT"; exit 1; fi

echo "== HF identity =="
"$HF" auth whoami || { echo "[ERROR] not logged in. Run: $HF auth login"; exit 1; }

# ── resolve which folder to upload per dimension ──────────────────────────
declare -A SRC
echo ""
echo "== resolving folder per dimension (final, else highest checkpoint-N) =="
for dim in $(ls "$LORA_ROOT"); do
    base="$LORA_ROOT/$dim"
    [ -d "$base" ] || continue
    if [ -d "$base/final" ]; then
        sel="final"
    else
        # highest-numbered checkpoint-N
        sel=$(ls "$base" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' \
              | sed -E 's/checkpoint-([0-9]+)/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    if [ -z "$sel" ] || [ ! -d "$base/$sel" ]; then
        echo "  [WARN] $dim: no final/checkpoint found (skipping)"
        continue
    fi
    SRC["$dim"]="$base/$sel"
    echo "  $dim -> $sel  ($(du -sh "$base/$sel" | cut -f1))"
done

if [ ${#SRC[@]} -eq 0 ]; then
    echo "[ERROR] nothing to upload."; exit 1
fi

# ── create repo (idempotent) ──────────────────────────────────────────────
echo ""
echo "== creating repo $REPO (if missing) =="
"$HF" repo create "$REPO" --repo-type model $VISIBILITY -y 2>/dev/null \
    || echo "  (repo may already exist — continuing)"

# ── upload each folder to <repo>/<dim>/ ───────────────────────────────────
echo ""
echo "== uploading =="
for dim in $(echo "${!SRC[@]}" | tr ' ' '\n' | sort); do
    src="${SRC[$dim]}"
    echo "--- $dim  ($src)  ->  $REPO:$dim/ ---"
    "$HF" upload "$REPO" "$src" "$dim" --repo-type model
done

echo ""
echo "== done =="
echo "  https://huggingface.co/$REPO"
