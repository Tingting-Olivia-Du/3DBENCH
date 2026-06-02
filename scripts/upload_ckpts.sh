#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# upload_ckpts.sh — upload the latest checkpoint of each active VLA ablation
#                   experiment to a Hugging Face model repo.
#
# Usage:
#   bash scripts/upload_ckpts.sh <HF_USER> [REPO_NAME] [--private] [--public]
#
#   <HF_USER>    your Hugging Face username/org (e.g. tingtingdu06)
#   [REPO_NAME]  repo name (default: vla-ablation-libero10-head)
#   --private    create as private (default)
#   --public     create as public
#
# Examples:
#   bash scripts/upload_ckpts.sh Tingtingdu
#   bash scripts/upload_ckpts.sh TingtingDu vla-ablation-libero10-head --public
#
# What it does:
#   1. Resolves the LATEST .ckpt for each experiment (by step number).
#   2. Creates the HF repo if missing.
#   3. Uploads each ckpt to <exp>/<filename> in the repo.
#
# Requires: hf CLI logged in (hf auth whoami).  ckpts are ~7GB each.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── args ─────────────────────────────────────────────────────────────────
HF_USER="${1:-}"
if [ -z "$HF_USER" ]; then
    echo "Usage: bash scripts/upload_ckpts.sh <HF_USER> [REPO_NAME] [--private|--public]"
    exit 1
fi
REPO_NAME="vla-ablation-libero10-head"
VISIBILITY="--private"
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --private) VISIBILITY="--private" ;;
        --public)  VISIBILITY="" ;;          # public = omit the flag
        *)         REPO_NAME="$1" ;;
    esac
    shift
done
REPO="$HF_USER/$REPO_NAME"

# ── config ───────────────────────────────────────────────────────────────
HF="/workspace/tingting/envs/vlmbench/bin/hf"
ROOT="/workspace/tingting/VLM4VLA/runs/vla_ablation"
# EXPS=(e0_head e8_head e1_head e2_head e4_head)   # the 5 active experiments
EXPS=(e2_head) 
# ── sanity ───────────────────────────────────────────────────────────────
if [ ! -x "$HF" ]; then echo "[ERROR] hf CLI not found at $HF"; exit 1; fi
echo "== HF identity =="
"$HF" auth whoami || { echo "[ERROR] not logged in. Run: $HF auth login"; exit 1; }

# ── find latest ckpt per experiment ──────────────────────────────────────
declare -A CKPT
echo ""
echo "== resolving latest checkpoint per experiment =="
missing=0
for e in "${EXPS[@]}"; do
    # newest by step number: sort by the numeric step in 'step=N.ckpt'
    ck=$(find "$ROOT/$e" -name "*.ckpt" 2>/dev/null \
         | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' \
         | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then
        echo "  [WARN] $e: no .ckpt found yet (skipping)"
        missing=$((missing+1))
        continue
    fi
    CKPT["$e"]="$ck"
    sz=$(du -h "$ck" | cut -f1)
    echo "  $e -> $(basename "$ck")  ($sz)"
done

if [ ${#CKPT[@]} -eq 0 ]; then
    echo "[ERROR] no checkpoints found for any experiment. Nothing to upload."
    exit 1
fi

# ── create repo (idempotent) ─────────────────────────────────────────────
echo ""
echo "== creating repo $REPO (if missing) =="
"$HF" repo create "$REPO" --repo-type model $VISIBILITY -y 2>/dev/null \
    || echo "  (repo may already exist — continuing)"

# ── upload ───────────────────────────────────────────────────────────────
echo ""
echo "== uploading =="
for e in "${EXPS[@]}"; do
    ck="${CKPT[$e]:-}"
    [ -z "$ck" ] && continue
    dest="$e/$(basename "$ck")"
    echo "--- $e -> $REPO:$dest ---"
    "$HF" upload "$REPO" "$ck" "$dest" --repo-type model
done

echo ""
echo "== done =="
echo "  https://huggingface.co/$REPO"
