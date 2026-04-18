#!/usr/bin/env bash
# run_benchmark.sh  –  End-to-end VLM spatial reasoning benchmark pipeline.
#
# Usage:
#   bash run_benchmark.sh [options]
#
# Options (all optional, defaults shown):
#   --suite            libero_spatial    LIBERO suite(s): single name, comma-separated, or 'all'
#   --n_states         5                 Init states per task (start-of-episode frames)
#   --n_traj_frames    0                 Frames to sample per init-state along the approach
#                                        trajectory (0 = disabled; >0 gives temporal coverage)
#   --n_close          3                 Extra frames per task where can_close=True
#                                        (P-controller approach; 0 = disabled)
#   --close_max_steps  300               Max sim steps for approach / trajectory runs
#   --models           "qwen2.5-vl-7b random"  Space-separated model list
#   --device           cuda:0            GPU device
#   --max_new_tokens   512
#   --gt_dir           data/gt           Where to save / read GT
#   --resp_dir         data/runs/<timestamp>      Response + results root (default: auto)
#   --results          <resp_dir>/results.json    Metrics output (default: auto)
#   --report           <resp_dir>/report.md       Markdown report (default: auto)
#   --skip_gt          (flag) skip step 1 if GT already exists
#   --skip_eval        (flag) skip step 2, only recompute metrics
#   --resume           (flag) pass --resume to 02_run_vlm_eval.py
#   --task_ids         ""                Space-separated task IDs (empty = all)
#   --libero_path      ""                Custom LIBERO install path
#
# Output layout (per run):
#   data/runs/<timestamp>/
#   ├── qwen2.5-vl-7b/sample_*.json   ← model responses
#   ├── random/sample_*.json
#   ├── results.json                  ← aggregated metrics
#   ├── comparison_qwen2.5-vl-7b.json ← per-sample GT vs pred
#   └── report.md                     ← full markdown report
#
# Examples:
#   # Full run on GPU 1, only Qwen
#   bash run_benchmark.sh --device cuda:1 --models "qwen2.5-vl-7b"
#
#   # Sample trajectory frames (8 per init-state) + no extra close frames
#   bash run_benchmark.sh --n_traj_frames 8 --n_close 0
#
#   # Richer dataset: 5 init states + 8 traj frames each + 3 can_close frames
#   bash run_benchmark.sh --n_states 5 --n_traj_frames 8 --n_close 3
#
#   # GT already extracted, just re-evaluate with a different model
#   bash run_benchmark.sh --skip_gt --models "internvl2-8b" --device cuda:0
#
#   # Resume an interrupted eval run (reuse same run folder)
#   bash run_benchmark.sh --skip_gt --resume --resp_dir data/runs/20250416_143022 \
#                         --results data/runs/20250416_143022/results.json \
#                         --report  data/runs/20250416_143022/report.md

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SUITE="libero_spatial"
N_STATES=5
N_TRAJ_FRAMES=5
N_CLOSE=3
CLOSE_MAX_STEPS=300
MODELS="qwen2.5-vl-7b random"
DEVICE="cuda:0"
MAX_NEW_TOKENS=512
GT_DIR="data/gt"
SKIP_GT=0
SKIP_EVAL=0
RESUME=""
TASK_IDS=""
LIBERO_PATH=""
# These three default to the same run-timestamped folder; override individually if needed.
RESP_DIR=""    # default: data/runs/<RUN_TS>
RESULTS=""     # default: data/runs/<RUN_TS>/results.json
REPORT=""      # default: data/runs/<RUN_TS>/report.md

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)            SUITE="$2";            shift 2 ;;
        --n_states)         N_STATES="$2";        shift 2 ;;
        --n_traj_frames)    N_TRAJ_FRAMES="$2";   shift 2 ;;
        --n_close)          N_CLOSE="$2";         shift 2 ;;
        --close_max_steps)  CLOSE_MAX_STEPS="$2"; shift 2 ;;
        --models)           MODELS="$2";          shift 2 ;;
        --device)           DEVICE="$2";          shift 2 ;;
        --max_new_tokens)   MAX_NEW_TOKENS="$2";  shift 2 ;;
        --gt_dir)           GT_DIR="$2";          shift 2 ;;
        --resp_dir)         RESP_DIR="$2";        shift 2 ;;
        --results)          RESULTS="$2";         shift 2 ;;
        --report)           REPORT="$2";          shift 2 ;;
        --skip_gt)          SKIP_GT=1;            shift   ;;
        --skip_eval)        SKIP_EVAL=1;          shift   ;;
        --resume)           RESUME="--resume";    shift   ;;
        --task_ids)         TASK_IDS="$2";        shift 2 ;;
        --libero_path)      LIBERO_PATH="$2";     shift 2 ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Timestamp for this run
RUN_TS="$(date '+%Y%m%d_%H%M%S')"

# All outputs default to the same run folder: data/runs/<RUN_TS>/
RUN_DIR="data/runs/${RUN_TS}"
[[ -z "$RESP_DIR" ]] && RESP_DIR="${RUN_DIR}"
[[ -z "$RESULTS"  ]] && RESULTS="${RUN_DIR}/results.json"
[[ -z "$REPORT"   ]] && REPORT="${RUN_DIR}/report.md"

# Write all console output to both terminal and the markdown file.
# The markdown file starts with a header; subsequent appends come from scripts.
mkdir -p "$(dirname "$REPORT")"
cat > "$REPORT" <<MDHEADER
# 3DBENCH Run — $(date '+%Y-%m-%d %H:%M:%S')

**Suite:** \`${SUITE}\`  |  **Init states/task:** ${N_STATES}  |  **Models:** \`${MODELS}\`  |  **Device:** \`${DEVICE}\`

\`\`\`
MDHEADER

# Tee everything that follows into the open code block in the markdown file
exec > >(tee -a "$REPORT") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "$(printf '=%.0s' {1..60})"; }

hr
log "3DBENCH  –  VLM Spatial Reasoning Benchmark"
hr
log "Suite:       $SUITE  |  Init states/task: $N_STATES  |  Traj frames: $N_TRAJ_FRAMES  |  Close frames: $N_CLOSE"
log "Models:      $MODELS"
log "Device:      $DEVICE"
log "GT dir:      $GT_DIR"
log "Run folder:  $RESP_DIR"
log "  results.json  → $RESULTS"
log "  report.md     → $REPORT"
[[ -n "$TASK_IDS" ]]    && log "Task IDs:    $TASK_IDS"
[[ -n "$LIBERO_PATH" ]] && log "LIBERO path: $LIBERO_PATH"
[[ $SKIP_GT   -eq 1 ]]  && log "(skip_gt)   using existing GT"
[[ $SKIP_EVAL -eq 1 ]]  && log "(skip_eval) skipping model inference"
[[ -n "$RESUME" ]]       && log "(resume)    skipping already-done samples"
hr

# ---------------------------------------------------------------------------
# Step 1 – Extract ground truth
# ---------------------------------------------------------------------------
if [[ $SKIP_GT -eq 0 ]]; then
    log "STEP 1/3  Extract ground truth from LIBERO sim"

    GT_ARGS=(
        --suite           "$SUITE"
        --n_states        "$N_STATES"
        --n_traj_frames   "$N_TRAJ_FRAMES"
        --n_close         "$N_CLOSE"
        --close_max_steps "$CLOSE_MAX_STEPS"
        --out_dir         "$GT_DIR"
    )
    [[ -n "$TASK_IDS" ]]    && GT_ARGS+=(--task_ids $TASK_IDS)
    [[ -n "$LIBERO_PATH" ]] && GT_ARGS+=(--libero_path "$LIBERO_PATH")

    python scripts/01_extract_gt.py "${GT_ARGS[@]}"
    log "STEP 1 done  →  $GT_DIR/manifest.json"
else
    if [[ ! -f "$GT_DIR/manifest.json" ]]; then
        log "ERROR: --skip_gt was set but $GT_DIR/manifest.json not found."
        exit 1
    fi
    log "STEP 1 skipped  (using existing $GT_DIR/manifest.json)"
fi

hr

# ---------------------------------------------------------------------------
# Step 2 – Run VLM evaluation
# ---------------------------------------------------------------------------
if [[ $SKIP_EVAL -eq 0 ]]; then
    log "STEP 2/3  VLM evaluation"

    EVAL_ARGS=(
        --manifest       "$GT_DIR/manifest.json"
        --out_dir        "$RESP_DIR"
        --models         $MODELS
        --device         "$DEVICE"
        --max_new_tokens "$MAX_NEW_TOKENS"
    )
    [[ -n "$RESUME" ]] && EVAL_ARGS+=($RESUME)

    python scripts/02_run_vlm_eval.py "${EVAL_ARGS[@]}"
    log "STEP 2 done  →  $RESP_DIR/"
else
    log "STEP 2 skipped  (using existing responses in $RESP_DIR/)"
fi

hr

# ---------------------------------------------------------------------------
# Step 3 – Compute metrics
# ---------------------------------------------------------------------------
log "STEP 3/3  Compute metrics"

python scripts/03_compute_metrics.py \
    --manifest      "$GT_DIR/manifest.json" \
    --responses_dir "$RESP_DIR" \
    --out           "$RESULTS" \
    --report        "$REPORT"

log "STEP 3 done  →  $RESULTS"
hr
log "All done.  Report: $REPORT"

# Close the console-log code block opened in the markdown header
printf '```\n' >> "$REPORT"
