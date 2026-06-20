#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# VLA Ablation Evaluation Launcher
#
# All settings live in the CONFIG block below and can be overridden from the
# command line, e.g.:
#
#   # Single experiment on all suites, GPU 6, osmesa rendering
#   CUDA_DEVICE=6 MUJOCO_GL=osmesa bash scripts/run_vla_ablation_eval.sh e0_full
#   CUDA_DEVICE=7 bash scripts/run_vla_ablation_eval.sh e0_full_dualcam_bs256

#   # One suite only, with live wandb success-rate curves
#   USE_WANDB=1 bash scripts/run_vla_ablation_eval.sh e0_full libero_10
#
#   # Evaluate e0..e3 full on libero_10 only
#   EXPS="e0_full e1_full e2_full e3_full" bash scripts/run_vla_ablation_eval.sh "" libero_10
#
#   # Evaluate every experiment that has a checkpoint
#   bash scripts/run_vla_ablation_eval.sh all
#
#   # Dry run (print commands, don't execute)
#   DRY_RUN=1 bash scripts/run_vla_ablation_eval.sh e0_full
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — everything below is overridable via environment variables.   ║
# ║  Edit the defaults here, or pass VAR=... on the command line.          ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                       # 3DBENCH/
VLM4VLA_ROOT=${VLM4VLA_ROOT:-"$(dirname "$PROJECT_ROOT")/VLM4VLA"}   # VLM4VLA/
CONFIG_DIR=${CONFIG_DIR:-"$PROJECT_ROOT/configs/vla_ablation_rlds"}
# CONFIG_DIR=${CONFIG_DIR:-"$PROJECT_ROOT/configs/vla_ablation_fm"}
RESULTS_DIR=${RESULTS_DIR:-"$PROJECT_ROOT/results/vla_ablation_rlds"}
# RESULTS_DIR=${RESULTS_DIR:-"$PROJECT_ROOT/results/vla_ablation_fm"}
# Subdir under VLM4VLA/runs/ holding the checkpoints. Must match the configs'
# output_root (currently runs/vla_ablation_flip_norm065/). Override if needed.
RUNS_SUBDIR=${RUNS_SUBDIR:-"vla_ablation_rlds"}
# RUNS_SUBDIR=${RUNS_SUBDIR:-"vla_ablation_fm"}

# ── Conda / Python env ─────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-"/workspace/tingting/envs/vlmbench-rlds"}   # set "" to skip activation
# Make the VLM4VLA repo root importable (fixes `ModuleNotFoundError: No module named 'eval'`)
export PYTHONPATH="${VLM4VLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}                # live stdout flushing

# ── GPU ────────────────────────────────────────────────────────────────
CUDA_DEVICE=${CUDA_DEVICE:-7}

# ── Rendering backend (MuJoCo / robosuite headless) ───────────────────
# osmesa = CPU software GL (most portable, slower); egl = headless GPU GL;
# glfw = on-screen (needs a display). LIBERO/robosuite are headless → osmesa/egl.
MUJOCO_GL=${MUJOCO_GL:-osmesa}
export MUJOCO_GL
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_DEVICE}  # only used when MUJOCO_GL=egl

# ── Eval hyperparameters ───────────────────────────────────────────────
EXECUTE_STEP=${EXECUTE_STEP:-1}
NUM_TRIALS=${NUM_TRIALS:-10}                                  # episodes per task
# Override per-suite max rollout steps. Empty = keep suite defaults
# (spatial=220, object=280, goal=300, libero_10=520, libero_90=400).
# Set e.g. MAX_STEPS=400 to give the policy more time; unset to revert.
MAX_STEPS=${MAX_STEPS:-""}
# Default ON: configs now train with image_aug=true (0.9 resized crop) on both
# train and val splits, so eval must apply the matching 0.9 CENTER crop. Set
# CENTER_CROP=False only to eval an OLD checkpoint trained without any crop.
CENTER_CROP=${CENTER_CROP:-True}
# Restrict to specific task ids, comma-separated (e.g. "0,1"). Empty = all tasks.
TASK_IDS=${TASK_IDS:-""}
# Which experiments to run when first arg is empty (space-separated).
EXPS=${EXPS:-""}
# Which suites to run when no suite arg is given (space-separated).
# TASK_SUITES_STR=${TASK_SUITES:-"libero_spatial libero_object libero_goal libero_10"}
TASK_SUITES_STR=${TASK_SUITES:-"libero_10"}
read -r -a TASK_SUITES <<< "$TASK_SUITES_STR"

# ── wandb (live success-rate logging) ──────────────────────────────────
USE_WANDB=${USE_WANDB:-1}                                     # 1 to enable
WANDB_PROJECT=${WANDB_PROJECT:-vla_eval_l1loss}

# ── Misc ───────────────────────────────────────────────────────────────
DRY_RUN=${DRY_RUN:-0}

# ╚═══════════════════════════ END CONFIG ═══════════════════════════════╝

# Activate conda env if requested and not already active.
if [ -n "$CONDA_ENV" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate "$CONDA_ENV" 2>/dev/null || echo "[WARN] could not 'conda activate $CONDA_ENV'"
    fi
fi

mkdir -p "$RESULTS_DIR"

find_checkpoint() {
    # Find the latest checkpoint for an experiment.
    # An explicit CKPT_PATH env var overrides the auto-discovery entirely.
    local exp_name="$1"

    # 1) Manual override: CKPT_PATH=/abs/path/to/epoch=22-step=50000.ckpt
    if [ -n "${CKPT_PATH:-}" ]; then
        echo "$CKPT_PATH"
        return
    fi

    local ckpt_root="$VLM4VLA_ROOT/runs/${RUNS_SUBDIR}/${exp_name}/checkpoints"

    if [ ! -d "$ckpt_root" ]; then
        echo ""
        return
    fi

    # 2) Auto: pick the most recently *modified* .ckpt (by mtime, not filename).
    # Sorting by filename is wrong because e.g. "epoch=22-..." < "epoch=3-..."
    # as strings, so a plain `sort | tail -1` would pick the older checkpoint.
    local latest
    latest=$(find "$ckpt_root" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
             | sort -n | tail -1 | cut -d' ' -f2-)
    echo "$latest"
}

run_eval() {
    local exp_name="$1"
    local suite="$2"
    local config_path="$CONFIG_DIR/${exp_name}.json"

    local ckpt_path
    ckpt_path=$(find_checkpoint "$exp_name")

    if [ -z "$ckpt_path" ]; then
        echo "[SKIP] $exp_name: no checkpoint found"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  Evaluating: $exp_name on $suite"
    echo "  Checkpoint: $ckpt_path"
    echo "  GPU:        $CUDA_DEVICE     render: MUJOCO_GL=$MUJOCO_GL"
    echo "  trials:     $NUM_TRIALS      execute_step: $EXECUTE_STEP   wandb: $USE_WANDB"
    echo "  task_ids:   ${TASK_IDS:-all}"
    echo "================================================================"

    local cmd="cd $VLM4VLA_ROOT && CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u eval/libero/run_libero_eval.py \
        --ckpt_path $ckpt_path \
        --config_path $config_path \
        --execute_step $EXECUTE_STEP \
        --task_suite_name $suite \
        --num_trials_per_task $NUM_TRIALS \
        --center_crop $CENTER_CROP"

    if [ -n "$TASK_IDS" ]; then
        cmd="$cmd --task_ids $TASK_IDS"
    fi

    if [ -n "$MAX_STEPS" ]; then
        cmd="$cmd --max_steps_override $MAX_STEPS"
    fi

    if [ "$USE_WANDB" = "1" ]; then
        cmd="$cmd --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name ${exp_name}_${suite}"
    fi

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd" 2>&1 | tee "$RESULTS_DIR/${exp_name}_${suite}.log"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────
EXP_TARGET="${1:-}"
SUITE_TARGET="${2:-}"

# Resolve the list of experiments to evaluate.
EXP_LIST=()
if [ "$EXP_TARGET" = "all" ]; then
    # Every experiment that has a config.
    for config_file in "$CONFIG_DIR"/e*_*.json; do
        EXP_LIST+=("$(basename "$config_file" .json)")
    done
elif [ -n "$EXP_TARGET" ]; then
    EXP_LIST=("$EXP_TARGET")
elif [ -n "$EXPS" ]; then
    # No positional exp given → use the EXPS env list (e.g. "e0_full e1_full ...").
    read -r -a EXP_LIST <<< "$EXPS"
else
    echo "Usage: bash scripts/run_vla_ablation_eval.sh <experiment|all> [suite]"
    echo "   or: EXPS=\"e0_full e1_full ...\" bash scripts/run_vla_ablation_eval.sh \"\" [suite]"
    echo ""
    echo "Suites: ${TASK_SUITES[*]}"
    exit 1
fi

# Resolve the list of suites to evaluate (positional arg overrides config).
if [ -n "$SUITE_TARGET" ]; then
    SUITE_LIST=("$SUITE_TARGET")
else
    SUITE_LIST=("${TASK_SUITES[@]}")
fi

for exp_name in "${EXP_LIST[@]}"; do
    for suite in "${SUITE_LIST[@]}"; do
        run_eval "$exp_name" "$suite"
    done
done

echo ""
echo "Evaluation complete. Logs saved to: $RESULTS_DIR/"
