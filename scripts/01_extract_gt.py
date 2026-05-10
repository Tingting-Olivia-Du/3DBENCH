#!/usr/bin/env python3
"""Extract ground truth spatial annotations from LIBERO simulations.

For each task in a LIBERO suite (default: libero_spatial), this script:
  1. Loads N initial states.
  2. Resets the sim to each initial state and lets physics settle.
  3. Captures the agent-view RGB image.
  4. Extracts ground truth:
       - EE position relative to robot base
       - All scene object positions relative to robot base
       - Best-guess target object (via keyword matching to task description)
       - can_close label            (dist(EE, target) < 0.04 m)
       - next_direction             (unit vector EE → target)
       - gripper_to_target_delta    (target_pos - eef_pos, [Δx, Δy, Δz] in meters)  ← Q5 GT
       - gripper_to_target_relation (axis-wise label dict, e.g.                      ← Q6 GT
                                     {"x": "in_front", "y": "left", "z": "below"})
  5. Saves images as PNG and GT records as JSON.

NOTE: If you previously extracted GT without Q5/Q6 fields, re-run this script
to regenerate the manifest so that 03_compute_metrics.py can evaluate Q5/Q6.

Usage
-----
  # Default: libero_spatial, 5 init states per task
  python scripts/01_extract_gt.py

  # Custom options
  python scripts/01_extract_gt.py \\
      --suite libero_goal \\
      --n_states 10 \\
      --out_dir data/gt-q6-test \\
      --task_ids 1 \\
      --libero_path /path/to/LIBERO

  # Build a TEST set using init states the training set never saw.
  # Training used --n_states 5 → indices [0,12,24,36,49].
  # Pass --exclude_n_states 5 to skip those and sample from the remaining 45.
  python scripts/01_extract_gt.py \\
      --suite all \\
      --n_states 10 \\
      --exclude_n_states 5 \\
      --n_traj_frames 8 \\
      --n_close 3 \\
      --out_dir data/gt-q6-mv-test

  # Or use explicit indices:
  python scripts/01_extract_gt.py \\
      --suite libero_spatial \\
      --state_indices 3 7 15 20 28 33 42 \\
      --out_dir data/gt-q6-mv-test
 

# 激活环境
source /umd-datapool/tingting/envs/vlmbench/bin/activate
# 或者
/umd-datapool/tingting/envs/vlmbench/bin/python 对应 python

# 直接跑（libero 已经装好，不需要 --libero_path）

bash run_benchmark.sh \
    --n_states 5 \
    --n_traj_frames 8 \
    --n_close 3 \
    --gt_dir data/gt-q6 \
    --device cuda:0 \
    --models "qwen2.5-vl-7b random"



Output
------
  data/gt/
    sample_0000.png
    sample_0000.json   <- contains task info + gt dict
    ...
    manifest.json      <- list of all samples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: make sure lerobot and src/bench are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from bench.gt_extractor import (
    CLOSE_THRESHOLD,
    extract_gt_from_obs,
    find_target_object,
    get_views,
    get_all_object_positions,
    get_robot_base_pos,
    get_sim_from_env,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite",     default="libero_spatial",
                   help=(
                       "LIBERO suite name(s), comma-separated or 'all'. "
                       "e.g. 'libero_spatial' | 'libero_spatial,libero_object' | 'all'. "
                       "Each suite is saved to <out_dir>/<suite_name>/. "
                       "Default: libero_spatial"
                   ))
    p.add_argument("--n_states",  type=int, default=5,
                   help="Number of initial states to sample per task (default: 5)")
    p.add_argument("--out_dir",   default="data/gt",
                   help="Output directory for images and JSON (default: data/gt)")
    p.add_argument("--n_wait",    type=int, default=10,
                   help="Settling steps after init-state reset (default: 10)")
    p.add_argument("--obs_size",  type=int, default=256,
                   help="Image resolution (square) (default: 256)")
    p.add_argument("--task_ids",  nargs="*", type=int, default=None,
                   help="Specific task IDs to run (default: all tasks)")
    p.add_argument("--n_close",   type=int, default=3,
                   help=(
                       "Per task: number of EXTRA frames to capture where can_close=True, "
                       "obtained by steering EE toward the target with a simple P-controller. "
                       "Set to 0 to disable. (default: 3)"
                   ))
    p.add_argument("--close_max_steps", type=int, default=300,
                   help="Max sim steps for the approach phase (default: 300)")
    p.add_argument("--n_traj_frames", type=int, default=0,
                   help=(
                       "Per init-state: run a P-controller approach and uniformly sample "
                       "this many frames along the trajectory (start → near-grasp), "
                       "like extracting keyframes from a task video. "
                       "Includes the initial frame and the final near-grasp frame when possible. "
                       "Set to 0 to disable (default: 0)."
                   ))
    p.add_argument("--libero_path", default=None,
                   help="Optional path prepended to sys.path for LIBERO import")

    # ── Test-set construction: exclude training init states ──────────────
    p.add_argument("--exclude_n_states", type=int, default=0,
                   help=(
                       "Exclude the N init-state indices that *would* have been selected "
                       "by --n_states=N on the full pool (i.e. np.linspace(0, 49, N)).  "
                       "Use this to build a test set from states the training set never saw.  "
                       "E.g. if training used --n_states 5, pass --exclude_n_states 5 here "
                       "to skip indices [0,12,24,36,49].  (default: 0 = no exclusion)"
                   ))
    p.add_argument("--state_indices", type=int, nargs="*", default=None,
                   help=(
                       "Explicit list of init-state indices to use (0-based).  "
                       "Overrides --n_states.  E.g. --state_indices 3 7 15 28 42"
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# LIBERO bootstrap
# ---------------------------------------------------------------------------

def _import_libero(libero_path: str | None):
    """Import LIBERO modules, optionally prepending a custom path."""
    if libero_path and libero_path not in sys.path:
        sys.path.insert(0, libero_path)
    try:
        from libero.libero import benchmark as _bench_mod
        from libero.libero import get_libero_path as _glp
        from libero.libero.envs import OffScreenRenderEnv as _OSR
        return _bench_mod, _glp, _OSR
    except ImportError as exc:
        print(
            f"ERROR: Cannot import LIBERO: {exc}\n"
            "Install it with:  pip install hf-libero\n"
            "Or point to your LIBERO checkout with --libero_path /path/to/LIBERO"
        )
        sys.exit(1)


def _load_init_states(task_suite, task_id: int, get_libero_path_fn) -> np.ndarray:
    """Load the pre-recorded init-state tensor for a given task."""
    import torch
    task = task_suite.tasks[task_id]

    # Prefer suite-level helper if available
    if hasattr(task_suite, "get_task_init_states"):
        return task_suite.get_task_init_states(task_id)

    # Fall back to LeRobot-style path construction
    init_path = (
        Path(get_libero_path_fn("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_path, weights_only=False)  # nosec B614


def _make_env(task, get_libero_path_fn, OffScreenRenderEnv, obs_size: int):
    """Create an OffScreenRenderEnv for a single LIBERO task."""
    import os
    bddl_file = os.path.join(
        get_libero_path_fn("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=obs_size,
        camera_widths=obs_size,
    )
    return env


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # no-op + gripper open

# Action scale: LIBERO/robosuite delta-position actions are clipped to [-1,1]
# and scaled internally (~0.05 m/step). We use a fraction of max speed so the
# robot approaches smoothly and doesn't overshoot.
_APPROACH_GAIN = 0.8


def extract_frame(
    env,
    init_state,
    task_description: str,
    n_wait: int,
) -> tuple[dict, dict] | None:
    """Reset env to init_state, settle, and extract GT + dual-view images.

    Returns (gt_dict, images_dict) where images_dict has keys 'agent' and 'wrist'.
    """
    try:
        env.reset()
        raw_obs = env.set_init_state(init_state)

        # Let physics settle
        for _ in range(n_wait):
            result = env.step(DUMMY_ACTION)
            raw_obs = result[0]
            done = result[2]
            if done:
                break

        sim = get_sim_from_env(env)
        gt = extract_gt_from_obs(raw_obs, sim, task_description)
        images = get_views(raw_obs)
        return gt, images

    except Exception as exc:
        print(f"    WARNING: Frame extraction failed: {exc}")
        return None


def extract_close_frame(
    env,
    init_state,
    task_description: str,
    n_wait: int,
    max_steps: int,
) -> tuple[dict, np.ndarray] | None:
    """Steer the EE toward the target with a proportional controller.

    Runs until the gripper is within CLOSE_THRESHOLD of the target or
    max_steps is reached, then captures a frame.  Returns (gt, image) where
    gt['can_close'] is True, or None if the target was not reached.

    The approach uses world-frame delta-position actions [dx,dy,dz,0,0,0,-1].
    LIBERO/robosuite clips actions to [-1,1] and scales position deltas by
    ~0.05 m/step, so we don't need to normalise precisely — just point the
    direction vector toward the target at _APPROACH_GAIN magnitude.
    """
    try:
        env.reset()
        raw_obs = env.set_init_state(init_state)

        # Settle
        for _ in range(n_wait):
            raw_obs = env.step(DUMMY_ACTION)[0]

        sim = get_sim_from_env(env)

        for _ in range(max_steps):
            base_pos = get_robot_base_pos(sim)
            eef_world = np.array(raw_obs["robot0_eef_pos"])
            eef_rel = eef_world - base_pos

            all_objects = get_all_object_positions(sim, base_pos)
            _, target_rel = find_target_object(all_objects, task_description)

            dist = float(np.linalg.norm(eef_rel - target_rel))

            if dist < CLOSE_THRESHOLD:
                gt = extract_gt_from_obs(raw_obs, sim, task_description)
                images = get_views(raw_obs)
                return gt, images

            # Delta action toward target (world frame, normalised)
            direction = (target_rel - eef_rel) / (dist + 1e-8)
            action = (direction * _APPROACH_GAIN).tolist() + [0.0, 0.0, 0.0, -1.0]
            result = env.step(action)
            raw_obs = result[0]
            if result[2]:  # done
                break

        return None

    except Exception as exc:
        print(f"    WARNING: Close-frame extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Record saving helper
# ---------------------------------------------------------------------------

def extract_traj_frames(
    env,
    init_state,
    task_description: str,
    n_wait: int,
    max_steps: int,
    n_frames: int,
) -> list[tuple[dict, dict, int, int]]:
    """Run P-controller from init_state → target and uniformly sample n_frames.

    Returns a list of (gt, images, traj_step, traj_len) tuples where `images`
    is a dict {'agent': ndarray, 'wrist': ndarray}, sampled at evenly-spaced
    steps across the full approach trajectory.  The list always includes the
    first frame (traj_step=0) and, when the target is reached, the final
    near-grasp frame.

    If the robot never reaches the target within max_steps the full run up to
    that point is still sampled — useful for long-range tasks.
    """
    if n_frames <= 0:
        return []

    try:
        env.reset()
        raw_obs = env.set_init_state(init_state)

        # Settle
        for _ in range(n_wait):
            raw_obs = env.step(DUMMY_ACTION)[0]

        sim = get_sim_from_env(env)

        # ── Collect every frame of the approach run ───────────────────────
        frames: list[tuple[dict, dict]] = []

        for _ in range(max_steps):
            base_pos = get_robot_base_pos(sim)
            eef_world = np.array(raw_obs["robot0_eef_pos"])
            eef_rel   = eef_world - base_pos
            all_objects = get_all_object_positions(sim, base_pos)
            _, target_rel = find_target_object(all_objects, task_description)

            gt     = extract_gt_from_obs(raw_obs, sim, task_description)
            images = get_views(raw_obs)
            frames.append((gt, images))

            if gt["can_close"]:
                break  # reached target — stop collecting

            dist = float(np.linalg.norm(eef_rel - target_rel))
            direction = (target_rel - eef_rel) / (dist + 1e-8)
            action = (direction * _APPROACH_GAIN).tolist() + [0.0, 0.0, 0.0, -1.0]
            result = env.step(action)
            raw_obs = result[0]
            if result[2]:  # env done
                break

        traj_len = len(frames)
        if traj_len == 0:
            return []

        # ── Uniform sampling across the collected trajectory ──────────────
        # Always include the first frame (t=0) and the last (near-grasp or
        # timeout).  linspace handles edge case n_frames=1 gracefully.
        indices = np.linspace(0, traj_len - 1, min(n_frames, traj_len), dtype=int)
        # Deduplicate while preserving order
        seen: set[int] = set()
        sampled = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                gt, images = frames[idx]
                sampled.append((gt, images, int(idx), traj_len))

        return sampled

    except Exception as exc:
        print(f"    WARNING: Trajectory extraction failed: {exc}")
        return []


def _save_record(
    gt: dict,
    images: dict,             # {"agent": ndarray, "wrist": ndarray}
    sample_id: int,
    suite_name: str,
    task_id: int,
    task_desc: str,
    init_state_idx: int,
    frame_type: str,          # "init" | "close" | "traj"
    out_dir: "Path",          # directory where the file will be written
    manifest: list,
    data_root: "Path" = None, # root for relative image_path (defaults to out_dir.parent)
    traj_step: int | None = None,
    traj_len:  int | None = None,
) -> int:
    """Save dual-view PNGs + JSON for one sample and append to manifest.

    Writes two PNGs per sample with `_agent` / `_wrist` suffixes:
      sample_0000_t00_s00_init_agent.png
      sample_0000_t00_s00_init_wrist.png
    The manifest record carries:
      - image_path        : legacy field, points to the AGENT view (back-compat)
      - image_paths       : {"agent": "...", "wrist": "..."}  (new, preferred)
    Returns next sample_id.
    """
    if frame_type == "traj" and traj_step is not None and traj_len is not None:
        type_suffix = f"traj_{traj_step:03d}of{traj_len:03d}"
    else:
        type_suffix = frame_type  # "init" or "close"
    stem = f"sample_{sample_id:04d}_t{task_id:02d}_s{init_state_idx:02d}_{type_suffix}"

    _data_root = data_root if data_root is not None else out_dir.parent
    rel_paths: dict = {}
    for view_name in ("agent", "wrist"):
        if view_name not in images:
            raise KeyError(
                f"_save_record: missing view '{view_name}' in images dict; got keys {list(images.keys())}"
            )
        png_path = out_dir / f"{stem}_{view_name}.png"
        Image.fromarray(images[view_name]).save(png_path)
        rel_paths[view_name] = str(png_path.resolve().relative_to(_data_root.resolve()))

    record: dict = {
        "sample_id": sample_id,
        "suite": suite_name,
        "task_id": int(task_id),
        "task_description": task_desc,
        "init_state_idx": init_state_idx,
        "frame_type": frame_type,
        # legacy: agent view, kept so older single-image consumers keep working
        "image_path": rel_paths["agent"],
        # new: dual-view dict
        "image_paths": rel_paths,
        "gt": gt,
    }
    if traj_step is not None:
        record["traj_step"] = traj_step
        record["traj_len"]  = traj_len
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2))
    manifest.append(record)

    if frame_type == "traj":
        tag = f"[TRAJ {traj_step:03d}/{traj_len:03d}]"
    elif frame_type == "close":
        tag = "[CLOSE]        "
    else:
        tag = "               "
    print(
        f"    {tag} sample {sample_id:04d} | state {init_state_idx:02d} | "
        f"EE=[{gt['eef_pos'][0]:.3f},{gt['eef_pos'][1]:.3f},{gt['eef_pos'][2]:.3f}] | "
        f"target={gt['target_object_name']} "
        f"dist={gt['distance_to_target']:.3f}m "
        f"can_close={gt['can_close']}"
    )
    return sample_id + 1


ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _resolve_state_indices(
    n_total: int, args: argparse.Namespace
) -> np.ndarray:
    """Decide which init-state indices to use for a task.

    Priority:
      1. --state_indices  (explicit list)
      2. --n_states + --exclude_n_states  (uniform sample minus training set)
      3. --n_states  (original behaviour)
    """
    if args.state_indices is not None:
        idx = np.array(sorted(set(args.state_indices)), dtype=int)
        idx = idx[(idx >= 0) & (idx < n_total)]
        if len(idx) == 0:
            raise ValueError(
                f"--state_indices produced no valid indices for pool size {n_total}"
            )
        return idx

    # Indices that the *training* run would have picked
    exclude_set: set[int] = set()
    if args.exclude_n_states > 0:
        train_idx = np.linspace(0, n_total - 1, args.exclude_n_states, dtype=int)
        exclude_set = set(train_idx.tolist())

    # Candidate pool: uniform sample of n_states from [0, n_total)
    n_want = min(args.n_states, n_total)
    if exclude_set:
        # Sample from the remaining indices
        remaining = np.array([i for i in range(n_total) if i not in exclude_set])
        if len(remaining) == 0:
            raise ValueError(
                f"--exclude_n_states={args.exclude_n_states} excludes ALL "
                f"{n_total} states — nothing left to sample"
            )
        n_want = min(n_want, len(remaining))
        idx = remaining[np.linspace(0, len(remaining) - 1, n_want, dtype=int)]
    else:
        idx = np.linspace(0, n_total - 1, n_want, dtype=int)

    return idx


def _resolve_suites(suite_arg: str, bench_dict: dict) -> list[str]:
    """Expand 'all' or comma-separated suite names into a validated list."""
    if suite_arg.strip().lower() == "all":
        names = ALL_SUITES
    else:
        names = [s.strip() for s in suite_arg.split(",") if s.strip()]

    unknown = [n for n in names if n not in bench_dict]
    if unknown:
        print(f"ERROR: Unknown suite(s): {unknown}. Available: {sorted(bench_dict.keys())}")
        sys.exit(1)
    return names


def _extract_suite(
    suite_name: str,
    task_suite,
    bench_mod,
    get_libero_path_fn,
    OffScreenRenderEnv,
    suite_out_dir: Path,
    args,
    global_sample_id: int,
    data_root: Path = None,
) -> tuple[list[dict], int]:
    """Extract GT for one suite into suite_out_dir. Returns (manifest_records, next_sample_id)."""
    suite_out_dir.mkdir(parents=True, exist_ok=True)

    # data_root is the common ancestor for relative image paths (e.g. "data/").
    # It is passed explicitly from main() to handle both single and multi-suite
    # layouts correctly.  Falls back to suite_out_dir.parent for safety.
    if data_root is None:
        data_root = suite_out_dir.parent

    total_tasks = len(task_suite.tasks)
    task_ids = args.task_ids if args.task_ids is not None else list(range(total_tasks))

    print(
        f"\n{'='*60}\n"
        f"Suite: {suite_name}  |  Tasks: {len(task_ids)}/{total_tasks}  "
        f"|  Init states/task: {args.n_states}  |  Output: {suite_out_dir}"
    )

    manifest: list[dict] = []
    sample_id = global_sample_id

    for tid in task_ids:
        task = task_suite.get_task(tid)
        task_desc = task.language
        print(f"\n[Task {tid:02d}] {task_desc}")

        try:
            init_states = _load_init_states(task_suite, tid, get_libero_path_fn)
        except Exception as exc:
            print(f"  ERROR loading init states: {exc} — skipping task.")
            continue

        # Folder layout:
        #   <suite_out_dir>/task_00/init_state_00/sample_0000.png
        task_out_dir = suite_out_dir / f"task_{tid:02d}"

        state_indices = _resolve_state_indices(len(init_states), args)
        if args.exclude_n_states > 0:
            print(f"    (excluded {args.exclude_n_states} training indices, "
                  f"using {len(state_indices)}: {state_indices.tolist()})")

        env = _make_env(task, get_libero_path_fn, OffScreenRenderEnv, args.obs_size)

        # ── Normal init-state frames ──────────────────────────────────────────
        for state_idx in tqdm(state_indices, desc=f"  Task {tid} (init)", leave=False):
            result = extract_frame(env, init_states[state_idx], task_desc, args.n_wait)
            if result is None:
                continue

            gt, images = result
            state_out_dir = task_out_dir / f"init_state_{int(state_idx):02d}"
            state_out_dir.mkdir(parents=True, exist_ok=True)
            sample_id = _save_record(
                gt, images, sample_id, suite_name, tid, task_desc,
                int(state_idx), "init", state_out_dir, manifest,
                data_root=data_root,
            )

        # ── Trajectory frames (start → near-grasp, uniformly sampled) ────────
        n_traj_frames = getattr(args, "n_traj_frames", 0)
        if n_traj_frames > 0:
            for state_idx in tqdm(state_indices, desc=f"  Task {tid} (traj)", leave=False):
                traj_samples = extract_traj_frames(
                    env, init_states[state_idx], task_desc,
                    args.n_wait, args.close_max_steps, n_traj_frames,
                )
                if not traj_samples:
                    print(f"    WARNING: No traj frames for task {tid} state {state_idx}")
                    continue
                state_out_dir = task_out_dir / f"init_state_{int(state_idx):02d}"
                state_out_dir.mkdir(parents=True, exist_ok=True)
                for gt, images, t_step, t_len in traj_samples:
                    sample_id = _save_record(
                        gt, images, sample_id, suite_name, tid, task_desc,
                        int(state_idx), "traj", state_out_dir, manifest,
                        data_root=data_root, traj_step=t_step, traj_len=t_len,
                    )

        # ── Extra can_close=True frames (P-controller approach) ───────────────
        n_close = getattr(args, "n_close", 0)
        if n_close > 0:
            # Sample close-frame init states from the same allowed pool
            close_indices = state_indices[
                np.linspace(0, len(state_indices) - 1, min(n_close, len(state_indices)), dtype=int)
            ]
            n_found = 0
            for state_idx in tqdm(close_indices, desc=f"  Task {tid} (close)", leave=False):
                result = extract_close_frame(
                    env, init_states[state_idx], task_desc,
                    args.n_wait, args.close_max_steps,
                )
                if result is None:
                    continue
                gt, images = result
                state_out_dir = task_out_dir / f"init_state_{int(state_idx):02d}"
                state_out_dir.mkdir(parents=True, exist_ok=True)
                sample_id = _save_record(
                    gt, images, sample_id, suite_name, tid, task_desc,
                    int(state_idx), "close", state_out_dir, manifest,
                    data_root=data_root,
                )
                n_found += 1
            if n_found == 0:
                print(f"    WARNING: Could not reach can_close=True for task {tid} "
                      f"within {args.close_max_steps} steps — try --close_max_steps larger.")

        env.close()

    # Per-suite manifest
    suite_manifest = suite_out_dir / "manifest.json"
    suite_manifest.write_text(json.dumps(manifest, indent=2))

    n_close_found = sum(1 for r in manifest if r["gt"]["can_close"])
    n_close_frames = sum(1 for r in manifest if r.get("frame_type") == "close")
    print(
        f"\n  [{suite_name}] {len(manifest)} samples saved  |  "
        f"can_close=True: {n_close_found}/{len(manifest)} ({100*n_close_found/max(1,len(manifest)):.0f}%)  "
        f"[{n_close_frames} from close-approach]  |  "
        f"manifest: {suite_manifest}"
    )
    return manifest, sample_id


def main() -> None:
    args = parse_args()
    bench_mod, get_libero_path_fn, OffScreenRenderEnv = _import_libero(args.libero_path)

    base_out_dir = Path(args.out_dir)   # e.g. data/gt
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # data_root is the "data/" directory — image_path in manifests is relative to it.
    # base_out_dir is typically  <cwd>/data/gt
    # so data_root = <cwd>/data/gt/..  = <cwd>/data/
    data_root = base_out_dir.resolve().parent

    bench_dict = bench_mod.get_benchmark_dict()
    suite_names = _resolve_suites(args.suite, bench_dict)
    multi = len(suite_names) > 1

    print(f"Suites to extract: {suite_names}")
    if multi:
        print(f"Each suite → {base_out_dir}/<suite_name>/")

    global_id = 0
    all_records: list[dict] = []

    for suite_name in suite_names:
        task_suite = bench_dict[suite_name]()
        # Each suite gets its own subdirectory and manifest:
        #   <out_dir>/<suite_name>/manifest.json
        suite_dir = base_out_dir / suite_name
        records, global_id = _extract_suite(
            suite_name, task_suite, bench_mod, get_libero_path_fn,
            OffScreenRenderEnv, suite_dir, args, global_id,
            data_root=data_root,
        )
        all_records.extend(records)

    # Write merged top-level manifest covering all suites
    top_manifest = base_out_dir / "manifest.json"
    top_manifest.write_text(json.dumps(all_records, indent=2))
    print(f"\nTop-level manifest: {len(all_records)} samples → {top_manifest}")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"Grand total: {global_id} samples across {len(suite_names)} suite(s)")
    for suite_name in suite_names:
        mf = base_out_dir / suite_name / "manifest.json"
        if mf.exists():
            import json as _json
            cnt = len(_json.loads(mf.read_text()))
            print(f"  {suite_name}: {cnt} samples  →  {mf}")


if __name__ == "__main__":
    main()
