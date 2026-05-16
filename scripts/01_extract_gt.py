#!/usr/bin/env python3
"""Extract ground truth spatial annotations from LIBERO demo trajectories.

Replays official human demonstration trajectories from LIBERO HDF5 files,
uniformly samples frames, and extracts ground truth annotations including:
  - EE position / orientation relative to robot base
  - All scene object positions relative to robot base
  - Active target object (following task-description order)
  - can_close, gripper_phase, grasp_success
  - next_direction, gripper_to_target_delta, gripper_to_target_relation
  - Pairwise object distances, gripper openness
  - Demo action at each timestep

Demo files are auto-downloaded from HuggingFace if not found locally.

Usage
-----
  # Extract 50 frames from 1 demo per task in libero_10
  
  python scripts/01_extract_gt.py \
      --suite all \
      --n_traj_frames 50 \
      --n_demos 10 \
      --out_dir data/gt-demo-libero-all-suite-train-fix

  # Extract from all suites, 3 demos each, 20 frames
  python scripts/01_extract_gt.py \
      --suite all \
      --n_traj_frames 20 \
      --n_demos 3 \
      --out_dir data/gt-demo-all

  # Custom demo directory
  python scripts/01_extract_gt.py \
      --suite libero_10 \
      --demo_dir /path/to/datasets \
      --n_traj_frames 20 \
      --out_dir data/gt-demo

Output
------
  <out_dir>/
    <suite_name>/
      task_00/
        demo_00/
          sample_0000_t00_s00_traj_000of388_agent.png
          sample_0000_t00_s00_traj_000of388_wrist.png
          sample_0000_t00_s00_traj_000of388.json
          ...
      manifest.json
    manifest.json    <- merged across all suites
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

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
    check_finger_contact,
    compute_gripper_phase,
    compute_release_windows,
    extract_eef_pose_7d,
    extract_gt_from_obs,
    get_robot_base_pos,
    get_views,
    get_sim_from_env,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial",
                   help=(
                       "LIBERO suite name(s), comma-separated or 'all'. "
                       "e.g. 'libero_spatial' | 'libero_spatial,libero_object' | 'all'. "
                       "Each suite is saved to <out_dir>/<suite_name>/. "
                       "Default: libero_spatial"
                   ))
    p.add_argument("--out_dir", default="data/gt",
                   help="Output directory for images and JSON (default: data/gt)")
    p.add_argument("--obs_size", type=int, default=256,
                   help="Image resolution (square) (default: 256)")
    p.add_argument("--task_ids", nargs="*", type=int, default=None,
                   help="Specific task IDs to run (default: all tasks)")
    p.add_argument("--n_traj_frames", type=int, default=50,
                   help="Number of frames to uniformly sample per demo (default: 50)")
    p.add_argument("--libero_path", default=None,
                   help="Optional path prepended to sys.path for LIBERO import")
    p.add_argument("--demo_dir", default="/workspace/tingting/LIBERO/libero/datasets",
                   help=(
                       "Directory containing demo HDF5 files (e.g. libero_10/*.hdf5). "
                       "If not found, auto-downloads from HuggingFace."
                   ))
    p.add_argument("--n_demos", type=int, default=50,
                   help=(
                       "Number of demonstration episodes to replay per task "
                       "(each HDF5 has ~50 demos). (default: 1)"
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
# Demo file management
# ---------------------------------------------------------------------------

_HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def _resolve_demo_dir(args, get_libero_path_fn) -> Path:
    """Determine the directory containing demo HDF5 files."""
    if args.demo_dir:
        return Path(args.demo_dir)

    try:
        cfg_path = get_libero_path_fn("datasets")
        if Path(cfg_path).exists():
            return Path(cfg_path)
    except Exception:
        pass

    fallback = _REPO_ROOT.parent / "LIBERO" / "libero" / "datasets"
    return fallback


def _ensure_demo_file(demo_dir: Path, suite_name: str, task_name: str) -> Path:
    """Return path to demo HDF5, downloading from HuggingFace if missing."""
    demo_path = demo_dir / suite_name / f"{task_name}_demo.hdf5"
    if demo_path.exists():
        return demo_path

    print(f"    Demo file not found: {demo_path}")
    print(f"    Downloading from HuggingFace ({_HF_REPO_ID})...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest_dir = demo_dir / suite_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        filename=f"{suite_name}/{task_name}_demo.hdf5",
        local_dir=str(demo_dir),
    )

    if not demo_path.exists():
        raise FileNotFoundError(
            f"Download succeeded but file not at expected path: {demo_path}"
        )
    print(f"    Downloaded: {demo_path}")
    return demo_path


# ---------------------------------------------------------------------------
# Demo trajectory extraction
# ---------------------------------------------------------------------------

def extract_demo_traj_frames(
    env,
    demo_file: Path,
    demo_idx: int,
    task_description: str,
    n_frames: int,
) -> list[tuple[dict, dict, int, int]]:
    """Replay an official demo trajectory and uniformly sample n_frames.

    Uses env.set_init_state(state_vector) to jump to each sampled timestep,
    avoiding the need to step through every action sequentially.

    Returns a list of (gt, images, traj_step, traj_len) tuples.
    """
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_idx}"
        if demo_key not in f:
            print(f"    WARNING: {demo_key} not found in {demo_file}")
            return []
        states = f[f"{demo_key}/states"][()]   # (T, state_dim)
        actions = f[f"{demo_key}/actions"][()]  # (T, 7)

    traj_len = len(states)
    if traj_len == 0:
        return []

    # Pre-compute release windows from full action sequence
    release_windows = compute_release_windows(actions)

    # ── Pre-compute keyframe poses (initial, grasp, release) ─────────
    # Find grasp transitions (-1→+1) and release transitions (+1→-1)
    gripper_cmds = actions[:, 6]
    grasp_steps = []   # steps where action[6] goes -1 → +1
    release_steps = [] # steps where action[6] goes +1 → -1
    for t in range(1, traj_len):
        if gripper_cmds[t - 1] < 0 and gripper_cmds[t] > 0:
            grasp_steps.append(t)
        elif gripper_cmds[t - 1] > 0 and gripper_cmds[t] < 0:
            release_steps.append(t)

    # Extract 7D poses at keyframes by restoring sim state
    env.reset()
    sim = get_sim_from_env(env)

    def _pose_at_step(step: int) -> dict:
        obs = env.set_init_state(states[step])
        s = get_sim_from_env(env)
        base = get_robot_base_pos(s)
        return extract_eef_pose_7d(obs, s, base)

    initial_eef_pose = _pose_at_step(0)

    grasp_poses = []
    for t in grasp_steps:
        grasp_poses.append({"step": t, "pose": _pose_at_step(t)})

    release_poses = []
    for t in release_steps:
        release_poses.append({"step": t, "pose": _pose_at_step(t)})

    # Build ordered list of keyframe targets for next_target_pose lookup:
    # grasp_0, release_0, grasp_1, release_1, ...
    keyframes = []
    gi, ri = 0, 0
    while gi < len(grasp_steps) or ri < len(release_steps):
        if gi < len(grasp_steps) and (ri >= len(release_steps) or grasp_steps[gi] <= release_steps[ri]):
            keyframes.append(("grasp", grasp_steps[gi], grasp_poses[gi]["pose"]))
            gi += 1
        else:
            keyframes.append(("release", release_steps[ri], release_poses[ri]["pose"]))
            ri += 1

    def _next_target_pose(step: int) -> dict | None:
        """Find the next keyframe pose after the given step."""
        for kf_type, kf_step, kf_pose in keyframes:
            if kf_step > step:
                return {"type": kf_type, "step": kf_step, "pose": kf_pose}
        return None  # past all keyframes (task finishing)

    # ── Uniform sampling ─────────────────────────────────────────────
    n_sample = min(n_frames, traj_len)
    sample_indices = np.linspace(0, traj_len - 1, n_sample, dtype=int)

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_indices = []
    for idx in sample_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(int(idx))

    sampled = []
    try:
        for idx in unique_indices:
            # Restore full simulator state at this timestep
            raw_obs = env.set_init_state(states[idx])
            sim = get_sim_from_env(env)

            gt = extract_gt_from_obs(raw_obs, sim, task_description, env=env)
            images = get_views(raw_obs)

            # ── Demo-specific fields ─────────────────────────────────
            if idx < len(actions):
                gt["demo_action"] = actions[idx].tolist()

            # Check physical finger contact from sim
            is_grasping, grasped_geom = check_finger_contact(sim)
            gt["finger_contact"] = is_grasping
            gt["grasped_object"] = grasped_geom

            # can_close: directly from demo action command
            action = actions[idx] if idx < len(actions) else np.zeros(7)
            gt["can_close"] = bool(action[6] > 0)

            # gripper_phase from action + contact
            if idx in release_windows:
                phase = "releasing"
            elif action[6] < 0:
                phase = "approaching"
            elif is_grasping:
                phase = "carrying"
            else:
                phase = "grasping"
            gt["gripper_phase"] = phase

            # Keyframe poses
            gt["initial_eef_pose"] = initial_eef_pose
            gt["grasp_poses"] = grasp_poses
            gt["release_poses"] = release_poses
            gt["next_target_pose"] = _next_target_pose(idx)

            sampled.append((gt, images, idx, traj_len))

    except Exception as exc:
        print(f"    WARNING: Demo replay failed at step {idx}: {exc}")
        import traceback; traceback.print_exc()

    return sampled


# ---------------------------------------------------------------------------
# Record saving helper
# ---------------------------------------------------------------------------

def _save_record(
    gt: dict,
    images: dict,             # {"agent": ndarray, "wrist": ndarray}
    sample_id: int,
    suite_name: str,
    task_id: int,
    task_desc: str,
    demo_idx: int,
    out_dir: "Path",          # directory where the file will be written
    manifest: list,
    data_root: "Path" = None, # root for relative image_path (defaults to out_dir.parent)
    traj_step: int | None = None,
    traj_len:  int | None = None,
) -> int:
    """Save dual-view PNGs + JSON for one sample and append to manifest.

    Returns next sample_id.
    """
    if traj_step is not None and traj_len is not None:
        type_suffix = f"traj_{traj_step:03d}of{traj_len:03d}"
    else:
        type_suffix = "traj"
    stem = f"sample_{sample_id:04d}_t{task_id:02d}_s{demo_idx:02d}_{type_suffix}"

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
        "demo_idx": demo_idx,
        "frame_type": "traj",
        "image_path": rel_paths["agent"],
        "image_paths": rel_paths,
        "gt": gt,
    }
    if traj_step is not None:
        record["traj_step"] = traj_step
        record["traj_len"]  = traj_len
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2))
    manifest.append(record)

    tag = f"[TRAJ {traj_step:03d}/{traj_len:03d}]" if traj_step is not None else ""
    phase = gt.get("gripper_phase", "")
    print(
        f"    {tag} sample {sample_id:04d} | demo {demo_idx:02d} | "
        f"EE=[{gt['eef_pos'][0]:.3f},{gt['eef_pos'][1]:.3f},{gt['eef_pos'][2]:.3f}] | "
        f"target={gt['target_object_name']} "
        f"dist={gt['distance_to_target']:.3f}m "
        f"phase={phase} can_close={gt['can_close']}"
    )
    return sample_id + 1


# ---------------------------------------------------------------------------
# Suite extraction
# ---------------------------------------------------------------------------

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


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

    if data_root is None:
        data_root = suite_out_dir.parent

    total_tasks = len(task_suite.tasks)
    task_ids = args.task_ids if args.task_ids is not None else list(range(total_tasks))

    print(
        f"\n{'='*60}\n"
        f"Suite: {suite_name}  |  Tasks: {len(task_ids)}/{total_tasks}  "
        f"|  Demos/task: {args.n_demos}  |  Frames/demo: {args.n_traj_frames}  "
        f"|  Output: {suite_out_dir}"
    )

    manifest: list[dict] = []
    sample_id = global_sample_id
    demo_dir = _resolve_demo_dir(args, get_libero_path_fn)

    for tid in task_ids:
        task = task_suite.get_task(tid)
        task_desc = task.language
        print(f"\n[Task {tid:02d}] {task_desc}")

        env = _make_env(task, get_libero_path_fn, OffScreenRenderEnv, args.obs_size)

        # Locate demo file (auto-download if missing)
        demo_file = _ensure_demo_file(demo_dir, suite_name, task.name)
        with h5py.File(demo_file, "r") as f:
            total_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
        n_demos = min(args.n_demos, total_demos)
        print(f"    Replaying {n_demos} demo(s) from {demo_file.name} "
              f"({total_demos} available), sampling {args.n_traj_frames} frames each")

        task_out_dir = suite_out_dir / f"task_{tid:02d}"

        for demo_idx in tqdm(range(n_demos), desc=f"  Task {tid} (demo)", leave=False):
            traj_samples = extract_demo_traj_frames(
                env, demo_file, demo_idx, task_desc, args.n_traj_frames,
            )
            if not traj_samples:
                print(f"    WARNING: No demo frames for task {tid} demo {demo_idx}")
                continue
            state_out_dir = task_out_dir / f"demo_{demo_idx:02d}"
            state_out_dir.mkdir(parents=True, exist_ok=True)
            for gt, images, t_step, t_len in traj_samples:
                sample_id = _save_record(
                    gt, images, sample_id, suite_name, tid, task_desc,
                    demo_idx, state_out_dir, manifest,
                    data_root=data_root, traj_step=t_step, traj_len=t_len,
                )

        env.close()

    # Per-suite manifest
    suite_manifest = suite_out_dir / "manifest.json"
    suite_manifest.write_text(json.dumps(manifest, indent=2))

    n_can_close = sum(1 for r in manifest if r["gt"]["can_close"])
    print(
        f"\n  [{suite_name}] {len(manifest)} samples saved  |  "
        f"can_close=True: {n_can_close}/{len(manifest)} "
        f"({100*n_can_close/max(1,len(manifest)):.0f}%)  |  "
        f"manifest: {suite_manifest}"
    )
    return manifest, sample_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    bench_mod, get_libero_path_fn, OffScreenRenderEnv = _import_libero(args.libero_path)

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

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
        suite_dir = base_out_dir / suite_name
        records, global_id = _extract_suite(
            suite_name, task_suite, bench_mod, get_libero_path_fn,
            OffScreenRenderEnv, suite_dir, args, global_id,
            data_root=data_root,
        )
        all_records.extend(records)

    # Write merged top-level manifest
    top_manifest = base_out_dir / "manifest.json"
    top_manifest.write_text(json.dumps(all_records, indent=2))
    print(f"\nTop-level manifest: {len(all_records)} samples → {top_manifest}")

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
