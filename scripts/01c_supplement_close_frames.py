#!/usr/bin/env python3
"""Supplement missing 'close' frames in an existing gt-q6-mv manifest.

Why this exists
---------------
`scripts/01_extract_gt.py` reaches the close frame via a simple P-controller
that sometimes fails inside 300 steps with a relatively aggressive gain — the
result is uneven coverage (e.g. libero_goal had only 13 close frames out of
30 target with 5 zero-coverage tasks). This supplement script keeps the
existing init/traj/close samples and *only adds more close frames* where
they're missing.

Strategy (more aggressive than 01_extract_gt.py for close frames):
  - target N close frames per (suite, task)               (default 5)
  - try up to M init states per task                      (default 6)
  - longer P-controller horizon                           (default 600 steps)
  - smaller gain to avoid overshoot                       (default 0.5)
  - skip init_state_idx values that already produced a    (so new frames are
    close frame in the existing manifest                   actually new)

Output
------
  - APPENDS new samples to <gt_root>/<suite>/manifest.json (and refreshes the
    per-sample JSON files), with new sample_ids continuing past the maximum
    existing id in that suite manifest.
  - Refreshes the top-level <gt_root>/manifest.json by concatenating all
    per-suite manifests and re-numbering sample_id globally.
  - Writes new PNG/JSON files alongside the existing ones; existing files
    are not touched.

Usage
-----
  python scripts/01c_supplement_close_frames.py \\
      --gt_root data/gt-q6-mv \\
      --target_close 5 \\
      --max_init_attempts 6 \\
      --close_max_steps 600 \\
      --approach_gain 0.5

  # smoke (single suite, single task)
  python scripts/01c_supplement_close_frames.py \\
      --gt_root data/gt-q6-mv \\
      --suites libero_goal --task_ids 0 \\
      --target_close 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so we can import LIBERO and src/bench
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
    get_all_object_positions,
    get_robot_base_pos,
    get_sim_from_env,
    get_views,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_root", required=True,
                   help="Root of the GT dataset (e.g. data/gt-q6-mv). Per-suite manifests live at "
                        "<gt_root>/<suite>/manifest.json.")
    p.add_argument("--suites", nargs="*", default=None,
                   help="Suites to process. Default: all suites that have a manifest.json under gt_root.")
    p.add_argument("--task_ids", nargs="*", type=int, default=None,
                   help="Restrict to specific task_ids (across all selected suites). Default: every task.")
    p.add_argument("--target_close", type=int, default=5,
                   help="Target number of close frames per (suite, task) (default 5).")
    p.add_argument("--max_init_attempts", type=int, default=6,
                   help="Max number of init states to try per task (default 6).")
    p.add_argument("--close_max_steps", type=int, default=600,
                   help="Max P-controller steps for the approach (default 600).")
    p.add_argument("--approach_gain", type=float, default=0.5,
                   help="Action gain for the approach controller (default 0.5).")
    p.add_argument("--n_wait", type=int, default=10,
                   help="Settling steps after init-state reset (default 10).")
    p.add_argument("--obs_size", type=int, default=256,
                   help="Image resolution (default 256, must match the existing dataset).")
    p.add_argument("--libero_path", default=None,
                   help="Optional LIBERO source dir to prepend to sys.path.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the gap analysis and exit without running any sim.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LIBERO bootstrap (mirrors 01_extract_gt.py)
# ---------------------------------------------------------------------------

def _import_libero(libero_path: str | None):
    if libero_path and libero_path not in sys.path:
        sys.path.insert(0, libero_path)
    try:
        from libero.libero import benchmark as _bench_mod
        from libero.libero import get_libero_path as _glp
        from libero.libero.envs import OffScreenRenderEnv as _OSR
        return _bench_mod, _glp, _OSR
    except ImportError as exc:
        print(f"ERROR: Cannot import LIBERO: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_init_states(task_suite, task_id: int, get_libero_path_fn) -> np.ndarray:
    import torch
    task = task_suite.tasks[task_id]
    if hasattr(task_suite, "get_task_init_states"):
        return task_suite.get_task_init_states(task_id)
    init_path = (
        Path(get_libero_path_fn("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_path, weights_only=False)  # nosec B614


def _make_env(task, get_libero_path_fn, OffScreenRenderEnv, obs_size: int):
    import os
    bddl_file = os.path.join(
        get_libero_path_fn("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=obs_size,
        camera_widths=obs_size,
    )


# ---------------------------------------------------------------------------
# P-controller approach (more conservative gain than 01_extract_gt.py)
# ---------------------------------------------------------------------------

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def extract_close_frame_aggressive(
    env,
    init_state,
    task_description: str,
    n_wait: int,
    max_steps: int,
    gain: float,
):
    """Like 01_extract_gt.py:extract_close_frame but with caller-controlled gain
    and step budget. Returns (gt, images_dict) on success, else None.
    """
    try:
        env.reset()
        raw_obs = env.set_init_state(init_state)

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

            direction = (target_rel - eef_rel) / (dist + 1e-8)
            action = (direction * gain).tolist() + [0.0, 0.0, 0.0, -1.0]
            result = env.step(action)
            raw_obs = result[0]
            if result[2]:  # env done
                break

        return None
    except Exception as exc:
        print(f"      WARNING: close-frame extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _existing_close_init_states(records: list[dict], task_id: int) -> set[int]:
    """init_state_idx values that already have a close frame in this suite."""
    return {
        r["init_state_idx"]
        for r in records
        if r.get("frame_type") == "close" and r["task_id"] == task_id
    }


def _close_count(records: list[dict], task_id: int) -> int:
    return sum(1 for r in records if r.get("frame_type") == "close" and r["task_id"] == task_id)


def _save_record(
    gt: dict,
    images: dict,
    sample_id: int,
    suite_name: str,
    task_id: int,
    task_desc: str,
    init_state_idx: int,
    out_dir: Path,
    manifest: list,
    data_root: Path,
) -> int:
    """Save dual-view PNGs + JSON for a NEW close-frame sample."""
    stem = f"sample_{sample_id:04d}_t{task_id:02d}_s{init_state_idx:02d}_close"

    rel: dict[str, str] = {}
    for view in ("agent", "wrist"):
        if view not in images:
            raise KeyError(f"missing view {view!r}; got {list(images.keys())}")
        png_path = out_dir / f"{stem}_{view}.png"
        Image.fromarray(images[view]).save(png_path)
        rel[view] = str(png_path.resolve().relative_to(data_root.resolve()))

    record: dict = {
        "sample_id": sample_id,
        "suite": suite_name,
        "task_id": int(task_id),
        "task_description": task_desc,
        "init_state_idx": init_state_idx,
        "frame_type": "close",
        "image_path": rel["agent"],
        "image_paths": rel,
        "gt": gt,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2))
    manifest.append(record)
    return sample_id + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    gt_root = Path(args.gt_root).resolve()
    if not gt_root.exists():
        print(f"ERROR: gt_root not found: {gt_root}", file=sys.stderr)
        sys.exit(1)

    # data_root for relative image paths in the manifest is gt_root.parent
    # (matches 01_extract_gt.py — image_path is "gt-q6-mv/<suite>/...").
    data_root = gt_root.parent

    # Discover suites with manifests.
    suite_dirs: dict[str, Path] = {}
    for child in sorted(gt_root.iterdir()):
        if (child / "manifest.json").exists():
            suite_dirs[child.name] = child
    if args.suites:
        suite_dirs = {s: suite_dirs[s] for s in args.suites if s in suite_dirs}
    if not suite_dirs:
        print(f"ERROR: no suite manifests under {gt_root}", file=sys.stderr)
        sys.exit(1)

    print("Suites to process:", list(suite_dirs.keys()))

    bench_mod, get_libero_path_fn, OffScreenRenderEnv = (None, None, None)
    if not args.dry_run:
        bench_mod, get_libero_path_fn, OffScreenRenderEnv = _import_libero(args.libero_path)

    # ── First pass: gap analysis ────────────────────────────────────────
    print("\n=== Gap analysis ===")
    print(f"Target close frames per task: {args.target_close}")
    plan: dict[str, dict[int, int]] = {}  # suite -> task_id -> needed
    for suite_name, suite_dir in suite_dirs.items():
        manifest = json.loads((suite_dir / "manifest.json").read_text())
        task_ids = sorted(set(r["task_id"] for r in manifest))
        if args.task_ids is not None:
            task_ids = [t for t in task_ids if t in args.task_ids]
        suite_plan: dict[int, int] = {}
        for tid in task_ids:
            cur = _close_count(manifest, tid)
            need = max(0, args.target_close - cur)
            if need > 0:
                suite_plan[tid] = need
        if suite_plan:
            plan[suite_name] = suite_plan
            print(f"  {suite_name}: missing {sum(suite_plan.values())} close frames across "
                  f"{len(suite_plan)} task(s) → "
                  + ", ".join(f"task_{t:02d}+{n}" for t, n in suite_plan.items()))
        else:
            print(f"  {suite_name}: already has ≥{args.target_close} close frames per task — nothing to do.")

    if not plan:
        print("\nNothing to supplement.")
        return
    if args.dry_run:
        print("\n[dry_run] exiting before running simulator.")
        return

    # ── Second pass: actually extract ───────────────────────────────────
    grand_added = 0
    for suite_name, task_plan in plan.items():
        suite_dir = suite_dirs[suite_name]
        manifest_path = suite_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        next_sample_id = max((r["sample_id"] for r in manifest), default=-1) + 1

        task_suite = bench_mod.get_benchmark_dict()[suite_name]()
        suite_added = 0

        for tid, need in task_plan.items():
            print(f"\n[{suite_name}] task {tid:02d} need {need} more close frame(s)")
            task = task_suite.get_task(tid)
            task_desc = task.language

            try:
                init_states = _load_init_states(task_suite, tid, get_libero_path_fn)
            except Exception as exc:
                print(f"  ERROR loading init states: {exc} — skipping task.")
                continue

            already = _existing_close_init_states(manifest, tid)

            # Spread attempts across init states we haven't already used for close,
            # up to --max_init_attempts. If we still need more, fall back to states
            # already used (the controller's stochasticity is low — this is the
            # last resort).
            unused_indices = [i for i in range(len(init_states)) if i not in already]
            attempt_indices_primary = list(np.linspace(
                0, len(unused_indices) - 1, min(args.max_init_attempts, len(unused_indices))
            ).astype(int)) if unused_indices else []
            attempt_indices = [unused_indices[i] for i in attempt_indices_primary]

            env = _make_env(task, get_libero_path_fn, OffScreenRenderEnv, args.obs_size)
            try:
                got = 0
                attempts_made = 0
                for state_idx in attempt_indices:
                    if got >= need:
                        break
                    attempts_made += 1
                    print(f"  trying init_state {state_idx:02d} (attempt {attempts_made}/{len(attempt_indices)})")
                    result = extract_close_frame_aggressive(
                        env,
                        init_states[state_idx],
                        task_desc,
                        n_wait=args.n_wait,
                        max_steps=args.close_max_steps,
                        gain=args.approach_gain,
                    )
                    if result is None:
                        print(f"    no close frame within {args.close_max_steps} steps")
                        continue
                    gt, images = result
                    state_out_dir = suite_dir / f"task_{tid:02d}" / f"init_state_{int(state_idx):02d}"
                    state_out_dir.mkdir(parents=True, exist_ok=True)
                    next_sample_id = _save_record(
                        gt, images, next_sample_id, suite_name, tid, task_desc,
                        int(state_idx), state_out_dir, manifest,
                        data_root=data_root,
                    )
                    got += 1
                    suite_added += 1
                    print(f"    [+] close frame saved (got {got}/{need} for this task)")
                if got == 0:
                    print(f"  ⚠ task {tid:02d}: still no close frame after {attempts_made} attempts")
                elif got < need:
                    print(f"  ⚠ task {tid:02d}: only got {got}/{need}")
            finally:
                env.close()

        # Persist updated suite manifest after each suite (resilient to crashes).
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  [{suite_name}] +{suite_added} close frames written → {manifest_path}")
        grand_added += suite_added

    # Refresh top-level manifest by re-concatenating + globally re-numbering.
    print("\n=== Refreshing top-level manifest ===")
    flat: list[dict] = []
    for suite_name, suite_dir in sorted(suite_dirs.items()):
        flat.extend(json.loads((suite_dir / "manifest.json").read_text()))
    for i, rec in enumerate(flat):
        rec["sample_id"] = i
    (gt_root / "manifest.json").write_text(json.dumps(flat, indent=2))
    print(f"  wrote {gt_root / 'manifest.json'}  ({len(flat)} samples total)")

    # Final stats
    print("\n=== Final close coverage ===")
    by_suite_task: dict[str, Counter] = defaultdict(Counter)
    for r in flat:
        if r.get("frame_type") == "close":
            by_suite_task[r["suite"]][r["task_id"]] += 1
    for suite_name in sorted(by_suite_task):
        c = by_suite_task[suite_name]
        n_tasks = len(c)
        avg = sum(c.values()) / max(1, n_tasks)
        zero = [t for t in range(max(c) + 1) if t not in c]
        print(f"  {suite_name:18s} total_close={sum(c.values()):3d}  "
              f"avg/task={avg:.1f}  zero-close tasks={zero}")
    print(f"\nGrand total close frames added: {grand_added}")


if __name__ == "__main__":
    main()
