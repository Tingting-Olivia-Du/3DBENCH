#!/usr/bin/env python3
"""Extract dual-view spatial QA ground truth from the CALVIN dataset.

Mirrors the LIBERO pipeline (scripts/01_extract_gt.py) so that the resulting
manifest is consumable by scripts/02_run_vlm_eval.py and 03_compute_metrics.py
without further modification.

Pipeline
--------
1. Parse <data_root>/lang_annotations/auto_lang_ann.npy into a list of
   (start, end, task_class, language) windows.
2. For each window, sample frames mirroring LIBERO (`init`, `traj`, `close`).
3. Load the corresponding episode_*.npz, write rgb_static and rgb_gripper as
   PNGs (named with `_agent` and `_wrist` suffixes for cross-pipeline
   compatibility), compute the 6 GT fields from robot_obs / scene_obs.
4. Save per-sample JSON and a `manifest.json` per suite (`calvin_D`).

Output layout (matches the LIBERO `gt-q6-mv` schema):

    <out_dir>/
      manifest.json                           <-- top-level (all samples flat)
      calvin_D/
        manifest.json
        task_<NN>/
          window_<MMMM>/
            sample_<gid>_t<NN>_w<MMMM>_init_agent.png
            sample_<gid>_t<NN>_w<MMMM>_init_wrist.png
            sample_<gid>_t<NN>_w<MMMM>_init.json
            sample_<gid>_t<NN>_w<MMMM>_traj_<step>of<len>_*.png
            sample_<gid>_t<NN>_w<MMMM>_close_*.png

Usage
-----
  python scripts/01b_extract_gt_calvin.py \\
      --data_root /umd-datapool/tingting/calvin_data/task_D_D \\
      --split validation \\
      --out_dir data/gt-q6-mv-calvin

  # smoke
  python scripts/01b_extract_gt_calvin.py \\
      --data_root /umd-datapool/tingting/calvin_data/task_D_D \\
      --split validation --max_windows 5 --out_dir /tmp/calvin_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from bench.calvin_loader import (
    CalvinWindow,
    load_lang_annotations,
    load_episode,
    sample_window_frames,
)
from bench.calvin_gt_extractor import extract_gt_from_calvin
from bench.calvin_taxonomy import TASK_CLASS_TO_TARGET, DENY_LIST


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="Root of the CALVIN split (containing lang_annotations/ and episode_*.npz).")
    p.add_argument("--split", default="validation",
                   help="Subdir under data_root (default: validation). Set to '' if data_root is already the split dir.")
    p.add_argument("--out_dir", required=True,
                   help="Output dir, e.g. data/gt-q6-mv-calvin")
    p.add_argument("--suite_name", default="calvin_D",
                   help="Suite label to record in the manifest (default: calvin_D)")
    p.add_argument("--max_windows", type=int, default=-1,
                   help="If > 0, cap the number of annotated windows processed (smoke).")
    p.add_argument("--n_traj_frames", type=int, default=3,
                   help="Number of trajectory frames sampled inside each window (default 3).")
    p.add_argument("--include_close", action="store_true", default=True,
                   help="Also save the last frame of each window as the 'close' frame.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_id_from_class(task_class: str, registry: dict[str, int]) -> int:
    """Stable integer task_id per task class, assigned in first-seen order."""
    if task_class not in registry:
        registry[task_class] = len(registry)
    return registry[task_class]


def _save_record(
    gt: dict,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    sample_id: int,
    suite_name: str,
    task_id: int,
    task_desc: str,
    window: CalvinWindow,
    frame_type: str,
    frame_idx: int,
    out_dir: Path,
    manifest: list,
    data_root_for_paths: Path,
    traj_step: int | None = None,
    traj_len: int | None = None,
) -> int:
    if frame_type == "traj" and traj_step is not None and traj_len is not None:
        type_suffix = f"traj_{traj_step:03d}of{traj_len:03d}"
    else:
        type_suffix = frame_type
    stem = f"sample_{sample_id:04d}_t{task_id:02d}_w{window.window_id:04d}_{type_suffix}"

    rel: dict[str, str] = {}
    for view, arr in (("agent", rgb_static), ("wrist", rgb_gripper)):
        png_path = out_dir / f"{stem}_{view}.png"
        Image.fromarray(arr).save(png_path)
        rel[view] = str(png_path.resolve().relative_to(data_root_for_paths.resolve()))

    record = {
        "sample_id": sample_id,
        "suite": suite_name,
        "task_id": int(task_id),
        "task_description": task_desc,
        # We reuse the LIBERO field name: init_state_idx == window's start frame index.
        "init_state_idx": int(window.start_idx),
        "frame_type": frame_type,
        "image_path": rel["agent"],          # legacy / agent
        "image_paths": rel,                  # full dual-view dict
        "calvin_window": [int(window.start_idx), int(window.end_idx)],
        "calvin_frame_idx": int(frame_idx),
        "calvin_task_class": window.task_class,
        "gt": gt,
    }
    if traj_step is not None:
        record["traj_step"] = traj_step
        record["traj_len"] = traj_len
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2))
    manifest.append(record)
    return sample_id + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base = Path(args.data_root)
    split_root = base / args.split if args.split else base
    if not split_root.exists():
        print(f"ERROR: split root not found: {split_root}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir)
    suite_dir = out_root / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    # data_root for relative image paths is the parent of out_root (mirrors
    # LIBERO's "data/" convention).
    data_root_for_paths = out_root.resolve().parent

    print(f"Reading annotations from {split_root}/lang_annotations/auto_lang_ann.npy")
    windows = load_lang_annotations(split_root)
    print(f"  total annotated windows: {len(windows)}")

    if args.max_windows and args.max_windows > 0:
        windows = windows[: args.max_windows]
        print(f"  capping at --max_windows={args.max_windows}")

    # Skip windows whose task_class is unknown or in the deny-list.
    skipped_unknown = 0
    skipped_deny = 0
    eligible: list[CalvinWindow] = []
    for w in windows:
        if w.task_class not in TASK_CLASS_TO_TARGET:
            skipped_unknown += 1
            continue
        if w.task_class in DENY_LIST:
            skipped_deny += 1
            continue
        eligible.append(w)
    print(f"  unknown task: {skipped_unknown}, deny-listed: {skipped_deny}, eligible: {len(eligible)}")

    task_id_registry: dict[str, int] = {}
    manifest: list[dict] = []
    sample_id = 0

    for w in tqdm(eligible, desc="windows"):
        tid = _task_id_from_class(w.task_class, task_id_registry)
        window_dir = suite_dir / f"task_{tid:02d}" / f"window_{w.window_id:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)

        frames = sample_window_frames(
            w,
            n_traj_frames=args.n_traj_frames,
            include_close=args.include_close,
        )
        # Need to know traj_len for naming.
        traj_indices = [(ft, idx) for (ft, idx) in frames if ft == "traj"]
        traj_len = len(traj_indices)

        for ft, idx in frames:
            try:
                ep = load_episode(split_root, idx)
            except FileNotFoundError as e:
                print(f"  WARN: skipping frame {idx} (window {w.window_id}): {e}")
                continue

            gt = extract_gt_from_calvin(
                robot_obs=ep["robot_obs"],
                scene_obs=ep["scene_obs"],
                task_class=w.task_class,
                task_description=w.language,
            )
            if gt is None:
                # Shouldn't happen because we filtered eligible, but guard anyway.
                continue

            traj_step = None
            traj_step_argv = None
            if ft == "traj":
                traj_step_argv = traj_indices.index((ft, idx))
                traj_step = traj_step_argv

            sample_id = _save_record(
                gt,
                rgb_static=ep["rgb_static"],
                rgb_gripper=ep["rgb_gripper"],
                sample_id=sample_id,
                suite_name=args.suite_name,
                task_id=tid,
                task_desc=w.language,
                window=w,
                frame_type=ft,
                frame_idx=idx,
                out_dir=window_dir,
                manifest=manifest,
                data_root_for_paths=data_root_for_paths,
                traj_step=traj_step,
                traj_len=traj_len if ft == "traj" else None,
            )

    suite_manifest = suite_dir / "manifest.json"
    suite_manifest.write_text(json.dumps(manifest, indent=2))

    # Top-level manifest (flat, mirrors per-suite for now since we have one suite).
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    by_frame_type: dict[str, int] = defaultdict(int)
    for r in manifest:
        by_frame_type[r["frame_type"]] += 1
    print(f"\n  [{args.suite_name}] {len(manifest)} samples saved.")
    print(f"  Per frame_type: {dict(by_frame_type)}")
    print(f"  Per task_class registry (task_id assignment):")
    for cls, tid in sorted(task_id_registry.items(), key=lambda kv: kv[1]):
        print(f"    task_{tid:02d}  {cls}")
    print(f"  Manifest: {suite_manifest}")


if __name__ == "__main__":
    main()
