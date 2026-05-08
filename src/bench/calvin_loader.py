"""I/O helpers for the CALVIN dataset (env D, validation split).

The official layout (after running calvin/dataset/download_data.sh):

    <data_root>/
      training/        episode_NNNNNNN.npz  (we don't use this for eval)
      validation/      episode_NNNNNNN.npz  ← we use this
        lang_annotations/
          auto_lang_ann.npy   { "language": {"ann": [...], "task": [...], ...},
                                 "info":     {"indx": [(start, end), ...]} }

Each episode_*.npz is a NumPy dict of per-timestep observations:
    rgb_static       (H, W, 3) uint8     200x200 third-person
    rgb_gripper      (H, W, 3) uint8      84x84  wrist
    robot_obs        (15,) float32       [tcp_pos(3), tcp_orn(3), gripper_width(1),
                                          arm_joint_pos(7), gripper_action(1)]
    scene_obs        (24,) float32       (see calvin_taxonomy.py)
    actions          (7,) float32        (we ignore for QA extraction)

Episodes are named by their absolute frame index, e.g. episode_0000123.npz —
exactly one .npz per timestep. Annotated windows index into this flat array.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CalvinWindow:
    """One annotated window from auto_lang_ann.npy."""
    start_idx: int
    end_idx: int
    task_class: str   # e.g. "lift_red_block_table"
    language: str     # natural-language instruction
    window_id: int    # 0-based index into the windows list (for stable filenames)


def load_lang_annotations(split_root: Path) -> list[CalvinWindow]:
    """Read lang_annotations/auto_lang_ann.npy and return a list of CalvinWindow.

    Robust to small format variations across calvin releases:
      - Some checkpoints store {"language": {"ann": [...], "task": [...]},
                                 "info":     {"indx": [...]}}
      - Older ones store {"language": [...], "task": [...], "indx": [...]}
    """
    ann_path = split_root / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        raise FileNotFoundError(f"Could not find {ann_path}")
    raw = np.load(ann_path, allow_pickle=True).item()

    # Format A — modern, nested
    if "language" in raw and isinstance(raw["language"], dict):
        anns = list(raw["language"]["ann"])
        tasks = list(raw["language"]["task"])
        indx = list(raw["info"]["indx"])
    # Format B — flat
    else:
        anns = list(raw["language"])
        tasks = list(raw["task"])
        indx = list(raw["indx"])

    if not (len(anns) == len(tasks) == len(indx)):
        raise ValueError(
            f"Mismatched annotation lengths: language={len(anns)} task={len(tasks)} indx={len(indx)}"
        )

    out: list[CalvinWindow] = []
    for i, ((s, e), task, lang) in enumerate(zip(indx, tasks, anns)):
        out.append(CalvinWindow(
            start_idx=int(s),
            end_idx=int(e),
            task_class=str(task),
            language=str(lang),
            window_id=i,
        ))
    return out


_FRAME_RE = re.compile(r"episode_(\d+)\.npz$")


def episode_path(split_root: Path, frame_idx: int) -> Path:
    """Resolve the path to the episode_*.npz file for a given absolute frame index.

    CALVIN pads the integer to (typically) 7 digits. We probe several widths
    in case the dataset uses a different padding.
    """
    for width in (7, 6, 8, 5, 9):
        p = split_root / f"episode_{frame_idx:0{width}d}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No episode npz found for frame {frame_idx} under {split_root} "
        f"(tried widths 5–9)"
    )


def load_episode(split_root: Path, frame_idx: int) -> dict:
    """Load one timestep into a dict (numpy arrays), no copy of unused keys."""
    path = episode_path(split_root, frame_idx)
    d = np.load(path)
    out = {
        "rgb_static": d["rgb_static"],          # (200,200,3) uint8
        "rgb_gripper": d["rgb_gripper"],        # ( 84, 84,3) uint8
        "robot_obs":   d["robot_obs"].astype(np.float32),
        "scene_obs":   d["scene_obs"].astype(np.float32),
    }
    return out


def sample_window_frames(
    window: CalvinWindow,
    n_traj_frames: int = 3,
    include_close: bool = True,
) -> list[tuple[str, int]]:
    """Pick (frame_type, frame_idx) tuples for a window, mirroring LIBERO.

      - "init"  : window.start_idx
      - "traj"  : up to n_traj_frames evenly spaced strictly between start and end-2
      - "close" : end_idx - 1 (the last recorded frame; the actual `can_close`
                  value will be filled in by the GT extractor)
    """
    s, e = window.start_idx, window.end_idx
    out: list[tuple[str, int]] = [("init", s)]
    if n_traj_frames > 0 and e - s > 3:
        traj_lo = s + 1
        traj_hi = e - 2
        idxs = np.linspace(traj_lo, traj_hi, n_traj_frames).astype(int)
        idxs = sorted(set(int(i) for i in idxs))
        for j in idxs:
            out.append(("traj", j))
    if include_close and e - 1 > s:
        out.append(("close", e - 1))
    return out
