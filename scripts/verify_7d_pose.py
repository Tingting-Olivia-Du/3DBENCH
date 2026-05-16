#!/usr/bin/env python3
"""Verify all 7 dimensions of the EE pose by nudging each independently.

For one LIBERO task, resets to an init state, then:
  1. Captures a baseline frame (no movement).
  2. Nudges the EE along +X, -X, +Y, -Y, +Z, -Z (translation).
  3. Nudges the EE along +Roll, -Roll, +Pitch, -Pitch, +Yaw, -Yaw (rotation).
  4. Toggles gripper open vs closed.

Each nudge is saved as a SEPARATE image: [Baseline | After Nudge] with
text annotations BELOW the images (not overlaid).

Usage:
  conda activate /workspace/tingting/envs/vlmbench
  python scripts/verify_7d_pose.py --out_dir data/verify/verify_7d-2

  # Specific suite/task
  python scripts/verify_7d_pose.py --suite libero_10 --task_id 0 --out_dir data/verify/verify_7d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.gt_extractor import (
    get_agentview_image,
    get_wrist_image,
    get_robot_base_pos,
    get_sim_from_env,
    quat_wxyz_from_xyzw,
    quat_to_euler_deg,
    _compute_gripper_openness_from_sim,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--state_idx", type=int, default=0)
    p.add_argument("--out_dir", default="data/verify/verify_7d")
    p.add_argument("--n_steps", type=int, default=40,
                   help="Number of sim steps per nudge (default 40)")
    p.add_argument("--obs_size", type=int, default=256)
    return p.parse_args()


def _import_libero():
    try:
        from libero.libero import benchmark as bench_mod
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        return bench_mod, get_libero_path, OffScreenRenderEnv
    except ImportError as exc:
        print(f"ERROR: Cannot import LIBERO: {exc}")
        sys.exit(1)


DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _get_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def get_ee_pose_7d(raw_obs, sim):
    """Extract current 7D EE pose."""
    base_pos = get_robot_base_pos(sim)
    eef_pos = np.array(raw_obs["robot0_eef_pos"]) - base_pos
    eef_quat_wxyz = quat_wxyz_from_xyzw(np.array(raw_obs["robot0_eef_quat"]))
    euler = quat_to_euler_deg(eef_quat_wxyz)
    openness = _compute_gripper_openness_from_sim(sim, raw_obs)
    return {
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "gripper": openness,
    }


def capture_baseline(env, init_state, n_settle=10):
    """Reset to init_state, settle, return raw_obs."""
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(n_settle):
        raw_obs = env.step(DUMMY_ACTION)[0]
    return raw_obs


def nudge_and_capture(env, init_state, action, n_steps, n_settle=10):
    """Reset, settle, apply action for n_steps, return (raw_obs, pose_before, pose_after)."""
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(n_settle):
        raw_obs = env.step(DUMMY_ACTION)[0]

    sim = get_sim_from_env(env)
    pose_before = get_ee_pose_7d(raw_obs, sim)

    for _ in range(n_steps):
        raw_obs = env.step(action)[0]

    sim = get_sim_from_env(env)
    pose_after = get_ee_pose_7d(raw_obs, sim)

    return raw_obs, pose_before, pose_after


def make_text_bar(lines: list[str], width: int, color=(255, 255, 255),
                  bg_color=(30, 30, 30), font_size=13) -> np.ndarray:
    """Create a text bar image (dark background with text) of given width."""
    font = _get_font(font_size)
    line_height = font_size + 4
    height = max(line_height * len(lines) + 8, 20)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    y = 4
    for line in lines:
        draw.text((8, y), line, fill=color, font=font)
        y += line_height
    return np.array(img)


def format_pose(pose: dict, label: str) -> list[str]:
    """Format a 7D pose as display lines."""
    return [
        f"{label}:",
        f"  pos=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) m",
        f"  rot=({pose['roll']:.1f}, {pose['pitch']:.1f}, {pose['yaw']:.1f}) deg",
        f"  gripper={pose['gripper']:.2f}",
    ]


def format_delta(pose_before: dict, pose_after: dict) -> list[str]:
    """Format the delta between two poses."""
    lines = ["Delta:"]
    for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]:
        delta = pose_after[key] - pose_before[key]
        unit = "m" if key in ("x", "y", "z") else ("" if key == "gripper" else "deg")
        if abs(delta) > 0.001:
            lines.append(f"  d{key} = {delta:+.4f} {unit}")
    return lines


def save_comparison(baseline_img: np.ndarray, nudged_img: np.ndarray,
                    baseline_pose: dict, nudged_pose_before: dict,
                    nudged_pose_after: dict,
                    title: str, axis_info: str,
                    out_path: Path):
    """Save a single comparison image: [Baseline | After Nudge] with text below."""
    h, w = baseline_img.shape[:2]
    full_w = w * 2 + 4  # 4px gap between images

    # Title bar
    title_lines = [title]
    if axis_info:
        title_lines.append(axis_info)
    title_bar = make_text_bar(title_lines, full_w, color=(255, 200, 80), font_size=15)

    # Column labels
    left_label = make_text_bar(["Baseline"], w, color=(180, 180, 180))
    gap_label = np.full((left_label.shape[0], 4, 3), 30, dtype=np.uint8)
    right_label = make_text_bar(["After Nudge"], w, color=(255, 120, 120))
    label_row = np.concatenate([left_label, gap_label, right_label], axis=1)

    # Image row: baseline | gap | nudged
    gap = np.zeros((h, 4, 3), dtype=np.uint8)
    img_row = np.concatenate([baseline_img, gap, nudged_img], axis=1)

    # Pose info below
    base_lines = format_pose(baseline_pose, "Before")
    after_lines = format_pose(nudged_pose_after, "After")
    delta_lines = format_delta(nudged_pose_before, nudged_pose_after)

    left_info = make_text_bar(base_lines, w, color=(180, 180, 180))
    right_info = make_text_bar(after_lines + [""] + delta_lines, w, color=(120, 255, 120))
    # Pad to same height
    max_h = max(left_info.shape[0], right_info.shape[0])
    if left_info.shape[0] < max_h:
        pad = np.full((max_h - left_info.shape[0], left_info.shape[1], 3), 30, dtype=np.uint8)
        left_info = np.concatenate([left_info, pad], axis=0)
    if right_info.shape[0] < max_h:
        pad = np.full((max_h - right_info.shape[0], right_info.shape[1], 3), 30, dtype=np.uint8)
        right_info = np.concatenate([right_info, pad], axis=0)
    gap_info = np.full((max_h, 4, 3), 30, dtype=np.uint8)
    info_row = np.concatenate([left_info, gap_info, right_info], axis=1)

    # Stack vertically: title | labels | images | info
    combined = np.concatenate([title_bar, label_row, img_row, info_row], axis=0)
    Image.fromarray(combined).save(out_path)


# ── Axis descriptions for each dimension ────────────────────────────────

AXIS_INFO = {
    "+X":      "X-axis = forward (away from robot into workspace)",
    "-X":      "X-axis = forward (away from robot into workspace)",
    "+Y":      "Y-axis = right (robot's right when facing forward)",
    "-Y":      "Y-axis = right (robot's right when facing forward)",
    "+Z":      "Z-axis = up (vertical)",
    "-Z":      "Z-axis = up (vertical)",
    "+Roll":   "Roll = rotation about X-axis (forward axis, tilts left/right)",
    "-Roll":   "Roll = rotation about X-axis (forward axis, tilts left/right)",
    "+Pitch":  "Pitch = rotation about Y-axis (right axis, nods up/down)",
    "-Pitch":  "Pitch = rotation about Y-axis (right axis, nods up/down)",
    "+Yaw":    "Yaw = rotation about Z-axis (up axis, turns left/right)",
    "-Yaw":    "Yaw = rotation about Z-axis (up axis, turns left/right)",
    "Gripper Close": "Gripper: -1=open, +1=close",
}


def main():
    args = parse_args()
    bench_mod, get_libero_path, OffScreenRenderEnv = _import_libero()
    import os, torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_dict = bench_mod.get_benchmark_dict()
    task_suite = bench_dict[args.suite]()
    task = task_suite.get_task(args.task_id)

    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path, weights_only=False)
    init_state = init_states[args.state_idx]

    print(f"Task: {task.language}")
    print(f"Suite: {args.suite}, task_id: {args.task_id}, state_idx: {args.state_idx}")

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file,
                             camera_heights=args.obs_size,
                             camera_widths=args.obs_size)

    # ── Define all nudges ────────────────────────────────────────────────
    SPEED = 0.4
    ROLL = 0.3
    PITCH =0.3
    YAW = 0.2
    nudges = [
        # Translation
        ("+X",             [+SPEED, 0, 0, 0, 0, 0, -1]),
        ("-X",             [-SPEED, 0, 0, 0, 0, 0, -1]),
        ("+Y",             [0, +SPEED, 0, 0, 0, 0, -1]),
        ("-Y",             [0, -SPEED, 0, 0, 0, 0, -1]),
        ("+Z",             [0, 0, +SPEED, 0, 0, 0, -1]),
        ("-Z",             [0, 0, -SPEED, 0, 0, 0, -1]),
        # Rotation
        ("+Roll",          [0, 0, 0, +ROLL, 0, 0, -1]),
        ("-Roll",          [0, 0, 0, -ROLL, 0, 0, -1]),
        ("+Pitch",         [0, 0, 0, 0, +PITCH, 0, -1]),
        ("-Pitch",         [0, 0, 0, 0, -PITCH, 0, -1]),
        ("+Yaw",           [0, 0, 0, 0, 0, +YAW, -1]),
        ("-Yaw",           [0, 0, 0, 0, 0, -YAW, -1]),
        # Gripper
        ("Gripper Close",  [0, 0, 0, 0, 0, 0, +1]),
    ]

    # ── Capture baseline ─────────────────────────────────────────────────
    print("\n--- Baseline ---")
    raw_obs = capture_baseline(env, init_state)
    sim = get_sim_from_env(env)
    baseline_pose = get_ee_pose_7d(raw_obs, sim)
    print(f"  pos=({baseline_pose['x']:.4f}, {baseline_pose['y']:.4f}, {baseline_pose['z']:.4f}) m")
    print(f"  rot=({baseline_pose['roll']:.1f}, {baseline_pose['pitch']:.1f}, {baseline_pose['yaw']:.1f}) deg")
    print(f"  gripper={baseline_pose['gripper']:.3f}")

    baseline_agent = get_agentview_image(raw_obs)
    baseline_wrist = get_wrist_image(raw_obs)

    # ── Capture and save each nudge as a separate image ──────────────────
    for name, action in nudges:
        print(f"\n--- {name} ---")
        raw_obs, pose_before, pose_after = nudge_and_capture(
            env, init_state, action, args.n_steps
        )

        # Print delta
        for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]:
            delta = pose_after[key] - pose_before[key]
            if abs(delta) > 0.001:
                unit = "m" if key in ("x", "y", "z") else ("" if key == "gripper" else "deg")
                print(f"  d{key} = {delta:+.4f} {unit}")

        agent_img = get_agentview_image(raw_obs)
        wrist_img = get_wrist_image(raw_obs)

        safe_name = name.replace("+", "plus").replace("-", "minus").replace(" ", "_")
        axis_info = AXIS_INFO.get(name, "")

        # Agent view comparison
        save_comparison(
            baseline_agent, agent_img,
            baseline_pose, pose_before, pose_after,
            title=f"Agent View: {name}",
            axis_info=axis_info,
            out_path=out_dir / f"agent_{safe_name}.png",
        )

        # Wrist view comparison
        save_comparison(
            baseline_wrist, wrist_img,
            baseline_pose, pose_before, pose_after,
            title=f"Wrist View: {name}",
            axis_info=axis_info,
            out_path=out_dir / f"wrist_{safe_name}.png",
        )

    env.close()

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Saved {len(nudges) * 2} comparison images to {out_dir}/")
    print(f"\nCoordinate frame (XYZ intrinsic Euler):")
    print(f"  X = forward (away from robot)")
    print(f"  Y = right   (robot's right)")
    print(f"  Z = up      (vertical)")
    print(f"  Roll  = rotation about X (forward axis, tilts left/right)")
    print(f"  Pitch = rotation about Y (right axis, nods up/down)")
    print(f"  Yaw   = rotation about Z (up axis, turns left/right)")
    print(f"  Gripper: 0.0=closed, 1.0=open")


if __name__ == "__main__":
    main()
