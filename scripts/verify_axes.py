#!/usr/bin/env python3
"""Verify coordinate axis directions by nudging the EE along each axis.

For one LIBERO task, resets to an init state, then:
  1. Captures a baseline frame (no movement).
  2. Nudges the EE along +X, +Y, +Z separately, capturing a frame after each.

Saves 4 agent-view images side by side so you can visually confirm
which image direction corresponds to which coordinate axis.

Usage:
  source /umd-datapool/tingting/envs/vlmbench/bin/activate
  python scripts/verify_axes.py --out_dir data/axis_verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.gt_extractor import get_agentview_image, get_wrist_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--state_idx", type=int, default=0)
    p.add_argument("--out_dir", default="data/axis_verify")
    p.add_argument("--n_steps", type=int, default=40,
                   help="Number of sim steps to nudge (default 40)")
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


def capture_frame(env, init_state, n_settle=10):
    """Reset to init_state, settle, return raw_obs."""
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(n_settle):
        raw_obs = env.step(DUMMY_ACTION)[0]
    return raw_obs


def nudge_and_capture(env, init_state, axis_action, n_steps, n_settle=10):
    """Reset, settle, then apply axis_action for n_steps, return final raw_obs."""
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(n_settle):
        raw_obs = env.step(DUMMY_ACTION)[0]

    # Record EE position before nudge
    eef_before = raw_obs["robot0_eef_pos"].copy()

    for _ in range(n_steps):
        raw_obs = env.step(axis_action)[0]

    eef_after = raw_obs["robot0_eef_pos"].copy()
    delta = eef_after - eef_before
    print(f"  EE moved: dx={delta[0]:.4f}  dy={delta[1]:.4f}  dz={delta[2]:.4f}")

    return raw_obs


def add_label(img_array, text, color=(255, 0, 0)):
    """Add a text label to the top-left of an image."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    # Use a large font size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    # Draw text with black outline for readability
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]:
        draw.text((10+dx, 10+dy), text, fill=(0, 0, 0), font=font)
    draw.text((10, 10), text, fill=color, font=font)
    return np.array(img)


def main():
    args = parse_args()
    bench_mod, get_libero_path, OffScreenRenderEnv = _import_libero()
    import os, torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_dict = bench_mod.get_benchmark_dict()
    task_suite = bench_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    task_desc = task.language

    # Load init states
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path, weights_only=False)
    init_state = init_states[args.state_idx]

    print(f"Task: {task_desc}")
    print(f"Suite: {args.suite}, task_id: {args.task_id}, state_idx: {args.state_idx}")

    # Create env
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file,
                             camera_heights=args.obs_size,
                             camera_widths=args.obs_size)

    # Define nudge actions: [dx, dy, dz, 0, 0, 0, gripper]
    # +1.0 = max speed in that axis direction
    nudges = {
        "baseline": DUMMY_ACTION,
        "+X (action dx=+1)": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "+Y (action dy=+1)": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "+Z (action dz=+1)": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
    }

    agent_images = []
    wrist_images = []

    for name, action in nudges.items():
        print(f"\n--- {name} ---")
        if name == "baseline":
            raw_obs = capture_frame(env, init_state)
            eef = raw_obs["robot0_eef_pos"]
            print(f"  EE position: x={eef[0]:.4f}  y={eef[1]:.4f}  z={eef[2]:.4f}")
        else:
            raw_obs = nudge_and_capture(env, init_state, action, args.n_steps)

        agent_img = get_agentview_image(raw_obs)
        wrist_img = get_wrist_image(raw_obs)

        agent_images.append(add_label(agent_img, name, color=(255, 50, 50)))
        wrist_images.append(add_label(wrist_img, name, color=(50, 255, 50)))

    env.close()

    # Compose side-by-side comparison
    agent_row = np.concatenate(agent_images, axis=1)
    wrist_row = np.concatenate(wrist_images, axis=1)
    combined = np.concatenate([agent_row, wrist_row], axis=0)

    out_path = out_dir / "axis_verification.png"
    Image.fromarray(combined).save(out_path)
    print(f"\nSaved: {out_path}")
    print("Top row: agent view | Bottom row: wrist view")
    print("Compare baseline vs +X/+Y/+Z to see which direction the EE moved in the image.")

    # Also save individual frames
    for i, name in enumerate(nudges.keys()):
        safe = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("=", "")
        Image.fromarray(agent_images[i]).save(out_dir / f"agent_{safe}.png")
        Image.fromarray(wrist_images[i]).save(out_dir / f"wrist_{safe}.png")
    print(f"Individual frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
