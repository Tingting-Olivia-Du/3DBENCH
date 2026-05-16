#!/usr/bin/env python3
"""Dense rotation sweep: visualize Roll, Pitch, Yaw through a full circle.

For each rotation axis, incrementally nudges the EE from baseline through
N evenly-spaced angles and captures a frame at each step. Overlays the
EE coordinate axes (X=red, Y=green, Z=blue) on each image.

Usage:
  conda activate /workspace/tingting/envs/vlmbench
  python scripts/verify_rotation_sweep.py --out_dir data/verify/rotation_sweep_vis

  # More/fewer samples
  python scripts/verify_rotation_sweep.py --n_angles 24 --out_dir data/verify/rotation_sweep

python scripts/verify_rotation_sweep.py --out_dir data/verify/rotation_sweep_vis-2


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
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--state_idx", type=int, default=0)
    p.add_argument("--out_dir", default="data/verify/rotation_sweep")
    p.add_argument("--n_angles", type=int, default=16,
                   help="Number of angles to sample per axis (default 16)")
    p.add_argument("--steps_per_angle", type=int, default=8,
                   help="Sim steps per angle increment (default 8)")
    p.add_argument("--speed", type=float, default=0.3,
                   help="Action magnitude for rotation (default 0.3)")
    p.add_argument("--obs_size", type=int, default=256)
    p.add_argument("--grid_cols", type=int, default=4,
                   help="Columns in the grid image (default 4)")
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


def get_euler_deg(raw_obs, sim):
    """Get current EE euler angles in degrees."""
    eef_quat_wxyz = quat_wxyz_from_xyzw(np.array(raw_obs["robot0_eef_quat"]))
    return quat_to_euler_deg(eef_quat_wxyz)  # [roll, pitch, yaw]


def get_ee_axes_world(sim):
    """Get EE position and local axis directions in world frame."""
    site_id = sim.model.site_name2id("gripper0_grip_site")
    ee_pos = sim.data.site_xpos[site_id].copy()
    ee_mat = sim.data.site_xmat[site_id].reshape(3, 3).copy()
    # Columns of ee_mat are the local X, Y, Z axes in world frame
    return ee_pos, ee_mat[:, 0], ee_mat[:, 1], ee_mat[:, 2]


# ---------------------------------------------------------------------------
# 3D → 2D projection using MuJoCo camera parameters
# ---------------------------------------------------------------------------

def project_to_camera(world_points: np.ndarray, sim, cam_name: str,
                      img_h: int, img_w: int) -> np.ndarray:
    """Project 3D world points to 2D pixel coordinates.

    Uses MuJoCo camera parameters. The cam_xmat is a 3x3 rotation matrix
    (camera-to-world), so world-to-camera = cam_xmat.T.
    MuJoCo uses OpenGL convention: camera looks along -Z, Y is up, X is right.

    Args:
        world_points: (N, 3) array of 3D points in world frame.
        sim:          MuJoCo sim object.
        cam_name:     Camera name (e.g. "agentview", "robot0_eye_in_hand").
        img_h, img_w: Image dimensions.

    Returns:
        (N, 2) array of (u, v) pixel coordinates. u=column, v=row.
    """
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    # cam_xmat: rotation from camera local frame to world frame (column = axis in world)
    cam_to_world = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    world_to_cam = cam_to_world.T  # inverse rotation
    fovy = sim.model.cam_fovy[cam_id]

    # Transform to camera frame
    pts = np.atleast_2d(world_points)
    rel = pts - cam_pos  # (N, 3) relative to camera position in world
    cam_coords = (world_to_cam @ rel.T).T  # (N, 3) in camera local frame
    # cam_coords: x=right, y=up, z=back (OpenGL: camera looks along -Z)

    # Perspective projection
    # Depth is -z in OpenGL convention (objects in front have negative cam z)
    depth = -cam_coords[:, 2].copy()
    depth[np.abs(depth) < 1e-8] = 1e-8

    f = 0.5 * img_h / np.tan(np.radians(fovy / 2.0))

    # LIBERO renders images flipped (upside-down), and get_agentview_image
    # does [::-1, ::-1] to correct. This flips both H and W, so we need to
    # account for that flip in the projection.
    u = img_w / 2.0 - f * cam_coords[:, 0] / depth  # flipped X
    v = img_h / 2.0 + f * cam_coords[:, 1] / depth   # flipped Y

    return np.stack([u, v], axis=1)


def _draw_arrow(draw, ox, oy, tx, ty, color, width=2, arrow_size=6, font=None, label=None):
    """Draw a line with arrowhead and optional label."""
    draw.line([(ox, oy), (tx, ty)], fill=color, width=width)
    # Arrowhead
    dx, dy = tx - ox, ty - oy
    length = max(np.sqrt(dx**2 + dy**2), 1e-3)
    ux, uy = dx / length, dy / length
    px1 = int(tx - arrow_size * (ux + uy * 0.5))
    py1 = int(ty - arrow_size * (uy - ux * 0.5))
    px2 = int(tx - arrow_size * (ux - uy * 0.5))
    py2 = int(ty - arrow_size * (uy + ux * 0.5))
    draw.polygon([(tx, ty), (px1, py1), (px2, py2)], fill=color)
    if label and font:
        draw.text((tx + 3, ty - 6), label, fill=color, font=font)


def draw_axes_on_image(img: np.ndarray, sim, cam_name: str,
                       axis_length: float = 0.08) -> np.ndarray:
    """Draw BOTH world axes (fixed) and EE local axes (rotating) on an image.

    World axes (drawn at EE position as reference, dashed/thin):
      Xw=light red, Yw=light green, Zw=light blue — these never change.
    EE local axes (solid/thick arrows):
      Xe=bright red, Ye=bright green, Ze=bright blue — these rotate with the EE.

    This lets you see how the EE frame rotates relative to the fixed world frame.
    """
    h, w = img.shape[:2]
    ee_pos, ax_x, ax_y, ax_z = get_ee_axes_world(sim)

    pil_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(11)

    # Project EE origin
    origin_px = project_to_camera(ee_pos.reshape(1, 3), sim, cam_name, h, w)[0]
    ox, oy = int(origin_px[0]), int(origin_px[1])

    # ── World axes (fixed reference, thin dashed lines) ──────────────
    world_axes = [
        ("+Xw", np.array([1, 0, 0]), (180, 80, 80)),    # light red
        ("+Yw", np.array([0, 1, 0]), (80, 180, 80)),     # light green
        ("+Zw", np.array([0, 0, 1]), (80, 80, 180)),     # light blue
    ]
    for label, axis_dir, color in world_axes:
        tip_pos = ee_pos + axis_dir * axis_length * 0.7
        tip_px = project_to_camera(tip_pos.reshape(1, 3), sim, cam_name, h, w)[0]
        tx, ty = int(tip_px[0]), int(tip_px[1])
        # Dashed effect: draw thin line
        draw.line([(ox, oy), (tx, ty)], fill=color, width=1)
        draw.text((tx + 2, ty - 5), label, fill=color, font=font)

    # ── EE local axes (rotating, thick solid arrows) ─────────────────
    ee_axes = [
        ("+X", ax_x, (255, 60, 60)),     # bright red
        ("+Y", ax_y, (60, 255, 60)),      # bright green
        ("+Z", ax_z, (80, 80, 255)),      # bright blue
    ]
    for label, axis_dir, color in ee_axes:
        tip_pos = ee_pos + axis_dir * axis_length
        tip_px = project_to_camera(tip_pos.reshape(1, 3), sim, cam_name, h, w)[0]
        tx, ty = int(tip_px[0]), int(tip_px[1])
        _draw_arrow(draw, ox, oy, tx, ty, color, width=3, arrow_size=7,
                    font=font, label=label)

    # Draw origin dot
    draw.ellipse([(ox - 3, oy - 3), (ox + 3, oy + 3)], fill=(255, 255, 255))

    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Sweep and visualization
# ---------------------------------------------------------------------------

def make_text_bar(lines: list[str], width: int, color=(255, 255, 255),
                  bg_color=(30, 30, 30), font_size=13) -> np.ndarray:
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


def sweep_axis(env, init_state, axis_idx, axis_name, n_angles, steps_per_angle,
               speed, obs_size):
    """Sweep one rotation axis and return list of frame tuples.

    Each frame: (agent_img, wrist_img, agent_axes_img, wrist_axes_img,
                 euler_deg, cum_steps, label)
    """
    frames = []

    def _capture_frame(raw_obs, cum_steps, label):
        sim = get_sim_from_env(env)
        euler = get_euler_deg(raw_obs, sim)
        agent_img = get_agentview_image(raw_obs)
        wrist_img = get_wrist_image(raw_obs)
        # Draw axes overlay
        agent_axes = draw_axes_on_image(agent_img, sim, "agentview")
        wrist_axes = draw_axes_on_image(wrist_img, sim, "robot0_eye_in_hand")
        return (agent_img, wrist_img, agent_axes, wrist_axes,
                euler.copy(), cum_steps, label)

    # ── Positive direction: 0 → +max ────────────────────────────────────
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(10):
        raw_obs = env.step(DUMMY_ACTION)[0]

    frames.append(_capture_frame(raw_obs, 0, "baseline"))

    action = list(DUMMY_ACTION)
    action[axis_idx] = +speed
    cum_steps = 0
    for i in range(n_angles):
        for _ in range(steps_per_angle):
            raw_obs = env.step(action)[0]
            cum_steps += 1
        frames.append(_capture_frame(raw_obs, cum_steps, f"+{axis_name} #{i+1}"))

    # ── Negative direction: baseline → -max ─────────────────────────────
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(10):
        raw_obs = env.step(DUMMY_ACTION)[0]

    action = list(DUMMY_ACTION)
    action[axis_idx] = -speed
    cum_steps = 0
    neg_frames = []
    for i in range(n_angles):
        for _ in range(steps_per_angle):
            raw_obs = env.step(action)[0]
            cum_steps += 1
        neg_frames.append(_capture_frame(raw_obs, cum_steps, f"-{axis_name} #{i+1}"))

    return list(reversed(neg_frames)) + frames


def build_grid(frames, axis_name, axis_desc, use_axes_overlay, grid_cols, obs_size):
    """Build a grid image from frames with labels below each cell."""
    euler_idx = {"Roll": 0, "Pitch": 1, "Yaw": 2}[axis_name]
    n = len(frames)
    n_rows = (n + grid_cols - 1) // grid_cols

    cell_w = obs_size
    label_h = 55
    cell_h = obs_size + label_h
    grid_w = cell_w * grid_cols
    grid_h = cell_h * n_rows

    title_lines = [
        f"{axis_name} Sweep ({'with axes' if use_axes_overlay else 'raw'}) - {n} frames",
        axis_desc,
        "Axes: X=Red(forward)  Y=Green(right)  Z=Blue(up)",
    ]
    title_bar = make_text_bar(title_lines, grid_w, color=(255, 200, 80), font_size=13)

    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)
    font = _get_font(11)

    for i, frame in enumerate(frames):
        (agent_img, wrist_img, agent_axes, wrist_axes,
         euler, cum_steps, label) = frame
        row = i // grid_cols
        col = i % grid_cols
        x0 = col * cell_w
        y0 = row * cell_h

        # Use axes overlay version
        if use_axes_overlay:
            img = agent_axes
        else:
            img = agent_img

        if img.shape[0] != obs_size or img.shape[1] != obs_size:
            img = np.array(Image.fromarray(img).resize((obs_size, obs_size)))

        grid[y0:y0 + obs_size, x0:x0 + obs_size] = img

        # Text label below
        text_img = Image.fromarray(grid[y0 + obs_size:y0 + cell_h, x0:x0 + cell_w])
        draw = ImageDraw.Draw(text_img)

        if "baseline" in label:
            color = (255, 255, 80)
        elif label.startswith("+"):
            color = (120, 255, 120)
        else:
            color = (255, 120, 120)

        draw.text((4, 2), f"{label}", fill=color, font=font)
        draw.text((4, 16), f"R={euler[0]:.1f}  P={euler[1]:.1f}  Y={euler[2]:.1f}", fill=(200, 200, 200), font=font)
        draw.text((4, 30), f"{axis_name}={euler[euler_idx]:.1f} deg", fill=(255, 255, 255), font=font)
        grid[y0 + obs_size:y0 + cell_h, x0:x0 + cell_w] = np.array(text_img)

    return np.concatenate([title_bar, grid], axis=0)


def build_gif(frames, axis_name, use_axes_overlay, obs_size):
    """Build list of PIL Images for GIF animation."""
    euler_idx = {"Roll": 0, "Pitch": 1, "Yaw": 2}[axis_name]
    pil_frames = []

    for frame in frames:
        (agent_img, wrist_img, agent_axes, wrist_axes,
         euler, cum_steps, label) = frame

        if use_axes_overlay:
            img = agent_axes
        else:
            img = agent_img

        if img.shape[0] != obs_size or img.shape[1] != obs_size:
            img = np.array(Image.fromarray(img).resize((obs_size, obs_size)))

        val = euler[euler_idx]
        bar = make_text_bar(
            [f"{label}  |  R={euler[0]:.1f}  P={euler[1]:.1f}  Y={euler[2]:.1f}",
             f"{axis_name}={val:.1f} deg  |  Axes: X=Red Y=Green Z=Blue"],
            obs_size, color=(255, 255, 255), font_size=12,
        )
        combined = np.concatenate([img, bar], axis=0)
        pil_frames.append(Image.fromarray(combined))

    return pil_frames


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
    print(f"Suite: {args.suite}, task_id: {args.task_id}")
    print(f"Angles per direction: {args.n_angles}, steps per angle: {args.steps_per_angle}")

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file,
                             camera_heights=args.obs_size,
                             camera_widths=args.obs_size)

    axes = [
        (3, "Roll",  "Roll = rotation about X-axis (forward). +Roll: Z toward -Y"),
        (4, "Pitch", "Pitch = rotation about Y-axis (right). +Pitch: Z toward -X"),
        (5, "Yaw",   "Yaw = rotation about Z-axis (up). +Yaw: X toward +Y (CCW from above)"),
    ]

    for axis_idx, axis_name, axis_desc in axes:
        print(f"\n{'='*50}")
        print(f"Sweeping {axis_name} (action dim {axis_idx})...")

        frames = sweep_axis(
            env, init_state, axis_idx, axis_name,
            args.n_angles, args.steps_per_angle, args.speed, args.obs_size,
        )

        euler_i = {"Roll": 0, "Pitch": 1, "Yaw": 2}[axis_name]
        print(f"  {len(frames)} frames captured:")
        for frame in frames:
            euler, label = frame[4], frame[6]
            print(f"    {label:20s} | roll={euler[0]:7.1f}  pitch={euler[1]:7.1f}  yaw={euler[2]:7.1f}")

        # Grid with axes overlay
        grid = build_grid(frames, axis_name, axis_desc, True, args.grid_cols, args.obs_size)
        grid_path = out_dir / f"{axis_name.lower()}_sweep_axes.png"
        Image.fromarray(grid).save(grid_path)
        print(f"  Saved grid (axes): {grid_path}")

        # Grid without overlay (raw)
        grid_raw = build_grid(frames, axis_name, axis_desc, False, args.grid_cols, args.obs_size)
        grid_raw_path = out_dir / f"{axis_name.lower()}_sweep_raw.png"
        Image.fromarray(grid_raw).save(grid_raw_path)
        print(f"  Saved grid (raw):  {grid_raw_path}")

        # GIF with axes
        pil_frames = build_gif(frames, axis_name, True, args.obs_size)
        gif_path = out_dir / f"{axis_name.lower()}_sweep_axes.gif"
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=250, loop=0,
        )
        print(f"  Saved GIF (axes):  {gif_path}")

        # GIF without overlay
        pil_raw = build_gif(frames, axis_name, False, args.obs_size)
        gif_raw_path = out_dir / f"{axis_name.lower()}_sweep_raw.gif"
        pil_raw[0].save(
            gif_raw_path, save_all=True, append_images=pil_raw[1:],
            duration=250, loop=0,
        )
        print(f"  Saved GIF (raw):   {gif_raw_path}")

    env.close()

    print(f"\n{'='*50}")
    print(f"All outputs saved to {out_dir}/")
    print(f"\nAxes color code:")
    print(f"  X = Red   (forward, away from robot)")
    print(f"  Y = Green (right)")
    print(f"  Z = Blue  (up)")
    print(f"\nEuler convention (XYZ intrinsic):")
    print(f"  Roll  = rotation about X. +Roll: Z toward -Y")
    print(f"  Pitch = rotation about Y. +Pitch: Z toward -X")
    print(f"  Yaw   = rotation about Z. +Yaw: X toward +Y")


if __name__ == "__main__":
    main()
