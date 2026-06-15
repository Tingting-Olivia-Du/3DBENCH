"""Visual check: project the NEW geometry-center Z (vs old body-origin Z) onto
the agent-view render so we can confirm the new GT point lands at the object's
visible middle.

For each scene object we draw:
  - RED dot   : old body-origin position (what GT used before)
  - GREEN dot : new geometry-center position (bottom + half-height)
  - cyan tick : object's true geometry bottom
  - magenta tick: object's true geometry top
A correct result: the GREEN dot sits halfway up the object's silhouette,
between the cyan (bottom) and magenta (top) ticks.
"""
import os
import sys

import numpy as np
import h5py
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libero.libero import get_libero_path, benchmark
from libero.libero.envs import OffScreenRenderEnv

from src.bench.gt_extractor import (
    get_all_object_positions,
    compute_object_center_z,
    _descendant_body_ids,
    _geom_world_z_extent,
    is_scene_object_body,
)
from src.bench.gt_extractor import get_agentview_image


def project_to_camera(world_points, sim, cam_name, img_h, img_w):
    """World -> pixel for the flip-corrected agentview image.

    NOTE: scripts/verify_rotation_sweep.py:project_to_camera has an inverted v
    axis (uses +f*y instead of -f*y), so higher world-Z maps to a LOWER row.
    That is a visualization-only bug there; here we use the correct sign so the
    overlay matches the rendered (``[::-1,::-1]`` flip-corrected) image.
    """
    cid = sim.model.camera_name2id(cam_name)
    cpos = sim.data.cam_xpos[cid].copy()
    c2w = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    w2c = c2w.T
    fovy = sim.model.cam_fovy[cid]
    pts = np.atleast_2d(world_points)
    cc = (w2c @ (pts - cpos).T).T
    depth = -cc[:, 2].copy()
    depth[np.abs(depth) < 1e-8] = 1e-8
    f = 0.5 * img_h / np.tan(np.radians(fovy / 2.0))
    u = img_w / 2.0 - f * cc[:, 0] / depth
    v = img_h / 2.0 - f * cc[:, 1] / depth
    return np.stack([u, v], axis=1)


def object_bottom_top_world(sim, body_id):
    """Return (z_bottom, z_top) world Z of an object's full collision geometry."""
    m = sim.model
    zmin, zmax = None, None
    for bid in _descendant_body_ids(sim, body_id):
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != bid or m.geom_group[gid] != 0:
                continue
            ext = _geom_world_z_extent(sim, gid)
            if ext is None:
                continue
            zmin = ext[0] if zmin is None else min(zmin, ext[0])
            zmax = ext[1] if zmax is None else max(zmax, ext[1])
    return zmin, zmax


def main(suite, task_id, demo_idx, frames):
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    dd = get_libero_path("datasets")
    hdf5 = os.path.join(dd, suite, task.bddl_file.replace(".bddl", "_demo.hdf5"))
    h = h5py.File(hdf5, "r")
    demo_key = sorted(h["data"].keys())[demo_idx]
    states = h["data"][demo_key]["states"][:]
    print(f"{suite} task {task_id} '{task.language}' | demo {demo_key} | {len(states)} states")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=512, camera_widths=512)
    env.reset()
    sim = env.env.sim
    bp = sim.data.get_body_xpos("robot0_base").copy()

    out_dir = os.path.join("rollouts", "center_z_check")
    os.makedirs(out_dir, exist_ok=True)

    for fi in frames:
        fi = min(fi, len(states) - 1)
        env.set_init_state(states[fi])
        for _ in range(2):
            sim.forward()
        raw_obs = env.env._get_observations()
        img = get_agentview_image(raw_obs)  # (H,W,3), already flip-corrected
        H, W = img.shape[:2]

        pil = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(pil)

        report = []
        for name in sim.model.body_names:
            if not is_scene_object_body(name) or "mount" in name.lower():
                continue
            if not name.lower().endswith("_main"):
                continue
            bid = sim.model.body_name2id(name)
            world_orig = sim.data.get_body_xpos(name).copy()
            center_z = compute_object_center_z(sim, bid, bp)
            if center_z is None:
                continue
            world_center = world_orig.copy()
            world_center[2] = center_z + bp[2]
            zb, zt = object_bottom_top_world(sim, bid)

            pts_world = np.array([
                world_orig,                                  # old origin
                world_center,                                # new center
                [world_orig[0], world_orig[1], zb],          # bottom
                [world_orig[0], world_orig[1], zt],          # top
            ])
            px = project_to_camera(pts_world, sim, "agentview", H, W)
            (ox, oy), (cx, cy), (bx, by), (tx, ty) = [(int(a), int(b)) for a, b in px]

            zb_rel = zb - bp[2]
            zt_rel = zt - bp[2]

            # bottom/top ticks (short horizontal marks) + Z value labels
            draw.line([(bx - 6, by), (bx + 6, by)], fill=(0, 230, 230), width=2)
            draw.text((bx + 8, by - 5), f"b {zb_rel:+.3f}", fill=(0, 230, 230))
            draw.line([(tx - 6, ty), (tx + 6, ty)], fill=(230, 0, 230), width=2)
            draw.text((tx + 8, ty - 5), f"t {zt_rel:+.3f}", fill=(230, 0, 230))
            # old origin = red, new center = green
            draw.ellipse([(ox - 5, oy - 5), (ox + 5, oy + 5)], outline=(255, 40, 40), width=2)
            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=(40, 255, 40))
            short = name.replace("_1_main", "")
            draw.text((cx + 8, cy - 6), f"{short}  c {center_z:+.3f}", fill=(40, 255, 40))
            report.append(
                f"  {short:18s} bottom={zb_rel:+.3f}  center={center_z:+.3f}  "
                f"top={zt_rel:+.3f}  (old_origin={world_orig[2]-bp[2]:+.3f})"
            )

        # legend
        draw.rectangle([(4, 4), (320, 72)], fill=(20, 20, 20))
        draw.ellipse([(10, 12), (18, 20)], outline=(255, 40, 40), width=2)
        draw.text((24, 10), "old body-origin Z", fill=(255, 255, 255))
        draw.ellipse([(10, 28), (18, 36)], fill=(40, 255, 40))
        draw.text((24, 26), "new geom-center Z", fill=(255, 255, 255))
        draw.line([(10, 48), (18, 48)], fill=(0, 230, 230), width=2)
        draw.line([(120, 48), (128, 48)], fill=(230, 0, 230), width=2)
        draw.text((24, 42), "bottom", fill=(255, 255, 255))
        draw.text((134, 42), "top", fill=(255, 255, 255))
        draw.text((24, 56), "labels: b/c/t = base-relative Z (m)", fill=(200, 200, 200))

        out = os.path.join(out_dir, f"{suite}_t{task_id}_d{demo_idx}_f{fi:04d}.png")
        pil.save(out)
        print(f"frame {fi}: saved {out}")
        for r in report:
            print(r)

    env.close()
    h.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 80, 160])
    a = ap.parse_args()
    main(a.suite, a.task, a.demo, a.frames)
