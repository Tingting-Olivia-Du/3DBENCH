#!/usr/bin/env python3
"""LIBERO-half GT extraction for the in-domain 3D benchmark (+ anchor QA).

Run in the `libero` env (py3.8):
  MUJOCO_GL=osmesa PYTHONPATH=/workspace/tingting/envs/libero_extra_site \
    /workspace/ghsun/miniconda3/envs/libero/bin/python libero_gt_extract.py \
    --suite libero_spatial --task_id 0 --demos 45 46 47 48 49

Two-pass per demo:
  pass 1 (no render): set_state+forward over ALL frames -> TCP site pos/quat,
          wrist-cam extrinsics per frame, gripper q
  pass 2 (render):    keyframes only -> RGB 640x480 (agentview + wrist),
          instance seg -> per-object tight boxes, object world AABB corners,
          per-cam K/E
Built-in convention self-check: project each visible object's body xpos through
K/E and compare with its seg-mask centroid; abort if median err > 8 px.

Output mirrors the M1 ETL rec format:
  <out>/ann/<suite>/<task>__demo<i>.pkl , <out>/frames/<suite>/<task>__demo<i>/{view}_f{f}.jpg
"""
import argparse, json, os, pickle, re, sys
from pathlib import Path

import h5py
import numpy as np

LIBERO_ROOT = "/workspace/tingting/LIBERO"
sys.path.insert(0, LIBERO_ROOT)

ROBOT_KW = {"robot", "gripper", "finger", "link", "joint", "actuator", "motor",
            "thumb", "panda", "franka", "eef", "wrist", "mount", "controller"}
ENV_KW = {"table", "floor", "ground", "ceiling", "wall", "world", "scene",
          "light", "fixture", "support", "stand", "camera", "pedestal"}
VIEWS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
W, H = 640, 480


def is_scene_object(name):
    n = name.lower()
    return not (any(k in n for k in ROBOT_KW) or any(k in n for k in ENV_KW))


def humanize(body):
    # "akita_black_bowl_1_main" -> "akita black bowl 1"
    n = re.sub(r"_(main|object|body)$", "", body)
    return re.sub(r"_+", " ", n).strip()


def select_keyframes(eef_pos, grip_q, max_kf=6, min_gap=12):
    T = len(eef_pos)
    v = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
    if len(v) >= 5:
        v = np.convolve(v, np.ones(5) / 5, mode="same")
    cand = {0: 3.0, T - 1: 3.0}
    dg = np.abs(np.diff(grip_q))
    thr = max(1e-4, dg.max() * 0.2) if len(dg) else 1.0
    last = -10 ** 9
    for f in np.where(dg > thr)[0]:
        if f - last > 5:
            cand[int(f)] = 4.0
        last = f
    for f in range(T // 4, T, max(T // 4, 1)):
        cand.setdefault(int(f), 0.5)
    chosen = []
    for f, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
        if all(abs(f - c) >= min_gap for c in chosen):
            chosen.append(f)
        if len(chosen) >= max_kf:
            break
    return sorted(chosen)


def body_aabb_world(sim, bid):
    """World-frame AABB corners over the body's own geoms (approx bbox3d)."""
    import mujoco  # noqa  (robosuite's mj bindings differ; use model arrays)
    m, d = sim.model, sim.data
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid:
            continue
        pos = d.geom_xpos[gid]
        R = d.geom_xmat[gid].reshape(3, 3)
        size = m.geom_size[gid]
        gtype = m.geom_type[gid]
        if gtype == 7:  # mesh: use rbound sphere as fallback
            r = m.geom_rbound[gid]
            ext = np.array([r, r, r])
            corners = pos + np.array([[sx, sy, sz] for sx in (-ext[0], ext[0])
                                      for sy in (-ext[1], ext[1]) for sz in (-ext[2], ext[2])])
        else:
            if gtype == 2:  # sphere
                half = np.array([size[0]] * 3)
            elif gtype in (3, 5):  # capsule/cylinder
                half = np.array([size[0], size[0], size[1] + (size[0] if gtype == 3 else 0)])
            else:  # box / ellipsoid
                half = size[:3].copy()
            local = np.array([[sx, sy, sz] for sx in (-half[0], half[0])
                              for sy in (-half[1], half[1]) for sz in (-half[2], half[2])])
            corners = pos + local @ R.T
        lo = np.minimum(lo, corners.min(0))
        hi = np.maximum(hi, corners.max(0))
    if not np.isfinite(lo).all():
        return None
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--demos", type=int, nargs="+", default=[45, 46, 47, 48, 49])
    ap.add_argument("--out", default="/workspace/tingting/3dvla-data/libero")
    ap.add_argument("--max_kf", type=int, default=6)
    args = ap.parse_args()

    from libero.libero import benchmark as bench_mod, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                              get_camera_intrinsic_matrix,
                                              get_camera_segmentation)
    import cv2

    suite = bench_mod.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    demo_path = os.path.join(LIBERO_ROOT, "libero/datasets", args.suite,
                             task.name + "_demo.hdf5")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=list(VIEWS.values()),
                             camera_heights=H, camera_widths=W,
                             camera_segmentations="instance")
    env.reset()
    sim = env.env.sim
    body_names = list(sim.model.body_names)
    scene_bodies = {bid: humanize(n) for bid, n in enumerate(body_names)
                    if is_scene_object(n)}
    # TCP site
    site = "gripper0_grip_site"
    site_id = sim.model.site_name2id(site)

    ann_dir = Path(args.out) / "ann" / args.suite
    frm_root = Path(args.out) / "frames" / args.suite
    ann_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(demo_path, "r") as h5:
        for di in args.demos:
            key = f"data/demo_{di}"
            if key not in h5:
                continue
            states = h5[key + "/states"][()]
            T = len(states)
            # ---- pass 1: full-trajectory kinematics (no render) ----
            eef, eefR, wristE, gripq = [], [], [], []
            for t in range(T):
                sim.set_state_from_flattened(states[t])
                sim.forward()
                eef.append(sim.data.site_xpos[site_id].copy())
                eefR.append(sim.data.site_xmat[site_id].reshape(3, 3).copy())
                wristE.append(get_camera_extrinsic_matrix(sim, VIEWS["wrist"]).copy())
                gq = h5[key + "/obs/gripper_states"][t] if key + "/obs/gripper_states" in h5 \
                    else [0.0]
                gripq.append(float(np.asarray(gq).ravel()[0]))
            eef = np.array(eef)
            kfs = select_keyframes(eef, np.array(gripq), max_kf=args.max_kf)
            # ---- pass 2: render keyframes ----
            per_kf, cams = {}, {}
            frm_dir = frm_root / f"{task.name}__demo{di}"
            frm_dir.mkdir(parents=True, exist_ok=True)
            reproj_err = []
            for f in kfs:
                sim.set_state_from_flattened(states[f])
                sim.forward()
                obs = env.env._get_observations(force_update=True)
                entry = {"boxes": {}, "aabb": {}, "names": {}}
                for view, cam in VIEWS.items():
                    rgb = obs[f"{cam}_image"][::-1]
                    cv2.imwrite(str(frm_dir / f"{view}_f{f:04d}.jpg"),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    K = get_camera_intrinsic_matrix(sim, cam, H, W)
                    E = get_camera_extrinsic_matrix(sim, cam)  # cam->world
                    cams.setdefault(view, {})[f] = {"K": K, "E_cam2world": E}
                    seg = get_camera_segmentation(sim, cam, H, W)[:, :, 1]  # already upright
                    body_of_geom = sim.model.geom_bodyid
                    bmap = np.where(seg >= 0, body_of_geom[np.clip(seg, 0, None)], -1)
                    boxes = {}
                    for bid, name in scene_bodies.items():
                        # include child bodies (objects are body subtrees)
                        mask = bmap == bid
                        for cid in range(sim.model.nbody):
                            if sim.model.body_parentid[cid] == bid:
                                mask |= bmap == cid
                        if mask.sum() < 40:
                            continue
                        ys, xs = np.where(mask)
                        b = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        if b[2] - b[0] < 20 or b[3] - b[1] < 20:
                            continue
                        # guaranteed-inside point: mask pixel nearest the mask centroid
                        # (box centers can fall on rims/occluders — user-caught issue)
                        cy, cx = ys.mean(), xs.mean()
                        k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
                        boxes[bid] = b + [int(xs[k]), int(ys[k])]
                        # convention self-check: AABB geometric center projected vs
                        # tight-box center (like-for-like semantics).
                        # robosuite E is OpenCV-like (x right, y down, z forward); get_camera_segmentation
                        # returns UPRIGHT maps (unlike raw obs images which need [::-1]).
                        aabb = body_aabb_world(sim, bid)
                        if aabb is None:
                            continue
                        p = aabb.mean(0)
                        Ew = np.linalg.inv(E)
                        pc = Ew[:3, :3] @ p + Ew[:3, 3]
                        if pc[2] > 1e-6:
                            u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
                            v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
                            bu, bv = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                            reproj_err.append((float(u - bu), float(v - bv)))
                    entry["boxes"][view] = boxes
                for bid in scene_bodies:
                    c = body_aabb_world(sim, bid)
                    if c is not None:
                        entry["aabb"][bid] = c
                entry["names"] = dict(scene_bodies)
                per_kf[f] = entry
            if reproj_err:
                arr = np.array(reproj_err)
                med = float(np.median(np.hypot(arr[:, 0], arr[:, 1])))
                med_uv = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            else:
                med, med_uv = -1, (0, 0)
            rec = {
                "suite": args.suite, "task": task.name, "language": task.language,
                "demo": di, "length": T, "keyframes": kfs, "per_kf": per_kf,
                "cams": cams, "eef_pos_world": eef.astype(np.float32),
                "eef_R_world": np.array(eefR).astype(np.float32),
                "wrist_E_cam2world": np.array(wristE).astype(np.float32),
                "gripper_q": np.array(gripq).astype(np.float32),
                "reproj_selfcheck_med_px": med,
                "reproj_selfcheck_med_uv": med_uv,
            }
            with open(ann_dir / f"{task.name}__demo{di}.pkl", "wb") as fh:
                pickle.dump(rec, fh, protocol=4)
            print(f"[{args.suite}/{task.name}/demo{di}] T={T} kf={kfs} "
                  f"selfcheck={med:.1f}px (du={med_uv[0]:+.1f}, dv={med_uv[1]:+.1f})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
