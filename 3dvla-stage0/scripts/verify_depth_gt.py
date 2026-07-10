#!/usr/bin/env python3
"""Independent verification of T3 depth GT via the renderer's depth buffer.

For each T3_depth_obj / T3_depth_tcp item in spotcheck_libero.csv:
  - replay the exact sim state, render a metric depth map (same renderer that
    produced the RGB — an evidence chain independent of our AABB/extrinsics math)
  - T3_obj: median ray-distance over the object's seg mask vs GT answer.
    Expected |diff| <= object's half-extent (surface vs center) + small margin.
  - T3_tcp: ray distance at the TCP pixel vs GT (gripper surface vs site point).
Report per-item and aggregate. Run in the libero env with MUJOCO_GL=osmesa.
"""
import csv, json, os, pickle, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

LIBERO_ROOT = "/workspace/tingting/LIBERO"
sys.path.insert(0, LIBERO_ROOT)
DATA = Path("/workspace/tingting/3dvla-data")
W, H = 640, 480


def real_depth(env, sim, cam):
    from robosuite.utils.camera_utils import get_real_depth_map
    obs = env.env._get_observations(force_update=True)
    d = obs[f"{cam}_depth"][::-1]  # flip to upright like RGB
    return get_real_depth_map(sim, d)[..., 0] if d.ndim == 3 else get_real_depth_map(sim, d)


def main():
    from libero.libero import benchmark as bench_mod, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (get_camera_intrinsic_matrix,
                                              get_camera_segmentation)

    bench = {json.loads(l)["qid"]: json.loads(l)
             for l in open(DATA / "benchmark" / "libero_bench.jsonl")}
    rows = [r for r in csv.DictReader(open(DATA / "benchmark" / "spotcheck_libero.csv"))
            if "T3_depth" in r["qid"]]
    # group by episode to reuse envs
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[bench[r["qid"]]["episode"]].append(bench[r["qid"]])

    suites = {}
    results = []
    for ep, items in sorted(by_ep.items()):
        suite_name, rest = ep.split("/", 1)
        task_name, demo_s = rest.rsplit("__demo", 1)
        if suite_name not in suites:
            suites[suite_name] = bench_mod.get_benchmark_dict()[suite_name]()
        suite = suites[suite_name]
        task = next(t for t in suite.tasks if t.name == task_name)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_names=["agentview", "robot0_eye_in_hand"],
                                 camera_heights=H, camera_widths=W,
                                 camera_depths=True, camera_segmentations="instance")
        env.reset()
        sim = env.env.sim
        rec = pickle.load(open(DATA / "libero" / "ann" / suite_name / f"{rest}.pkl", "rb"))
        with h5py.File(os.path.join(LIBERO_ROOT, "libero/datasets", suite_name,
                                    task_name + "_demo.hdf5"), "r") as h5:
            states = h5[f"data/demo_{demo_s}/states"][()]
        for q in items:
            f = q["frame"]
            sim.set_state_from_flattened(states[f])
            sim.forward()
            cam = "agentview" if q["view"] == "agentview" else "robot0_eye_in_hand"
            K = get_camera_intrinsic_matrix(sim, cam, H, W)
            depth = real_depth(env, sim, cam)  # upright, z-depth (meters)
            gt = float(q["answer"]) / 100.0
            if q["template"] == "T3_depth_obj":
                # object mask: largest named object used by the generator
                from qa_gen_libero import named_boxes
                entry = rec["per_kf"][f]
                named = [(bid, n) for bid, n, _ in named_boxes(entry, q["view"])
                         if bid in entry["aabb"]]
                if not named:
                    continue
                bid = named[0][0]
                seg = get_camera_segmentation(sim, cam, H, W)[:, :, 1]
                bmap = sim.model.geom_bodyid[np.clip(seg, 0, None)]
                mask = bmap == bid
                for cid in range(sim.model.nbody):
                    if sim.model.body_parentid[cid] == bid:
                        mask |= bmap == cid
                if mask.sum() < 40:
                    continue
                ys, xs = np.where(mask)
                z = depth[ys, xs]
                # z-depth -> ray distance per pixel
                rays = np.sqrt(((xs - K[0, 2]) / K[0, 0]) ** 2 +
                               ((ys - K[1, 2]) / K[1, 1]) ** 2 + 1.0)
                ray_med = float(np.median(z * rays))
                half_ext = float(np.linalg.norm(
                    np.asarray(entry["aabb"][bid]).max(0) -
                    np.asarray(entry["aabb"][bid]).min(0)) / 2)
                results.append({"qid": q["qid"], "t": "obj", "gt_m": gt,
                                "depthbuf_m": ray_med, "diff_cm": (gt - ray_med) * 100,
                                "half_extent_cm": half_ext * 100,
                                "ok": abs(gt - ray_med) <= half_ext + 0.03})
            else:  # T3_depth_tcp
                uv = q["meta"]["tcp_uv"]
                px = int(uv[0] / 2000 * W)
                py = int(uv[1] / 2000 * H)
                patch = depth[max(0, py-3):py+4, max(0, px-3):px+4]
                ray = float(np.sqrt(((px - K[0, 2]) / K[0, 0]) ** 2 +
                                    ((py - K[1, 2]) / K[1, 1]) ** 2 + 1.0))
                zmin = float(np.min(patch)) * ray
                results.append({"qid": q["qid"], "t": "tcp", "gt_m": gt,
                                "depthbuf_m": zmin, "diff_cm": (gt - zmin) * 100,
                                "half_extent_cm": 8.0,
                                "ok": abs(gt - zmin) <= 0.12})
        env.close()
        print(f"[{ep}] checked {len(items)}", flush=True)

    ok = sum(r["ok"] for r in results)
    d = np.array([abs(r["diff_cm"]) for r in results])
    print(f"\nTOTAL {len(results)} items: PASS {ok} ({ok/len(results)*100:.0f}%), "
          f"|GT - depth-buffer| median {np.median(d):.1f}cm p95 {np.percentile(d,95):.1f}cm")
    with open(DATA / "benchmark" / "depth_gt_verification.json", "w") as fh:
        json.dump(results, fh, indent=1)
    for r in [x for x in results if not x["ok"]][:8]:
        print("  FAIL:", r)


if __name__ == "__main__":
    main()
