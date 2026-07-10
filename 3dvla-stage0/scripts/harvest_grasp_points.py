#!/usr/bin/env python3
"""Harvest grasp-point SETS for T5b evaluation: for every (suite, task), run
kinematics-only pass over ALL 50 demos and record each gripper-closing event's
TCP world position + nearest-object identity. The per-(task, object) point set
approximates the valid-grasp manifold (rim of a bowl gets sampled at many
azimuths by different demos).

Kinematics only (no rendering) — fast. Run in the libero env.
Output: <out>/libero/grasp_sets.json
  {suite/task: {clean_object_name: [[x,y,z], ...]}}
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

LIBERO_ROOT = "/workspace/tingting/LIBERO"
sys.path.insert(0, LIBERO_ROOT)
sys.path.insert(0, str(Path(__file__).parent))
DATA = Path("/workspace/tingting/3dvla-data")


def main():
    from libero.libero import benchmark as bench_mod, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from qa_gen_libero import clean_name

    suite_name = sys.argv[1]
    task_id = int(sys.argv[2])
    suite = bench_mod.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview"],
                             camera_heights=128, camera_widths=128)
    env.reset()
    sim = env.env.sim
    site_id = sim.model.site_name2id("gripper0_grip_site")
    body_names = list(sim.model.body_names)
    from libero_gt_extract import is_scene_object, body_aabb_world

    out = defaultdict(list)
    demo_path = os.path.join(LIBERO_ROOT, "libero/datasets", suite_name, task.name + "_demo.hdf5")
    with h5py.File(demo_path, "r") as h5:
        demos = sorted(int(k.split("_")[1]) for k in h5["data"].keys())
        for di in demos:
            key = f"data/demo_{di}"
            states = h5[key + "/states"][()]
            gq = (h5[key + "/obs/gripper_states"][()][:, 0]
                  if key + "/obs/gripper_states" in h5 else None)
            if gq is None:
                continue
            dg = np.diff(gq)
            thr = max(1e-4, np.abs(dg).max() * 0.3)
            events, last = [], -10 ** 9
            for f in np.where(dg < -thr)[0]:  # closing only
                if f - last > 8:
                    events.append(int(f))
                last = f
            for ev in events:
                sim.set_state_from_flattened(states[ev])
                sim.forward()
                tcp = sim.data.site_xpos[site_id].copy()
                best = None
                for bid, name in enumerate(body_names):
                    if not is_scene_object(name):
                        continue
                    aabb = body_aabb_world(sim, bid)
                    if aabb is None:
                        continue
                    d = float(np.linalg.norm(aabb.mean(0) - tcp))
                    if d <= 0.20 and (best is None or d < best[1]):
                        best = (name, d)
                if best:
                    bid = body_names.index(best[0])
                    aabb = body_aabb_world(sim, bid)
                    R = sim.data.body_xmat[bid].reshape(3, 3)
                    out[clean_name(best[0].replace("_", " "))].append({
                        "tcp": [round(float(x), 4) for x in tcp],
                        "obj_c": [round(float(x), 4) for x in aabb.mean(0)],
                        "obj_R": [round(float(x), 5) for x in R.ravel()]})
    env.close()
    rec = {f"{suite_name}/{task.name}": {k: v for k, v in out.items()}}
    p = DATA / "libero" / "grasp_sets" / f"{suite_name}__{task_id}.json"
    p.parent.mkdir(exist_ok=True)
    json.dump(rec, open(p, "w"))
    print(f"[{suite_name}/{task.name}] objects={list(out)} "
          f"points={sum(len(v) for v in out.values())}")


if __name__ == "__main__":
    main()
