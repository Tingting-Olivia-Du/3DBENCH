#!/usr/bin/env python3
"""T5b grasp-affordance QA — InternData-M1 half.

Grasp contact point = fingertip = flange + R_flange @ [0,0,D_TOOL] at the
gripper-close command edge (-1 -> +1), in base frame -> world via per-episode
base. Calibration (2026-07-08, 395 events): D_TOOL=0.15 m, euler extrinsic
'xyz' -> fingertip inside picked object's bbox3d in 100% of events (median
3.1 cm from object center). Target object = dataset `pick_obj_uid` (sim truth).

Emitted at pre-grasp keyframes (object static, point on/near object box), both
fixed views. Output: qa/<chunk>_t5b.jsonl (separate pool).
"""
import argparse, glob, json, pickle, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from m1_calib import BASE_Z, K_PIXEL_FIXED, world_to_cam_fixed, project
from qa_gen import W, H, NORM, uid_boxes, uid_center3d, norm_box, Gen

D_TOOL = 0.15
EULER = "xyz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--chunk", default="chunk-000")
    args = ap.parse_args()
    gen = Gen(Path(args.data) / "ann" / args.chunk, Path(args.data) / "frames" / args.chunk)

    out_path = Path(args.data) / "qa" / f"{args.chunk}_t5b.jsonl"
    n = 0
    with open(out_path, "w") as fh:
        for fp in sorted(glob.glob(str(Path(args.data) / "ann" / args.chunk / "episode_*.pkl"))):
            rec = pickle.load(open(fp, "rb"))
            if rec["base_solve_rms_px"] > 0.5:
                continue
            uid = rec["pick_obj_uid"]
            name = gen.uid2desc.get(uid)
            if not name:
                continue
            g = rec["gripper_cmd"].astype(float)
            evs = [int(f) for f in np.where(np.diff(g) > 0.5)[0]]
            if not evs:
                continue
            ev = min(evs[0] + 5, rec["length"] - 1)
            base = np.array([rec["base_xy"][0], rec["base_xy"][1], BASE_Z])
            p = rec["eef_pos_base"][ev].astype(float)
            R = Rotation.from_euler(EULER, rec["eef_orn_base"][ev].astype(float)).as_matrix()
            grasp_w = base + p + R @ np.array([0, 0, D_TOOL])
            kf_ev = min(rec["keyframes"], key=lambda k: abs(k - ev))
            c_ev, _ = uid_center3d(rec, kf_ev, uid)
            if c_ev is None:
                continue
            task = gen.tasks.get(rec["task_index"], "")
            for f in [k for k in rec["keyframes"] if k <= ev]:
                c_f, _ = uid_center3d(rec, f, uid)
                if c_f is None or np.linalg.norm(c_f - c_ev) > 0.015:
                    continue
                for view in ("base_view", "base_view_2"):
                    img = gen.img(rec["episode"], view, f)
                    b = uid_boxes(rec, f, view).get(uid)
                    if img is None or b is None:
                        continue
                    R_cv, C = world_to_cam_fixed(view)
                    uv, vis = project(grasp_w, R_cv, C, K_PIXEL_FIXED)
                    if not vis[0]:
                        continue
                    u, v = float(uv[0, 0]), float(uv[0, 1])
                    mx = 0.15 * (b[2] - b[0]) + 8
                    my = 0.15 * (b[3] - b[1]) + 8
                    if not (b[0] - mx <= u <= b[2] + mx and b[1] - my <= v <= b[3] + my):
                        continue
                    q = {"episode": rec["episode"], "task": task, "domain": "m1",
                         "template": "T5b_grasp_point", "frame": f, "view": view,
                         "image": img,
                         "question": f"Task: {task} Point to the exact spot where the "
                                     f"gripper should grasp the {name}. Answer [x,y], "
                                     f"normalized to [0,2000).",
                         "answer": str([round(u / W * NORM, 1), round(v / H * NORM, 1)]),
                         "meta": {"uid": uid, "name": name, "grasp_frame": ev,
                                  "box": norm_box(b)},
                         "qid": f"{rec['episode']}_T5b_{f}_{view}"}
                    fh.write(json.dumps(q, ensure_ascii=False) + "\n")
                    n += 1
    print(f"M1 T5b: {n} items -> {out_path}")


if __name__ == "__main__":
    main()
