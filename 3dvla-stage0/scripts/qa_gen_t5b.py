#!/usr/bin/env python3
"""T5b grasp-affordance QA (LIBERO half): GT = the demo's actual grasp contact
point (TCP grip-site at the gripper-closing event), projected into agentview at
PRE-GRASP keyframes. Unlike T5 (object identification), the answer lands where
the demo grasped — mug handle, bowl rim, bottle neck.

Validity per item: object static between question frame and grasp event
(AABB-center displacement < 1.5 cm) and projected point within the target's
(slightly expanded) box. Output: qa/libero_t5b.jsonl (separate pool — keeps the
main benchmark sampling untouched).
"""
import glob, json, pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from qa_gen_libero import (W, H, NORM, clean_name, grasp_targets, norm_box, project)

DATA = Path("/workspace/tingting/3dvla-data")


def main():
    import argparse, re
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-min", type=int, default=0)
    ap.add_argument("--demo-max", type=int, default=99)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else DATA / "qa" / "libero_t5b.jsonl"
    n = 0
    with open(out_path, "w") as fh:
        for fp in sorted(glob.glob(str(DATA / "libero/ann/*/*.pkl"))):
            di = int(re.search(r"__demo(\d+)\.pkl$", fp).group(1))
            if not (args.demo_min <= di <= args.demo_max):
                continue
            rec = pickle.load(open(fp, "rb"))
            if rec.get("reproj_selfcheck_med_px", 99) > 15:
                continue
            ep = f"{rec['suite']}/{rec['task']}__demo{rec['demo']}"
            frm = DATA / "libero/frames" / rec["suite"] / f"{rec['task']}__demo{rec['demo']}"
            eef = rec["eef_pos_world"].astype(float)
            kfs = rec["keyframes"]
            for ev, bid in grasp_targets(rec):
                kf_ev = min(kfs, key=lambda k: abs(k - ev))
                aabb_ev = rec["per_kf"][kf_ev]["aabb"].get(bid)
                if aabb_ev is None:
                    continue
                c_ev = np.asarray(aabb_ev, float).mean(0)
                grasp_w = eef[min(ev, len(eef) - 1)]  # world grasp contact point
                for f in [k for k in kfs if k <= ev]:
                    entry = rec["per_kf"][f]
                    if f not in rec["cams"].get("agentview", {}):
                        continue
                    aabb_f = entry["aabb"].get(bid)
                    if aabb_f is None:
                        continue
                    if np.linalg.norm(np.asarray(aabb_f, float).mean(0) - c_ev) > 0.015:
                        continue  # object moved since this frame
                    b = entry["boxes"].get("agentview", {}).get(bid)
                    if b is None:
                        continue
                    K = np.asarray(rec["cams"]["agentview"][f]["K"], float)
                    E = np.asarray(rec["cams"]["agentview"][f]["E_cam2world"], float)
                    uv, vis = project(grasp_w, K, E)
                    if not vis[0]:
                        continue
                    u, v = float(uv[0, 0]), float(uv[0, 1])
                    mx, my = 0.15 * (b[2] - b[0]) + 8, 0.15 * (b[3] - b[1]) + 8
                    if not (b[0] - mx <= u <= b[2] + mx and b[1] - my <= v <= b[3] + my):
                        continue  # projected grasp point not on/near the object
                    name = clean_name(entry["names"][bid])
                    img = frm / f"agentview_f{f:04d}.jpg"
                    if not img.exists():
                        continue
                    q = {"episode": ep, "task": rec["language"], "domain": "libero",
                         "template": "T5b_grasp_point", "frame": f, "view": "agentview",
                         "image": str(img),
                         "question": f"Task: {rec['language']} Point to the exact spot "
                                     f"where the gripper should grasp the {name}. "
                                     f"Answer [x,y], normalized to [0,2000).",
                         "answer": str([round(u / W * NORM, 1), round(v / H * NORM, 1)]),
                         "meta": {"bid": int(bid), "name": name, "grasp_frame": ev,
                                  "box": norm_box(b)}}
                    q["qid"] = f"{ep}_T5b_{f}_{ev}"
                    fh.write(json.dumps(q) + "\n")
                    n += 1
    print(f"T5b: {n} items -> {out_path}")


if __name__ == "__main__":
    main()
