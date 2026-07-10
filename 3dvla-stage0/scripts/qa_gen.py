#!/usr/bin/env python3
"""QA synthesis from ETL output (ann/*.pkl + frames/*.jpg) — InternData-M1 franka.

Implements plan §3.3 templates on the pinned geometry (m1_calib):
  T1 grounding            "locate the {name}" -> box [0,2000)
  T2 object metric 3D     distance / size between pick & place objects (meters)
  T3 metric depth         camera->object / camera->TCP distance (meters)
  T4 TCP 2D-trace         future TCP waypoints projected into the view
  T5 where-to-act         pick/place target -> point
  T6 cross-view           box in base_view -> box in base_view_2
  T7 ego-motion           wrist camera translation/rotation between two keyframes

Conventions: pixel coords normalized to [0,2000) on both axes (resize-invariant);
metric answers in meters/centimeters as stated per template; camera frame = OpenCV.
Output: jsonl records {qid, template, episode, frame, view, image[, image2], question,
answer, meta}. Deterministic given (episode, seed).
"""
import argparse, glob, json, pickle, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m1_calib import (K_PIXEL_FIXED, M_OPT, WristChain, project,
                      world_to_cam_fixed, quat_to_R_wxyz, BASE_Z)
# All projections here land on RGB pixels -> use K_PIXEL_* (see m1_calib dual-K note)
K_FIXED = K_PIXEL_FIXED

W, H = 640, 480
NORM = 2000.0
FIXED_VIEWS = ("base_view", "base_view_2")
MIN_SIDE = 20          # px; boxes smaller than this are visually unreliable targets
MAX_T1_PER_VIEW = 4    # cap grounding QAs per (frame, view)
MAX_T6_PER_FRAME = 3   # cap cross-view QAs per frame

# Official GenManip asset semantics: uid -> {short_description, caption, color, ...}
# (visually grounded GPT-4V annotations; `category` field is noisier — do not use).
OBJAVERSE_ANN = "/workspace/tingting/GenManip/assets/objects/objaverse_annotation_refined_pnp.pickle"

_TASK_RE = re.compile(r"^move the (.+?) to (?:the top of )?the (.+?)\.?$", re.I)


def norm_uv(uv):
    return [round(float(u) / W * NORM, 1) for u in np.atleast_1d(uv[0:1])] + \
           [round(float(v) / H * NORM, 1) for v in np.atleast_1d(uv[1:2])]


def norm_box(b):
    return [round(b[0] / W * NORM, 1), round(b[1] / H * NORM, 1),
            round(b[2] / W * NORM, 1), round(b[3] / H * NORM, 1)]


def parse_names(task):
    m = _TASK_RE.match(task.strip())
    return (m.group(1), m.group(2)) if m else (None, None)


def uid_boxes(rec, f, view):
    """uid -> tight box [x0,y0,x1,y1] for keyframe f in view."""
    d = rec["per_kf"][f]
    arr = d[f"{view}.bbox2d_tight"]
    id2l = d[f"{view}.bbox2d_tight_id2labels"] or {}
    sem2uid = {int(k): v["class"] for k, v in id2l.items()}
    out = {}
    if arr is None:
        return out
    for row in arr:
        uid = sem2uid.get(int(row["semanticId"]))
        if uid is None or uid == "defaultgroundplane":
            continue
        x0, y0, x1, y1 = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
            continue
        out[uid] = [x0, y0, x1, y1]
    return out


def uid_center3d(rec, f, uid):
    """Match bbox3d entries by their own `class` field (= semanticId) — the entry
    list order does NOT follow the id2labels key order (order-based matching
    silently picked the wrong object in some episodes)."""
    d = rec["per_kf"][f]
    id2l = d["bbox3d_id2labels"] or {}
    sem2uid = {int(k): v["class"] for k, v in id2l.items()}
    for ob in (d["bbox3d"] or []):
        sem = int(np.asarray(ob["class"]).item())
        if sem2uid.get(sem) == uid:
            corners = np.asarray(ob["corners"], float).reshape(8, 3)
            return corners.mean(0), corners
    return None, None


class Gen:
    def __init__(self, ann_dir, frames_dir, seed=0):
        self.ann_dir, self.frames_dir = Path(ann_dir), Path(frames_dir)
        self.rng = np.random.default_rng(seed)
        self.wc = WristChain()
        with open("/workspace/tingting/Datasets/InternData-M1/simulated/franka/meta/tasks.jsonl") as fh:
            self.tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in fh}
        with open(OBJAVERSE_ANN, "rb") as fh:
            ann = pickle.load(fh)
        # visually-grounded name per asset; drop leading article for prompt slotting
        self.uid2desc = {}
        for uid, e in ann.items():
            name = (e.get("short_description") or e.get("caption") or "").strip().rstrip(".")
            if name:
                self.uid2desc[uid] = re.sub(r"^(a|an|the)\s+", "", name, flags=re.I)

    def named_boxes(self, rec, f, view, task_names):
        """Visible boxes with unambiguous names: official short_description first,
        task-parsed name as fallback for pick/place. Objects whose name appears
        more than once among visible boxes are dropped (grounding would be
        ambiguous). Returns [(uid, name, box)] sorted by area desc."""
        boxes = uid_boxes(rec, f, view)
        cand = {}
        for uid, b in boxes.items():
            name = self.uid2desc.get(uid) or task_names.get(uid)
            if name:
                cand[uid] = (name.lower(), b)
        counts = {}
        for name, _ in cand.values():
            counts[name] = counts.get(name, 0) + 1
        out = [(uid, name, b) for uid, (name, b) in cand.items() if counts[name] == 1]
        out.sort(key=lambda t: -(t[2][2] - t[2][0]) * (t[2][3] - t[2][1]))
        return out

    def img(self, ep, view, f):
        p = self.frames_dir / ep / f"{view}_f{f:04d}.jpg"
        return str(p) if p.exists() else None

    def episode_qas(self, pkl_path):
        rec = pickle.load(open(pkl_path, "rb"))
        if rec["base_solve_rms_px"] > 0.5:  # bad base calibration -> geometry untrustworthy
            return []
        ep = rec["episode"]
        task = self.tasks.get(rec["task_index"], "")
        tp_pick, tp_place = parse_names(task)
        task_names = {}
        if tp_pick and rec["pick_obj_uid"]:
            task_names[rec["pick_obj_uid"]] = tp_pick
        if tp_place and rec["place_obj_uid"]:
            task_names[rec["place_obj_uid"]] = tp_place
        # QA-facing names: official visually-grounded description > task noun
        pick_name = self.uid2desc.get(rec["pick_obj_uid"]) or tp_pick
        place_name = self.uid2desc.get(rec["place_obj_uid"]) or tp_place
        base_xy = rec["base_xy"]
        T_wb = np.eye(4); T_wb[:3, 3] = [base_xy[0], base_xy[1], BASE_Z]
        eef_w = rec["eef_pos_base"].astype(float) @ T_wb[:3, :3].T + T_wb[:3, 3]

        out = []
        emit = lambda **kw: out.append({"episode": ep, "task": task, **kw})
        kfs = rec["keyframes"]

        for f in kfs:
            for view in FIXED_VIEWS:
                img = self.img(ep, view, f)
                if img is None:
                    continue
                R_cv, C = world_to_cam_fixed(view)
                boxes = uid_boxes(rec, f, view)

                # T1 grounding (all unambiguously-named visible objects, biggest first)
                named = self.named_boxes(rec, f, view, task_names)
                for uid, name, b in named[:MAX_T1_PER_VIEW]:
                    emit(template="T1_grounding", frame=f, view=view, image=img,
                         question=f"Locate the {name} in the image. Answer with a bounding "
                                  f"box [x0,y0,x1,y1], coordinates normalized to [0,2000).",
                         answer=str(norm_box(b)), meta={"uid": uid, "name_src":
                                  "official" if uid in self.uid2desc else "task"})
                if rec["pick_obj_uid"] in boxes:
                    b = boxes[rec["pick_obj_uid"]]
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    emit(template="T5_where_to_act", frame=f, view=view, image=img,
                         question=f"Task: {task} Point to the object that should be picked up. "
                                  f"Answer [x,y], normalized to [0,2000).",
                         answer=str([round(cx / W * NORM, 1), round(cy / H * NORM, 1)]),
                         meta={"uid": rec["pick_obj_uid"]})

                # T2 object-level metric 3D (pick <-> place)
                c_pick, _ = uid_center3d(rec, f, rec["pick_obj_uid"])
                c_place, _ = uid_center3d(rec, f, rec["place_obj_uid"])
                if pick_name and place_name and c_pick is not None and c_place is not None \
                        and rec["pick_obj_uid"] in boxes and rec["place_obj_uid"] in boxes:
                    dist = float(np.linalg.norm(c_pick - c_place))
                    emit(template="T2_metric3d_dist", frame=f, view=view, image=img,
                         question=f"What is the 3D distance between the center of the {pick_name} "
                                  f"and the center of the {place_name}, in centimeters?",
                         answer=f"{dist*100:.1f}", meta={})

                # T3 metric depth: camera->object, camera->TCP
                if pick_name and c_pick is not None and rec["pick_obj_uid"] in boxes:
                    emit(template="T3_depth_obj", frame=f, view=view, image=img,
                         question=f"How far is the center of the {pick_name} from the camera, "
                                  f"in centimeters?",
                         answer=f"{float(np.linalg.norm(c_pick - C))*100:.1f}", meta={})
                uv_tcp, vis = project(eef_w[f], R_cv, C, K_FIXED)
                if vis[0] and 0 <= uv_tcp[0, 0] < W and 0 <= uv_tcp[0, 1] < H:
                    emit(template="T3_depth_tcp", frame=f, view=view, image=img,
                         question="How far is the robot wrist (gripper flange) from the camera, "
                                  "in centimeters?",
                         answer=f"{float(np.linalg.norm(eef_w[f] - C))*100:.1f}",
                         meta={"tcp_uv": norm_uv(uv_tcp[0])})

                # T4 TCP 2D-trace (future waypoints).
                # Require the CURRENT TCP in view (else the prediction is unanswerable
                # from pixels — e.g. home pose projects above base_view's top edge),
                # and truncate the trace at the first out-of-view waypoint.
                uv_now, vis_now = project(eef_w[f], R_cv, C, K_FIXED)
                tcp_in_view = bool(vis_now[0]) and 0 <= uv_now[0, 0] < W and 0 <= uv_now[0, 1] < H
                deltas = (10, 20, 30, 45, 60)
                fut = [f + d for d in deltas if f + d < rec["length"]]
                if tcp_in_view and len(fut) >= 3:
                    uvs, vs = project(eef_w[fut], R_cv, C, K_FIXED)
                    keep = 0
                    for i in range(len(fut)):
                        if vs[i] and 0 <= uvs[i, 0] < W and 0 <= uvs[i, 1] < H:
                            keep += 1
                        else:
                            break
                    if keep >= 3:
                        pts = [[round(float(u) / W * NORM, 1), round(float(v) / H * NORM, 1)]
                               for u, v in uvs[:keep]]
                        steps = ", ".join(str(d) for d in deltas[:keep])
                        emit(template="T4_tcp_trace", frame=f, view=view, image=img,
                             question=f"Task: {task} Predict the robot wrist's future 2D trajectory "
                                      f"in this view: its position after {steps} frames (30 fps), "
                                      f"as [x,y] normalized to [0,2000).",
                             answer=str(pts), meta={"future_frames": fut[:keep]})

            # T6 cross-view (base_view -> base_view_2), all named objects visible in both
            named1 = {u: (n, b) for u, n, b in self.named_boxes(rec, f, "base_view", task_names)}
            named2 = {u: (n, b) for u, n, b in self.named_boxes(rec, f, "base_view_2", task_names)}
            img1, img2 = self.img(ep, "base_view", f), self.img(ep, "base_view_2", f)
            if img1 and img2:
                common = [u for u in named1 if u in named2][:MAX_T6_PER_FRAME]
                for uid in common:
                    name, bb1 = named1[uid]
                    _, bb2 = named2[uid]
                    emit(template="T6_cross_view", frame=f, view="base_view->base_view_2",
                         image=img1, image2=img2,
                         question=f"The {name} is at {norm_box(bb1)} in the first view "
                                  f"([x0,y0,x1,y1], normalized [0,2000)). Give its bounding "
                                  f"box in the second view, same format.",
                         answer=str(norm_box(bb2)), meta={"uid": uid})

        # T7 ego-motion between keyframe pairs with gap in [10, 45]
        for a, b in zip(kfs[:-1], kfs[1:]):
            gap = b - a
            if not (10 <= gap <= 45):
                continue
            ia, ib = self.img(ep, "ego_view", a), self.img(ep, "ego_view", b)
            if not ia or not ib:
                continue
            Ta = self.wc.cam_pose_world(rec["joints"][a].astype(float), base_xy)
            Tb = self.wc.cam_pose_world(rec["joints"][b].astype(float), base_xy)
            # relative motion in OpenCV optical frame of camera a
            Ra_cv = M_OPT @ Ta[:3, :3].T
            t_rel = Ra_cv @ (Tb[:3, 3] - Ta[:3, 3])
            R_rel = (M_OPT @ Tb[:3, :3].T) @ (M_OPT @ Ta[:3, :3].T).T
            dist = float(np.linalg.norm(t_rel))
            if dist < 0.01 and self.rng.random() > 0.1:
                continue
            yaw = np.degrees(np.arctan2(R_rel[0, 2], R_rel[2, 2]))
            pitch = np.degrees(np.arcsin(np.clip(-R_rel[1, 2], -1, 1)))
            roll = np.degrees(np.arctan2(R_rel[1, 0], R_rel[1, 1]))
            dirs = []
            for val, pos, neg in ((t_rel[0], "right", "left"),
                                  (t_rel[1], "down", "up"),
                                  (t_rel[2], "forward", "backward")):
                if abs(val) > 0.005:
                    dirs.append(f"{pos if val > 0 else neg} {abs(val)*100:.1f}")
            move = "; ".join(dirs) if dirs else "no significant translation"
            emit(template="T7_ego_motion", frame=a, view="ego_view", image=ia, image2=ib,
                 question=f"These are two wrist-camera frames {gap} frames apart (30 fps). "
                          f"Describe the camera's motion in its own optical frame, where forward "
                          f"means the direction the camera is looking (note: for a downward-looking "
                          f"camera, rising in the world means moving backward): translation per axis "
                          f"in centimeters (right/left, up/down, forward/backward) and rotation "
                          f"as yaw/pitch/roll in degrees.",
                 answer=f"translation: {move} cm; rotation: yaw {yaw:.0f}, pitch {pitch:.0f}, "
                        f"roll {roll:.0f} deg",
                 meta={"gap": gap, "t_rel_cm": [round(x * 100, 2) for x in t_rel],
                       "dist_cm": round(dist * 100, 2)})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--chunk", default="chunk-000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ann = Path(args.data) / "ann" / args.chunk
    frames = Path(args.data) / "frames" / args.chunk
    gen = Gen(ann, frames)
    files = sorted(ann.glob("episode_*.pkl"))
    if args.limit:
        files = files[: args.limit]
    out_path = args.out or str(Path(args.data) / "qa" / f"{args.chunk}.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out_path, "w") as fh:
        for i, fp in enumerate(files):
            for q in gen.episode_qas(fp):
                q["qid"] = f"{q['episode']}_{q['template']}_{q['frame']}_{q.get('view','')}"
                counts[q["template"]] = counts.get(q["template"], 0) + 1
                fh.write(json.dumps(q, default=str) + "\n")
    print(f"wrote {sum(counts.values())} QAs from {len(files)} episodes -> {out_path}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
