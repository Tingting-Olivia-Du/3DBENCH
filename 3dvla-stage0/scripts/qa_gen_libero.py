#!/usr/bin/env python3
"""QA synthesis for the LIBERO half — consumes libero_gt_extract.py recs.

Same 7 templates and conventions as qa_gen.py (M1 half): pixel coords normalized
[0,2000) (top-left origin), metric in cm, T7 in the wrist camera's OpenCV optical
frame. Differences: names from de-instanced MuJoCo body names (ambiguity guard:
duplicate base names are skipped); T5 only when the task language uniquely matches
one scene object; T6 is agentview<->wrist.

Runs in the plain python3 env (no sim needed — pure geometry on the recs).
"""
import argparse, glob, json, pickle, re
from pathlib import Path

import numpy as np

W, H, NORM = 640, 480, 2000.0
MIN_SIDE = 20
MAX_T1_PER_VIEW = 4
MAX_T6_PER_FRAME = 3


def norm_box(b):
    return [round(b[0] / W * NORM, 1), round(b[1] / H * NORM, 1),
            round(b[2] / W * NORM, 1), round(b[3] / H * NORM, 1)]


def clean_name(name):
    """'akita black bowl 1' -> 'akita black bowl'; 'wooden cabinet 1 cabinet top'
    -> 'wooden cabinet top'."""
    n = re.sub(r"\b\d+\b", " ", name)
    n = re.sub(r"\bcabinet cabinet\b", "cabinet", re.sub(r"\s+", " ", n)).strip()
    return n


def project(p_world, K, E_cam2world):
    Ew = np.linalg.inv(E_cam2world)
    pc = np.atleast_2d(p_world) @ Ew[:3, :3].T + Ew[:3, 3]
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.where(z > 1e-6, z, 1) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.where(z > 1e-6, z, 1) + K[1, 2]
    return np.stack([u, v], 1), z > 1e-6


def named_boxes(entry, view):
    """[(bid, clean_name, box)] with duplicate-name and size guards, area-desc."""
    boxes = entry["boxes"].get(view, {})
    cand = {}
    for bid, b in boxes.items():
        if b[2] - b[0] < MIN_SIDE or b[3] - b[1] < MIN_SIDE:
            continue
        cand[bid] = (clean_name(entry["names"][bid]), b)
    counts = {}
    for n, _ in cand.values():
        counts[n] = counts.get(n, 0) + 1
    out = [(bid, n, b) for bid, (n, b) in cand.items() if counts[n] == 1]
    out.sort(key=lambda t: -(t[2][2] - t[2][0]) * (t[2][3] - t[2][1]))
    return out


_PICK_RES = [
    re.compile(r"pick up the (.+?)(?: and | from | next | between | on | in |$)"),
    re.compile(r"(?:put|place) the (.+?) (?:in|into|on|onto|under) "),
    re.compile(r"(?:open|close|turn on|turn off|push|pull) the (.+?)(?: and |$)"),
]


def match_task_object(language, entry):
    """Unique scene object matching the MANIPULATED-object phrase of the task
    language (not the whole sentence — otherwise the *place target* can match,
    e.g. 'pick up the black bowl ... place it on the plate' with two identical
    bowls would wrongly resolve to the plate). None unless exactly one object's
    content words all appear in the extracted pick-phrase."""
    lang = language.lower()
    phrase = None
    for rx in _PICK_RES:
        m = rx.search(lang)
        if m:
            phrase = m.group(1)
            break
    if not phrase:
        return None
    hits = []
    seen = set()
    for bid, raw in entry["names"].items():
        n = clean_name(raw)
        if n in seen:
            continue
        seen.add(n)
        words = [w for w in n.split() if len(w) > 2]
        if words and all(w in phrase or w in ("the",) for w in words):
            hits.append((bid, n))
    return hits[0] if len(hits) == 1 else None


def rel_motion_text(Ea, Eb, gap):
    """Camera motion between two wrist poses, in camera-a's optical frame."""
    Ra, ta = Ea[:3, :3], Ea[:3, 3]
    Rb, tb = Eb[:3, :3], Eb[:3, 3]
    t_rel = Ra.T @ (tb - ta)
    R_rel = Rb.T @ Ra
    yaw = np.degrees(np.arctan2(R_rel[0, 2], R_rel[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-R_rel[1, 2], -1, 1)))
    roll = np.degrees(np.arctan2(R_rel[1, 0], R_rel[1, 1]))
    dirs = []
    for val, pos, neg in ((t_rel[0], "right", "left"), (t_rel[1], "down", "up"),
                          (t_rel[2], "forward", "backward")):
        if abs(val) > 0.005:
            dirs.append(f"{pos if val > 0 else neg} {abs(val)*100:.1f}")
    move = "; ".join(dirs) if dirs else "no significant translation"
    ans = (f"translation: {move} cm; rotation: yaw {yaw:.0f}, pitch {pitch:.0f}, "
           f"roll {roll:.0f} deg")
    return ans, t_rel, float(np.linalg.norm(t_rel))


def grasp_targets(rec):
    """Sim-grounded manipulation targets: at each gripper-closing event, the scene
    object whose AABB center is nearest the TCP (<=20cm). Replaces text matching —
    covers relational referring ('the bowl between the plate and the ramekin')
    that language matching must skip, and multi-step tasks get per-phase targets.
    Returns [(event_frame, bid)] sorted by frame."""
    g = np.asarray(rec["gripper_q"], float)
    eef = rec["eef_pos_world"].astype(float)
    dg = np.diff(g)
    thr = max(1e-4, np.abs(dg).max() * 0.3)
    events, last = [], -10**9
    for f in np.where(dg < -thr)[0]:  # closing only: grasp semantics
        if f - last > 8:
            events.append(int(f))
        last = f
    kfs = rec["keyframes"]
    out = []
    for ev in events:
        kf = min(kfs, key=lambda k: abs(k - ev))  # nearest keyframe has AABBs
        entry = rec["per_kf"][kf]
        best = None
        for bid, aabb in entry["aabb"].items():
            c = np.asarray(aabb, float).mean(0)
            d_ = float(np.linalg.norm(c - eef[min(ev, len(eef) - 1)]))
            if d_ <= 0.20 and (best is None or d_ < best[1]):
                best = (bid, d_)
        if best:
            out.append((ev, best[0]))
    return out


def episode_qas(rec, frames_root, rng):
    ep = f"{rec['suite']}/{rec['task']}__demo{rec['demo']}"
    task = rec["language"]
    frm = Path(frames_root) / rec["suite"] / f"{rec['task']}__demo{rec['demo']}"
    out = []
    emit = lambda **kw: out.append({"episode": ep, "task": task, "domain": "libero", **kw})
    kfs = rec["keyframes"]
    eef = rec["eef_pos_world"].astype(float)
    targets = grasp_targets(rec)  # [(event_frame, bid)]

    for f in kfs:
        entry = rec["per_kf"][f]
        for view in ("agentview", "wrist"):
            img = frm / f"{view}_f{f:04d}.jpg"
            if not img.exists() or f not in rec["cams"].get(view, {}):
                continue
            img = str(img)
            K = np.asarray(rec["cams"][view][f]["K"], float)
            E = np.asarray(rec["cams"][view][f]["E_cam2world"], float)
            C = E[:3, 3]
            named = named_boxes(entry, view)
            for bid, name, b in named[:MAX_T1_PER_VIEW]:
                emit(template="T1_grounding", frame=f, view=view, image=img,
                     question=f"Locate the {name} in the image. Answer with a bounding "
                              f"box [x0,y0,x1,y1], coordinates normalized to [0,2000).",
                     answer=str(norm_box(b)), meta={"bid": int(bid)})
            # T2: two largest unique-named objects with AABBs
            withc = [(bid, n) for bid, n, _ in named if bid in entry["aabb"]]
            if len(withc) >= 2:
                (b1, n1), (b2, n2) = withc[0], withc[1]
                c1 = np.asarray(entry["aabb"][b1], float).mean(0)
                c2 = np.asarray(entry["aabb"][b2], float).mean(0)
                emit(template="T2_metric3d_dist", frame=f, view=view, image=img,
                     question=f"What is the 3D distance between the center of the {n1} "
                              f"and the center of the {n2}, in centimeters?",
                     answer=f"{np.linalg.norm(c1-c2)*100:.1f}", meta={})
                emit(template="T3_depth_obj", frame=f, view=view, image=img,
                     question=f"How far is the center of the {n1} from the camera, "
                              f"in centimeters?",
                     answer=f"{np.linalg.norm(c1-C)*100:.1f}", meta={})
            # T3 tcp + T4 trace
            uv, vis = project(eef[f], K, E)
            # T3_depth_tcp only from the fixed camera: wrist-cam->TCP distance is a
            # near-constant mount property, answerable without looking at the image.
            if view == "agentview" and vis[0] and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
                emit(template="T3_depth_tcp", frame=f, view=view, image=img,
                     question="How far is the robot gripper (TCP) from the camera, "
                              "in centimeters?",
                     answer=f"{np.linalg.norm(eef[f]-C)*100:.1f}",
                     meta={"tcp_uv": [round(uv[0,0]/W*NORM,1), round(uv[0,1]/H*NORM,1)]})
                if view == "agentview":
                    deltas = (5, 10, 15, 22, 30)
                    fut = [f + d for d in deltas if f + d < rec["length"]]
                    if len(fut) >= 3:
                        uvs, vs = project(eef[fut], K, E)
                        keep = 0
                        for i in range(len(fut)):
                            if vs[i] and 0 <= uvs[i, 0] < W and 0 <= uvs[i, 1] < H:
                                keep += 1
                            else:
                                break
                        if keep >= 3:
                            pts = [[round(float(u)/W*NORM,1), round(float(v)/H*NORM,1)]
                                   for u, v in uvs[:keep]]
                            steps = ", ".join(str(d) for d in deltas[:keep])
                            emit(template="T4_tcp_trace", frame=f, view=view, image=img,
                                 question=f"Task: {task} Predict the robot gripper's future "
                                          f"2D trajectory in this view: its position after "
                                          f"{steps} frames (20 fps), as [x,y] normalized "
                                          f"to [0,2000).",
                                 answer=str(pts), meta={"future_frames": fut[:keep]})
            # T5 (agentview only): sim-grounded target = object grasped at the NEXT
            # gripper-closing event after this frame (time-aware for multi-step tasks).
            if view == "agentview":
                nxt = next(((ev, bid) for ev, bid in targets if ev >= f), None)
                allb = {bid: b for bid, b in entry["boxes"].get(view, {}).items()}
                if nxt and nxt[1] in allb:
                    b = allb[nxt[1]]
                    cx, cy = (b[4], b[5]) if len(b) >= 6 else ((b[0]+b[2])/2, (b[1]+b[3])/2)
                    emit(template="T5_where_to_act", frame=f, view=view, image=img,
                         question=f"Task: {task} Point to the object that should be "
                                  f"manipulated next. Answer [x,y], normalized to [0,2000).",
                         answer=str([round(cx/W*NORM,1), round(cy/H*NORM,1)]),
                         meta={"bid": int(nxt[1]), "name": clean_name(entry["names"][nxt[1]]),
                               "box": norm_box(b), "grasp_frame": nxt[0], "src": "sim"})
        # T6 agentview <-> wrist
        na = {b: (n, x) for b, n, x in named_boxes(entry, "agentview")}
        nw = {b: (n, x) for b, n, x in named_boxes(entry, "wrist")}
        ia, iw = frm / f"agentview_f{f:04d}.jpg", frm / f"wrist_f{f:04d}.jpg"
        if ia.exists() and iw.exists():
            for bid in [b for b in na if b in nw][:MAX_T6_PER_FRAME]:
                n, b1 = na[bid]
                _, b2 = nw[bid]
                emit(template="T6_cross_view", frame=f, view="agentview->wrist",
                     image=str(ia), image2=str(iw),
                     question=f"The {n} is at {norm_box(b1)} in the first view (a fixed "
                              f"camera; [x0,y0,x1,y1] normalized [0,2000)). Give its "
                              f"bounding box in the second view (the wrist camera), "
                              f"same format.",
                     answer=str(norm_box(b2)), meta={"bid": int(bid)})
    # T7 ego-motion between keyframe pairs (LIBERO demos are 20 fps, shorter)
    for a, b in zip(kfs[:-1], kfs[1:]):
        gap = b - a
        if not (8 <= gap <= 45):
            continue
        ia, ib = frm / f"wrist_f{a:04d}.jpg", frm / f"wrist_f{b:04d}.jpg"
        if not (ia.exists() and ib.exists()):
            continue
        Ea = np.asarray(rec["wrist_E_cam2world"][a], float)
        Eb = np.asarray(rec["wrist_E_cam2world"][b], float)
        ans, t_rel, dist = rel_motion_text(Ea, Eb, gap)
        if dist < 0.01 and rng.random() > 0.1:
            continue
        emit(template="T7_ego_motion", frame=a, view="wrist", image=str(ia), image2=str(ib),
             question=f"These are two wrist-camera frames {gap} frames apart (20 fps). "
                      f"Describe the camera's motion in its own optical frame, where forward "
                      f"means the direction the camera is looking (note: for a downward-looking "
                      f"camera, rising in the world means moving backward): translation per axis "
                      f"in centimeters (right/left, up/down, forward/backward) and rotation "
                      f"as yaw/pitch/roll in degrees.",
             answer=ans, meta={"gap": gap, "t_rel_cm": [round(x*100,2) for x in t_rel],
                               "dist_cm": round(dist*100, 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo-min", type=int, default=0)
    ap.add_argument("--demo-max", type=int, default=99)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    ann = [f for f in sorted(glob.glob(f"{args.data}/libero/ann/*/*.pkl"))
           if args.demo_min <= int(re.search(r"__demo(\d+)\.pkl$", f).group(1)) <= args.demo_max]
    out_path = args.out or f"{args.data}/qa/libero.jsonl"
    counts, bad = {}, 0
    with open(out_path, "w") as fh:
        for fp in ann:
            rec = pickle.load(open(fp, "rb"))
            if rec.get("reproj_selfcheck_med_px", 99) > 15:
                bad += 1
                continue
            for q in episode_qas(rec, f"{args.data}/libero/frames", rng):
                q["qid"] = f"{q['episode']}_{q['template']}_{q['frame']}_{q.get('view','')}"
                counts[q["template"]] = counts.get(q["template"], 0) + 1
                fh.write(json.dumps(q, default=str) + "\n")
    print(f"wrote {sum(counts.values())} QAs from {len(ann)} recs "
          f"({bad} skipped by selfcheck) -> {out_path}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
