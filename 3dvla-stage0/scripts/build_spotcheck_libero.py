#!/usr/bin/env python3
"""LIBERO-half spot-check reviewer: render GT overlays for spotcheck_libero.csv
items and emit out/spotcheck_libero.html (same UI as the M1 reviewer).
GT provenance drawn on image: red = answer; blue = auxiliary (projected AABB
centers / trace links); yellow = question-side input (T6 view-1 box).
"""
import ast, csv, json, pickle, re as _re0, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qa_gen_libero import named_boxes, project, clean_name

DATA = Path("/workspace/tingting/3dvla-data")
OUT = Path("/workspace/tingting/3dvla-stage0/out")
IMG_DIR = OUT / "spotcheck_libero_imgs"
import shutil
shutil.rmtree(IMG_DIR, ignore_errors=True)  # stale overlays from older qid numbering caused image/text mismatches
IMG_DIR.mkdir(exist_ok=True)
W, H, NORM = 640, 480, 2000.0
RED, BLU, YEL = (46, 46, 230), (220, 200, 40), (60, 200, 240)


def d(v, ax):
    return int(v / NORM * (W if ax == "x" else H))


def dbox(b):
    return d(b[0], "x"), d(b[1], "y"), d(b[2], "x"), d(b[3], "y")


bench = {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "benchmark" / "libero_bench.jsonl")}
rows = list(csv.DictReader(open(DATA / "benchmark" / "spotcheck_libero.csv")))
recs = {}


def rec_of(ep):  # ep = "suite/task__demoN"
    if ep not in recs:
        suite, name = ep.split("/", 1)
        recs[ep] = pickle.load(open(DATA / "libero" / "ann" / suite / f"{name}.pkl", "rb"))
    return recs[ep]


import hashlib as _h
TOK = _h.md5(open(DATA / "benchmark" / "spotcheck_libero.csv","rb").read()).hexdigest()[:6]
items = []
for r in rows:
    q = bench[r["qid"]]
    t, ep, f = q["template"], q["episode"], q["frame"]
    rec = rec_of(ep)
    entry = rec["per_kf"][f]
    img = cv2.imread(q["image"])
    if t == "T1_grounding":
        b = ast.literal_eval(q["answer"])
        cv2.rectangle(img, dbox(b)[:2], dbox(b)[2:], RED, 2)
    elif t == "T5_where_to_act":
        p = ast.literal_eval(q["answer"])
        cv2.drawMarker(img, (d(p[0], "x"), d(p[1], "y")), RED, cv2.MARKER_CROSS, 20, 2)
    elif t == "T4_tcp_trace":
        pts = [(d(p[0], "x"), d(p[1], "y")) for p in ast.literal_eval(q["answer"])]
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, RED, -1)
            if i:
                cv2.line(img, pts[i - 1], p, BLU, 1)
    elif t == "T3_depth_tcp":
        uv = q["meta"]["tcp_uv"]
        cv2.drawMarker(img, (d(uv[0], "x"), d(uv[1], "y")), RED, cv2.MARKER_CROSS, 20, 2)
    elif t in ("T2_metric3d_dist", "T3_depth_obj"):
        view = q["view"]
        K = np.asarray(rec["cams"][view][f]["K"], float)
        E = np.asarray(rec["cams"][view][f]["E_cam2world"], float)
        # re-derive the same two objects the generator used (largest unique-named)
        named = named_boxes(entry, view)
        withc = [(bid, n, b) for bid, n, b in named if bid in entry["aabb"]]
        cts = []
        for bid, n, b in withc[: (2 if t == "T2_metric3d_dist" else 1)]:
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:4]), RED, 2)
            c3 = np.asarray(entry["aabb"][bid], float).mean(0)
            uv, vis = project(c3, K, E)
            if vis[0]:
                cts.append((int(uv[0, 0]), int(uv[0, 1])))
        if t == "T2_metric3d_dist" and len(cts) == 2:
            cv2.line(img, cts[0], cts[1], BLU, 2)
            mid = ((cts[0][0] + cts[1][0]) // 2, (cts[0][1] + cts[1][1]) // 2)
            cv2.putText(img, f"{float(q['answer']):.0f}cm", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif t == "T6_cross_view":
        img2 = cv2.imread(q["image2"])
        qb = ast.literal_eval(q["question"].split("is at ")[1].split(" in the first")[0])
        ab = ast.literal_eval(q["answer"])
        cv2.rectangle(img, dbox(qb)[:2], dbox(qb)[2:], YEL, 2)
        cv2.rectangle(img2, dbox(ab)[:2], dbox(ab)[2:], RED, 2)
        img = np.concatenate([img, img2], 1)
    elif t == "T7_ego_motion":
        img2 = cv2.imread(q["image2"])
        cv2.putText(img, "t", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2)
        cv2.putText(img2, f"t+{q['meta']['gap']}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2)
        img = np.concatenate([img, img2], 1)
    cv2.imwrite(str(IMG_DIR / f"{TOK}_{q['qid']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    items.append({"qid": q["qid"], "t": t, "ep": ep, "frame": f, "view": str(q["view"]),
                  "q": q["question"], "a": q["answer"]})

# reuse the M1 reviewer page with swapped payload/paths/title/storage key
tmpl = (OUT / "spotcheck.html").read_text()
page = tmpl.replace("spotcheck_imgs/", "spotcheck_libero_imgs/")
page = _re0.sub(r"const TOKEN = '[a-f0-9]*'", f"const TOKEN = '{TOK}'", page)
page = page.replace("const KEY = 'spotcheck_v1';", "const KEY = 'spotcheck_libero_v3';")
page = page.replace("<title>Benchmark GT 人工抽检</title>",
                    "<title>LIBERO 半 GT 人工抽检</title>")
page = page.replace("<h1>GT 人工抽检</h1>", "<h1>GT 人工抽检 — LIBERO 半</h1>")
page = page.replace("a.download = 'spotcheck_verdicts.csv'",
                    "a.download = 'spotcheck_libero_verdicts.csv'")
import re as _re
page = _re.sub(r"const ITEMS = \[.*?\];\n", lambda m: "const ITEMS = " + json.dumps(items, ensure_ascii=False) + ";\n",
               page, count=1, flags=_re.S)
(OUT / "spotcheck_libero.html").write_text(page)
print(f"rendered {len(items)} overlays -> {IMG_DIR}")
print("reviewer -> localhost:8737/spotcheck_libero.html")
