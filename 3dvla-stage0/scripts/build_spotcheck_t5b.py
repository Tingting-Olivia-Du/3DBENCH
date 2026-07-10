#!/usr/bin/env python3
"""T5b grasp-affordance spot-check reviewer: 15 items per suite (60 total) from
qa/libero_t5b.jsonl. Overlay: red cross = grasp-point GT (the demo's actual grasp
contact, projected to this pre-grasp frame); yellow box = target object.
Output: out/spotcheck_t5b.html (+ imgs). Reuses the reviewer UI template.
"""
import hashlib, json, re, shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/workspace/tingting/3dvla-data")
OUT = Path("/workspace/tingting/3dvla-stage0/out")
IMG_DIR = OUT / "spotcheck_t5b_imgs"
shutil.rmtree(IMG_DIR, ignore_errors=True)
IMG_DIR.mkdir(exist_ok=True)
W, H, NORM = 640, 480, 2000.0
PER_SUITE = 15

qs = [json.loads(l) for l in open(DATA / "qa" / "libero_t5b.jsonl")]
by_suite = defaultdict(list)
for q in qs:
    by_suite[q["episode"].split("/")[0]].append(q)
rng = np.random.default_rng(11)
picks = []
for s, items in sorted(by_suite.items()):
    idx = rng.choice(len(items), size=min(PER_SUITE, len(items)), replace=False)
    picks += [items[i] for i in idx]

TOK = hashlib.md5(json.dumps([q["qid"] for q in picks]).encode()).hexdigest()[:6]
items = []
for q in picks:
    img = cv2.imread(q["image"])
    b = q["meta"]["box"]
    bb = [int(b[0] / NORM * W), int(b[1] / NORM * H), int(b[2] / NORM * W), int(b[3] / NORM * H)]
    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (60, 200, 240), 1)
    p = json.loads(q["answer"])
    cv2.drawMarker(img, (int(p[0] / NORM * W), int(p[1] / NORM * H)), (46, 46, 230),
                   cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, f"grasp@f{q['meta']['grasp_frame']}  target: {q['meta']['name'][:22]}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    safe = q["qid"].replace("/", "_")
    cv2.imwrite(str(IMG_DIR / f"{TOK}_{safe}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    items.append({"qid": safe, "t": "T5b_grasp_point", "ep": q["episode"],
                  "frame": q["frame"], "view": "agentview",
                  "q": q["question"], "a": q["answer"]})

tmpl = (OUT / "spotcheck.html").read_text()
page = tmpl.replace("spotcheck_imgs/", "spotcheck_t5b_imgs/")
page = re.sub(r"const TOKEN = '[a-f0-9]*'", f"const TOKEN = '{TOK}'", page)
page = page.replace("const KEY = 'spotcheck_m1_v2';", "const KEY = 'spotcheck_t5b_v1';")
page = page.replace("<title>Benchmark GT 人工抽检</title>", "<title>T5b 抓取点 GT 抽检</title>")
page = page.replace("<h1>GT 人工抽检</h1>",
                    "<h1>T5b 抓取 affordance GT 抽检 — 十字应落在合理抓取位（把手/沿/颈）</h1>")
page = page.replace("a.download = 'spotcheck_verdicts.csv'",
                    "a.download = 'spotcheck_t5b_verdicts.csv'")
page = re.sub(r"const ITEMS = \[.*?\];\n",
              lambda m: "const ITEMS = " + json.dumps(items, ensure_ascii=False) + ";\n",
              page, count=1, flags=re.S)
(OUT / "spotcheck_t5b.html").write_text(page)
print(f"rendered {len(items)} overlays (token {TOK}) -> localhost:8737/spotcheck_t5b.html")
