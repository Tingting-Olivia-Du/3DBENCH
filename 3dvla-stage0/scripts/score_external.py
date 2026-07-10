#!/usr/bin/env python3
"""Score the external benchmark battery (preds from eval_external.py).

Metrics:
  cvbench / erqa / mmmu-MC / vsi-MC : accuracy (letter match)
  mmmu-open                         : normalized exact match
  robospatial compat/config         : yes-no accuracy
  robospatial context               : mean fraction of predicted points inside GT
                                      mask; scale read {[0,1] as-written | /1000 |
                                      /2000 | abs px} reported separately
  vsi numeric                       : MRA (mean over theta in .5...95 of
                                      1[relerr < 1-theta], official protocol)

Usage: python score_external.py [--models base stage1_full] [--bench cvbench ...]
"""
import argparse, io, json, re
from collections import defaultdict
from pathlib import Path

DATA = Path("/workspace/tingting/3dvla-data/external_bench")
RSH = Path("/root/angli/hf_cache/hub/datasets--chanhee-luke--RoboSpatial-Home/"
           "snapshots/7ef1b43b20087ef284e0e9017ef51dd143814c59/data")

LETTERS = "ABCDEFGHI"


def extract_letter(pred):
    p = pred.strip()
    m = re.match(r"^\(?([A-I])\)?\b", p)
    if m and len(p) <= 40:          # direct short answer, e.g. "C", "(C) 1"
        return m.group(1)
    # long CoT: take the LAST explicit answer statement (earlier "(A)"-style
    # hits are usually the model discussing options, not answering)
    ms = re.findall(r"answer(?:\s+is)?\s*[:：]?\s*\**\(?([A-I])\)?\b", p, re.I)
    if ms:
        return ms[-1].upper()
    ms = re.findall(r"\(([A-I])\)", p)
    if ms:
        return ms[-1]
    m = re.match(r"^\s*([A-I])\b", p)
    return m.group(1) if m else None


def norm_open(s):
    s = str(s).strip().lower().rstrip(".").replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def open_match(pred, gt):
    p, g = norm_open(pred), norm_open(gt)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        pass
    return p == g or (len(g) > 2 and g in p)


def first_float(s):
    m = re.search(r"-?\d+\.?\d*", str(s).replace(",", ""))
    return float(m.group()) if m else None


def mra(pred, gt):
    p, g = first_float(pred), first_float(gt)
    if p is None or g is None or g == 0:
        return 0.0
    rel = abs(p - g) / abs(g)
    thetas = [0.5 + 0.05 * i for i in range(10)]
    return sum(rel < (1 - t) for t in thetas) / len(thetas)


def score_mc_like(rows):
    ok = sum(extract_letter(r["pred"]) == r["gt"].strip().upper() for r in rows)
    return {"n": len(rows), "acc": round(100 * ok / len(rows), 1)}


def score_yesno(rows):
    ok = 0
    for r in rows:
        m = re.search(r"\b(yes|no)\b", r["pred"].lower())
        ok += bool(m) and m.group(1) == r["gt"].strip().lower()
    return {"n": len(rows), "acc": round(100 * ok / len(rows), 1)}


def parse_points(pred):
    return [(float(a), float(b)) for a, b in
            re.findall(r"[(\[]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[)\]]", pred)]


def score_context(rows):
    import numpy as np
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(RSH / "context-00000-of-00001.parquet")
    masks = {}
    for i, rr in df.iterrows():
        mk = np.array(Image.open(io.BytesIO(rr["mask"]["bytes"])).convert("L"))
        masks[f"rs_context_{i}"] = mk
    scales = {"as01": None, "d1000": 1000.0, "d2000": 2000.0, "abspx": "abs"}
    out = {}
    for tag, sc in scales.items():
        vals = []
        for r in rows:
            mk = masks.get(r["qid"])
            if mk is None:
                continue
            H, W = mk.shape
            pts = parse_points(r["pred"])
            if not pts:
                vals.append(0.0)
                continue
            hit = []
            for x, y in pts:
                if sc == "abs":
                    px, py = x, y
                elif sc is None:
                    px, py = x * W, y * H
                else:
                    px, py = x / sc * W, y / sc * H
                xi, yi = int(round(px)), int(round(py))
                hit.append(0 <= xi < W and 0 <= yi < H and mk[yi, xi] > 127)
            vals.append(sum(hit) / len(hit))
        out[tag] = round(100 * sum(vals) / len(vals), 1) if vals else None
    return {"n": len(rows), **out}


def score_bench(bench, rows):
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r["subtask"]].append(r)
    res = {}
    for sub, rs in sorted(by_sub.items()):
        if bench == "robospatial":
            res[sub] = score_yesno(rs) if sub != "context" else score_context(rs)
        elif bench == "mmmu":
            res[sub] = score_mc_like(rs) if sub == "multiple-choice" else \
                {"n": len(rs), "acc": round(100 * sum(
                    open_match(r["pred"], r["gt"]) for r in rs) / len(rs), 1)}
        elif bench == "vsi":
            if sub in {"object_counting", "object_abs_distance",
                       "object_size_estimation", "room_size_estimation"}:
                res[sub] = {"n": len(rs), "MRA": round(100 * sum(
                    mra(r["pred"], r["gt"]) for r in rs) / len(rs), 1)}
            else:
                res[sub] = score_mc_like(rs)
        else:
            res[sub] = score_mc_like(rs)
    # bench-level aggregate: mean of subtask primary metrics (official style for
    # cvbench is 2D/3D averaged separately; we print subtasks, reader aggregates)
    prim = [v.get("acc", v.get("MRA")) for v in res.values()
            if v.get("acc", v.get("MRA")) is not None]
    if prim:
        res["__mean__"] = {"n": sum(v["n"] for k, v in res.items() if k != "__mean__"),
                           "acc": round(sum(prim) / len(prim), 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "stage1"])
    ap.add_argument("--bench", nargs="+",
                    default=["cvbench", "erqa", "robospatial", "mmmu", "vsi"])
    args = ap.parse_args()
    for m in args.models:
        print(f"===== model: {m}")
        for b in args.bench:
            p = DATA / f"preds_{b}_{m}.jsonl"
            if not p.exists():
                print(f"  -- {b}: no preds")
                continue
            rows = [json.loads(l) for l in open(p)]
            errs = sum(r["pred"].startswith("__ERROR__") for r in rows)
            res = score_bench(b, rows)
            for sub, v in res.items():
                print(f"  {b:12s} {sub:22s} {v}")
            if errs:
                print(f"  {b:12s} !! {errs} __ERROR__ preds")


if __name__ == "__main__":
    main()
