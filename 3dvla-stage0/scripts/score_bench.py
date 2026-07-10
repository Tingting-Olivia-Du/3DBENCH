#!/usr/bin/env python3
"""Score benchmark predictions. Reports per-template metric + parse rate
(parse failures are counted separately, never silently scored as wrong-value —
3DBENCH lesson).

Metrics:
  T1/T6 (box)    : acc@IoU>=0.5, mean IoU (over parsed)
  T5 (point)     : acc = point inside GT box (needs GT box from meta; falls back
                   to normalized distance < 100 units of GT point), mean dist
  T2/T3 (scalar) : acc@relerr<=25% (SpatialRGPT convention), median rel-err
  T4 (trace)     : mean per-waypoint L2 in normalized units (matched prefix)
  T7 (text)      : translation dominant-direction accuracy + magnitude rel-err
                   (parsed from free text)
Usage: python score_bench.py [--models qwen3vl4b vlaser2b internvla-m1]
"""
import argparse, json, re
from pathlib import Path

import numpy as np


def parse_box(s):
    nums = re.findall(r"-?\d+\.?\d*", s)
    if len(nums) < 4:
        return None
    b = [float(x) for x in nums[:4]]
    return b if b[2] > b[0] and b[3] > b[1] else None


def parse_point(s):
    nums = re.findall(r"-?\d+\.?\d*", s)
    return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None


def parse_scalar(s):
    m = re.findall(r"-?\d+\.?\d*", s)
    return float(m[0]) if m else None


def parse_trace(s):
    pairs = re.findall(r"\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?", s)
    return [[float(a), float(b)] for a, b in pairs] or None


def iou(a, b):
    xi = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - xi
    return xi / (ua + 1e-9)


def score_template(t, gt_items, preds, coord_scale=(1.0, 1.0), unit_scale=1.0):
    """coord_scale: multiply predicted pixel-normalized coords (2.0 = model answered
    in [0,1000)); unit_scale: multiply predicted scalars (100 = model answered meters).
    One global convention per (model, template) — never per item."""
    rows, n_parse = [], 0
    for q in gt_items:
        p = preds.get(q["qid"])
        if p is None or p.startswith("__ERROR__"):
            continue
        if t in ("T1_grounding", "T6_cross_view"):
            pb, gb = parse_box(p), json.loads(q["answer"].replace("'", '"'))
            if pb is None:
                rows.append(None); continue
            n_parse += 1
            sx, sy = coord_scale
            pb = [pb[0] * sx, pb[1] * sy, pb[2] * sx, pb[3] * sy]
            rows.append(iou(pb, gb))
        elif t == "T5_where_to_act":
            pp, gp = parse_point(p), json.loads(q["answer"].replace("'", '"'))
            if pp is None:
                rows.append(None); continue
            n_parse += 1
            sx, sy = coord_scale
            pp = [pp[0] * sx, pp[1] * sy]
            gb = q.get("meta", {}).get("box")
            if gb:  # primary: point-inside-object-box (fair to e.g. handle-pointing)
                rows.append(1.0 if gb[0] <= pp[0] <= gb[2] and gb[1] <= pp[1] <= gb[3]
                            else 0.0)
            else:
                rows.append(float(np.hypot(pp[0] - gp[0], pp[1] - gp[1])))
        elif t in ("T2_metric3d_dist", "T3_depth_obj", "T3_depth_tcp"):
            pv, gv = parse_scalar(p), float(q["answer"])
            if pv is None:
                rows.append(None); continue
            n_parse += 1
            rows.append(abs(pv * unit_scale - gv) / max(gv, 1e-6))
        elif t == "T4_tcp_trace":
            pt, gt_ = parse_trace(p), json.loads(q["answer"].replace("'", '"'))
            if not pt:
                rows.append(None); continue
            n_parse += 1
            k = min(len(pt), len(gt_))
            sx, sy = coord_scale
            d = np.mean([np.hypot(pt[i][0] * sx - gt_[i][0],
                                  pt[i][1] * sy - gt_[i][1]) for i in range(k)])
            rows.append(float(d))
        elif t == "T7_ego_motion":
            gt_t = q["meta"]["t_rel_cm"]
            gax = int(np.argmax(np.abs(gt_t)))
            words = {0: ("right", "left"), 1: ("down", "up"), 2: ("forward", "backward")}[gax]
            expect = words[0] if gt_t[gax] > 0 else words[1]
            other = words[1] if gt_t[gax] > 0 else words[0]
            pl = p.lower()
            if expect in pl and other not in pl.replace(expect, ""):
                ok = 1.0
            elif other in pl:
                ok = 0.0
            else:
                rows.append(None); continue
            n_parse += 1
            rows.append(ok)
    total = len(rows)
    vals = [r for r in rows if r is not None]
    if not vals:
        return {"n": total, "parse%": 0.0}
    out = {"n": total, "parse%": round(100 * n_parse / max(total, 1), 1)}
    v = np.array(vals)
    if t in ("T1_grounding", "T6_cross_view"):
        out |= {"acc@IoU.5": round(float((v >= 0.5).mean()) * 100, 1),
                "meanIoU": round(float(v.mean()), 3)}
    elif t == "T5_where_to_act":
        if set(np.unique(v)) <= {0.0, 1.0}:  # inside-box mode
            out |= {"accInBox%": round(float(v.mean()) * 100, 1)}
        else:
            out |= {"acc@100u": round(float((v <= 100).mean()) * 100, 1),
                    "medDist_u": round(float(np.median(v)), 1)}
    elif t in ("T2_metric3d_dist", "T3_depth_obj", "T3_depth_tcp"):
        out |= {"acc@25%": round(float((v <= 0.25).mean()) * 100, 1),
                "medRelErr": round(float(np.median(v)), 3)}
    elif t == "T4_tcp_trace":
        out |= {"medErr_u": round(float(np.median(v)), 1)}
    elif t == "T7_ego_motion":
        out |= {"dirAcc%": round(float(v.mean()) * 100, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--models", nargs="+",
                    default=["qwen3vl4b", "internvla-m1", "vlaser2b"])
    ap.add_argument("--bench", default="m1_bench.jsonl")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    bench = [json.loads(l) for l in open(f"{args.data}/benchmark/{args.bench}")]
    by_t = {}
    for q in bench:
        if q["template"] == "T3_depth_tcp" and q.get("view") == "wrist":
            continue  # degenerate: wrist-cam->TCP is a constant mount property
        by_t.setdefault(q["template"], []).append(q)
    PRIMARY = {"T1_grounding": ("acc@IoU.5", 1), "T6_cross_view": ("acc@IoU.5", 1),
               "T5_where_to_act": ("medDist_u", -1), "T4_tcp_trace": ("medErr_u", -1),
               "T2_metric3d_dist": ("medRelErr", -1), "T3_depth_obj": ("medRelErr", -1),
               "T3_depth_tcp": ("medRelErr", -1), "T7_ego_motion": ("dirAcc%", 1)}
    for m in args.models:
        pp = Path(f"{args.data}/benchmark/preds_{m}{args.suffix}.jsonl")
        if not pp.exists():
            print(f"== {m}: no predictions yet"); continue
        preds = {json.loads(l)["qid"]: json.loads(l)["pred"] for l in open(pp)}
        print(f"== {m} ({len(preds)} preds)")
        for t in sorted(by_t):
            strict = score_template(t, by_t[t], preds)
            # convention oracle: one global rescale per (model, template)
            key, sign = PRIMARY[t]
            if t == "T5_where_to_act" and "accInBox%" in strict:
                key, sign = "accInBox%", 1
            best, best_conv = strict, "as-instructed"
            for cs, us, name in (((2.0, 2.0), 1.0, "coords x2 [0,1000)->[0,2000)"),
                                 ((2000/640, 2000/480), 1.0, "abs pixels->norm"),
                                 ((1.0, 1.0), 100.0, "meters->cm")):
                cand = score_template(t, by_t[t], preds, coord_scale=cs, unit_scale=us)
                if key in cand and (key not in best or
                                    sign * cand[key] > sign * best.get(key, sign * -1e18)):
                    best, best_conv = cand, name
            line = f"  {t:<18} strict {strict}"
            if best_conv != "as-instructed":
                line += f"\n  {'':<18} oracle {best}  [{best_conv}]"
            print(line)


if __name__ == "__main__":
    main()
