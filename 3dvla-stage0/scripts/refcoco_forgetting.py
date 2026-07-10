#!/usr/bin/env python3
"""RefCOCO-val forgetting audit (plan: Stage-1 dual audit, forgetting half).

Protocol: REC direction — LLaVA-style native prompt (NO [0,2000) preamble):
  "Please provide the bounding box coordinate of the region this sentence
   describes: <expr>"
Scored under a per-model convention oracle over {[0,1] floats, [0,1000],
absolute pixels} (the same generosity Exp0 gave zero-shot baselines), plus
the strict-native ([0,1]) reading. Forgetting = finetuned drop vs base under
each model's best convention; the two-dialect design predicts the finetuned
model stays in [0,1] on native prompts.

  python refcoco_forgetting.py --ckpt base|<safetensors> --device cuda:0 -n 500
"""
import argparse, io, json, random, re, sys

import torch

LIST_RE = re.compile(r"\[\s*-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*){3}\s*\]")
PROMPT = ("Please provide the bounding box coordinate of the region this "
          "sentence describes: {expr}")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    random.seed(0)

    sys.path.insert(0, "/workspace/tingting/3dvla-stage0/scripts")
    from eval_bench import load_qwen, load_stage1
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    idx = random.sample(range(len(ds)), min(args.n, len(ds)))

    if args.ckpt == "base":
        run = load_qwen("Qwen/Qwen3-VL-4B-Instruct", args.device)
        is_stage1 = False
    else:
        run = load_stage1(args.ckpt, args.device)
        is_stage1 = True

    import ast, tempfile, os
    from collections import Counter
    hits = {c: 0 for c in ("01", "1000", "abs", "2000")}
    stats = Counter()
    tmpdir = tempfile.mkdtemp(prefix="refcoco_")
    for k, i in enumerate(idx):
        r = ds[i]
        a = r["answer"]
        exprs = a if isinstance(a, list) else (ast.literal_eval(a) if a.startswith("[") else [a])
        expr = exprs[0]
        im = r["image"]
        W, H = im.size
        bb = r["bbox"]
        x, y, w, h = ast.literal_eval(bb) if isinstance(bb, str) else bb
        gt = (x, y, x + w, y + h)
        p = os.path.join(tmpdir, "cur.jpg")
        im.convert("RGB").save(p)
        try:
            if is_stage1:
                pred = run([p], PROMPT.format(expr=expr), views=[None])  # native: NO unify, no K
            else:
                pred = run([p], PROMPT.format(expr=expr))
        except Exception:
            stats["error"] += 1
            continue
        m = LIST_RE.search(pred)
        if not m:
            stats["noparse"] += 1
            continue
        v = [float(t) for t in re.findall(r"-?\d+\.?\d*", m.group(0))]
        mx = max(v)
        stats["mag_le1"] += mx <= 1.05
        stats["mag_1000"] += 1.05 < mx <= 1200
        stats["mag_2000"] += 1200 < mx <= 2100
        stats["parsed"] += 1
        for conv, box in (("01", [v[0]*W, v[1]*H, v[2]*W, v[3]*H]),
                          ("1000", [v[0]/1000*W, v[1]/1000*H, v[2]/1000*W, v[3]/1000*H]),
                          ("2000", [v[0]/2000*W, v[1]/2000*H, v[2]/2000*W, v[3]/2000*H]),
                          ("abs", v)):
            if iou(box, gt) >= 0.5:
                hits[conv] += 1
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(idx)}", flush=True)

    n = len(idx)
    print(f"== RefCOCO-val forgetting probe: {args.ckpt} {args.tag}")
    print(f"  n={n} parsed={stats['parsed']} noparse={stats['noparse']} error={stats['error']}")
    print(f"  output-dialect magnitudes: <=1 {stats['mag_le1']}, ~[0,1000] {stats['mag_1000']}, "
          f"~[0,2000) {stats['mag_2000']}")
    for c in ("01", "1000", "2000", "abs"):
        print(f"  acc@IoU.5 under {c:>4}: {100*hits[c]/n:.1f}")
    print(f"  ORACLE acc@IoU.5: {100*max(hits.values())/n:.1f}")


if __name__ == "__main__":
    main()
