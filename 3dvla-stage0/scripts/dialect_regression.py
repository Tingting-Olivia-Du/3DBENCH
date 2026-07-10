#!/usr/bin/env python3
"""Two-dialect format regression (plan: Stage-1 audit; DATA_SPEC invariant #4).

Direction A (ours): benchmark T1 prompts (carry the [0,2000) preamble) ->
  responses must be bracketed numbers in (1.05, 2000); values <=1.05 = leakage
  of the native dialect.
Direction B (native): LLaVA-style grounding prompts sampled from the general
  slice (native [0,1] floats, NO preamble) -> responses must stay <=1.05;
  values >1.05 = leakage of our dialect (the forgetting-adjacent failure).

Reports per-direction leakage %, non-parse %, and n. Run per checkpoint:
  python dialect_regression.py --ckpt <safetensors|base> --device cuda:0 -n 200
"""
import argparse, json, random, re, sys

import torch

LIST_RE = re.compile(r"\[\s*-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*){1,3}\s*\]")


def classify(text):
    m = LIST_RE.search(text)
    if not m:
        return "noparse", None
    vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", m.group(0))]
    mx = max(vals)
    if mx <= 1.05:
        return "native", mx
    if mx < 2001:
        return "ours", mx
    return "other", mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage1 safetensors path or 'base'")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()
    random.seed(0)

    sys.path.insert(0, "/workspace/tingting/3dvla-stage0/scripts")
    from eval_bench import load_qwen, load_stage1
    run = (load_qwen("Qwen/Qwen3-VL-4B-Instruct", args.device) if args.ckpt == "base"
           else load_stage1(args.ckpt, args.device))
    is_stage1 = args.ckpt != "base"

    # Direction A: our-dialect T1 prompts from both bench halves
    ours_items = []
    for f, viewful in [("m1_bench.jsonl", True), ("libero_bench.jsonl", True)]:
        for l in open(f"/workspace/tingting/3dvla-data/benchmark/{f}"):
            d = json.loads(l)
            if d["template"] == "T1_grounding":
                ours_items.append(d)
    ours_items = random.sample(ours_items, min(args.n, len(ours_items)))

    # Direction B: native grounding prompts (answer carries [0,1] coords)
    native_items = []
    for l in open("/workspace/tingting/3dvla-data/sft/general_llava_coco.jsonl"):
        d = json.loads(l)
        conv = d["conversations"]
        for i in range(0, len(conv) - 1, 2):
            q, a = conv[i]["value"], conv[i + 1]["value"]
            if LIST_RE.search(a) and not LIST_RE.search(q):
                native_items.append({"image": "/workspace/tingting/3dvla-data/general/" + d["images"][0],
                                     "question": q.replace("<image>", "").strip()})
                break
        if len(native_items) >= args.n:
            break

    def eval_dir(items, expect, views=False):
        from collections import Counter
        c = Counter()
        for it in items:
            imgs = [it["image"]]
            try:
                if is_stage1:
                    v = [it.get("view")] if views else [None]
                    pred = run(imgs, it["question"], views=v)
                else:
                    pred = run(imgs, it["question"])
            except Exception:
                c["error"] += 1
                continue
            kind, _ = classify(pred)
            c[kind] += 1
        n = sum(c.values())
        leak = {"ours": c["native"], "native": c["ours"]}[expect]
        print(f"  expect={expect:<7} n={n} correct-dialect={c[expect]} "
              f"leak={leak} ({100*leak/max(1,n):.1f}%) noparse={c['noparse']} "
              f"other={c['other']} error={c['error']}")
        return c

    print(f"== dialect regression: {args.ckpt}")
    print("Direction A (our [0,2000) prompts):")
    eval_dir(ours_items, "ours", views=True)
    print("Direction B (native LLaVA grounding prompts):")
    eval_dir(native_items, "native")


if __name__ == "__main__":
    main()
