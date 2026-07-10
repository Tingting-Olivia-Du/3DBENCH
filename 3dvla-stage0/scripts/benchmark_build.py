#!/usr/bin/env python3
"""Build the in-domain 3D benchmark (InternData-M1 half) from a HELD-OUT chunk's
QA pool (odd chunks are never used for training by construction).

Selection:
  - stratified: up to N_PER_TEMPLATE per template
  - diversity: <= MAX_PER_EP items per (episode, template); balance across views
  - metric templates (T2/T3/T7): sample uniformly across answer quartiles so the
    benchmark can't be gamed by predicting the prior mean

Outputs (under <data>/benchmark/):
  m1_bench.jsonl       the benchmark items (same schema as training QA + qid)
  spotcheck.csv        SPOT_PER_TEMPLATE random items per template for human verify
                       (columns: qid, image, question, answer, verdict[blank], note[blank])
  stats.json           per-template counts + answer distributions
"""
import argparse, collections, csv, json
from pathlib import Path

import numpy as np

N_PER_TEMPLATE = 250
MAX_PER_EP = 3
SPOT_PER_TEMPLATE = 50


def metric_value(q):
    t = q["template"]
    if t in ("T2_metric3d_dist", "T3_depth_obj", "T3_depth_tcp"):
        return float(q["answer"])
    if t == "T7_ego_motion":
        return float(q["meta"]["dist_cm"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--chunks", nargs="+", default=["chunk-001"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pool = []
    for c in args.chunks:
        with open(Path(args.data) / "qa" / f"{c}.jsonl") as fh:
            pool += [json.loads(l) for l in fh]
    by_t = collections.defaultdict(list)
    for q in pool:
        by_t[q["template"]].append(q)

    bench, stats = [], {}
    for t, items in sorted(by_t.items()):
        rng.shuffle(items)
        vals = [metric_value(q) for q in items]
        if vals[0] is not None:
            # quartile-stratified for metric answers
            qs = np.percentile([v for v in vals], [25, 50, 75])
            buckets = collections.defaultdict(list)
            for q, v in zip(items, vals):
                buckets[int(np.searchsorted(qs, v))].append(q)
            order = []
            for i in range(N_PER_TEMPLATE):
                b = buckets[i % 4]
                if b:
                    order.append(b.pop())
            items = order
        picked, per_ep = [], collections.Counter()
        for q in items:
            key = (q["episode"], t)
            if per_ep[key] >= MAX_PER_EP:
                continue
            per_ep[key] += 1
            picked.append(q)
            if len(picked) >= N_PER_TEMPLATE:
                break
        for q in picked:
            q["qid"] = f"bench_{t}_{len(bench):05d}"
            bench.append(q)
        mv = [metric_value(q) for q in picked]
        stats[t] = {"n": len(picked),
                    "episodes": len({q['episode'] for q in picked}),
                    "answer_p5_p50_p95": ([round(float(np.percentile([v for v in mv], p)), 1)
                                           for p in (5, 50, 95)] if mv[0] is not None else None)}

    out = Path(args.data) / "benchmark"
    out.mkdir(exist_ok=True)
    with open(out / f"{args.tag}_bench.jsonl", "w") as fh:
        for q in bench:
            fh.write(json.dumps(q, default=str) + "\n")
    # human spot-check sheet
    with open(out / f"spotcheck_{args.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "image", "image2", "question", "answer", "verdict", "note"])
        for t in sorted(by_t):
            tq = [q for q in bench if q["template"] == t]
            for q in (tq if len(tq) <= SPOT_PER_TEMPLATE else
                      [tq[i] for i in rng.choice(len(tq), SPOT_PER_TEMPLATE, replace=False)]):
                w.writerow([q["qid"], q["image"], q.get("image2", ""),
                            q["question"], q["answer"], "", ""])
    with open(out / f"stats_{args.tag}.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"benchmark: {len(bench)} items -> {out}/{args.tag}_bench.jsonl")
    for t, s in stats.items():
        print(f"  {t}: n={s['n']} eps={s['episodes']} ans_p5/50/95={s['answer_p5_p50_p95']}")


if __name__ == "__main__":
    main()
