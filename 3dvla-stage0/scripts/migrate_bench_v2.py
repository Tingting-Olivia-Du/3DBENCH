#!/usr/bin/env python3
"""Rebuild the LIBERO bench (v2: sim-grounded T5, inside-points) and migrate
still-valid predictions: an old pred transfers iff its qid exists in v2 with
IDENTICAL question+answer+image. Run AFTER all v1 evals finish; then re-launch
evals to fill the remaining (changed) items.
"""
import json, shutil, subprocess, sys
from pathlib import Path

B = Path("/workspace/tingting/3dvla-data/benchmark")
MODELS = ["qwen3vl4b", "internvla-m1", "vlaser2b"]

# 1. snapshot v1
shutil.copy(B / "libero_bench.jsonl", B / "libero_bench_v1.jsonl")
old = {json.loads(l)["qid"]: json.loads(l) for l in open(B / "libero_bench_v1.jsonl")}

# 2. rebuild (writes libero_bench.jsonl, spotcheck_libero.csv, stats_libero.json)
subprocess.run([sys.executable, "scripts/benchmark_build.py", "--chunks", "libero",
                "--tag", "libero"], check=True,
               cwd="/workspace/tingting/3dvla-stage0")
new = {json.loads(l)["qid"]: json.loads(l) for l in open(B / "libero_bench.jsonl")}

# 3. migrate predictions whose item content is unchanged
same = {qid for qid, q in new.items()
        if qid in old and all(old[qid].get(k) == q.get(k)
                              for k in ("question", "answer", "image", "image2"))}
print(f"v2 items: {len(new)}, content-identical with v1: {len(same)}")
for m in MODELS:
    p = B / f"preds_{m}_libero.jsonl"
    if not p.exists():
        continue
    shutil.copy(p, B / f"preds_{m}_libero_v1.jsonl")
    kept = [l for l in open(B / f"preds_{m}_libero_v1.jsonl")
            if json.loads(l)["qid"] in same]
    with open(p, "w") as fh:
        fh.writelines(kept)
    print(f"  {m}: kept {len(kept)} preds, to re-run: {len(new) - len(kept)}")
