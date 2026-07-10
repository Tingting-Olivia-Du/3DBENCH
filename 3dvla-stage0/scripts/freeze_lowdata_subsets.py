#!/usr/bin/env python3
"""Freeze the Stage-2 LIBERO low-data demo subsets (Option A, decided 2026-07-10).

Protocol:
  - Demo pool per task: 50 (indices 0-49). Stage-1 allocation is FROZEN upstream:
    anchor-QA keyframes = demos 0-44, VLM-QA bench = demos 45-49.
  - Low-data budgets are sampled from 0-44 (Option A): the backbone saw these
    demos' KEYFRAME IMAGES (and geometric QA about them) in Stage-1 — never
    actions. The anchor slice is identical across variants, so intervention
    contrasts difference this exposure out; absolute data-efficiency claims
    carry the image-exposure caveat (paper §Setup). Exposure-robust readouts:
    LIBERO-Plus OOD + SimplerEnv-Bridge second domain.
  - p10 = 5 demos/task (10% of 50), p25 = 12 demos/task (25%, floor of 12.5),
    NESTED (p10 ⊂ p25) so budget comparisons are monotone.
  - Sampled ONCE with per-task deterministic seeds; shared across ALL variants
    and ALL Stage-2 seeds. Re-running this script reproduces the file bit-exact.

v2 (2026-07-10): sampling pool now intersects the lerobot episode map —
the *_no_noops lerobot conversions dropped replay-failed demos (spatial
432/500 ... libero_10 379/500), so a v1 subset could reference demos that
do not exist as training episodes. v2 samples ONLY from demos present in
lerobot AND in 0-44, and records the lerobot episode indices directly
(what the training filter consumes), plus the bench-episode ban list.
Requires: stage2/map_lerobot_episodes.py output (action-sequence-verified).

Output: /workspace/tingting/3dvla-data/stage2/lowdata_subsets_v2.json
"""
import hashlib, json, random
from pathlib import Path

import h5py

DATASETS = Path("/workspace/tingting/LIBERO/libero/datasets")
OUT = Path("/workspace/tingting/3dvla-data/stage2/lowdata_subsets_v2.json")
EP_MAP = Path("/workspace/tingting/3dvla-stage0/stage2/lerobot_episode_map_v1.json")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SEED = 42
ANCHOR_MAX = 44          # demos 0-44 = anchor pool (Stage-1 QA keyframes)
BENCH = set(range(45, 50))  # demos 45-49 = VLM-QA bench, NEVER in any budget
P10_N, P25_N = 5, 12

ep_map = json.loads(EP_MAP.read_text())

result = {"_meta": {
    "version": "v2", "frozen": "2026-07-10", "seed": SEED,
    "rule": "p10=5, p25=12 (floor of 25% of 50), nested p10 subset-of p25, "
            "sampled from demos {0-44} INTERSECT lerobot-present (Option A); "
            "demos 45-49 = VLM-QA bench, excluded from every budget; "
            "*_episodes fields carry the lerobot episode indices the training "
            "filter consumes; ban_episodes = bench demos' episodes per suite",
    "exposure_caveat": "backbone saw keyframe IMAGES of demos 0-44 as Stage-1 "
                       "anchor QA (~11 keyframes/demo, zero action tokens); "
                       "identical across variants -> intervention contrasts "
                       "unbiased; absolute claims carry this caveat",
    "zero_exposure_task": "libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_"
                          "bottom_drawer_of_the_cabinet_and_close_it was skipped "
                          "by anchor QA generation (ambiguity guard) -> its "
                          "images are FULLY unexposed; usable as a within-suite "
                          "exposure-free spot check",
}}

for suite in SUITES:
    sdir = DATASETS / suite
    assert sdir.is_dir(), f"missing suite dir {sdir}"
    d2e = ep_map[suite]["demo_to_episode"]
    result[suite] = {"ban_episodes": sorted(ep_map[suite]["bench_episodes"])}
    min_pool = 99
    for h5path in sorted(sdir.glob("*.hdf5")):
        task = h5path.stem.replace("_demo", "")
        with h5py.File(h5path, "r") as f:
            n = len(f["data"].keys())
        assert n == 50, f"{task}: expected 50 demos, found {n}"
        pool = sorted(int(k.rsplit("demo_", 1)[1]) for k in d2e
                      if k.rsplit("__demo_", 1)[0] == task
                      and int(k.rsplit("demo_", 1)[1]) <= ANCHOR_MAX)
        min_pool = min(min_pool, len(pool))
        assert len(pool) >= P25_N, \
            f"{suite}/{task}: only {len(pool)} lerobot-present demos in 0-44 (< {P25_N})"
        rng = random.Random(f"{SEED}:{suite}:{task}")
        p25 = sorted(rng.sample(pool, P25_N))
        p10 = sorted(rng.sample(p25, P10_N))
        assert not (set(p25) & BENCH) and set(p10) <= set(p25)
        result[suite][task] = {
            "p10": p10, "p25": p25, "pool_size": len(pool),
            "p10_episodes": sorted(d2e[f"{task}__demo_{i}"] for i in p10),
            "p25_episodes": sorted(d2e[f"{task}__demo_{i}"] for i in p25),
        }
    n_tasks = len(result[suite]) - 1
    print(f"{suite}: {n_tasks} tasks frozen, min pool {min_pool}, "
          f"ban_episodes {len(result[suite]['ban_episodes'])}")

OUT.parent.mkdir(parents=True, exist_ok=True)
blob = json.dumps(result, indent=1, sort_keys=True)
OUT.write_text(blob)
md5 = hashlib.md5(blob.encode()).hexdigest()
print(f"wrote {OUT}  md5={md5}")
