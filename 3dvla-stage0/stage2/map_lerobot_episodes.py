#!/usr/bin/env python3
"""Build the (suite, task, demo_id) <-> lerobot episode_index mapping.

Why: the *_no_noops lerobot datasets kept only replay-successful demos
(e.g. libero_spatial 432/500) in non-task-major order, and record no original
demo ids. The Stage-2 low-data filter (and the bench-demo exclusion list)
need an exact mapping.

Method: ACTION-SEQUENCE fingerprint. (First-frame state matching FAILED:
t=0 is the reset arm pose, near-identical across all demos — per-demo
variation lives in object layout, not the arm.) The demo's action sequence
is its control signal and is unique; the no_noops conversion preserves the
non-noop rows in order (gripper dim possibly sign-flipped). We build an
inverted index over rounded action rows, find candidate (demo, t0) pairs
for an episode's first action, then verify the FULL episode is an ordered
subsequence of the demo's actions. Both gripper conventions are tried;
language is a sanity check only.

Output: lerobot_episode_map_v1.json
  {suite: {"demo_to_episode": {"<task>__demo<i>": ep_idx, ...},
           "bench_episodes": [...],       # demos 45-49 present in lerobot
           "missing_demos": [...]}}       # demos dropped by the replay
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HDF5_ROOT = Path("/workspace/tingting/LIBERO/libero/datasets")
LEROBOT_ROOT = Path("/workspace/tingting/playground_data/libero")
OUT = Path(__file__).parent / "lerobot_episode_map_v1.json"
SUITES = {
    "libero_spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object": "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10": "libero_10_no_noops_1.0.0_lerobot",
}
ATOL = 1e-5           # float64 (hdf5) vs float32 (lerobot) row tolerance
BENCH = set(range(45, 50))


def is_subseq(ep_act, da, t0):
    """episode rows form an ordered subsequence of demo actions, anchored at t0."""
    j = t0
    for row in ep_act:
        while j < len(da) and not np.allclose(da[j], row, atol=ATOL):
            j += 1
        if j >= len(da):
            return False
        j += 1
    return True

result = {}
for suite, ldir in SUITES.items():
    lpath = LEROBOT_ROOT / ldir
    episodes = [json.loads(l) for l in open(lpath / "meta" / "episodes.jsonl")]

    # --- load all demo action sequences + one big arm-dim row matrix
    # (rounded-tuple hashing FAILED: teleop values sit on rounding knife
    # edges after float32<->float64 trips; vector-scan with tolerance instead)
    demo_actions, fp_keys = {}, []
    rows, row_key, row_t = [], [], []
    for h5file in sorted((HDF5_ROOT / suite).glob("*.hdf5")):
        task = h5file.stem.replace("_demo", "")
        with h5py.File(h5file, "r") as f:
            for dname in f["data"]:
                key = f"{task}__{dname}"
                da = np.asarray(f[f"data/{dname}/actions"], dtype=np.float64)
                fp_keys.append(key)
                demo_actions[key] = da
                rows.append(da[:, :6])
                row_key.extend([key] * len(da))
                row_t.extend(range(len(da)))
    ALL = np.concatenate(rows)                     # [N, 6] arm dims
    row_key, row_t = np.array(row_key), np.array(row_t)

    demo_to_ep, problems = {}, []
    for ep in episodes:
        idx = ep["episode_index"]
        pq = lpath / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        df = pd.read_parquet(pq, columns=["action"])
        ep_raw = np.stack(df["action"].to_numpy()).astype(np.float64)
        cand = np.abs(ALL - ep_raw[0, :6]).sum(1) < 1e-6
        # lerobot gripper is {0,1}; hdf5 demo gripper is {-1,+1}: demo = 1-2*ep
        ep_act = ep_raw.copy()
        ep_act[:, 6] = 1.0 - 2.0 * ep_act[:, 6]
        hits = [key for key, t0 in zip(row_key[cand], row_t[cand])
                if is_subseq(ep_act, demo_actions[key], int(t0))]
        uniq = sorted(set(hits))
        if len(uniq) != 1:
            problems.append((idx, "n_matches", len(uniq), uniq[:3]))
            continue
        key = uniq[0]
        # language sanity
        lang = ep["tasks"][0].lower().replace(" ", "_")
        if lang not in key.lower():
            problems.append((idx, key, "language-mismatch", ep["tasks"][0]))
            continue
        assert key not in demo_to_ep, f"demo {key} matched twice"
        demo_to_ep[key] = idx

    all_keys = set(fp_keys)
    matched = set(demo_to_ep)
    missing = sorted(all_keys - matched)
    bench_eps = sorted(v for k, v in demo_to_ep.items()
                       if int(k.rsplit("demo_", 1)[1]) in BENCH)
    result[suite] = {
        "n_episodes": len(episodes), "n_matched": len(demo_to_ep),
        "n_demos_total": len(fp_keys), "missing_demos": missing,
        "bench_episodes": bench_eps,
        "demo_to_episode": dict(sorted(demo_to_ep.items())),
    }
    print(f"{suite}: episodes={len(episodes)} matched={len(demo_to_ep)} "
          f"missing_demos={len(missing)} bench_eps={len(bench_eps)} "
          f"problems={len(problems)}")
    for p in problems[:5]:
        print("   PROBLEM:", p)
    assert not problems, f"{suite}: {len(problems)} unmatched/ambiguous episodes"

OUT.write_text(json.dumps(result, indent=1))
print(f"wrote {OUT}")
