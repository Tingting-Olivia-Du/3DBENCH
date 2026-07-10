#!/usr/bin/env python3
"""Token calibration (plan §3.2: "token-level rebalancing happens after
tokenizer pass"). Conversation-count weights badly miss token shares because
per-conv token cost varies 5.6x across slices (measured, full_smoke5000 via
the real dataset class incl. focal unification):

  general 448 | embodied 658 | fine3d 802 | anchor 1423 | generic3d 2509

Keeping the total conversation budget N, solve for token total T and per-slice
conv counts n_i so that token shares hit the plan weights:
  n_i = w_i * T / m_i,  T = N / sum_i(w_i / m_i)

Writes sft/variants_tokcal/<v>.json (same schema as variants/, consumed by
materialize_variant_data.py via --spec-dir) + sft/token_stats.json.
Caveat recorded per spec: axis variants reuse the unfiltered fine3d mean
(filtered convs are shorter); acceptable for the full/none headline runs.
"""
import json
from pathlib import Path

SFT = Path("/workspace/tingting/3dvla-data/sft")
MEASURED = {  # mean tokens/conv, 2026-07-08, n=5000, stage1_dryrun_audit
    "general": 448, "embodied": 658, "fine3d": 802,
    "anchor": 1423, "generic3d": 2509,
}
MAX_EPOCHS = 3.0

json.dump({"means": MEASURED, "measured_on": "full_smoke5000.jsonl n=5000",
           "date": "2026-07-08", "truncated_gt8192": 0},
          open(SFT / "token_stats.json", "w"), indent=1)

man = json.load(open(SFT / "mixture_manifest.json"))
avail = {k: v["convs_available"] for k, v in man["slices"].items()}
out_dir = SFT / "variants_tokcal"
out_dir.mkdir(exist_ok=True)

# Anchor ALL variants to the full-variant token total: equal token exposure
# across variants is the controlled-comparison invariant (conv counts float).
full_spec = json.load(open(SFT / "variants/full.json"))
N = full_spec["budget"]
w_full = {k: s["take_convs"] / N for k, s in full_spec["slices"].items()}
T_FULL = N / sum(wi / MEASURED[k] for k, wi in w_full.items() if wi > 0)

for vf in sorted((SFT / "variants").glob("*.json")):
    spec = json.load(open(vf))
    # effective weights: none-variant moved fine3d budget into embodied
    w = {}
    for name, s in spec["slices"].items():
        w[name] = s["take_convs"] / N  # conv-share as implemented in the spec
    T = T_FULL
    tok_spec = {"budget": N, "token_total_est": int(T),
                "token_weights": w, "means": MEASURED,
                "caveat": ("axis variants: filtered fine3d convs are shorter than the "
                           "unfiltered mean used here -> their fine3d token share runs "
                           "low; close via measure-then-refill pass before those runs"),
                "slices": {}}
    for name, s in spec["slices"].items():
        s = dict(s)
        if w[name] > 0:
            n = round(w[name] * T / MEASURED[name])
            cap = int(avail[name] * MAX_EPOCHS)
            if n > cap:
                s["note"] = (s.get("note") or "") + f" TOKCAL capped {n}->{cap}"
                n = cap
            s["take_convs"] = n
            s["epochs"] = round(n / avail[name], 2) if avail[name] else 0
            s["token_share_est"] = round(n * MEASURED[name] / T, 3)
        tok_spec["slices"][name] = s
    json.dump(tok_spec, open(out_dir / vf.name, "w"), indent=1)
    parts = [f"{k}:{v['take_convs']//1000}K(x{v['epochs']})"
             for k, v in tok_spec["slices"].items() if v["take_convs"]]
    print(f"{vf.stem:<16} T≈{T/1e6:.0f}M tok  " + "  ".join(parts))
