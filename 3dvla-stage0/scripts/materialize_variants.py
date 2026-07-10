#!/usr/bin/env python3
"""Materialize per-variant sampling specs from the mixture manifest.

Solves per-slice sample counts for a total conversation budget under the
40/20/20/10/10 weights, with explicit repetition (epoch) reporting — small
slices (anchor) repeat rather than silently shrinking the mixture (VLM3 warns
small sets overfit: repetition is capped and REPORTED, never hidden).

Variant semantics (plan §3.4):
  full            fine3d = all templates
  none            fine3d budget -> EXTRA embodied (coarse) sampling
  <axis>_only     fine3d filtered to that axis's templates; the REMAINDER of the
                  fine3d budget is refilled with embodied coarse data so total
                  token exposure stays matched across variants.

Output: sft/variants/<name>.json  {slices: {name: {files, take_convs, epochs,
template_filter}}, budget, weights}
"""
import argparse, json
from pathlib import Path

SFT = Path("/workspace/tingting/3dvla-data/sft")
MAX_EPOCHS = 3.0  # repetition cap for small slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1_000_000, help="total conversations")
    args = ap.parse_args()
    man = json.load(open(SFT / "mixture_manifest.json"))
    out_dir = SFT / "variants"
    out_dir.mkdir(exist_ok=True)

    for vname, tfilter in man["variants"].items():
        spec = {"budget": args.budget, "weights": {}, "slices": {}}
        for sname, s in man["slices"].items():
            target = round(args.budget * s["weight"])
            avail = s["convs_available"]
            # variant surgery on fine3d
            note = ""
            if sname == "fine3d":
                if tfilter == []:  # none-variant: budget goes to embodied
                    spec["slices"]["fine3d"] = {"files": [], "take_convs": 0,
                                                "epochs": 0, "template_filter": [],
                                                "note": "budget moved to embodied (coarse filler)"}
                    emb = spec["slices"].get("embodied")
                    if emb:
                        emb["take_convs"] += target
                        emb["epochs"] = round(emb["take_convs"] /
                                              man["slices"]["embodied"]["convs_available"], 2)
                        emb["note"] = "includes none-variant fine3d refill"
                    continue
                if tfilter:
                    note = (f"filtered to {tfilter}; est. availability scales by "
                            f"template share — trainer-side filter, refill from embodied "
                            f"to keep token exposure matched")
            take = min(target, int(avail * MAX_EPOCHS)) if avail else 0
            epochs = round(take / avail, 2) if avail else 0
            if epochs > 1.0:
                note = (note + f" REPEATS x{epochs}").strip()
            spec["slices"][sname] = {
                "files": [e["file"] for e in s["files"] if e["status"] == "ok"],
                "take_convs": take, "epochs": epochs,
                "template_filter": (tfilter if sname == "fine3d" else None),
                "note": note}
            spec["weights"][sname] = s["weight"]
        json.dump(spec, open(out_dir / f"{vname}.json", "w"), indent=1)
        parts = [f"{k}:{v['take_convs']//1000}K" + (f"(x{v['epochs']})" if v["epochs"] > 1 else "")
                 for k, v in spec["slices"].items()]
        print(f"{vname:<16} " + " ".join(parts))


if __name__ == "__main__":
    main()
