#!/usr/bin/env python3
"""Build the Stage-1 mixture manifest (plan §3.2) + capability-controlled variant
definitions (§3.4).

Five slices (initial weights, tunable):
  general   0.40  anti-forgetting (LLaVA-COCO, ShareGPT4V-COCO)
  embodied  0.20  Vlaser selection (coarse embodied QA/grounding)
  fine3d    0.20  OUR InternData-M1 QA (the studied variable)
  anchor    0.10  LIBERO in-domain QA (train demos 0-44 only)
  generic3d 0.10  Vlaser SPAR/VSI/vlm_3r (ScanNet-style indoor spatial)

Capability-controlled variants replace ONLY the fine3d slice's template subset;
all variants share the same keyframe images by construction (templates are
different questions on the same frames). The `none` variant fills fine3d's
budget with EXTRA embodied (coarse) data => headline contrast is
fine-grained-metric vs equal-budget coarse (user-approved design).

Output: sft/mixture_manifest.json with per-source conversation counts and
per-variant template filters. Sampling weights operate at the CONVERSATION
level; token-level rebalancing happens after tokenizer pass (Stage-1 config).
"""
import glob, json
from pathlib import Path

DATA = Path("/workspace/tingting/3dvla-data")
SFT = DATA / "sft"

SLICES = {
    "general": {"weight": 0.40, "files": ["general_llava_coco.jsonl",
                                          "general_sharegpt4v_coco.jsonl"]},
    # bounding_box_data: PARTITIONED 2026-07-08. Single-image records (134,864;
    # VG_100K + coco) verified 100% within declared [0,1000] -> reinstated.
    # Two-image records (50,000) dropped: boxes straddle the 1000 boundary and
    # cross unrelated photos; +1000-offset / concat-canvas hypotheses all fail ->
    # genuinely broken annotations. (Initial "92% dirty" claim was an artifact of
    # a number-concatenation bug in our own survey — corrected.)
    "embodied": {"weight": 0.20, "files": ["vlaser_refspatial_3d.jsonl",
                                           "vlaser_refspatial_simulator.jsonl",
                                           "vlaser_bounding_box_data_single.jsonl",
                                           "vlaser_affordance.jsonl",
                                           "vlaser_trajectory.jsonl",
                                           "vlaser_paco_lvis_v1_train.jsonl"]},
    "fine3d": {"weight": 0.20, "files": "M1_PACKS"},   # resolved from qa/ chunks
    "anchor": {"weight": 0.10, "files": ["libero_anchor_pack.jsonl",
                                         "libero_anchor_t5b_pack.jsonl"]},
    "generic3d": {"weight": 0.10, "files": ["vlaser_vsi_100k_merged_all.jsonl",
                                            "vlaser_vlm_3r_merged_all.jsonl"]},
}

# §3.4 variants: template filters applied to the fine3d slice only.
VARIANTS = {
    "full": None,  # all templates
    "none": [],    # fine3d budget refilled with EXTRA embodied (coarse) data
    "grounding_only": ["T1_grounding", "T5_where_to_act", "T5b_grasp_point"],
    "trace_only": ["T4_tcp_trace"],
    "metric3d_only": ["T2_metric3d_dist", "T3_depth_obj", "T3_depth_tcp"],
    "egomotion_only": ["T7_ego_motion"],
    # cross-view rides with metric3d in VLM3-style splits? NO — keep it a
    # separate probe axis inside `full` only; too small to carry its own variant.
}


def count(path):
    with open(path) as fh:
        return sum(1 for _ in fh)


def main():
    manifest = {"slices": {}, "variants": VARIANTS,
                "notes": {
                    "shared_keyframes": "all fine3d variants question the same frames",
                    "none_filler": "embodied slice upsampled to fill fine3d budget",
                    "coordinate_convention": "[0,2000) both axes, preamble in first turn",
                    "focal_unification": "K!=null sources only, dynamic-resolution mode",
                }}
    for name, spec in SLICES.items():
        files = spec["files"]
        if files == "M1_PACKS":
            files = sorted(glob.glob(str(SFT / "m1_chunk*_pack.jsonl")))
            files = [Path(f).name for f in files]
        entries = []
        total = 0
        for f in files:
            p = SFT / f
            if not p.exists():
                entries.append({"file": f, "convs": None, "status": "PENDING"})
                continue
            c = count(p)
            total += c
            entries.append({"file": f, "convs": c, "status": "ok"})
        manifest["slices"][name] = {"weight": spec["weight"], "files": entries,
                                    "convs_available": total}
    out = SFT / "mixture_manifest.json"
    json.dump(manifest, open(out, "w"), indent=1)
    print(f"manifest -> {out}\n")
    for name, s in manifest["slices"].items():
        pend = sum(1 for e in s["files"] if e["status"] == "PENDING")
        print(f"  {name:<10} w={s['weight']:.2f} convs={s['convs_available']:>9,}"
              f"{'  (+%d files pending)' % pend if pend else ''}")


if __name__ == "__main__":
    main()
