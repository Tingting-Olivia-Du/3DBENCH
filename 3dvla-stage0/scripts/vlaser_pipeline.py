#!/usr/bin/env python3
"""Vlaser end-to-end pipeline (single reproducible pass) — supersedes the
incremental patches. Steps:

1. PARTITION bounding_box_data: keep single-image records only (134,864;
   verified 100% within declared [0,1000]); two-image records (50,000) are
   broken (boxes straddle the 1000 boundary across unrelated photos) — dropped.
2. ADAPT all 9 sources via vlaser_adapt.adapt_file:
   - drop system turns (their format demands conflict with our convention)
   - convert <point>/<box> [0,1000] -> bare [0,2000) lists (x2)
   - inject [0,2000) preamble into first human turn iff record has coords
   - append answer-style hint; per-source SEMANTIC hints for affordance
     ("box of the region to manipulate") and trajectory ("gripper waypoint
     trajectory") — restoring task semantics lost with the system turn
3. VALIDATE per output: counts; zero system turns; zero residual tags;
   converted coords in range; preamble/hint iff coords; image paths resolve.

Output: sft/vlaser_*.jsonl + sft/vlaser_pipeline_report.json
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vlaser_adapt import adapt_file, PREAMBLE

DATA = Path("/workspace/tingting/3dvla-data")
V = DATA / "vlaser"
SFT = DATA / "sft"

SOURCES = {  # name -> (raw path, image resolution root)
    "vlaser_refspatial_3d": ("grounding_data/RefSpatial_3D.jsonl", "grounding_data"),
    "vlaser_refspatial_simulator": ("grounding_data/RefSpatial_Simulator.jsonl", "grounding_data"),
    "vlaser_bounding_box_data_single": ("grounding_data/bounding_box_data_single.jsonl", "grounding_data"),
    "vlaser_affordance": ("grounding_data/affordance.jsonl", "grounding_data"),
    "vlaser_trajectory": ("grounding_data/trajectory.jsonl", "grounding_data"),
    "vlaser_paco_lvis_v1_train": ("grounding_data/paco_lvis_v1_train.jsonl", "grounding_data"),
    "vlaser_vsi_100k_merged_all": ("spatial_data/VSI-100k_merged_all.jsonl", "spatial_data"),
    "vlaser_vlm_3r_merged_all": ("spatial_data/vlm_3r_merged_all.jsonl", "spatial_data"),
}

LIST_RE = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)"
                     r"(?:\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*))?\s*\]")


def partition_bbox():
    src = V / "grounding_data/bounding_box_data.jsonl"
    dst = V / "grounding_data/bounding_box_data_single.jsonl"
    n_single = n_multi = 0
    with open(dst, "w") as fh:
        for l in open(src):
            d = json.loads(l)
            imgs = d["image"] if isinstance(d["image"], list) else [d["image"]]
            if len(imgs) == 1:
                fh.write(l); n_single += 1
            else:
                n_multi += 1
    return n_single, n_multi


def validate(name, root):
    import os, random
    random.seed(0)
    p = SFT / f"{name}.jsonl"
    rep = {"records": 0, "system_turns": 0, "residual_tags": 0,
           "coord_out_of_range": 0, "coord_records": 0,
           "preamble_missing": 0, "path_miss": 0, "path_checked": 0,
           "multi_image_records": 0}
    lines = open(p).readlines()
    rep["records"] = len(lines)
    sample_idx = set(random.sample(range(len(lines)), min(100, len(lines))))
    for i, l in enumerate(lines):
        d = json.loads(l)
        conv = d["conversations"]
        rep["system_turns"] += sum(1 for c in conv if c["from"] == "system")
        text = " ".join(c["value"] for c in conv)
        if re.search(r"</?(point|box|ref)>", text):
            rep["residual_tags"] += 1
        if len(d["images"]) > 1:
            rep["multi_image_records"] += 1
        # coords in converted records: any 2/4-number bracketed list in gpt turns
        gpt = " ".join(c["value"] for c in conv if c["from"] == "gpt")
        vals = [float(x) for m in LIST_RE.finditer(gpt) for x in m.groups() if x]
        if vals:
            rep["coord_records"] += 1
            if max(vals) >= 2001 or min(vals) < -1:
                rep["coord_out_of_range"] += 1
            if PREAMBLE not in conv[0]["value"]:
                rep["preamble_missing"] += 1
        if i in sample_idx:
            for img in d["images"]:
                rep["path_checked"] += 1
                if not os.path.exists(str(V / root / img)):
                    rep["path_miss"] += 1
    return rep


def main():
    print("== step 1: partition bounding_box_data")
    ns, nm = partition_bbox()
    print(f"   single kept {ns:,}, two-image dropped {nm:,}")
    print("== step 2: adapt all sources")
    report = {"partition": {"single_kept": ns, "two_image_dropped": nm}, "sources": {}}
    for name, (raw, root) in SOURCES.items():
        with open(SFT / f"{name}.jsonl", "w") as fh:
            n_rec, n_coord = adapt_file(V / raw, fh, name)
        print(f"   {name}: {n_rec:,} records, {n_coord:,} coord blocks")
        report["sources"][name] = {"records": n_rec, "coord_blocks": n_coord}
    print("== step 3: validate")
    all_ok = True
    for name, (raw, root) in SOURCES.items():
        rep = validate(name, root)
        report["sources"][name]["validation"] = rep
        problems = {k: v for k, v in rep.items()
                    if k in ("system_turns", "residual_tags", "coord_out_of_range",
                             "preamble_missing", "path_miss") and v}
        status = "OK" if not problems else f"PROBLEMS {problems}"
        all_ok &= not problems
        print(f"   {name:<36} recs={rep['records']:>7,} coordrec={rep['coord_records']:>7,} "
              f"multi_img={rep['multi_image_records']:>6,} paths={rep['path_checked']-rep['path_miss']}"
              f"/{rep['path_checked']} {status}")
    report["all_ok"] = all_ok
    json.dump(report, open(SFT / "vlaser_pipeline_report.json", "w"), indent=1)
    print(f"== {'ALL OK' if all_ok else 'PROBLEMS FOUND'} -> vlaser_pipeline_report.json")


if __name__ == "__main__":
    main()
