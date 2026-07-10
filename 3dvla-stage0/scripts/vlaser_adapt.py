#!/usr/bin/env python3
"""Adapt Vlaser jsonl sources to our SFT pack format.

Coordinate convention conversion: Vlaser uses [0,1000] inside <point>/<box> tags;
we use bare lists normalized to [0,2000) with a preamble. This rewrites BOTH
question and answer text: tags stripped, every coordinate x2, our list syntax.
Multi-image samples keep their image lists; MCQ/text answers pass through.

Output records: {"images": [...], "K": null, "source": "vlaser_<name>",
"templates": ["vlaser_<name>"], "conversations": [...]} — same schema as
pack_sharegpt.py so the mixture assembler treats all sources uniformly.
Image paths are emitted RELATIVE (resolved against --images-root at load time);
use --check-images to verify resolution once tars are unpacked.
"""
import argparse, json, re
from pathlib import Path

PREAMBLE = ("Pixel coordinates in this conversation are normalized to [0,2000) "
            "on both axes, origin at the image top-left. ")

TAG_RE = re.compile(r"<(point|box)>\s*(\[\[.*?\]\])\s*</\1>", re.S)


def convert_coords(text):
    """<point>[[x,y]]</point> / <box>[[...],...]</box> with [0,1000] -> our bare
    [0,2000) lists. Returns (new_text, n_converted)."""
    n = 0

    def repl(m):
        nonlocal n
        try:
            arr = json.loads(m.group(2))
        except json.JSONDecodeError:
            return m.group(0)
        scaled = [[round(v * 2.0, 1) for v in row] for row in arr]
        n += 1
        out = scaled[0] if len(scaled) == 1 else scaled
        return json.dumps(out)

    return TAG_RE.sub(repl, text), n


HINT = {"point": " Answer with point coordinates [x,y].",
        "box": " Answer with bounding boxes [x0,y0,x1,y1]."}
# per-source semantic hints: the dropped system turns carried TASK semantics
# (what to box / what the points mean) — restore them here (user-caught gap).
SOURCE_HINT = {
    "vlaser_affordance": " Answer with the bounding box of the region to "
                         "manipulate for this task, [x0,y0,x1,y1].",
    "vlaser_trajectory": " Answer with the 2D trajectory of points the gripper "
                         "should follow to execute the task, as [x,y] waypoints.",
}


# sources whose multi-image records are PACKED unrelated turns (turn k <-> image
# slot k): split into one record per unique image (merging same-image turns into
# one conversation with a single <image>) — kills the ~61% duplicate-slot vision
# token waste in refspatial_3d and makes affordance/trajectory records
# self-contained (user-caught: "why 4 frames"). Genuine multi-frame reasoning
# sources (vsi/vlm_3r) are NOT split.
SPLIT_PER_IMAGE = {"vlaser_refspatial_3d", "vlaser_affordance", "vlaser_trajectory"}

# generic3d metric consistency (user-caught contradiction): metric-numeric QA
# without K would teach a focal-inconsistent metric scale. ScanNet frames ship at
# native 1296x968 -> published constant intrinsics apply exactly; ScanNet++ has
# per-scene intrinsics we cannot verify offline -> its metric-NUMERIC questions
# are dropped (MCQ/qualitative kept).
SPATIAL_SOURCES = {"vlaser_vsi_100k_merged_all", "vlaser_vlm_3r_merged_all"}
K_SCANNET = [1170.19, 1170.19, 647.75, 483.75]  # ScanNet RGB @1296x968 (published)
METRIC_Q = re.compile(r"\bin (meters|centimeters)\b")
NUM_A = re.compile(r"^-?\d+\.?\d*$")
IMG_TAG = re.compile(r"(Image-\d+:\s*)?<image>\s*")


def split_records(d):
    """Yield (images, conversations) per unique image slot group."""
    conv = [c for c in d.get("conversations", []) if c["from"] != "system"]
    imgs = d.get("image") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    pairs = [(conv[i], conv[i+1]) for i in range(0, len(conv)-1, 2)
             if conv[i]["from"] == "human" and conv[i+1]["from"] == "gpt"]
    if len(pairs) != len(imgs) or not imgs:
        yield imgs, conv  # structure mismatch: pass through unsplit
        return
    groups = {}
    for k, pr in enumerate(pairs):
        groups.setdefault(imgs[k], []).append(pr)
    for img, prs in groups.items():
        new = []
        for j, (h, g) in enumerate(prs):
            hv = IMG_TAG.sub("", h["value"]).strip()
            if j == 0:
                hv = "<image>\n" + hv
            new.append({"from": "human", "value": hv})
            new.append({"from": "gpt", "value": g["value"]})
        yield [img], new


def adapt_file(src, out_fh, source_name):
    n_rec = n_conv = 0
    for line in open(src):
        d = json.loads(line)
        if source_name in SPLIT_PER_IMAGE:
            units = list(split_records(d))
        else:
            conv0 = [c for c in d.get("conversations", []) if c["from"] != "system"]
            imgs0 = d.get("image") or d.get("video") or []
            if isinstance(imgs0, str):
                imgs0 = [imgs0]
            units = [(imgs0, conv0)] if conv0 else []
        for imgs, conv in units:
            if not conv:
                continue
            raw_gpt = " ".join(c["value"] for c in conv if c["from"] == "gpt")
            style = "point" if "<point>" in raw_gpt else ("box" if "<box>" in raw_gpt else None)
            new_conv = []
            rec_coords = 0
            for c in conv:
                v, k = convert_coords(c["value"])
                rec_coords += k
                new_conv.append({"from": c["from"], "value": v})
            n_conv += rec_coords
            if rec_coords and new_conv and new_conv[0]["from"] == "human":
                v0 = new_conv[0]["value"]
                v0 = (re.sub(r"((?:<image>\s*|Image-\d+:\s*<image>\s*)+)", r"\1" + PREAMBLE,
                             v0, count=1) if "<image>" in v0 else PREAMBLE + v0)
                if source_name in SOURCE_HINT:
                    v0 = v0.rstrip() + SOURCE_HINT[source_name]
                elif style:
                    v0 = v0.rstrip() + HINT[style]
                new_conv[0]["value"] = v0
            K = None
            if source_name in SPATIAL_SOURCES:
                is_pp = any("scannetpp" in i for i in imgs)
                if not is_pp:
                    K = K_SCANNET
                elif (METRIC_Q.search(new_conv[0]["value"])
                      and any(NUM_A.match(c["value"].strip()) for c in new_conv
                              if c["from"] == "gpt")):
                    continue  # scannetpp metric-numeric: unverifiable K -> drop
            out_fh.write(json.dumps({"images": imgs, "K": K,
                                     "Ks": ([K] * len(imgs) if K else None),
                                     "source": source_name,
                                     "templates": [source_name],
                                     "conversations": new_conv}, ensure_ascii=False) + "\n")
            n_rec += 1
    return n_rec, n_conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="jsonl paths")
    ap.add_argument("--out-dir", default="/workspace/tingting/3dvla-data/sft")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        name = "vlaser_" + Path(src).stem.lower().replace("-", "_")
        out = Path(args.out_dir) / f"{name}.jsonl"
        with open(out, "w") as fh:
            n_rec, n_coord = adapt_file(src, fh, name)
        print(f"{name}: {n_rec} records, {n_coord} coord blocks converted -> {out}")


if __name__ == "__main__":
    main()
