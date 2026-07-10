#!/usr/bin/env python3
"""Adapt general multimodal SFT sources (the 40% anti-forgetting slice) to our
pack format: LLaVA-1.5 mix665k (COCO-image subset — no extra image downloads)
and ShareGPT4V-100K captions (COCO subset likewise).

These carry no metric supervision: K=null (no focal unification), no coordinate
preamble. TWO-DIALECT POLICY (final, plan-aligned): metric/embodied sources (M1, LIBERO,
Vlaser) use OUR [0,2000) convention (plan §3.1-3.2 mandates re-normalization);
this anti-forgetting slice KEEPS native conventions verbatim (RefCOCO-style
[0,1] floats, no preamble) so that (a) the RefCOCO forgetting audit stays
interpretable — a dialect overwrite would masquerade as capability loss, and
(b) the model learns context-conditioned conventions instead of anchoring to a
single dialect (the exact failure Exp0 exposed in zero-shot baselines).
A brief ×2000 conversion was applied on 2026-07-08 and REVERTED same day.

Output: sft/general_llava_coco.jsonl, sft/general_sharegpt4v_coco.jsonl
(image paths relative to general/coco/).
"""
import json
from pathlib import Path

DATA = Path("/workspace/tingting/3dvla-data")
ANN = DATA / "general" / "ann"
OUT = DATA / "sft"


def emit(fh, imgs, conv, source):
    fh.write(json.dumps({"images": imgs, "K": None, "source": source,
                         "templates": [source], "conversations": conv},
                        ensure_ascii=False) + "\n")


def llava():
    src = json.load(open(ANN / "llava_v1_5_mix665k.json"))
    n = 0
    with open(OUT / "general_llava_coco.jsonl", "w") as fh:
        for d in src:
            img = d.get("image")
            if not img or not img.startswith("coco/"):
                continue  # keep only COCO-backed samples (single image download)
            conv = d["conversations"]
            if not conv or conv[0]["from"] != "human":
                continue
            emit(fh, [img.replace("coco/", "coco/")], conv, "general_llava")
            n += 1
    print(f"llava_coco: {n} records")


def sharegpt4v():
    src = json.load(open(ANN / "sharegpt4v_instruct_gpt4-vision_cap100k.json"))
    n = 0
    with open(OUT / "general_sharegpt4v_coco.jsonl", "w") as fh:
        for d in src:
            img = d.get("image", "")
            if "coco/" not in img:
                continue
            conv = d["conversations"]
            emit(fh, [img[img.index("coco/"):]], conv, "general_sharegpt4v")
            n += 1
    print(f"sharegpt4v_coco: {n} records")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    llava()
    sharegpt4v()
