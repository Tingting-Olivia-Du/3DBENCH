#!/usr/bin/env python3
"""Pack our QA jsonl pools into ShareGPT-style conversations for Stage-1 SFT
(compatible with starVLA vlm_datasets / QwenVL conversation format).

Packing (plan §3.1 / VLM³): group QAs by image, up to PACK_MAX QAs per sample as
a multi-turn conversation — near-zero extra vision tokens, ~10x label density.
Multi-image templates (T6/T7) are emitted unpacked (one conversation each, two
images). The first human turn carries <image> token(s) and the convention
preamble; follow-up turns are bare questions.

Each sample carries source metadata for the focal-unification loader:
  {"images": [...], "K": [fx,fy,cx,cy] | null, "source": "m1"|"libero"|...}

Usage: pack_sharegpt.py --pools qa/chunk-000.jsonl ... --source m1 --out sft/m1_pack.jsonl
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PACK_MAX = 10
PREAMBLE = ("Pixel coordinates in this conversation are normalized to [0,2000) "
            "on both axes, origin at the image top-left. ")

K_REGISTRY = {  # per (source, view) pixel-domain intrinsics; None -> loader skips unification
    ("m1", "base_view"): [606.99, 606.49, 321.73, 260.76],
    ("m1", "base_view_2"): [606.99, 606.49, 321.73, 260.76],
    ("m1", "ego_view"): [609.79, 608.92, 323.59, 236.77],
    ("libero", "agentview"): [579.41, 579.41, 320.0, 240.0],  # constant across suites (verified)
    ("libero", "wrist"): [312.77, 312.77, 320.0, 240.0],
}


def ks_of(source, q, n_images):
    """Per-image K list (mixed-K multi-image e.g. LIBERO T6 agentview->wrist)."""
    v = str(q.get("view", ""))
    if source == "libero" and "->" in v:
        a, b = v.split("->")
        return [K_REGISTRY.get((source, a.strip())), K_REGISTRY.get((source, b.strip()))]
    k = k_of(source, q)
    return [k] * n_images


def k_of(source, q):
    v = str(q.get("view", ""))
    if source == "m1":
        for key in ("base_view_2", "base_view", "ego_view"):
            if key in v:
                return K_REGISTRY[("m1", key)]
        return None
    if source == "libero":
        for key in ("agentview", "wrist"):
            if key in v:
                return K_REGISTRY[("libero", key)]
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    qs = []
    for p in args.pools:
        with open(p) as fh:
            qs += [json.loads(l) for l in fh]
    single = defaultdict(list)
    multi = []
    for q in qs:
        (multi.append(q) if q.get("image2") else single[q["image"]].append(q))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_conv = n_qa = 0
    with open(out_path, "w") as fh:
        for img, items in single.items():
            rng.shuffle(items)
            for i in range(0, len(items), PACK_MAX):
                chunk = items[i:i + PACK_MAX]
                conv = []
                for j, q in enumerate(chunk):
                    prefix = ("<image>\n" + PREAMBLE) if j == 0 else ""
                    conv.append({"from": "human", "value": prefix + q["question"]})
                    conv.append({"from": "gpt", "value": q["answer"]})
                k = k_of(args.source, chunk[0])
                fh.write(json.dumps({
                    "images": [img], "K": k, "Ks": [k],
                    "source": args.source,
                    "templates": [q["template"] for q in chunk],
                    "conversations": conv}, ensure_ascii=False) + "\n")
                n_conv += 1
                n_qa += len(chunk)
        for q in multi:
            conv = [{"from": "human",
                     "value": "<image>\n<image>\n" + PREAMBLE + q["question"]},
                    {"from": "gpt", "value": q["answer"]}]
            ks = ks_of(args.source, q, 2)
            fh.write(json.dumps({
                "images": [q["image"], q["image2"]], "K": ks[0], "Ks": ks,
                "source": args.source, "templates": [q["template"]],
                "conversations": conv}, ensure_ascii=False) + "\n")
            n_conv += 1
            n_qa += 1
    print(f"packed {n_qa} QAs -> {n_conv} conversations "
          f"({n_qa/max(n_conv,1):.1f} QA/conv) -> {out_path}")


if __name__ == "__main__":
    main()
