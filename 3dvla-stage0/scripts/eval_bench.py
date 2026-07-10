#!/usr/bin/env python3
"""Experiment 0: zero-shot audit of VLMs on the in-domain 3D benchmark.

Runners: qwen3vl4b (Qwen/Qwen3-VL-4B-Instruct), internvla-m1 (InternRobotics/InternVLA-M1,
Qwen2.5-VL-3B based), vlaser2b (OpenGVLab/Vlaser-2B, InternVL3 arch, trust_remote_code).
Images are presented RAW (no focal unification) — zero-shot models have no convention;
the unified-convention probe applies only to our own Stage-1 variants later.

Usage: python eval_bench.py --model qwen3vl4b --device cuda:4 [--limit 50]
Writes <data>/benchmark/preds_<model>.jsonl (one {qid, pred} per line, resumable).
"""
import argparse, json, sys
from pathlib import Path

import torch


def load_qwen(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    def run(images, question):
        content = [{"type": "image", "image": p} for p in images]
        content.append({"type": "text", "text": question})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()
    return run


def load_internvl(model_id, device):
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(MEAN, STD)])

    def run(images, question):
        pv = torch.stack([tf(Image.open(p)) for p in images]).to(torch.bfloat16).to(device)
        npl = [1] * len(images)
        q = "".join(f"Image-{i+1}: <image>\n" for i in range(len(images))) + question \
            if len(images) > 1 else "<image>\n" + question
        with torch.no_grad():
            return model.chat(tok, pv, q, dict(max_new_tokens=128, do_sample=False),
                              num_patches_list=npl).strip()
    return run


# view -> fx for focal unification of OUR Stage-1 variants (F_tgt=600).
# Zero-shot baselines stay raw (they have no learned convention to preserve).
FX = {"base_view": 606.99, "base_view_2": 606.99, "ego_view": 609.79,
      "agentview": 579.41, "wrist": 312.77}
F_TGT = 600.0


def load_stage1(ckpt_path, device):
    """Our Stage-1 checkpoint: base Qwen3-VL-4B architecture + trained weights
    (state dict prefixed `qwen_vl_interface.model.` by the starVLA wrapper),
    evaluated UNDER THE TRAINING CONTRACT: images focal-unified to F_tgt=600
    per view before inference (deployment-contract probe)."""
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16)
    sd = load_file(ckpt_path)
    prefix = "qwen_vl_interface.model."
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all("rotary_emb" in k or "position_ids" in k for k in missing), \
        f"missing non-buffer keys: {missing[:5]}"
    model = model.to(device).eval()

    def unify(path, view):
        im = Image.open(path).convert("RGB")
        fx = FX.get(view)
        if fx:
            s = F_TGT / fx
            im = im.resize((max(28, round(im.size[0]*s)),
                            max(28, round(im.size[1]*s))), Image.BICUBIC)
        return im

    def run(images, question, views=None):
        views = views or [None] * len(images)
        content = [{"type": "image", "image": unify(p, v)}
                   for p, v in zip(images, views)]
        content.append({"type": "text", "text": question})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()
    return run


RUNNERS = {
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", load_qwen),
    "internvla-m1": ("InternRobotics/InternVLA-M1", load_qwen),
    "vlaser2b": ("OpenGVLab/Vlaser-2B", load_internvl),
    "stage1": (None, load_stage1),  # requires --ckpt
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=RUNNERS)
    ap.add_argument("--data", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--ckpt", default=None, help="stage1 safetensors checkpoint")
    args = ap.parse_args()

    bench_path = args.bench or f"{args.data}/benchmark/m1_bench.jsonl"
    items = [json.loads(l) for l in open(bench_path)]
    if args.limit:
        items = items[: args.limit]
    out_path = Path(f"{args.data}/benchmark/preds_{args.model}{args.suffix}.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in open(out_path)}
    model_id, loader = RUNNERS[args.model]
    if args.model == "stage1":
        assert args.ckpt, "--ckpt required for stage1"
        run = loader(args.ckpt, args.device)
    else:
        run = loader(model_id, args.device)
    n_new = 0
    with open(out_path, "a") as fh:
        for i, q in enumerate(items):
            if q["qid"] in done:
                continue
            images = [q["image"]] + ([q["image2"]] if q.get("image2") else [])
            views = q["view"].split("->") if "->" in q.get("view", "") else [q.get("view")]
            try:
                pred = (run(images, q["question"], views=views)
                        if args.model == "stage1" else run(images, q["question"]))
            except Exception as e:
                pred = f"__ERROR__ {type(e).__name__}: {e}"
            fh.write(json.dumps({"qid": q["qid"], "template": q["template"],
                                 "pred": pred}) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 50 == 0:
                print(f"[{args.model}] {n_new} done ({i+1}/{len(items)})", flush=True)
    print(f"[{args.model}] finished: +{n_new} preds -> {out_path}")


if __name__ == "__main__":
    main()
