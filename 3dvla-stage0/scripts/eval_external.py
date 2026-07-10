#!/usr/bin/env python3
"""External benchmark battery: CV-Bench / ERQA / RoboSpatial-Home / MMMU-val / VSI-Bench.

Capability-profile matrix for the Stage-1 variants: these benchmarks are outside
BOTH variants' training distributions, so neither row holds an in-domain advantage
(unlike the M1 half of our own bench, which is in-domain for `full`).

Presentation protocol — IDENTICAL inputs for every model row (bias-free):
external images carry no K, so no focal unification is possible; we mirror the
training no-K contract instead: <=4 images -> downscale-only to area<=378,000 px
each (SINGLE_NOK_AREA); >4 images (video regime) -> area<=72,000 px each
(VIDEO_FRAME_AREA). VSI videos: 16 uniform frames, pre-extracted at the capped area.

Usage:
  python eval_external.py --model base --bench cvbench --device cuda:4
  python eval_external.py --model stage1 --ckpt <safetensors> --bench vsi --device cuda:5
  python eval_external.py --prep-vsi          # one-off frame extraction after download
Writes <data>/external_bench/preds_<bench>_<model>.jsonl (qid/subtask/gt/pred, resumable).
Scoring: score_external.py
"""
import argparse, ast, io, json, math, os, re, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/angli/hf_cache")

DATA = Path("/workspace/tingting/3dvla-data/external_bench")
HUB = Path("/root/angli/hf_cache/hub")
CVB = HUB / "datasets--nyu-visionx--CV-Bench/snapshots/bc284db50d036958861cb60cdd7b77612052ce0d"
ERQA = HUB / "datasets--FlagEval--ERQA/snapshots/6bf393250b0981c3f6dd77eb58b6d6879038633f/data/test-00000-of-00001.parquet"
RSH = HUB / "datasets--chanhee-luke--RoboSpatial-Home/snapshots/7ef1b43b20087ef284e0e9017ef51dd143814c59/data"
MMMU_SNAP = HUB / "datasets--MMMU--MMMU/snapshots/4619a102cf5ad2da1abf7e220fde1258d2434cb7"

SINGLE_NOK_AREA = 378_000   # == training no-K single-image cap (vlm_datasets_3d)
VIDEO_FRAME_AREA = 72_000   # == training no-K video-frame cap
VSI_FRAMES = 16

MC_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def cap_image(im, area):
    from PIL import Image
    im = im.convert("RGB")
    w, h = im.size
    s = min(1.0, math.sqrt(area / float(w * h)))
    if s < 1.0:
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    return im


def from_bytes(obj):
    from PIL import Image
    if isinstance(obj, dict):
        obj = obj["bytes"]
    return Image.open(io.BytesIO(obj))


# ---------------------------------------------------------------- bench loaders
def iter_cvbench():
    import pandas as pd
    for part in ["test_2d.parquet", "test_3d.parquet"]:
        df = pd.read_parquet(CVB / part)
        for _, r in df.iterrows():
            yield {"qid": f"cv_{r['idx']}", "subtask": f"{r['type']}_{r['task']}",
                   "images": [from_bytes(r["image"])],
                   "prompt": r["prompt"] + MC_SUFFIX,
                   "gt": str(r["answer"]).strip("() ")}


def iter_erqa():
    import pandas as pd
    df = pd.read_parquet(ERQA)
    for _, r in df.iterrows():
        q = r["question"]
        if "answer directly" not in q.lower():
            q += "\nPlease answer directly with only the letter of the correct option and nothing else."
        yield {"qid": str(r["question_id"]), "subtask": str(r["question_type"]),
               "images": [from_bytes(im) for im in r["images"]],
               "prompt": q, "gt": str(r["answer"]).strip()}


def iter_robospatial():
    import pandas as pd
    for sub in ["compatibility", "configuration", "context"]:
        df = pd.read_parquet(RSH / f"{sub}-00000-of-00001.parquet")
        for i, r in df.iterrows():
            gt = str(r["answer"]) if sub != "context" else "__MASK__"
            yield {"qid": f"rs_{sub}_{i}", "subtask": sub,
                   "images": [from_bytes(r["img"])],
                   "prompt": str(r["question"]), "gt": gt}


def iter_mmmu():
    import pandas as pd
    parqs = sorted(MMMU_SNAP.glob("*/validation-*.parquet"))
    assert len(parqs) == 30, f"expected 30 MMMU subjects, found {len(parqs)}"
    for pq in parqs:
        df = pd.read_parquet(pq)
        for _, r in df.iterrows():
            refs = re.findall(r"<image (\d)>", r["question"])
            opts = r["options"]
            if isinstance(opts, str):
                opts = ast.literal_eval(opts)
            opts = list(opts)
            for o in opts:
                refs += re.findall(r"<image (\d)>", str(o))
            ks = sorted({int(k) for k in refs}) or [1]
            images = [from_bytes(r[f"image_{k}"]) for k in ks
                      if r.get(f"image_{k}") is not None]
            if r["question_type"] == "multiple-choice":
                letters = "ABCDEFGHI"
                body = "\n".join(f"({letters[j]}) {o}" for j, o in enumerate(opts))
                prompt = f"{r['question']}\n{body}{MC_SUFFIX}"
            else:
                prompt = r["question"] + "\nAnswer with a single word, phrase, or number."
            yield {"qid": str(r["id"]), "subtask": r["question_type"],
                   "images": images, "prompt": prompt, "gt": str(r["answer"])}


def vsi_frame_dir(dataset, scene):
    return DATA / "vsi_frames" / dataset / scene


def prep_vsi():
    """One-off: extract VSI_FRAMES uniform frames per video at the capped area."""
    import cv2
    items = [json.loads(l) for l in open(DATA_VSI_JSONL())]
    scenes = {(it["dataset"], str(it["scene_name"])) for it in items}
    print(f"[prep] {len(scenes)} scenes")
    for n, (ds, sc) in enumerate(sorted(scenes)):
        out = vsi_frame_dir(ds, sc)
        if (out / ".done").exists():
            continue
        vid = DATA / "vsi_videos" / ds / f"{sc}.mp4"
        if not vid.exists():
            hits = list((DATA / "vsi_videos").rglob(f"{sc}.mp4"))
            assert hits, f"video not found for {ds}/{sc}"
            vid = hits[0]
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = [int(i * (total - 1) / (VSI_FRAMES - 1)) for i in range(VSI_FRAMES)]
        out.mkdir(parents=True, exist_ok=True)
        for j, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            assert ok, f"decode fail {vid} @{fi}"
            h, w = fr.shape[:2]
            s = min(1.0, math.sqrt(VIDEO_FRAME_AREA / float(w * h)))
            if s < 1.0:
                fr = cv2.resize(fr, (max(28, int(w * s)), max(28, int(h * s))),
                                interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"f{j:02d}.jpg"), fr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        cap.release()
        (out / ".done").touch()
        if (n + 1) % 20 == 0:
            print(f"[prep] {n+1}/{len(scenes)}", flush=True)
    print("[prep] VSI frames ready")
    (DATA / ".vsi_frames_ready").touch()


def DATA_VSI_JSONL():
    p = list(HUB.glob("datasets--nyu-visionx--VSI-Bench/snapshots/*/test.jsonl"))
    assert p, "VSI test.jsonl not downloaded yet"
    return p[0]


def iter_vsi():
    from PIL import Image
    assert (DATA / ".vsi_frames_ready").exists(), "run --prep-vsi first"
    for i, l in enumerate(open(DATA_VSI_JSONL())):
        it = json.loads(l)
        qt = it["question_type"]
        fdir = vsi_frame_dir(it["dataset"], str(it["scene_name"]))
        images = [Image.open(fdir / f"f{j:02d}.jpg") for j in range(VSI_FRAMES)]
        q = ("These are frames of a video.\n" + it["question"])
        opts = it.get("options")
        if opts is not None and len(opts):
            q += "\n" + "\n".join(str(o) for o in opts) + MC_SUFFIX
            gt = str(it["ground_truth"]).strip()
        else:
            q += "\nAnswer with a single number."
            gt = str(it["ground_truth"])
        yield {"qid": f"vsi_{it.get('id', i)}", "subtask": qt,
               "images": images, "prompt": q, "gt": gt}


BENCHES = {"cvbench": iter_cvbench, "erqa": iter_erqa,
           "robospatial": iter_robospatial, "mmmu": iter_mmmu, "vsi": iter_vsi}


# ---------------------------------------------------------------- model runners
def load_runner(name, ckpt, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16)
    if name == "stage1":
        from safetensors.torch import load_file
        sd = load_file(ckpt)
        prefix = "qwen_vl_interface.model."
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        assert all("rotary_emb" in k or "position_ids" in k for k in missing), \
            f"missing non-buffer keys: {missing[:5]}"
    model = model.to(device).eval()

    def run(images, prompt, max_new_tokens=96):
        area = SINGLE_NOK_AREA if len(images) <= 4 else VIDEO_FRAME_AREA
        content = [{"type": "image", "image": cap_image(im, area)} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "stage1"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bench", choices=BENCHES)
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prep-vsi", action="store_true")
    args = ap.parse_args()

    if args.prep_vsi:
        prep_vsi()
        return
    assert args.model and args.bench
    if args.model == "stage1":
        assert args.ckpt, "--ckpt required for stage1"

    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / f"preds_{args.bench}_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in open(out_path)}
    run = load_runner(args.model, args.ckpt, args.device)
    n_new, n_seen = 0, 0
    with open(out_path, "a") as fh:
        for item in BENCHES[args.bench]():
            n_seen += 1
            if args.limit and n_seen > args.limit:
                break
            if item["qid"] in done:
                continue
            try:
                # MMMU: models legitimately reason before the letter; don't truncate
                mnt = 512 if args.bench == "mmmu" else 96
                pred = run(item["images"], item["prompt"], max_new_tokens=mnt)
            except Exception as e:
                pred = f"__ERROR__ {type(e).__name__}: {e}"
            fh.write(json.dumps({"qid": item["qid"], "subtask": item["subtask"],
                                 "gt": item["gt"], "pred": pred}) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 100 == 0:
                print(f"[{args.bench}/{args.model}] {n_new} done", flush=True)
    print(f"[{args.bench}/{args.model}] finished: +{n_new} -> {out_path}")


if __name__ == "__main__":
    main()
