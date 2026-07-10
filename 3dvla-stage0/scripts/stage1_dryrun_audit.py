#!/usr/bin/env python3
"""CPU dry-run audit of the Stage-1 data pipeline (no GPU, no model weights).

Feeds the materialized smoke file through the REAL training dataset class
(Stage1MixtureDataset) and verifies, per sample:
  1. LABEL FIDELITY — decoded unmasked label spans must equal the record's gpt
     answers verbatim (catches chat-template drift / unmask offset bugs)
  2. no zero-supervision, truncation counted vs model_max_length
  3. focal unification arrived at the processor: image_grid_thw implies the
     unified (not native) resolution; budget assertion never fired
  4. per-slice token statistics (drives the token-calibration step)
  5. dialect spot samples printed for eyeball check
Finally runs the real collator on a batch for shape sanity.

Run inside the starVLA env from the starVLA repo root:
  python /workspace/tingting/3dvla-stage0/scripts/stage1_dryrun_audit.py \
      --ann /workspace/tingting/3dvla-data/sft/materialized/full_smoke5000.jsonl -n 400
"""
import argparse, json, re, sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, "/workspace/tingting/starVLA")
import torch  # noqa: E402
import transformers  # noqa: E402
from starVLA.dataloader.vlm_datasets import IGNORE_INDEX, DataCollatorForSupervisedDataset  # noqa: E402
from starVLA.dataloader.vlm_datasets_3d import Stage1MixtureDataset  # noqa: E402

BASE_VLM = "Qwen/Qwen3-VL-4B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("-n", type=int, default=400)
    args = ap.parse_args()

    processor = transformers.AutoProcessor.from_pretrained(BASE_VLM)
    processor.tokenizer.model_max_length = 8192
    ns = SimpleNamespace(
        annotation_path=args.ann, model_type="qwen3vl",
        model_max_length=8192, max_pixels=1254400, min_pixels=784,
        video_min_pixels=784, video_max_pixels=401408,
        video_min_frames=4, video_max_frames=32, video_fps=1,
        per_device_batch_size=2, dataset_use="materialized")
    ds = Stage1MixtureDataset(processor, ns)
    tok = processor.tokenizer

    stats = defaultdict(lambda: {"n": 0, "tokens": 0, "sup": 0, "max": 0})
    mismatches, truncs = [], 0
    n = min(args.n, len(ds))
    for i in range(n):
        rec = ds.list_data_dict[i]
        item = ds[i]
        ids = item["input_ids"][0]
        labels = item["labels"][0]
        L = ids.size(0)
        sl = rec.get("slice", "?")
        s = stats[sl]
        s["n"] += 1; s["tokens"] += L; s["max"] = max(s["max"], L)
        s["sup"] += int((labels != IGNORE_INDEX).sum())
        if L > 8192:
            truncs += 1

        # 1. label fidelity: unmasked spans == gpt answers verbatim
        lab = labels.tolist()
        spans, cur = [], []
        for t, m in zip(ids.tolist(), lab):
            if m != IGNORE_INDEX:
                cur.append(t)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        answers = [c["value"] for c in rec["conversations"] if c["from"] == "gpt"]
        if len(spans) != len(answers):
            mismatches.append((i, sl, f"span-count {len(spans)} vs answers {len(answers)}"))
            continue
        for sp, ans in zip(spans, answers):
            got = tok.decode(sp, skip_special_tokens=False)
            got = got.replace("<|im_end|>", "").strip()
            if got != ans.strip():
                mismatches.append((i, sl, f"'{got[:70]}' != '{ans[:70]}'"))
                break

        # 3. unified resolution reached the processor (single-image K records)
        if rec.get("K") and len(rec["image"]) == 1 and "image_grid_thw" in item:
            t_, gh, gw = item["image_grid_thw"][0].tolist()
            patch = getattr(processor.image_processor, "patch_size", 16)  # Qwen3-VL: 16
            got_px = (gh * patch) * (gw * patch)
            fx = rec["K"][0]
            import PIL.Image
            w0, h0 = PIL.Image.open(rec["image"][0]).size
            want = (w0 * (600.0 / fx)) * (h0 * (600.0 / fx))
            if abs(got_px - want) / want > 0.10:
                mismatches.append((i, sl, f"grid px {got_px} vs unified target {want:.0f}"))

    print(f"\n== label fidelity: {n - len(mismatches)}/{n} exact; mismatches={len(mismatches)}")
    for m in mismatches[:8]:
        print("   MISMATCH", m)
    print(f"== truncated >8192: {truncs}")
    print(f"== dataset audit counters: {dict(ds.audit)}")
    print("\n== per-slice tokens (n / mean / max / sup-ratio):")
    for sl, s in sorted(stats.items()):
        print(f"  {sl:<10} n={s['n']:>4} mean={s['tokens']/s['n']:>7.0f} "
              f"max={s['max']:>6} sup={s['sup']/max(1,s['tokens']):.3f}")

    # 5. dialect eyeball samples
    print("\n== dialect samples:")
    shown = set()
    for rec in ds.list_data_dict:
        sl = rec.get("slice")
        if sl in shown:
            continue
        shown.add(sl)
        q = rec["conversations"][0]["value"].replace("\n", " ")[:130]
        a = rec["conversations"][1]["value"].replace("\n", " ")[:90]
        print(f"  [{sl}] Q: {q}\n            A: {a}")

    # 6. collator shape sanity
    batch = DataCollatorForSupervisedDataset(tok)([ds[0], ds[1]])
    print("\n== collator batch:", {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
                                   for k, v in batch.items() if v is not None})
    ok = not mismatches and truncs == 0
    print("\n==", "DRY-RUN OK" if ok else "DRY-RUN PROBLEMS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
