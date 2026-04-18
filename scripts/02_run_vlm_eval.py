#!/usr/bin/env python3
"""Run zero-shot VLM evaluation on the spatial reasoning benchmark.

Loads the benchmark dataset from ``data/gt/manifest.json``, builds a
four-question spatial prompt for each sample, queries each model, and
saves raw responses under ``data/responses/<model_slug>/``.

Supported models
----------------
  qwen2.5-vl-7b     Qwen/Qwen2.5-VL-7B-Instruct  (default open model)
  internvl2-8b      OpenGVLab/InternVL2-8B
  random            Random-number baseline (no GPU needed)

Usage
-----
  # Run all models
  python scripts/02_run_vlm_eval.py

  # Run a single model
  python scripts/02_run_vlm_eval.py --models qwen2.5-vl-7b

  # Override paths
  python scripts/02_run_vlm_eval.py \
      --manifest data/gt/manifest.json \
      --out_dir  data/responses-0415 \
      --device   cuda:0
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
from tqdm import tqdm

from bench.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/gt/manifest.json",
                   help="Path to GT manifest from 01_extract_gt.py (default: data/gt/manifest.json)")
    p.add_argument("--out_dir",  default="data/responses",
                   help="Output directory for model responses (default: data/responses)")
    p.add_argument("--models",   nargs="+", default=["qwen2.5-vl-7b", "internvl2-8b", "random"],
                   help="Models to evaluate (default: all three)")
    p.add_argument("--device",   default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--resume",   action="store_true",
                   help="Skip samples that already have a response file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Abstract VLM interface
# ---------------------------------------------------------------------------

class VLMBase(ABC):
    """Common interface for all VLMs and the random baseline."""

    slug: str  # used as directory name under out_dir

    @abstractmethod
    def generate(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Return the model's text response given one image and two prompt strings."""


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

class Qwen25VL(VLMBase):
    slug = "qwen2.5-vl-7b"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"Loading {model_id} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model.eval()
        self._device = next(self._model.parameters()).device
        import torch as _torch
        self._torch = _torch

    def generate(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Strip the prompt tokens from output
        generated_ids = [
            out[len(inp):]
            for out, inp in zip(out_ids, inputs.input_ids)
        ]
        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ---------------------------------------------------------------------------
# InternVL2
# ---------------------------------------------------------------------------

class InternVL2(VLMBase):
    slug = "internvl2-8b"

    # Preprocessing constants used by InternVL2
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)
    _INPUT_SIZE = 448

    def __init__(self, model_id: str = "OpenGVLab/InternVL2-8B", device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading {model_id} ...")
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self._model.eval()
        import torch as _torch
        import torchvision.transforms as T
        self._torch = _torch
        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    def _load_pixel_values(self, image_path: str):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0)
        device = next(self._model.parameters()).device
        return tensor.to(self._torch.bfloat16).to(device)

    def generate(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        pixel_values = self._load_pixel_values(image_path)
        # InternVL2 uses <image> token in the question
        question = f"<image>\n{system_prompt}\n\n{user_prompt}"
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            gen_config,
        )
        return response


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomBaseline(VLMBase):
    """Generates valid-format JSON with random spatial values.

    Serves as a lower-bound reference: any model scoring below this is
    pathologically wrong (e.g. systematic bias or refusal to answer).
    """
    slug = "random"

    # Approximate workspace bounds [min, max] in robot-base frame (meters).
    # These match the WORKSPACE CONTEXT in spatial_qa.yaml so the random
    # baseline samples from the same distribution the VLMs are told about.
    _BOUNDS = {
        "x": (0.10, 0.80),
        "y": (-0.35, 0.35),
        "z": (-0.02, 0.60),  # table surface ≈ -0.02; gripper raised ≈ 0.40-0.60
    }

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def _rand_pos(self) -> dict:
        return {
            ax: float(self._rng.uniform(*bounds))
            for ax, bounds in self._BOUNDS.items()
        }

    def _rand_unit_vec(self) -> dict:
        v = self._rng.standard_normal(3)
        v = v / (np.linalg.norm(v) + 1e-8)
        return {"dx": float(v[0]), "dy": float(v[1]), "dz": float(v[2])}

    def generate(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        response = {
            "q1": self._rand_pos(),
            "q2": self._rand_pos(),
            "q3": {"can_close": "yes" if self._rng.random() > 0.5 else "no"},
            "q4": self._rand_unit_vec(),
        }
        return json.dumps(response)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[VLMBase]] = {
    "qwen2.5-vl-7b": Qwen25VL,
    "internvl2-8b": InternVL2,
    "random": RandomBaseline,
}


def _free_model(model: VLMBase) -> None:
    """Delete a model and release its GPU memory."""
    import gc
    if hasattr(model, "_model"):
        del model._model
    if hasattr(model, "_processor"):
        del model._processor
    if hasattr(model, "_tokenizer"):
        del model._tokenizer
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        print("GPU cache cleared.")
    except Exception:
        pass


def build_model(slug: str, device: str) -> VLMBase:
    if slug not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{slug}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[slug]
    if slug == "random":
        return cls()
    return cls(device=device)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_model(
    model: VLMBase,
    manifest: list[dict],
    out_dir: Path,
    builder: PromptBuilder,
    max_new_tokens: int,
    resume: bool,
    data_root: Path,
    run_ts: str,
) -> None:
    # Directory: responses/<model>/<YYYYMMDD_HHMMSS>/
    model_out = out_dir / model.slug / run_ts
    model_out.mkdir(parents=True, exist_ok=True)

    # Also keep a symlink "latest" pointing to the newest run
    latest_link = out_dir / model.slug / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_ts)

    print(f"\n{'='*60}")
    print(f"Model: {model.slug}  |  Run: {run_ts}  |  Output: {model_out}")
    print(f"{'='*60}")

    n_ok = n_fail = n_skip = 0

    for record in tqdm(manifest, desc=model.slug):
        sample_id = record["sample_id"]
        resp_path = model_out / f"sample_{sample_id:04d}.json"

        if resume and resp_path.exists():
            n_skip += 1
            continue

        image_path = str(data_root / record["image_path"])
        task_desc = record["task_description"]

        system_prompt, user_prompt = builder.build(task_desc)

        t0 = time.perf_counter()
        sample_ts = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            raw_response = model.generate(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.perf_counter() - t0
            n_ok += 1
        except Exception as exc:
            raw_response = f"ERROR: {exc}"
            elapsed = time.perf_counter() - t0
            n_fail += 1
            print(f"  sample {sample_id:04d} FAILED: {exc}")

        output = {
            "sample_id": sample_id,
            "model": model.slug,
            "run_timestamp": run_ts,
            "sample_timestamp": sample_ts,
            "task_description": task_desc,
            "image_path": record["image_path"],
            "raw_response": raw_response,
            "elapsed_s": round(elapsed, 3),
        }
        resp_path.write_text(json.dumps(output, indent=2))

    print(
        f"Done: {n_ok} ok  |  {n_fail} failed  |  {n_skip} skipped"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"ERROR: Manifest not found at '{manifest_path}'.\n"
            "Run 01_extract_gt.py first to generate the benchmark dataset."
        )
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    data_root = manifest_path.resolve().parent.parent  # data/gt/.. = data/
    print(f"Loaded {len(manifest)} samples from {manifest_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single timestamp shared across all models in this run
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {run_ts}")

    builder = PromptBuilder()

    for slug in args.models:
        try:
            model = build_model(slug, args.device)
        except ValueError as exc:
            print(f"Skipping: {exc}")
            continue
        except Exception as exc:
            print(f"ERROR loading model '{slug}': {exc}")
            continue

        try:
            run_model(
                model=model,
                manifest=manifest,
                out_dir=out_dir,
                builder=builder,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
                data_root=data_root,
                run_ts=run_ts,
            )
        finally:
            # Explicitly free GPU memory before loading the next model.
            # Without this, multiple 8B models will OOM on a 44GB card.
            _free_model(model)

    print(f"\nAll done. Responses saved under {out_dir}/")
    print("Next: python scripts/03_compute_metrics.py")


if __name__ == "__main__":
    main()
