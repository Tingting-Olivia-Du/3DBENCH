#!/usr/bin/env python3
"""Run zero-shot VLM evaluation on the spatial reasoning benchmark.

Loads the benchmark dataset from ``data/gt/manifest.json``, builds a
four-question spatial prompt for each sample, queries each model, and
saves raw responses under ``data/responses/<model_slug>/``.

Supported models
----------------
  qwen2.5-vl-3b     Qwen/Qwen2.5-VL-3B-Instruct
  qwen2.5-vl-7b     Qwen/Qwen2.5-VL-7B-Instruct
  qwen3-vl-2b       Qwen/Qwen3-VL-2B-Instruct
  qwen3-vl-4b       Qwen/Qwen3-VL-4B-Instruct
  qwen3-vl-8b       Qwen/Qwen3-VL-8B-Instruct
  qwen3-vl-30b-a3b  Qwen/Qwen3-VL-30B-A3B-Instruct
  paligemma-1       google/paligemma-3b-mix-448
  paligemma-2       google/paligemma2-3b-mix-448
  kosmos-2          microsoft/kosmos-2-patch14-224
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
    p.add_argument("--models",   nargs="+",
                   default=["qwen2.5-vl-3b", "qwen2.5-vl-7b",
                            "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b", "qwen3-vl-30b-a3b",
                            "paligemma-1", "paligemma-2", "kosmos-2",
                            "internvl2-8b", "random"],
                   help="Models to evaluate (default: all three)")
    p.add_argument("--device",   default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--resume",   action="store_true",
                   help="Skip samples that already have a response file")
    # LoRA-augmented Qwen2.5-VL-3B: pass --lora_dir to load a PEFT adapter
    # produced by scripts/06_finetune_qwen.py and use the per-dim stripped
    # prompt for that adapter's dimension. Use slug 'qwen2.5-vl-3b-mv-lora'.
    p.add_argument("--lora_dir", default=None,
                   help="Path to a LoRA adapter directory (e.g. models/qwen2.5-vl-3b-mv-lora/q3/final). "
                        "When set, the model slug 'qwen2.5-vl-3b-mv-lora' loads it.")
    p.add_argument("--lora_dim", default=None,
                   choices=[None, "q1", "q2", "q3", "q4", "q5", "q6"],
                   help="Which dimension this LoRA targets. If omitted, read from "
                        "<lora_dir>/adapter_meta.json or <lora_dir>/../adapter_meta.json.")
    p.add_argument("--prompt_yaml", default=None,
                   help="Override the prompt YAML used by base PromptBuilder. "
                        "Useful for evaluating on CALVIN with prompts/spatial_qa_calvin.yaml.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Abstract VLM interface
# ---------------------------------------------------------------------------

class VLMBase(ABC):
    """Common interface for all VLMs and the random baseline.

    Dual-view contract: callers pass an `image_paths` dict like
        {"agent": "/abs/path/sample_xxx_agent.png",
         "wrist": "/abs/path/sample_xxx_wrist.png"}
    Subclasses decide how to consume it: native multi-image models stream both
    images into the model; single-image models concatenate them side-by-side.
    """

    slug: str  # used as directory name under out_dir
    # Set to True by subclasses that natively accept multiple images.
    # Single-image models (PaliGemma, KosMos2) leave this False and the base
    # helper concatenates the two views horizontally before inference.
    supports_multi_image: bool = False

    @abstractmethod
    def generate(
        self,
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Return the model's text response given dual-view images and prompts.

        `image_paths` is a dict with at minimum a "agent" key; "wrist" is
        optional but expected for the spatial QA benchmark.
        """


# ---------------------------------------------------------------------------
# Helpers shared across single-image models
# ---------------------------------------------------------------------------

def _hconcat_views(image_paths: dict, target_size: int = 448):
    """Combine agent + wrist views into a single PIL.Image, side by side.

    Used by models that don't natively support multi-image input. Each view is
    resized to target_size×target_size and placed left (agent) and right
    (wrist). Returns a target_size × 2*target_size RGB image.
    If only one view is present the function returns that view resized to a
    square target_size×target_size.
    """
    from PIL import Image
    if "agent" in image_paths and "wrist" in image_paths:
        agent = Image.open(image_paths["agent"]).convert("RGB").resize((target_size, target_size))
        wrist = Image.open(image_paths["wrist"]).convert("RGB").resize((target_size, target_size))
        out = Image.new("RGB", (target_size * 2, target_size))
        out.paste(agent, (0, 0))
        out.paste(wrist, (target_size, 0))
        return out
    # Fallback: single view
    only = next(iter(image_paths.values()))
    return Image.open(only).convert("RGB").resize((target_size, target_size))


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

class Qwen25VL(VLMBase):
    slug = "qwen2.5-vl-7b"
    supports_multi_image = True
    _DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: str = "cuda"):
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
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        # Stream both views, with explicit text labels so the model knows which is which.
        user_content: list = []
        if "agent" in image_paths:
            user_content.append({"type": "text", "text": "[AGENT VIEW] (third-person, full workspace)"})
            user_content.append({"type": "image", "image": f"file://{Path(image_paths['agent']).resolve()}"})
        if "wrist" in image_paths:
            user_content.append({"type": "text", "text": "[WRIST VIEW] (first-person, mounted on the gripper)"})
            user_content.append({"type": "image", "image": f"file://{Path(image_paths['wrist']).resolve()}"})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
# Qwen2.5-VL variants (3B)
# ---------------------------------------------------------------------------

class Qwen25VL3B(Qwen25VL):
    slug = "qwen2.5-vl-3b"
    _DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        super().__init__(model_id=model_id, device=device)


class Qwen25VL3B_LoRA(Qwen25VL3B):
    """Qwen2.5-VL-3B with a per-dimension LoRA adapter merged at load time.

    Constructed via build_model() with `lora_dir` set. The dimension can be
    inferred from <lora_dir>/adapter_meta.json (or its parent), in which case
    the eval driver swaps the default `PromptBuilder` for `PerDimPromptBuilder`.

    The merged-and-unloaded weights are equivalent to the base model with LoRA
    applied; subsequent inference uses Qwen25VL.generate() unchanged.

    The instance `slug` is rewritten at construction time to include the
    target dim (e.g. "qwen2.5-vl-3b-mv-lora-q3"), so each adapter writes its
    responses to an independent output directory. This is critical: without it,
    multiple --lora_dir invocations in the same --out_dir clobber each other,
    or worse, --resume silently skips them all because sample_<id>.json from a
    previous dim already exists.
    """
    slug = "qwen2.5-vl-3b-mv-lora"  # class default; instance overrides below

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        lora_dir: str | None = None,
    ):
        if lora_dir is None:
            raise ValueError("Qwen25VL3B_LoRA requires lora_dir")
        super().__init__(model_id=model_id, device=device)
        from peft import PeftModel
        print(f"  [LoRA] loading adapter from {lora_dir}")
        peft_model = PeftModel.from_pretrained(self._model, lora_dir)
        self._model = peft_model.merge_and_unload()
        self._model.eval()
        # Stash the dim if the adapter has metadata, so the eval loop can pick it up.
        meta_path = Path(lora_dir) / "adapter_meta.json"
        if not meta_path.exists():
            meta_path = Path(lora_dir).parent / "adapter_meta.json"
        if meta_path.exists():
            self.lora_meta = json.loads(meta_path.read_text())
            print(f"  [LoRA] dim={self.lora_meta.get('dim')}")
        else:
            self.lora_meta = {}
        # Rewrite the instance slug to include the dim → unique output dir.
        dim = self.lora_meta.get("dim")
        if dim:
            self.slug = f"qwen2.5-vl-3b-mv-lora-{dim}"
            print(f"  [LoRA] output slug = {self.slug}")


# ---------------------------------------------------------------------------
# Qwen3-VL  (shares the same transformers class as Qwen2.5-VL)
# ---------------------------------------------------------------------------

class Qwen3VL(VLMBase):
    """Qwen3-VL family — uses Qwen3VLForConditionalGeneration (distinct from Qwen2.5-VL)."""

    slug = "qwen3-vl-2b"
    supports_multi_image = True

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        print(f"Loading {model_id} ...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        user_content: list = []
        if "agent" in image_paths:
            user_content.append({"type": "text", "text": "[AGENT VIEW] (third-person, full workspace)"})
            user_content.append({"type": "image", "image": f"file://{Path(image_paths['agent']).resolve()}"})
        if "wrist" in image_paths:
            user_content.append({"type": "text", "text": "[WRIST VIEW] (first-person, mounted on the gripper)"})
            user_content.append({"type": "image", "image": f"file://{Path(image_paths['wrist']).resolve()}"})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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

        generated_ids = [
            out[len(inp):]
            for out, inp in zip(out_ids, inputs.input_ids)
        ]
        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class Qwen3VL4B(Qwen3VL):
    slug = "qwen3-vl-4b"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda"):
        super().__init__(model_id=model_id, device=device)


class Qwen3VL8B(Qwen3VL):
    slug = "qwen3-vl-8b"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        super().__init__(model_id=model_id, device=device)


class Qwen3VL30BA3B(Qwen3VL):
    slug = "qwen3-vl-30b-a3b"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", device: str = "cuda"):
        super().__init__(model_id=model_id, device=device)


# ---------------------------------------------------------------------------
# PaliGemma
# ---------------------------------------------------------------------------

class PaliGemma(VLMBase):
    slug = "paligemma-1"

    def __init__(self, model_id: str = "google/paligemma-3b-mix-448", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        print(f"Loading {model_id} ...")
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
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
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        # PaliGemma is single-image — concatenate agent + wrist horizontally.
        image = _hconcat_views(image_paths, target_size=448)
        layout_note = (
            "The image you see is two views concatenated side by side: "
            "LEFT half = AGENT VIEW (third-person workspace), "
            "RIGHT half = WRIST VIEW (first-person from the gripper).\n\n"
        )
        prompt = f"{system_prompt}\n\n{layout_note}{user_prompt}"
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # PaliGemma output includes the input tokens; strip them
        generated = out_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated, skip_special_tokens=True)


class PaliGemma2(PaliGemma):
    slug = "paligemma-2"

    def __init__(self, model_id: str = "google/paligemma2-3b-mix-448", device: str = "cuda"):
        super().__init__(model_id=model_id, device=device)


# ---------------------------------------------------------------------------
# KosMos-2
# ---------------------------------------------------------------------------

class KosMos2(VLMBase):
    slug = "kosmos-2"

    def __init__(self, model_id: str = "microsoft/kosmos-2-patch14-224", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
        print(f"Loading {model_id} ...")
        self._model = Kosmos2ForConditionalGeneration.from_pretrained(
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
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        # Kosmos-2 is single-image — concatenate agent + wrist horizontally.
        image = _hconcat_views(image_paths, target_size=224)
        layout_note = (
            "The image is two views concatenated: LEFT = AGENT VIEW (workspace), "
            "RIGHT = WRIST VIEW (gripper-mounted).\n\n"
        )
        prompt = f"<grounding>{system_prompt}\n\n{layout_note}{user_prompt}"
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = out_ids[0][inputs["input_ids"].shape[-1]:]
        text, _ = self._processor.post_process_generation(
            self._processor.decode(generated, skip_special_tokens=True)
        )
        return text


# ---------------------------------------------------------------------------
# InternVL2
# ---------------------------------------------------------------------------

class InternVL2(VLMBase):
    slug = "internvl2-8b"
    supports_multi_image = True

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
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        # InternVL2 native multi-image: stack pixel_values along dim 0 and pass
        # one <image> token per view + a num_patches_list of [1, 1, ...].
        view_order = [v for v in ("agent", "wrist") if v in image_paths]
        per_view = [self._load_pixel_values(image_paths[v]) for v in view_order]
        pixel_values = self._torch.cat(per_view, dim=0)
        num_patches_list = [1] * len(view_order)

        # Two <image> tokens, one per view, with explicit labels.
        view_labels = {
            "agent": "Agent view (third-person workspace):",
            "wrist": "Wrist view (first-person from gripper):",
        }
        image_block = "\n".join(
            f"{view_labels[v]} <image>" for v in view_order
        )
        question = f"{system_prompt}\n\n{image_block}\n\n{user_prompt}"

        gen_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            gen_config,
            num_patches_list=num_patches_list,
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
    # Realistic delta range: target is typically 0.05–0.50 m away from gripper
    _DELTA_BOUNDS = {
        "dx": (-0.50, 0.50),
        "dy": (-0.40, 0.40),
        "dz": (-0.40, 0.40),
    }
    _X_LABELS = ["in_front", "behind", "aligned_x"]
    _Y_LABELS = ["left", "right", "aligned_y"]
    _Z_LABELS = ["above", "below", "aligned_z"]

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

    def _rand_delta(self) -> dict:
        return {
            ax: float(self._rng.uniform(*bounds))
            for ax, bounds in self._DELTA_BOUNDS.items()
        }

    def _rand_relation(self) -> dict:
        return {
            "x": self._rng.choice(self._X_LABELS),
            "y": self._rng.choice(self._Y_LABELS),
            "z": self._rng.choice(self._Z_LABELS),
        }

    def generate(
        self,
        image_paths: dict,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        response = {
            "task_type": self._rng.choice(["pick_and_place", "articulation"]),
            "q1": self._rand_pos(),
            "q1_dest": self._rand_pos(),
            "q2": self._rand_pos(),
            "q3": {"can_close": "yes" if self._rng.random() > 0.5 else "no"},
            "q4": self._rand_unit_vec(),
            "q5": self._rand_delta(),
            "q6": self._rand_relation(),
        }
        return json.dumps(response)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[VLMBase]] = {
    # Qwen2.5-VL
    "qwen2.5-vl-3b":         Qwen25VL3B,
    "qwen2.5-vl-3b-mv-lora": Qwen25VL3B_LoRA,
    "qwen2.5-vl-7b":         Qwen25VL,
    # Qwen3-VL
    "qwen3-vl-2b":        Qwen3VL,
    "qwen3-vl-4b":        Qwen3VL4B,
    "qwen3-vl-8b":        Qwen3VL8B,
    "qwen3-vl-30b-a3b":   Qwen3VL30BA3B,
    # PaliGemma
    "paligemma-1":        PaliGemma,
    "paligemma-2":        PaliGemma2,
    # KosMos
    "kosmos-2":           KosMos2,
    # InternVL2
    "internvl2-8b":       InternVL2,
    # Baseline
    "random":             RandomBaseline,
}

# HuggingFace repo id for each slug — used to find the local cache path.
_MODEL_HF_ID: dict[str, str] = {
    "qwen2.5-vl-3b":    "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-2b":      "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b":      "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b":      "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "paligemma-1":      "google/paligemma-3b-mix-448",
    "paligemma-2":      "google/paligemma2-3b-mix-448",
    "kosmos-2":         "microsoft/kosmos-2-patch14-224",
    "internvl2-8b":     "OpenGVLab/InternVL2-8B",
}

_LOCAL_MODEL_ROOT = Path("/workspace/tingting/models")


def _is_local_complete(local: Path) -> bool:
    """Return True only if the local model directory has all weight shards.

    A sharded model has model.safetensors.index.json listing every shard file.
    A single-file model has model.safetensors (or pytorch_model.bin).
    We consider the directory incomplete if any listed shard is missing.
    """
    if not local.exists():
        return False
    index = local / "model.safetensors.index.json"
    if index.exists():
        import json as _json
        idx = _json.loads(index.read_text())
        shards = set(idx.get("weight_map", {}).values())
        return all((local / s).exists() for s in shards)
    # Non-sharded: just needs model.safetensors or pytorch_model.bin
    return (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists()


def _resolve_model_id(slug: str) -> str:
    """Return local path if complete, otherwise download from HF to _LOCAL_MODEL_ROOT."""
    hf_id = _MODEL_HF_ID.get(slug)
    if hf_id is None:
        return slug
    local = _LOCAL_MODEL_ROOT / hf_id.split("/")[-1]
    if _is_local_complete(local):
        print(f"  [local] {slug} → {local}")
        return str(local)
    # Download (or resume partial download) to the local model root
    print(f"  [download] {hf_id} → {local}  (this may take a while)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(local),
            resume_download=True,
        )
        print(f"  [download] done → {local}")
        return str(local)
    except Exception as exc:
        print(f"  [download] failed ({exc}), falling back to HF repo id")
        return hf_id


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


def build_model(slug: str, device: str, lora_dir: str | None = None) -> VLMBase:
    if slug not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{slug}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[slug]
    if slug == "random":
        return cls()
    if slug == "qwen2.5-vl-3b-mv-lora":
        if lora_dir is None:
            raise ValueError("--lora_dir is required when using slug 'qwen2.5-vl-3b-mv-lora'")
        # Always resolve the base model id from the qwen2.5-vl-3b slug.
        model_id = _resolve_model_id("qwen2.5-vl-3b")
        return cls(model_id=model_id, device=device, lora_dir=lora_dir)
    model_id = _resolve_model_id(slug)
    return cls(model_id=model_id, device=device)


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
    # Base directory: responses/<model>/<YYYYMMDD_HHMMSS>/
    # Samples go into a suite subdirectory: .../<suite>/sample_*.json
    latest_link = out_dir / model.slug / "latest"

    # When resuming, reuse the existing run directory (via the latest symlink)
    # so that already-written sample files are found by resp_path.exists().
    if resume and latest_link.exists():
        effective_ts = latest_link.resolve().name
        print(f"  [resume] reusing run {effective_ts}")
    else:
        effective_ts = run_ts

    model_run_dir = out_dir / model.slug / effective_ts
    model_run_dir.mkdir(parents=True, exist_ok=True)

    # Update the latest symlink to point to this run
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(effective_ts)

    print(f"\n{'='*60}")
    print(f"Model: {model.slug}  |  Run: {effective_ts}  |  Output: {model_run_dir}")
    print(f"{'='*60}")

    n_ok = n_fail = n_skip = 0

    # Cache suite→dir so we only mkdir once per suite
    _suite_dirs: dict[str, Path] = {}

    for record in tqdm(manifest, desc=model.slug):
        sample_id = record["sample_id"]
        suite = record.get("suite", "unknown")

        if suite not in _suite_dirs:
            suite_dir = model_run_dir / suite
            suite_dir.mkdir(parents=True, exist_ok=True)
            _suite_dirs[suite] = suite_dir

        resp_path = _suite_dirs[suite] / f"sample_{sample_id:04d}.json"

        if resume and resp_path.exists():
            n_skip += 1
            continue

        # Resolve dual-view image paths (new schema). Fall back to single-image
        # legacy manifests by treating the lone image_path as the agent view.
        rel_paths = record.get("image_paths")
        if rel_paths is None:
            rel_paths = {"agent": record["image_path"]}
        image_paths = {k: str(data_root / v) for k, v in rel_paths.items()}

        task_desc = record["task_description"]

        system_prompt, user_prompt = builder.build(task_desc)

        t0 = time.perf_counter()
        sample_ts = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            raw_response = model.generate(
                image_paths=image_paths,
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
            "suite": suite,
            "model": model.slug,
            "run_timestamp": effective_ts,
            "sample_timestamp": sample_ts,
            "task_description": task_desc,
            "image_path": record["image_path"],          # legacy / agent view
            "image_paths": rel_paths,                    # full dual-view dict
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
    # image_path in manifest is always relative to the repo's data/ directory
    data_root = Path(__file__).resolve().parent.parent / "data"
    print(f"Loaded {len(manifest)} samples from {manifest_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single timestamp shared across all models in this run
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {run_ts}")

    # Default builder: full 6-question prompt (optionally overridden via --prompt_yaml,
    # e.g. for CALVIN). Per-LoRA evaluation uses PerDimPromptBuilder, picked below.
    builder = PromptBuilder(template_path=args.prompt_yaml) if args.prompt_yaml else PromptBuilder()

    for slug in args.models:
        try:
            model = build_model(slug, args.device, lora_dir=args.lora_dir)
        except ValueError as exc:
            print(f"Skipping: {exc}")
            continue
        except Exception as exc:
            print(f"ERROR loading model '{slug}': {exc}")
            continue

        # If this is a LoRA-augmented model, swap the prompt builder for the
        # per-dim one matching this adapter's dimension. Dim is taken from CLI
        # --lora_dim or the adapter's metadata file (set by 06_finetune_qwen.py).
        # Critically, also make `model.slug` carry the dim so that 6 different
        # adapters write to 6 distinct output dirs (otherwise --resume silently
        # skips runs because sample_NNNN.json from a previous adapter exists).
        active_builder = builder
        if slug == "qwen2.5-vl-3b-mv-lora":
            from bench.per_dim_prompt_builder import PerDimPromptBuilder, VALID_DIMS
            dim = args.lora_dim
            if dim is None:
                dim = getattr(model, "lora_meta", {}).get("dim")
            if dim not in VALID_DIMS:
                print(f"ERROR: could not determine LoRA dim (got {dim!r}); pass --lora_dim.")
                _free_model(model)
                continue
            # Override the instance slug so each dim writes to its own output dir.
            # (Qwen25VL3B_LoRA also sets this from adapter_meta in __init__; we
            # repeat here so a CLI --lora_dim wins over a stale meta file.)
            model.slug = f"qwen2.5-vl-3b-mv-lora-{dim}"
            # If --prompt_yaml is supplied (e.g. CALVIN-aware per-dim prompts),
            # honour it; otherwise PerDimPromptBuilder loads the default LIBERO
            # template (prompts/spatial_qa_per_dim.yaml).
            active_builder = PerDimPromptBuilder(dim, template_path=args.prompt_yaml)
            tag = args.prompt_yaml if args.prompt_yaml else "default (LIBERO)"
            print(f"  [LoRA] PerDimPromptBuilder({dim})  template={tag}  slug={model.slug}")

        try:
            run_model(
                model=model,
                manifest=manifest,
                out_dir=out_dir,
                builder=active_builder,
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
