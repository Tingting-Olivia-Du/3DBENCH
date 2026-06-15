#!/usr/bin/env python3
"""Merge per-dimension 3DBENCH LoRA adapters into the base Qwen2.5-VL-3B model.

Produces standalone HuggingFace checkpoints that can be loaded directly by
VLM4VLA's build_vlm() without any PEFT dependency.

Usage:
    # Merge all available adapters
    python scripts/merge_lora_adapters.py

    # Merge specific dimensions
    python scripts/merge_lora_adapters.py --dims q1 q8 all

    # Dry run (check paths only)
    python scripts/merge_lora_adapters.py --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # 3DBENCH/
MODELS_ROOT  = PROJECT_ROOT.parent / "models"                  # models/
BASE_MODEL   = MODELS_ROOT / "Qwen2.5-VL-3B-Instruct"
OUTPUT_ROOT  = MODELS_ROOT / "merged"

# Map dimension -> downloaded adapter tag. Adapters live at
#   models/downloads_<dim>-<tag>/<dim>-<tag>/
ADAPTER_MAP = {
    "q1": "final",
    "q2": "final",
    "q3": "ckpt4k",
    "q4": "ckpt4k",
    "q5": "ckpt4k",
    "q6": "ckpt4k",
    "q7": "final",
    "q8": "final",
}


def find_adapter_path(dim: str) -> Path | None:
    """Return the adapter directory for a given dimension, or None."""
    tag = ADAPTER_MAP.get(dim)
    if tag is None:
        return None
    path = MODELS_ROOT / f"downloads_{dim}-{tag}" / f"{dim}-{tag}"
    if not (path / "adapter_config.json").exists():
        return None
    return path


def merge_one(dim: str, adapter_path: Path, output_dir: Path,
              base_model_path: Path, dtype: str = "bf16"):
    """Merge a single LoRA adapter into the base model and save."""
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"\n{'='*60}")
    print(f"Merging: {dim}")
    print(f"  Adapter : {adapter_path}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    if output_dir.exists():
        print(f"  [SKIP] Output already exists. Delete to re-merge.")
        return

    # 1. Load base model in FP32 for precise merge
    print("  Loading base model (float32)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 2. Load LoRA adapter
    print(f"  Loading LoRA adapter from {adapter_path.name}...")
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))

    # 3. Merge and unload
    print("  Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # 4. Convert to target dtype for saving
    if dtype == "bf16":
        print("  Converting to bfloat16 for storage...")
        merged_model = merged_model.to(torch.bfloat16)

    # 5. Save merged model
    print(f"  Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))

    # 6. Copy processor/tokenizer from base model
    print("  Copying processor/tokenizer files...")
    processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    # 7. Save merge metadata
    meta = {
        "base_model": str(base_model_path),
        "adapter_path": str(adapter_path),
        "dimension": dim,
        "checkpoint": adapter_path.name,
        "merge_dtype": dtype,
    }
    (output_dir / "merge_info.json").write_text(json.dumps(meta, indent=2))

    # Free memory
    del merged_model, peft_model, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  [DONE] {dim} merged successfully.")


def main():
    parser = argparse.ArgumentParser(description="Merge 3DBENCH LoRA adapters into base model")
    parser.add_argument("--dims", nargs="+", default=None,
                        help="Dimensions to merge (default: all available)")
    parser.add_argument("--base-model", type=str, default=None,
                        help=f"Base model path (default: {BASE_MODEL})")
    parser.add_argument("--output-root", type=str, default=None,
                        help=f"Output root dir (default: {OUTPUT_ROOT})")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                        help="Save dtype (default: bf16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check paths, don't merge")
    args = parser.parse_args()

    base_model = Path(args.base_model) if args.base_model else BASE_MODEL
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT

    # Validate base model
    if not base_model.exists():
        print(f"ERROR: Base model not found at {base_model}")
        sys.exit(1)

    # Determine which dims to merge
    dims = args.dims or list(ADAPTER_MAP.keys())

    # Check all adapter paths
    print("Checking adapter paths...")
    tasks = []
    for dim in dims:
        adapter_path = find_adapter_path(dim)
        output_dir = output_root / f"Qwen2.5-VL-3B-{dim}-Merged"
        if adapter_path is None:
            print(f"  [WARN] {dim}: no adapter found, skipping")
            continue
        exists = output_dir.exists()
        status = "EXISTS" if exists else "READY"
        print(f"  [{status}] {dim}: {adapter_path.name} -> {output_dir.name}")
        tasks.append((dim, adapter_path, output_dir))

    if args.dry_run:
        print(f"\nDry run complete. {len(tasks)} adapters ready to merge.")
        return

    if not tasks:
        print("No adapters to merge.")
        return

    print(f"\nWill merge {len(tasks)} adapters...")
    for dim, adapter_path, output_dir in tasks:
        merge_one(dim, adapter_path, output_dir, base_model, dtype=args.dtype)

    print(f"\n{'='*60}")
    print(f"All done! Merged models saved to: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
