#!/usr/bin/env python3
"""Generate VLA ablation experiment configs for per-dimension spatial LoRA study.

Creates 22 JSON configs (11 backbones x 2 training strategies) under
3DBENCH/configs/vla_ablation/, based on VLM4VLA's LIBERO template.

Usage:
    python scripts/generate_ablation_configs.py
    python scripts/generate_ablation_configs.py --data-root /path/to/modified_libero_rlds
"""
import argparse
import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # 3DBENCH/
VLM4VLA_ROOT = PROJECT_ROOT.parent / "VLM4VLA"
OUTPUT_DIR   = PROJECT_ROOT / "configs" / "vla_ablation"

# ── Default paths (update these for your machine) ─────────────────────────
BASE_MODEL_PATH = str(PROJECT_ROOT.parent / "models" / "Qwen2.5-VL-3B-Instruct")
MERGED_ROOT     = str(PROJECT_ROOT.parent / "models" / "merged")
DATA_ROOT       = str(PROJECT_ROOT.parent / "modified_libero_rlds")  # RLDS data

# ── Experiment definitions ────────────────────────────────────────────────
DIMS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "all"]

EXPERIMENTS = []
# E0: vanilla baseline
EXPERIMENTS.append({
    "id": "e0",
    "dim": None,
    "label": "vanilla (no LoRA)",
    "model_path": BASE_MODEL_PATH,
})
# E1-E8: per-dimension LoRA
for i, dim in enumerate(DIMS[:8], start=1):
    EXPERIMENTS.append({
        "id": f"e{i}",
        "dim": dim,
        "label": f"{dim} LoRA merged",
        "model_path": f"{MERGED_ROOT}/Qwen2.5-VL-3B-{dim}-Merged",
    })
# E9: all-Q combined
EXPERIMENTS.append({
    "id": "e9",
    "dim": "all",
    "label": "all-Q LoRA merged",
    "model_path": f"{MERGED_ROOT}/Qwen2.5-VL-3B-all-Merged",
})


def load_template() -> dict:
    """Load the VLM4VLA LIBERO config template."""
    template_path = VLM4VLA_ROOT / "configs" / "oxe_training" / "libero10" / "finetune_qwen25vl-3b_libero10.json"
    with open(template_path) as f:
        return json.load(f)


def make_config(template: dict, exp: dict, strategy: str, data_root: str) -> dict:
    """Create a single experiment config from template."""
    cfg = copy.deepcopy(template)

    model_path = exp["model_path"]
    dim_tag = exp["dim"] or "vanilla"
    exp_id = exp["id"]

    # ── Task name for W&B ──
    cfg["task_name"] = f"vla_ablation_{exp_id}_{strategy}"

    # ── Model paths ──
    cfg["model_path"] = model_path
    cfg["model_config"] = f"{model_path}/config.json"
    cfg["vlm"]["pretrained_model_name_or_path"] = model_path
    cfg["tokenizer"]["pretrained_model_name_or_path"] = model_path

    # ── Data paths ──
    cfg["train_dataset"]["data_root_dir"] = data_root
    cfg["val_dataset"]["data_root_dir"] = data_root

    # ── Output paths (separate per experiment) ──
    cfg["output_root"] = f"runs/vla_ablation/{exp_id}_{strategy}/checkpoints"
    cfg["log_root"]    = f"runs/vla_ablation/{exp_id}_{strategy}/logs"
    cfg["cache_root"]  = f"runs/vla_ablation/{exp_id}_{strategy}/cache"

    # ── Training strategy ──
    if strategy == "head":
        # Freeze backbone, only train action token + FCDecoder
        cfg["train_setup"]["freeze_backbone"] = True
        cfg["train_setup"]["train_vision"] = False
        cfg["train_setup"]["train_text_embedding"] = False
    else:
        # Full finetune (default)
        cfg["train_setup"]["freeze_backbone"] = False
        cfg["train_setup"]["train_vision"] = True
        cfg["train_setup"]["train_text_embedding"] = True

    # ── Metadata (not used by VLM4VLA but useful for tracking) ──
    cfg["_ablation_meta"] = {
        "experiment_id": exp_id,
        "dimension": dim_tag,
        "strategy": strategy,
        "label": exp["label"],
    }

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate VLA ablation configs")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help=f"RLDS data root (default: {DATA_ROOT})")
    parser.add_argument("--strategies", nargs="+", default=["full", "head"],
                        choices=["full", "head"],
                        help="Training strategies to generate (default: both)")
    args = parser.parse_args()

    template = load_template()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for exp in EXPERIMENTS:
        for strategy in args.strategies:
            cfg = make_config(template, exp, strategy, args.data_root)
            filename = f"{exp['id']}_{strategy}.json"
            filepath = OUTPUT_DIR / filename
            with open(filepath, "w") as f:
                json.dump(cfg, f, indent=4)
            generated.append(filename)
            print(f"  Generated: {filename}  ({exp['label']}, {strategy})")

    print(f"\n{len(generated)} configs written to {OUTPUT_DIR}/")

    # Generate a summary table
    summary_path = OUTPUT_DIR / "README.md"
    lines = [
        "# VLA Ablation Experiment Configs\n",
        "| Config | Experiment | Dimension | Strategy | Model Path |",
        "|--------|-----------|-----------|----------|------------|",
    ]
    for exp in EXPERIMENTS:
        for strategy in args.strategies:
            filename = f"{exp['id']}_{strategy}.json"
            dim_tag = exp["dim"] or "vanilla"
            model_short = Path(exp["model_path"]).name
            lines.append(f"| `{filename}` | {exp['id']} | {dim_tag} | {strategy} | `{model_short}` |")
    lines.append(f"\nGenerated by `scripts/generate_ablation_configs.py`")
    summary_path.write_text("\n".join(lines))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
