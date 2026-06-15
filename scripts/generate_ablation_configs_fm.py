#!/usr/bin/env python3
"""Generate FMDecoder (flow-matching) variants of the VLA ablation configs.

Derives one FMDecoder config per existing `configs/vla_ablation/*.json` file so
that every ablation checkpoint (e0-e9, full + head) can be trained with the
flow-matching action head instead of FCDecoder. Only the action-head related
fields are changed; model paths, data paths, norm settings, training strategy
and ablation metadata are preserved from the source configs.

Output goes to a NEW folder: 3DBENCH/configs/vla_ablation_fm/

Usage:
    python scripts/generate_ablation_configs_fm.py
"""
import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # 3DBENCH/
SRC_DIR      = PROJECT_ROOT / "configs" / "vla_ablation"
OUT_DIR      = PROJECT_ROOT / "configs" / "vla_ablation_fm"

# Action chunk length predicted by the flow-matching head. Kept equal to the
# top-level fwd_pred_next_n the dataset already produces (4) to avoid changing
# the data pipeline. Both the top-level field and the act_head field must match.
FWD_PRED_NEXT_N = 4

# Qwen2.5-VL-3B language hidden size -> condition dim for layer-wise cross-attn.
QWEN25VL_3B_HIDDEN = 2048

# DiT config mirrors VLM4VLA's flowmatching template, with cross_attention_dim
# and num_layers adapted to the Qwen2.5-VL-3B backbone (36 layers, hidden 2048).
DIT_CONFIG = {
    "input_embedding_dim": 1536,
    "num_layers": 12,
    "num_attention_heads": 16,
    "attention_head_dim": 96,
    "dropout": 0.0,
    "attention_bias": True,
    "activation_fn": "gelu-approximate",
    "norm_type": "ada_norm",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-05,
    "max_num_positional_embeddings": 512,
    "final_dropout": True,
    "positional_embeddings": "sinusoidal",
    "interleave_self_attention": False,
    "cross_attention_dim": QWEN25VL_3B_HIDDEN,   # = backbone hidden_size (2048)
    "output_dim": 1024,
}


def make_fm_act_head(src_act_head: dict) -> dict:
    """Build an FMDecoder act_head, carrying over compatible fields."""
    return {
        "type": "FMDecoder",
        "hidden_size": src_act_head.get("hidden_size", 1024),
        "action_dim": src_act_head.get("action_dim", 7),
        "down_sample": "none",
        "latent": src_act_head.get("latent", 1),
        "fwd_pred_next_n": FWD_PRED_NEXT_N,
        "window_size": src_act_head.get("window_size", 1),
        "action_space": "continuous",
        "with_history": src_act_head.get("with_history", True),
        "history_type": src_act_head.get("history_type", "post"),
        "DiTConfig": copy.deepcopy(DIT_CONFIG),
        "add_pos_embed": True,
        "num_inference_timesteps": 20,
        "num_timestep_buckets": 1000,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "state_dim": 0,
        "num_target_vision_tokens": 32,
        "max_seq_len": 1024,
    }


def convert(src_path: Path) -> dict:
    with open(src_path) as f:
        cfg = json.load(f)

    cfg = copy.deepcopy(cfg)

    # Swap action head to FMDecoder.
    cfg["act_head"] = make_fm_act_head(cfg.get("act_head", {}))

    # Flow matching predicts an action chunk; keep top-level chunk length in sync.
    cfg["fwd_pred_next_n"] = FWD_PRED_NEXT_N

    # Distinguish task name and run outputs from the FCDecoder runs.
    base_task = cfg.get("task_name", src_path.stem)
    cfg["task_name"] = f"{base_task}_fm"

    for key in ("output_root", "log_root", "cache_root"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace("vla_ablation_flip_norm065", "vla_ablation_fm")

    # Record the head variant in metadata.
    meta = cfg.get("_ablation_meta", {})
    meta["action_head"] = "FMDecoder"
    cfg["_ablation_meta"] = meta

    return cfg


def main():
    src_files = sorted(p for p in SRC_DIR.glob("e*.json"))
    if not src_files:
        raise SystemExit(f"No source configs found in {SRC_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for src in src_files:
        cfg = convert(src)
        out_path = OUT_DIR / src.name
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=4)
        generated.append(src.name)
        print(f"  Generated: {out_path.relative_to(PROJECT_ROOT)}")

    # README
    readme = OUT_DIR / "README.md"
    lines = [
        "# VLA Ablation Configs — FMDecoder (Flow Matching) variants\n",
        "Flow-matching action-head variants of `configs/vla_ablation/`. Same backbones,",
        "model paths, data, norm settings and training strategies; the `act_head` is",
        "`FMDecoder` (DiT + layer-wise cross-attention) instead of `FCDecoder`.\n",
        f"- `fwd_pred_next_n`: {FWD_PRED_NEXT_N} (action chunk)",
        f"- DiT `cross_attention_dim`: {QWEN25VL_3B_HIDDEN} (= Qwen2.5-VL-3B hidden size)",
        f"- DiT `num_layers`: {DIT_CONFIG['num_layers']} (<= 36 backbone layers)\n",
        "Requires the FMDecoder support added to `RoboQwen25VL.forward_continuous`.\n",
        "| Config | Source |",
        "|--------|--------|",
    ]
    for name in generated:
        lines.append(f"| `{name}` | `../vla_ablation/{name}` |")
    lines.append("\nGenerated by `scripts/generate_ablation_configs_fm.py`")
    readme.write_text("\n".join(lines))

    print(f"\n{len(generated)} FMDecoder configs written to {OUT_DIR}/")
    print(f"README written to {readme}")


if __name__ == "__main__":
    main()
