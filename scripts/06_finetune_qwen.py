#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL-3B on a single QA dimension with dual-view input.

Reads JSONL produced by `scripts/05_prepare_finetune_data.py` (one dimension's
`train.jsonl` and `val.jsonl`) and trains a LoRA adapter on the LLM half of
Qwen2.5-VL while keeping the vision tower frozen.

Each training example has fields:
    image_paths : {"agent": "<rel>", "wrist": "<rel>"}
    system      : str
    user        : str
    assistant   : str   # the JSON answer for THIS dimension only

The vision tower (vit + merger) is frozen — LoRA targets only the LLM
attention/MLP projections (q/k/v/o/gate/up/down). This is the "safe" recipe
for small datasets and matches the plan in
/root/.claude/plans/libero-qa-...md.

Usage
-----
  python scripts/06_finetune_qwen.py \\
      --dim q3 \\
      --train_jsonl data/finetune-mv/q3/train.jsonl \\
      --val_jsonl   data/finetune-mv/q3/val.jsonl \\
      --data_root   data \\
      --base_model  /umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct \\
      --output_dir  models/qwen2.5-vl-3b-mv-lora/q3 \\
      --epochs 3 --per_device_batch_size 1 --grad_accum 8 --lr 1e-4

  # quick smoke (3 steps, no eval)
  python scripts/06_finetune_qwen.py --dim q3 \\
      --train_jsonl /tmp/finetune_mini/q3/train.jsonl \\
      --val_jsonl   /tmp/finetune_mini/q3/val.jsonl \\
      --output_dir  /tmp/lora_mv_test_q3 \\
      --max_steps 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", required=True, choices=["q1", "q2", "q3", "q4", "q5", "q6"],
                   help="Which dimension this adapter targets.")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--data_root", default="data",
                   help="Root for resolving relative image_paths inside the JSONL.")
    p.add_argument("--base_model",
                   default="/umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct",
                   help="Path or HF id of the base model.")
    p.add_argument("--output_dir", required=True)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If > 0, overrides --epochs (useful for smoke tests).")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--save_steps", type=int, default=0,
                   help="If > 0, save a checkpoint every N steps. 0 = save at epoch end only.")
    p.add_argument("--eval_steps", type=int, default=0,
                   help="If > 0, evaluate every N steps. 0 = evaluate at epoch end only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DualViewSFTDataset(torch.utils.data.Dataset):
    """Wraps a per-dim JSONL file. Returns dicts of (system, user, assistant, agent_path, wrist_path)."""

    def __init__(self, jsonl_path: str, data_root: str):
        self._records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        self._data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        img_paths = rec["image_paths"]
        return {
            "system": rec["system"],
            "user": rec["user"],
            "assistant": rec["assistant"],
            "agent_path": str(self._data_root / img_paths["agent"]),
            "wrist_path": str(self._data_root / img_paths.get("wrist", img_paths["agent"])),
            "has_wrist": "wrist" in img_paths,
        }


def _build_messages(rec: dict) -> list[dict]:
    """Build the Qwen chat-format messages for a single training example."""
    user_content: list = []
    user_content.append({"type": "text", "text": "[AGENT VIEW] (third-person, full workspace)"})
    user_content.append({"type": "image", "image": f"file://{rec['agent_path']}"})
    if rec["has_wrist"]:
        user_content.append({"type": "text", "text": "[WRIST VIEW] (first-person, mounted on the gripper)"})
        user_content.append({"type": "image", "image": f"file://{rec['wrist_path']}"})
    user_content.append({"type": "text", "text": rec["user"]})
    return [
        {"role": "system", "content": rec["system"]},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": rec["assistant"]},
    ]


def make_collator(processor, max_seq_len: int):
    """Return a collate fn that builds a Qwen-VL training batch.

    Loss is computed only on assistant tokens. We achieve this by tokenising
    the prompt (system + user + assistant header) once to find the assistant
    span, then masking labels=-100 for everything before that span.
    """
    from qwen_vl_utils import process_vision_info

    def collate(batch: list[dict]):
        msgs_list = [_build_messages(rec) for rec in batch]

        # Full text (with assistant) for input_ids; prompt-only text for masking.
        texts_full = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in msgs_list
        ]
        texts_prompt = [
            processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            for m in msgs_list
        ]

        image_inputs, video_inputs = process_vision_info(msgs_list)

        inputs = processor(
            text=texts_full,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        # Build labels = input_ids, then mask everything up to the prompt length.
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        # Mask prompt portion per-sample
        prompt_token_lens = [
            len(processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in texts_prompt
        ]
        for i, plen in enumerate(prompt_token_lens):
            plen = min(plen, labels.size(1))
            labels[i, :plen] = -100
        # Mask out padding positions too (attention_mask == 0)
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs

    return collate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred so --help is fast and import errors are visible.
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    print(f"Loading base model: {args.base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.base_model)

    # ── Freeze the vision tower (vit + merger). LoRA only on LLM modules. ──
    # Qwen2.5-VL exposes the vision tower as `model.visual` (or `model.model.visual`
    # depending on transformers version). Freeze any param whose name contains
    # "visual." — robust across versions.
    n_frozen = n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        if "visual" in name:
            p.requires_grad = False
            n_frozen += 1
    print(f"Frozen {n_frozen}/{n_total} parameters (vision tower).")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Datasets ──
    train_ds = DualViewSFTDataset(args.train_jsonl, args.data_root)
    val_ds   = DualViewSFTDataset(args.val_jsonl, args.data_root)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    collator = make_collator(processor, max_seq_len=args.max_seq_len)

    # ── TrainingArguments ──
    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    eval_strategy = "steps" if args.eval_steps > 0 else "epoch"
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=5,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if args.eval_steps > 0 else 500,
        save_total_limit=2,
        load_best_model_at_end=True if args.max_steps <= 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        remove_unused_columns=False,
        report_to=[],   # no wandb/tb — we'll add it later if needed
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    print(f"Starting training: dim={args.dim}  out={args.output_dir}")
    trainer.train()

    # Save the LoRA adapter (PEFT handles the rest of the layout).
    final_dir = Path(args.output_dir) / "final"
    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    # Also drop a small metadata file so 02_run_vlm_eval.py knows the dim.
    (final_dir / "adapter_meta.json").write_text(json.dumps({
        "dim": args.dim,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
    }, indent=2))
    print(f"Saved LoRA adapter → {final_dir}")


if __name__ == "__main__":
    main()
