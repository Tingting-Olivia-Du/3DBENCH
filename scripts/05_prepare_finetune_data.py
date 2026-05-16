#!/usr/bin/env python3
"""Convert one or more LIBERO QA manifests into per-dimension SFT JSONL files.

For every manifest sample, produces 8 training examples — one per QA dimension.
Q1_dest is folded into Q1: the Q1 example asks for both `q1` and `q1_dest`
(the latter is null for articulation samples).

Train/val/test split is **by `(suite, task_id)`**, not random:
init/traj/close frames within the same episode share the scene and would
trivially leak across a random split. Per suite we hold out one task for
validation and one task for test; everything else goes to training.

Output layout:
    <out_dir>/
      q1/{train,val,test}.jsonl
      q2/{train,val,test}.jsonl
      ...
      q8/{train,val,test}.jsonl
      split_info.json   <-- records which task_ids per suite landed in each fold

Each line in a *.jsonl file:
    {
      "sample_id": ...,
      "suite": ...,
      "task_id": ...,
      "frame_type": ...,
      "image_paths": {"agent": "...", "wrist": "..."},
      "system": "<dim-specific stripped system prompt>",
      "user":   "<dim-specific user prompt with task description>",
      "assistant": "<JSON containing only the dim's keys>"
    }

The `image_paths` are the same relative paths as in the source manifest, so
the training script is responsible for joining them with `--data_root`.

Usage
-----

cd /workspace/tingting/3DBENCH

python scripts/05_prepare_finetune_data.py \
    --manifest_glob 'data/gt-demo-libero-all-suite-train-fix/*/manifest.json' \
    --out_dir data/finetune-0515-fix \
    --val_task 8 --test_task 9



  # 1928-sample, 4-suite production
  python scripts/05_prepare_finetune_data.py \
      --manifest_glob 'data/gt-q6-mv/*/manifest.json' \
      --out_dir data/finetune-mv
      --val_task 8 --test_task 9



  # 40-sample mini smoke
  python scripts/05_prepare_finetune_data.py \
      --manifest_glob 'data/gt-q6-mv-mini/*/manifest.json' \
      --out_dir data/finetune-mv-mini \
      --val_task 8 --test_task 9
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so we can import src/bench
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.per_dim_prompt_builder import PerDimPromptBuilder, VALID_DIMS
from bench.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest_glob", required=True,
                   help="Glob matching one or more manifest.json files "
                        "(e.g. 'data/gt-q6-mv/*/manifest.json').")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for {q1..q8}/{train,val,test}.jsonl.")
    p.add_argument("--val_task", type=int, default=8,
                   help="Per-suite task_id to hold out for validation (default 8).")
    p.add_argument("--test_task", type=int, default=9,
                   help="Per-suite task_id to hold out for testing (default 9).")
    p.add_argument("--float_decimals_m", type=int, default=3,
                   help="Decimal places for meter values in assistant JSON (default 3).")
    p.add_argument("--float_decimals_unit", type=int, default=2,
                   help="Decimal places for unit-vector components in assistant JSON (default 2).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Assistant-target builders (one per dim)
# ---------------------------------------------------------------------------

def _r_m(v, decimals: int):
    """Round a float (or list of floats) to `decimals` places, leave None as is."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [round(float(x), decimals) for x in v]
    return round(float(v), decimals)


def _xyz(v, decimals: int):
    """Convert a [x,y,z] list to {'x':..,'y':..,'z':..} with rounding."""
    if v is None:
        return {"x": None, "y": None, "z": None}
    return {"x": _r_m(v[0], decimals), "y": _r_m(v[1], decimals), "z": _r_m(v[2], decimals)}


def _dxdydz(v, decimals: int):
    return {"dx": _r_m(v[0], decimals), "dy": _r_m(v[1], decimals), "dz": _r_m(v[2], decimals)}


def build_assistant_payload(dim: str, gt: dict, dec_m: int, dec_unit: int) -> dict:
    """Return the dict the assistant should emit for `dim` given the sample's GT.

    New question numbering (May 2026):
        Q1  – source object position       (GT: target_pos)
        Q1_dest – dest position             (GT: dest_pos)
        Q2  – gripper position              (GT: eef_pos)
        Q3  – gripper-to-target offset      (GT: gripper_to_target_delta)
        Q4  – spatial relation              (GT: gripper_to_target_relation)
        Q5  – pairwise distance             (GT: pairwise_distance)
        Q6  – EE orientation                (GT: eef_orientation_euler_deg)
        Q7  – gripper openness              (GT: gripper_openness)
        Q8  – 7-D action                    (GT: demo_action)
    """
    if dim == "q1":
        return {
            "task_type": gt["task_type"],
            "q1":      _xyz(gt["target_pos"], dec_m),
            "q1_dest": _xyz(gt.get("dest_pos"), dec_m),
        }
    if dim == "q2":
        return {"q2": _xyz(gt["eef_pos"], dec_m)}
    if dim == "q3":
        return {"q3": _dxdydz(gt["gripper_to_target_delta"], dec_m)}
    if dim == "q4":
        return {"q4": gt["gripper_to_target_relation"]}
    if dim == "q5":
        pd = gt.get("pairwise_distance")
        if pd is None:
            return {"q5": None}
        return {"q5": {"object_a": pd["object_a"], "object_b": pd["object_b"],
                        "distance_m": _r_m(pd["distance_m"], dec_m)}}
    if dim == "q6":
        e = gt.get("eef_orientation_euler_deg")
        if e is None:
            return {"q6": None}
        return {"q6": {"roll": _r_m(e[0], dec_unit), "pitch": _r_m(e[1], dec_unit), "yaw": _r_m(e[2], dec_unit)}}
    if dim == "q7":
        openness = gt.get("gripper_openness")
        if openness is None:
            return {"q7": None}
        return {"q7": {"openness": _r_m(openness, 2)}}
    if dim == "q8":
        action = gt.get("demo_action")
        if action is None:
            return {"q8": None}
        a = _r_m(action, dec_m)
        return {"q8": {"dx": a[0], "dy": a[1], "dz": a[2],
                        "droll": a[3], "dpitch": a[4], "dyaw": a[5],
                        "gripper": a[6]}}
    raise ValueError(f"unknown dim {dim!r}")


def build_all_dim_payload(gt: dict, dec_m: int, dec_unit: int) -> dict:
    """Build a single assistant JSON containing ALL dimensions (for all-in-one SFT)."""
    payload = {}
    for dim in VALID_DIMS:
        payload.update(build_assistant_payload(dim, gt, dec_m, dec_unit))
    return payload


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(glob.glob(args.manifest_glob))
    if not manifest_paths:
        print(f"ERROR: no manifests matched {args.manifest_glob!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(manifest_paths)} manifest(s):")
    for p in manifest_paths:
        print(f"  {p}")

    # Load all manifests, tag with their source suite for split bookkeeping.
    all_records: list[dict] = []
    for mp in manifest_paths:
        records = json.loads(Path(mp).read_text())
        if records:
            all_records.extend(records)
            print(f"    {mp}: +{len(records)} samples (suite={records[0].get('suite','?')})")

    print(f"Total source samples: {len(all_records)}")

    # Pre-build the 8 prompt builders.
    builders = {d: PerDimPromptBuilder(d) for d in VALID_DIMS}

    # Bucket records by fold.
    folds: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    suites_to_tasks_train: dict[str, set] = defaultdict(set)
    for rec in all_records:
        suite = rec["suite"]
        tid = int(rec["task_id"])
        if tid == args.val_task:
            folds["val"].append(rec)
        elif tid == args.test_task:
            folds["test"].append(rec)
        else:
            folds["train"].append(rec)
            suites_to_tasks_train[suite].add(tid)

    print("Fold sizes (raw samples):", {k: len(v) for k, v in folds.items()})

    # Q1_dest is null for articulation; we still keep the example (target=null is a
    # valid training signal). No skipping needed.
    written_counts: dict[str, dict[str, int]] = {d: {"train": 0, "val": 0, "test": 0} for d in VALID_DIMS}

    for dim in VALID_DIMS:
        dim_dir = out_root / dim
        dim_dir.mkdir(parents=True, exist_ok=True)
        builder = builders[dim]
        sys_prompt = builder.system_prompt  # constant for the dim
        for fold_name, recs in folds.items():
            jsonl_path = dim_dir / f"{fold_name}.jsonl"
            with jsonl_path.open("w") as f:
                for rec in recs:
                    user_prompt = builder.build_user_prompt(rec["task_description"])
                    payload = build_assistant_payload(
                        dim, rec["gt"], args.float_decimals_m, args.float_decimals_unit
                    )
                    # image_paths: prefer the new dual-view dict, fall back to legacy single
                    img_paths = rec.get("image_paths") or {"agent": rec["image_path"]}
                    line = {
                        "sample_id": rec["sample_id"],
                        "suite": rec["suite"],
                        "task_id": int(rec["task_id"]),
                        "frame_type": rec.get("frame_type"),
                        "image_paths": img_paths,
                        "system": sys_prompt,
                        "user": user_prompt,
                        "assistant": json.dumps(payload, ensure_ascii=False),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    written_counts[dim][fold_name] += 1

    # ── All-in-one: single prompt asking all Q1–Q8 (uses spatial_qa.yaml) ──
    all_dir = out_root / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    all_builder = PromptBuilder()  # loads spatial_qa.yaml
    all_sys_prompt = all_builder.system_prompt
    written_counts["all"] = {"train": 0, "val": 0, "test": 0}
    for fold_name, recs in folds.items():
        jsonl_path = all_dir / f"{fold_name}.jsonl"
        with jsonl_path.open("w") as f:
            for rec in recs:
                user_prompt = all_builder.build_user_prompt(rec["task_description"])
                payload = build_all_dim_payload(
                    rec["gt"], args.float_decimals_m, args.float_decimals_unit
                )
                img_paths = rec.get("image_paths") or {"agent": rec["image_path"]}
                line = {
                    "sample_id": rec["sample_id"],
                    "suite": rec["suite"],
                    "task_id": int(rec["task_id"]),
                    "frame_type": rec.get("frame_type"),
                    "image_paths": img_paths,
                    "system": all_sys_prompt,
                    "user": user_prompt,
                    "assistant": json.dumps(payload, ensure_ascii=False),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written_counts["all"][fold_name] += 1

    # Split info for reproducibility / audit.
    split_info = {
        "manifest_glob": args.manifest_glob,
        "manifests": manifest_paths,
        "val_task": args.val_task,
        "test_task": args.test_task,
        "fold_sizes": {k: len(v) for k, v in folds.items()},
        "train_tasks_per_suite": {s: sorted(t) for s, t in suites_to_tasks_train.items()},
        "per_dim_counts": written_counts,
    }
    (out_root / "split_info.json").write_text(json.dumps(split_info, indent=2))

    print("\n=== Per-dim JSONL counts ===")
    for dim in list(VALID_DIMS) + ["all"]:
        c = written_counts[dim]
        print(f"  {dim}: train={c['train']}  val={c['val']}  test={c['test']}")

    print(f"\nWrote split_info.json → {out_root / 'split_info.json'}")
    print(f"Done. Output: {out_root}/")


if __name__ == "__main__":
    main()
