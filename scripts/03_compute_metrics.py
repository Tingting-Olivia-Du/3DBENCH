#!/usr/bin/env python3
"""Compute and report metrics for the spatial reasoning benchmark.

Reads ground truth from ``data/gt/manifest.json`` and model responses from
``data/responses/<model>/``, then computes per-model, per-question metrics:

  Q1  (target object position)   → MAE/RMSE per axis + overall
  Q1d (destination position)     → MAE/RMSE per axis + overall
  Q2  (gripper position)         → MAE/RMSE per axis + overall
  Q3  (gripper→target offset)    → MAE/RMSE per axis + overall
  Q4  (spatial relation)         → Per-axis accuracy, macro-F1, all-axes accuracy
  Q5  (pairwise distance)        → Scalar MAE, RMSE (m)
  Q6  (EE orientation)           → Per-axis circular MAE (°), geodesic distance (°)
  Q7  (gripper openness)         → Scalar MAE, RMSE
  Q8  (7D next action)           → Per-component MAE, translation/rotation MAE, gripper accuracy

A full results JSON is saved to ``data/results.json``.

Usage
-----
  python scripts/03_compute_metrics.py

  # Custom paths
  python scripts/03_compute_metrics.py \
      --manifest data/gt-q11-libero-test/manifest.json \
      --responses_dir rollout/libero-test-qwen-0512 \
      --out results/libero-test-qwen-0512.json \
      --report results/libero-test-qwen-0512-allq.md

python scripts/03_compute_metrics.py \
    --responses_dir data/runs/20260425_051738 \
    --out data/runs/20260425_051738/results-stat.json \
    --report data/runs/20260425_051738/results.md

python scripts/03_compute_metrics.py \
    --manifest data/gt-demo-libero-all-suite-train-fix/manifest.json \
    --responses_dir rollout/debug/libero-train-qwen-0515-baseline-all-models-all-suite-selected-task \
    --report results/0516/libero-test-baseline-small-cos-simi-nonzero.md

python scripts/03_compute_metrics.py \
    --manifest data/gt-demo-libero-all-suite-train-fix/manifest.json \
    --responses_dir rollout/lora-perdim-ckpt4000 \
    --report results/0516/libero-lora-perdim-ckpt4000.md


python scripts/03_compute_metrics.py \
    --manifest data/gt-demo-libero-all-suite-train-fix/manifest.json \
    --responses_dir rollout/lora-perdim-final \
    --report results/0516/libero-lora-perdim-final.md

    

3DBENCH/rollout/debug/libero-train-qwen-0515-baseline-all-models-all-suite-selected-task
      
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.metrics import (
    _circular_distance_deg,
    angular_metrics,
    format_results_table,
    position_metrics,
    scalar_metrics,
    spatial_relation_metrics,
)
from bench.output_parser import ResponseParser


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",       default="data/gt/manifest.json",
                   help="Path to manifest.json. For multi-suite GT use data/gt/manifest.json "
                        "(combined) or data/gt/<suite>/manifest.json (per-suite).")
    p.add_argument("--responses_dir",  default="data/responses")
    p.add_argument("--suite_filter",   default=None,
                   help="Only evaluate samples from this suite (e.g. libero_spatial). "
                        "Default: all suites in manifest.")
    p.add_argument("--out",            default="data/results.json",
                   help="Path for the full JSON results output")
    p.add_argument("--verbose",        action="store_true",
                   help="Print per-sample parse failures")
    p.add_argument("--report",         default=None,
                   help="Append markdown report to this file (e.g. data/run.md)")
    p.add_argument("--out_csv",        default=None,
                   help="Write per-sample detail CSV to this path (for spreadsheet import). "
                        "If not set, no CSV is written.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: Manifest not found at '{path}'. Run 01_extract_gt.py first.")
        sys.exit(1)
    return json.loads(path.read_text())


def _load_responses(model_dir: Path) -> tuple[dict[int, dict], str]:
    """Load response JSON files for one model from the latest timestamped run.

    Directory layouts supported:
      New with suite:  responses/<model>/<YYYYMMDD_HHMMSS>/<suite>/sample_*.json
      New flat:        responses/<model>/<YYYYMMDD_HHMMSS>/sample_*.json
      Legacy flat:     responses/<model>/sample_*.json
      responses/<model>/latest  (symlink → newest run timestamp)

    Returns ({sample_id: record}, run_timestamp_str).
    """
    # Prefer latest/ symlink or newest timestamped subdirectory
    latest = model_dir / "latest"
    if latest.exists():
        search_dir = latest.resolve()
        run_ts = search_dir.name
    else:
        subdirs = sorted(
            (d for d in model_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        search_dir = subdirs[-1] if subdirs else model_dir
        run_ts = search_dir.name if subdirs else "unknown"

    responses: dict[int, dict] = {}
    # rglob picks up both flat layout (sample_*.json directly) and
    # suite-subdirectory layout (<suite>/sample_*.json)
    for fp in sorted(search_dir.rglob("sample_*.json")):
        try:
            rec = json.loads(fp.read_text())
            responses[int(rec["sample_id"])] = rec
        except Exception as exc:
            print(f"  WARNING: Could not load {fp}: {exc}")
    return responses, run_ts


def _safe_list(val) -> list[float] | None:
    """Return val as list if it is list/tuple, else None."""
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [float(v) for v in val]
    return None


# ---------------------------------------------------------------------------
# Q8 (7D action) metrics helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Returns 0 if either is zero-length."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _compute_q8_metrics(gt_q8: list[list[float]], pred_q8: list[list[float]], n_response: int) -> dict:
    """Compute per-component and aggregate metrics for the 7D next-action prediction.

    Each element is a 7-vector: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    Returns MAE per component, overall translation/rotation MAE, cosine similarity
    for translation direction, and gripper accuracy.
    """
    n = len(gt_q8)
    if n == 0:
        return {"n": 0, "parse_rate": 0.0}

    gt = np.array(gt_q8)   # (n, 7)
    pr = np.array(pred_q8)  # (n, 7)
    ae = np.abs(pr - gt)    # (n, 7)

    comp_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    result: dict = {"n": n, "parse_rate": n / n_response if n_response > 0 else 0.0}

    # Per-component MAE
    for i, name in enumerate(comp_names):
        result[f"mae_{name}"] = float(np.mean(ae[:, i]))

    # Translation MAE (average of per-axis MAEs)
    result["mae_translation"] = float(np.mean(ae[:, :3]))
    # Rotation MAE (average of per-axis MAEs)
    result["mae_rotation"] = float(np.mean(ae[:, 3:6]))

    # Translation cosine similarity (direction accuracy)
    cos_sims = [_cosine_similarity(gt[i, :3], pr[i, :3]) for i in range(n)]
    result["translation_cos_sim"] = float(np.mean(cos_sims))

    # Non-zero translation prediction ratio (pred norm > 1e-6)
    pred_norms = np.linalg.norm(pr[:, :3], axis=1)
    n_nonzero = int(np.sum(pred_norms > 1e-6))
    result["translation_nonzero_ratio"] = n_nonzero / n if n > 0 else 0.0

    # Cosine similarity only on non-zero predictions (more meaningful)
    if n_nonzero > 0:
        nonzero_mask = pred_norms > 1e-6
        cos_sims_nonzero = [_cosine_similarity(gt[i, :3], pr[i, :3])
                            for i in range(n) if nonzero_mask[i]]
        result["translation_cos_sim_nonzero"] = float(np.mean(cos_sims_nonzero))
    else:
        result["translation_cos_sim_nonzero"] = None

    # Gripper accuracy: sign match (both > 0 or both < 0)
    gt_sign = gt[:, 6] > 0
    pr_sign = pr[:, 6] > 0
    result["gripper_accuracy"] = float(np.mean(gt_sign == pr_sign))

    return result


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_slug: str,
    manifest: list[dict],
    responses: dict[int, dict],
    parser: ResponseParser,
    verbose: bool = False,
) -> dict:
    """Return a metrics dict for one model, including per-sample comparison."""
    gt_q1, pred_q1 = [], []
    gt_q1d, pred_q1d = [], []   # destination position
    gt_q2, pred_q2 = [], []
    gt_q3, pred_q3 = [], []     # gripper-to-target offset vector
    gt_q4, pred_q4 = [], []     # spatial relation dicts
    # Q5: pairwise distance (scalar)
    gt_q5, pred_q5 = [], []
    # Q6: EE orientation (euler + quaternion pairs)
    gt_q6e, pred_q6e = [], []   # euler deg
    gt_q6q, pred_q6q = [], []   # quaternion wxyz
    # Q7: gripper openness (scalar)
    gt_q7, pred_q7 = [], []
    # Q8: 7D next action [dx,dy,dz,droll,dpitch,dyaw,gripper]
    gt_q8, pred_q8 = [], []
    task_type_correct = []      # task_type classification

    n_total = len(manifest)
    n_response = 0

    # Per-sample records for the comparison table
    sample_records: list[dict] = []

    for record in manifest:
        sid = record["sample_id"]
        gt = record["gt"]

        if sid not in responses:
            if verbose:
                print(f"  [missing] sample {sid:04d}")
            continue

        n_response += 1
        raw = responses[sid].get("raw_response", "")
        parsed = parser.parse(raw)

        # ---- task_type classification ----
        gt_task_type = gt.get("task_type")
        pred_task_type = parsed.get("task_type")
        if gt_task_type and pred_task_type:
            task_type_correct.append(int(gt_task_type == pred_task_type))

        # ---- Q1: source object position ----
        gt_q1_val = _safe_list(gt.get("target_pos"))
        pred_q1_val = _safe_list(parsed["q1_object_pos"])
        if gt_q1_val and pred_q1_val:
            gt_q1.append(gt_q1_val)
            pred_q1.append(pred_q1_val)

        # ---- Q1_dest: destination position ----
        gt_q1d_val = _safe_list(gt.get("dest_pos"))
        pred_q1d_val = _safe_list(parsed["q1_dest_pos"])
        if gt_q1d_val and pred_q1d_val:
            gt_q1d.append(gt_q1d_val)
            pred_q1d.append(pred_q1d_val)

        # ---- Q2: gripper position ----
        gt_q2_val = _safe_list(gt.get("eef_pos"))
        pred_q2_val = _safe_list(parsed["q2_gripper_pos"])
        if gt_q2_val and pred_q2_val:
            gt_q2.append(gt_q2_val)
            pred_q2.append(pred_q2_val)

        # ---- Q3: gripper-to-target offset vector ----
        gt_q3_val = _safe_list(gt.get("gripper_to_target_delta"))
        pred_q3_val = _safe_list(parsed.get("q3_gripper_to_target"))
        if gt_q3_val and pred_q3_val:
            gt_q3.append(gt_q3_val)
            pred_q3.append(pred_q3_val)

        # ---- Q4: spatial relation classification ----
        gt_q4_val = gt.get("gripper_to_target_relation")
        pred_q4_val = parsed.get("q4_spatial_relation")
        # Only include if GT has the field and all axes of pred are non-None
        if (
            isinstance(gt_q4_val, dict)
            and isinstance(pred_q4_val, dict)
            and all(pred_q4_val.get(ax) is not None for ax in ("x", "y", "z"))
        ):
            gt_q4.append(gt_q4_val)
            pred_q4.append(pred_q4_val)

        # ---- Q5: Pairwise distance ----
        gt_q5_dict = gt.get("pairwise_distance")
        pred_q5_dict = parsed.get("q5_pairwise_distance")
        if isinstance(gt_q5_dict, dict) and isinstance(pred_q5_dict, dict):
            gt_d = gt_q5_dict.get("distance_m")
            pred_d = pred_q5_dict.get("distance_m")
            if gt_d is not None and pred_d is not None:
                gt_q5.append(float(gt_d))
                pred_q5.append(float(pred_d))

        # ---- Q6: EE orientation ----
        gt_q6e_val = _safe_list(gt.get("eef_orientation_euler_deg"))
        pred_q6e_val = _safe_list(parsed.get("q6_eef_orientation_euler"))
        if gt_q6e_val and pred_q6e_val:
            gt_q6e.append(gt_q6e_val); pred_q6e.append(pred_q6e_val)
            # Also collect quaternions for geodesic metric
            gt_q6q_val = gt.get("eef_orientation_quat")
            if gt_q6q_val:
                gt_q6q.append(gt_q6q_val)
                # VLM only returns euler; convert pred euler to quat for geodesic
                from bench.gt_extractor import euler_deg_to_quat_wxyz
                pred_q6q.append(euler_deg_to_quat_wxyz(pred_q6e_val))

        # ---- Q7: Gripper openness ----
        gt_q7_val = gt.get("gripper_openness")
        pred_q7_val = parsed.get("q7_gripper_openness")
        if gt_q7_val is not None and pred_q7_val is not None:
            gt_q7.append(float(gt_q7_val))
            pred_q7.append(float(pred_q7_val))

        # ---- Q8: 7D next action ----
        gt_q8_val = gt.get("demo_action")   # list of 7 floats
        pred_q8_val = parsed.get("q8_next_action")  # dict with translation, rotation, gripper
        # Normalise pred into a 7-element list [dx,dy,dz,droll,dpitch,dyaw,gripper]
        pred_q8_list = None
        if isinstance(pred_q8_val, dict):
            tr = pred_q8_val.get("translation")
            ro = pred_q8_val.get("rotation")
            gr = pred_q8_val.get("gripper")
            if (
                isinstance(tr, (list, tuple)) and len(tr) == 3
                and isinstance(ro, (list, tuple)) and len(ro) == 3
                and gr is not None
            ):
                pred_q8_list = [float(v) for v in tr] + [float(v) for v in ro] + [float(gr)]
        if (
            isinstance(gt_q8_val, (list, tuple)) and len(gt_q8_val) == 7
            and pred_q8_list is not None
        ):
            gt_q8.append([float(v) for v in gt_q8_val])
            pred_q8.append(pred_q8_list)

        # ---- per-sample error fields ----
        q1_err = (
            [round(abs(p - g), 4) for p, g in zip(pred_q1_val, gt_q1_val)]
            if gt_q1_val and pred_q1_val else None
        )
        q1d_err = (
            [round(abs(p - g), 4) for p, g in zip(pred_q1d_val, gt_q1d_val)]
            if gt_q1d_val and pred_q1d_val else None
        )
        q2_err = (
            [round(abs(p - g), 4) for p, g in zip(pred_q2_val, gt_q2_val)]
            if gt_q2_val and pred_q2_val else None
        )

        q3_err = (
            [round(abs(p - g), 4) for p, g in zip(pred_q3_val, gt_q3_val)]
            if gt_q3_val and pred_q3_val else None
        )

        # Q4 per-axis correctness
        q4_correct = None
        if isinstance(gt_q4_val, dict) and isinstance(pred_q4_val, dict):
            q4_correct = {
                ax: (pred_q4_val.get(ax) == gt_q4_val.get(ax))
                for ax in ("x", "y", "z")
            }

        # Q5 per-sample scalar error
        q5_err = None
        if isinstance(gt_q5_dict, dict) and isinstance(pred_q5_dict, dict):
            gt_d = gt_q5_dict.get("distance_m")
            pred_d = pred_q5_dict.get("distance_m")
            if gt_d is not None and pred_d is not None:
                q5_err = round(abs(float(pred_d) - float(gt_d)), 4)

        # Q6 per-sample circular error (degrees)
        def _euler_err(gt_e, pred_e):
            if gt_e and pred_e:
                return [round(_circular_distance_deg(p, g), 2) for p, g in zip(pred_e, gt_e)]
            return None

        q6_err = _euler_err(gt_q6e_val, pred_q6e_val)

        # Q7 per-sample scalar error
        q7_err = None
        if gt_q7_val is not None and pred_q7_val is not None:
            q7_err = round(abs(float(pred_q7_val) - float(gt_q7_val)), 4)

        # Q8 per-sample action errors
        q8_trans_err = None
        q8_rot_err = None
        q8_grip_err = None
        q8_trans_cos_sim = None
        if isinstance(gt_q8_val, (list, tuple)) and len(gt_q8_val) == 7 and pred_q8_list is not None:
            g8 = [float(v) for v in gt_q8_val]
            p8 = pred_q8_list
            q8_trans_err = [round(abs(p8[i] - g8[i]), 4) for i in range(3)]
            q8_rot_err = [round(abs(p8[i] - g8[i]), 4) for i in range(3, 6)]
            q8_grip_err = round(abs(p8[6] - g8[6]), 4)
            q8_trans_cos_sim = round(_cosine_similarity(
                np.array(g8[:3]), np.array(p8[:3])), 4)

        def _round_list(v):
            return [round(x, 4) for x in v] if v else None

        sample_records.append({
            "sample_id": sid,
            "task_id": record.get("task_id"),
            "task_description": record.get("task_description"),
            "task_type": {
                "gt": gt.get("task_type"),
                "pred": pred_task_type,
                "correct": (gt_task_type == pred_task_type) if gt_task_type and pred_task_type else None,
            },
            "target_object": gt.get("target_object_name"),
            "dest_object": gt.get("dest_object_name"),
            "q1": {
                "gt":   _round_list(gt_q1_val),
                "pred": _round_list(pred_q1_val),
                "abs_err_xyz": q1_err,
            },
            "q1_dest": {
                "gt":   _round_list(gt_q1d_val),
                "pred": _round_list(pred_q1d_val),
                "abs_err_xyz": q1d_err,
            },
            "q2": {
                "gt":   _round_list(gt_q2_val),
                "pred": _round_list(pred_q2_val),
                "abs_err_xyz": q2_err,
            },
            "q3": {
                "gt":   _round_list(gt_q3_val),
                "pred": _round_list(pred_q3_val),
                "abs_err_xyz": q3_err,
            },
            "q4": {
                "gt":   gt_q4_val,
                "pred": pred_q4_val if isinstance(pred_q4_val, dict) else None,
                "correct": q4_correct,
            },
            "q5": {
                "gt":   gt_q5_dict,
                "pred": pred_q5_dict,
                "abs_err": q5_err,
            },
            "q6": {
                "gt":   _round_list(gt_q6e_val),
                "pred": _round_list(pred_q6e_val),
                "err_deg": q6_err,
            },
            "q7": {
                "gt":   float(gt_q7_val) if gt_q7_val is not None else None,
                "pred": float(pred_q7_val) if pred_q7_val is not None else None,
                "abs_err": q7_err,
            },
            "q8": {
                "gt":   [float(v) for v in gt_q8_val] if isinstance(gt_q8_val, (list, tuple)) and len(gt_q8_val) == 7 else None,
                "pred": pred_q8_list,
                "trans_err": q8_trans_err,
                "trans_cos_sim": q8_trans_cos_sim,
                "rot_err": q8_rot_err,
                "grip_err": q8_grip_err,
            },
        })

    return {
        "model": model_slug,
        "n_total": n_total,
        "n_response": n_response,
        "parse_rate": n_response / n_total if n_total > 0 else 0.0,
        "task_type_accuracy": float(np.mean(task_type_correct)) if task_type_correct else None,
        "q1": {
            **position_metrics(pred_q1, gt_q1),
            "parse_rate": len(pred_q1) / n_response if n_response > 0 else 0.0,
        },
        "q1_dest": {
            **position_metrics(pred_q1d, gt_q1d),
            "parse_rate": len(pred_q1d) / n_response if n_response > 0 else 0.0,
        },
        "q2": {
            **position_metrics(pred_q2, gt_q2),
            "parse_rate": len(pred_q2) / n_response if n_response > 0 else 0.0,
        },
        "q3": {
            **position_metrics(pred_q3, gt_q3),
            "parse_rate": len(pred_q3) / n_response if n_response > 0 else 0.0,
        },
        "q4": {
            **spatial_relation_metrics(pred_q4, gt_q4),
            "parse_rate": len(pred_q4) / n_response if n_response > 0 else 0.0,
        },
        "q5": {
            **scalar_metrics(pred_q5, gt_q5),
            "parse_rate": len(pred_q5) / n_response if n_response > 0 else 0.0,
        },
        "q6": {
            **angular_metrics(pred_q6e, gt_q6e, pred_q6q if pred_q6q else None, gt_q6q if gt_q6q else None),
            "parse_rate": len(pred_q6e) / n_response if n_response > 0 else 0.0,
        },
        "q7": {
            **scalar_metrics(pred_q7, gt_q7),
            "parse_rate": len(pred_q7) / n_response if n_response > 0 else 0.0,
        },
        "q8": _compute_q8_metrics(gt_q8, pred_q8, n_response),
        "samples": sample_records,
    }


# ---------------------------------------------------------------------------
# Per-axis detail table
# ---------------------------------------------------------------------------

def _axis_table(results: dict[str, dict]) -> str:
    """Print per-axis MAE breakdown for position-type questions."""
    lines = ["\nPer-axis MAE breakdown (meters)", "-" * 70]
    header = f"{'Model':<30}  {'Q':>2}  {'MAE_x':>8}  {'MAE_y':>8}  {'MAE_z':>8}  {'note'}"
    lines.append(header)
    lines.append("-" * 70)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    for slug, m in results.items():
        for q_key, label in (("q1", "Q1"), ("q2", "Q2"), ("q3", "Q3")):
            qm = m.get(q_key, {})
            lines.append(
                f"{slug:<30}  {label}  "
                f"{_f(qm.get('mae_x')):>8}  "
                f"{_f(qm.get('mae_y')):>8}  "
                f"{_f(qm.get('mae_z')):>8}  "
                f"n={qm.get('n', 0)}"
            )
    lines.append("-" * 70)
    return "\n".join(lines)


def _q4_detail(results: dict[str, dict]) -> str:
    """Print Q4 spatial relation per-axis accuracy stats."""
    lines = ["\nQ4 (spatial_relation) detail", "-" * 90]
    lines.append(
        f"{'Model':<30}  {'AccX':>8}  {'AccY':>8}  {'AccZ':>8}  "
        f"{'AccAll':>8}  {'F1X':>8}  {'F1Y':>8}  {'F1Z':>8}  {'Parse%':>8}"
    )
    lines.append("-" * 90)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    def _p(v):
        return f"{v:.1%}" if v is not None else "  N/A  "

    for slug, m in results.items():
        qm = m.get("q4", {})
        lines.append(
            f"{slug:<30}  "
            f"{_f(qm.get('acc_x')):>8}  "
            f"{_f(qm.get('acc_y')):>8}  "
            f"{_f(qm.get('acc_z')):>8}  "
            f"{_f(qm.get('acc_all')):>8}  "
            f"{_f(qm.get('f1_x')):>8}  "
            f"{_f(qm.get('f1_y')):>8}  "
            f"{_f(qm.get('f1_z')):>8}  "
            f"{_p(qm.get('parse_rate')):>8}"
        )
    lines.append("-" * 90)
    return "\n".join(lines)


def _q8_detail(results: dict[str, dict]) -> str:
    """Print Q8 (7D next action) detail."""
    lines = ["\nQ8 (7D next action) detail", "-" * 135]
    lines.append(
        f"{'Model':<30}  {'MAE_dx':>8}  {'MAE_dy':>8}  {'MAE_dz':>8}  "
        f"{'MAE_dr':>8}  {'MAE_dp':>8}  {'MAE_dw':>8}  "
        f"{'TransMAE':>8}  {'TransCos':>8}  {'NZ_Cos':>8}  {'NZ%':>8}  "
        f"{'RotMAE':>8}  {'GripAcc':>8}  {'Parse%':>8}"
    )
    lines.append("-" * 135)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    def _p(v):
        return f"{v:.1%}" if v is not None else "  N/A  "

    for slug, m in results.items():
        qm = m.get("q8", {})
        lines.append(
            f"{slug:<30}  "
            f"{_f(qm.get('mae_dx')):>8}  "
            f"{_f(qm.get('mae_dy')):>8}  "
            f"{_f(qm.get('mae_dz')):>8}  "
            f"{_f(qm.get('mae_droll')):>8}  "
            f"{_f(qm.get('mae_dpitch')):>8}  "
            f"{_f(qm.get('mae_dyaw')):>8}  "
            f"{_f(qm.get('mae_translation')):>8}  "
            f"{_f(qm.get('translation_cos_sim')):>8}  "
            f"{_f(qm.get('translation_cos_sim_nonzero')):>8}  "
            f"{_p(qm.get('translation_nonzero_ratio')):>8}  "
            f"{_f(qm.get('mae_rotation')):>8}  "
            f"{_p(qm.get('gripper_accuracy')):>8}  "
            f"{_p(qm.get('parse_rate')):>8}"
        )
    lines.append("-" * 135)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _extract_numeric(cell: str) -> float | None:
    """Try to extract a numeric value from a formatted markdown cell.

    Handles formats like: `0.1234`, 85.3%, `0.1234`, plain numbers, —/N/A.
    """
    s = str(cell).strip()
    if s in ("—", "N/A", "", "—"):
        return None
    # Strip markdown backticks
    s = s.strip("`")
    # Strip percentage sign and convert
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    # Try direct float parse
    try:
        return float(s)
    except ValueError:
        return None


def _highlight_best(
    rows: list[list[str]],
    col_directions: dict[int, str],
) -> list[list[str]]:
    """Highlight best (bold) and second-best (underline) values per column.

    Parameters
    ----------
    rows : list of list of str
        Table rows (each row is a list of cell strings).
    col_directions : dict[int, str]
        Maps column index → "lower" (lower is better) or "higher" (higher is better).
        Columns not in this dict are left untouched.

    Returns a new list of rows with markdown formatting applied.
    Only highlights when there are ≥ 2 models (rows).
    """
    if len(rows) < 2:
        return rows

    new_rows = [list(r) for r in rows]  # deep copy

    for col_idx, direction in col_directions.items():
        # Collect (row_index, numeric_value) for valid entries
        values: list[tuple[int, float]] = []
        for i, row in enumerate(rows):
            if col_idx >= len(row):
                continue
            v = _extract_numeric(row[col_idx])
            if v is not None:
                values.append((i, v))

        if len(values) < 2:
            continue

        # Sort by value
        reverse = (direction == "higher")
        values.sort(key=lambda x: x[1], reverse=reverse)

        best_idx = values[0][0]
        second_idx = values[1][0]

        # Don't highlight if values are identical
        if values[0][1] == values[1][1]:
            continue

        # Apply formatting
        new_rows[best_idx][col_idx] = f"**{rows[best_idx][col_idx]}**"
        new_rows[second_idx][col_idx] = f"<u>{rows[second_idx][col_idx]}</u>"

    return new_rows


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a list of rows as a GitHub-Flavoured Markdown table."""
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: str,
    all_results: dict[str, dict],
    manifest: list[dict],
    suite: str = "",
) -> None:
    """Append a structured markdown section to *path* (creates file if needed)."""
    import datetime as dt

    def _f(v, fmt=".4f"):
        return f"`{v:{fmt}}`" if v is not None else "—"
    def _p(v):
        return f"{v:.1%}" if v is not None else "—"

    display = {k: {ek: ev for ek, ev in v.items() if ek != "samples"}
               for k, v in all_results.items()}

    lines: list[str] = []
    lines.append(f"\n## Results  —  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if suite:
        lines.append(f"**Suite:** `{suite}`  |  **Samples:** {len(manifest)}\n")

    # --- Summary table ---
    lines.append("### Summary\n")
    hdrs = ["Model", "Run", "Parse %",
            "Q1 Source Obj Pos MAE (m)", "Q1d Dest Pos MAE (m)",
            "Q2 Gripper Pos MAE (m)", "Q3 Grip→Target Offset MAE (m)",
            "Q4 Spatial Relation AccAll",
            "Q5 Pairwise Dist MAE (m)", "Q6 EE Orientation MAE (°)",
            "Q7 Gripper Openness MAE",
            "Q8 Action Trans MAE", "Q8 Action Trans CosSim",
            "Q8 Action Trans CosSim(NZ)", "Q8 Action NZ%",
            "Q8 Action Rot MAE", "Q8 Action Grip Acc",
            "TaskType Acc"]
    rows = []
    for slug, m in display.items():
        rows.append([
            f"`{slug}`",
            f"`{m.get('run_timestamp', '—')}`",
            _p(m.get("parse_rate")),
            _f(m.get("q1", {}).get("mae_overall")),
            _f(m.get("q1_dest", {}).get("mae_overall")),
            _f(m.get("q2", {}).get("mae_overall")),
            _f(m.get("q3", {}).get("mae_overall")),
            _f(m.get("q4", {}).get("acc_all")),
            _f(m.get("q5", {}).get("mae")),
            _f(m.get("q6", {}).get("mae_overall_deg"), ".1f"),
            _f(m.get("q7", {}).get("mae")),
            _f(m.get("q8", {}).get("mae_translation")),
            _f(m.get("q8", {}).get("translation_cos_sim")),
            _f(m.get("q8", {}).get("translation_cos_sim_nonzero")),
            _p(m.get("q8", {}).get("translation_nonzero_ratio")) if m.get("q8", {}).get("translation_nonzero_ratio") is not None else "—",
            _f(m.get("q8", {}).get("mae_rotation")),
            _p(m.get("q8", {}).get("gripper_accuracy")) if m.get("q8", {}).get("gripper_accuracy") is not None else "—",
            _p(m.get("task_type_accuracy")) if m.get("task_type_accuracy") is not None else "—",
        ])
    rows = _highlight_best(rows, {
        2: "higher", 3: "lower", 4: "lower", 5: "lower",
        6: "lower", 7: "higher", 8: "lower", 9: "lower",
        10: "lower", 11: "lower", 12: "higher", 13: "higher", 14: "higher",
        15: "lower", 16: "higher",
        17: "higher",
    })
    lines.append(_md_table(hdrs, rows))
    lines.append("")

    # --- Per-axis MAE ---
    lines.append("### Per-axis MAE (meters)\n")
    hdrs2 = ["Model", "Dimension", "MAE x", "MAE y", "MAE z", "RMSE overall", "n"]
    # Build rows grouped by Q-type so highlighting compares same metric
    # across models. Layout: for each Q-type, one row per model.
    rows2 = []
    for q_key, label in (
        ("q1", "Q1 Source Obj Pos"), ("q1_dest", "Q1d Dest Pos"),
        ("q2", "Q2 Gripper Pos"), ("q3", "Q3 Grip→Target Offset"),
    ):
        group = []
        for slug, m in display.items():
            qm = m.get(q_key, {})
            group.append([
                f"`{slug}`", label,
                _f(qm.get("mae_x")), _f(qm.get("mae_y")), _f(qm.get("mae_z")),
                _f(qm.get("rmse_overall")), str(qm.get("n", 0)),
            ])
        # Highlight within this Q-group (all lower-is-better)
        group = _highlight_best(group, {2: "lower", 3: "lower", 4: "lower", 5: "lower"})
        rows2.extend(group)
    lines.append(_md_table(hdrs2, rows2))
    lines.append("")

    # --- Q4 detail ---
    lines.append("### Q4 Spatial Relation — Per-axis Accuracy & F1\n")
    hdrs4_sr = ["Model", "Acc X", "Acc Y", "Acc Z", "Acc All", "F1 X", "F1 Y", "F1 Z", "Parse %"]
    rows4_sr = []
    for slug, m in display.items():
        qm = m.get("q4", {})
        rows4_sr.append([
            f"`{slug}`",
            _f(qm.get("acc_x")), _f(qm.get("acc_y")), _f(qm.get("acc_z")),
            _f(qm.get("acc_all")),
            _f(qm.get("f1_x")), _f(qm.get("f1_y")), _f(qm.get("f1_z")),
            _p(qm.get("parse_rate")),
        ])
    rows4_sr = _highlight_best(rows4_sr, {
        1: "higher", 2: "higher", 3: "higher", 4: "higher",
        5: "higher", 6: "higher", 7: "higher", 8: "higher",
    })
    lines.append(_md_table(hdrs4_sr, rows4_sr))
    lines.append("")

    # --- Q6 Orientation detail ---
    lines.append("### Q6 EE Orientation — Circular MAE & Geodesic (degrees)\n")
    hdrs6_or = ["Model", "MAE Roll", "MAE Pitch", "MAE Yaw", "MAE Overall",
                "Geodesic Mean", "Geodesic Median", "n"]
    rows6_or = []
    for slug, m in display.items():
        qm = m.get("q6", {})
        rows6_or.append([
            f"`{slug}`",
            _f(qm.get("mae_roll_deg"), ".1f"),
            _f(qm.get("mae_pitch_deg"), ".1f"),
            _f(qm.get("mae_yaw_deg"), ".1f"),
            _f(qm.get("mae_overall_deg"), ".1f"),
            _f(qm.get("geodesic_mean_deg"), ".1f"),
            _f(qm.get("geodesic_median_deg"), ".1f"),
            str(qm.get("n", 0)),
        ])
    rows6_or = _highlight_best(rows6_or, {
        1: "lower", 2: "lower", 3: "lower", 4: "lower", 5: "lower", 6: "lower",
    })
    lines.append(_md_table(hdrs6_or, rows6_or))
    lines.append("")

    # --- Q5/Q7 Scalar detail ---
    lines.append("### Q5 Pairwise Dist / Q7 Gripper Openness — Scalar MAE & RMSE\n")
    hdrs_sc = ["Model", "Dimension", "MAE", "RMSE", "n"]
    rows_sc = []
    for q_key, label in (("q5", "Q5 Pairwise Dist (m)"), ("q7", "Q7 Gripper Openness")):
        group = []
        for slug, m in display.items():
            qm = m.get(q_key, {})
            group.append([
                f"`{slug}`", label,
                _f(qm.get("mae")), _f(qm.get("rmse")),
                str(qm.get("n", 0)),
            ])
        group = _highlight_best(group, {2: "lower", 3: "lower"})
        rows_sc.extend(group)
    lines.append(_md_table(hdrs_sc, rows_sc))
    lines.append("")

    # --- Q8 7D action detail ---
    lines.append("### Q8 7D Next Action — Per-component MAE, CosSim, Grip Acc\n")
    hdrs8 = ["Model", "MAE dx", "MAE dy", "MAE dz", "MAE droll", "MAE dpitch", "MAE dyaw",
             "Trans MAE", "Trans CosSim", "Trans CosSim(NZ)", "NZ%", "Rot MAE", "Grip Acc", "n"]
    rows8 = []
    for slug, m in display.items():
        qm = m.get("q8", {})
        rows8.append([
            f"`{slug}`",
            _f(qm.get("mae_dx")), _f(qm.get("mae_dy")), _f(qm.get("mae_dz")),
            _f(qm.get("mae_droll")), _f(qm.get("mae_dpitch")), _f(qm.get("mae_dyaw")),
            _f(qm.get("mae_translation")), _f(qm.get("translation_cos_sim")),
            _f(qm.get("translation_cos_sim_nonzero")),
            _p(qm.get("translation_nonzero_ratio")) if qm.get("translation_nonzero_ratio") is not None else "—",
            _f(qm.get("mae_rotation")),
            _p(qm.get("gripper_accuracy")) if qm.get("gripper_accuracy") is not None else "—",
            str(qm.get("n", 0)),
        ])
    rows8 = _highlight_best(rows8, {
        1: "lower", 2: "lower", 3: "lower", 4: "lower", 5: "lower", 6: "lower",
        7: "lower", 8: "higher", 9: "higher", 10: "higher", 11: "lower", 12: "higher",
    })
    lines.append(_md_table(hdrs8, rows8))
    lines.append("")

    # --- Full per-sample comparison (all models) ---
    def _xyz(d, key):
        v = d.get(key) if d else None
        if isinstance(v, list) and len(v) == 3:
            return f"`[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]`"
        return "—"

    def _mae_xyz(d):
        v = d.get("abs_err_xyz") if d else None
        if isinstance(v, list) and len(v) == 3:
            overall = (sum(x**2 for x in v) / 3) ** 0.5
            return f"`[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]` → `{overall:.4f}`"
        return "—"

    for slug, result in all_results.items():
        samples = result.get("samples", [])
        if not samples:
            continue
        lines.append(f"### Per-Sample Detail  (`{slug}`, {len(samples)} samples)\n")
        hdrs4 = [
            "ID", "Task", "Target Obj",
            "Q1 Source Obj gt", "Q1 Source Obj pred", "Q1 Source Obj MAE",
            "Q1d Dest Pos gt", "Q1d Dest Pos pred", "Q1d Dest Pos MAE",
            "Q2 Gripper Pos gt", "Q2 Gripper Pos pred", "Q2 Gripper Pos MAE",
            "Q3 Grip→Target gt", "Q3 Grip→Target pred", "Q3 Grip→Target MAE",
            "Q4 Spatial Rel gt", "Q4 Spatial Rel pred", "Q4 Spatial Rel correct",
            "Q5 Pairwise Dist gt", "Q5 Pairwise Dist pred", "Q5 Pairwise Dist err",
            "Q6 EE Orient gt", "Q6 EE Orient pred", "Q6 EE Orient err (°)",
            "Q7 Grip Open gt", "Q7 Grip Open pred", "Q7 Grip Open err",
            "Q8 Action gt", "Q8 Action pred",
            "Q8 Action trans err", "Q8 Action trans cos",
            "Q8 Action rot err", "Q8 Action grip err",
        ]

        def _relation(d, key):
            v = d.get(key) if d else None
            if isinstance(v, dict):
                return f"`{v.get('x','?')}/{v.get('y','?')}/{v.get('z','?')}`"
            return "—"

        def _q4_correct(d):
            v = d.get("correct") if d else None
            if isinstance(v, dict):
                return "`{}/{}/{}`".format(
                    "✓" if v.get("x") else "✗",
                    "✓" if v.get("y") else "✗",
                    "✓" if v.get("z") else "✗",
                )
            return "—"

        def _euler_fmt(d, key):
            v = d.get(key) if d else None
            if isinstance(v, list) and len(v) == 3:
                return f"`[{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]`"
            return "—"

        def _scalar_fmt(v):
            if v is None:
                return "—"
            return f"`{v:.4f}`"

        def _q5_fmt(d, key):
            v = d.get(key) if d else None
            if isinstance(v, dict) and v.get("distance_m") is not None:
                return f"`{float(v['distance_m']):.4f}`"
            return "—"

        def _action_xyz(vals):
            """Format a 3-element sub-list from Q8 action."""
            if isinstance(vals, list) and len(vals) == 3:
                return f"`[{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}]`"
            return "—"

        def _action7(vals):
            """Format a 7-element action list."""
            if isinstance(vals, list) and len(vals) == 7:
                return f"`[{', '.join(f'{v:.4f}' for v in vals)}]`"
            return "—"

        rows4 = []
        for s in samples:
            q5 = s.get("q5", {})
            q7 = s.get("q7", {})
            q8 = s.get("q8", {})
            rows4.append([
                str(s["sample_id"]),
                s["task_description"][:50] + ("…" if len(s["task_description"]) > 50 else ""),
                s.get("target_object", "—"),
                _xyz(s.get("q1", {}), "gt"),
                _xyz(s.get("q1", {}), "pred"),
                _mae_xyz(s.get("q1", {})),
                _xyz(s.get("q1_dest", {}), "gt"),
                _xyz(s.get("q1_dest", {}), "pred"),
                _mae_xyz(s.get("q1_dest", {})),
                _xyz(s.get("q2", {}), "gt"),
                _xyz(s.get("q2", {}), "pred"),
                _mae_xyz(s.get("q2", {})),
                _xyz(s.get("q3", {}), "gt"),
                _xyz(s.get("q3", {}), "pred"),
                _mae_xyz(s.get("q3", {})),
                _relation(s.get("q4", {}), "gt"),
                _relation(s.get("q4", {}), "pred"),
                _q4_correct(s.get("q4", {})),
                _q5_fmt(q5, "gt"),
                _q5_fmt(q5, "pred"),
                _scalar_fmt(q5.get("abs_err")),
                _euler_fmt(s.get("q6", {}), "gt"),
                _euler_fmt(s.get("q6", {}), "pred"),
                _euler_fmt(s.get("q6", {}), "err_deg"),
                _scalar_fmt(q7.get("gt")),
                _scalar_fmt(q7.get("pred")),
                _scalar_fmt(q7.get("abs_err")),
                _action7(q8.get("gt")),
                _action7(q8.get("pred")),
                _action_xyz(q8.get("trans_err")),
                _scalar_fmt(q8.get("trans_cos_sim")),
                _action_xyz(q8.get("rot_err")),
                _scalar_fmt(q8.get("grip_err")),
            ])
        lines.append(_md_table(hdrs4, rows4))
        lines.append("")

    lines.append("---\n")

    with open(path, "a") as f:
        f.write("\n".join(lines))
    print(f"Markdown report appended to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    manifest = _load_manifest(Path(args.manifest))

    # Optional per-suite filter
    if args.suite_filter:
        before = len(manifest)
        manifest = [r for r in manifest if r.get("suite") == args.suite_filter]
        print(f"Suite filter '{args.suite_filter}': {len(manifest)}/{before} samples kept.")
        if not manifest:
            print("No samples matched. Check --suite_filter value.")
            sys.exit(1)

    responses_dir = Path(args.responses_dir)

    if not responses_dir.exists():
        print(
            f"ERROR: Responses directory '{responses_dir}' not found.\n"
            "Run 02_run_vlm_eval.py first."
        )
        sys.exit(1)

    model_dirs = sorted(d for d in responses_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No model subdirectories found under {responses_dir}.")
        sys.exit(1)

    print(f"Found {len(manifest)} GT samples, {len(model_dirs)} model(s): "
          f"{[d.name for d in model_dirs]}")

    parser = ResponseParser()
    all_results: dict[str, dict] = {}

    for model_dir in model_dirs:
        slug = model_dir.name
        responses, run_ts = _load_responses(model_dir)
        print(f"\nEvaluating {slug} (run: {run_ts}): {len(responses)}/{len(manifest)} responses found.")
        metrics = evaluate_model(slug, manifest, responses, parser, verbose=args.verbose)
        metrics["run_timestamp"] = run_ts
        all_results[slug] = metrics

    # ---------------------------------------------------------------------------
    # Merge per-dim LoRA results into a single synthetic "combined" row.
    # Detects slugs matching "...-lora-q<N>" and picks each Q's best metrics
    # from the corresponding per-dim model. This gives a single row that shows
    # the best achievable per-dim LoRA performance across all dimensions.
    # ---------------------------------------------------------------------------
    import re as _re
    _perdim_pattern = _re.compile(r'^(.+)-lora-(q\d+)$')
    _perdim_groups: dict[str, dict[str, str]] = {}  # prefix → {dim: slug}
    for slug in list(all_results.keys()):
        m = _perdim_pattern.match(slug)
        if m:
            prefix, dim = m.group(1), m.group(2)
            _perdim_groups.setdefault(prefix, {})[dim] = slug

    for prefix, dim_slugs in _perdim_groups.items():
        if len(dim_slugs) < 2:
            continue  # not worth merging a single dim
        combined_slug = f"{prefix}-lora-combined"
        # Map Q key → dim that provides it
        _q_to_dim = {
            "q1": "q1", "q1_dest": "q1", "q2": "q2", "q3": "q3",
            "q4": "q4", "q5": "q5", "q6": "q6", "q7": "q7", "q8": "q8",
        }
        combined: dict = {
            "model": combined_slug,
            "n_total": 0,
            "n_response": 0,
            "parse_rate": 0.0,
            "task_type_accuracy": None,
            "samples": [],
        }
        # Pick each Q's metrics from its matching per-dim model
        for q_key, dim in _q_to_dim.items():
            src_slug = dim_slugs.get(dim)
            if src_slug and src_slug in all_results:
                combined[q_key] = all_results[src_slug].get(q_key, {})
                # Use that model's n_response for parse rate of this Q
                src = all_results[src_slug]
                if src.get("n_response", 0) > combined["n_response"]:
                    combined["n_response"] = src["n_response"]
                    combined["n_total"] = src.get("n_total", 0)
            else:
                combined[q_key] = {}
        if combined["n_total"] > 0:
            combined["parse_rate"] = combined["n_response"] / combined["n_total"]
        # Grab run timestamp from first available dim
        first_src = next((all_results[s] for s in dim_slugs.values() if s in all_results), {})
        combined["run_timestamp"] = first_src.get("run_timestamp", "—")

        # --- Merge per-sample records from each per-dim model ---
        # Build {sample_id → sample_record} index for each per-dim model
        _dim_sample_maps: dict[str, dict[int, dict]] = {}
        for dim, src_slug in dim_slugs.items():
            if src_slug in all_results:
                samples_list = all_results[src_slug].get("samples", [])
                _dim_sample_maps[dim] = {s["sample_id"]: s for s in samples_list}

        # Collect all sample_ids across per-dim models (use first available model's order)
        _all_sample_ids: list[int] = []
        _seen_ids: set[int] = set()
        for dim_map in _dim_sample_maps.values():
            for sid in dim_map:
                if sid not in _seen_ids:
                    _all_sample_ids.append(sid)
                    _seen_ids.add(sid)
        _all_sample_ids.sort()

        # Q key → which dim provides it
        _q_keys_by_dim: dict[str, list[str]] = {}
        for q_key, dim in _q_to_dim.items():
            _q_keys_by_dim.setdefault(dim, []).append(q_key)

        combined_samples: list[dict] = []
        for sid in _all_sample_ids:
            # Start from any per-dim model's sample record for base fields
            base = None
            for dim_map in _dim_sample_maps.values():
                if sid in dim_map:
                    base = dim_map[sid]
                    break
            if base is None:
                continue

            merged = {
                "sample_id": sid,
                "task_id": base.get("task_id"),
                "task_description": base.get("task_description"),
                "task_type": base.get("task_type"),
                "target_object": base.get("target_object"),
                "dest_object": base.get("dest_object"),
            }
            # For each Q key, pick data from its corresponding per-dim model
            for q_key, dim in _q_to_dim.items():
                dim_map = _dim_sample_maps.get(dim, {})
                src_sample = dim_map.get(sid)
                if src_sample and q_key in src_sample:
                    merged[q_key] = src_sample[q_key]
                else:
                    # Fallback: use base sample's data (likely all None/—)
                    merged[q_key] = base.get(q_key, {})

            combined_samples.append(merged)

        combined["samples"] = combined_samples

        all_results[combined_slug] = combined
        dims_str = ", ".join(sorted(dim_slugs.keys()))
        print(f"\n[merged] {combined_slug} from {len(dim_slugs)} per-dim models ({dims_str})")

    # ---------------------------------------------------------------------------
    # Print summary tables
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SPATIAL REASONING BENCHMARK RESULTS")
    print("=" * 90)
    # Strip per-sample data from display copy
    display_results = {k: {ek: ev for ek, ev in v.items() if ek != "samples"}
                       for k, v in all_results.items()}
    print(format_results_table(display_results))
    print(_axis_table(display_results))
    print(_q4_detail(display_results))
    print(_q8_detail(display_results))

    # ---------------------------------------------------------------------------
    # Save results: summary (no sample detail) + per-sample comparison
    # ---------------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary file (compact — no per-sample rows)
    out_path.write_text(json.dumps(display_results, indent=2))
    print(f"\nSummary results saved to {out_path}")

    # Per-sample comparison file (one file per model)
    for slug, result in all_results.items():
        samples = result.get("samples", [])
        if not samples:
            continue
        cmp_path = out_path.parent / f"comparison_{slug}.json"
        cmp_path.write_text(json.dumps(samples, indent=2))
        print(f"Per-sample comparison saved to {cmp_path}")

    # ---------------------------------------------------------------------------
    # Print first-N sample comparison preview
    # ---------------------------------------------------------------------------
    preview_model = next(
        (s for s in all_results if s != "random"),
        next(iter(all_results), None),
    )
    if preview_model:
        samples = all_results[preview_model].get("samples", [])[:5]
        if samples:
            print(f"\nSample comparison preview ({preview_model}, first {len(samples)} samples):")
            print(f"{'ID':>4}  {'Task':>40}  {'Q1 gt_z':>8}  {'Q1 pr_z':>8}  "
                  f"{'Q2 gt_z':>8}  {'Q2 pr_z':>8}")
            print("-" * 85)
            def _z(d, key):
                if not d:
                    return "   N/A "
                v = d.get(key)
                return f"{v[2]:.3f}" if v else "   N/A "

            for s in samples:
                print(
                    f"{s['sample_id']:>4}  "
                    f"{s['task_description'][:40]:>40}  "
                    f"{_z(s['q1'], 'gt'):>8}  {_z(s['q1'], 'pred'):>8}  "
                    f"{_z(s['q2'], 'gt'):>8}  {_z(s['q2'], 'pred'):>8}"
                )

    # ---------------------------------------------------------------------------
    # Interpretation hints
    # ---------------------------------------------------------------------------
    print("\nInterpretation notes:")
    print("  Q1/Q2 MAE (m): lower = better. Random baseline ~ 0.15–0.20 m")
    print("  Q3 MAE (m)  : gripper-to-target offset error; lower = better")
    print("  Q4 AccAll   : fraction of samples where all 3 axes are correctly labeled")
    print("  Q5 MAE (m)  : pairwise object distance error; lower = better")
    print("  Q6 (°)      : orientation error in degrees; lower = better")
    print("  Q7 MAE      : gripper openness error [0,1]; lower = better")
    print("  Q8 TransMAE : 7D action translation component error; lower = better")
    print("  Q8 TransCos : translation direction cosine similarity; higher = better (1.0 = perfect)")
    print("  Q8 RotMAE   : 7D action rotation component error; lower = better")
    print("  Q8 GripAcc  : gripper sign accuracy; higher = better")
    print("  Parse rate  : fraction of responses parseable as valid JSON")
    print(f"\n  Per-sample comparisons: data/comparison_<model>.json")

    # ---------------------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------------------
    if args.report:
        suite_label = args.suite_filter or "all"
        write_markdown_report(
            path=args.report,
            all_results=all_results,
            manifest=manifest,
            suite=suite_label,
        )

    # ---------------------------------------------------------------------------
    # CSV export (per-sample detail, one row per sample per model)
    # ---------------------------------------------------------------------------
    if args.out_csv:
        import csv
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        def _csv_xyz(v):
            if isinstance(v, list) and len(v) == 3:
                return f"{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}"
            return ""

        def _csv_scalar(v):
            if v is None:
                return ""
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        def _csv_rel(v):
            if isinstance(v, dict):
                return f"{v.get('x','')}/{v.get('y','')}/{v.get('z','')}"
            return ""

        def _csv_q4ok(v):
            if isinstance(v, dict):
                return "{}/{}/{}".format(
                    "T" if v.get("x") else "F",
                    "T" if v.get("y") else "F",
                    "T" if v.get("z") else "F",
                )
            return ""

        def _csv_action7(v):
            if isinstance(v, list) and len(v) == 7:
                return ",".join(f"{x:.4f}" for x in v)
            return ""

        csv_hdrs = [
            "model", "sample_id", "task_id", "task_description", "target_object", "dest_object",
            "Q1_Source_Obj_gt_x", "Q1_Source_Obj_gt_y", "Q1_Source_Obj_gt_z",
            "Q1_Source_Obj_pred_x", "Q1_Source_Obj_pred_y", "Q1_Source_Obj_pred_z",
            "Q1_Source_Obj_err_x", "Q1_Source_Obj_err_y", "Q1_Source_Obj_err_z",
            "Q1d_Dest_gt_x", "Q1d_Dest_gt_y", "Q1d_Dest_gt_z",
            "Q1d_Dest_pred_x", "Q1d_Dest_pred_y", "Q1d_Dest_pred_z",
            "Q1d_Dest_err_x", "Q1d_Dest_err_y", "Q1d_Dest_err_z",
            "Q2_Gripper_gt_x", "Q2_Gripper_gt_y", "Q2_Gripper_gt_z",
            "Q2_Gripper_pred_x", "Q2_Gripper_pred_y", "Q2_Gripper_pred_z",
            "Q2_Gripper_err_x", "Q2_Gripper_err_y", "Q2_Gripper_err_z",
            "Q3_Offset_gt_x", "Q3_Offset_gt_y", "Q3_Offset_gt_z",
            "Q3_Offset_pred_x", "Q3_Offset_pred_y", "Q3_Offset_pred_z",
            "Q3_Offset_err_x", "Q3_Offset_err_y", "Q3_Offset_err_z",
            "Q4_SpatialRel_gt", "Q4_SpatialRel_pred", "Q4_SpatialRel_correct",
            "Q5_PairDist_gt", "Q5_PairDist_pred", "Q5_PairDist_err",
            "Q6_Orient_gt_roll", "Q6_Orient_gt_pitch", "Q6_Orient_gt_yaw",
            "Q6_Orient_pred_roll", "Q6_Orient_pred_pitch", "Q6_Orient_pred_yaw",
            "Q6_Orient_err_roll", "Q6_Orient_err_pitch", "Q6_Orient_err_yaw",
            "Q7_GripOpen_gt", "Q7_GripOpen_pred", "Q7_GripOpen_err",
            "Q8_Action_gt", "Q8_Action_pred",
            "Q8_trans_err_x", "Q8_trans_err_y", "Q8_trans_err_z",
            "Q8_trans_cos_sim",
            "Q8_rot_err_roll", "Q8_rot_err_pitch", "Q8_rot_err_yaw",
            "Q8_grip_err",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_hdrs)
            for slug, result in all_results.items():
                for s in result.get("samples", []):
                    q1 = s.get("q1", {})
                    q1d = s.get("q1_dest", {})
                    q2 = s.get("q2", {})
                    q3 = s.get("q3", {})
                    q4 = s.get("q4", {})
                    q5 = s.get("q5", {})
                    q6 = s.get("q6", {})
                    q7 = s.get("q7", {})
                    q8 = s.get("q8", {})

                    def _expand3(v):
                        if isinstance(v, list) and len(v) == 3:
                            return [f"{x:.4f}" for x in v]
                        return ["", "", ""]

                    def _q5_scalar(d, key):
                        v = d.get(key)
                        if isinstance(v, dict) and v.get("distance_m") is not None:
                            return f"{float(v['distance_m']):.4f}"
                        if isinstance(v, (int, float)):
                            return f"{float(v):.4f}"
                        return ""

                    row = [
                        slug,
                        s.get("sample_id", ""),
                        s.get("task_id", ""),
                        s.get("task_description", ""),
                        s.get("target_object", ""),
                        s.get("dest_object", ""),
                        *_expand3(q1.get("gt")),
                        *_expand3(q1.get("pred")),
                        *_expand3(q1.get("abs_err_xyz")),
                        *_expand3(q1d.get("gt")),
                        *_expand3(q1d.get("pred")),
                        *_expand3(q1d.get("abs_err_xyz")),
                        *_expand3(q2.get("gt")),
                        *_expand3(q2.get("pred")),
                        *_expand3(q2.get("abs_err_xyz")),
                        *_expand3(q3.get("gt")),
                        *_expand3(q3.get("pred")),
                        *_expand3(q3.get("abs_err_xyz")),
                        _csv_rel(q4.get("gt")),
                        _csv_rel(q4.get("pred")),
                        _csv_q4ok(q4.get("correct")),
                        _q5_scalar(q5, "gt"),
                        _q5_scalar(q5, "pred"),
                        _csv_scalar(q5.get("abs_err")),
                        *_expand3(q6.get("gt")),
                        *_expand3(q6.get("pred")),
                        *_expand3(q6.get("err_deg")),
                        _csv_scalar(q7.get("gt")),
                        _csv_scalar(q7.get("pred")),
                        _csv_scalar(q7.get("abs_err")),
                        _csv_action7(q8.get("gt")),
                        _csv_action7(q8.get("pred")),
                        *_expand3(q8.get("trans_err")),
                        _csv_scalar(q8.get("trans_cos_sim")),
                        *_expand3(q8.get("rot_err")),
                        _csv_scalar(q8.get("grip_err")),
                    ]
                    writer.writerow(row)
        print(f"Per-sample CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
