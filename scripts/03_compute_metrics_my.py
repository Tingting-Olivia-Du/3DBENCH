#!/usr/bin/env python3
"""Compute and report metrics for the spatial reasoning benchmark.

Reads ground truth from ``data/gt/manifest.json`` and model responses from
``data/responses/<model>/``, then computes per-model, per-question metrics:

  Q1 (target object position)  → MAE/RMSE per axis + overall
  Q2 (gripper position)        → MAE/RMSE per axis + overall
  Q3 (can gripper close)       → Accuracy, F1, positive rates
  Q4 (next move direction)     → Mean & median cosine similarity

A full results JSON is saved to ``data/results.json``.

Usage
-----
  python scripts/03_compute_metrics.py

  # Custom paths
  python scripts/03_compute_metrics.py \
      --manifest data/gt/manifest.json \
      --responses_dir data/runs/20260416_213810 \
      --out data/runs/20260416_213810/results-stat.json

python scripts/03_compute_metrics.py \
    --responses_dir data/runs/20260425_051738 \
    --out data/runs/20260425_051738/results-stat.json \
    --report data/runs/20260425_051738/results.md

      
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
    classification_metrics,
    direction_metrics,
    format_results_table,
    parse_rate,
    position_metrics,
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
    gt_q3, pred_q3 = [], []
    gt_q4, pred_q4 = [], []
    gt_q5, pred_q5 = [], []     # gripper-to-target offset vector
    gt_q6, pred_q6 = [], []     # spatial relation dicts
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

        # ---- Q3: can close ----
        gt_q3_val = gt.get("can_close")
        pred_q3_val = parsed["q3_can_close"]
        if pred_q3_val is not None:
            gt_q3.append(bool(gt_q3_val))
            pred_q3.append(bool(pred_q3_val))
        elif verbose:
            print(f"  [q3 parse fail] sample {sid:04d}: {raw[:80]!r}")

        # ---- Q4: next direction ----
        gt_q4_val = _safe_list(gt.get("next_direction"))
        pred_q4_val = _safe_list(parsed["q4_next_dir"])
        if gt_q4_val and pred_q4_val:
            gt_q4.append(gt_q4_val)
            pred_q4.append(pred_q4_val)

        # ---- Q5: gripper-to-target offset vector ----
        gt_q5_val = _safe_list(gt.get("gripper_to_target_delta"))
        pred_q5_val = _safe_list(parsed["q5_gripper_to_target"])
        if gt_q5_val and pred_q5_val:
            gt_q5.append(gt_q5_val)
            pred_q5.append(pred_q5_val)

        # ---- Q6: spatial relation classification ----
        gt_q6_val = gt.get("gripper_to_target_relation")
        pred_q6_val = parsed["q6_spatial_relation"]
        # Only include if GT has the field and all axes of pred are non-None
        if (
            isinstance(gt_q6_val, dict)
            and isinstance(pred_q6_val, dict)
            and all(pred_q6_val.get(ax) is not None for ax in ("x", "y", "z"))
        ):
            gt_q6.append(gt_q6_val)
            pred_q6.append(pred_q6_val)

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
        q4_cos = None
        if gt_q4_val and pred_q4_val:
            a = np.array(pred_q4_val); b = np.array(gt_q4_val)
            a /= np.linalg.norm(a) + 1e-8; b /= np.linalg.norm(b) + 1e-8
            q4_cos = round(float(np.dot(a, b)), 4)

        q5_err = (
            [round(abs(p - g), 4) for p, g in zip(pred_q5_val, gt_q5_val)]
            if gt_q5_val and pred_q5_val else None
        )

        # Q6 per-axis correctness
        q6_correct = None
        if isinstance(gt_q6_val, dict) and isinstance(pred_q6_val, dict):
            q6_correct = {
                ax: (pred_q6_val.get(ax) == gt_q6_val.get(ax))
                for ax in ("x", "y", "z")
            }

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
                "gt":   [round(v, 4) for v in gt_q1_val] if gt_q1_val else None,
                "pred": [round(v, 4) for v in pred_q1_val] if pred_q1_val else None,
                "abs_err_xyz": q1_err,
            },
            "q1_dest": {
                "gt":   [round(v, 4) for v in gt_q1d_val] if gt_q1d_val else None,
                "pred": [round(v, 4) for v in pred_q1d_val] if pred_q1d_val else None,
                "abs_err_xyz": q1d_err,
            },
            "q2": {
                "gt":   [round(v, 4) for v in gt_q2_val] if gt_q2_val else None,
                "pred": [round(v, 4) for v in pred_q2_val] if pred_q2_val else None,
                "abs_err_xyz": q2_err,
            },
            "q3": {
                "gt":   bool(gt_q3_val),
                "pred": bool(pred_q3_val) if pred_q3_val is not None else None,
            },
            "q4": {
                "gt":   [round(v, 4) for v in gt_q4_val] if gt_q4_val else None,
                "pred": [round(v, 4) for v in pred_q4_val] if pred_q4_val else None,
                "cosine_sim": q4_cos,
            },
            "q5": {
                "gt":   [round(v, 4) for v in gt_q5_val] if gt_q5_val else None,
                "pred": [round(v, 4) for v in pred_q5_val] if pred_q5_val else None,
                "abs_err_xyz": q5_err,
            },
            "q6": {
                "gt":   gt_q6_val,
                "pred": pred_q6_val if isinstance(pred_q6_val, dict) else None,
                "correct": q6_correct,
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
            **classification_metrics(pred_q3, gt_q3),
            "parse_rate": len(pred_q3) / n_response if n_response > 0 else 0.0,
        },
        "q4": {
            **direction_metrics(pred_q4, gt_q4),
            "parse_rate": len(pred_q4) / n_response if n_response > 0 else 0.0,
        },
        "q5": {
            **position_metrics(pred_q5, gt_q5),
            "parse_rate": len(pred_q5) / n_response if n_response > 0 else 0.0,
        },
        "q6": {
            **spatial_relation_metrics(pred_q6, gt_q6),
            "parse_rate": len(pred_q6) / n_response if n_response > 0 else 0.0,
        },
        "samples": sample_records,
    }


# ---------------------------------------------------------------------------
# Per-axis detail table
# ---------------------------------------------------------------------------

def _axis_table(results: dict[str, dict]) -> str:
    """Print per-axis MAE breakdown for Q1 and Q2."""
    lines = ["\nPer-axis MAE breakdown (meters)", "-" * 70]
    header = f"{'Model':<30}  {'Q':>2}  {'MAE_x':>8}  {'MAE_y':>8}  {'MAE_z':>8}  {'note'}"
    lines.append(header)
    lines.append("-" * 70)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    for slug, m in results.items():
        for q_key, label in (("q1", "Q1"), ("q2", "Q2"), ("q5", "Q5")):
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


def _q6_detail(results: dict[str, dict]) -> str:
    """Print Q6 spatial relation per-axis accuracy stats."""
    lines = ["\nQ6 (spatial_relation) detail", "-" * 90]
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
        qm = m.get("q6", {})
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


def _q3_detail(results: dict[str, dict]) -> str:
    """Print Q3 positive-rate stats."""
    lines = ["\nQ3 (can_close) detail", "-" * 70]
    lines.append(f"{'Model':<30}  {'Acc':>8}  {'F1':>8}  {'GT+%':>8}  {'Pred+%':>8}  {'Parse%':>8}")
    lines.append("-" * 70)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    def _p(v):
        return f"{v:.1%}" if v is not None else "  N/A  "

    for slug, m in results.items():
        qm = m.get("q3", {})
        lines.append(
            f"{slug:<30}  "
            f"{_f(qm.get('accuracy')):>8}  "
            f"{_f(qm.get('f1')):>8}  "
            f"{_p(qm.get('positive_rate_gt')):>8}  "
            f"{_p(qm.get('positive_rate_pred')):>8}  "
            f"{_p(qm.get('parse_rate')):>8}"
        )
    lines.append("-" * 70)
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
    hdrs = ["Model", "Run", "Parse %", "Q1 MAE (m)", "Q1_dest MAE (m)",
            "Q2 MAE (m)", "Q3 Acc", "Q3 F1", "Q4 CosSim",
            "Q5 MAE (m)", "Q6 AccAll", "TaskType Acc"]
    rows = []
    for slug, m in display.items():
        rows.append([
            f"`{slug}`",
            f"`{m.get('run_timestamp', '—')}`",
            _p(m.get("parse_rate")),
            _f(m.get("q1", {}).get("mae_overall")),
            _f(m.get("q1_dest", {}).get("mae_overall")),
            _f(m.get("q2", {}).get("mae_overall")),
            _f(m.get("q3", {}).get("accuracy")),
            _f(m.get("q3", {}).get("f1")),
            _f(m.get("q4", {}).get("mean_cosine_sim")),
            _f(m.get("q5", {}).get("mae_overall")),
            _f(m.get("q6", {}).get("acc_all")),
            _p(m.get("task_type_accuracy")) if m.get("task_type_accuracy") is not None else "—",
        ])
    # Highlight best/second-best across models in the summary table
    #   cols: 2=Parse%(higher), 3=Q1 MAE(lower), 4=Q1d MAE(lower),
    #         5=Q2 MAE(lower), 6=Q3 Acc(higher), 7=Q3 F1(higher),
    #         8=Q4 CosSim(higher), 9=Q5 MAE(lower), 10=Q6 AccAll(higher),
    #         11=TaskType Acc(higher)
    rows = _highlight_best(rows, {
        2: "higher", 3: "lower", 4: "lower", 5: "lower",
        6: "higher", 7: "higher", 8: "higher", 9: "lower",
        10: "higher", 11: "higher",
    })
    lines.append(_md_table(hdrs, rows))
    lines.append("")

    # --- Per-axis MAE ---
    lines.append("### Per-axis MAE (meters)\n")
    hdrs2 = ["Model", "Q", "MAE x", "MAE y", "MAE z", "RMSE overall", "n"]
    # Build rows grouped by Q-type so highlighting compares same metric
    # across models. Layout: for each Q-type, one row per model.
    rows2 = []
    for q_key, label in (
        ("q1", "Q1 source"), ("q1_dest", "Q1 dest"),
        ("q2", "Q2 gripper"), ("q5", "Q5 offset"),
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

    # --- Q3 detail ---
    lines.append("### Q3 — Can Gripper Close\n")
    hdrs3 = ["Model", "Accuracy", "F1", "GT positive %", "Pred positive %", "Parse %"]
    rows3 = []
    for slug, m in display.items():
        qm = m.get("q3", {})
        rows3.append([
            f"`{slug}`",
            _f(qm.get("accuracy")), _f(qm.get("f1")),
            _p(qm.get("positive_rate_gt")), _p(qm.get("positive_rate_pred")),
            _p(qm.get("parse_rate")),
        ])
    rows3 = _highlight_best(rows3, {1: "higher", 2: "higher", 5: "higher"})
    lines.append(_md_table(hdrs3, rows3))
    lines.append("")

    # --- Q6 detail ---
    lines.append("### Q6 — Spatial Relation (Language)\n")
    hdrs6 = ["Model", "Acc X", "Acc Y", "Acc Z", "Acc All", "F1 X", "F1 Y", "F1 Z", "Parse %"]
    rows6 = []
    for slug, m in display.items():
        qm = m.get("q6", {})
        rows6.append([
            f"`{slug}`",
            _f(qm.get("acc_x")), _f(qm.get("acc_y")), _f(qm.get("acc_z")),
            _f(qm.get("acc_all")),
            _f(qm.get("f1_x")), _f(qm.get("f1_y")), _f(qm.get("f1_z")),
            _p(qm.get("parse_rate")),
        ])
    rows6 = _highlight_best(rows6, {
        1: "higher", 2: "higher", 3: "higher", 4: "higher",
        5: "higher", 6: "higher", 7: "higher", 8: "higher",
    })
    lines.append(_md_table(hdrs6, rows6))
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

    def _cos(d):
        v = d.get("cosine_sim") if d else None
        return f"`{v:.4f}`" if v is not None else "—"

    def _bool_sym(v):
        if v is None:
            return "—"
        return "✓" if v else "✗"

    for slug, result in all_results.items():
        samples = result.get("samples", [])
        if not samples:
            continue
        lines.append(f"### Per-Sample Detail  (`{slug}`, {len(samples)} samples)\n")
        hdrs4 = [
            "ID", "Task", "Target Obj",
            "Q1 src gt", "Q1 src pred", "Q1 src MAE (xyz → overall)",
            "Q1 dst gt", "Q1 dst pred", "Q1 dst MAE (xyz → overall)",
            "Q2 gt", "Q2 pred", "Q2 MAE (xyz → overall)",
            "Q3 gt", "Q3 pred",
            "Q4 gt", "Q4 pred", "Q4 CosSim",
            "Q5 gt", "Q5 pred", "Q5 MAE (xyz → overall)",
            "Q6 gt", "Q6 pred", "Q6 correct (xyz)",
        ]

        def _relation(d, key):
            v = d.get(key) if d else None
            if isinstance(v, dict):
                return f"`{v.get('x','?')}/{v.get('y','?')}/{v.get('z','?')}`"
            return "—"

        def _q6_correct(d):
            v = d.get("correct") if d else None
            if isinstance(v, dict):
                return "`{}/{}/{}`".format(
                    "✓" if v.get("x") else "✗",
                    "✓" if v.get("y") else "✗",
                    "✓" if v.get("z") else "✗",
                )
            return "—"

        rows4 = []
        for s in samples:
            q3 = s.get("q3", {})
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
                _bool_sym(q3.get("gt")),
                _bool_sym(q3.get("pred")),
                _xyz(s.get("q4", {}), "gt"),
                _xyz(s.get("q4", {}), "pred"),
                _cos(s.get("q4", {})),
                _xyz(s.get("q5", {}), "gt"),
                _xyz(s.get("q5", {}), "pred"),
                _mae_xyz(s.get("q5", {})),
                _relation(s.get("q6", {}), "gt"),
                _relation(s.get("q6", {}), "pred"),
                _q6_correct(s.get("q6", {})),
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
    print(_q3_detail(display_results))
    print(_q6_detail(display_results))

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
                  f"{'Q2 gt_z':>8}  {'Q2 pr_z':>8}  {'Q4 cos':>8}")
            print("-" * 95)
            def _z(d, key):
                if not d:
                    return "   N/A "
                v = d.get(key)
                return f"{v[2]:.3f}" if v else "   N/A "

            def _cos(d):
                if not d:
                    return "   N/A "
                v = d.get("cosine_sim")
                return f"{v:.4f}" if v is not None else "   N/A "

            for s in samples:
                print(
                    f"{s['sample_id']:>4}  "
                    f"{s['task_description'][:40]:>40}  "
                    f"{_z(s['q1'], 'gt'):>8}  {_z(s['q1'], 'pred'):>8}  "
                    f"{_z(s['q2'], 'gt'):>8}  {_z(s['q2'], 'pred'):>8}  "
                    f"{_cos(s['q4']):>8}"
                )

    # ---------------------------------------------------------------------------
    # Interpretation hints
    # ---------------------------------------------------------------------------
    print("\nInterpretation notes:")
    print("  Q1/Q2 MAE (m): lower = better. Random baseline ~ 0.15–0.20 m")
    print("  Q3 F1       : near 0 = model always says 'no' (correct at task reset)")
    print("  Q4 CosSim   : range [-1,1]; random ~0.0; perfect=1.0")
    print("  Q5 MAE (m)  : gripper-to-target offset error; lower = better")
    print("  Q6 AccAll   : fraction of samples where all 3 axes are correctly labeled")
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


if __name__ == "__main__":
    main()
