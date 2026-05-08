#!/usr/bin/env python3
"""Aggregate results across multiple runs, broken down by suite and model.

Usage
-----
  # Aggregate specific runs
  python scripts/04_aggregate_results.py \
      --runs data/runs/20260425_054955 data/runs/20260425_055015 \
             data/runs/20260425_055019 data/runs/20260425_055037 \
             data/runs/20260425_055105 \
      --out data/runs/aggregate_results.json \
      --report data/runs/aggregate_report.md

  # Auto-discover all runs under data/runs/
  python scripts/04_aggregate_results.py --auto --out data/runs/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.metrics import (
    classification_metrics,
    direction_metrics,
    position_metrics,
    spatial_relation_metrics,
)
from bench.output_parser import ResponseParser


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=None,
                   help="List of run directories (e.g. data/runs/20260425_054955 ...)")
    p.add_argument("--auto", action="store_true",
                   help="Auto-discover all timestamped run dirs under --runs_root")
    p.add_argument("--runs_root", default="data/runs",
                   help="Root directory to scan when --auto is used")
    p.add_argument("--out", default="data/runs/aggregate_results.json")
    p.add_argument("--report", default="data/runs/aggregate_report.md")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_list(val) -> list[float] | None:
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [float(v) for v in val]
    return None


def _load_responses_from_run(run_dir: Path, model_name: str) -> dict[int, dict]:
    """Load all sample JSON files for a model within a run directory."""
    model_dir = run_dir / model_name
    if not model_dir.exists():
        return {}
    # find latest timestamped subdir or use model_dir directly
    latest = model_dir / "latest"
    if latest.exists():
        search_dir = latest.resolve()
    else:
        subdirs = sorted(d for d in model_dir.iterdir() if d.is_dir())
        search_dir = subdirs[-1] if subdirs else model_dir

    responses: dict[int, dict] = {}
    for fp in sorted(search_dir.rglob("sample_*.json")):
        try:
            rec = json.loads(fp.read_text())
            responses[int(rec["sample_id"])] = rec
        except Exception as exc:
            print(f"  WARNING: Could not load {fp}: {exc}")
    return responses


def _evaluate_samples(manifest_subset: list[dict], responses: dict[int, dict],
                       parser: ResponseParser, verbose: bool = False) -> dict:
    """Compute metrics for a subset of samples (one suite × one model)."""
    gt_q1, pred_q1 = [], []
    gt_q1d, pred_q1d = [], []
    gt_q2, pred_q2 = [], []
    gt_q3, pred_q3 = [], []
    gt_q4, pred_q4 = [], []
    gt_q5, pred_q5 = [], []
    gt_q6, pred_q6 = [], []
    task_type_correct = []
    n_total = len(manifest_subset)
    n_response = 0

    for record in manifest_subset:
        sid = record["sample_id"]
        gt = record["gt"]
        if sid not in responses:
            continue
        n_response += 1
        raw = responses[sid].get("raw_response", "")
        parsed = parser.parse(raw)

        gt_task_type = gt.get("task_type")
        pred_task_type = parsed.get("task_type")
        if gt_task_type and pred_task_type:
            task_type_correct.append(int(gt_task_type == pred_task_type))

        gt_q1_val = _safe_list(gt.get("target_pos"))
        pred_q1_val = _safe_list(parsed["q1_object_pos"])
        if gt_q1_val and pred_q1_val:
            gt_q1.append(gt_q1_val); pred_q1.append(pred_q1_val)

        gt_q1d_val = _safe_list(gt.get("dest_pos"))
        pred_q1d_val = _safe_list(parsed["q1_dest_pos"])
        if gt_q1d_val and pred_q1d_val:
            gt_q1d.append(gt_q1d_val); pred_q1d.append(pred_q1d_val)

        gt_q2_val = _safe_list(gt.get("eef_pos"))
        pred_q2_val = _safe_list(parsed["q2_gripper_pos"])
        if gt_q2_val and pred_q2_val:
            gt_q2.append(gt_q2_val); pred_q2.append(pred_q2_val)

        gt_q3_val = gt.get("can_close")
        pred_q3_val = parsed["q3_can_close"]
        if pred_q3_val is not None:
            gt_q3.append(bool(gt_q3_val)); pred_q3.append(bool(pred_q3_val))

        gt_q4_val = _safe_list(gt.get("next_direction"))
        pred_q4_val = _safe_list(parsed["q4_next_dir"])
        if gt_q4_val and pred_q4_val:
            gt_q4.append(gt_q4_val); pred_q4.append(pred_q4_val)

        gt_q5_val = _safe_list(gt.get("gripper_to_target_delta"))
        pred_q5_val = _safe_list(parsed["q5_gripper_to_target"])
        if gt_q5_val and pred_q5_val:
            gt_q5.append(gt_q5_val); pred_q5.append(pred_q5_val)

        gt_q6_val = gt.get("gripper_to_target_relation")
        pred_q6_val = parsed["q6_spatial_relation"]
        if (
            isinstance(gt_q6_val, dict)
            and isinstance(pred_q6_val, dict)
            and all(pred_q6_val.get(ax) is not None for ax in ("x", "y", "z"))
        ):
            gt_q6.append(gt_q6_val); pred_q6.append(pred_q6_val)

    return {
        "n_total": n_total,
        "n_response": n_response,
        "parse_rate": n_response / n_total if n_total > 0 else 0.0,
        "task_type_accuracy": float(np.mean(task_type_correct)) if task_type_correct else None,
        "q1":     {**position_metrics(pred_q1, gt_q1),
                   "parse_rate": len(pred_q1) / n_response if n_response else 0.0},
        "q1_dest":{**position_metrics(pred_q1d, gt_q1d),
                   "parse_rate": len(pred_q1d) / n_response if n_response else 0.0},
        "q2":     {**position_metrics(pred_q2, gt_q2),
                   "parse_rate": len(pred_q2) / n_response if n_response else 0.0},
        "q3":     {**classification_metrics(pred_q3, gt_q3),
                   "parse_rate": len(pred_q3) / n_response if n_response else 0.0},
        "q4":     {**direction_metrics(pred_q4, gt_q4),
                   "parse_rate": len(pred_q4) / n_response if n_response else 0.0},
        "q5":     {**position_metrics(pred_q5, gt_q5),
                   "parse_rate": len(pred_q5) / n_response if n_response else 0.0},
        "q6":     {**spatial_relation_metrics(pred_q6, gt_q6),
                   "parse_rate": len(pred_q6) / n_response if n_response else 0.0},
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "—"

def _p(v):
    return f"{v:.1%}" if v is not None else "—"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Collect run directories
    if args.auto:
        root = Path(args.runs_root)
        run_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and d.name[:8].isdigit())
    elif args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        print("Provide --runs or --auto.")
        sys.exit(1)

    run_dirs = [d for d in run_dirs if d.exists()]
    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    print(f"Processing {len(run_dirs)} run(s): {[d.name for d in run_dirs]}")

    parser = ResponseParser()

    # -----------------------------------------------------------------------
    # Collect data: {model: {suite: metrics}}
    # -----------------------------------------------------------------------
    # aggregate_data[model][suite] = metrics_dict
    aggregate_data: dict[str, dict[str, dict]] = {}

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"  WARNING: No manifest.json in {run_dir}, skipping.")
            continue
        manifest = json.loads(manifest_path.read_text())

        # Group manifest by suite
        suites: dict[str, list] = {}
        for rec in manifest:
            suite = rec.get("suite", "unknown")
            suites.setdefault(suite, []).append(rec)

        # Find model directories
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            model = model_dir.name
            responses = _load_responses_from_run(run_dir, model)
            if not responses:
                continue

            aggregate_data.setdefault(model, {})

            print(f"  {run_dir.name} / {model}: {len(responses)} responses, "
                  f"suites={list(suites.keys())}")

            for suite, suite_manifest in suites.items():
                print(f"    suite={suite} ({len(suite_manifest)} samples) ...", end=" ", flush=True)
                metrics = _evaluate_samples(suite_manifest, responses, parser, args.verbose)
                metrics["run"] = run_dir.name
                aggregate_data[model][suite] = metrics
                print(f"parse={metrics['parse_rate']:.0%}  "
                      f"q1_mae={_f(metrics['q1'].get('mae_overall'))}  "
                      f"q3_acc={_f(metrics['q3'].get('accuracy'))}")

            # Also compute "all" (full manifest)
            print(f"    suite=ALL ({len(manifest)} samples) ...", end=" ", flush=True)
            metrics_all = _evaluate_samples(manifest, responses, parser, args.verbose)
            metrics_all["run"] = run_dir.name
            aggregate_data[model]["all"] = metrics_all
            print(f"parse={metrics_all['parse_rate']:.0%}  "
                  f"q1_mae={_f(metrics_all['q1'].get('mae_overall'))}  "
                  f"q3_acc={_f(metrics_all['q3'].get('accuracy'))}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate_data, indent=2))
    print(f"\nAggregate JSON saved to {out_path}")

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    if not args.report:
        return

    import datetime as dt
    lines: list[str] = []
    lines.append(f"# Aggregate Benchmark Results\n")
    lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Runs: {', '.join(d.name for d in run_dirs)}\n")

    all_suites = sorted({s for m in aggregate_data.values() for s in m if s != "all"}) + ["all"]
    models = sorted(aggregate_data.keys())

    for suite in all_suites:
        lines.append(f"\n## Suite: `{suite}`\n")

        # Summary table
        hdrs = ["Model", "Run", "N", "Parse%",
                "Q1 MAE", "Q1_dest MAE", "Q2 MAE",
                "Q3 Acc", "Q3 F1", "Q4 CosSim",
                "Q5 MAE", "Q6 AccAll", "TaskType Acc"]
        rows = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            rows.append([
                f"`{model}`",
                f"`{m.get('run', '—')}`",
                str(m.get("n_total", "—")),
                _p(m.get("parse_rate")),
                _f(m.get("q1", {}).get("mae_overall")),
                _f(m.get("q1_dest", {}).get("mae_overall")),
                _f(m.get("q2", {}).get("mae_overall")),
                _f(m.get("q3", {}).get("accuracy")),
                _f(m.get("q3", {}).get("f1")),
                _f(m.get("q4", {}).get("mean_cosine_sim")),
                _f(m.get("q5", {}).get("mae_overall")),
                _f(m.get("q6", {}).get("acc_all")),
                _p(m.get("task_type_accuracy")),
            ])
        lines.append(_md_table(hdrs, rows))
        lines.append("")

        # Per-axis MAE detail
        lines.append(f"### Per-axis MAE — Suite `{suite}`\n")
        hdrs2 = ["Model", "Q", "MAE x", "MAE y", "MAE z", "RMSE", "n"]
        rows2 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            for q_key, label in (
                ("q1", "Q1 source"), ("q1_dest", "Q1 dest"),
                ("q2", "Q2 gripper"), ("q5", "Q5 offset"),
            ):
                qm = m.get(q_key, {})
                rows2.append([
                    f"`{model}`", label,
                    _f(qm.get("mae_x")), _f(qm.get("mae_y")), _f(qm.get("mae_z")),
                    _f(qm.get("rmse_overall")), str(qm.get("n", 0)),
                ])
        lines.append(_md_table(hdrs2, rows2))
        lines.append("")

        # Q3 detail
        lines.append(f"### Q3 Detail — Suite `{suite}`\n")
        hdrs3 = ["Model", "Accuracy", "F1", "GT positive%", "Pred positive%", "Parse%"]
        rows3 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            qm = m.get("q3", {})
            rows3.append([
                f"`{model}`",
                _f(qm.get("accuracy")), _f(qm.get("f1")),
                _p(qm.get("positive_rate_gt")), _p(qm.get("positive_rate_pred")),
                _p(qm.get("parse_rate")),
            ])
        lines.append(_md_table(hdrs3, rows3))
        lines.append("")

        # Q6 detail
        lines.append(f"### Q6 Detail — Suite `{suite}`\n")
        hdrs6 = ["Model", "Acc X", "Acc Y", "Acc Z", "Acc All", "F1 X", "F1 Y", "F1 Z", "Parse%"]
        rows6 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            qm = m.get("q6", {})
            rows6.append([
                f"`{model}`",
                _f(qm.get("acc_x")), _f(qm.get("acc_y")), _f(qm.get("acc_z")),
                _f(qm.get("acc_all")),
                _f(qm.get("f1_x")), _f(qm.get("f1_y")), _f(qm.get("f1_z")),
                _p(qm.get("parse_rate")),
            ])
        lines.append(_md_table(hdrs6, rows6))
        lines.append("")

    lines.append("---\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Markdown report saved to {report_path}")


if __name__ == "__main__":
    main()
