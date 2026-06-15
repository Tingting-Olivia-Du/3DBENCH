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
    angular_metrics,
    position_metrics,
    scalar_metrics,
    spatial_relation_metrics,
)
from bench.output_parser import ResponseParser
from bench.gt_extractor import task_named_objects


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


def _compute_q8_metrics(gt_q8: list[list[float]], pred_q8: list[list[float]], n_response: int) -> dict:
    """Compute per-component and aggregate metrics for the 7D next-action prediction.

    Each element is a 7-vector: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """
    n = len(gt_q8)
    if n == 0:
        return {"n": 0, "parse_rate": 0.0}

    gt = np.array(gt_q8)
    pr = np.array(pred_q8)
    ae = np.abs(pr - gt)

    comp_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    result: dict = {"n": n, "parse_rate": n / n_response if n_response > 0 else 0.0}
    for i, name in enumerate(comp_names):
        result[f"mae_{name}"] = float(np.mean(ae[:, i]))
    result["mae_translation"] = float(np.mean(ae[:, :3]))
    result["mae_rotation"] = float(np.mean(ae[:, 3:6]))
    gt_sign = gt[:, 6] > 0
    pr_sign = pr[:, 6] > 0
    result["gripper_accuracy"] = float(np.mean(gt_sign == pr_sign))
    return result


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
    """Compute metrics for a subset of samples (one suite × one model).

    Question numbering (new):
      Q1  – per-object positions           (position_metrics, over all named objects)
      Q2  – gripper position               (position_metrics)
      Q3  – gripper-to-target offset       (position_metrics)
      Q4  – spatial relation               (spatial_relation_metrics)
      Q5  – pairwise distance              (scalar_metrics)
      Q6  – EE orientation                 (angular_metrics)
      Q7  – gripper openness               (scalar_metrics)
      Q8  – 7D next action                 (custom)
    """
    gt_q1, pred_q1 = [], []     # per-object positions (one pair per named object)
    gt_q2, pred_q2 = [], []
    gt_q3, pred_q3 = [], []     # gripper-to-target offset vector
    gt_q4, pred_q4 = [], []     # spatial relation dicts
    gt_q5, pred_q5 = [], []     # pairwise distance (scalar)
    gt_q6e, pred_q6e = [], []   # EE orientation euler deg
    gt_q6q, pred_q6q = [], []   # EE orientation quaternion wxyz
    gt_q7, pred_q7 = [], []     # gripper openness (scalar)
    gt_q8, pred_q8 = [], []     # 7D next action
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

        # Q1: per-object positions (one (gt, pred) pair per named object)
        pred_objs = parsed.get("q1_object_positions") or {}
        for obj in task_named_objects(gt):
            obj_gt_pos = _safe_list(obj.get("pos"))
            # Match by exact MuJoCo body name (the prompt lists these verbatim).
            obj_pred_pos = _safe_list(pred_objs.get(obj["name"]))
            if obj_gt_pos and obj_pred_pos:
                gt_q1.append(obj_gt_pos); pred_q1.append(obj_pred_pos)

        # Q2: gripper position
        gt_q2_val = _safe_list(gt.get("eef_pos"))
        pred_q2_val = _safe_list(parsed["q2_gripper_pos"])
        if gt_q2_val and pred_q2_val:
            gt_q2.append(gt_q2_val); pred_q2.append(pred_q2_val)

        # Q3: gripper-to-target offset vector
        gt_q3_val = _safe_list(gt.get("gripper_to_target_delta"))
        pred_q3_val = _safe_list(parsed.get("q3_gripper_to_target"))
        if gt_q3_val and pred_q3_val:
            gt_q3.append(gt_q3_val); pred_q3.append(pred_q3_val)

        # Q4: spatial relation classification
        gt_q4_val = gt.get("gripper_to_target_relation")
        pred_q4_val = parsed.get("q4_spatial_relation")
        if (
            isinstance(gt_q4_val, dict)
            and isinstance(pred_q4_val, dict)
            and all(pred_q4_val.get(ax) is not None for ax in ("x", "y", "z"))
        ):
            gt_q4.append(gt_q4_val); pred_q4.append(pred_q4_val)

        # Q5: pairwise distance (scalar)
        gt_q5_dict = gt.get("pairwise_distance")
        pred_q5_dict = parsed.get("q5_pairwise_distance")
        if isinstance(gt_q5_dict, dict) and isinstance(pred_q5_dict, dict):
            gt_d = gt_q5_dict.get("distance_m")
            pred_d = pred_q5_dict.get("distance_m")
            if gt_d is not None and pred_d is not None:
                gt_q5.append(float(gt_d))
                pred_q5.append(float(pred_d))

        # Q6: EE orientation
        gt_q6e_val = _safe_list(gt.get("eef_orientation_euler_deg"))
        pred_q6e_val = _safe_list(parsed.get("q6_eef_orientation_euler"))
        if gt_q6e_val and pred_q6e_val:
            gt_q6e.append(gt_q6e_val); pred_q6e.append(pred_q6e_val)
            gt_q6q_val = gt.get("eef_orientation_quat")
            if gt_q6q_val:
                gt_q6q.append(gt_q6q_val)
                from bench.gt_extractor import euler_deg_to_quat_wxyz
                pred_q6q.append(euler_deg_to_quat_wxyz(pred_q6e_val))

        # Q7: gripper openness
        gt_q7_val = gt.get("gripper_openness")
        pred_q7_val = parsed.get("q7_gripper_openness")
        if gt_q7_val is not None and pred_q7_val is not None:
            gt_q7.append(float(gt_q7_val))
            pred_q7.append(float(pred_q7_val))

        # Q8: 7D next action
        gt_q8_val = gt.get("demo_action")
        pred_q8_val = parsed.get("q8_next_action")
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

    return {
        "n_total": n_total,
        "n_response": n_response,
        "parse_rate": n_response / n_total if n_total > 0 else 0.0,
        "task_type_accuracy": float(np.mean(task_type_correct)) if task_type_correct else None,
        "q1":     {**position_metrics(pred_q1, gt_q1),
                   "parse_rate": len(pred_q1) / n_response if n_response else 0.0},
        "q2":     {**position_metrics(pred_q2, gt_q2),
                   "parse_rate": len(pred_q2) / n_response if n_response else 0.0},
        "q3":     {**position_metrics(pred_q3, gt_q3),
                   "parse_rate": len(pred_q3) / n_response if n_response else 0.0},
        "q4":     {**spatial_relation_metrics(pred_q4, gt_q4),
                   "parse_rate": len(pred_q4) / n_response if n_response else 0.0},
        "q5":     {**scalar_metrics(pred_q5, gt_q5),
                   "parse_rate": len(pred_q5) / n_response if n_response else 0.0},
        "q6":     {**angular_metrics(pred_q6e, gt_q6e,
                                     pred_q6q if pred_q6q else None,
                                     gt_q6q if gt_q6q else None),
                   "parse_rate": len(pred_q6e) / n_response if n_response else 0.0},
        "q7":     {**scalar_metrics(pred_q7, gt_q7),
                   "parse_rate": len(pred_q7) / n_response if n_response else 0.0},
        "q8":     _compute_q8_metrics(gt_q8, pred_q8, n_response),
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
                      f"q3_mae={_f(metrics['q3'].get('mae_overall'))}")

            # Also compute "all" (full manifest)
            print(f"    suite=ALL ({len(manifest)} samples) ...", end=" ", flush=True)
            metrics_all = _evaluate_samples(manifest, responses, parser, args.verbose)
            metrics_all["run"] = run_dir.name
            aggregate_data[model]["all"] = metrics_all
            print(f"parse={metrics_all['parse_rate']:.0%}  "
                  f"q1_mae={_f(metrics_all['q1'].get('mae_overall'))}  "
                  f"q3_mae={_f(metrics_all['q3'].get('mae_overall'))}")

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
                "Q1 MAE", "Q2 MAE",
                "Q3 MAE", "Q4 AccAll",
                "Q5 MAE", "Q6 MAE (°)",
                "Q7 MAE", "Q8 TransMAE", "Q8 GripAcc",
                "TaskType Acc"]
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
                _f(m.get("q2", {}).get("mae_overall")),
                _f(m.get("q3", {}).get("mae_overall")),
                _f(m.get("q4", {}).get("acc_all")),
                _f(m.get("q5", {}).get("mae")),
                _f(m.get("q6", {}).get("mae_overall_deg")),
                _f(m.get("q7", {}).get("mae")),
                _f(m.get("q8", {}).get("mae_translation")),
                _p(m.get("q8", {}).get("gripper_accuracy")),
                _p(m.get("task_type_accuracy")),
            ])
        lines.append(_md_table(hdrs, rows))
        lines.append("")

        # Per-axis MAE detail (position-type questions)
        lines.append(f"### Per-axis MAE — Suite `{suite}`\n")
        hdrs2 = ["Model", "Q", "MAE x", "MAE y", "MAE z", "RMSE", "n"]
        rows2 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            for q_key, label in (
                ("q1", "Q1 object"),
                ("q2", "Q2 gripper"), ("q3", "Q3 offset"),
            ):
                qm = m.get(q_key, {})
                rows2.append([
                    f"`{model}`", label,
                    _f(qm.get("mae_x")), _f(qm.get("mae_y")), _f(qm.get("mae_z")),
                    _f(qm.get("rmse_overall")), str(qm.get("n", 0)),
                ])
        lines.append(_md_table(hdrs2, rows2))
        lines.append("")

        # Q4 detail (spatial relation)
        lines.append(f"### Q4 Spatial Relation Detail — Suite `{suite}`\n")
        hdrs4 = ["Model", "Acc X", "Acc Y", "Acc Z", "Acc All", "F1 X", "F1 Y", "F1 Z", "Parse%"]
        rows4 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            qm = m.get("q4", {})
            rows4.append([
                f"`{model}`",
                _f(qm.get("acc_x")), _f(qm.get("acc_y")), _f(qm.get("acc_z")),
                _f(qm.get("acc_all")),
                _f(qm.get("f1_x")), _f(qm.get("f1_y")), _f(qm.get("f1_z")),
                _p(qm.get("parse_rate")),
            ])
        lines.append(_md_table(hdrs4, rows4))
        lines.append("")

        # Q5/Q7 scalar detail
        lines.append(f"### Q5/Q7 Scalar Detail — Suite `{suite}`\n")
        hdrs_sc = ["Model", "Q", "MAE", "RMSE", "n"]
        rows_sc = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            for q_key, label in (("q5", "Q5 Pairwise Dist"), ("q7", "Q7 Openness")):
                qm = m.get(q_key, {})
                rows_sc.append([
                    f"`{model}`", label,
                    _f(qm.get("mae")), _f(qm.get("rmse")),
                    str(qm.get("n", 0)),
                ])
        lines.append(_md_table(hdrs_sc, rows_sc))
        lines.append("")

        # Q6 orientation detail
        lines.append(f"### Q6 EE Orientation Detail — Suite `{suite}`\n")
        hdrs6 = ["Model", "MAE Roll", "MAE Pitch", "MAE Yaw", "MAE Overall",
                  "Geodesic Mean", "Geodesic Median", "n"]
        rows6 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            qm = m.get("q6", {})
            rows6.append([
                f"`{model}`",
                _f(qm.get("mae_roll_deg")), _f(qm.get("mae_pitch_deg")),
                _f(qm.get("mae_yaw_deg")), _f(qm.get("mae_overall_deg")),
                _f(qm.get("geodesic_mean_deg")), _f(qm.get("geodesic_median_deg")),
                str(qm.get("n", 0)),
            ])
        lines.append(_md_table(hdrs6, rows6))
        lines.append("")

        # Q8 7D action detail
        lines.append(f"### Q8 7D Action Detail — Suite `{suite}`\n")
        hdrs8 = ["Model", "MAE dx", "MAE dy", "MAE dz",
                  "Trans MAE", "Rot MAE", "Grip Acc", "n"]
        rows8 = []
        for model in models:
            m = aggregate_data[model].get(suite)
            if m is None:
                continue
            qm = m.get("q8", {})
            rows8.append([
                f"`{model}`",
                _f(qm.get("mae_dx")), _f(qm.get("mae_dy")), _f(qm.get("mae_dz")),
                _f(qm.get("mae_translation")), _f(qm.get("mae_rotation")),
                _p(qm.get("gripper_accuracy")),
                str(qm.get("n", 0)),
            ])
        lines.append(_md_table(hdrs8, rows8))
        lines.append("")

    lines.append("---\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Markdown report saved to {report_path}")


if __name__ == "__main__":
    main()
