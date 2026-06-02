#!/usr/bin/env python3
"""Analyze VLA ablation experiment results.

Reads evaluation logs, compiles success rates into tables, and generates
correlation plots between 3DBENCH spatial reasoning improvements and
VLA task performance.

Usage:
    python scripts/analyze_vla_ablation.py
    python scripts/analyze_vla_ablation.py --results-dir results/vla_ablation
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "vla_ablation"

# ── 3DBENCH baseline vs LoRA improvements ─────────────────────────────
# From benchmark results: per-dimension improvement ratios
BENCH_IMPROVEMENTS = {
    "q1": {"baseline_mae": 0.2136, "lora_mae": 0.0555, "reduction_pct": 74.0},
    "q2": {"baseline_mae": 0.1899, "lora_mae": 0.0330, "reduction_pct": 82.6},
    "q3": {"baseline_mae": 0.0696, "lora_mae": 0.0408, "reduction_pct": 41.4},
    "q4": {"baseline_acc": 0.2100, "lora_acc": 0.3985, "gain_pct": 89.8},
    "q5": {"baseline_mae": 0.0895, "lora_mae": 0.0666, "reduction_pct": 25.6},
    "q6": {"baseline_mae": 26.2,   "lora_mae": 13.1,   "reduction_pct": 50.0},
    "q7": {"baseline_mae": 0.2591, "lora_mae": 0.0415, "reduction_pct": 84.0},
    "q8": {"baseline_grip_acc": 0.468, "lora_grip_acc": 0.840, "gain_pct": 79.5},
}

EXPERIMENT_META = {
    "e0": {"dim": None,  "label": "Vanilla (baseline)"},
    "e1": {"dim": "q1",  "label": "Q1: Object position"},
    "e2": {"dim": "q2",  "label": "Q2: Gripper position"},
    "e3": {"dim": "q3",  "label": "Q3: Offset vector"},
    "e4": {"dim": "q4",  "label": "Q4: Spatial relation"},
    "e5": {"dim": "q5",  "label": "Q5: Pairwise distance"},
    "e6": {"dim": "q6",  "label": "Q6: Orientation"},
    "e7": {"dim": "q7",  "label": "Q7: Gripper state"},
    "e8": {"dim": "q8",  "label": "Q8: 7D action"},
    "e9": {"dim": "all", "label": "All-Q combined"},
}

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_eval_log(log_path: Path) -> dict | None:
    """Parse success rate from an evaluation log file.

    Returns dict with per-task and average success rates, or None if
    the log can't be parsed. Adjust parsing logic based on actual
    VLM4VLA eval output format.
    """
    if not log_path.exists():
        return None

    text = log_path.read_text()

    # Try to find success rate patterns in the log
    # VLM4VLA typically outputs: "Task X: success_rate = Y"
    # or a summary like "Average success rate: Z"
    results = {}

    # Pattern 1: per-task success rates
    task_pattern = re.compile(r"Task\s+(\d+).*?success.*?(\d+\.?\d*)%", re.IGNORECASE)
    for match in task_pattern.finditer(text):
        task_id = int(match.group(1))
        sr = float(match.group(2))
        results[f"task_{task_id}"] = sr

    # Pattern 2: average success rate
    avg_pattern = re.compile(r"(?:average|mean|overall).*?success.*?(\d+\.?\d*)%", re.IGNORECASE)
    avg_match = avg_pattern.search(text)
    if avg_match:
        results["average"] = float(avg_match.group(1))

    # Pattern 3: JSON results block
    json_pattern = re.compile(r'\{[^{}]*"success_rate"[^{}]*\}', re.DOTALL)
    json_match = json_pattern.search(text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "success_rate" in data:
                results["average"] = float(data["success_rate"]) * 100
        except json.JSONDecodeError:
            pass

    return results if results else None


def collect_results(results_dir: Path) -> dict:
    """Collect all evaluation results into a structured dict."""
    all_results = defaultdict(lambda: defaultdict(dict))

    for log_file in sorted(results_dir.glob("*.log")):
        # Parse filename: e0_full_libero_spatial.log
        parts = log_file.stem.split("_", 2)
        if len(parts) < 3:
            continue
        exp_id = parts[0]            # e0
        strategy = parts[1]          # full or head
        suite = "_".join(parts[2:])  # libero_spatial

        parsed = parse_eval_log(log_file)
        if parsed:
            key = f"{exp_id}_{strategy}"
            all_results[key][suite] = parsed

    return dict(all_results)


def generate_summary_table(results: dict, output_path: Path):
    """Generate a markdown summary table of success rates."""
    lines = [
        "# VLA Ablation Results\n",
        "## Success Rate (%) by Experiment and LIBERO Suite\n",
        "| Experiment | Strategy | Dimension | " + " | ".join(SUITES) + " | Average |",
        "|------------|----------|-----------|" + "|".join(["------" for _ in SUITES]) + "|---------|",
    ]

    for exp_id, meta in EXPERIMENT_META.items():
        for strategy in ["full", "head"]:
            key = f"{exp_id}_{strategy}"
            if key not in results:
                # Still generate row with dashes
                suite_vals = " | ".join(["--" for _ in SUITES])
                lines.append(f"| {exp_id} | {strategy} | {meta['label']} | {suite_vals} | -- |")
                continue

            suite_results = results[key]
            vals = []
            for suite in SUITES:
                if suite in suite_results and "average" in suite_results[suite]:
                    vals.append(f"{suite_results[suite]['average']:.1f}")
                else:
                    vals.append("--")

            # Compute overall average
            numeric_vals = [float(v) for v in vals if v != "--"]
            avg = f"{sum(numeric_vals)/len(numeric_vals):.1f}" if numeric_vals else "--"

            suite_str = " | ".join(vals)
            lines.append(f"| {exp_id} | {strategy} | {meta['label']} | {suite_str} | {avg} |")

    # Add 3DBENCH improvement reference
    lines.extend([
        "\n## 3DBENCH Spatial Reasoning Improvements (Reference)\n",
        "| Dimension | Description | Baseline | LoRA | Improvement |",
        "|-----------|-------------|----------|------|-------------|",
    ])
    for dim, info in BENCH_IMPROVEMENTS.items():
        if "reduction_pct" in info:
            lines.append(
                f"| {dim} | MAE | {info.get('baseline_mae', '--')} | "
                f"{info.get('lora_mae', '--')} | {info['reduction_pct']:.1f}% reduction |"
            )
        else:
            lines.append(
                f"| {dim} | Accuracy | {info.get('baseline_acc', info.get('baseline_grip_acc', '--'))} | "
                f"{info.get('lora_acc', info.get('lora_grip_acc', '--'))} | {info['gain_pct']:.1f}% gain |"
            )

    output_path.write_text("\n".join(lines))
    print(f"Summary written to {output_path}")


def generate_correlation_data(results: dict, output_path: Path):
    """Generate CSV for correlation analysis between 3DBENCH improvement and VLA success."""
    lines = ["experiment,dimension,strategy,bench_improvement_pct,vla_avg_success_pct"]

    baseline_full = results.get("e0_full", {})
    baseline_head = results.get("e0_head", {})

    for exp_id, meta in EXPERIMENT_META.items():
        if exp_id == "e0" or meta["dim"] is None:
            continue

        dim = meta["dim"]
        bench_info = BENCH_IMPROVEMENTS.get(dim, {})
        bench_pct = bench_info.get("reduction_pct", bench_info.get("gain_pct", 0))

        for strategy in ["full", "head"]:
            key = f"{exp_id}_{strategy}"
            baseline = baseline_full if strategy == "full" else baseline_head

            if key not in results:
                continue

            suite_results = results[key]
            numeric_vals = []
            for suite in SUITES:
                if suite in suite_results and "average" in suite_results[suite]:
                    numeric_vals.append(suite_results[suite]["average"])

            if numeric_vals:
                vla_avg = sum(numeric_vals) / len(numeric_vals)
                lines.append(f"{exp_id},{dim},{strategy},{bench_pct:.1f},{vla_avg:.1f}")

    output_path.write_text("\n".join(lines))
    print(f"Correlation data written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze VLA ablation results")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run evaluations first with: bash scripts/run_vla_ablation_eval.sh all")
        return

    print("Collecting results...")
    results = collect_results(results_dir)

    if not results:
        print("No evaluation logs found yet.")
        print("Expected log files like: e0_full_libero_spatial.log")

        # Still generate empty template
        generate_summary_table({}, results_dir / "summary.md")
        return

    print(f"Found results for {len(results)} experiments.\n")

    # Generate outputs
    generate_summary_table(results, results_dir / "summary.md")
    generate_correlation_data(results, results_dir / "correlation.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
