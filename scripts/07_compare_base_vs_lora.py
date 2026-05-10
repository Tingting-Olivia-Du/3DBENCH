#!/usr/bin/env python3
"""Compare Qwen2.5-VL-3B base vs per-dim LoRA on CALVIN or LIBERO.

For each QA dimension (Q1, Q1_dest, Q2, Q3, Q4, Q5, Q6) the script:
  - reads the base run    from <base_root>/<base_slug>/<latest>/...
  - reads the LoRA run    from <lora_root>/<lora_slug_template.format(dim=...)>/<latest>/...
  - parses each response with the existing ResponseParser
  - computes the dimension's primary metric using src/bench/metrics.py
  - reports base, LoRA, and the LoRA-vs-base delta

Per-dim primary metric (lower-is-better unless noted):
  Q1, Q1_dest, Q2, Q5  : MAE overall (m), RMSE overall (m)
  Q3                   : accuracy (higher is better), F1
  Q4                   : mean cosine similarity (higher is better)
  Q6                   : per-axis accuracy + all-axes accuracy (higher is better)

Usage:
  # CALVIN (default)
  python scripts/07_compare_base_vs_lora.py --benchmark calvin

  # LIBERO, optionally restricted to one suite
  python scripts/07_compare_base_vs_lora.py --benchmark libero \\
      --suite_filter libero_goal

  # Fully explicit paths still work
  python scripts/07_compare_base_vs_lora.py \\
      --base_root  data/runs-mv-base-libero \\
      --lora_root  data/runs-mv-lora-libero-new \\
      --manifest   data/gt-q6-mv/manifest.json \\
      --suite_filter libero_goal \\
      --out_json   data/results-base-vs-lora-libero-goal.json \\
      --out_md     data/results-base-vs-lora-libero-goal.md

# libero 全 suite(LoRA 跑齐后)
python scripts/07_compare_base_vs_lora.py --benchmark libero

# libero 单 suite
python scripts/07_compare_base_vs_lora.py --benchmark libero --suite_filter libero_goal

# calvin(行为不变)
python scripts/07_compare_base_vs_lora.py --benchmark calvin

      
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.output_parser import ResponseParser
from bench.metrics import (
    classification_metrics,
    direction_metrics,
    position_metrics,
    spatial_relation_metrics,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

# Per-benchmark defaults. Used when the user passes --benchmark and does not
# explicitly override the corresponding flag.
BENCHMARK_PRESETS = {
    "calvin": {
        "base_root": "data/runs-mv-base-calvin",
        "lora_root": "data/runs-mv-lora-calvin",
        "manifest":  "data/gt-q6-mv-calvin/manifest.json",
        "out_json":  "data/results-base-vs-lora-calvin.json",
        "out_md":    "data/results-base-vs-lora-calvin.md",
        "title":     "Qwen2.5-VL-3B base vs LoRA on CALVIN",
    },
    "libero": {
        "base_root": "data/runs-mv-base-libero",
        # Note: current LoRA runs live under runs-mv-lora-libero-new; if you
        # have an older runs-mv-lora-libero/ pass --lora_root explicitly.
        "lora_root": "data/runs-mv-lora-libero-new",
        "manifest":  "data/gt-q6-mv/manifest.json",
        "out_json":  "data/results-base-vs-lora-libero.json",
        "out_md":    "data/results-base-vs-lora-libero.md",
        "title":     "Qwen2.5-VL-3B base vs LoRA on LIBERO",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", choices=sorted(BENCHMARK_PRESETS.keys()), default="calvin",
                   help="Benchmark preset; sets defaults for --base_root / --lora_root / "
                        "--manifest / --out_json / --out_md unless those are passed explicitly.")
    p.add_argument("--base_root",  default=None,
                   help="Root containing <base_slug>/<latest>/<suite>/sample_*.json. "
                        "Defaults from --benchmark.")
    p.add_argument("--lora_root",  default=None,
                   help="Root containing <lora_slug>/<latest>/... . Defaults from --benchmark.")
    p.add_argument("--base_slug",  default="qwen2.5-vl-3b",
                   help="Subdir name of the base model under base_root.")
    p.add_argument("--lora_slug_template", default="qwen2.5-vl-3b-mv-lora-{dim}",
                   help="Subdir name pattern for LoRA-augmented runs.")
    p.add_argument("--manifest", default=None,
                   help="Path to the GT manifest. Defaults from --benchmark.")
    p.add_argument("--suite_filter", default=None,
                   help="If set, only sample_ids whose manifest entry has suite=<this> "
                        "are scored (e.g. libero_goal). Default: all suites in the manifest.")
    p.add_argument("--out_json", default=None,
                   help="Output JSON path. Defaults from --benchmark (auto-suffixed with "
                        "suite when --suite_filter is given).")
    p.add_argument("--out_md",   default=None,
                   help="Output markdown path. Same defaulting rule as --out_json.")
    p.add_argument("--dims", nargs="+", default=["q1", "q2", "q3", "q4", "q5", "q6"])
    args = p.parse_args()

    preset = BENCHMARK_PRESETS[args.benchmark]
    if args.base_root is None: args.base_root = preset["base_root"]
    if args.lora_root is None: args.lora_root = preset["lora_root"]
    if args.manifest  is None: args.manifest  = preset["manifest"]

    # When a suite filter is set, suffix the default output paths so different
    # suites don't overwrite each other. If the user gave explicit paths, leave them.
    suite_suffix = f"-{args.suite_filter}" if args.suite_filter else ""
    if args.out_json is None:
        args.out_json = preset["out_json"].replace(".json", f"{suite_suffix}.json")
    if args.out_md is None:
        args.out_md = preset["out_md"].replace(".md", f"{suite_suffix}.md")
    args._title = preset["title"] + (f" — {args.suite_filter}" if args.suite_filter else "")
    return args


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir(model_dir: Path) -> Path:
    """Resolve the latest run timestamp dir under model_dir."""
    latest = model_dir / "latest"
    if latest.is_symlink() or latest.exists():
        return latest.resolve()
    # Fallback: pick the lexicographically largest YYYYMMDD_HHMMSS subdir.
    runs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run subdirs under {model_dir}")
    return sorted(runs)[-1]


def load_responses_by_sample(run_dir: Path) -> dict[str, dict]:
    """Load all sample_*.json under run_dir/<suite>/, keyed by image_path.

    NOTE: We key on image_path rather than sample_id because the two run
    pipelines disagree on what sample_id means — base runs use a global id
    that spans all suites, LoRA runs use a per-suite local id. image_path is
    identical across both for the same underlying frame.
    """
    out: dict[str, dict] = {}
    for sj in run_dir.rglob("sample_*.json"):
        rec = json.loads(sj.read_text())
        out[rec["image_path"]] = rec
    return out


def load_manifest_by_sample(manifest_path: Path, suite_filter: str | None = None) -> dict[str, dict]:
    """Load the GT manifest, keyed by image_path. Optionally restrict to one suite."""
    raw = json.loads(manifest_path.read_text())
    if suite_filter:
        raw = [r for r in raw if r.get("suite") == suite_filter]
    return {r["image_path"]: r for r in raw}


# ---------------------------------------------------------------------------
# Per-dimension scoring
# ---------------------------------------------------------------------------

def _score_q1(parser, samples, gt_by_id, key: str) -> dict:
    """key is either 'q1_object_pos' (matched against gt['target_pos'])
    or 'q1_dest_pos' (matched against gt['dest_pos'], skipping nulls)."""
    gt_field = "target_pos" if key == "q1_object_pos" else "dest_pos"
    preds, gts = [], []
    n_total = n_parsed = n_skipped_null_gt = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        gt_val = gt_by_id[sid]["gt"].get(gt_field)
        if gt_val is None:
            n_skipped_null_gt += 1
            continue
        parsed = parser.parse(rec["raw_response"])
        v = parsed[key]
        if v is None:
            continue
        if any(x is None for x in v):
            continue
        preds.append(v)
        gts.append(gt_val)
        n_parsed += 1
    m = position_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["n_skipped_null_gt"] = n_skipped_null_gt
    m["parse_rate"] = (n_parsed / max(1, n_total - n_skipped_null_gt))
    return m


def _score_q2(parser, samples, gt_by_id) -> dict:
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed["q2_gripper_pos"]
        if v is None or any(x is None for x in v):
            continue
        preds.append(v)
        gts.append(gt_by_id[sid]["gt"]["eef_pos"])
        n_parsed += 1
    m = position_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q3(parser, samples, gt_by_id) -> dict:
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed["q3_can_close"]
        if v is None:
            continue
        preds.append(bool(v))
        gts.append(bool(gt_by_id[sid]["gt"]["can_close"]))
        n_parsed += 1
    m = classification_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q4(parser, samples, gt_by_id) -> dict:
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed["q4_next_dir"]
        if v is None or any(x is None for x in v):
            continue
        preds.append(v)
        gts.append(gt_by_id[sid]["gt"]["next_direction"])
        n_parsed += 1
    m = direction_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q5(parser, samples, gt_by_id) -> dict:
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed["q5_gripper_to_target"]
        if v is None or any(x is None for x in v):
            continue
        preds.append(v)
        gts.append(gt_by_id[sid]["gt"]["gripper_to_target_delta"])
        n_parsed += 1
    m = position_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q6(parser, samples, gt_by_id) -> dict:
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed["q6_spatial_relation"]
        if v is None or any(v.get(ax) is None for ax in ("x", "y", "z")):
            continue
        preds.append(v)
        gts.append(gt_by_id[sid]["gt"]["gripper_to_target_relation"])
        n_parsed += 1
    m = spatial_relation_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


# ---------------------------------------------------------------------------
# Per-dim primary metric extraction (for the headline table)
# ---------------------------------------------------------------------------

# (metric_name, lower_is_better)
PRIMARY = {
    "q1":      ("mae_overall",      True),
    "q1_dest": ("mae_overall",      True),
    "q2":      ("mae_overall",      True),
    "q3":      ("accuracy",         False),
    "q4":      ("mean_cosine_sim",  False),
    "q5":      ("mae_overall",      True),
    "q6":      ("acc_all",          False),
}

SECONDARY = {
    "q1":      "rmse_overall",
    "q1_dest": "rmse_overall",
    "q2":      "rmse_overall",
    "q3":      "f1",
    "q4":      "median_cosine_sim",
    "q5":      "rmse_overall",
    "q6":      "f1_x",   # show one of the per-axis F1s; full breakdown lives in JSON
}


def _fmt_metric(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _delta_str(base, lora, lower_better: bool) -> str:
    if base is None or lora is None:
        return "—"
    d = lora - base
    if abs(d) < 1e-9:
        return "  0.0000"
    arrow = "↓" if d < 0 else "↑"
    is_improvement = (d < 0) if lower_better else (d > 0)
    marker = "✓" if is_improvement else "✗"
    return f"{arrow}{abs(d):.4f} {marker}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_root = Path(args.base_root)
    lora_root = Path(args.lora_root)
    manifest_path = Path(args.manifest)

    print(f"manifest:  {manifest_path}")
    print(f"base_root: {base_root}/{args.base_slug}")
    print(f"lora_root: {lora_root}/{args.lora_slug_template.format(dim='<dim>')}")

    gt_by_id = load_manifest_by_sample(manifest_path)
    print(f"GT samples: {len(gt_by_id)}")

    base_run = _resolve_run_dir(base_root / args.base_slug)
    print(f"base run:  {base_run}")
    base_samples = load_responses_by_sample(base_run)
    print(f"  base responses: {len(base_samples)}")

    parser = ResponseParser()
    scorers = {
        "q1":      lambda s: _score_q1(parser, s, gt_by_id, "q1_object_pos"),
        "q1_dest": lambda s: _score_q1(parser, s, gt_by_id, "q1_dest_pos"),
        "q2":      lambda s: _score_q2(parser, s, gt_by_id),
        "q3":      lambda s: _score_q3(parser, s, gt_by_id),
        "q4":      lambda s: _score_q4(parser, s, gt_by_id),
        "q5":      lambda s: _score_q5(parser, s, gt_by_id),
        "q6":      lambda s: _score_q6(parser, s, gt_by_id),
    }

    # Score base on every dim
    base_results: dict[str, dict] = {}
    for dim_name, fn in scorers.items():
        base_results[dim_name] = fn(base_samples)

    # Score each LoRA on its target dim (and Q1 LoRA additionally on Q1_dest,
    # since Q1 is folded with Q1_dest in the per-dim prompt).
    lora_results: dict[str, dict] = {}
    for dim in args.dims:
        slug = args.lora_slug_template.format(dim=dim)
        lora_dir = lora_root / slug
        if not lora_dir.exists():
            print(f"  ⚠ skipping {dim}: {lora_dir} not found")
            continue
        try:
            run = _resolve_run_dir(lora_dir)
        except FileNotFoundError:
            print(f"  ⚠ skipping {dim}: no run timestamp under {lora_dir}")
            continue
        print(f"LoRA {dim} run: {run}")
        samples = load_responses_by_sample(run)
        print(f"  responses: {len(samples)}")

        # The per-dim LoRA only emits its own dim, so we score only that dim
        # (plus q1_dest for the q1 LoRA).
        scored_dims = [dim] + (["q1_dest"] if dim == "q1" else [])
        for sd in scored_dims:
            lora_results[sd] = scorers[sd](samples)

    # ---- Build the headline table ---------------------------------------
    rows = []
    headline_dims = ["q1", "q1_dest", "q2", "q3", "q4", "q5", "q6"]
    for d in headline_dims:
        b = base_results.get(d, {})
        l = lora_results.get(d, {})
        primary_key, lower_better = PRIMARY[d]
        sec_key = SECONDARY[d]
        rows.append({
            "dim": d,
            "base_n_parsed":  b.get("n_parsed"),
            "lora_n_parsed":  l.get("n_parsed") if l else None,
            "base_parse_rate": b.get("parse_rate"),
            "lora_parse_rate": l.get("parse_rate") if l else None,
            "primary_metric": primary_key,
            "lower_is_better": lower_better,
            "base_primary":  b.get(primary_key),
            "lora_primary":  l.get(primary_key) if l else None,
            "secondary_metric": sec_key,
            "base_secondary": b.get(sec_key),
            "lora_secondary": l.get(sec_key) if l else None,
        })

    # ---- Render markdown -------------------------------------------------
    md_lines = []
    md_lines.append("# Qwen2.5-VL-3B base vs LoRA on CALVIN")
    md_lines.append("")
    md_lines.append(f"- GT manifest: `{manifest_path}` ({len(gt_by_id)} samples)")
    md_lines.append(f"- Base responses: `{base_root}/{args.base_slug}`")
    md_lines.append(f"- LoRA responses: `{lora_root}/qwen2.5-vl-3b-mv-lora-<dim>`")
    md_lines.append("")
    md_lines.append("## Headline metrics (LoRA only answers its own dim; q1 LoRA also covers q1_dest)")
    md_lines.append("")
    md_lines.append("| Dim | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |")
    md_lines.append("|-----|--------|------|------|------------------|---------------------------|")
    for r in rows:
        delta = _delta_str(r["base_primary"], r["lora_primary"], r["lower_is_better"])
        improvement_dir = "↓" if r["lower_is_better"] else "↑"
        pr_b = _fmt_metric(r["base_parse_rate"])
        pr_l = _fmt_metric(r["lora_parse_rate"])
        md_lines.append(
            f"| **{r['dim'].upper()}** "
            f"| {r['primary_metric']} ({improvement_dir} better) "
            f"| {_fmt_metric(r['base_primary'])} "
            f"| {_fmt_metric(r['lora_primary'])} "
            f"| {delta} "
            f"| {pr_b} / {pr_l} |"
        )

    md_lines.append("")
    md_lines.append("### Secondary metrics")
    md_lines.append("")
    md_lines.append("| Dim | Metric | Base | LoRA |")
    md_lines.append("|-----|--------|------|------|")
    for r in rows:
        md_lines.append(
            f"| **{r['dim'].upper()}** "
            f"| {r['secondary_metric']} "
            f"| {_fmt_metric(r['base_secondary'])} "
            f"| {_fmt_metric(r['lora_secondary'])} |"
        )

    md_lines.append("")
    md_lines.append("### Notes")
    md_lines.append("- ✓ marks an improvement (LoRA better than base on the primary metric).")
    md_lines.append("- ✗ marks a regression.")
    md_lines.append("- Q3 is binary (yes/no can_close); accuracy / F1 are higher-better.")
    md_lines.append("- Q4 is unit-vector cosine similarity (range [-1, 1]); higher is better.")
    md_lines.append("- Q1/Q1_dest/Q2/Q5 are 3D positions/offsets in meters; MAE / RMSE are lower-better.")
    md_lines.append("- Q6 is per-axis categorical; `acc_all` is the strict-all-3-axes accuracy.")
    md_lines.append("- Parse rate = fraction of samples where the model emitted a parseable value for the dim.")
    md_lines.append("")
    md_lines.append("## Full per-dim results (incl. per-axis breakdown)")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps({
        "base":  base_results,
        "lora":  lora_results,
    }, indent=2))
    md_lines.append("```")

    out_md_path = Path(args.out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md_lines))
    print(f"\nWrote markdown report → {out_md_path}")

    # ---- Save full JSON --------------------------------------------------
    out_json_path = Path(args.out_json)
    out_json_path.write_text(json.dumps({
        "manifest": str(manifest_path),
        "n_gt": len(gt_by_id),
        "base": base_results,
        "lora": lora_results,
        "headline": rows,
    }, indent=2))
    print(f"Wrote JSON         → {out_json_path}")

    # ---- Console summary -------------------------------------------------
    print("\n=== Headline ===")
    for r in rows:
        delta = _delta_str(r["base_primary"], r["lora_primary"], r["lower_is_better"])
        print(f"  {r['dim']:8s} {r['primary_metric']:20s} "
              f"base={_fmt_metric(r['base_primary'])}  "
              f"lora={_fmt_metric(r['lora_primary'])}  "
              f"Δ={delta}  "
              f"parse={_fmt_metric(r['base_parse_rate'])}/{_fmt_metric(r['lora_parse_rate'])}")


if __name__ == "__main__":
    main()
