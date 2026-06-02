#!/usr/bin/env python3
"""Compare Qwen2.5-VL-3B base vs per-dim LoRA on CALVIN or LIBERO.

For each QA dimension (Q1, Q1_dest, Q2, Q3, Q4, Q5, Q6, Q7, Q8) the script:
  - reads the base run    from <base_root>/<base_slug>/<latest>/...
  - reads the LoRA run    from <lora_root>/<lora_slug_template.format(dim=...)>/<latest>/...
  - parses each response with the existing ResponseParser
  - computes the dimension's primary metric using src/bench/metrics.py
  - reports base, LoRA, and the LoRA-vs-base delta

Per-dim primary metric (lower-is-better unless noted):
  Q1, Q1_dest, Q2, Q3   : MAE overall (m), RMSE overall (m)
  Q4                    : per-axis accuracy + all-axes accuracy (higher is better)
  Q5                    : MAE (m) — pairwise distance (lower is better)
  Q6                    : MAE overall (°) — EE orientation (lower is better)
  Q7                    : MAE — gripper openness (lower is better)
  Q8                    : MAE translation — 7D next action (lower is better)

Usage:
  # CALVIN (default)
  python scripts/07_compare_base_vs_lora.py --benchmark calvin

  # LIBERO, optionally restricted to one suite
  python scripts/07_compare_base_vs_lora.py --benchmark libero \\
      --suite_filter libero_goal

  # Fully explicit paths still work

  python scripts/07_compare_base_vs_lora.py \
      --benchmark libero \
      --base_root  rollout/baseline/libero-test-9-qwen-0515-baseline-all-models-all-suite \
      --lora_root  rollout/lora-perdim-ckpt4000 \
      --manifest   data/gt-demo-libero-all-suite-train-fix/manifest.json \
      --out_md     results/0516/results-base-vs-lora-libero-detailed-0516.md

/workspace/tingting/3DBENCH/
      
  python scripts/07_compare_base_vs_lora.py \
      --benchmark calvin \
      --base_root  data/runs-mv-base-calvin-0509-fixed \
      --lora_root  data/runs-mv-lora-calvin-0509-fixed \
      --manifest   data/gt-q6-mv-calvin/manifest.json \
      --out_json   data/results-base-vs-lora-calvin-0509.json \
      --out_md     data/results-base-vs-lora-calvin-0509.md



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
    _circular_distance_deg,
    angular_metrics,
    position_metrics,
    scalar_metrics,
    spatial_relation_metrics,
)
import numpy as np


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
    p.add_argument("--dims", nargs="+", default=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"])
    p.add_argument("--out_csv", default=None,
                   help="Write per-sample detail CSV (base vs LoRA side-by-side). "
                        "If not set, no CSV is written.")
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
    """Q3: gripper-to-target offset vector."""
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed.get("q3_gripper_to_target")
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


def _score_q4(parser, samples, gt_by_id) -> dict:
    """Q4: spatial relation classification."""
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        v = parsed.get("q4_spatial_relation")
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


def _score_q5(parser, samples, gt_by_id) -> dict:
    """Q5: pairwise distance (scalar)."""
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        pred_dict = parsed.get("q5_pairwise_distance")
        gt_dict = gt_by_id[sid]["gt"].get("pairwise_distance")
        if not isinstance(pred_dict, dict) or not isinstance(gt_dict, dict):
            continue
        pred_d = pred_dict.get("distance_m")
        gt_d = gt_dict.get("distance_m")
        if pred_d is None or gt_d is None:
            continue
        preds.append(float(pred_d))
        gts.append(float(gt_d))
        n_parsed += 1
    m = scalar_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q6(parser, samples, gt_by_id) -> dict:
    """Q6: EE orientation (euler degrees)."""
    def _safe_list(val):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return [float(v) for v in val]
        return None

    pred_euler, gt_euler = [], []
    pred_quat, gt_quat = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        pred_e = _safe_list(parsed.get("q6_eef_orientation_euler"))
        gt_e = _safe_list(gt_by_id[sid]["gt"].get("eef_orientation_euler_deg"))
        if pred_e is None or gt_e is None:
            continue
        pred_euler.append(pred_e)
        gt_euler.append(gt_e)
        n_parsed += 1
        gt_q = gt_by_id[sid]["gt"].get("eef_orientation_quat")
        if gt_q:
            gt_quat.append(gt_q)
            from bench.gt_extractor import euler_deg_to_quat_wxyz
            pred_quat.append(euler_deg_to_quat_wxyz(pred_e))
    m = angular_metrics(
        pred_euler, gt_euler,
        pred_quat if pred_quat else None,
        gt_quat if gt_quat else None,
    )
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q7(parser, samples, gt_by_id) -> dict:
    """Q7: gripper openness (scalar)."""
    preds, gts = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        pred_v = parsed.get("q7_gripper_openness")
        gt_v = gt_by_id[sid]["gt"].get("gripper_openness")
        if pred_v is None or gt_v is None:
            continue
        preds.append(float(pred_v))
        gts.append(float(gt_v))
        n_parsed += 1
    m = scalar_metrics(preds, gts)
    m["n_total"] = n_total
    m["n_parsed"] = n_parsed
    m["parse_rate"] = n_parsed / max(1, n_total)
    return m


def _score_q8(parser, samples, gt_by_id) -> dict:
    """Q8: 7D next action [dx,dy,dz,droll,dpitch,dyaw,gripper]."""
    gt_q8, pred_q8 = [], []
    n_total = n_parsed = 0
    for sid, rec in samples.items():
        if sid not in gt_by_id:
            continue
        n_total += 1
        parsed = parser.parse(rec["raw_response"])
        pred_val = parsed.get("q8_next_action")
        gt_val = gt_by_id[sid]["gt"].get("demo_action")
        pred_list = None
        if isinstance(pred_val, dict):
            tr = pred_val.get("translation")
            ro = pred_val.get("rotation")
            gr = pred_val.get("gripper")
            if (
                isinstance(tr, (list, tuple)) and len(tr) == 3
                and isinstance(ro, (list, tuple)) and len(ro) == 3
                and gr is not None
            ):
                pred_list = [float(v) for v in tr] + [float(v) for v in ro] + [float(gr)]
        if (
            isinstance(gt_val, (list, tuple)) and len(gt_val) == 7
            and pred_list is not None
        ):
            gt_q8.append([float(v) for v in gt_val])
            pred_q8.append(pred_list)
            n_parsed += 1

    n = len(gt_q8)
    if n == 0:
        return {"n": 0, "n_total": n_total, "n_parsed": 0, "parse_rate": 0.0}
    gt_arr = np.array(gt_q8)
    pr_arr = np.array(pred_q8)
    ae = np.abs(pr_arr - gt_arr)
    comp_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    m: dict = {"n": n, "n_total": n_total, "n_parsed": n_parsed,
               "parse_rate": n_parsed / max(1, n_total)}
    for i, name in enumerate(comp_names):
        m[f"mae_{name}"] = float(np.mean(ae[:, i]))
    m["mae_translation"] = float(np.mean(ae[:, :3]))
    m["mae_rotation"] = float(np.mean(ae[:, 3:6]))
    # Cosine similarity for translation (dx, dy, dz)
    gt_trans = gt_arr[:, :3]
    pr_trans = pr_arr[:, :3]
    dot = np.sum(gt_trans * pr_trans, axis=1)
    norm_gt = np.linalg.norm(gt_trans, axis=1)
    norm_pr = np.linalg.norm(pr_trans, axis=1)
    denom = norm_gt * norm_pr
    # Avoid division by zero when either vector is zero
    valid = denom > 1e-12
    if valid.any():
        cos_sim = dot[valid] / denom[valid]
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        m["cos_sim_translation_mean"] = float(np.mean(cos_sim))
        m["cos_sim_translation_median"] = float(np.median(cos_sim))
        m["cos_sim_translation_n_valid"] = int(valid.sum())
    else:
        m["cos_sim_translation_mean"] = None
        m["cos_sim_translation_median"] = None
        m["cos_sim_translation_n_valid"] = 0
    gt_sign = gt_arr[:, 6] > 0
    pr_sign = pr_arr[:, 6] > 0
    m["gripper_accuracy"] = float(np.mean(gt_sign == pr_sign))
    return m


# ---------------------------------------------------------------------------
# Per-sample detail collection
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _safe_list(val):
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [float(v) for v in val]
    return None


def _round_list(v):
    return [round(x, 4) for x in v] if v else None


def collect_per_sample_details(
    parser: ResponseParser,
    gt_by_id: dict[str, dict],
    base_samples: dict[str, dict],
    lora_samples_by_dim: dict[str, dict[str, dict]],
) -> list[dict]:
    """Walk through every GT sample and collect per-sample gt / base / lora predictions and errors.

    ``lora_samples_by_dim`` maps dim name (e.g. "q1", "q8") to the LoRA
    response dict keyed by image_path.  q1_dest reuses the q1 LoRA responses.
    """

    records: list[dict] = []

    for sid, gt_rec in gt_by_id.items():
        gt = gt_rec["gt"]
        base_rec = base_samples.get(sid)
        base_parsed = parser.parse(base_rec["raw_response"]) if base_rec else {}

        rec: dict = {
            "image_path": sid,
            "sample_id": gt_rec.get("sample_id"),
            "task_id": gt_rec.get("task_id"),
            "task_description": gt_rec.get("task_description", ""),
            "suite": gt_rec.get("suite"),
            "target_object": gt.get("target_object_name"),
            "dest_object": gt.get("dest_object_name"),
        }

        # --- helper closures ---
        def _xyz_err(pred, gt_val):
            if pred and gt_val:
                return [round(abs(p - g), 4) for p, g in zip(pred, gt_val)]
            return None

        # --- Q1 source ---
        gt_q1 = _safe_list(gt.get("target_pos"))
        base_q1 = _safe_list(base_parsed.get("q1_object_pos"))
        lora_q1_parsed = None
        lora_q1_samples = lora_samples_by_dim.get("q1", {})
        if sid in lora_q1_samples:
            lora_q1_parsed = parser.parse(lora_q1_samples[sid]["raw_response"])
        lora_q1 = _safe_list(lora_q1_parsed.get("q1_object_pos")) if lora_q1_parsed else None
        rec["q1"] = {
            "gt": _round_list(gt_q1),
            "base_pred": _round_list(base_q1),
            "lora_pred": _round_list(lora_q1),
            "base_err": _xyz_err(base_q1, gt_q1),
            "lora_err": _xyz_err(lora_q1, gt_q1),
        }

        # --- Q1 dest ---
        gt_q1d = _safe_list(gt.get("dest_pos"))
        base_q1d = _safe_list(base_parsed.get("q1_dest_pos"))
        # q1_dest reuses q1 LoRA
        lora_q1d = _safe_list(lora_q1_parsed.get("q1_dest_pos")) if lora_q1_parsed else None
        rec["q1_dest"] = {
            "gt": _round_list(gt_q1d),
            "base_pred": _round_list(base_q1d),
            "lora_pred": _round_list(lora_q1d),
            "base_err": _xyz_err(base_q1d, gt_q1d),
            "lora_err": _xyz_err(lora_q1d, gt_q1d),
        }

        # --- Q2 gripper pos ---
        gt_q2 = _safe_list(gt.get("eef_pos"))
        base_q2 = _safe_list(base_parsed.get("q2_gripper_pos"))
        lora_q2_parsed = None
        lora_q2_samples = lora_samples_by_dim.get("q2", {})
        if sid in lora_q2_samples:
            lora_q2_parsed = parser.parse(lora_q2_samples[sid]["raw_response"])
        lora_q2 = _safe_list(lora_q2_parsed.get("q2_gripper_pos")) if lora_q2_parsed else None
        rec["q2"] = {
            "gt": _round_list(gt_q2),
            "base_pred": _round_list(base_q2),
            "lora_pred": _round_list(lora_q2),
            "base_err": _xyz_err(base_q2, gt_q2),
            "lora_err": _xyz_err(lora_q2, gt_q2),
        }

        # --- Q3 gripper-to-target offset ---
        gt_q3 = _safe_list(gt.get("gripper_to_target_delta"))
        base_q3 = _safe_list(base_parsed.get("q3_gripper_to_target"))
        lora_q3_parsed = None
        lora_q3_samples = lora_samples_by_dim.get("q3", {})
        if sid in lora_q3_samples:
            lora_q3_parsed = parser.parse(lora_q3_samples[sid]["raw_response"])
        lora_q3 = _safe_list(lora_q3_parsed.get("q3_gripper_to_target")) if lora_q3_parsed else None
        rec["q3"] = {
            "gt": _round_list(gt_q3),
            "base_pred": _round_list(base_q3),
            "lora_pred": _round_list(lora_q3),
            "base_err": _xyz_err(base_q3, gt_q3),
            "lora_err": _xyz_err(lora_q3, gt_q3),
        }

        # --- Q4 spatial relation ---
        gt_q4 = gt.get("gripper_to_target_relation")
        base_q4 = base_parsed.get("q4_spatial_relation")
        lora_q4_parsed = None
        lora_q4_samples = lora_samples_by_dim.get("q4", {})
        if sid in lora_q4_samples:
            lora_q4_parsed = parser.parse(lora_q4_samples[sid]["raw_response"])
        lora_q4 = lora_q4_parsed.get("q4_spatial_relation") if lora_q4_parsed else None

        def _q4_correct(pred, gt_val):
            if isinstance(pred, dict) and isinstance(gt_val, dict):
                return {ax: (pred.get(ax) == gt_val.get(ax)) for ax in ("x", "y", "z")}
            return None

        def _q4_fmt(v):
            if isinstance(v, dict):
                return v
            return None

        rec["q4"] = {
            "gt": _q4_fmt(gt_q4),
            "base_pred": _q4_fmt(base_q4),
            "lora_pred": _q4_fmt(lora_q4),
            "base_correct": _q4_correct(base_q4, gt_q4),
            "lora_correct": _q4_correct(lora_q4, gt_q4),
        }

        # --- Q5 pairwise distance ---
        gt_q5_dict = gt.get("pairwise_distance")
        base_q5_dict = base_parsed.get("q5_pairwise_distance")
        lora_q5_parsed = None
        lora_q5_samples = lora_samples_by_dim.get("q5", {})
        if sid in lora_q5_samples:
            lora_q5_parsed = parser.parse(lora_q5_samples[sid]["raw_response"])
        lora_q5_dict = lora_q5_parsed.get("q5_pairwise_distance") if lora_q5_parsed else None

        def _q5_val(d):
            if isinstance(d, dict):
                v = d.get("distance_m")
                return round(float(v), 4) if v is not None else None
            return None

        gt_q5 = _q5_val(gt_q5_dict)
        base_q5 = _q5_val(base_q5_dict)
        lora_q5 = _q5_val(lora_q5_dict)
        rec["q5"] = {
            "gt": gt_q5,
            "base_pred": base_q5,
            "lora_pred": lora_q5,
            "base_err": round(abs(base_q5 - gt_q5), 4) if base_q5 is not None and gt_q5 is not None else None,
            "lora_err": round(abs(lora_q5 - gt_q5), 4) if lora_q5 is not None and gt_q5 is not None else None,
        }

        # --- Q6 EE orientation ---
        gt_q6 = _safe_list(gt.get("eef_orientation_euler_deg"))
        base_q6 = _safe_list(base_parsed.get("q6_eef_orientation_euler"))
        lora_q6_parsed = None
        lora_q6_samples = lora_samples_by_dim.get("q6", {})
        if sid in lora_q6_samples:
            lora_q6_parsed = parser.parse(lora_q6_samples[sid]["raw_response"])
        lora_q6 = _safe_list(lora_q6_parsed.get("q6_eef_orientation_euler")) if lora_q6_parsed else None

        def _euler_err(pred, gt_val):
            if pred and gt_val:
                return [round(_circular_distance_deg(p, g), 2) for p, g in zip(pred, gt_val)]
            return None

        rec["q6"] = {
            "gt": _round_list(gt_q6),
            "base_pred": _round_list(base_q6),
            "lora_pred": _round_list(lora_q6),
            "base_err_deg": _euler_err(base_q6, gt_q6),
            "lora_err_deg": _euler_err(lora_q6, gt_q6),
        }

        # --- Q7 gripper openness ---
        gt_q7 = gt.get("gripper_openness")
        base_q7 = base_parsed.get("q7_gripper_openness")
        lora_q7_parsed = None
        lora_q7_samples = lora_samples_by_dim.get("q7", {})
        if sid in lora_q7_samples:
            lora_q7_parsed = parser.parse(lora_q7_samples[sid]["raw_response"])
        lora_q7 = lora_q7_parsed.get("q7_gripper_openness") if lora_q7_parsed else None
        rec["q7"] = {
            "gt": round(float(gt_q7), 4) if gt_q7 is not None else None,
            "base_pred": round(float(base_q7), 4) if base_q7 is not None else None,
            "lora_pred": round(float(lora_q7), 4) if lora_q7 is not None else None,
            "base_err": round(abs(float(base_q7) - float(gt_q7)), 4) if base_q7 is not None and gt_q7 is not None else None,
            "lora_err": round(abs(float(lora_q7) - float(gt_q7)), 4) if lora_q7 is not None and gt_q7 is not None else None,
        }

        # --- Q8 7D next action ---
        gt_q8_val = gt.get("demo_action")
        base_q8_val = base_parsed.get("q8_next_action")
        lora_q8_parsed = None
        lora_q8_samples = lora_samples_by_dim.get("q8", {})
        if sid in lora_q8_samples:
            lora_q8_parsed = parser.parse(lora_q8_samples[sid]["raw_response"])
        lora_q8_val = lora_q8_parsed.get("q8_next_action") if lora_q8_parsed else None

        def _parse_q8_list(val):
            if isinstance(val, dict):
                tr = val.get("translation")
                ro = val.get("rotation")
                gr = val.get("gripper")
                if (isinstance(tr, (list, tuple)) and len(tr) == 3
                        and isinstance(ro, (list, tuple)) and len(ro) == 3
                        and gr is not None):
                    return [float(v) for v in tr] + [float(v) for v in ro] + [float(gr)]
            return None

        gt_q8 = [float(v) for v in gt_q8_val] if isinstance(gt_q8_val, (list, tuple)) and len(gt_q8_val) == 7 else None
        base_q8 = _parse_q8_list(base_q8_val)
        lora_q8 = _parse_q8_list(lora_q8_val)

        def _q8_detail(pred, gt_val):
            if pred is None or gt_val is None:
                return None, None, None, None
            trans_err = [round(abs(pred[i] - gt_val[i]), 4) for i in range(3)]
            rot_err = [round(abs(pred[i] - gt_val[i]), 4) for i in range(3, 6)]
            grip_err = round(abs(pred[6] - gt_val[6]), 4)
            cos_sim = round(_cosine_similarity(gt_val[:3], pred[:3]), 4)
            return trans_err, rot_err, grip_err, cos_sim

        base_trans_err, base_rot_err, base_grip_err, base_cos = _q8_detail(base_q8, gt_q8)
        lora_trans_err, lora_rot_err, lora_grip_err, lora_cos = _q8_detail(lora_q8, gt_q8)
        rec["q8"] = {
            "gt": _round_list(gt_q8),
            "base_pred": _round_list(base_q8),
            "lora_pred": _round_list(lora_q8),
            "base_trans_err": base_trans_err,
            "base_trans_cos_sim": base_cos,
            "base_rot_err": base_rot_err,
            "base_grip_err": base_grip_err,
            "lora_trans_err": lora_trans_err,
            "lora_trans_cos_sim": lora_cos,
            "lora_rot_err": lora_rot_err,
            "lora_grip_err": lora_grip_err,
        }

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Per-dim primary metric extraction (for the headline table)
# ---------------------------------------------------------------------------

# (metric_name, lower_is_better)
PRIMARY = {
    "q1":      ("mae_overall",      True),
    "q1_dest": ("mae_overall",      True),
    "q2":      ("mae_overall",      True),
    "q3":      ("mae_overall",      True),
    "q4":      ("acc_all",          False),
    "q5":      ("mae",              True),
    "q6":      ("mae_overall_deg",  True),
    "q7":      ("mae",              True),
    "q8":      ("mae_translation",  True),
}

SECONDARY = {
    "q1":      "rmse_overall",
    "q1_dest": "rmse_overall",
    "q2":      "rmse_overall",
    "q3":      "rmse_overall",
    "q4":      "f1_x",   # show one of the per-axis F1s; full breakdown lives in JSON
    "q5":      "rmse",
    "q6":      "geodesic_mean_deg",
    "q7":      "rmse",
    "q8":      "gripper_accuracy",
}

# Human-readable dimension names for table headers
DIM_LABEL = {
    "q1":      "Q1 Source Obj Pos",
    "q1_dest": "Q1d Dest Pos",
    "q2":      "Q2 Gripper Pos",
    "q3":      "Q3 Grip→Target Offset",
    "q4":      "Q4 Spatial Relation",
    "q5":      "Q5 Pairwise Dist",
    "q6":      "Q6 EE Orientation",
    "q7":      "Q7 Gripper Openness",
    "q8":      "Q8 7D Action",
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
        "q7":      lambda s: _score_q7(parser, s, gt_by_id),
        "q8":      lambda s: _score_q8(parser, s, gt_by_id),
    }

    # Score base on every dim
    base_results: dict[str, dict] = {}
    for dim_name, fn in scorers.items():
        base_results[dim_name] = fn(base_samples)

    # Score each LoRA on its target dim (and Q1 LoRA additionally on Q1_dest,
    # since Q1 is folded with Q1_dest in the per-dim prompt).
    lora_results: dict[str, dict] = {}
    lora_samples_by_dim: dict[str, dict[str, dict]] = {}   # dim → {image_path: rec}
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
        lora_samples_by_dim[dim] = samples

        # The per-dim LoRA only emits its own dim, so we score only that dim
        # (plus q1_dest for the q1 LoRA).
        scored_dims = [dim] + (["q1_dest"] if dim == "q1" else [])
        for sd in scored_dims:
            lora_results[sd] = scorers[sd](samples)

    # ---- Build the headline table ---------------------------------------
    rows = []
    headline_dims = ["q1", "q1_dest", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]
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
    md_lines.append("| Dimension | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |")
    md_lines.append("|-----------|--------|------|------|------------------|---------------------------|")
    for r in rows:
        delta = _delta_str(r["base_primary"], r["lora_primary"], r["lower_is_better"])
        improvement_dir = "↓" if r["lower_is_better"] else "↑"
        pr_b = _fmt_metric(r["base_parse_rate"])
        pr_l = _fmt_metric(r["lora_parse_rate"])
        dim_label = DIM_LABEL.get(r["dim"], r["dim"].upper())
        md_lines.append(
            f"| **{dim_label}** "
            f"| {r['primary_metric']} ({improvement_dir} better) "
            f"| {_fmt_metric(r['base_primary'])} "
            f"| {_fmt_metric(r['lora_primary'])} "
            f"| {delta} "
            f"| {pr_b} / {pr_l} |"
        )

    md_lines.append("")
    md_lines.append("### Secondary metrics")
    md_lines.append("")
    md_lines.append("| Dimension | Metric | Base | LoRA |")
    md_lines.append("|-----------|--------|------|------|")
    for r in rows:
        dim_label = DIM_LABEL.get(r["dim"], r["dim"].upper())
        md_lines.append(
            f"| **{dim_label}** "
            f"| {r['secondary_metric']} "
            f"| {_fmt_metric(r['base_secondary'])} "
            f"| {_fmt_metric(r['lora_secondary'])} |"
        )
        # Extra row for Q8: cosine similarity of translation
        if r["dim"] == "q8":
            b8 = base_results.get("q8", {})
            l8 = lora_results.get("q8", {})
            md_lines.append(
                f"| **{dim_label}** "
                f"| cos_sim_translation_mean "
                f"| {_fmt_metric(b8.get('cos_sim_translation_mean'))} "
                f"| {_fmt_metric(l8.get('cos_sim_translation_mean'))} |"
            )

    md_lines.append("")
    md_lines.append("### Notes")
    md_lines.append("- ✓ marks an improvement (LoRA better than base on the primary metric).")
    md_lines.append("- ✗ marks a regression.")
    md_lines.append("- Q1/Q1_dest/Q2/Q3 are 3D positions/offsets in meters; MAE / RMSE are lower-better.")
    md_lines.append("- Q4 is per-axis categorical; `acc_all` is the strict-all-3-axes accuracy (higher is better).")
    md_lines.append("- Q5 is pairwise distance in meters; MAE is lower-better.")
    md_lines.append("- Q6 is EE orientation in degrees; MAE is lower-better.")
    md_lines.append("- Q7 is gripper openness [0,1]; MAE is lower-better.")
    md_lines.append("- Q8 is 7D next action; translation MAE is lower-better, gripper accuracy is higher-better.")
    md_lines.append("- Parse rate = fraction of samples where the model emitted a parseable value for the dim.")

    # ---- Collect per-sample details ----------------------------------------
    print("\nCollecting per-sample details ...")
    sample_details = collect_per_sample_details(
        parser, gt_by_id, base_samples, lora_samples_by_dim,
    )
    print(f"  {len(sample_details)} per-sample records collected.")

    md_lines.append("")
    md_lines.append("## Full per-dim results (incl. per-axis breakdown)")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps({
        "base":  base_results,
        "lora":  lora_results,
    }, indent=2))
    md_lines.append("```")

    # ---- Per-sample detail table -------------------------------------------
    md_lines.append("")
    md_lines.append(f"## Per-Sample Detail ({len(sample_details)} samples)")
    md_lines.append("")

    def _xyz_fmt(v):
        if isinstance(v, list) and len(v) == 3:
            return f"`[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]`"
        return "—"

    def _mae_xyz_fmt(v):
        if isinstance(v, list) and len(v) == 3:
            overall = (sum(x**2 for x in v) / 3) ** 0.5
            return f"`[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]` → `{overall:.4f}`"
        return "—"

    def _scalar_fmt(v):
        if v is None:
            return "—"
        return f"`{v:.4f}`"

    def _rel_fmt(v):
        if isinstance(v, dict):
            return f"`{v.get('x','?')}/{v.get('y','?')}/{v.get('z','?')}`"
        return "—"

    def _q4_ok(v):
        if isinstance(v, dict):
            return "`{}/{}/{}`".format(
                "✓" if v.get("x") else "✗",
                "✓" if v.get("y") else "✗",
                "✓" if v.get("z") else "✗",
            )
        return "—"

    def _euler_fmt(v):
        if isinstance(v, list) and len(v) == 3:
            return f"`[{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]`"
        return "—"

    def _action7_fmt(v):
        if isinstance(v, list) and len(v) == 7:
            return f"`[{', '.join(f'{x:.4f}' for x in v)}]`"
        return "—"

    ps_hdrs = [
        "ID", "Task", "Target",
        "Q1 Source Obj gt", "Q1 Source Obj base", "Q1 Source Obj lora",
        "Q1 Source Obj base err", "Q1 Source Obj lora err",
        "Q2 Gripper Pos gt", "Q2 Gripper Pos base", "Q2 Gripper Pos lora",
        "Q2 Gripper Pos base err", "Q2 Gripper Pos lora err",
        "Q3 Grip→Target gt", "Q3 Grip→Target base", "Q3 Grip→Target lora",
        "Q3 Grip→Target base err", "Q3 Grip→Target lora err",
        "Q4 Spatial Rel gt", "Q4 Spatial Rel base", "Q4 Spatial Rel lora",
        "Q4 Spatial Rel base ok", "Q4 Spatial Rel lora ok",
        "Q5 Pairwise Dist gt", "Q5 Pairwise Dist base", "Q5 Pairwise Dist lora",
        "Q5 Pairwise Dist base err", "Q5 Pairwise Dist lora err",
        "Q6 EE Orient gt", "Q6 EE Orient base", "Q6 EE Orient lora",
        "Q6 EE Orient base err°", "Q6 EE Orient lora err°",
        "Q7 Grip Open gt", "Q7 Grip Open base", "Q7 Grip Open lora",
        "Q7 Grip Open base err", "Q7 Grip Open lora err",
        "Q8 Action gt", "Q8 Action base", "Q8 Action lora",
        "Q8 Action base trans err", "Q8 Action base cos",
        "Q8 Action base rot err", "Q8 Action base grip err",
        "Q8 Action lora trans err", "Q8 Action lora cos",
        "Q8 Action lora rot err", "Q8 Action lora grip err",
    ]
    md_lines.append("| " + " | ".join(ps_hdrs) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(ps_hdrs)) + " |")

    for s in sample_details:
        q1 = s.get("q1", {})
        q1d = s.get("q1_dest", {})
        q2 = s.get("q2", {})
        q3 = s.get("q3", {})
        q4 = s.get("q4", {})
        q5 = s.get("q5", {})
        q6 = s.get("q6", {})
        q7 = s.get("q7", {})
        q8 = s.get("q8", {})
        desc = s.get("task_description", "")
        desc_short = desc[:50] + ("…" if len(desc) > 50 else "")
        cells = [
            str(s.get("sample_id", "")),
            desc_short,
            s.get("target_object", "—") or "—",
            # Q1
            _xyz_fmt(q1.get("gt")), _xyz_fmt(q1.get("base_pred")), _xyz_fmt(q1.get("lora_pred")),
            _mae_xyz_fmt(q1.get("base_err")), _mae_xyz_fmt(q1.get("lora_err")),
            # Q2
            _xyz_fmt(q2.get("gt")), _xyz_fmt(q2.get("base_pred")), _xyz_fmt(q2.get("lora_pred")),
            _mae_xyz_fmt(q2.get("base_err")), _mae_xyz_fmt(q2.get("lora_err")),
            # Q3
            _xyz_fmt(q3.get("gt")), _xyz_fmt(q3.get("base_pred")), _xyz_fmt(q3.get("lora_pred")),
            _mae_xyz_fmt(q3.get("base_err")), _mae_xyz_fmt(q3.get("lora_err")),
            # Q4
            _rel_fmt(q4.get("gt")), _rel_fmt(q4.get("base_pred")), _rel_fmt(q4.get("lora_pred")),
            _q4_ok(q4.get("base_correct")), _q4_ok(q4.get("lora_correct")),
            # Q5
            _scalar_fmt(q5.get("gt")), _scalar_fmt(q5.get("base_pred")), _scalar_fmt(q5.get("lora_pred")),
            _scalar_fmt(q5.get("base_err")), _scalar_fmt(q5.get("lora_err")),
            # Q6
            _euler_fmt(q6.get("gt")), _euler_fmt(q6.get("base_pred")), _euler_fmt(q6.get("lora_pred")),
            _euler_fmt(q6.get("base_err_deg")), _euler_fmt(q6.get("lora_err_deg")),
            # Q7
            _scalar_fmt(q7.get("gt")), _scalar_fmt(q7.get("base_pred")), _scalar_fmt(q7.get("lora_pred")),
            _scalar_fmt(q7.get("base_err")), _scalar_fmt(q7.get("lora_err")),
            # Q8
            _action7_fmt(q8.get("gt")), _action7_fmt(q8.get("base_pred")), _action7_fmt(q8.get("lora_pred")),
            _xyz_fmt(q8.get("base_trans_err")), _scalar_fmt(q8.get("base_trans_cos_sim")),
            _xyz_fmt(q8.get("base_rot_err")), _scalar_fmt(q8.get("base_grip_err")),
            _xyz_fmt(q8.get("lora_trans_err")), _scalar_fmt(q8.get("lora_trans_cos_sim")),
            _xyz_fmt(q8.get("lora_rot_err")), _scalar_fmt(q8.get("lora_grip_err")),
        ]
        md_lines.append("| " + " | ".join(cells) + " |")

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

    # ---- Save per-sample detail JSON -------------------------------------
    detail_json_path = out_json_path.parent / out_json_path.name.replace(".json", "-samples.json")
    detail_json_path.write_text(json.dumps(sample_details, indent=2))
    print(f"Wrote per-sample   → {detail_json_path}")

    # ---- Console summary -------------------------------------------------
    print("\n=== Headline ===")
    for r in rows:
        delta = _delta_str(r["base_primary"], r["lora_primary"], r["lower_is_better"])
        print(f"  {DIM_LABEL.get(r['dim'], r['dim']):24s} {r['primary_metric']:20s} "
              f"base={_fmt_metric(r['base_primary'])}  "
              f"lora={_fmt_metric(r['lora_primary'])}  "
              f"Δ={delta}  "
              f"parse={_fmt_metric(r['base_parse_rate'])}/{_fmt_metric(r['lora_parse_rate'])}")

    # ---- CSV export (per-sample, base vs LoRA side-by-side) ----------------
    if args.out_csv:
        import csv
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        def _expand3(v):
            if isinstance(v, list) and len(v) == 3:
                return [f"{x:.4f}" for x in v]
            return ["", "", ""]

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
            "sample_id", "suite", "task_id", "task_description",
            "target_object", "dest_object",
            # Q1
            "Q1_SrcObj_gt_x", "Q1_SrcObj_gt_y", "Q1_SrcObj_gt_z",
            "Q1_SrcObj_base_x", "Q1_SrcObj_base_y", "Q1_SrcObj_base_z",
            "Q1_SrcObj_lora_x", "Q1_SrcObj_lora_y", "Q1_SrcObj_lora_z",
            "Q1_SrcObj_base_err_x", "Q1_SrcObj_base_err_y", "Q1_SrcObj_base_err_z",
            "Q1_SrcObj_lora_err_x", "Q1_SrcObj_lora_err_y", "Q1_SrcObj_lora_err_z",
            # Q2
            "Q2_Grip_gt_x", "Q2_Grip_gt_y", "Q2_Grip_gt_z",
            "Q2_Grip_base_x", "Q2_Grip_base_y", "Q2_Grip_base_z",
            "Q2_Grip_lora_x", "Q2_Grip_lora_y", "Q2_Grip_lora_z",
            "Q2_Grip_base_err_x", "Q2_Grip_base_err_y", "Q2_Grip_base_err_z",
            "Q2_Grip_lora_err_x", "Q2_Grip_lora_err_y", "Q2_Grip_lora_err_z",
            # Q3
            "Q3_Offset_gt_x", "Q3_Offset_gt_y", "Q3_Offset_gt_z",
            "Q3_Offset_base_x", "Q3_Offset_base_y", "Q3_Offset_base_z",
            "Q3_Offset_lora_x", "Q3_Offset_lora_y", "Q3_Offset_lora_z",
            "Q3_Offset_base_err_x", "Q3_Offset_base_err_y", "Q3_Offset_base_err_z",
            "Q3_Offset_lora_err_x", "Q3_Offset_lora_err_y", "Q3_Offset_lora_err_z",
            # Q4
            "Q4_SpatialRel_gt", "Q4_SpatialRel_base", "Q4_SpatialRel_lora",
            "Q4_SpatialRel_base_ok", "Q4_SpatialRel_lora_ok",
            # Q5
            "Q5_PairDist_gt", "Q5_PairDist_base", "Q5_PairDist_lora",
            "Q5_PairDist_base_err", "Q5_PairDist_lora_err",
            # Q6
            "Q6_Orient_gt_roll", "Q6_Orient_gt_pitch", "Q6_Orient_gt_yaw",
            "Q6_Orient_base_roll", "Q6_Orient_base_pitch", "Q6_Orient_base_yaw",
            "Q6_Orient_lora_roll", "Q6_Orient_lora_pitch", "Q6_Orient_lora_yaw",
            "Q6_Orient_base_err_roll", "Q6_Orient_base_err_pitch", "Q6_Orient_base_err_yaw",
            "Q6_Orient_lora_err_roll", "Q6_Orient_lora_err_pitch", "Q6_Orient_lora_err_yaw",
            # Q7
            "Q7_GripOpen_gt", "Q7_GripOpen_base", "Q7_GripOpen_lora",
            "Q7_GripOpen_base_err", "Q7_GripOpen_lora_err",
            # Q8
            "Q8_Action_gt", "Q8_Action_base", "Q8_Action_lora",
            "Q8_base_trans_err_x", "Q8_base_trans_err_y", "Q8_base_trans_err_z",
            "Q8_base_trans_cos",
            "Q8_base_rot_err_roll", "Q8_base_rot_err_pitch", "Q8_base_rot_err_yaw",
            "Q8_base_grip_err",
            "Q8_lora_trans_err_x", "Q8_lora_trans_err_y", "Q8_lora_trans_err_z",
            "Q8_lora_trans_cos",
            "Q8_lora_rot_err_roll", "Q8_lora_rot_err_pitch", "Q8_lora_rot_err_yaw",
            "Q8_lora_grip_err",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_hdrs)
            for s in sample_details:
                q1 = s.get("q1", {})
                q2 = s.get("q2", {})
                q3 = s.get("q3", {})
                q4 = s.get("q4", {})
                q5 = s.get("q5", {})
                q6 = s.get("q6", {})
                q7 = s.get("q7", {})
                q8 = s.get("q8", {})
                row = [
                    s.get("sample_id", ""),
                    s.get("suite", ""),
                    s.get("task_id", ""),
                    s.get("task_description", ""),
                    s.get("target_object", ""),
                    s.get("dest_object", ""),
                    # Q1
                    *_expand3(q1.get("gt")),
                    *_expand3(q1.get("base_pred")),
                    *_expand3(q1.get("lora_pred")),
                    *_expand3(q1.get("base_err")),
                    *_expand3(q1.get("lora_err")),
                    # Q2
                    *_expand3(q2.get("gt")),
                    *_expand3(q2.get("base_pred")),
                    *_expand3(q2.get("lora_pred")),
                    *_expand3(q2.get("base_err")),
                    *_expand3(q2.get("lora_err")),
                    # Q3
                    *_expand3(q3.get("gt")),
                    *_expand3(q3.get("base_pred")),
                    *_expand3(q3.get("lora_pred")),
                    *_expand3(q3.get("base_err")),
                    *_expand3(q3.get("lora_err")),
                    # Q4
                    _csv_rel(q4.get("gt")),
                    _csv_rel(q4.get("base_pred")),
                    _csv_rel(q4.get("lora_pred")),
                    _csv_q4ok(q4.get("base_correct")),
                    _csv_q4ok(q4.get("lora_correct")),
                    # Q5
                    _csv_scalar(q5.get("gt")),
                    _csv_scalar(q5.get("base_pred")),
                    _csv_scalar(q5.get("lora_pred")),
                    _csv_scalar(q5.get("base_err")),
                    _csv_scalar(q5.get("lora_err")),
                    # Q6
                    *_expand3(q6.get("gt")),
                    *_expand3(q6.get("base_pred")),
                    *_expand3(q6.get("lora_pred")),
                    *_expand3(q6.get("base_err_deg")),
                    *_expand3(q6.get("lora_err_deg")),
                    # Q7
                    _csv_scalar(q7.get("gt")),
                    _csv_scalar(q7.get("base_pred")),
                    _csv_scalar(q7.get("lora_pred")),
                    _csv_scalar(q7.get("base_err")),
                    _csv_scalar(q7.get("lora_err")),
                    # Q8
                    _csv_action7(q8.get("gt")),
                    _csv_action7(q8.get("base_pred")),
                    _csv_action7(q8.get("lora_pred")),
                    *_expand3(q8.get("base_trans_err")),
                    _csv_scalar(q8.get("base_trans_cos_sim")),
                    *_expand3(q8.get("base_rot_err")),
                    _csv_scalar(q8.get("base_grip_err")),
                    *_expand3(q8.get("lora_trans_err")),
                    _csv_scalar(q8.get("lora_trans_cos_sim")),
                    *_expand3(q8.get("lora_rot_err")),
                    _csv_scalar(q8.get("lora_grip_err")),
                ]
                writer.writerow(row)
        print(f"Wrote CSV          → {csv_path}")


if __name__ == "__main__":
    main()
