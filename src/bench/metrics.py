"""Metric computation for the spatial reasoning benchmark.

Metrics per question:
  Q1 / Q2 (3D position): MAE per axis, MAE overall, RMSE overall
  Q3 (can_close label):  Accuracy, F1-score, positive-rate
  Q4 (next direction):   Mean cosine similarity, median cosine similarity
  Q5 (gripper→target offset): MAE per axis, MAE overall, RMSE overall (reuses position_metrics)
  Q6 (spatial relation):  Per-axis accuracy + macro-F1, overall (all-axes) accuracy
  Q7 / Q8 / Q9 (orientation): Per-axis circular MAE (°), geodesic distance (°)
  Q10 (pairwise distance): Scalar MAE, RMSE (m)
  Q11 (gripper openness):  Scalar MAE, RMSE
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ----- Position metrics ------------------------------------------------

def mae_per_axis(preds: list[list[float]], gts: list[list[float]]) -> np.ndarray:
    """Per-axis mean absolute error. Returns shape (3,) array [x, y, z]."""
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    return np.abs(p - g).mean(axis=0)


def rmse_per_axis(preds: list[list[float]], gts: list[list[float]]) -> np.ndarray:
    """Per-axis root mean squared error. Returns shape (3,) array."""
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    return np.sqrt(((p - g) ** 2).mean(axis=0))


def mae_overall(preds: list[list[float]], gts: list[list[float]]) -> float:
    """Overall MAE (mean over all axes and samples)."""
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    return float(np.abs(p - g).mean())


def rmse_overall(preds: list[list[float]], gts: list[list[float]]) -> float:
    """Overall RMSE."""
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    return float(np.sqrt(((p - g) ** 2).mean()))


def position_metrics(
    preds: list[list[float]], gts: list[list[float]]
) -> dict:
    """Compute all position metrics for a list of (pred, gt) coordinate pairs."""
    if not preds:
        return {"n": 0, "mae_x": None, "mae_y": None, "mae_z": None,
                "mae_overall": None, "rmse_overall": None}
    mae_ax = mae_per_axis(preds, gts)
    return {
        "n": len(preds),
        "mae_x": float(mae_ax[0]),
        "mae_y": float(mae_ax[1]),
        "mae_z": float(mae_ax[2]),
        "mae_overall": mae_overall(preds, gts),
        "rmse_overall": rmse_overall(preds, gts),
    }


# ----- Classification metrics ------------------------------------------

def classification_metrics(preds: list[bool], gts: list[bool]) -> dict:
    """Accuracy and F1 for the can-close binary classification."""
    if not preds:
        return {"n": 0, "accuracy": None, "f1": None, "positive_rate_gt": None,
                "positive_rate_pred": None}
    p = [int(v) for v in preds]
    g = [int(v) for v in gts]
    return {
        "n": len(preds),
        "accuracy": float(accuracy_score(g, p)),
        "f1": float(f1_score(g, p, zero_division=0)),
        "positive_rate_gt": float(np.mean(g)),
        "positive_rate_pred": float(np.mean(p)),
    }


# ----- Direction metrics -----------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def direction_metrics(
    preds: list[list[float]], gts: list[list[float]]
) -> dict:
    """Mean and median cosine similarity between predicted and GT directions."""
    if not preds:
        return {"n": 0, "mean_cosine_sim": None, "median_cosine_sim": None}
    sims = [
        _cosine_sim(np.array(p, dtype=float), np.array(g, dtype=float))
        for p, g in zip(preds, gts)
    ]
    return {
        "n": len(sims),
        "mean_cosine_sim": float(np.mean(sims)),
        "median_cosine_sim": float(np.median(sims)),
    }


# ----- Spatial relation metrics (Q6) -----------------------------------

def spatial_relation_metrics(
    preds: list[dict[str, str | None]],
    gts: list[dict[str, str]],
) -> dict:
    """Per-axis accuracy and macro-F1 for the axis-wise spatial relation classification (Q6).

    Args:
        preds: list of {"x": label_or_None, "y": label_or_None, "z": label_or_None}
        gts:   list of {"x": label, "y": label, "z": label}

    Returns dict with per-axis accuracy, macro-F1, and all-axes accuracy.
    """
    if not preds:
        return {
            "n": 0,
            "acc_x": None, "acc_y": None, "acc_z": None, "acc_all": None,
            "f1_x": None, "f1_y": None, "f1_z": None,
        }

    results: dict[str, dict] = {}
    for axis in ("x", "y", "z"):
        gt_labels = [g[axis] for g in gts]
        pred_labels = [p[axis] for p in preds]
        results[axis] = {
            "gt": gt_labels,
            "pred": pred_labels,
        }

    acc_x = float(accuracy_score(results["x"]["gt"], results["x"]["pred"]))
    acc_y = float(accuracy_score(results["y"]["gt"], results["y"]["pred"]))
    acc_z = float(accuracy_score(results["z"]["gt"], results["z"]["pred"]))

    # All-axes accuracy: 1 only if all three axes are correct for a sample
    all_correct = [
        int(p["x"] == g["x"] and p["y"] == g["y"] and p["z"] == g["z"])
        for p, g in zip(preds, gts)
    ]
    acc_all = float(np.mean(all_correct))

    f1_x = float(f1_score(results["x"]["gt"], results["x"]["pred"], average="macro", zero_division=0))
    f1_y = float(f1_score(results["y"]["gt"], results["y"]["pred"], average="macro", zero_division=0))
    f1_z = float(f1_score(results["z"]["gt"], results["z"]["pred"], average="macro", zero_division=0))

    return {
        "n": len(preds),
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "acc_all": acc_all,
        "f1_x": f1_x,
        "f1_y": f1_y,
        "f1_z": f1_z,
    }


# ----- Angular / orientation metrics (Q7, Q8, Q9) ---------------------

def _circular_distance_deg(a: float, b: float) -> float:
    """Circular distance between two angles in degrees, handling ±180° wrap."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def geodesic_distance_deg(q1_wxyz: list[float], q2_wxyz: list[float]) -> float:
    """Geodesic angular distance (degrees) between two unit quaternions [w,x,y,z].

    Uses formula: angle = 2 * arccos(|q1 · q2|).
    Handles q / -q sign ambiguity via absolute value of dot product.
    """
    q1 = np.array(q1_wxyz, dtype=float)
    q2 = np.array(q2_wxyz, dtype=float)
    # Normalize
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


def angular_metrics(
    preds_euler: list[list[float]],
    gts_euler: list[list[float]],
    preds_quat: list[list[float]] | None = None,
    gts_quat: list[list[float]] | None = None,
) -> dict:
    """Compute orientation error metrics.

    Args:
        preds_euler, gts_euler: lists of [roll, pitch, yaw] in degrees.
        preds_quat, gts_quat:  optional lists of [w,x,y,z] for geodesic error.

    Returns dict with mae_roll_deg, mae_pitch_deg, mae_yaw_deg,
    mae_overall_deg, geodesic_mean_deg, geodesic_median_deg.
    """
    if not preds_euler:
        return {
            "n": 0,
            "mae_roll_deg": None, "mae_pitch_deg": None, "mae_yaw_deg": None,
            "mae_overall_deg": None,
            "geodesic_mean_deg": None, "geodesic_median_deg": None,
        }

    # Per-axis circular MAE
    roll_errs = [_circular_distance_deg(p[0], g[0]) for p, g in zip(preds_euler, gts_euler)]
    pitch_errs = [_circular_distance_deg(p[1], g[1]) for p, g in zip(preds_euler, gts_euler)]
    yaw_errs = [_circular_distance_deg(p[2], g[2]) for p, g in zip(preds_euler, gts_euler)]

    mae_roll = float(np.mean(roll_errs))
    mae_pitch = float(np.mean(pitch_errs))
    mae_yaw = float(np.mean(yaw_errs))
    mae_overall = float(np.mean([mae_roll, mae_pitch, mae_yaw]))

    # Geodesic distance (requires quaternions)
    geo_mean, geo_median = None, None
    if preds_quat and gts_quat and len(preds_quat) == len(gts_quat):
        geo_dists = [
            geodesic_distance_deg(p, g)
            for p, g in zip(preds_quat, gts_quat)
        ]
        geo_mean = float(np.mean(geo_dists))
        geo_median = float(np.median(geo_dists))

    return {
        "n": len(preds_euler),
        "mae_roll_deg": mae_roll,
        "mae_pitch_deg": mae_pitch,
        "mae_yaw_deg": mae_yaw,
        "mae_overall_deg": mae_overall,
        "geodesic_mean_deg": geo_mean,
        "geodesic_median_deg": geo_median,
    }


# ----- Scalar metrics (Q10 distance, Q11 openness) --------------------

def scalar_metrics(
    preds: list[float],
    gts: list[float],
) -> dict:
    """Metrics for a single scalar prediction (distance or openness).

    Returns dict with n, mae, rmse.
    """
    if not preds:
        return {"n": 0, "mae": None, "rmse": None}
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    return {
        "n": len(preds),
        "mae": float(np.abs(p - g).mean()),
        "rmse": float(np.sqrt(((p - g) ** 2).mean())),
    }


# ----- Parse-rate helper -----------------------------------------------

def parse_rate(parsed_list: list[bool | None], total: int) -> float:
    """Fraction of responses where a field was successfully parsed."""
    n_parsed = sum(1 for v in parsed_list if v is not None)
    return n_parsed / total if total > 0 else 0.0


# ----- Summary table ---------------------------------------------------

def format_results_table(results: dict[str, dict]) -> str:
    """Format a per-model results dict as a human-readable ASCII table.

    ``results`` structure::

        {
            "model_name": {
                "parse_rate": float,
                "q1": {...position_metrics output...},
                "q2": {...position_metrics output...},
                "q3": {...classification_metrics output...},
                "q4": {...direction_metrics output...},
            },
            ...
        }
    """
    lines = []
    sep = "-" * 160

    lines.append(sep)
    lines.append(
        f"{'Model':<30}  {'Parse%':>7}  "
        f"{'Q1 MAE(m)':>10}  {'Q2 MAE(m)':>10}  "
        f"{'Q3 MAE(m)':>10}  {'Q4 AccAll':>10}  "
        f"{'Q5 MAE(m)':>10}  {'Q6 MAE(°)':>10}  "
        f"{'Q7 MAE':>8}  {'Q8 TrMAE':>8}"
    )
    lines.append(sep)

    for model, m in results.items():
        def _f(val, fmt=".4f"):
            return f"{val:{fmt}}" if val is not None else "   N/A  "

        q1_mae  = m.get("q1", {}).get("mae_overall")
        q2_mae  = m.get("q2", {}).get("mae_overall")
        q3_mae  = m.get("q3", {}).get("mae_overall")
        q4_acc  = m.get("q4", {}).get("acc_all")
        q5_mae  = m.get("q5", {}).get("mae")
        q6_mae  = m.get("q6", {}).get("mae_overall_deg")
        q7_mae  = m.get("q7", {}).get("mae")
        q8_mae  = m.get("q8", {}).get("mae_translation")
        parse   = m.get("parse_rate")

        lines.append(
            f"{model:<30}  {_f(parse, '.1%'):>7}  "
            f"{_f(q1_mae):>10}  {_f(q2_mae):>10}  "
            f"{_f(q3_mae):>10}  {_f(q4_acc):>10}  "
            f"{_f(q5_mae):>10}  {_f(q6_mae, '.1f'):>10}  "
            f"{_f(q7_mae):>8}  {_f(q8_mae):>8}"
        )

    lines.append(sep)
    return "\n".join(lines)
