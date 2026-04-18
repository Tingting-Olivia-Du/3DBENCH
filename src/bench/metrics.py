"""Metric computation for the spatial reasoning benchmark.

Metrics per question:
  Q1 / Q2 (3D position): MAE per axis, MAE overall, RMSE overall
  Q3 (can_close label):  Accuracy, F1-score, positive-rate
  Q4 (next direction):   Mean cosine similarity, median cosine similarity
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
    sep = "-" * 90

    lines.append(sep)
    lines.append(
        f"{'Model':<30}  {'Parse%':>7}  "
        f"{'Q1 MAE(m)':>10}  {'Q2 MAE(m)':>10}  "
        f"{'Q3 Acc':>8}  {'Q3 F1':>8}  "
        f"{'Q4 CosSim':>10}"
    )
    lines.append(sep)

    for model, m in results.items():
        def _f(val, fmt=".4f"):
            return f"{val:{fmt}}" if val is not None else "   N/A  "

        q1_mae = m.get("q1", {}).get("mae_overall")
        q2_mae = m.get("q2", {}).get("mae_overall")
        q3_acc = m.get("q3", {}).get("accuracy")
        q3_f1  = m.get("q3", {}).get("f1")
        q4_cos = m.get("q4", {}).get("mean_cosine_sim")
        parse  = m.get("parse_rate")

        lines.append(
            f"{model:<30}  {_f(parse, '.1%'):>7}  "
            f"{_f(q1_mae):>10}  {_f(q2_mae):>10}  "
            f"{_f(q3_acc):>8}  {_f(q3_f1):>8}  "
            f"{_f(q4_cos):>10}"
        )

    lines.append(sep)
    return "\n".join(lines)
