"""
Evaluation metrics for VLM spatial reasoning benchmark.

Metrics:
- Q1 (Object Localization): MAE, RMSE, Euclidean distance, per-axis error
- Q2 (Gripper Close): Accuracy, F1-score, Precision, Recall
- Q3 (Move Direction): Cosine similarity, Angle error
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class CoordinateMetrics:
    """Metrics for coordinate prediction evaluation."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    euclidean_distance: float  # Euclidean distance
    x_error: float  # X-axis absolute error
    y_error: float  # Y-axis absolute error
    z_error: float  # Z-axis absolute error


@dataclass
class GraspTimingMetrics:
    """Metrics for grasp timing evaluation."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass
class DirectionMetrics:
    """Metrics for direction prediction evaluation."""
    cosine_similarity: float
    angle_error_degrees: float


def evaluate_coordinate_prediction(
    pred_coords: Union[Dict, List, np.ndarray],
    gt_coords: Union[List, np.ndarray],
) -> CoordinateMetrics:
    """
    Evaluate coordinate prediction accuracy.

    Args:
        pred_coords: Predicted coordinates as dict {"x": float, "y": float, "z": float}
                    or list/array [x, y, z]
        gt_coords: Ground truth coordinates as list/array [x, y, z]

    Returns:
        CoordinateMetrics with MAE, RMSE, Euclidean distance, and per-axis errors
    """
    # Parse prediction
    if isinstance(pred_coords, dict):
        pred = np.array([pred_coords['x'], pred_coords['y'], pred_coords['z']])
    else:
        pred = np.array(pred_coords)

    gt = np.array(gt_coords)

    # Per-axis absolute errors
    axis_errors = np.abs(pred - gt)

    # Mean Absolute Error
    mae = float(np.mean(axis_errors))

    # Root Mean Square Error
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    # Euclidean distance
    euclidean = float(np.linalg.norm(pred - gt))

    return CoordinateMetrics(
        mae=mae,
        rmse=rmse,
        euclidean_distance=euclidean,
        x_error=float(axis_errors[0]),
        y_error=float(axis_errors[1]),
        z_error=float(axis_errors[2]),
    )


def evaluate_grasp_timing(
    predictions: List[bool],
    ground_truths: List[bool],
) -> GraspTimingMetrics:
    """
    Evaluate grasp timing (can_close) predictions.

    Args:
        predictions: List of predicted boolean values
        ground_truths: List of ground truth boolean values

    Returns:
        GraspTimingMetrics with accuracy, F1, precision, recall
    """
    predictions = np.array(predictions, dtype=bool)
    ground_truths = np.array(ground_truths, dtype=bool)

    # Confusion matrix elements
    tp = int(np.sum(predictions & ground_truths))
    tn = int(np.sum(~predictions & ~ground_truths))
    fp = int(np.sum(predictions & ~ground_truths))
    fn = int(np.sum(~predictions & ground_truths))

    total = len(predictions)

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return GraspTimingMetrics(
        accuracy=accuracy,
        f1_score=f1_score,
        precision=precision,
        recall=recall,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_direction_prediction(
    pred_direction: Union[Dict, List, np.ndarray],
    gt_direction: Union[List, np.ndarray],
) -> DirectionMetrics:
    """
    Evaluate move direction prediction.

    Args:
        pred_direction: Predicted direction as dict {"dx": float, "dy": float, "dz": float}
                       or list/array [dx, dy, dz]
        gt_direction: Ground truth direction as list/array [dx, dy, dz]

    Returns:
        DirectionMetrics with cosine similarity and angle error
    """
    # Parse prediction
    if isinstance(pred_direction, dict):
        pred = np.array([pred_direction['dx'], pred_direction['dy'], pred_direction['dz']])
    else:
        pred = np.array(pred_direction)

    gt = np.array(gt_direction)

    # Normalize vectors
    pred_norm = np.linalg.norm(pred)
    gt_norm = np.linalg.norm(gt)

    if pred_norm < 1e-8 or gt_norm < 1e-8:
        # Handle zero vectors
        return DirectionMetrics(
            cosine_similarity=0.0,
            angle_error_degrees=90.0,  # Perpendicular assumption
        )

    pred_unit = pred / pred_norm
    gt_unit = gt / gt_norm

    # Cosine similarity
    cosine_sim = float(np.dot(pred_unit, gt_unit))

    # Clamp to [-1, 1] for numerical stability
    cosine_sim_clamped = np.clip(cosine_sim, -1.0, 1.0)

    # Angle error in degrees
    angle_error = float(np.arccos(cosine_sim_clamped) * 180.0 / np.pi)

    return DirectionMetrics(
        cosine_similarity=cosine_sim,
        angle_error_degrees=angle_error,
    )


def aggregate_results(
    coordinate_results: List[CoordinateMetrics],
    grasp_results: Optional[GraspTimingMetrics] = None,
    direction_results: List[DirectionMetrics] = None,
) -> Dict:
    """
    Aggregate evaluation results across all samples.

    Args:
        coordinate_results: List of coordinate metrics for each sample
        grasp_results: Aggregated grasp timing metrics (already computed)
        direction_results: List of direction metrics for each sample

    Returns:
        Dictionary with aggregated statistics
    """
    results = {}

    # Q1: Coordinate localization
    if coordinate_results:
        n = len(coordinate_results)
        results["object_localization"] = {
            "n_samples": n,
            "mae_mean": np.mean([r.mae for r in coordinate_results]),
            "mae_std": np.std([r.mae for r in coordinate_results]),
            "rmse_mean": np.mean([r.rmse for r in coordinate_results]),
            "rmse_std": np.std([r.rmse for r in coordinate_results]),
            "euclidean_mean": np.mean([r.euclidean_distance for r in coordinate_results]),
            "euclidean_std": np.std([r.euclidean_distance for r in coordinate_results]),
            "x_error_mean": np.mean([r.x_error for r in coordinate_results]),
            "y_error_mean": np.mean([r.y_error for r in coordinate_results]),
            "z_error_mean": np.mean([r.z_error for r in coordinate_results]),
            "x_error_std": np.std([r.x_error for r in coordinate_results]),
            "y_error_std": np.std([r.y_error for r in coordinate_results]),
            "z_error_std": np.std([r.z_error for r in coordinate_results]),
        }

    # Q2: Grasp timing
    if grasp_results:
        results["grasp_timing"] = {
            "accuracy": grasp_results.accuracy,
            "f1_score": grasp_results.f1_score,
            "precision": grasp_results.precision,
            "recall": grasp_results.recall,
            "true_positives": grasp_results.true_positives,
            "true_negatives": grasp_results.true_negatives,
            "false_positives": grasp_results.false_positives,
            "false_negatives": grasp_results.false_negatives,
        }

    # Q3: Move direction
    if direction_results:
        n = len(direction_results)
        results["move_direction"] = {
            "n_samples": n,
            "cosine_similarity_mean": np.mean([r.cosine_similarity for r in direction_results]),
            "cosine_similarity_std": np.std([r.cosine_similarity for r in direction_results]),
            "angle_error_mean": np.mean([r.angle_error_degrees for r in direction_results]),
            "angle_error_std": np.std([r.angle_error_degrees for r in direction_results]),
            "angle_error_median": np.median([r.angle_error_degrees for r in direction_results]),
        }

    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("VLM SPATIAL REASONING BENCHMARK RESULTS")
    print("=" * 60)

    if "object_localization" in results:
        print("\n--- Q1: Object Localization ---")
        ol = results["object_localization"]
        print(f"  Samples: {ol['n_samples']}")
        print(f"  MAE:       {ol['mae_mean']:.4f} ± {ol['mae_std']:.4f} m")
        print(f"  RMSE:      {ol['rmse_mean']:.4f} ± {ol['rmse_std']:.4f} m")
        print(f"  Euclidean: {ol['euclidean_mean']:.4f} ± {ol['euclidean_std']:.4f} m")
        print(f"  Per-axis errors (mean):")
        print(f"    X: {ol['x_error_mean']:.4f} m")
        print(f"    Y: {ol['y_error_mean']:.4f} m")
        print(f"    Z: {ol['z_error_mean']:.4f} m")

    if "grasp_timing" in results:
        print("\n--- Q2: Grasp Timing ---")
        gt = results["grasp_timing"]
        print(f"  Accuracy:  {gt['accuracy']:.2%}")
        print(f"  F1 Score:  {gt['f1_score']:.4f}")
        print(f"  Precision: {gt['precision']:.4f}")
        print(f"  Recall:    {gt['recall']:.4f}")
        print(f"  Confusion: TP={gt['true_positives']}, TN={gt['true_negatives']}, "
              f"FP={gt['false_positives']}, FN={gt['false_negatives']}")

    if "move_direction" in results:
        print("\n--- Q3: Move Direction ---")
        md = results["move_direction"]
        print(f"  Samples: {md['n_samples']}")
        print(f"  Cosine Similarity: {md['cosine_similarity_mean']:.4f} ± {md['cosine_similarity_std']:.4f}")
        print(f"  Angle Error:       {md['angle_error_mean']:.2f}° ± {md['angle_error_std']:.2f}°")
        print(f"  Angle Error (median): {md['angle_error_median']:.2f}°")

    print("\n" + "=" * 60)
