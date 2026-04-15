#!/usr/bin/env python3
"""
VLM Spatial Reasoning Benchmark Runner

This script runs the zero-shot evaluation of VLM spatial reasoning capabilities
on LIBERO manipulation tasks.

Usage:
    # Run with default settings (requires LIBERO installed)
    python run_benchmark.py

    # Run with mock VLM (for testing)
    python run_benchmark.py --mock

    # Run evaluation only (if data already extracted)
    python run_benchmark.py --eval-only

    # Use specific model via API
    python run_benchmark.py --use-api --api-base http://localhost:8000/v1
"""

# Add LIBERO to path if installed in editable mode
import sys
import os
_libero_path = os.path.join(os.path.dirname(__file__), 'LIBERO/libero')
if os.path.exists(_libero_path):
    sys.path.insert(0, os.path.dirname(_libero_path))

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from datetime import datetime

from vlm_benchmark.data_extraction import BenchmarkSample, sample_benchmark_data
from vlm_benchmark.data_extraction.extract_gt import load_benchmark_samples, save_benchmark_samples
from vlm_benchmark.prompts import (
    get_object_localization_prompt,
    get_gripper_close_prompt,
    get_move_direction_prompt,
)
from vlm_benchmark.evaluation import (
    evaluate_coordinate_prediction,
    evaluate_grasp_timing,
    evaluate_direction_prediction,
    aggregate_results,
)
from vlm_benchmark.evaluation.metrics import print_results, CoordinateMetrics, DirectionMetrics
from vlm_benchmark.vlm_inference import QwenVL, VLMResponse
from vlm_benchmark.vlm_inference.qwen_vl import MockVLM


def run_data_extraction(args) -> List[BenchmarkSample]:
    """Extract benchmark data from LIBERO environment."""
    print("\n=== Data Extraction Phase ===\n")

    samples = sample_benchmark_data(
        benchmark_name=args.benchmark,
        num_tasks=args.num_tasks,
        samples_per_task=args.samples_per_task,
        output_dir=args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
    )

    # Save samples
    gt_path = Path(args.data_dir) / "ground_truth" / "benchmark_samples.json"
    save_benchmark_samples(samples, str(gt_path))

    return samples


def run_vlm_inference(
    vlm,
    samples: List[BenchmarkSample],
    question_types: List[str],
) -> Dict[str, List[Dict]]:
    """
    Run VLM inference on all samples for specified question types.

    Returns:
        Dictionary mapping question type to list of (prediction, response) pairs
    """
    print("\n=== VLM Inference Phase ===\n")

    results = {q: [] for q in question_types}

    for i, sample in enumerate(samples):
        print(f"Processing sample {i + 1}/{len(samples)}: {sample.sample_id}")

        # Prepare images
        images = [sample.agentview_image_path, sample.eye_in_hand_image_path]

        # Skip if images don't exist
        if not all(Path(p).exists() for p in images if p):
            print(f"  Warning: Missing images for {sample.sample_id}")
            continue

        for question_type in question_types:
            # Generate prompt
            if question_type == "object_localization":
                prompt = get_object_localization_prompt(
                    sample.eef_pos, sample.target_object_name
                )
            elif question_type == "gripper_close":
                prompt = get_gripper_close_prompt(
                    sample.eef_pos, sample.target_object_name
                )
            elif question_type == "move_direction":
                prompt = get_move_direction_prompt(
                    sample.eef_pos, sample.target_object_name
                )
            else:
                continue

            # Run inference
            response = vlm.inference(images, prompt)

            # Store result
            results[question_type].append({
                "sample_id": sample.sample_id,
                "sample": sample,
                "response": response,
                "parsed": response.parsed_json,
            })

            if response.success:
                print(f"  {question_type}: {response.parsed_json}")
            else:
                print(f"  {question_type}: Failed - {response.error_message or 'Parse error'}")

    return results


def evaluate_results(
    inference_results: Dict[str, List[Dict]],
    samples: List[BenchmarkSample],
) -> Dict:
    """Evaluate VLM predictions against ground truth."""
    print("\n=== Evaluation Phase ===\n")

    # Create sample lookup
    sample_lookup = {s.sample_id: s for s in samples}

    evaluation = {}

    # Q1: Object Localization
    if "object_localization" in inference_results:
        coord_metrics = []
        for result in inference_results["object_localization"]:
            if result["parsed"] is None:
                continue

            sample = result["sample"]
            pred = result["parsed"]

            # Validate response
            if not all(k in pred for k in ["x", "y", "z"]):
                continue

            metrics = evaluate_coordinate_prediction(
                pred_coords=pred,
                gt_coords=sample.target_object_pos,
            )
            coord_metrics.append(metrics)

        evaluation["object_localization"] = coord_metrics
        print(f"Q1 Object Localization: {len(coord_metrics)} valid predictions")

    # Q2: Gripper Close
    if "gripper_close" in inference_results:
        predictions = []
        ground_truths = []

        for result in inference_results["gripper_close"]:
            if result["parsed"] is None:
                continue

            sample = result["sample"]
            pred = result["parsed"]

            if "can_close" not in pred:
                continue

            predictions.append(bool(pred["can_close"]))
            ground_truths.append(sample.can_gripper_close)

        if predictions:
            grasp_metrics = evaluate_grasp_timing(predictions, ground_truths)
            evaluation["grasp_timing"] = grasp_metrics
            print(f"Q2 Gripper Close: {len(predictions)} valid predictions")

    # Q3: Move Direction
    if "move_direction" in inference_results:
        direction_metrics = []
        for result in inference_results["move_direction"]:
            if result["parsed"] is None:
                continue

            sample = result["sample"]
            pred = result["parsed"]

            if not all(k in pred for k in ["dx", "dy", "dz"]):
                continue

            metrics = evaluate_direction_prediction(
                pred_direction=pred,
                gt_direction=sample.move_direction,
            )
            direction_metrics.append(metrics)

        evaluation["move_direction"] = direction_metrics
        print(f"Q3 Move Direction: {len(direction_metrics)} valid predictions")

    return evaluation


def save_results(
    inference_results: Dict,
    evaluation: Dict,
    output_dir: str,
    model_name: str,
):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]

    # Save raw predictions
    predictions_file = output_dir / f"predictions_{model_short}_{timestamp}.json"
    predictions_data = {}
    for q_type, results in inference_results.items():
        predictions_data[q_type] = [
            {
                "sample_id": r["sample_id"],
                "parsed": r["parsed"],
                "raw_text": r["response"].raw_text,
                "success": r["response"].success,
            }
            for r in results
        ]

    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Saved predictions to {predictions_file}")

    # Save evaluation metrics
    metrics_file = output_dir / f"metrics_{model_short}_{timestamp}.json"

    # Convert dataclasses to dict
    metrics_data = {}
    if "object_localization" in evaluation:
        metrics_data["object_localization"] = [
            {
                "mae": m.mae,
                "rmse": m.rmse,
                "euclidean_distance": m.euclidean_distance,
                "x_error": m.x_error,
                "y_error": m.y_error,
                "z_error": m.z_error,
            }
            for m in evaluation["object_localization"]
        ]

    if "grasp_timing" in evaluation:
        gt = evaluation["grasp_timing"]
        metrics_data["grasp_timing"] = {
            "accuracy": gt.accuracy,
            "f1_score": gt.f1_score,
            "precision": gt.precision,
            "recall": gt.recall,
        }

    if "move_direction" in evaluation:
        metrics_data["move_direction"] = [
            {
                "cosine_similarity": m.cosine_similarity,
                "angle_error_degrees": m.angle_error_degrees,
            }
            for m in evaluation["move_direction"]
        ]

    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Spatial Reasoning Benchmark"
    )

    # Data extraction args
    parser.add_argument("--benchmark", type=str, default="libero_spatial",
                        help="LIBERO benchmark name")
    parser.add_argument("--num-tasks", type=int, default=10,
                        help="Number of tasks to sample")
    parser.add_argument("--samples-per-task", type=int, default=5,
                        help="Samples per task")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data output directory")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # VLM args
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="VLM model name")
    parser.add_argument("--use-api", action="store_true",
                        help="Use API instead of local inference")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for local inference")

    # Execution args
    parser.add_argument("--mock", action="store_true",
                        help="Use mock VLM for testing")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (data must exist)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract data (no VLM inference)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results output directory")

    # Question types
    parser.add_argument("--questions", type=str, nargs="+",
                        default=["object_localization", "gripper_close", "move_direction"],
                        help="Question types to evaluate")

    args = parser.parse_args()

    print("=" * 60)
    print("VLM SPATIAL REASONING BENCHMARK")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Questions: {args.questions}")
    print(f"Total samples: {args.num_tasks * args.samples_per_task}")
    print("=" * 60)

    # Step 1: Data extraction or loading
    gt_path = Path(args.data_dir) / "ground_truth" / "benchmark_samples.json"

    if args.eval_only and gt_path.exists():
        print("\nLoading existing benchmark data...")
        samples = load_benchmark_samples(str(gt_path))
    else:
        samples = run_data_extraction(args)

    if args.extract_only:
        print("\nData extraction complete. Exiting.")
        return

    # Step 2: VLM inference
    if args.mock:
        vlm = MockVLM()
        print("\nUsing MockVLM for testing")
    else:
        vlm = QwenVL(
            model_name=args.model,
            use_api=args.use_api,
            api_base=args.api_base,
            api_key=args.api_key,
            device=args.device,
        )

    inference_results = run_vlm_inference(vlm, samples, args.questions)

    # Step 3: Evaluation
    evaluation = evaluate_results(inference_results, samples)

    # Step 4: Aggregate and print results
    aggregated = aggregate_results(
        coordinate_results=evaluation.get("object_localization", []),
        grasp_results=evaluation.get("grasp_timing"),
        direction_results=evaluation.get("move_direction", []),
    )

    print_results(aggregated)

    # Step 5: Save results
    save_results(
        inference_results=inference_results,
        evaluation=evaluation,
        output_dir=args.results_dir,
        model_name=args.model,
    )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
