"""
Sample states from LIBERO environment for benchmark.

Sampling strategy:
- 10 tasks from libero_spatial
- 5 states per task (different phases of manipulation)
- Total: 50 samples for first version
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import os
from PIL import Image

from .extract_gt import BenchmarkSample, extract_ground_truth


def get_target_object_from_task(task_language: str) -> str:
    """
    Extract target object name from task language description.

    Examples:
    - "pick up the black bowl..." -> "black_bowl"
    - "pick up the alphabet soup..." -> "alphabet_soup"
    """
    # Common patterns in libero_spatial tasks
    task_lower = task_language.lower()

    # Try to extract "pick up the X" pattern
    if "pick up the" in task_lower:
        # Get text after "pick up the"
        after_pickup = task_lower.split("pick up the")[1].strip()
        # Get first object phrase (before "and" or "from" or other connectors)
        for connector in [" and ", " from ", " on ", " in ", " between "]:
            if connector in after_pickup:
                after_pickup = after_pickup.split(connector)[0]

        # Convert to object name format (replace spaces with underscores)
        obj_name = after_pickup.strip().replace(" ", "_")
        return obj_name

    # Fallback: return first noun-like phrase
    return "target_object"


def sample_benchmark_data(
    benchmark_name: str = "libero_spatial",
    num_tasks: int = 10,
    samples_per_task: int = 5,
    output_dir: str = "data",
    image_size: int = 256,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """
    Sample benchmark data from LIBERO environment.

    This function requires LIBERO to be installed and available.

    Args:
        benchmark_name: Name of LIBERO benchmark suite
        num_tasks: Number of tasks to sample from
        samples_per_task: Number of samples per task
        output_dir: Directory to save images and ground truth
        image_size: Image resolution (default 256x256)
        seed: Random seed for reproducibility

    Returns:
        List of BenchmarkSample objects
    """
    # Import LIBERO (will fail if not installed)
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise ImportError(
            "LIBERO is not installed. Please install it first:\n"
            "pip install -e LIBERO/libero"
        ) from e

    np.random.seed(seed)

    # Setup output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    gt_dir = output_dir / "ground_truth"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    print(f"Loading benchmark: {benchmark_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[benchmark_name]()

    # Get number of available tasks
    n_available_tasks = task_suite.n_tasks
    num_tasks = min(num_tasks, n_available_tasks)
    print(f"Sampling from {num_tasks} tasks (available: {n_available_tasks})")

    samples = []

    for task_id in range(num_tasks):
        print(f"\n--- Task {task_id + 1}/{num_tasks} ---")

        # Get task info
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_language = task.language

        print(f"Task: {task_name}")
        print(f"Language: {task_language}")

        # Get target object from task description
        target_object = get_target_object_from_task(task_language)
        print(f"Target object: {target_object}")

        # Get BDDL file path
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=image_size,
            camera_widths=image_size,
        )

        # Get initial states for reproducibility
        init_states = task_suite.get_task_init_states(task_id)
        n_init_states = len(init_states) if init_states is not None else 1

        # Sample different time steps
        # Strategy: initial, 25%, 50%, 75%, near-grasp
        sample_phases = [0.0, 0.25, 0.5, 0.75, 0.95]

        for sample_idx in range(samples_per_task):
            print(f"  Sample {sample_idx + 1}/{samples_per_task}")

            # Use different init state for variety
            init_state_idx = sample_idx % n_init_states
            episode_id = sample_idx

            # Reset environment
            env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[init_state_idx])
            else:
                obs = env.reset()

            # Simulate some random actions to get different states
            phase = sample_phases[sample_idx % len(sample_phases)]
            n_steps = int(phase * 100)  # Max 100 steps

            action = None
            for step in range(n_steps):
                # Random action for exploration
                action = np.random.uniform(-0.1, 0.1, size=7)
                action[6] = -1  # Keep gripper open
                obs, _, done, _ = env.step(action)
                if done:
                    break

            # Create sample ID
            sample_id = f"task{task_id:02d}_ep{episode_id:02d}_step{n_steps:03d}"

            # Save images
            agentview_path = images_dir / f"{sample_id}_agentview.png"
            eye_in_hand_path = images_dir / f"{sample_id}_eye_in_hand.png"

            Image.fromarray(obs["agentview_image"]).save(agentview_path)
            Image.fromarray(obs["robot0_eye_in_hand_image"]).save(eye_in_hand_path)

            # Extract ground truth
            gt = extract_ground_truth(
                env=env,
                obs=obs,
                target_object_name=target_object,
                action=action,
            )

            # Create benchmark sample
            sample = BenchmarkSample(
                sample_id=sample_id,
                task_id=task_id,
                task_name=task_name,
                task_language=task_language,
                episode_id=episode_id,
                step_id=n_steps,
                agentview_image_path=str(agentview_path),
                eye_in_hand_image_path=str(eye_in_hand_path),
                eef_pos=gt["eef_pos"],
                eef_quat=gt["eef_quat"],
                target_object_name=target_object,
                target_object_pos=gt.get("target_object_pos", np.zeros(3)),
                target_object_quat=gt.get("target_object_quat", np.array([1, 0, 0, 0])),
                objects_pos=gt["objects_pos"],
                can_gripper_close=gt["can_gripper_close"],
                eef_to_target_distance=gt["eef_to_target_distance"],
                move_direction=gt["move_direction"],
                action=gt.get("action"),
                gripper_state=gt["gripper_state"],
            )

            samples.append(sample)
            print(f"    EEF pos: {sample.eef_pos}")
            print(f"    Distance to target: {sample.eef_to_target_distance:.4f}m")
            print(f"    Can close: {sample.can_gripper_close}")

        env.close()

    print(f"\n=== Sampled {len(samples)} total samples ===")
    return samples


def sample_from_hf_dataset(
    dataset_name: str = "physical-intelligence/libero",
    num_samples: int = 50,
    output_dir: str = "data",
    seed: int = 42,
) -> List[BenchmarkSample]:
    """
    Sample benchmark data from HuggingFace LIBERO dataset.

    This is an alternative to running the simulation directly.

    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to extract
        output_dir: Directory to save images and ground truth
        seed: Random seed

    Returns:
        List of BenchmarkSample objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library not installed: pip install datasets")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    np.random.seed(seed)

    # Sample random indices
    total_samples = len(dataset)
    indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for i, idx in enumerate(indices):
        print(f"Processing sample {i + 1}/{len(indices)}")

        row = dataset[int(idx)]

        # Extract data
        sample_id = f"hf_{idx:06d}"
        task_id = row.get("task_index", 0)
        episode_id = row.get("episode_index", 0)
        step_id = row.get("frame_index", 0)

        # Save images
        agentview_path = images_dir / f"{sample_id}_agentview.png"
        eye_in_hand_path = images_dir / f"{sample_id}_eye_in_hand.png"

        if "image" in row:
            row["image"].save(agentview_path)
        if "wrist_image" in row:
            row["wrist_image"].save(eye_in_hand_path)

        # Extract state and action
        state = np.array(row.get("state", [0] * 8))
        action = np.array(row.get("actions", [0] * 7))

        # State format: [eef_pos(3), eef_quat(4), gripper(1)] or similar
        # This needs to be verified with actual dataset
        eef_pos = state[:3] if len(state) >= 3 else np.zeros(3)

        # Compute move direction from action
        delta_pos = action[:3]
        norm = np.linalg.norm(delta_pos)
        move_direction = delta_pos / norm if norm > 1e-6 else np.zeros(3)

        sample = BenchmarkSample(
            sample_id=sample_id,
            task_id=task_id,
            task_name=f"task_{task_id}",
            task_language="",  # Not available in HF dataset
            episode_id=episode_id,
            step_id=step_id,
            agentview_image_path=str(agentview_path),
            eye_in_hand_image_path=str(eye_in_hand_path),
            eef_pos=eef_pos,
            move_direction=move_direction,
            action=action,
        )

        samples.append(sample)

    print(f"Sampled {len(samples)} from HuggingFace dataset")
    return samples


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Sample benchmark data from LIBERO")
    parser.add_argument("--source", choices=["sim", "hf"], default="sim",
                        help="Data source: 'sim' for simulation, 'hf' for HuggingFace")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.source == "sim":
        samples = sample_benchmark_data(
            num_tasks=args.num_tasks,
            samples_per_task=args.samples_per_task,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        samples = sample_from_hf_dataset(
            num_samples=args.num_tasks * args.samples_per_task,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    # Save samples
    from .extract_gt import save_benchmark_samples
    save_benchmark_samples(samples, f"{args.output_dir}/ground_truth/benchmark_samples.json")
