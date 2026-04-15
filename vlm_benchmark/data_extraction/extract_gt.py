"""
Ground Truth extraction from LIBERO environment.

Extracts:
- Object positions in world frame
- End-effector (gripper) positions
- Grasp affordance (can gripper close?)
- Move direction from actions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class BenchmarkSample:
    """A single benchmark sample with images and ground truth."""

    # Identifiers
    sample_id: str
    task_id: int
    task_name: str
    task_language: str
    episode_id: int
    step_id: int

    # Images (paths or arrays)
    agentview_image_path: Optional[str] = None
    eye_in_hand_image_path: Optional[str] = None
    agentview_image: Optional[np.ndarray] = None
    eye_in_hand_image: Optional[np.ndarray] = None

    # Ground truth - End effector state
    eef_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eef_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Ground truth - Target object
    target_object_name: str = ""
    target_object_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_object_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Ground truth - All objects in scene
    objects_pos: Dict[str, np.ndarray] = field(default_factory=dict)

    # Ground truth - Gripper close judgment
    can_gripper_close: bool = False
    eef_to_target_distance: float = 0.0

    # Ground truth - Move direction (from action)
    move_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    action: Optional[np.ndarray] = None

    # Metadata
    gripper_state: float = 0.0  # gripper qpos

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_language": self.task_language,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "agentview_image_path": self.agentview_image_path,
            "eye_in_hand_image_path": self.eye_in_hand_image_path,
            "eef_pos": self.eef_pos.tolist(),
            "eef_quat": self.eef_quat.tolist(),
            "target_object_name": self.target_object_name,
            "target_object_pos": self.target_object_pos.tolist(),
            "target_object_quat": self.target_object_quat.tolist(),
            "objects_pos": {k: v.tolist() for k, v in self.objects_pos.items()},
            "can_gripper_close": self.can_gripper_close,
            "eef_to_target_distance": self.eef_to_target_distance,
            "move_direction": self.move_direction.tolist(),
            "action": self.action.tolist() if self.action is not None else None,
            "gripper_state": self.gripper_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkSample":
        """Create from dictionary."""
        sample = cls(
            sample_id=data["sample_id"],
            task_id=data["task_id"],
            task_name=data["task_name"],
            task_language=data["task_language"],
            episode_id=data["episode_id"],
            step_id=data["step_id"],
        )
        sample.agentview_image_path = data.get("agentview_image_path")
        sample.eye_in_hand_image_path = data.get("eye_in_hand_image_path")
        sample.eef_pos = np.array(data["eef_pos"])
        sample.eef_quat = np.array(data["eef_quat"])
        sample.target_object_name = data["target_object_name"]
        sample.target_object_pos = np.array(data["target_object_pos"])
        sample.target_object_quat = np.array(data["target_object_quat"])
        sample.objects_pos = {k: np.array(v) for k, v in data.get("objects_pos", {}).items()}
        sample.can_gripper_close = data["can_gripper_close"]
        sample.eef_to_target_distance = data["eef_to_target_distance"]
        sample.move_direction = np.array(data["move_direction"])
        sample.action = np.array(data["action"]) if data.get("action") is not None else None
        sample.gripper_state = data.get("gripper_state", 0.0)
        return sample


def extract_ground_truth(
    env,
    obs: dict,
    target_object_name: str,
    action: Optional[np.ndarray] = None,
    grasp_distance_threshold: float = 0.045,
) -> dict:
    """
    Extract ground truth data from LIBERO environment.

    Args:
        env: LIBERO OffScreenRenderEnv instance
        obs: Observation dictionary from env.step() or env.reset()
        target_object_name: Name of the target object to grasp
        action: Action array (7D OSC_POSE: dx,dy,dz,drx,dry,drz,gripper)
        grasp_distance_threshold: Distance threshold for can_gripper_close (meters)

    Returns:
        Dictionary with ground truth data
    """
    gt = {}

    # 1. End-effector state
    gt["eef_pos"] = obs["robot0_eef_pos"].copy()
    gt["eef_quat"] = obs["robot0_eef_quat"].copy()
    gt["gripper_state"] = obs.get("robot0_gripper_qpos", np.array([0.0]))[0]

    # 2. All object positions
    objects_pos = {}
    objects_quat = {}

    # Access inner environment for object data
    inner_env = env.env if hasattr(env, 'env') else env

    if hasattr(inner_env, 'objects_dict') and hasattr(inner_env, 'obj_body_id'):
        for obj_name in inner_env.objects_dict.keys():
            try:
                body_id = inner_env.obj_body_id[obj_name]
                objects_pos[obj_name] = env.sim.data.body_xpos[body_id].copy()
                objects_quat[obj_name] = env.sim.data.body_xquat[body_id].copy()
            except (KeyError, IndexError):
                continue

    gt["objects_pos"] = objects_pos
    gt["objects_quat"] = objects_quat

    # 3. Target object position
    if target_object_name in objects_pos:
        gt["target_object_pos"] = objects_pos[target_object_name]
        gt["target_object_quat"] = objects_quat[target_object_name]
    else:
        # Try to find object with similar name
        for obj_name in objects_pos.keys():
            if target_object_name.lower() in obj_name.lower():
                gt["target_object_pos"] = objects_pos[obj_name]
                gt["target_object_quat"] = objects_quat[obj_name]
                gt["target_object_name_matched"] = obj_name
                break
        else:
            gt["target_object_pos"] = np.zeros(3)
            gt["target_object_quat"] = np.array([1, 0, 0, 0])

    # 4. Gripper close judgment
    eef_pos = gt["eef_pos"]
    target_pos = gt.get("target_object_pos", np.zeros(3))
    distance = np.linalg.norm(eef_pos - target_pos)

    gt["eef_to_target_distance"] = distance
    gt["can_gripper_close"] = distance < grasp_distance_threshold

    # 5. Move direction from action
    if action is not None:
        delta_pos = action[:3]
        norm = np.linalg.norm(delta_pos)
        if norm > 1e-6:
            gt["move_direction"] = delta_pos / norm
        else:
            gt["move_direction"] = np.zeros(3)
        gt["action"] = action.copy()
    else:
        # Compute direction from eef to target
        direction = target_pos - eef_pos
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            gt["move_direction"] = direction / norm
        else:
            gt["move_direction"] = np.zeros(3)
        gt["action"] = None

    return gt


def save_benchmark_samples(samples: List[BenchmarkSample], output_path: str):
    """Save benchmark samples to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [s.to_dict() for s in samples]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def load_benchmark_samples(input_path: str) -> List[BenchmarkSample]:
    """Load benchmark samples from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    samples = [BenchmarkSample.from_dict(d) for d in data]
    print(f"Loaded {len(samples)} samples from {input_path}")
    return samples
