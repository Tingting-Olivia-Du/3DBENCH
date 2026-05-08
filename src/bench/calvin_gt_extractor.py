"""Compute the 6-dimensional spatial QA ground truth for a CALVIN frame.

Mirrors `src/bench/gt_extractor.py:extract_gt_from_obs(...)` which is used for
LIBERO. The geometric helpers (`compute_can_close`, `compute_next_direction`,
`compute_gripper_to_target_delta`, `compute_spatial_relation`) are imported
unchanged — they only see a delta vector and don't care about the source
coordinate frame.

We keep CALVIN coordinates in the world frame (no rotation into a libero-like
base-relative frame) because:
  - the X/Y axes don't align with LIBERO's anyway;
  - the Calvin-specific prompt (prompts/spatial_qa_calvin.yaml) explicitly
    documents the world frame to the model.
"""
from __future__ import annotations

import numpy as np

from .gt_extractor import (
    CLOSE_THRESHOLD,
    compute_can_close,
    compute_gripper_to_target_delta,
    compute_next_direction,
    compute_spatial_relation,
)
from .calvin_taxonomy import (
    ROBOT_BASE_POS_WORLD,
    all_object_positions_calvin,
    find_target_calvin,
)


def extract_gt_from_calvin(
    robot_obs: np.ndarray,
    scene_obs: np.ndarray,
    task_class: str,
    task_description: str,
) -> dict | None:
    """Build the same `gt` dict shape that LIBERO produces.

    Returns None if the task class is unknown or in the deny-list.
    """
    resolved = find_target_calvin(task_class, scene_obs)
    if resolved is None:
        return None
    target_name, target_pos, task_type, dest_pos = resolved
    eef_pos = np.asarray(robot_obs[0:3], dtype=np.float32)

    distance = float(np.linalg.norm(eef_pos - target_pos))

    return {
        "task_type": task_type,
        "eef_pos": eef_pos.tolist(),
        # CALVIN reports robot_obs and scene_obs in world frame. We surface the
        # real Panda base (sourced from validation/.hydra/merged_config.yaml →
        # scene.robot_base_position) so downstream consumers can convert world →
        # base-relative by subtracting it. The `eef_pos`/`target_pos` etc. above
        # remain in world frame to avoid silently rewriting GT values.
        "base_pos_world": ROBOT_BASE_POS_WORLD.tolist(),
        "target_object_name": target_name,
        "target_pos": target_pos.tolist(),
        "dest_object_name": None if dest_pos is None else "receptacle",
        "dest_pos": None if dest_pos is None else dest_pos.tolist(),
        "all_objects": all_object_positions_calvin(scene_obs),
        "can_close": compute_can_close(eef_pos, target_pos),
        "next_direction": compute_next_direction(eef_pos, target_pos),
        "distance_to_target": distance,
        "gripper_to_target_delta": compute_gripper_to_target_delta(eef_pos, target_pos),
        "gripper_to_target_relation": compute_spatial_relation(eef_pos, target_pos),
    }
