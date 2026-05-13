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
    compute_gripper_openness,
    compute_gripper_to_target_delta,
    compute_next_direction,
    compute_orientation,
    compute_pairwise_distance,
    compute_relative_rotation,
    compute_spatial_relation,
    euler_deg_to_quat_wxyz,
    select_object_pair,
)
from .calvin_taxonomy import (
    BLOCK_EULER_INDEX,
    ROBOT_BASE_POS_WORLD,
    TASK_CLASS_TO_TARGET,
    all_object_positions_calvin,
    find_target_calvin,
)


def extract_gt_from_calvin(
    robot_obs: np.ndarray,
    scene_obs: np.ndarray,
    task_class: str,
    task_description: str,
) -> dict | None:
    """Build the same `gt` dict shape that LIBERO produces (Q1-Q6 + Q7-Q11).

    Returns None if the task class is unknown or in the deny-list.
    """
    resolved = find_target_calvin(task_class, scene_obs)
    if resolved is None:
        return None
    target_name, target_pos, task_type, dest_pos = resolved
    eef_pos = np.asarray(robot_obs[0:3], dtype=np.float32)

    distance = float(np.linalg.norm(eef_pos - target_pos))

    dest_object_name = None if dest_pos is None else "receptacle"
    all_objects = all_object_positions_calvin(scene_obs)

    # --- Q7: EE orientation ---
    # CALVIN robot_obs[3:6] = tcp_orn (euler, radians)
    eef_euler_deg = np.degrees(np.asarray(robot_obs[3:6], dtype=np.float64)).tolist()
    eef_quat_wxyz = euler_deg_to_quat_wxyz(eef_euler_deg)
    eef_orn = {"quat": eef_quat_wxyz, "euler_deg": eef_euler_deg}

    # --- Q8: Target object orientation ---
    entry = TASK_CLASS_TO_TARGET.get(task_class)
    if entry is not None and entry["target_kind"] == "block" and target_name in BLOCK_EULER_INDEX:
        obj_euler_deg = np.degrees(
            np.asarray(scene_obs[BLOCK_EULER_INDEX[target_name]], dtype=np.float64)
        ).tolist()
        obj_quat_wxyz = euler_deg_to_quat_wxyz(obj_euler_deg)
        target_orn = {"quat": obj_quat_wxyz, "euler_deg": obj_euler_deg}
    else:
        # Articulated objects: fixed orientation, use identity quaternion
        target_orn = {"quat": [1.0, 0.0, 0.0, 0.0], "euler_deg": [0.0, 0.0, 0.0]}

    # --- Q9: Relative rotation ---
    rel_rot = compute_relative_rotation(
        np.array(eef_orn["quat"]), np.array(target_orn["quat"])
    )

    # --- Q10: Pairwise distance ---
    pair = select_object_pair(all_objects, target_name, dest_object_name)
    pairwise = compute_pairwise_distance(all_objects, *pair) if pair else None

    # --- Q11: Gripper openness ---
    # CALVIN robot_obs[6] = gripper_width (open ≈ 0.08, closed ≈ 0.0)
    openness = compute_gripper_openness(float(robot_obs[6]), min_val=0.0, max_val=0.08)

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
        "dest_object_name": dest_object_name,
        "dest_pos": None if dest_pos is None else dest_pos.tolist(),
        "all_objects": all_objects,
        "can_close": compute_can_close(eef_pos, target_pos),
        "next_direction": compute_next_direction(eef_pos, target_pos),
        "distance_to_target": distance,
        "gripper_to_target_delta": compute_gripper_to_target_delta(eef_pos, target_pos),
        "gripper_to_target_relation": compute_spatial_relation(eef_pos, target_pos),
        # Q7
        "eef_orientation_quat": eef_orn["quat"],
        "eef_orientation_euler_deg": eef_orn["euler_deg"],
        # Q8
        "target_orientation_quat": target_orn["quat"],
        "target_orientation_euler_deg": target_orn["euler_deg"],
        # Q9
        "relative_rotation_quat": rel_rot["quat"],
        "relative_rotation_euler_deg": rel_rot["euler_deg"],
        # Q10
        "pairwise_distance": pairwise,
        # Q11
        "gripper_openness": openness,
    }
