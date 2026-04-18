"""Ground truth extractor for LIBERO environments.

Extracts object positions, EE position, can-close label, and next-direction
from a MuJoCo simulation state, all expressed relative to the robot base frame.

Coordinate frame (robot base = origin):
  X : forward (into the table workspace, away from the robot)
  Y : left    (robot's left when facing forward)
  Z : up
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ----- Constants -------------------------------------------------------

# Distance threshold (m) for "can gripper close to grasp target object".
# Panda gripper half-width when open ≈ 0.04 m.
CLOSE_THRESHOLD = 0.04

# Body name fragments that belong to the robot or the environment,
# NOT to manipulable scene objects.
_ROBOT_KEYWORDS: frozenset[str] = frozenset(
    {
        "robot", "gripper", "finger", "link", "joint", "actuator", "motor",
        "thumb", "panda", "franka", "eef", "wrist",
    }
)
_ENV_KEYWORDS: frozenset[str] = frozenset(
    {
        "table", "floor", "ground", "ceiling", "wall", "world", "scene",
        "light", "fixture", "support", "stand", "camera", "pedestal",
    }
)
# Body-name suffixes that indicate a pure visual / collision sub-body.
_GEOM_SUFFIXES: frozenset[str] = frozenset(
    {"collision", "visual", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"}
)


# ----- Utilities -------------------------------------------------------

def is_scene_object_body(name: str) -> bool:
    """Return True if a MuJoCo body name belongs to a manipulable scene object."""
    lower = name.lower()
    if not lower or lower == "world":
        return False
    if any(kw in lower for kw in _ROBOT_KEYWORDS | _ENV_KEYWORDS):
        return False
    if any(lower.endswith(f"_{suf}") or lower == suf for suf in _GEOM_SUFFIXES):
        return False
    return True


def get_sim_from_env(env: Any):
    """Retrieve the MuJoCo ``sim`` object through various wrapper depths.

    LIBERO's ``OffScreenRenderEnv`` wraps a robosuite environment.
    The robosuite env keeps ``sim`` as a direct attribute; LIBERO may add
    one or two extra wrapper levels.
    """
    for candidate in (env, getattr(env, "env", None), getattr(getattr(env, "env", None), "env", None)):
        if candidate is not None and hasattr(candidate, "sim"):
            return candidate.sim
    raise AttributeError(
        f"Cannot locate 'sim' attribute in env of type {type(env).__name__}. "
        "Check that the underlying OffScreenRenderEnv exposes .sim."
    )


def get_robot_base_pos(sim) -> np.ndarray:
    """Return the world-frame position of the robot base body."""
    for candidate_name in ("robot0_base", "base", "robot_base", "Panda_link0"):
        try:
            return sim.data.get_body_xpos(candidate_name).copy()
        except Exception:
            continue
    # Fallback: assume world origin
    return np.zeros(3)


def get_all_object_positions(sim, base_pos: np.ndarray) -> dict[str, list[float]]:
    """Return {body_name: [x,y,z]} for all scene-object bodies, relative to base."""
    objects: dict[str, list[float]] = {}
    for name in sim.model.body_names:
        if is_scene_object_body(name):
            pos_world = sim.data.get_body_xpos(name).copy()
            objects[name] = (pos_world - base_pos).tolist()
    return objects


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------

# Verbs that indicate articulation tasks (no separate destination object).
_ARTICULATION_VERBS = frozenset({
    "open", "close", "turn on", "turn off", "push", "pull",
    "press", "switch", "rotate", "twist", "toggle",
})

# Prepositions / phrases that separate source from destination in pick-and-place.
_PLACE_PREPOSITIONS = (
    " and place it in ", " and place it on ", " and place it into ",
    " and put it in ", " and put it on ", " and put it into ",
    " into ", " onto ", " in the ", " on the ",
)


def classify_task(task_description: str) -> str:
    """Return 'pick_and_place' or 'articulation' based on the task description."""
    desc_lower = task_description.lower()
    if any(desc_lower.startswith(v) or f" {v} " in desc_lower for v in _ARTICULATION_VERBS):
        return "articulation"
    return "pick_and_place"


def _score_objects(
    object_positions: dict[str, list[float]],
    phrase: str,
) -> dict[str, int]:
    """Score each object body name by keyword overlap with a text phrase."""
    scores: dict[str, int] = {}
    for name in object_positions:
        parts = [p for p in name.lower().replace("_", " ").split() if len(p) > 2]
        scores[name] = sum(1 for p in parts if p in phrase.lower())
    return scores


def find_target_object(
    object_positions: dict[str, list[float]],
    task_description: str,
) -> tuple[str, np.ndarray]:
    """Identify the SOURCE object (thing to pick up / articulate).

    Returns (body_name, position_array). Falls back to the first object
    if no keyword match is found.
    """
    if not object_positions:
        return "unknown", np.array([0.20, 0.0, 0.0])

    # For pick-and-place, score only against the part before the preposition.
    source_phrase = task_description
    for prep in _PLACE_PREPOSITIONS:
        if prep in task_description.lower():
            idx = task_description.lower().index(prep)
            source_phrase = task_description[:idx]
            break

    scores = _score_objects(object_positions, source_phrase)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = next(iter(object_positions))

    return best, np.array(object_positions[best])


def find_destination_object(
    object_positions: dict[str, list[float]],
    task_description: str,
    source_name: str,
) -> tuple[str, np.ndarray] | tuple[None, None]:
    """Identify the DESTINATION object for pick-and-place tasks.

    Looks at the part of the task description after the place-preposition and
    scores body names against that phrase. Excludes the source object.

    Returns (body_name, position_array) or (None, None) for articulation tasks.
    """
    if classify_task(task_description) == "articulation":
        return None, None

    dest_phrase = ""
    for prep in _PLACE_PREPOSITIONS:
        if prep in task_description.lower():
            idx = task_description.lower().index(prep) + len(prep)
            dest_phrase = task_description[idx:]
            break

    if not dest_phrase:
        return None, None

    # Exclude source object from candidates
    candidates = {k: v for k, v in object_positions.items() if k != source_name}
    if not candidates:
        return None, None

    scores = _score_objects(candidates, dest_phrase)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None, None

    return best, np.array(candidates[best])


# ----- Core extraction -------------------------------------------------

def compute_can_close(eef_pos: np.ndarray, target_pos: np.ndarray) -> bool:
    """True if gripper is within closing distance of the target object."""
    return bool(np.linalg.norm(eef_pos - target_pos) < CLOSE_THRESHOLD)


def compute_next_direction(eef_pos: np.ndarray, target_pos: np.ndarray) -> list[float]:
    """Unit vector from current EE toward target object center."""
    delta = target_pos - eef_pos
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return [1.0, 0.0, 0.0]
    return (delta / norm).tolist()


def extract_gt_from_obs(
    raw_obs: dict,
    sim,
    task_description: str,
) -> dict:
    """Extract all ground truth fields from a raw LIBERO observation + sim state.

    Args:
        raw_obs: Dict returned by ``OffScreenRenderEnv.step()`` / ``.reset()``.
        sim:     The MuJoCo ``sim`` object (e.g. ``env.sim``).
        task_description: Natural-language task string used to identify
                          the target object.

    Returns:
        Dict with keys:
          eef_pos           – [x, y, z] in robot-base frame (m)
          base_pos_world    – [x, y, z] of robot base in world frame (m)
          target_object_name – str, matched MuJoCo body name
          target_pos        – [x, y, z] of target in robot-base frame (m)
          all_objects       – {name: [x, y, z]} for all scene objects
          can_close         – bool
          next_direction    – [dx, dy, dz] unit vector toward target
          distance_to_target – float (m)
    """
    base_pos = get_robot_base_pos(sim)
    eef_pos = np.array(raw_obs["robot0_eef_pos"]) - base_pos
    all_objects = get_all_object_positions(sim, base_pos)
    task_type = classify_task(task_description)
    target_name, target_pos = find_target_object(all_objects, task_description)
    dest_name, dest_pos = find_destination_object(all_objects, task_description, target_name)

    return {
        "task_type": task_type,
        "eef_pos": eef_pos.tolist(),
        "base_pos_world": base_pos.tolist(),
        "target_object_name": target_name,
        "target_pos": target_pos.tolist(),
        "dest_object_name": dest_name,
        "dest_pos": dest_pos.tolist() if dest_pos is not None else None,
        "all_objects": all_objects,
        "can_close": compute_can_close(eef_pos, target_pos),
        "next_direction": compute_next_direction(eef_pos, target_pos),
        "distance_to_target": float(np.linalg.norm(eef_pos - target_pos)),
    }


def get_agentview_image(raw_obs: dict) -> np.ndarray:
    """Return the agent-view RGB image, flipped to match display convention.

    LIBERO stores images upside-down in the raw obs; flipping both H and W
    gives the natural top-up view used throughout LeRobot and beta-vla.
    """
    img = raw_obs["agentview_image"]
    return img[::-1, ::-1].copy()
