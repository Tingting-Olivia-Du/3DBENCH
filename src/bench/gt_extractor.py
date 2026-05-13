"""Ground truth extractor for LIBERO environments.

Extracts object positions, EE position/orientation, can-close label,
next-direction, and additional spatial GT (Q7-Q11) from a MuJoCo simulation
state, all expressed relative to the robot base frame.

Coordinate frame (robot base = origin):
  X : forward (into the table workspace, away from the robot)
  Y : right   (robot's right when facing forward)
  Z : up
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# ----- Constants -------------------------------------------------------

# Distance threshold (m) for "can gripper close to grasp target object".
# Panda gripper half-width when open ≈ 0.04 m.
CLOSE_THRESHOLD = 0.04

# Alignment threshold (m) for spatial relation classification (Q6).
# Differences smaller than this are considered "aligned" on that axis.
ALIGN_THRESHOLD = 0.02

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


# ----- Quaternion / Euler helpers -----------------------------------------
# Convention: all quaternions stored as scalar-first [w, x, y, z].
# All Euler angles are XYZ intrinsic (roll, pitch, yaw) in degrees.

def quat_wxyz_from_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert robosuite [x,y,z,w] quaternion to standard [w,x,y,z]."""
    q = np.asarray(q_xyzw, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_euler_deg(q_wxyz: np.ndarray) -> list[float]:
    """Convert [w,x,y,z] quaternion to [roll, pitch, yaw] in degrees (XYZ intrinsic)."""
    q = np.asarray(q_wxyz, dtype=float)
    # scipy expects [x, y, z, w]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q_xyzw).as_euler('xyz', degrees=True).tolist()


def euler_deg_to_quat_wxyz(euler_deg) -> list[float]:
    """Convert [roll, pitch, yaw] in degrees (XYZ intrinsic) to [w,x,y,z] quaternion."""
    r = Rotation.from_euler('xyz', euler_deg, degrees=True)
    q_xyzw = r.as_quat()  # scipy returns [x, y, z, w]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def compute_orientation(quat_wxyz: np.ndarray) -> dict:
    """Return {"quat": [w,x,y,z], "euler_deg": [roll,pitch,yaw]} from a quaternion."""
    q = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        q = q / norm
    return {
        "quat": q.tolist(),
        "euler_deg": quat_to_euler_deg(q),
    }


def compute_relative_rotation(
    eef_quat_wxyz: np.ndarray,
    obj_quat_wxyz: np.ndarray,
) -> dict:
    """Compute relative rotation: q_rel = conj(q_eef) * q_obj.

    This represents the object's orientation expressed in the EE frame,
    i.e. the rotation the EE must undergo to align with the object.

    Returns {"quat": [w,x,y,z], "euler_deg": [roll,pitch,yaw]}.
    """
    # Convert to scipy [x,y,z,w] format
    def _to_xyzw(q_wxyz):
        q = np.asarray(q_wxyz, dtype=float)
        return np.array([q[1], q[2], q[3], q[0]])

    r_eef = Rotation.from_quat(_to_xyzw(eef_quat_wxyz))
    r_obj = Rotation.from_quat(_to_xyzw(obj_quat_wxyz))
    r_rel = r_eef.inv() * r_obj

    q_rel_xyzw = r_rel.as_quat()  # [x, y, z, w]
    q_rel_wxyz = [float(q_rel_xyzw[3]), float(q_rel_xyzw[0]),
                  float(q_rel_xyzw[1]), float(q_rel_xyzw[2])]
    euler_deg = r_rel.as_euler('xyz', degrees=True).tolist()

    return {"quat": q_rel_wxyz, "euler_deg": euler_deg}


def _object_base_name(name: str) -> str:
    """Strip part suffixes to get the base object identity.

    E.g. 'flat_stove_1_base' and 'flat_stove_1_button' both share the
    base name 'flat_stove_1'.  This lets us detect sub-parts of the same
    composite object so we don't pair them in Q10.
    """
    _PART_SUFFIXES = {
        "main", "base", "button", "handle", "lid", "knob", "door",
        "link", "link0", "link1", "link2", "hinge", "joint", "body",
    }
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in _PART_SUFFIXES:
        return parts[0]
    return name


def select_object_pair(
    all_objects: dict[str, list[float]],
    target_name: str,
    dest_name: str | None = None,
) -> tuple[str, str] | None:
    """Select the closest pair of non-target, non-destination objects.

    Excludes:
      - The target and destination objects
      - mount/base-like robot bodies
      - Pairs whose distance is 0 (co-located sub-bodies)
      - Pairs that share the same base object name (e.g. stove_base + stove_button)

    Returns (name_a, name_b) with name_a < name_b lexicographically,
    or None if fewer than 2 candidate objects exist.
    """
    exclude = {target_name}
    if dest_name:
        exclude.add(dest_name)
    # Also exclude mount/base-like bodies that leak through
    candidates = {
        name: pos for name, pos in all_objects.items()
        if name not in exclude and "mount" not in name.lower()
    }
    if len(candidates) < 2:
        return None

    best_pair: tuple[str, str] | None = None
    best_dist = float("inf")
    for (a, pos_a), (b, pos_b) in combinations(candidates.items(), 2):
        # Skip sub-parts of the same composite object
        if _object_base_name(a) == _object_base_name(b):
            continue
        d = float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))
        # Skip zero-distance pairs (co-located sub-bodies)
        if d < 1e-6:
            continue
        # Tie-break by lexicographic order
        pair = (min(a, b), max(a, b))
        if d < best_dist or (d == best_dist and (best_pair is None or pair < best_pair)):
            best_dist = d
            best_pair = pair
    return best_pair


def compute_pairwise_distance(
    all_objects: dict[str, list[float]],
    object_a: str,
    object_b: str,
) -> dict:
    """Compute Euclidean distance between two named objects.

    Returns {"object_a", "object_b", "distance_m", "pos_a", "pos_b"}.
    """
    pos_a = all_objects[object_a]
    pos_b = all_objects[object_b]
    dist = float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))
    return {
        "object_a": object_a,
        "object_b": object_b,
        "distance_m": dist,
        "pos_a": list(pos_a),
        "pos_b": list(pos_b),
    }


def compute_gripper_openness(
    raw_value: float,
    min_val: float = 0.0,
    max_val: float = 0.04,
) -> float:
    """Normalize a raw gripper reading to [0.0, 1.0].

    0.0 = fully closed, 1.0 = fully open.
    """
    if max_val <= min_val:
        return 0.0
    return float(np.clip((raw_value - min_val) / (max_val - min_val), 0.0, 1.0))


def compute_gripper_phases(
    actions: np.ndarray,
    min_hold_steps: int = 30,
    window: int = 5,
    **kwargs,
) -> list[str]:
    """Compute gripper phase for every timestep in a demo trajectory.

    Uses the full actions array (T, 7) and close-segment duration to
    distinguish successful grasps from failed attempts.

    Phases:
      "approaching"  – gripper open command, moving toward target
      "grasping"     – in a close-segment that is too short to be a real
                       grasp (failed attempt, < min_hold_steps)
      "carrying"     – in a close-segment long enough to be a successful
                       grasp (≥ min_hold_steps)
      "releasing"    – transition region around close→open (for successful
                       grasps only)

    Args:
        actions:          (T, 7) array of demo actions.
        min_hold_steps:   Minimum close-segment length to count as a
                          successful grasp. Default 30 (≈ 1.5s at 20Hz).
        window:           Half-width for releasing transition zones.

    Returns:
        List of T phase strings, one per timestep.
    """
    T = len(actions)
    gripper_cmds = actions[:, 6]  # -1 = open, +1 = close

    # Identify close-segments
    segments: list[tuple[int, int, bool]] = []  # (start, end, is_success)
    in_close = False
    start = 0
    for t in range(T):
        if gripper_cmds[t] > 0 and not in_close:
            start = t
            in_close = True
        elif gripper_cmds[t] < 0 and in_close:
            success = (t - start) >= min_hold_steps
            segments.append((start, t, success))
            in_close = False
    if in_close:
        success = (T - start) >= min_hold_steps
        segments.append((start, T, success))

    # Build release zones (only for successful grasps)
    release_zone: set[int] = set()
    for seg_start, seg_end, is_success in segments:
        if is_success and seg_end < T:
            for tt in range(max(0, seg_end - window), min(T, seg_end + window)):
                release_zone.add(tt)

    # Assign phases
    # First, map each timestep to its segment info
    seg_map: dict[int, tuple[int, int, bool]] = {}
    for seg_start, seg_end, is_success in segments:
        for t in range(seg_start, seg_end):
            seg_map[t] = (seg_start, seg_end, is_success)

    phases = [""] * T
    for t in range(T):
        if t in release_zone:
            phases[t] = "releasing"
        elif gripper_cmds[t] < 0:
            phases[t] = "approaching"
        elif t in seg_map:
            _, _, is_success = seg_map[t]
            phases[t] = "carrying" if is_success else "grasping"
        else:
            phases[t] = "approaching"

    return phases


def compute_grasp_success(
    actions: np.ndarray,
    min_hold_steps: int = 30,
    **kwargs,
) -> list[bool | None]:
    """For each timestep, determine if the current grasp attempt succeeds.

    A successful grasp is a close-segment (contiguous run of action[6] > 0)
    that lasts at least ``min_hold_steps`` steps. Short close-segments are
    failed attempts where the operator tried to grasp but quickly released.

    At 20 Hz control frequency, 30 steps ≈ 1.5 seconds — a reasonable
    minimum for pick-up + transport.

    Returns a list of T values:
      - True:  this close-segment is long enough → successful grasp
      - False: this close-segment is too short → failed attempt
      - None:  not in a close-segment (action[6] < 0)
    """
    T = len(actions)
    gripper_cmds = actions[:, 6]

    # Identify close-segments: contiguous runs of action[6] > 0
    segments: list[tuple[int, int]] = []  # (start, end_exclusive)
    in_close = False
    start = 0
    for t in range(T):
        if gripper_cmds[t] > 0 and not in_close:
            start = t
            in_close = True
        elif gripper_cmds[t] < 0 and in_close:
            segments.append((start, t))
            in_close = False
    if in_close:
        segments.append((start, T))

    result: list[bool | None] = [None] * T
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        success = seg_len >= min_hold_steps
        for t in range(seg_start, seg_end):
            result[t] = success

    return result


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


# ----- Multi-target support --------------------------------------------

import re as _re


def find_all_target_objects(
    object_positions: dict[str, list[float]],
    task_description: str,
) -> list[tuple[str, np.ndarray]]:
    """Parse multi-step task descriptions and return ALL target objects.

    Handles patterns like:
      - "put both the X and the Y in the Z"
      - "put the X on the A and put the Y on the B"
      - "turn on the stove and put the X on it"
      - "put the X in the Y and close it"

    Returns a list of (body_name, position_array) for each target found.
    Falls back to the single-target result from find_target_object().
    """
    desc = task_description.lower()

    # Collect candidate phrases for target objects
    target_phrases: list[str] = []

    # Pattern 1: "put both the X and the Y in/on the Z"
    m = _re.match(r"put both the (.+?) and the (.+?) (?:in|on|into) the ", desc)
    if m:
        target_phrases = [m.group(1), m.group(2)]

    # Pattern 2: "put the X on/in the A and put the Y on/in the B"
    if not target_phrases:
        m = _re.match(
            r"put the (.+?) (?:on|in|into) the .+? and put the (.+?) (?:on|in|into|to) ",
            desc,
        )
        if m:
            target_phrases = [m.group(1), m.group(2)]

    # Pattern 3: "verb the X and verb the Y ..." (e.g. "turn on the stove and put the moka pot on it")
    if not target_phrases:
        # Split on " and " and try to extract an object from each sub-clause
        clauses = desc.split(" and ")
        for clause in clauses:
            # Try to find "the <object>" after a verb
            m2 = _re.search(r"(?:put|pick up|place|turn on|turn off|open|close) the (.+?)(?:\s+(?:in|on|into|to|from)\s|$)", clause)
            if m2:
                target_phrases.append(m2.group(1).strip())

    if not target_phrases:
        # Single-target fallback
        name, pos = find_target_object(object_positions, task_description)
        return [(name, pos)]

    # Match each phrase to the best-scoring object body
    results: list[tuple[str, np.ndarray]] = []
    used_names: set[str] = set()
    for phrase in target_phrases:
        scores = _score_objects(object_positions, phrase)
        # Exclude already-matched objects to avoid duplicates
        candidates = {k: v for k, v in scores.items() if k not in used_names}
        if not candidates:
            continue
        best = max(candidates, key=candidates.get)
        if candidates[best] == 0:
            continue
        used_names.add(best)
        results.append((best, np.array(object_positions[best])))

    if not results:
        name, pos = find_target_object(object_positions, task_description)
        return [(name, pos)]

    return results


def find_active_target(
    eef_pos: np.ndarray,
    targets: list[tuple[str, np.ndarray]],
    dest_pos: np.ndarray | None = None,
    placed_threshold: float = 0.08,
) -> tuple[str, np.ndarray]:
    """Pick the active target following task-description order.

    For multi-target tasks (e.g. "put both X and Y in the basket"),
    the demo always operates on targets in the order they appear in
    the language instruction.  A target is considered "placed" (done)
    when it is close to the destination (< placed_threshold meters).
    We then advance to the next target in sequence.

    Args:
        eef_pos:           Current gripper position (robot-base frame).
        targets:           Ordered list of (name, pos) from find_all_target_objects
                           (already in language-instruction order).
        dest_pos:          Destination position (e.g. basket center). If None,
                           falls back to the first (language-order) target.
        placed_threshold:  Max distance from destination to count as "placed".

    Returns:
        (active_target_name, active_target_pos)
    """
    if len(targets) == 1:
        return targets[0]

    if dest_pos is not None:
        # Walk through targets in order; skip any already placed at destination
        for name, pos in targets:
            dist_to_dest = float(np.linalg.norm(pos - dest_pos))
            if dist_to_dest > placed_threshold:
                # This target hasn't been placed yet → it's the active one
                return name, pos

        # All targets placed → return the last one
        return targets[-1]

    # No destination info → default to first target in language order
    return targets[0]


def compute_targets_info(
    eef_pos: np.ndarray,
    targets: list[tuple[str, np.ndarray]],
    active_name: str,
) -> list[dict]:
    """Compute distance info for all target objects.

    Returns a list of dicts, one per target, each with:
      name, pos, distance, is_active, can_close
    """
    info = []
    for name, pos in targets:
        dist = float(np.linalg.norm(eef_pos - pos))
        info.append({
            "name": name,
            "pos": pos.tolist(),
            "distance": dist,
            "is_active": name == active_name,
        })
    return info


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


def compute_gripper_to_target_delta(eef_pos: np.ndarray, target_pos: np.ndarray) -> list[float]:
    """3D offset vector from gripper to target: target_pos - eef_pos.

    Positive dx means target is further forward (X+) than gripper.
    Positive dy means target is further right (Y+) than gripper.
    Positive dz means target is higher (Z+) than gripper.
    """
    return (target_pos - eef_pos).tolist()


def compute_spatial_relation(
    eef_pos: np.ndarray,
    target_pos: np.ndarray,
    threshold: float = ALIGN_THRESHOLD,
) -> dict[str, str]:
    """Classify the axis-wise spatial relationship of target relative to gripper.

    Returns a dict with keys "x", "y", "z", each taking one of three labels:
      X: "in_front" (target further forward) | "behind" | "aligned_x"
      Y: "right"    (target to robot's right) | "left"  | "aligned_y"
      Z: "above"    (target higher)           | "below"  | "aligned_z"

    The perspective is "where is the TARGET relative to the GRIPPER".
    delta = target_pos - eef_pos; positive delta_x means target is in front of gripper.
    """
    delta = target_pos - eef_pos

    if delta[0] > threshold:
        x_rel = "in_front"
    elif delta[0] < -threshold:
        x_rel = "behind"
    else:
        x_rel = "aligned_x"

    # Y+ = robot's right; positive delta_y means target is to the right of gripper
    if delta[1] > threshold:
        y_rel = "right"
    elif delta[1] < -threshold:
        y_rel = "left"
    else:
        y_rel = "aligned_y"

    if delta[2] > threshold:
        z_rel = "above"
    elif delta[2] < -threshold:
        z_rel = "below"
    else:
        z_rel = "aligned_z"

    return {"x": x_rel, "y": y_rel, "z": z_rel}


def _compute_gripper_openness_from_sim(sim, raw_obs: dict) -> float:
    """Read gripper openness directly from MuJoCo sim state.

    Tries to find Panda finger joints in sim.data.qpos first (reliable after
    set_state_from_flattened), then falls back to raw_obs['robot0_gripper_qpos'].

    Returns float in [0, 1]: 0 = closed, 1 = fully open.
    """
    # Panda finger joint names in robosuite/MuJoCo
    _FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2",
                      "finger_joint1", "finger_joint2")
    _PANDA_OPEN = 0.04  # max finger joint value when fully open

    # Try reading from sim.data.qpos via joint name
    finger_vals = []
    for jname in _FINGER_JOINTS:
        try:
            jid = sim.model.joint_name2id(jname)
            addr = sim.model.jnt_qposadr[jid]
            finger_vals.append(abs(float(sim.data.qpos[addr])))
        except Exception:
            continue

    if finger_vals:
        raw = float(np.mean(finger_vals))
        return compute_gripper_openness(raw, min_val=0.0, max_val=_PANDA_OPEN)

    # Fallback to robosuite observable
    gripper_qpos = raw_obs.get("robot0_gripper_qpos")
    if gripper_qpos is not None:
        return compute_gripper_openness(float(np.mean(gripper_qpos)),
                                        min_val=0.0, max_val=_PANDA_OPEN)
    return 0.0


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
        Dict with keys (Q1-Q6 legacy + Q7-Q11 new + multi-target):
          eef_pos                      – [x, y, z] in robot-base frame (m)
          base_pos_world               – [x, y, z] of robot base in world frame (m)
          target_object_name           – str, *active* target (closest to gripper)
          target_pos                   – [x, y, z] of active target in robot-base frame (m)
          all_objects                  – {name: [x, y, z]} for all scene objects
          can_close                    – bool (against active target)
          next_direction               – [dx, dy, dz] unit vector toward active target
          distance_to_target           – float (m) to active target
          gripper_to_target_delta      – [Δx, Δy, Δz] in meters
          gripper_to_target_relation   – {"x": str, "y": str, "z": str}
          eef_orientation_quat         – [w, x, y, z]
          eef_orientation_euler_deg    – [roll, pitch, yaw] in degrees
          target_orientation_quat      – [w, x, y, z]
          target_orientation_euler_deg – [roll, pitch, yaw] in degrees
          relative_rotation_quat       – [w, x, y, z]
          relative_rotation_euler_deg  – [roll, pitch, yaw] in degrees
          pairwise_distance            – dict or None
          gripper_openness             – float [0, 1]
          targets_info                 – list of dicts, one per target object with
                                         {name, pos, distance, is_active, can_close}
    """
    base_pos = get_robot_base_pos(sim)
    eef_pos = np.array(raw_obs["robot0_eef_pos"]) - base_pos
    all_objects = get_all_object_positions(sim, base_pos)
    task_type = classify_task(task_description)

    # --- Multi-target: find all targets, then pick the active one ---
    all_targets = find_all_target_objects(all_objects, task_description)

    # Find destination first so we can determine which target has been placed
    # Use the first target's name for destination lookup (dest is shared in
    # "put both X and Y in the basket" style tasks)
    first_target_name = all_targets[0][0] if all_targets else "unknown"
    dest_name, dest_pos = find_destination_object(
        all_objects, task_description, first_target_name,
    )

    target_name, target_pos = find_active_target(
        eef_pos, all_targets, dest_pos=dest_pos,
    )
    targets_info = compute_targets_info(eef_pos, all_targets, target_name)

    # --- Q7: EE orientation ---
    eef_quat_wxyz = quat_wxyz_from_xyzw(np.array(raw_obs["robot0_eef_quat"]))
    eef_orn = compute_orientation(eef_quat_wxyz)

    # --- Q8: Target object orientation ---
    # MuJoCo get_body_xquat returns [w, x, y, z] natively
    target_quat_wxyz = sim.data.get_body_xquat(target_name).copy()
    target_orn = compute_orientation(target_quat_wxyz)

    # --- Q9: Relative rotation (EE → object) ---
    rel_rot = compute_relative_rotation(eef_quat_wxyz, target_quat_wxyz)

    # --- Q10: Pairwise distance between two non-target objects ---
    pair = select_object_pair(all_objects, target_name, dest_name)
    pairwise = compute_pairwise_distance(all_objects, *pair) if pair else None

    # --- Q11: Gripper openness ---
    # Panda gripper has 2 finger joints ("finger_joint1", "finger_joint2"),
    # each ranges from ~0.0 (closed) to ~0.04 (open).
    # We read directly from sim.data.qpos for reliability — the robosuite
    # observable robot0_gripper_qpos may not update correctly after
    # set_state_from_flattened() in demo replay mode.
    openness = _compute_gripper_openness_from_sim(sim, raw_obs)

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
        # Multi-target info
        "targets_info": targets_info,
    }


def get_agentview_image(raw_obs: dict) -> np.ndarray:
    """Return the agent-view RGB image, flipped to match display convention.

    LIBERO stores images upside-down in the raw obs; flipping both H and W
    gives the natural top-up view used throughout LeRobot and beta-vla.
    """
    img = raw_obs["agentview_image"]
    return img[::-1, ::-1].copy()


def get_wrist_image(raw_obs: dict) -> np.ndarray:
    """Return the wrist-mounted (eye-in-hand) RGB image, flipped to match display.

    LIBERO renders both `agentview_image` and `robot0_eye_in_hand_image`
    upside-down, so apply the same H+W flip as `get_agentview_image`.
    """
    img = raw_obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1].copy()


def get_views(raw_obs: dict) -> dict:
    """Return both available views as a dict {'agent': ndarray, 'wrist': ndarray}."""
    return {
        "agent": get_agentview_image(raw_obs),
        "wrist": get_wrist_image(raw_obs),
    }
