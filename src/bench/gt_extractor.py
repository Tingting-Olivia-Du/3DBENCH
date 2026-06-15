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


def _geom_world_z_extent(sim, gid: int) -> tuple[float, float] | None:
    """Return the world-frame (z_min, z_max) of a single geom's bounding box.

    Handles the primitive types plus mesh (by transforming mesh vertices).
    Returns None for geom types we cannot bound (e.g. planes).
    """
    import mujoco

    m, d = sim.model, sim.data
    gtype = m.geom_type[gid]
    size = m.geom_size[gid]
    pos = d.geom_xpos[gid]
    mat = d.geom_xmat[gid].reshape(3, 3)
    T = mujoco.mjtGeom

    if gtype == T.mjGEOM_BOX:
        hs = size[:3]
    elif gtype == T.mjGEOM_CYLINDER:
        hs = np.array([size[0], size[0], size[1]])
    elif gtype == T.mjGEOM_CAPSULE:
        hs = np.array([size[0], size[0], size[1] + size[0]])
    elif gtype == T.mjGEOM_SPHERE:
        hs = np.array([size[0]] * 3)
    elif gtype == T.mjGEOM_ELLIPSOID:
        hs = size[:3]
    elif gtype == T.mjGEOM_MESH:
        dataid = m.geom_dataid[gid]
        if dataid < 0:
            return None
        vadr = m.mesh_vertadr[dataid]
        vnum = m.mesh_vertnum[dataid]
        verts = m.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
        world_z = (verts @ mat.T)[:, 2] + pos[2]
        return float(world_z.min()), float(world_z.max())
    else:
        return None

    # Axis-aligned z half-extent of an oriented box.
    z_ext = (abs(mat[2, 0] * hs[0]) + abs(mat[2, 1] * hs[1]) + abs(mat[2, 2] * hs[2]))
    return float(pos[2] - z_ext), float(pos[2] + z_ext)


def _descendant_body_ids(sim, root_bid: int) -> list[int]:
    """Return root_bid plus all bodies in its kinematic sub-tree.

    Articulated objects (cabinets, stoves) keep their geometry on child bodies
    (e.g. ``wooden_cabinet_1_cabinet_top``), so the center of the whole object
    requires walking the sub-tree rather than reading only the ``_main`` body.
    """
    m = sim.model
    ids = [root_bid]
    i = 0
    while i < len(ids):
        cur = ids[i]
        for bid in range(m.nbody):
            if m.body_parentid[bid] == cur and bid != cur:
                ids.append(bid)
        i += 1
    return ids


def compute_object_center_z(sim, body_id: int, base_pos: np.ndarray) -> float | None:
    """Return the base-relative Z of an object's visible geometry center.

    Center = midpoint of the object's full collision-geometry bounding box in Z,
    i.e. roughly ``bottom + half_height``. This is a visually locatable point
    (the middle of the object's silhouette) rather than the internal body origin,
    which can sit anywhere inside the mesh.

    Aggregates over the body's whole kinematic sub-tree so articulated objects
    are bounded correctly. Returns None if the object has no bounded geoms (the
    caller should fall back to the body origin).
    """
    m = sim.model
    z_min, z_max = None, None
    for bid in _descendant_body_ids(sim, body_id):
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != bid:
                continue
            # Collision geoms only (robosuite convention: group 0). Visual meshes
            # (group 1) give a near-identical center (<=0.4 mm in practice) but are
            # less robust, so we prefer collision geometry.
            if m.geom_group[gid] != 0:
                continue
            ext = _geom_world_z_extent(sim, gid)
            if ext is None:
                continue
            z_min = ext[0] if z_min is None else min(z_min, ext[0])
            z_max = ext[1] if z_max is None else max(z_max, ext[1])
    if z_min is None:
        return None
    return float((z_min + z_max) / 2.0 - base_pos[2])


def get_all_object_positions(sim, base_pos: np.ndarray) -> dict[str, list[float]]:
    """Return {body_name: [x, y, z]} for all scene-object bodies, relative to base.

    X and Y are the body-origin (column) position. Z is the visible-geometry
    center (≈ bottom + half-height) so the model is asked to localize the middle
    of the object's silhouette, not an invisible internal reference point. Falls
    back to the body-origin Z when no bounded collision geometry is available.
    """
    objects: dict[str, list[float]] = {}
    for name in sim.model.body_names:
        if is_scene_object_body(name):
            pos_world = sim.data.get_body_xpos(name).copy()
            x = float(pos_world[0] - base_pos[0])
            y = float(pos_world[1] - base_pos[1])
            body_id = sim.model.body_name2id(name)
            center_z = compute_object_center_z(sim, body_id, base_pos)
            z = center_z if center_z is not None else float(pos_world[2] - base_pos[2])
            objects[name] = [x, y, z]
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

    Deprecated in favour of :func:`_object_instance_id`, which does not rely
    on a hand-maintained suffix whitelist. Kept for backward compatibility.
    """
    _PART_SUFFIXES = {
        "main", "base", "button", "handle", "lid", "knob", "door",
        "link", "link0", "link1", "link2", "hinge", "joint", "body",
    }
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in _PART_SUFFIXES:
        return parts[0]
    return name


def _object_instance_id(name: str) -> str:
    """Return the physical-object instance id for a MuJoCo body name.

    Groups every sub-body of the same composite object under one id by
    truncating at the first purely-numeric token (the instance index):
        'microwave_1_main'         -> 'microwave_1'
        'microwave_1_microdoorroot'-> 'microwave_1'
        'flat_stove_1_base'        -> 'flat_stove_1'
        'flat_stove_1_burner'      -> 'flat_stove_1'
    This is whitelist-free (unlike :func:`_object_base_name`) so it correctly
    de-duplicates ANY internal body name.
    """
    parts = name.split("_")
    kept: list[str] = []
    for p in parts:
        kept.append(p)
        if p.isdigit():
            break
    return "_".join(kept)


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

    Objects are de-duplicated to ONE representative body per physical instance
    (by :func:`_object_instance_id`), so sub-parts of the same composite object
    (e.g. microwave body + microwave door) are never paired together and a
    single object never appears twice.
    """
    # Exclude the target and destination by INSTANCE, so every sub-body of
    # those objects is removed (not just the exact body name).
    exclude_instances = {_object_instance_id(target_name)}
    if dest_name:
        exclude_instances.add(_object_instance_id(dest_name))

    # Collapse to one representative body per instance. Prefer the '_main'
    # body; otherwise keep the first body seen for that instance.
    rep_by_instance: dict[str, tuple[str, list[float]]] = {}
    for name, pos in all_objects.items():
        if "mount" in name.lower():
            continue
        inst = _object_instance_id(name)
        if inst in exclude_instances:
            continue
        cur = rep_by_instance.get(inst)
        if cur is None or (not cur[0].endswith("_main") and name.endswith("_main")):
            rep_by_instance[inst] = (name, pos)

    candidates = dict(rep_by_instance.values())
    if len(candidates) < 2:
        return None

    best_pair: tuple[str, str] | None = None
    best_dist = float("inf")
    for (a, pos_a), (b, pos_b) in combinations(candidates.items(), 2):
        d = float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))
        # Skip zero-distance pairs (co-located bodies)
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

    ``object_a``/``object_b`` are the raw MuJoCo body names, matching the names
    listed in the prompt and used everywhere else in the GT.

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


def check_finger_contact(sim) -> tuple[bool, str | None]:
    """Check if the gripper fingers are in contact with any object.

    Uses MuJoCo's contact data to detect whether both left and right
    finger pads are touching the same object — the robosuite definition
    of a successful grasp.

    Returns:
        (is_grasping, object_name):
          - (True, "alphabet_soup_1_g6") if both fingers contact an object
          - (False, None) if not grasping
    """
    left_contacts: set[str] = set()
    right_contacts: set[str] = set()

    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if c.geom1 >= sim.model.ngeom or c.geom2 >= sim.model.ngeom:
            continue
        g1 = sim.model.geom_id2name(c.geom1) or ""
        g2 = sim.model.geom_id2name(c.geom2) or ""

        # Identify which is the finger and which is the object
        for finger_geom, other_geom in [(g1, g2), (g2, g1)]:
            if "finger" not in finger_geom:
                continue
            # Skip robot self-collision
            if "finger" in other_geom or "gripper" in other_geom or "robot" in other_geom:
                continue
            if "finger1" in finger_geom:
                left_contacts.add(other_geom)
            elif "finger2" in finger_geom:
                right_contacts.add(other_geom)

    # Both fingers must contact at least one shared object
    shared = left_contacts & right_contacts
    if shared:
        return True, sorted(shared)[0]
    return False, None


def compute_gripper_phase(action: np.ndarray, is_grasping: bool, window_label: str | None = None) -> str:
    """Compute gripper phase for a single timestep.

    Args:
        action:        (7,) action vector. action[6]: -1=open, +1=close.
        is_grasping:   True if finger contact detected (from check_finger_contact).
        window_label:  Optional override — "releasing" if in a release window.

    Returns:
        One of: "approaching", "grasping", "carrying", "releasing".
    """
    if window_label == "releasing":
        return "releasing"
    if action[6] < 0:
        return "approaching"
    # action[6] > 0 (close command)
    return "carrying" if is_grasping else "grasping"


def compute_release_windows(actions: np.ndarray, window: int = 5) -> set[int]:
    """Find timesteps near close→open transitions (releasing zones).

    Only marks transitions where the gripper was closed for a meaningful
    duration (≥ 10 steps) to avoid marking failed-grasp releases.

    Returns:
        Set of timestep indices that fall in a release window.
    """
    T = len(actions)
    gripper_cmds = actions[:, 6]
    release_steps: set[int] = set()

    # Find close→open transitions
    in_close = False
    close_start = 0
    for t in range(T):
        if gripper_cmds[t] > 0 and not in_close:
            close_start = t
            in_close = True
        elif gripper_cmds[t] < 0 and in_close:
            in_close = False
            seg_len = t - close_start
            if seg_len >= 10:  # only meaningful close segments
                for tt in range(max(0, t - window), min(T, t + window)):
                    release_steps.add(tt)

    return release_steps


# ---------------------------------------------------------------------------
# Robosuite / LIBERO sim-based detection
# ---------------------------------------------------------------------------

def body_to_object_key(env, body_name: str) -> str:
    """Map a MuJoCo body name to LIBERO's BDDL object key.

    Target/destination objects are tracked by MuJoCo body name (e.g.
    'alphabet_soup_1_main', 'wooden_cabinet_1_cabinet_top'), but the sim-state
    lookups (goal predicates, grasp, placement) are keyed by the BDDL object id
    (e.g. 'alphabet_soup_1', 'wooden_cabinet_1'). This converts between them.

    Strategy (returns the first candidate that exists in the env's object dicts,
    else the best textual guess):
      1. the body name as-is
      2. the body name with a trailing '_main' stripped
      3. the instance id (truncate at the first numeric token)
    """
    candidates = [body_name]
    if body_name.endswith("_main"):
        candidates.append(body_name[: -len("_main")])
    inst = _object_instance_id(body_name)
    if inst not in candidates:
        candidates.append(inst)

    # Validate against whatever object dicts the env exposes.
    inner = env.env if (env is not None and hasattr(env, "env")) else env
    known: set[str] = set()
    if inner is not None:
        for attr in ("objects_dict", "object_states_dict"):
            d = getattr(inner, attr, None)
            if isinstance(d, dict):
                known |= set(d.keys())
    if known:
        for c in candidates:
            if c in known:
                return c
    # Fall back to the instance id (correct for both '_main' and multi-part
    # bodies like 'wooden_cabinet_1_cabinet_top').
    return inst


def check_grasp_robosuite(env, object_name: str) -> bool:
    """Check if the gripper is grasping a specific object using robosuite's API.

    Uses env._check_grasp() which checks that both left and right finger pads
    are in contact with the object's collision geoms.

    Args:
        env:          The LIBERO OffScreenRenderEnv (or its inner env).
        object_name:  MuJoCo body name OR BDDL object key; converted internally.

    Returns:
        True if the gripper is grasping the object.
    """
    inner = env.env if hasattr(env, "env") else env
    if not hasattr(inner, "_check_grasp") or not hasattr(inner, "objects_dict"):
        return False
    key = body_to_object_key(env, object_name)
    if key not in inner.objects_dict:
        return False
    robot = inner.robots[0]
    return bool(inner._check_grasp(robot.gripper, inner.objects_dict[key]))


def check_task_success(env) -> bool:
    """Check if the full task goal is achieved using LIBERO's BDDL predicates.

    Evaluates all goal predicates (e.g. "in(soup, basket)" AND "in(tomato, basket)").

    Args:
        env: The LIBERO OffScreenRenderEnv.

    Returns:
        True if all goal predicates are satisfied.
    """
    inner = env.env if hasattr(env, "env") else env
    if not hasattr(inner, "_check_success"):
        return False
    try:
        return bool(inner._check_success())
    except Exception:
        return False


def check_goal_predicates(env) -> list[dict]:
    """Evaluate each goal predicate individually and return detailed results.

    Parses the BDDL goal_state and evaluates each predicate against the
    current sim state using LIBERO's object_states_dict and eval_predicate_fn.

    Returns a list of dicts, one per goal predicate:
      {"predicate": "in", "args": ["alphabet_soup_1", "basket_1_contain_region"],
       "satisfied": True}
    """
    inner = env.env if hasattr(env, "env") else env
    results = []

    if not hasattr(inner, "parsed_problem") or not hasattr(inner, "object_states_dict"):
        return results

    try:
        from libero.libero.envs.predicates import eval_predicate_fn
    except ImportError:
        return results

    goal_state = inner.parsed_problem.get("goal_state", [])
    osd = inner.object_states_dict

    for state in goal_state:
        entry = {"predicate": state[0], "args": list(state[1:]), "satisfied": False}
        try:
            if len(state) == 3 and state[1] in osd and state[2] in osd:
                entry["satisfied"] = bool(eval_predicate_fn(state[0], osd[state[1]], osd[state[2]]))
            elif len(state) == 2 and state[1] in osd:
                entry["satisfied"] = bool(eval_predicate_fn(state[0], osd[state[1]]))
        except Exception:
            pass
        results.append(entry)

    return results


def check_object_placed(env, object_name: str) -> bool:
    """Check if a specific object has been placed at its goal destination.

    Looks through the goal predicates for ones involving this object
    (e.g. "in(alphabet_soup_1, basket_1_contain_region)") and evaluates them.

    Args:
        env:          The LIBERO OffScreenRenderEnv.
        object_name:  MuJoCo body name OR BDDL object key (e.g.
                      "alphabet_soup_1_main" or "alphabet_soup_1"); converted
                      to the BDDL key internally.

    Returns:
        True if the object satisfies its goal predicate.
    """
    key = body_to_object_key(env, object_name)
    predicates = check_goal_predicates(env)
    for p in predicates:
        if key in p["args"]:
            return p["satisfied"]
    return False


def get_all_grasp_states(env, object_names: list[str]) -> dict[str, bool]:
    """Check grasp state for multiple objects at once.

    Args:
        env:           The LIBERO OffScreenRenderEnv.
        object_names:  List of object keys in env.objects_dict.

    Returns:
        Dict mapping object_name → is_grasped.
    """
    return {name: check_grasp_robosuite(env, name) for name in object_names}


def get_articulation_states(env) -> dict[str, dict]:
    """Get open/close/turn_on/turn_off states for all articulated objects.

    Scans object_states_dict for objects that have joints (articulated),
    and reads their current state.

    Returns:
        Dict mapping object_name → {"is_open": bool, "is_close": bool,
                                     "turn_on": bool|None, "turn_off": bool|None}
    """
    inner = env.env if hasattr(env, "env") else env
    if not hasattr(inner, "object_states_dict"):
        return {}

    osd = inner.object_states_dict
    results = {}

    for name, state in osd.items():
        # Only check ObjectState (not SiteObjectState) with joints
        if state.object_state_type != "object":
            continue
        if name not in inner.objects_dict:
            continue
        obj = inner.objects_dict[name]
        if not hasattr(obj, "joints") or not obj.joints:
            continue

        entry: dict = {}
        try:
            entry["is_open"] = bool(state.is_open())
        except Exception:
            entry["is_open"] = None
        try:
            entry["is_close"] = bool(state.is_close())
        except Exception:
            entry["is_close"] = None
        # turn_on / turn_off only for objects with that affordance
        if hasattr(state, "has_turnon_affordance") and state.has_turnon_affordance:
            try:
                entry["turn_on"] = bool(state.turn_on())
            except Exception:
                entry["turn_on"] = None
            try:
                entry["turn_off"] = bool(state.turn_off())
            except Exception:
                entry["turn_off"] = None

        if entry:
            results[name] = entry

    return results


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


def _has_articulation_verb(desc_lower: str) -> bool:
    """True if any articulation verb appears as a leading or whole-word token."""
    return any(
        desc_lower.startswith(v) or f" {v} " in desc_lower
        for v in _ARTICULATION_VERBS
    )


# Phrasal-verb particles whose " on the " / " off the " must NOT be read as a
# place-preposition, e.g. "turn on the stove", "switch off the light".
_PHRASAL_ON_OFF_VERBS = ("turn", "switch")


def _has_place_destination(desc_lower: str) -> bool:
    """True if the description contains a place-preposition (source → destination).

    Guards against false positives where " on the " is actually the particle of
    a phrasal articulation verb ("turn on the ...", "switch off the ...").
    """
    for prep in _PLACE_PREPOSITIONS:
        idx = desc_lower.find(prep)
        if idx == -1:
            continue
        # Reject " on the " / " off the " when it belongs to a phrasal verb.
        if prep in (" on the ",):
            preceding = desc_lower[:idx].rstrip().rsplit(" ", 1)
            if preceding and preceding[-1] in _PHRASAL_ON_OFF_VERBS:
                # try a later occurrence of this same prep before giving up
                later = desc_lower.find(prep, idx + 1)
                if later == -1:
                    continue
            return True
        return True
    return False


def classify_task(task_description: str) -> str:
    """Classify a task as 'pick_and_place', 'articulation', or 'compound'.

    - 'compound': a pick-and-place with a place destination AND a trailing
      articulation step, e.g. "put the mug in the microwave and close it".
    - 'articulation': pure articulation with no place destination, e.g.
      "open the top drawer of the cabinet".
    - 'pick_and_place': a place with no articulation verb.
    """
    desc_lower = task_description.lower()
    has_artic = _has_articulation_verb(desc_lower)
    has_dest = _has_place_destination(desc_lower)

    if has_artic and has_dest:
        return "compound"
    if has_artic:
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

    Returns (body_name, position_array) or (None, None) for pure articulation
    tasks. Compound tasks (place + trailing articulation) DO have a destination.
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


# ----- Human-readable object names -------------------------------------


def task_named_objects(gt: dict) -> list[dict]:
    """Return the task-relevant named objects for the per-object coordinate Q.

    Reads an already-extracted GT dict and returns an ordered, de-duplicated
    list of objects the model should localize: every target (in task order)
    followed by the destination (if any). Each entry is:
        {"name": <MuJoCo body name>, "body": <MuJoCo body name>, "pos": [x, y, z]}

    Names are the raw MuJoCo body names (e.g. "white_yellow_mug_1_main"); the
    Q1 prompt lists these explicitly so the model echoes them back verbatim and
    GT/prediction match exactly. ``name`` and ``body`` are kept identical for
    backward compatibility with callers that read either key.

    This is the single source of truth shared by the prompt builder (which
    object names to ask for) and the metrics scorer (the GT position for each).
    """
    entries: list[dict] = []
    seen: set[str] = set()

    def _add(body: str, pos) -> None:
        if not body or body in seen or pos is None:
            return
        seen.add(body)
        entries.append({
            "name": body,
            "body": body,
            "pos": list(pos),
        })

    # Targets first, in task order (targets_info preserves description order).
    for info in gt.get("targets_info") or []:
        _add(info.get("name"), info.get("pos"))

    # Fall back to single target if targets_info is absent.
    if not entries and gt.get("target_object_name"):
        _add(gt["target_object_name"], gt.get("target_pos"))

    # Destination last.
    if gt.get("dest_object_name"):
        _add(gt["dest_object_name"], gt.get("dest_pos"))

    return entries


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
    env=None,
    dest_pos: np.ndarray | None = None,
    placed_threshold: float = 0.08,
) -> tuple[str, np.ndarray]:
    """Pick the active target following task-description order.

    For multi-target tasks (e.g. "put both X and Y in the basket"),
    the demo always operates on targets in the order they appear in
    the language instruction.  A target is considered "placed" (done)
    when it satisfies its goal predicate (checked via LIBERO's BDDL
    predicates).  Falls back to distance-based check if env is unavailable.

    Args:
        eef_pos:           Current gripper position (robot-base frame).
        targets:           Ordered list of (name, pos) from find_all_target_objects
                           (already in language-instruction order).
        env:               Optional LIBERO env for predicate-based checks.
        dest_pos:          Fallback destination position for distance check.
        placed_threshold:  Fallback max distance to count as "placed".

    Returns:
        (active_target_name, active_target_pos)
    """
    if len(targets) == 1:
        return targets[0]

    # Preferred: use LIBERO predicates to check if target is placed
    if env is not None:
        for name, pos in targets:
            if not check_object_placed(env, name):
                return name, pos
        return targets[-1]

    # Fallback: distance-based check
    if dest_pos is not None:
        for name, pos in targets:
            dist_to_dest = float(np.linalg.norm(pos - dest_pos))
            if dist_to_dest > placed_threshold:
                return name, pos
        return targets[-1]

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


def extract_eef_pose_7d(raw_obs: dict, sim, base_pos: np.ndarray) -> dict:
    """Extract 7D end-effector pose: [x, y, z, roll, pitch, yaw, gripper_openness].

    All values in robot-base frame.

    Returns:
        Dict with keys: x, y, z, roll, pitch, yaw, gripper_openness
    """
    eef_pos = np.array(raw_obs["robot0_eef_pos"]) - base_pos
    eef_quat_wxyz = quat_wxyz_from_xyzw(np.array(raw_obs["robot0_eef_quat"]))
    euler = quat_to_euler_deg(eef_quat_wxyz)
    openness = _compute_gripper_openness_from_sim(sim, raw_obs)
    return {
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "gripper_openness": openness,
    }


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
    env=None,
) -> dict:
    """Extract all ground truth fields from a raw LIBERO observation + sim state.

    Args:
        raw_obs: Dict returned by ``OffScreenRenderEnv.step()`` / ``.reset()``.
        sim:     The MuJoCo ``sim`` object (e.g. ``env.sim``).
        task_description: Natural-language task string used to identify
                          the target object.
        env:     Optional LIBERO OffScreenRenderEnv — when provided, enables
                 sim-based detection: task_success, goal_predicates,
                 active_target via predicates, grasp states, articulation states.

    Returns:
        Dict with spatial GT fields + sim-based detection fields.
    """
    base_pos = get_robot_base_pos(sim)
    eef_pos = np.array(raw_obs["robot0_eef_pos"]) - base_pos
    all_objects = get_all_object_positions(sim, base_pos)
    task_type = classify_task(task_description)

    # --- Multi-target: find all targets, then pick the active one ---
    all_targets = find_all_target_objects(all_objects, task_description)

    # Find destination first so we can determine which target has been placed
    first_target_name = all_targets[0][0] if all_targets else "unknown"
    dest_name, dest_pos = find_destination_object(
        all_objects, task_description, first_target_name,
    )

    target_name, target_pos = find_active_target(
        eef_pos, all_targets, env=env, dest_pos=dest_pos,
    )
    targets_info = compute_targets_info(eef_pos, all_targets, target_name)

    # --- Q7: EE orientation ---
    eef_quat_wxyz = quat_wxyz_from_xyzw(np.array(raw_obs["robot0_eef_quat"]))
    eef_orn = compute_orientation(eef_quat_wxyz)

    # --- Q8: Target object orientation ---
    target_quat_wxyz = sim.data.get_body_xquat(target_name).copy()
    target_orn = compute_orientation(target_quat_wxyz)

    # --- Q9: Relative rotation (EE → object) ---
    rel_rot = compute_relative_rotation(eef_quat_wxyz, target_quat_wxyz)

    # --- Q10: Pairwise distance between two non-target objects ---
    pair = select_object_pair(all_objects, target_name, dest_name)
    pairwise = compute_pairwise_distance(all_objects, *pair) if pair else None

    # --- Q11: Gripper openness ---
    openness = _compute_gripper_openness_from_sim(sim, raw_obs)

    # --- Sim-based detection (when env is available) ---
    task_success = check_task_success(env) if env else None
    goal_predicates = check_goal_predicates(env) if env else []
    articulation_states = get_articulation_states(env) if env else {}

    # Grasp states for all target objects
    target_names = [name for name, _ in all_targets]
    grasp_states = get_all_grasp_states(env, target_names) if env else {}

    # Enrich targets_info with placement and grasp from sim
    for info in targets_info:
        name = info["name"]
        info["placed"] = check_object_placed(env, name) if env else None
        info["grasped"] = grasp_states.get(name, None)

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
        # Multi-target info (now includes placed/grasped from sim)
        "targets_info": targets_info,
        # Sim-based detection
        "task_success": task_success,
        "goal_predicates": goal_predicates,
        "articulation_states": articulation_states if articulation_states else None,
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
