"""Mapping from CALVIN task class strings to scene-state targets.

CALVIN env D scene_obs (24-D) layout (verified empirically from
episode_0000000.npz — the official README is sparse and the actual layout
interleaves each block's xyz with its euler):
    [0]  sliding_door_position   (slider on the shelf, scalar)
    [1]  drawer_position         (lower drawer, scalar)
    [2]  button_position         (push-button, scalar)
    [3]  switch_position         (light switch, scalar)
    [4]  bulb                    (lightbulb on/off, binary)
    [5]  green_light             (green LED on/off, binary)
    [6:9]    red_block  xyz       (world frame)
    [9:12]   red_block  euler
    [12:15]  blue_block xyz
    [15:18]  blue_block euler
    [18:21]  pink_block xyz
    [21:24]  pink_block euler

The articulated objects (slider, drawer, button, switch, lightbulb, LED) live
on the **panel** at fixed world positions. The scene_obs scalar is the joint
state; the spatial position of the object itself is fixed. We expose them as
constants so Q1 / Q5 / Q6 can be computed.

The auto_lang_ann.npy file stores per-window:
    "language" : list[str]      free-text instruction
    "task"     : list[str]      one of the discrete classes below
    "indx"     : list[(start_idx, end_idx)] inclusive

For each task class we record:
    target_kind        : "block" | "articulated"
    target_key         : index/name into the lookup
    task_type          : "pick_and_place" | "articulation"
    has_destination    : bool — whether the task has a separate destination

The exact set of 34 classes was verified against the official mees/calvin repo
(dataset/conf/dataset/calvin_env.yaml + lang_ann auto-labelled tasks).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Robot base in CALVIN env D world frame.
# Source: validation/.hydra/merged_config.yaml → scene.robot_base_position
# (also config.yaml line 61). The Panda is mounted at this fixed pose; all
# robot_obs positions are in world frame, so subtracting this constant gives
# base-relative coordinates (mirroring the LIBERO convention).
# ---------------------------------------------------------------------------
ROBOT_BASE_POS_WORLD = np.array([-0.34, -0.46, 0.24], dtype=np.float32)

# ---------------------------------------------------------------------------
# Fixed world-frame positions of the articulated objects on the play panel.
# Sourced from the CALVIN env URDF in the official simulator. These are
# approximate centroids of the articulated elements; sufficient for Q1/Q5/Q6.
# ---------------------------------------------------------------------------
ARTICULATED_POSITIONS = {
    "slider":   np.array([-0.10, -0.10,  0.55], dtype=np.float32),  # sliding-door knob (top of shelf)
    "drawer":   np.array([ 0.20, -0.10,  0.40], dtype=np.float32),  # lower drawer handle
    "button":   np.array([ 0.10,  0.00,  0.51], dtype=np.float32),  # push-button on panel
    "switch":   np.array([-0.05,  0.00,  0.51], dtype=np.float32),  # light switch
    "bulb":     np.array([-0.20,  0.00,  0.55], dtype=np.float32),  # lightbulb glass
    "led":      np.array([ 0.20,  0.00,  0.55], dtype=np.float32),  # green LED
}

# Indices into scene_obs for block xyz.
# NOTE: blocks are interleaved as (xyz, euler) per block, NOT (all xyz, all euler).
BLOCK_INDEX = {
    "red_block":  slice(6, 9),
    "blue_block": slice(12, 15),
    "pink_block": slice(18, 21),
}

# Indices into scene_obs for block euler angles (radians).
BLOCK_EULER_INDEX = {
    "red_block":  slice(9, 12),
    "blue_block": slice(15, 18),
    "pink_block": slice(21, 24),
}

# Indices into scene_obs for articulation joint scalars (informational; the
# *positions* of the articulated objects live in ARTICULATED_POSITIONS).
ARTICULATION_JOINT_IDX = {
    "slider":   0,
    "drawer":   1,
    "button":   2,
    "switch":   3,
    "bulb":     4,
    "led":      5,
}

# Receptacles for pick_and_place destinations. CALVIN's tabletop has only a
# few well-defined places to put a block: the slider (shelf), the drawer, or
# anywhere on the table. We map task classes to one of these.
RECEPTACLE_POSITIONS = {
    "slider":     ARTICULATED_POSITIONS["slider"],   # "place block in slider"
    "drawer":     ARTICULATED_POSITIONS["drawer"],   # "place block in drawer"
    "table":      np.array([ 0.00, -0.20,  0.46], dtype=np.float32),  # generic open table area
}

# ---------------------------------------------------------------------------
# Per-task-class entry. Each value:
#   target_kind   : "block"      → look up in BLOCK_INDEX (uses scene_obs)
#                   "articulated" → look up in ARTICULATED_POSITIONS (constant)
#   target_key    : the lookup key
#   task_type     : "pick_and_place" | "articulation"
#   destination   : None | str (RECEPTACLE_POSITIONS key) | str (block name → BLOCK_INDEX)
# ---------------------------------------------------------------------------

TASK_CLASS_TO_TARGET: dict[str, dict] = {
    # ── Block lifts (treated as pick_and_place with no fixed destination) ──
    "lift_red_block_table":   dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "lift_blue_block_table":  dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "lift_pink_block_table":  dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
    "lift_red_block_slider":  dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "lift_blue_block_slider": dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "lift_pink_block_slider": dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
    "lift_red_block_drawer":  dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "lift_blue_block_drawer": dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "lift_pink_block_drawer": dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),

    # ── Block placements (pick_and_place WITH a destination) ──
    "place_in_slider":        dict(target_kind="block", target_key="ANY_BLOCK", task_type="pick_and_place", destination="slider"),
    "place_in_drawer":        dict(target_kind="block", target_key="ANY_BLOCK", task_type="pick_and_place", destination="drawer"),

    # ── Block pushes (pick_and_place style: source = block, dest = ?) ──
    "push_red_block_left":    dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "push_red_block_right":   dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "push_blue_block_left":   dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "push_blue_block_right":  dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "push_pink_block_left":   dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
    "push_pink_block_right":  dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
    "push_into_drawer":       dict(target_kind="block", target_key="ANY_BLOCK", task_type="pick_and_place", destination="drawer"),

    # ── Articulations ──
    "open_drawer":            dict(target_kind="articulated", target_key="drawer", task_type="articulation", destination=None),
    "close_drawer":           dict(target_kind="articulated", target_key="drawer", task_type="articulation", destination=None),
    "move_slider_left":       dict(target_kind="articulated", target_key="slider", task_type="articulation", destination=None),
    "move_slider_right":      dict(target_kind="articulated", target_key="slider", task_type="articulation", destination=None),
    "turn_on_lightbulb":      dict(target_kind="articulated", target_key="bulb",   task_type="articulation", destination=None),
    "turn_off_lightbulb":     dict(target_kind="articulated", target_key="bulb",   task_type="articulation", destination=None),
    "turn_on_led":            dict(target_kind="articulated", target_key="led",    task_type="articulation", destination=None),
    "turn_off_led":           dict(target_kind="articulated", target_key="led",    task_type="articulation", destination=None),
    "push_button":            dict(target_kind="articulated", target_key="button", task_type="articulation", destination=None),
    "press_button":           dict(target_kind="articulated", target_key="button", task_type="articulation", destination=None),
    "toggle_switch":          dict(target_kind="articulated", target_key="switch", task_type="articulation", destination=None),

    # ── Stack / unstack: ambiguous source/dest — by default we drop these ──
    # Keep here so the loader can detect and skip them via DENY_LIST below.
    "stack_block":            dict(target_kind="block", target_key="ANY_BLOCK", task_type="pick_and_place", destination="OTHER_BLOCK"),
    "unstack_block":          dict(target_kind="block", target_key="ANY_BLOCK", task_type="pick_and_place", destination=None),
    "rotate_red_block_left":  dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "rotate_red_block_right": dict(target_kind="block", target_key="red_block",  task_type="pick_and_place", destination=None),
    "rotate_blue_block_left": dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "rotate_blue_block_right":dict(target_kind="block", target_key="blue_block", task_type="pick_and_place", destination=None),
    "rotate_pink_block_left": dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
    "rotate_pink_block_right":dict(target_kind="block", target_key="pink_block", task_type="pick_and_place", destination=None),
}

# Multi-block / ambiguous tasks where target identity isn't deterministic from
# language alone. The loader skips windows whose `task` is in this set.
DENY_LIST = {"stack_block", "unstack_block",
             "place_in_slider", "place_in_drawer", "push_into_drawer"}
# (place_in_* and push_into_* are skipped because the *source* block isn't
#  named in the task class; we'd need to inspect the moving block at runtime.
#  Future work: pick the block that moves the most across the window.)


def find_target_calvin(
    task_class: str,
    scene_obs: np.ndarray,
) -> tuple[str, np.ndarray, str, np.ndarray | None] | None:
    """Resolve (target_name, target_xyz, task_type, dest_xyz) for a CALVIN task class.

    Returns None if the task class is unknown or in the deny-list.
    """
    if task_class not in TASK_CLASS_TO_TARGET:
        return None
    if task_class in DENY_LIST:
        return None
    entry = TASK_CLASS_TO_TARGET[task_class]
    kind = entry["target_kind"]
    key = entry["target_key"]
    task_type = entry["task_type"]

    if kind == "block":
        if key == "ANY_BLOCK":
            # Without runtime motion analysis we can't tell which block is the source.
            return None
        target_xyz = np.asarray(scene_obs[BLOCK_INDEX[key]], dtype=np.float32)
    elif kind == "articulated":
        target_xyz = ARTICULATED_POSITIONS[key].copy()
    else:
        return None

    dest_key = entry["destination"]
    if dest_key is None or dest_key == "OTHER_BLOCK":
        dest_xyz = None
    elif dest_key in RECEPTACLE_POSITIONS:
        dest_xyz = RECEPTACLE_POSITIONS[dest_key].copy()
    else:
        dest_xyz = None

    return key, target_xyz, task_type, dest_xyz


def all_object_positions_calvin(scene_obs: np.ndarray) -> dict[str, list]:
    """Build a {name: [x,y,z]} dict for *all* known scene objects.

    Useful for dumping a `gt['all_objects']` that mirrors the LIBERO schema.
    Articulated objects use their fixed world positions; blocks use scene_obs.
    """
    out: dict[str, list] = {}
    for name, sl in BLOCK_INDEX.items():
        out[name] = np.asarray(scene_obs[sl], dtype=np.float32).tolist()
    for name, pos in ARTICULATED_POSITIONS.items():
        out[name] = pos.tolist()
    return out
