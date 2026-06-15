"""Parse VLM text responses into structured ground-truth-comparable dicts.

Expected VLM output (single JSON object) — new numbering:
  {
    "q1": {"x": float, "y": float, "z": float},            # source object position
    "q1_dest": {"x": float, "y": float, "z": float},       # destination position
    "q2": {"x": float, "y": float, "z": float},            # gripper position
    "q3": {"dx": float, "dy": float, "dz": float},         # gripper→target offset
    "q4": {"x": str, "y": str, "z": str},                  # spatial relation labels
    "q5": {"object_a": str, "object_b": str, "distance_m": float},  # pairwise distance
    "q6": {"roll": float, "pitch": float, "yaw": float},   # EE orientation (degrees)
    "q7": {"openness": float},                              # gripper openness [0,1]
    "q8": {"dx": float, "dy": float, "dz": float,          # next action (7D)
           "droll": float, "dpitch": float, "dyaw": float,
           "gripper": float}
  }

Mapping from old → new numbering:
  Q1      = Q1  (source object position)
  Q1_dest = Q1_dest (destination position)
  Q2      = Q2  (gripper position)
  Q3      = old Q5  (gripper-to-target offset)
  Q4      = old Q6  (spatial relationship)
  Q5      = old Q10 (pairwise object distance)
  Q6      = old Q7  (EE orientation)
  Q7      = old Q11 (gripper openness)
  Q8      = new     (next action, 7D demo action)
  Removed: old Q3 (can_close), old Q8 (target orientation), old Q9 (relative rotation)

The parser is lenient: it tolerates markdown code fences, alternative key
names, and partial responses.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


# ----- JSON extraction -------------------------------------------------

def _extract_json_from_text(text: str) -> dict | None:
    """Try several strategies to pull a JSON object from raw model output."""
    text = text.strip()

    # 1. Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences then parse
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Find the first {...} block that parses (handles prose + JSON)
    brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
    for match in brace_pattern.finditer(text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ----- Field-level parsers ---------------------------------------------

def _parse_xyz(data: Any) -> list[float] | None:
    """Parse {"x": ..., "y": ..., "z": ...} → [x, y, z]."""
    if not isinstance(data, dict):
        return None
    try:
        return [float(data["x"]), float(data["y"]), float(data["z"])]
    except (KeyError, TypeError, ValueError):
        return None


def _parse_object_positions(data: Any) -> dict[str, list[float]] | None:
    """Parse the per-object Q1 dict {name: {x,y,z}} → {mujoco_name: [x, y, z]}.

    Keys are the model's object names verbatim (only stripped of surrounding
    whitespace). The Q1 prompt lists the explicit MuJoCo body names, so the
    model echoes them back and they match the GT names directly — no
    normalization needed. Returns None if no valid position is parsed.
    """
    if not isinstance(data, dict):
        return None
    out: dict[str, list[float]] = {}
    for name, val in data.items():
        xyz = _parse_xyz(val)
        if xyz is not None:
            out[str(name).strip()] = xyz
    return out or None


def _parse_delta(data: Any) -> list[float] | None:
    """Parse {"dx": ..., "dy": ..., "dz": ...} → [dx, dy, dz] without normalizing."""
    if not isinstance(data, dict):
        return None
    try:
        return [float(data["dx"]), float(data["dy"]), float(data["dz"])]
    except (KeyError, TypeError, ValueError):
        return None


# Valid label sets for Q4 spatial relation parsing
_X_LABELS = {"in_front", "behind", "aligned_x"}
_Y_LABELS = {"left", "right", "aligned_y"}
_Z_LABELS = {"above", "below", "aligned_z"}

# Alias normalization for Q4: map common VLM alternatives to canonical labels
_X_ALIASES: dict[str, str] = {
    "forward": "in_front", "front": "in_front", "ahead": "in_front",
    "backward": "behind", "back": "behind",
    "aligned": "aligned_x",
}
_Y_ALIASES: dict[str, str] = {
    "aligned": "aligned_y",
}
_Z_ALIASES: dict[str, str] = {
    "up": "above", "higher": "above", "over": "above",
    "down": "below", "lower": "below", "under": "below",
    "aligned": "aligned_z",
}


def _normalize_relation_label(raw: str, valid: set[str], aliases: dict[str, str]) -> str | None:
    """Normalize a raw label string to a canonical relation label."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s in valid:
        return s
    if s in aliases:
        candidate = aliases[s]
        if candidate in valid:
            return candidate
    return None


def _parse_spatial_relation(data: Any) -> dict[str, str] | None:
    """Parse {"x": str, "y": str, "z": str} → canonical relation dict or None."""
    if not isinstance(data, dict):
        return None
    x_label = _normalize_relation_label(data.get("x", ""), _X_LABELS, _X_ALIASES)
    y_label = _normalize_relation_label(data.get("y", ""), _Y_LABELS, _Y_ALIASES)
    z_label = _normalize_relation_label(data.get("z", ""), _Z_LABELS, _Z_ALIASES)
    if x_label is None and y_label is None and z_label is None:
        return None
    return {"x": x_label, "y": y_label, "z": z_label}


def _parse_euler(data: Any) -> list[float] | None:
    """Parse {"roll": float, "pitch": float, "yaw": float} → [roll, pitch, yaw] degrees."""
    if not isinstance(data, dict):
        return None
    try:
        return [float(data["roll"]), float(data["pitch"]), float(data["yaw"])]
    except (KeyError, TypeError, ValueError):
        return None


def _parse_pairwise_distance(data: Any) -> dict | None:
    """Parse {"object_a": str, "object_b": str, "distance_m": float}.

    Object names are kept as-is (the prompt asks for MuJoCo body names).
    """
    if not isinstance(data, dict):
        return None
    try:
        return {
            "object_a": str(data["object_a"]),
            "object_b": str(data["object_b"]),
            "distance_m": float(data["distance_m"]),
        }
    except (KeyError, TypeError, ValueError):
        # Also try alternative key names
        try:
            dist = data.get("distance_m") or data.get("distance") or data.get("dist")
            obj_a = data.get("object_a") or data.get("obj_a")
            obj_b = data.get("object_b") or data.get("obj_b")
            if dist is not None and obj_a and obj_b:
                return {
                    "object_a": str(obj_a),
                    "object_b": str(obj_b),
                    "distance_m": float(dist),
                }
        except (TypeError, ValueError):
            pass
        return None


def _parse_openness(data: Any) -> float | None:
    """Parse {"openness": float} → float in [0, 1]."""
    if isinstance(data, dict):
        val = data.get("openness")
    elif isinstance(data, (int, float)):
        val = data
    else:
        return None
    try:
        return float(np.clip(float(val), 0.0, 1.0))
    except (TypeError, ValueError):
        return None


def _parse_action_7d(data: Any) -> dict | None:
    """Parse 7D action: {"dx","dy","dz","droll","dpitch","dyaw","gripper"}.

    Returns dict with keys: translation [dx,dy,dz], rotation [droll,dpitch,dyaw],
    gripper float.
    """
    if not isinstance(data, dict):
        return None
    try:
        return {
            "translation": [float(data["dx"]), float(data["dy"]), float(data["dz"])],
            "rotation": [float(data["droll"]), float(data["dpitch"]), float(data["dyaw"])],
            "gripper": float(data["gripper"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


# ----- Top-level resolver ----------------------------------------------

def _resolve_subfield(top: dict, primary_key: str, alt_keys: tuple[str, ...]) -> Any:
    """Look up a field by primary key or fallback alternatives."""
    if primary_key in top:
        return top[primary_key]
    for k in alt_keys:
        if k in top:
            return top[k]
    return None


class ResponseParser:
    """Parse a single VLM response string into a structured answer dict.

    New Q numbering (v2):
      Q1      source object position
      Q1_dest destination position
      Q2      gripper position
      Q3      gripper-to-target offset
      Q4      spatial relation (language)
      Q5      pairwise object distance
      Q6      EE orientation (euler)
      Q7      gripper openness
      Q8      next action (7D)
    """

    def parse(self, response_text: str) -> dict:
        """Return a dict with keys:

        raw                           – original response string
        parsed_ok                     – bool
        task_type                     – str or None
        q1_object_positions           – {mujoco_name: [x, y, z]} or None
        q2_gripper_pos                – [x, y, z] or None
        q3_gripper_to_target          – [dx, dy, dz] or None
        q4_spatial_relation           – {"x": str, "y": str, "z": str} or None
        q5_pairwise_distance          – {"object_a", "object_b", "distance_m"} or None
        q6_eef_orientation_euler      – [roll, pitch, yaw] degrees or None
        q7_gripper_openness           – float in [0,1] or None
        q8_next_action                – {"translation", "rotation", "gripper"} or None
        """
        result: dict = {
            "raw": response_text,
            "parsed_ok": False,
            "task_type": None,
            "q1_object_positions": None,
            "q2_gripper_pos": None,
            "q3_gripper_to_target": None,
            "q4_spatial_relation": None,
            "q5_pairwise_distance": None,
            "q6_eef_orientation_euler": None,
            "q7_gripper_openness": None,
            "q8_next_action": None,
        }

        data = _extract_json_from_text(response_text)
        if data is None:
            return result

        # task_type classification
        tt = data.get("task_type")
        if isinstance(tt, str) and tt in ("pick_and_place", "articulation", "compound"):
            result["task_type"] = tt

        # Q1 – per-object positions: {object_name: {x, y, z}}
        q1_raw = _resolve_subfield(
            data, "q1",
            ("object_positions", "objects", "positions"),
        )
        result["q1_object_positions"] = _parse_object_positions(q1_raw)

        # Q2 – gripper position
        q2_raw = _resolve_subfield(
            data, "q2",
            ("gripper_position", "eef_position", "end_effector_position"),
        )
        result["q2_gripper_pos"] = _parse_xyz(q2_raw)

        # Q3 – gripper-to-target offset (was old Q5)
        q3_raw = _resolve_subfield(data, "q3", ("gripper_to_target", "relative_position", "delta",
                                                  "q5",))  # fallback to old q5 key
        result["q3_gripper_to_target"] = _parse_delta(q3_raw)

        # Q4 – spatial relation (was old Q6)
        q4_raw = _resolve_subfield(data, "q4", ("spatial_relation", "relation", "spatial_relationship",
                                                  "q6",))  # fallback to old q6 key
        result["q4_spatial_relation"] = _parse_spatial_relation(q4_raw)

        # Q5 – pairwise distance (was old Q10)
        q5_raw = _resolve_subfield(data, "q5", ("pairwise_distance", "object_distance",
                                                  "q10",))  # fallback to old q10 key
        result["q5_pairwise_distance"] = _parse_pairwise_distance(q5_raw)

        # Q6 – EE orientation (was old Q7)
        q6_raw = _resolve_subfield(data, "q6", ("eef_orientation", "gripper_orientation", "ee_orientation",
                                                  "q7",))  # fallback to old q7 key
        result["q6_eef_orientation_euler"] = _parse_euler(q6_raw)

        # Q7 – gripper openness (was old Q11)
        q7_raw = _resolve_subfield(data, "q7", ("gripper_openness", "openness",
                                                  "q11",))  # fallback to old q11 key
        result["q7_gripper_openness"] = _parse_openness(q7_raw)

        # Q8 – next action 7D (new)
        q8_raw = _resolve_subfield(data, "q8", ("next_action", "action", "demo_action",
                                                  "q4",))  # fallback to old q4 key
        result["q8_next_action"] = _parse_action_7d(q8_raw)

        result["parsed_ok"] = any(
            result[k] is not None
            for k in ("q1_object_positions", "q2_gripper_pos", "q3_gripper_to_target",
                       "q4_spatial_relation", "q6_eef_orientation_euler",
                       "q7_gripper_openness", "q8_next_action")
        )
        return result
