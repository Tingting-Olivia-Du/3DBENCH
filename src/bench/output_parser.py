"""Parse VLM text responses into structured ground-truth-comparable dicts.

Expected VLM output (single JSON object):
  {
    "q1": {"x": float, "y": float, "z": float},   # target object position
    "q2": {"x": float, "y": float, "z": float},   # gripper position
    "q3": {"can_close": "yes"|"no"},               # gripper close label
    "q4": {"dx": float, "dy": float, "dz": float}, # next move direction (unit vector)
    "q5": {"dx": float, "dy": float, "dz": float}, # gripper→target offset (not normalized)
    "q6": {"x": str, "y": str, "z": str}           # axis-wise spatial relation labels
  }

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


def _parse_direction(data: Any) -> list[float] | None:
    """Parse {"dx": ..., "dy": ..., "dz": ...} → normalized [dx, dy, dz]."""
    if not isinstance(data, dict):
        return None
    try:
        vec = np.array([float(data["dx"]), float(data["dy"]), float(data["dz"])], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).tolist()


def _parse_can_close(data: Any) -> bool | None:
    """Parse {"can_close": "yes"|"no"} → bool."""
    if isinstance(data, dict):
        val = data.get("can_close")
    else:
        val = data

    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"yes", "true", "1"}
    return None


def _parse_delta(data: Any) -> list[float] | None:
    """Parse {"dx": ..., "dy": ..., "dz": ...} → [dx, dy, dz] without normalizing.

    Used for Q5 (gripper-to-target offset), unlike Q4 which normalizes.
    """
    if not isinstance(data, dict):
        return None
    try:
        return [float(data["dx"]), float(data["dy"]), float(data["dz"])]
    except (KeyError, TypeError, ValueError):
        return None


# Valid label sets for Q6 spatial relation parsing
_X_LABELS = {"in_front", "behind", "aligned_x"}
_Y_LABELS = {"left", "right", "aligned_y"}
_Z_LABELS = {"above", "below", "aligned_z"}

# Alias normalization for Q6: map common VLM alternatives to canonical labels
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
    """Parse a single VLM response string into a structured answer dict."""

    def parse(self, response_text: str) -> dict:
        """Return a dict with keys:

        raw                  – original response string
        parsed_ok            – bool, True if at least one field was successfully parsed
        q1_object_pos        – [x, y, z] or None
        q2_gripper_pos       – [x, y, z] or None
        q3_can_close         – bool or None
        q4_next_dir          – [dx, dy, dz] (unit vector) or None
        q5_gripper_to_target – [dx, dy, dz] (raw offset, not normalized) or None
        q6_spatial_relation  – {"x": str, "y": str, "z": str} or None
        """
        result: dict = {
            "raw": response_text,
            "parsed_ok": False,
            "task_type": None,
            "q1_object_pos": None,
            "q1_dest_pos": None,
            "q2_gripper_pos": None,
            "q3_can_close": None,
            "q4_next_dir": None,
            "q5_gripper_to_target": None,
            "q6_spatial_relation": None,
        }

        data = _extract_json_from_text(response_text)
        if data is None:
            return result

        # task_type classification
        tt = data.get("task_type")
        if isinstance(tt, str) and tt in ("pick_and_place", "articulation"):
            result["task_type"] = tt

        # Q1 – source object position
        q1_raw = _resolve_subfield(
            data, "q1",
            ("object_position", "target_position", "source_object_position"),
        )
        result["q1_object_pos"] = _parse_xyz(q1_raw)

        # Q1_dest – destination position (None for articulation tasks)
        q1d_raw = _resolve_subfield(data, "q1_dest", ("destination", "dest_position", "place_position"))
        q1d = _parse_xyz(q1d_raw)
        # If model explicitly returned nulls, treat as no destination
        if q1d is not None and all(v is None or (isinstance(v, float) and v != v) for v in q1d):
            q1d = None
        result["q1_dest_pos"] = q1d

        # Q2 – gripper position
        q2_raw = _resolve_subfield(
            data, "q2",
            ("gripper_position", "eef_position", "end_effector_position"),
        )
        result["q2_gripper_pos"] = _parse_xyz(q2_raw)

        # Q3 – can close
        q3_raw = _resolve_subfield(data, "q3", ("can_close", "gripper_close"))
        result["q3_can_close"] = _parse_can_close(q3_raw)

        # Q4 – next direction (normalized unit vector)
        q4_raw = _resolve_subfield(data, "q4", ("next_direction", "direction", "move_direction"))
        result["q4_next_dir"] = _parse_direction(q4_raw)

        # Q5 – gripper-to-target offset vector (raw, not normalized)
        q5_raw = _resolve_subfield(data, "q5", ("gripper_to_target", "relative_position", "delta"))
        result["q5_gripper_to_target"] = _parse_delta(q5_raw)

        # Q6 – axis-wise spatial relation labels
        q6_raw = _resolve_subfield(data, "q6", ("spatial_relation", "relation", "spatial_relationship"))
        result["q6_spatial_relation"] = _parse_spatial_relation(q6_raw)

        result["parsed_ok"] = any(
            result[k] is not None
            for k in ("q1_object_pos", "q2_gripper_pos", "q3_can_close", "q4_next_dir")
        )
        return result
