"""Parse VLM text responses into structured ground-truth-comparable dicts.

Expected VLM output (single JSON object):
  {
    "q1": {"x": float, "y": float, "z": float},   # target object position
    "q2": {"x": float, "y": float, "z": float},   # gripper position
    "q3": {"can_close": "yes"|"no"},               # gripper close label
    "q4": {"dx": float, "dy": float, "dz": float}  # next move direction
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

        raw           – original response string
        parsed_ok     – bool, True if at least one field was successfully parsed
        q1_object_pos – [x, y, z] or None
        q2_gripper_pos– [x, y, z] or None
        q3_can_close  – bool or None
        q4_next_dir   – [dx, dy, dz] (unit vector) or None
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

        # Q4 – next direction
        q4_raw = _resolve_subfield(data, "q4", ("next_direction", "direction", "move_direction"))
        result["q4_next_dir"] = _parse_direction(q4_raw)

        result["parsed_ok"] = any(
            result[k] is not None
            for k in ("q1_object_pos", "q2_gripper_pos", "q3_can_close", "q4_next_dir")
        )
        return result
