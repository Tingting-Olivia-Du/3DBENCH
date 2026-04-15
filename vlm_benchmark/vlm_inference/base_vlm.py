"""
Base class for Vision-Language Model inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import re


@dataclass
class VLMResponse:
    """Response from VLM inference."""
    raw_text: str
    parsed_json: Optional[Dict] = None
    success: bool = False
    error_message: Optional[str] = None


class BaseVLM(ABC):
    """Abstract base class for VLM inference."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def inference(
        self,
        images: List[Union[str, Path]],
        prompt: str,
    ) -> VLMResponse:
        """
        Run inference on images with prompt.

        Args:
            images: List of image paths (agentview, eye_in_hand)
            prompt: Text prompt

        Returns:
            VLMResponse with raw text and parsed JSON
        """
        pass

    def parse_json_response(self, text: str) -> Optional[Dict]:
        """
        Parse JSON from VLM response text.

        Handles cases where JSON is embedded in other text.
        """
        # Try direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text using regex
        # Pattern for {...}
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Try to find JSON with nested braces
        # Pattern for {...} allowing nested content
        nested_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
        matches = re.findall(nested_pattern, text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def validate_coordinate_response(self, parsed: Dict) -> bool:
        """Validate coordinate prediction response."""
        required_keys = ["x", "y", "z"]
        if not all(k in parsed for k in required_keys):
            return False
        try:
            for k in required_keys:
                float(parsed[k])
            return True
        except (ValueError, TypeError):
            return False

    def validate_grasp_response(self, parsed: Dict) -> bool:
        """Validate grasp timing response."""
        if "can_close" not in parsed:
            return False
        return isinstance(parsed["can_close"], bool)

    def validate_direction_response(self, parsed: Dict) -> bool:
        """Validate direction prediction response."""
        required_keys = ["dx", "dy", "dz"]
        if not all(k in parsed for k in required_keys):
            return False
        try:
            for k in required_keys:
                float(parsed[k])
            return True
        except (ValueError, TypeError):
            return False
