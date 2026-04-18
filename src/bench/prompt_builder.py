"""Build prompts for the spatial QA benchmark.

Each frame gets a single VLM call with four questions; the VLM is expected
to return one JSON object with keys q1–q4.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PromptBuilder:
    """Loads the YAML template and formats system + user prompts per frame."""

    def __init__(self, template_path: str | Path | None = None):
        path = Path(template_path) if template_path else PROMPTS_DIR / "spatial_qa.yaml"
        with open(path) as f:
            self._templates = yaml.safe_load(f)

    @property
    def system_prompt(self) -> str:
        return self._templates["system"].strip()

    def build_user_prompt(self, task_description: str) -> str:
        """Return the formatted user message for a given task description."""
        return self._templates["user_template"].format(
            task_description=task_description
        ).strip()

    def build(self, task_description: str) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a task frame.

        The caller is responsible for attaching the image to the user message
        in whatever format the target VLM expects.
        """
        return self.system_prompt, self.build_user_prompt(task_description)
