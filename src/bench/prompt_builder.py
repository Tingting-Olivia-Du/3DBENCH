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

    @staticmethod
    def _format_object_names(object_names) -> str:
        """Render a list of object names as a bulleted block for the prompt."""
        if not object_names:
            # No named objects resolved — let the model infer from the task text.
            return "(identify every object named in the task description above)"
        return "\n".join(f'    - "{n}"' for n in object_names)

    def build_user_prompt(
        self,
        task_description: str,
        object_names: list[str] | None = None,
    ) -> str:
        """Return the formatted user message for a given task description.

        ``object_names`` is the explicit list of objects the model must report
        coordinates for (e.g. ["alphabet soup", "tomato sauce", "basket"]).
        """
        return self._templates["user_template"].format(
            task_description=task_description,
            object_names=self._format_object_names(object_names),
        ).strip()

    def build(
        self,
        task_description: str,
        object_names: list[str] | None = None,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a task frame.

        The caller is responsible for attaching the image to the user message
        in whatever format the target VLM expects.
        """
        return self.system_prompt, self.build_user_prompt(
            task_description, object_names
        )
