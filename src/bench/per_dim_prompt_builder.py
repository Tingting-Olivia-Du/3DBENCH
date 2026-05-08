"""Per-dimension prompt builder for the spatial QA benchmark.

Loads `prompts/spatial_qa_per_dim.yaml` and produces stripped system + user
prompts that ask only ONE dimension (q1, q2, q3, q4, q5, or q6).

Q1_dest is folded into Q1: a single Q1 prompt asks for both source and
destination positions.

Drop-in compatible with `bench.prompt_builder.PromptBuilder.build(...)`,
so `02_run_vlm_eval.py` can swap one for the other.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
DEFAULT_TEMPLATE = PROMPTS_DIR / "spatial_qa_per_dim.yaml"

VALID_DIMS = ("q1", "q2", "q3", "q4", "q5", "q6")


class PerDimPromptBuilder:
    """Build (system, user) prompts that target a single QA dimension.

    Usage:
        builder = PerDimPromptBuilder("q3")
        system, user = builder.build("pick up the red block")
    """

    def __init__(self, dim: str, template_path: str | Path | None = None):
        if dim not in VALID_DIMS:
            raise ValueError(f"dim must be one of {VALID_DIMS}, got {dim!r}")
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE
        with open(path) as f:
            self._templates = yaml.safe_load(f)
        if dim not in self._templates:
            raise KeyError(f"template {path} is missing section {dim!r}")
        self._dim = dim
        # Inline shared blocks once, so .format(task_description=...) is safe later.
        view_note = self._templates.get("shared_view_note", "").strip()
        frame_block = self._templates.get("shared_frame_block", "").strip()
        self._system = (
            self._templates[dim]["system"]
            .replace("{shared_view_note}", view_note)
            .replace("{shared_frame_block}", frame_block)
            .strip()
        )
        # The user template still has {task_description} as a real format placeholder.
        self._user_template = self._templates[dim]["user"]

    @property
    def dim(self) -> str:
        return self._dim

    @property
    def system_prompt(self) -> str:
        return self._system

    def build_user_prompt(self, task_description: str) -> str:
        return self._user_template.format(task_description=task_description).strip()

    def build(self, task_description: str) -> tuple[str, str]:
        return self.system_prompt, self.build_user_prompt(task_description)
