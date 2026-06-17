"""Prompt dataset loading and answer-format directives.

Prompts deliberately contain typos and casual phrasing to mirror real user
input. Hard-truth prompts carry an ``evaluation`` spec describing how to grade
the answer objectively; subjective prompts carry a ``rubric`` used by the judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "data" / "prompts.json"

# Appended to the user prompt for hard-truth tasks so answer extraction is
# deterministic. The realistic/typo-laden prompt itself is left untouched.
ANSWER_FORMAT_DIRECTIVES: dict[str, str] = {
    "code": (
        "\n\n---\nReturn your solution as a single self-contained Python function "
        "named `solve` inside exactly one ```python code block. Do not include "
        "example usage, tests, input() calls, or extra print statements."
    ),
    "final_line": (
        "\n\n---\nThink it through, then end your reply with a line in exactly "
        "this format:\nFINAL: <answer>\nwhere <answer> is just the numeric result."
    ),
}


@dataclass(slots=True)
class Prompt:
    id: str
    category: str
    kind: str  # "hard_truth" | "subjective"
    prompt: str
    answer_format: str | None = None
    evaluation: dict = field(default_factory=dict)
    rubric: str | None = None

    @property
    def is_hard_truth(self) -> bool:
        return self.kind == "hard_truth"

    @property
    def is_subjective(self) -> bool:
        return self.kind == "subjective"

    def rendered_prompt(self) -> str:
        """The prompt actually sent to the model, incl. any format directive."""
        directive = ANSWER_FORMAT_DIRECTIVES.get(self.answer_format or "", "")
        return self.prompt + directive


def load_prompts(path: str | Path | None = None) -> list[Prompt]:
    data_path = Path(path) if path else _DATA_FILE
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return [
        Prompt(
            id=item["id"],
            category=item["category"],
            kind=item["kind"],
            prompt=item["prompt"],
            answer_format=item.get("answer_format"),
            evaluation=item.get("evaluation", {}),
            rubric=item.get("rubric"),
        )
        for item in raw["prompts"]
    ]
