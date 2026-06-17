"""Assessment of model answers.

Two paths, mirroring the design:

* **Hard truth** — for coding/calculus the answer is graded objectively by
  running the code or checking the number, yielding correct / incorrect.
* **Subjective** — for emails/essays/advice the answers are ranked relative to
  each other via blind, order-randomised pairwise matchups judged by a model
  (or a small panel), producing a per-prompt rank for every model.
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .client import ModelClient
from .dataset import Prompt
from .registry import ModelSpec

# --------------------------------------------------------------------------- #
# Hard-truth grading
# --------------------------------------------------------------------------- #

_PY_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FINAL_LINE = re.compile(r"FINAL:\s*([^\n]+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(slots=True)
class HardTruthResult:
    correct: bool
    detail: str


def extract_python_code(text: str) -> str:
    """Pull the first ```python``` block, falling back to any fenced block."""
    match = _PY_FENCE.search(text)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return text.strip()


def extract_numeric_answer(text: str) -> float | None:
    """Read the ``FINAL: <n>`` line, else fall back to the last number seen."""
    final = _FINAL_LINE.search(text)
    search_space = final.group(1) if final else text
    numbers = _NUMBER.findall(search_space)
    if not numbers:
        numbers = _NUMBER.findall(text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _run_python(code: str, harness: str, expected: str, timeout: float = 5.0) -> HardTruthResult:
    script = f"{code}\n\n{harness}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return HardTruthResult(False, "timeout (>5s)")
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        return HardTruthResult(False, f"runtime error: {err[-1] if err else 'non-zero exit'}")
    output = (result.stdout or "").strip()
    if output == expected.strip():
        return HardTruthResult(True, f"output '{output}' matched")
    return HardTruthResult(False, f"expected '{expected}', got '{output}'")


def grade_hard_truth(prompt: Prompt, answer_text: str) -> HardTruthResult:
    spec = prompt.evaluation
    kind = spec.get("type")

    if kind == "code_exec":
        if (spec.get("language") or "python") != "python":
            return HardTruthResult(False, f"unsupported language: {spec.get('language')}")
        code = extract_python_code(answer_text)
        if not code:
            return HardTruthResult(False, "no code found in answer")
        return _run_python(code, spec.get("harness", ""), str(spec.get("expected_output", "")))

    if kind == "numeric":
        value = extract_numeric_answer(answer_text)
        if value is None:
            return HardTruthResult(False, "no numeric answer found")
        expected = float(spec["expected"])
        tol = float(spec.get("tolerance", 0.0))
        if abs(value - expected) <= tol:
            return HardTruthResult(True, f"{value} matched {expected}")
        return HardTruthResult(False, f"expected {expected}, got {value}")

    return HardTruthResult(False, f"unknown evaluation type: {kind}")


# --------------------------------------------------------------------------- #
# Subjective matchup ranking
# --------------------------------------------------------------------------- #


class PairwiseTextJudge(Protocol):
    def judge(self, prompt: str, rubric: str | None, answer_a: str, answer_b: str) -> str:
        """Return 'A', 'B', or 'tie' for which answer better satisfies the prompt."""
        ...


_VERDICT = re.compile(r"\b(A|B|TIE)\b", re.IGNORECASE)


class ModelMatchupJudge:
    """A model-backed blind pairwise judge using the GitHub Models surface."""

    def __init__(self, client: ModelClient, judge_model: ModelSpec):
        self.client = client
        self.judge_model = judge_model

    def judge(self, prompt: str, rubric: str | None, answer_a: str, answer_b: str) -> str:
        rubric_line = f"\nJudging rubric: {rubric}\n" if rubric else "\n"
        instruction = (
            "You are an impartial judge. Two assistants answered the same user "
            "request. Decide which answer is better overall."
            f"{rubric_line}"
            f"\n[User request]\n{prompt}\n"
            f"\n[Answer A]\n{answer_a}\n"
            f"\n[Answer B]\n{answer_b}\n"
            "\nReply with exactly one token: A, B, or TIE."
        )
        result = self.client.complete(self.judge_model, instruction)
        if not result.ok:
            return "tie"
        match = _VERDICT.search(result.text.strip().upper())
        if not match:
            return "tie"
        token = match.group(1).upper()
        return {"A": "A", "B": "B", "TIE": "tie"}[token]


@dataclass(slots=True)
class MatchupTally:
    wins: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    games: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _ranks_from_scores(scores: dict[str, float]) -> dict[str, float]:
    """Assign 1-based ranks (best = 1); tied scores share the averaged rank."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    position = 1
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        span = list(range(position, position + (j - i + 1)))
        avg_rank = sum(span) / len(span)
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        position += j - i + 1
        i = j + 1
    return ranks


def rank_answers(
    prompt: Prompt,
    answers: dict[str, str],
    judges: list[PairwiseTextJudge],
    *,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Round-robin pairwise matchups over ``answers`` -> per-model rank (1=best).

    Answer order is randomised per matchup to remove positional bias, and every
    unordered pair is judged by every judge in the panel.
    """
    rng = rng or random.Random(0)
    model_ids = list(answers)
    if len(model_ids) < 2:
        return {mid: 1.0 for mid in model_ids}

    scores: dict[str, float] = {mid: 0.0 for mid in model_ids}
    for idx_a in range(len(model_ids)):
        for idx_b in range(idx_a + 1, len(model_ids)):
            left, right = model_ids[idx_a], model_ids[idx_b]
            for judge in judges:
                # Randomise which answer is shown as "A".
                if rng.random() < 0.5:
                    first, second = left, right
                else:
                    first, second = right, left
                verdict = judge.judge(
                    prompt.prompt, prompt.rubric, answers[first], answers[second]
                )
                if verdict == "A":
                    scores[first] += 1.0
                elif verdict == "B":
                    scores[second] += 1.0
                else:
                    scores[first] += 0.5
                    scores[second] += 0.5

    return _ranks_from_scores(scores)
