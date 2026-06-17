"""Assessment of model answers.

Two paths, mirroring the design:

* **Hard truth** — for coding/calculus the answer is graded objectively by
  running the code or checking the number, yielding correct / incorrect.
* **Subjective** — for emails/essays/advice the answers are ranked relative to
  each other via blind, order-randomised pairwise matchups judged by a model
  (or a small panel), producing a per-prompt rank for every model.
"""

from __future__ import annotations

import math
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


def _ranks_from_sorted(ordered: list[tuple[str, object]]) -> dict[str, float]:
    """Assign 1-based ranks (best first) over a pre-sorted list of (id, key).

    Entries whose sort key is exactly equal share the averaged rank.
    """
    ranks: dict[str, float] = {}
    i = 0
    position = 1
    n = len(ordered)
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        span = list(range(position, position + (j - i + 1)))
        avg_rank = sum(span) / len(span)
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        position += j - i + 1
        i = j + 1
    return ranks


def _ranks_from_scores(scores: dict[str, float]) -> dict[str, float]:
    """Assign 1-based ranks (best = 1); tied scores share the averaged rank."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return _ranks_from_sorted(ordered)


def _play_match(
    prompt: Prompt,
    answers: dict[str, str],
    left: str,
    right: str,
    judges: list[PairwiseTextJudge],
    rng: random.Random,
    scores: dict[str, float],
) -> None:
    """Judge ``left`` vs ``right`` with every judge and add win/tie points.

    The answer shown as "A" is randomised per matchup to remove positional bias.
    """
    for judge in judges:
        if rng.random() < 0.5:
            first, second = left, right
        else:
            first, second = right, left
        verdict = judge.judge(prompt.prompt, prompt.rubric, answers[first], answers[second])
        if verdict == "A":
            scores[first] += 1.0
        elif verdict == "B":
            scores[second] += 1.0
        else:
            scores[first] += 0.5
            scores[second] += 0.5


def _round_robin_rank(
    prompt: Prompt,
    answers: dict[str, str],
    judges: list[PairwiseTextJudge],
    rng: random.Random,
) -> dict[str, float]:
    """Exact ranking: every unordered pair is judged by every judge."""
    model_ids = list(answers)
    scores: dict[str, float] = {mid: 0.0 for mid in model_ids}
    for idx_a in range(len(model_ids)):
        for idx_b in range(idx_a + 1, len(model_ids)):
            _play_match(
                prompt, answers, model_ids[idx_a], model_ids[idx_b], judges, rng, scores
            )
    return _ranks_from_scores(scores)


def _swiss_rank(
    prompt: Prompt,
    answers: dict[str, str],
    judges: list[PairwiseTextJudge],
    rng: random.Random,
    rounds: int,
) -> dict[str, float]:
    """Swiss-system ranking: ~``rounds * (n // 2)`` matchups instead of C(n, 2).

    Each round pairs answers of similar running score (winners meet winners),
    avoiding rematches; final standings are ranked by score with a Buchholz
    tiebreak (sum of opponents' scores) so close finishers separate cleanly.
    """
    ids = list(answers)
    scores: dict[str, float] = {mid: 0.0 for mid in ids}
    opponents: dict[str, list[str]] = {mid: [] for mid in ids}
    played: set[frozenset[str]] = set()
    had_bye: set[str] = set()

    for _ in range(rounds):
        order = sorted(ids, key=lambda m: (scores[m], rng.random()), reverse=True)
        unpaired = order[:]

        if len(unpaired) % 2 == 1:
            # Bye for the lowest-ranked answer that hasn't had one; worth half a
            # point (the expected value of a coin-flip game) to avoid distortion.
            byer = next((m for m in reversed(unpaired) if m not in had_bye), unpaired[-1])
            had_bye.add(byer)
            scores[byer] += 0.5
            unpaired.remove(byer)

        while unpaired:
            a = unpaired.pop(0)
            partner_idx = next(
                (k for k, b in enumerate(unpaired) if frozenset((a, b)) not in played),
                0,  # everyone already played: allow a rematch rather than stall
            )
            b = unpaired.pop(partner_idx)
            played.add(frozenset((a, b)))
            opponents[a].append(b)
            opponents[b].append(a)
            _play_match(prompt, answers, a, b, judges, rng, scores)

    buchholz = {m: sum(scores[o] for o in opponents[m]) for m in ids}
    keyed = sorted(
        ((m, (scores[m], buchholz[m])) for m in ids),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return _ranks_from_sorted(keyed)


def rank_answers(
    prompt: Prompt,
    answers: dict[str, str],
    judges: list[PairwiseTextJudge],
    *,
    rng: random.Random | None = None,
    rounds: int | None = None,
    max_round_robin: int = 4,
) -> dict[str, float]:
    """Rank ``answers`` via pairwise matchups -> per-model rank (1 = best).

    Round-robin (every pair) is exact but costs C(n, 2) judge calls, which gets
    expensive (and trips rate limits) for large panels. Instead this runs a
    Swiss-system tournament that reaches a stable ranking in only
    ``ceil(log2(n))`` rounds -- "just enough" matchups -- while small fields fall
    back to exact round-robin since it is already cheap there.

    Parameters
    ----------
    rounds:
        Swiss rounds. ``None`` -> ``ceil(log2(n))`` (auto). ``0`` -> force exact
        round-robin.
    max_round_robin:
        Panels with at most this many answers always use round-robin.
    """
    rng = rng or random.Random(0)
    ids = list(answers)
    n = len(ids)
    if n < 2:
        return {mid: 1.0 for mid in ids}

    if rounds is None:
        rounds = max(1, math.ceil(math.log2(n)))

    round_robin_games = n * (n - 1) // 2
    swiss_games = rounds * (n // 2)
    if rounds <= 0 or n <= max_round_robin or round_robin_games <= swiss_games:
        return _round_robin_rank(prompt, answers, judges, rng)
    return _swiss_rank(prompt, answers, judges, rng, rounds)
