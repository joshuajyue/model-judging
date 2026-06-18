"""Benchmark orchestrator.

Calls every model on every prompt with identical context, grades hard-truth
answers objectively, ranks subjective answers via pairwise matchups, and
aggregates latency/cost statistics. Because all models go through the same
GitHub Models surface, the latency and cost numbers are meaningful as a
*relative* scale even though the proxy adds overhead.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .assess import (
    ModelMatchupJudge,
    PairwiseTextJudge,
    grade_hard_truth,
    rank_answers,
)
from .client import CompletionResult, ModelClient
from .dataset import Prompt
from .registry import ModelSpec

ProgressFn = Callable[[str], None]


@dataclass(slots=True)
class CellResult:
    """One (model, prompt) outcome."""

    prompt_id: str
    category: str
    kind: str
    model_id: str
    model_name: str
    tier: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None
    # Hard-truth outcome:
    correct: bool | None = None
    detail: str = ""
    # Subjective outcome:
    rank: float | None = None
    answer: str = ""


@dataclass(slots=True)
class BenchmarkResult:
    cells: list[CellResult] = field(default_factory=list)

    def by_model(self) -> dict[str, list[CellResult]]:
        out: dict[str, list[CellResult]] = {}
        for cell in self.cells:
            out.setdefault(cell.model_id, []).append(cell)
        return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


@dataclass(slots=True)
class LatencyStats:
    avg: float
    p50: float
    p95: float

    @classmethod
    def of(cls, values: list[float]) -> "LatencyStats":
        if not values:
            return cls(0.0, 0.0, 0.0)
        return cls(
            avg=statistics.fmean(values),
            p50=_percentile(values, 50),
            p95=_percentile(values, 95),
        )


def _stable_seed(prompt_id: str, base: int = 0) -> int:
    """Deterministic per-prompt RNG seed (independent of Python's hash salt)."""
    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8]
    return base ^ int(digest, 16)


def _build_cell(prompt: Prompt, model: ModelSpec, completion: CompletionResult) -> CellResult:
    return CellResult(
        prompt_id=prompt.id,
        category=prompt.category,
        kind=prompt.kind,
        model_id=model.id,
        model_name=model.name,
        tier=model.tier,
        latency_ms=completion.latency_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.cost_usd,
        error=completion.error,
        answer=completion.text,
    )


def _apply_hard_truth(cell: CellResult, prompt: Prompt, completion: CompletionResult) -> None:
    if completion.ok:
        graded = grade_hard_truth(prompt, completion.text)
        cell.correct = graded.correct
        cell.detail = graded.detail
    else:
        cell.correct = False
        cell.detail = completion.error or "no response"


def run_benchmark(
    prompts: list[Prompt],
    models: list[ModelSpec],
    client: ModelClient,
    *,
    judge_models: list[ModelSpec] | None = None,
    judges: list[PairwiseTextJudge] | None = None,
    rng: random.Random | None = None,
    matchup_rounds: int | None = None,
    concurrency: int = 1,
    progress: ProgressFn | None = None,
) -> BenchmarkResult:
    """Run the benchmark.

    With ``concurrency > 1`` the independent answer calls (phase 1) and the
    per-prompt subjective rankings (phase 2) are dispatched across a thread pool.
    The Copilot CLI tolerates this comfortably (probed safe to ~20 concurrent),
    and each call is its own process, so threads are an appropriate fit. The
    shared client serialises only its short spawn-spacing critical section.

    To stay reproducible under parallelism, each subjective prompt is ranked with
    its own RNG seeded from the prompt id rather than the single shared ``rng``.
    """
    rng = rng or random.Random(0)
    raw_log = progress or (lambda _msg: None)
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            raw_log(msg)

    if judges is None:
        panel = judge_models if judge_models is not None else models[:1]
        judges = [ModelMatchupJudge(client, jm) for jm in panel]

    result = BenchmarkResult()
    answers: dict[str, dict[str, str]] = {p.id: {} for p in prompts}
    cell_index: dict[tuple[str, str], CellResult] = {}
    concurrency = max(1, concurrency)

    # ----------------------------------------------------------------- #
    # Phase 1: collect every model's answer to every prompt.
    # ----------------------------------------------------------------- #
    tasks = [(prompt, model) for prompt in prompts for model in models]

    def answer_task(prompt: Prompt, model: ModelSpec) -> CompletionResult:
        log(f"calling {model.name} on {prompt.id}")
        return client.complete(model, prompt.rendered_prompt())

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            completions = list(pool.map(lambda t: answer_task(*t), tasks))
    else:
        completions = [answer_task(prompt, model) for prompt, model in tasks]

    # Assemble cells in deterministic task order regardless of completion order.
    for (prompt, model), completion in zip(tasks, completions):
        cell = _build_cell(prompt, model, completion)
        result.cells.append(cell)
        cell_index[(prompt.id, model.id)] = cell
        if prompt.is_hard_truth:
            _apply_hard_truth(cell, prompt, completion)
        elif completion.ok and completion.text:
            answers[prompt.id][model.id] = completion.text

    # ----------------------------------------------------------------- #
    # Phase 2: rank subjective answers via pairwise matchups.
    # ----------------------------------------------------------------- #
    rankable = [
        p for p in prompts if p.is_subjective and len(answers[p.id]) >= 2
    ]
    # Single-answer subjective prompts trivially rank 1.
    for prompt in prompts:
        if prompt.is_subjective and len(answers[prompt.id]) == 1:
            only_id = next(iter(answers[prompt.id]))
            cell_index[(prompt.id, only_id)].rank = 1.0

    def rank_task(prompt: Prompt) -> tuple[str, dict[str, float]]:
        log(f"ranking {len(answers[prompt.id])} answers for {prompt.id}")
        prompt_rng = (
            rng if concurrency == 1 else random.Random(_stable_seed(prompt.id))
        )
        ranks = rank_answers(
            prompt, answers[prompt.id], judges, rng=prompt_rng, rounds=matchup_rounds
        )
        return prompt.id, ranks

    if concurrency > 1 and rankable:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            ranked = list(pool.map(rank_task, rankable))
    else:
        ranked = [rank_task(p) for p in rankable]

    ranks_by_prompt = dict(ranked)
    for prompt in rankable:
        ranks = ranks_by_prompt[prompt.id]
        worst = max(ranks.values()) + 1 if ranks else 1.0
        for model in models:
            cell = cell_index[(prompt.id, model.id)]
            cell.rank = ranks.get(model.id, worst if cell.error else None)

    return result
