"""Benchmark orchestrator.

Calls every model on every prompt with identical context, grades hard-truth
answers objectively, ranks subjective answers via pairwise matchups, and
aggregates latency/cost statistics. Because all models go through the same
GitHub Models surface, the latency and cost numbers are meaningful as a
*relative* scale even though the proxy adds overhead.
"""

from __future__ import annotations

import random
import statistics
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


def run_benchmark(
    prompts: list[Prompt],
    models: list[ModelSpec],
    client: ModelClient,
    *,
    judge_models: list[ModelSpec] | None = None,
    judges: list[PairwiseTextJudge] | None = None,
    rng: random.Random | None = None,
    progress: ProgressFn | None = None,
) -> BenchmarkResult:
    rng = rng or random.Random(0)
    log = progress or (lambda _msg: None)

    if judges is None:
        panel = judge_models if judge_models is not None else models[:1]
        judges = [ModelMatchupJudge(client, jm) for jm in panel]

    result = BenchmarkResult()

    # Phase 1: collect every model's answer to every prompt.
    answers: dict[str, dict[str, str]] = {}  # prompt_id -> {model_id -> text}
    cell_index: dict[tuple[str, str], CellResult] = {}

    for prompt in prompts:
        answers[prompt.id] = {}
        for model in models:
            log(f"calling {model.name} on {prompt.id}")
            completion: CompletionResult = client.complete(model, prompt.rendered_prompt())
            cell = CellResult(
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
            result.cells.append(cell)
            cell_index[(prompt.id, model.id)] = cell

            if prompt.is_hard_truth and completion.ok:
                graded = grade_hard_truth(prompt, completion.text)
                cell.correct = graded.correct
                cell.detail = graded.detail
            elif prompt.is_hard_truth:
                cell.correct = False
                cell.detail = completion.error or "no response"
            elif completion.ok and completion.text:
                answers[prompt.id][model.id] = completion.text

    # Phase 2: rank subjective answers via pairwise matchups.
    for prompt in prompts:
        if not prompt.is_subjective:
            continue
        pool = answers[prompt.id]
        if len(pool) < 2:
            for model_id in pool:
                cell_index[(prompt.id, model_id)].rank = 1.0
            continue
        log(f"ranking {len(pool)} answers for {prompt.id}")
        ranks = rank_answers(prompt, pool, judges, rng=rng)
        worst = max(ranks.values()) + 1 if ranks else 1.0
        for model in models:
            cell = cell_index[(prompt.id, model.id)]
            cell.rank = ranks.get(model.id, worst if cell.error else None)

    return result
