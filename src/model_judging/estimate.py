"""Pre-run cost / time estimator for the benchmark.

Projects how many ``copilot`` calls a run will make, the resulting Copilot
premium-request spend, a marginal USD estimate, and a rough wall-clock time --
*before* spending anything. The call-count math mirrors the harness exactly:

* Phase 1 — answers:        ``n_prompts * n_models``
* Phase 1.5 — proof judging: ``n_semantic_prompts * n_models * n_judges``
* Phase 2 — subjective:      ``sum_over_subjective_prompts(matchups(n_models)) * n_judges``

where ``matchups`` comes from :func:`assess.plan_matchups`, the same decision the
ranker uses, so the estimate cannot drift from reality.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assess import plan_matchups
from .copilot_client import PREMIUM_REQUEST_USD
from .dataset import Prompt
from .registry import ModelSpec

# Rough throughput model derived from a live run (concurrency 8 -> ~0.56 calls/s):
# each call is a ~6 s cold-start process, and contention keeps real throughput to
# roughly half of the ideal ``concurrency / call_seconds``.
_CALL_SECONDS = 6.0
_CONTENTION_EFFICIENCY = 0.55


@dataclass(slots=True)
class RunEstimate:
    n_prompts: int
    n_models: int
    n_judges: int
    answer_calls: int
    semantic_judge_calls: int
    ranking_judge_calls: int
    answer_premium: float
    judge_premium: float

    @property
    def judge_calls(self) -> int:
        return self.semantic_judge_calls + self.ranking_judge_calls

    @property
    def total_calls(self) -> int:
        return self.answer_calls + self.judge_calls

    @property
    def total_premium(self) -> float:
        return self.answer_premium + self.judge_premium

    @property
    def est_usd(self) -> float:
        return self.total_premium * PREMIUM_REQUEST_USD

    def est_seconds(self, concurrency: int, throttle: float) -> float:
        concurrency = max(1, concurrency)
        rate = concurrency / _CALL_SECONDS * _CONTENTION_EFFICIENCY  # calls/sec
        if throttle and throttle > 0:
            rate = min(rate, 1.0 / throttle)  # spawn spacing also caps the rate
        return self.total_calls / rate if rate > 0 else 0.0


def estimate_run(
    prompts: list[Prompt],
    models: list[ModelSpec],
    judges: list[ModelSpec],
    *,
    matchup_rounds: int | None = None,
    max_round_robin: int = 4,
) -> RunEstimate:
    n_models = len(models)
    n_judges = len(judges)
    subjective = [p for p in prompts if p.is_subjective]
    semantic = [p for p in prompts if p.is_semantic_truth]

    answer_calls = len(prompts) * n_models

    # Every model's proof is assessed by every validity judge.
    semantic_judge_calls = len(semantic) * n_models * n_judges

    # Each subjective prompt: matchups over the answering models, every judge votes.
    matchups_total = sum(
        plan_matchups(n_models, matchup_rounds, max_round_robin)[1] for _ in subjective
    )
    ranking_judge_calls = matchups_total * n_judges

    sum_model_premium = sum(m.premium_per_call for m in models)
    sum_judge_premium = sum(j.premium_per_call for j in judges)

    answer_premium = len(prompts) * sum_model_premium
    judge_premium = (
        len(semantic) * n_models * sum_judge_premium  # proof judging
        + matchups_total * sum_judge_premium           # subjective matchups
    )

    return RunEstimate(
        n_prompts=len(prompts),
        n_models=n_models,
        n_judges=n_judges,
        answer_calls=answer_calls,
        semantic_judge_calls=semantic_judge_calls,
        ranking_judge_calls=ranking_judge_calls,
        answer_premium=answer_premium,
        judge_premium=judge_premium,
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} hr"


def format_estimate(
    est: RunEstimate,
    *,
    concurrency: int = 1,
    throttle: float = 0.0,
    show_cost: bool = True,
) -> str:
    """Render a human-readable estimate block."""
    lines = [
        "Estimate:",
        f"  prompts x models      {est.n_prompts} x {est.n_models}",
        f"  answer calls          {est.answer_calls}",
        f"  judge calls           {est.judge_calls}"
        f"  (proof {est.semantic_judge_calls} + ranking {est.ranking_judge_calls},"
        f" panel of {est.n_judges})",
        f"  TOTAL calls           {est.total_calls}",
    ]
    if show_cost:
        lines += [
            f"  premium requests      ~{est.total_premium:.0f}"
            f"  (answers {est.answer_premium:.0f} + judges {est.judge_premium:.0f})",
            f"  est. USD (@${PREMIUM_REQUEST_USD:g}/premium)  ~${est.est_usd:.2f}",
            f"  est. wall time        ~{_fmt_duration(est.est_seconds(concurrency, throttle))}"
            f"  (concurrency {concurrency}, throttle {throttle:g}s)",
        ]
    return "\n".join(lines)
