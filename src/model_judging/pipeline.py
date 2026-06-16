from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from .pairwise import aggregate_pairwise_outcomes
from .scoring import ScoringWeights, score_candidate
from .types import Candidate, JudgeDecision, ObjectiveCheck, PairwiseOutcome, TaskKind


class ObjectiveJudge(Protocol):
    def evaluate(self, candidate: Candidate, artifact: object | None) -> list[ObjectiveCheck]:
        ...


class PairwiseJudge(Protocol):
    def compare(
        self,
        left: Candidate,
        right: Candidate,
        artifact: object | None,
    ) -> PairwiseOutcome:
        ...


@dataclass(slots=True)
class JudgingPipeline:
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    def rank_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        latencies = [c.latency_ms for c in candidates if c.latency_ms is not None]
        costs = [c.cost_per_1k_tokens for c in candidates if c.cost_per_1k_tokens is not None]
        latency_bounds = (min(latencies), max(latencies)) if latencies else (0.0, 1.0)
        cost_bounds = (min(costs), max(costs)) if costs else (0.0, 1.0)

        return sorted(
            candidates,
            key=lambda c: score_candidate(
                c,
                latency_bounds=latency_bounds,
                cost_bounds=cost_bounds,
                weights=self.weights,
            ),
            reverse=True,
        )

    def judge(
        self,
        task_kind: TaskKind,
        candidates: list[Candidate],
        *,
        objective_judge: ObjectiveJudge | None = None,
        artifact: object | None = None,
    ) -> JudgeDecision:
        if not candidates:
            return JudgeDecision(winner_id=None, ranking=[], rationale="No candidates provided.")

        if task_kind == TaskKind.HARD_TRUTH:
            if objective_judge is None:
                raise ValueError("objective_judge is required for hard truth tasks")

            checks_by_candidate: dict[str, list[ObjectiveCheck]] = {}
            passing: list[Candidate] = []
            for candidate in candidates:
                checks = objective_judge.evaluate(candidate, artifact)
                checks_by_candidate[candidate.id] = checks
                if all(check.passed for check in checks):
                    passing.append(candidate)

            ranked = self.rank_candidates(passing or candidates)
            winner = ranked[0] if ranked else None
            rationale = "Objective checks passed." if passing else "No candidate passed all objective checks."
            checks = checks_by_candidate.get(winner.id, []) if winner else []
            return JudgeDecision(
                winner_id=winner.id if winner else None,
                ranking=[candidate.id for candidate in ranked],
                rationale=rationale,
                objective_checks=checks,
            )

        ranked = self.rank_candidates(candidates)
        winner = ranked[0]
        return JudgeDecision(
            winner_id=winner.id,
            ranking=[candidate.id for candidate in ranked],
            rationale="Ranked by quality, latency, and cost.",
        )

    def judge_subjective_panel(
        self,
        candidates: list[Candidate],
        *,
        judges: list[PairwiseJudge],
        artifact: object | None = None,
    ) -> JudgeDecision:
        if len(candidates) < 2:
            return self.judge(TaskKind.SUBJECTIVE, candidates, artifact=artifact)
        if not judges:
            raise ValueError("judges is required for panel judging")

        outcomes: list[PairwiseOutcome] = []
        for left, right in combinations(candidates, 2):
            for judge in judges:
                outcomes.append(judge.compare(left, right, artifact))

        ranking = aggregate_pairwise_outcomes(outcomes, [candidate.id for candidate in candidates])
        winner = ranking[0] if ranking else None
        return JudgeDecision(
            winner_id=winner,
            ranking=ranking,
            rationale="Aggregated pairwise judgments from a judge panel.",
        )
