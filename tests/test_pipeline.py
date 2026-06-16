from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_judging.pipeline import JudgingPipeline
from model_judging.types import Candidate, ObjectiveCheck, PairwiseOutcome, TaskKind


class FakeObjectiveJudge:
    def __init__(self, passing_ids: set[str]):
        self.passing_ids = passing_ids

    def evaluate(self, candidate: Candidate, artifact: object | None):
        return [ObjectiveCheck(name="check", passed=candidate.id in self.passing_ids)]


class FakePairwiseJudge:
    def __init__(self, preferred_ids: set[str]):
        self.preferred_ids = preferred_ids

    def compare(self, left: Candidate, right: Candidate, artifact: object | None):
        if left.id in self.preferred_ids and right.id not in self.preferred_ids:
            return PairwiseOutcome(left_id=left.id, right_id=right.id, winner_id=left.id)
        if right.id in self.preferred_ids and left.id not in self.preferred_ids:
            return PairwiseOutcome(left_id=left.id, right_id=right.id, winner_id=right.id)
        return PairwiseOutcome(left_id=left.id, right_id=right.id, winner_id=None)


class JudgingPipelineTests(unittest.TestCase):
    def test_ranks_by_quality_then_latency_and_cost(self) -> None:
        pipeline = JudgingPipeline()
        candidates = [
            Candidate(id="a", name="A", latency_ms=200, cost_per_1k_tokens=2.0, quality_factor=0.5),
            Candidate(id="b", name="B", latency_ms=100, cost_per_1k_tokens=1.0, quality_factor=0.7),
        ]

        decision = pipeline.judge(TaskKind.SUBJECTIVE, candidates)

        self.assertEqual("b", decision.winner_id)
        self.assertEqual(["b", "a"], decision.ranking)

    def test_hard_truth_uses_objective_checks(self) -> None:
        pipeline = JudgingPipeline()
        candidates = [
            Candidate(id="a", name="A", latency_ms=200, cost_per_1k_tokens=2.0, quality_factor=0.5),
            Candidate(id="b", name="B", latency_ms=100, cost_per_1k_tokens=1.0, quality_factor=0.7),
        ]

        decision = pipeline.judge(
            TaskKind.HARD_TRUTH,
            candidates,
            objective_judge=FakeObjectiveJudge({"a"}),
            artifact=None,
        )

        self.assertEqual("a", decision.winner_id)
        self.assertEqual(["a"], decision.ranking)

    def test_pairwise_panel_aggregates_votes(self) -> None:
        pipeline = JudgingPipeline()
        candidates = [
            Candidate(id="a", name="A"),
            Candidate(id="b", name="B"),
            Candidate(id="c", name="C"),
        ]

        decision = pipeline.judge_subjective_panel(
            candidates,
            judges=[FakePairwiseJudge({"b"})],
            artifact=None,
        )

        self.assertEqual("b", decision.winner_id)
        self.assertEqual("b", decision.ranking[0])


if __name__ == "__main__":
    unittest.main()
