#!/usr/bin/env python3
"""
Example of using the model-judging pipeline for:
1. Hard-truth checks (code execution)
2. Subjective judging with a judge panel
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from model_judging import (
    Candidate,
    CodeExecutionJudge,
    JudgingPipeline,
    TaskKind,
)


def example_hard_truth_code_judging():
    """Example: Judge two code solutions for correctness."""
    print("=" * 60)
    print("HARD-TRUTH: Code Execution Judging")
    print("=" * 60)

    judge = CodeExecutionJudge()
    pipeline = JudgingPipeline()

    # Two candidates with different code solutions
    candidates = [
        Candidate(
            id="gpt4o",
            name="GPT-4o",
            provider="OpenAI",
            model="gpt-4o",
            latency_ms=500,
            cost_per_1k_tokens=0.03,
            quality_factor=0.9,
        ),
        Candidate(
            id="claude",
            name="Claude 3.5",
            provider="Anthropic",
            model="claude-3-5-sonnet",
            latency_ms=1200,
            cost_per_1k_tokens=0.015,
            quality_factor=0.85,
        ),
    ]

    # A simple code artifact to judge
    code_artifact = {
        "code": "print(2 + 2)",
        "language": "python",
        "expected_output": "4",
    }

    decision = pipeline.judge(
        TaskKind.HARD_TRUTH,
        candidates,
        objective_judge=judge,
        artifact=code_artifact,
    )

    print(f"Winner: {decision.winner_id}")
    print(f"Ranking: {decision.ranking}")
    print(f"Rationale: {decision.rationale}")
    print(f"Objective checks: {decision.objective_checks}")
    print()


def example_subjective_utility_ranking():
    """Example: Rank candidates by quality/latency/cost utility."""
    print("=" * 60)
    print("SUBJECTIVE: Utility-Based Ranking")
    print("=" * 60)

    pipeline = JudgingPipeline()

    candidates = [
        Candidate(
            id="gpt4o",
            name="GPT-4o",
            provider="OpenAI",
            model="gpt-4o",
            latency_ms=500,
            cost_per_1k_tokens=0.03,
            quality_factor=0.95,
        ),
        Candidate(
            id="claude",
            name="Claude 3.5",
            provider="Anthropic",
            model="claude-3-5-sonnet",
            latency_ms=1200,
            cost_per_1k_tokens=0.015,
            quality_factor=0.92,
        ),
        Candidate(
            id="mini",
            name="GPT-4o-mini",
            provider="OpenAI",
            model="gpt-4o-mini",
            latency_ms=200,
            cost_per_1k_tokens=0.0001,
            quality_factor=0.70,
        ),
    ]

    decision = pipeline.judge(TaskKind.SUBJECTIVE, candidates)

    print(f"Winner: {decision.winner_id}")
    print(f"Ranking: {decision.ranking}")
    print(f"Rationale: {decision.rationale}")
    print()


if __name__ == "__main__":
    example_hard_truth_code_judging()
    example_subjective_utility_ranking()
    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
