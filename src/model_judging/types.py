from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskKind(str, Enum):
    HARD_TRUTH = "hard_truth"
    SUBJECTIVE = "subjective"


@dataclass(slots=True)
class Candidate:
    id: str
    name: str
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    cost_per_1k_tokens: float | None = None
    quality_factor: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectiveCheck:
    name: str
    passed: bool
    details: str | None = None


@dataclass(slots=True)
class JudgeDecision:
    winner_id: str | None
    ranking: list[str]
    rationale: str
    objective_checks: list[ObjectiveCheck] = field(default_factory=list)


@dataclass(slots=True)
class PairwiseOutcome:
    left_id: str
    right_id: str
    winner_id: str | None
    judge_id: str | None = None
    rationale: str | None = None
