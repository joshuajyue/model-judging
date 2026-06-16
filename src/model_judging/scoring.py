from __future__ import annotations

from dataclasses import dataclass

from .types import Candidate


@dataclass(slots=True)
class ScoringWeights:
    quality: float = 1.0
    latency: float = 0.35
    cost: float = 0.2


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (value - low) / (high - low)


def score_candidate(
    candidate: Candidate,
    *,
    quality_floor: float = 0.0,
    latency_bounds: tuple[float, float] = (0.0, 1.0),
    cost_bounds: tuple[float, float] = (0.0, 1.0),
    weights: ScoringWeights | None = None,
) -> float:
    weights = weights or ScoringWeights()
    quality = candidate.quality_factor if candidate.quality_factor is not None else quality_floor
    latency = candidate.latency_ms if candidate.latency_ms is not None else latency_bounds[1]
    cost = candidate.cost_per_1k_tokens if candidate.cost_per_1k_tokens is not None else cost_bounds[1]

    latency_score = 1.0 - _normalize(latency, *latency_bounds)
    cost_score = 1.0 - _normalize(cost, *cost_bounds)
    return (
        weights.quality * quality
        + weights.latency * latency_score
        + weights.cost * cost_score
    )

