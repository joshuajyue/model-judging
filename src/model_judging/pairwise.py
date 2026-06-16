from __future__ import annotations

from collections import defaultdict

from .types import PairwiseOutcome


def aggregate_pairwise_outcomes(outcomes: list[PairwiseOutcome], candidate_ids: list[str]) -> list[str]:
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)

    for outcome in outcomes:
        if outcome.winner_id is None:
            continue
        loser_id = outcome.right_id if outcome.winner_id == outcome.left_id else outcome.left_id
        wins[outcome.winner_id] += 1
        losses[loser_id] += 1

    return sorted(
        candidate_ids,
        key=lambda candidate_id: (
            wins[candidate_id],
            -losses[candidate_id],
            -candidate_ids.index(candidate_id),
        ),
        reverse=True,
    )

