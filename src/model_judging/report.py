"""CSV reporting for benchmark results.

Two files are produced:

* **detailed** — one row per (model, prompt): ``correct``/``incorrect`` for
  hard-truth prompts or the per-prompt ``rank`` for subjective prompts, plus
  latency / token / cost columns.
* **summary** — one row per (model, category): pass-rate or average rank, with
  latency avg/p50/p95 and average cost, the table used for routing weights.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .harness import BenchmarkResult, CellResult, LatencyStats


def _score_cell(cell: CellResult) -> str:
    if cell.error:
        return "error"
    if cell.kind == "hard_truth":
        return "correct" if cell.correct else "incorrect"
    return "" if cell.rank is None else f"{cell.rank:g}"


def write_detailed_csv(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "tier", "category", "prompt_id", "kind", "score",
        "latency_ms", "input_tokens", "output_tokens", "cost_usd", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in result.cells:
            writer.writerow(
                {
                    "model_name": cell.model_name,
                    "tier": cell.tier,
                    "category": cell.category,
                    "prompt_id": cell.prompt_id,
                    "kind": cell.kind,
                    "score": _score_cell(cell),
                    "latency_ms": round(cell.latency_ms, 1),
                    "input_tokens": cell.input_tokens,
                    "output_tokens": cell.output_tokens,
                    "cost_usd": round(cell.cost_usd, 6),
                    "note": cell.error or cell.detail,
                }
            )
    return path


def write_summary_csv(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[CellResult]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for cell in result.cells:
        groups.setdefault((cell.model_id, cell.category), []).append(cell)
        meta[cell.model_id] = (cell.model_name, cell.tier)

    fields = [
        "model_name", "tier", "category", "kind", "metric", "value", "n",
        "latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "avg_cost_usd",
    ]
    rows = []
    for (model_id, category), cells in groups.items():
        model_name, tier = meta[model_id]
        kind = cells[0].kind
        latency = LatencyStats.of([c.latency_ms for c in cells if c.error is None])
        costs = [c.cost_usd for c in cells if c.error is None]
        avg_cost = statistics.fmean(costs) if costs else 0.0

        if kind == "hard_truth":
            graded = [c for c in cells if c.correct is not None]
            value = (
                statistics.fmean([1.0 if c.correct else 0.0 for c in graded])
                if graded else 0.0
            )
            metric = "pass_rate"
        else:
            ranked = [c.rank for c in cells if c.rank is not None]
            value = statistics.fmean(ranked) if ranked else 0.0
            metric = "avg_rank"

        rows.append(
            {
                "model_name": model_name,
                "tier": tier,
                "category": category,
                "kind": kind,
                "metric": metric,
                "value": round(value, 4),
                "n": len(cells),
                "latency_avg_ms": round(latency.avg, 1),
                "latency_p50_ms": round(latency.p50, 1),
                "latency_p95_ms": round(latency.p95, 1),
                "avg_cost_usd": round(avg_cost, 6),
            }
        )

    rows.sort(key=lambda r: (r["category"], r["tier"], r["model_name"]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
