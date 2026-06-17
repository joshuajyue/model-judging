#!/usr/bin/env python3
"""CLI for the model benchmark.

Examples
--------
Offline dry-run (no token, no network) — exercises the whole pipeline:

    python run_benchmark.py run

Live run against GitHub Models (needs a PAT with the 'models: read' scope):

    python run_benchmark.py run --live --token ghp_xxx
    GITHUB_TOKEN=ghp_xxx python run_benchmark.py run --live

Verify the registry model IDs against the live catalog before a paid run:

    python run_benchmark.py verify-models --token ghp_xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model_judging.client import GitHubModelsClient, fetch_catalog
from model_judging.dataset import load_prompts
from model_judging.harness import run_benchmark
from model_judging.mock import MockModelClient
from model_judging.registry import default_models, models_by_id
from model_judging.report import write_detailed_csv, write_summary_csv


def _select_models(filter_expr: str | None):
    models = default_models()
    if not filter_expr:
        return models
    terms = [t.strip().lower() for t in filter_expr.split(",") if t.strip()]
    return [m for m in models if any(t in m.id.lower() or t in m.tier.lower() for t in terms)]


def cmd_run(args: argparse.Namespace) -> int:
    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]
    models = _select_models(args.models)
    if not models:
        print("No models matched the filter.", file=sys.stderr)
        return 2

    if args.live:
        client = GitHubModelsClient(token=args.token)
        print(f"LIVE run: {len(models)} models x {len(prompts)} prompts via GitHub Models")
    else:
        client = MockModelClient()
        print(f"DRY-RUN (offline mock): {len(models)} models x {len(prompts)} prompts")

    judge_models = None
    if args.judge:
        registry = models_by_id()
        if args.judge not in registry:
            print(f"Unknown judge model id: {args.judge}", file=sys.stderr)
            return 2
        judge_models = [registry[args.judge]]

    def progress(msg: str) -> None:
        if args.verbose:
            print(f"  {msg}")

    result = run_benchmark(
        prompts, models, client, judge_models=judge_models, progress=progress
    )

    out_dir = Path(args.out)
    detailed = write_detailed_csv(result, out_dir / "detailed.csv")
    summary = write_summary_csv(result, out_dir / "summary.csv")

    errors = [c for c in result.cells if c.error]
    print(f"\nWrote {detailed}")
    print(f"Wrote {summary}")
    print(f"Cells: {len(result.cells)}  Errors: {len(errors)}")
    if errors:
        print("First error:", errors[0].error[:200])
    return 0


def cmd_verify_models(args: argparse.Namespace) -> int:
    try:
        catalog = fetch_catalog(token=args.token)
    except Exception as exc:  # noqa: BLE001
        print(f"Catalog lookup failed: {exc}", file=sys.stderr)
        return 1
    catalog_ids = {str(m.get("id", "")).lower() for m in catalog}
    print(f"Catalog returned {len(catalog_ids)} models.\n")
    for model in default_models():
        ok = model.id.lower() in catalog_ids
        mark = "OK " if ok else "MISSING"
        print(f"  [{mark}] {model.id}  ({model.tier})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GitHub Models benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the benchmark and write CSVs")
    run.add_argument("--live", action="store_true", help="Call GitHub Models for real")
    run.add_argument("--token", default=None, help="GitHub PAT (else uses $GITHUB_TOKEN)")
    run.add_argument("--limit", type=int, default=None, help="Cap number of prompts")
    run.add_argument("--models", default=None, help="Comma-separated id/tier filter")
    run.add_argument("--judge", default=None, help="Model id to use as the matchup judge")
    run.add_argument("--out", default="results", help="Output directory (default: results)")
    run.add_argument("--verbose", action="store_true", help="Print per-call progress")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify-models", help="Check registry IDs against the live catalog")
    verify.add_argument("--token", default=None, help="GitHub PAT (else uses $GITHUB_TOKEN)")
    verify.set_defaults(func=cmd_verify_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
