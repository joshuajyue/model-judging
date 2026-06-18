#!/usr/bin/env python3
"""CLI for the model benchmark.

Examples
--------
Offline dry-run (no auth, no network) -- exercises the whole pipeline:

    python run_benchmark.py run

Live run against the **Copilot CLI** models (uses your existing `copilot` login):

    python run_benchmark.py run --live

Live run against GitHub Models instead (needs a PAT with the 'models: read' scope):

    python run_benchmark.py run --live --provider github --token ghp_xxx

Verify the registry model IDs are accepted before a paid run:

    python run_benchmark.py verify-models                 # pings each model via the CLI
    python run_benchmark.py verify-models --provider github --token ghp_xxx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model_judging.client import GitHubModelsClient, fetch_catalog
from model_judging.copilot_client import CopilotCliClient, _resolve_copilot_home, verify_model
from model_judging.dataset import load_prompts
from model_judging.harness import run_benchmark
from model_judging.mock import MockModelClient
from model_judging.registry import default_judge_models, default_models, models_by_id
from model_judging.report import write_detailed_csv, write_summary_csv
from model_judging.sessions import clean_sessions, default_real_home, purge_home


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
        if args.provider == "github":
            client = GitHubModelsClient(token=args.token)
            print(f"LIVE run: {len(models)} models x {len(prompts)} prompts via GitHub Models")
        else:
            client = CopilotCliClient(
                min_interval=args.throttle,
                max_retries=args.max_retries,
            )
            print(
                f"LIVE run: {len(models)} models x {len(prompts)} prompts via Copilot CLI "
                f"(throttle={args.throttle:g}s, max_retries={args.max_retries}, "
                f"concurrency={args.concurrency})"
            )
            if args.provider != "github" and args.concurrency > 1:
                print(
                    "  NOTE: each call is a separate ~180 MB copilot process. High "
                    "concurrency saturates CPU (can freeze the desktop) AND inflates the "
                    "reported per-call latency through contention.\n"
                    "        -> For trustworthy latency numbers run with --concurrency 1.\n"
                    "        -> To go fast without freezing, keep --concurrency modest "
                    "(~4-6) and a small --throttle (e.g. 1) to stagger spawns.",
                    file=sys.stderr,
                )
    else:
        client = MockModelClient()
        print(f"DRY-RUN (offline mock): {len(models)} models x {len(prompts)} prompts")

    judge_models = None
    if args.judge:
        registry = models_by_id()
        judge_ids = [j.strip() for j in args.judge.split(",") if j.strip()]
        unknown = [j for j in judge_ids if j not in registry]
        if unknown:
            print(f"Unknown judge model id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        judge_models = [registry[j] for j in judge_ids]

    panel = judge_models if judge_models is not None else default_judge_models()
    print("Judge panel: " + ", ".join(m.name for m in panel))

    def progress(msg: str) -> None:
        if args.verbose:
            print(f"  {msg}")

    result = run_benchmark(
        prompts, models, client, judge_models=judge_models,
        matchup_rounds=args.matchup_rounds, concurrency=args.concurrency,
        progress=progress,
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
    if args.provider == "github":
        try:
            catalog = fetch_catalog(token=args.token)
        except Exception as exc:  # noqa: BLE001
            print(f"Catalog lookup failed: {exc}", file=sys.stderr)
            return 1
        catalog_ids = {str(m.get("id", "")).lower() for m in catalog}
        print(f"Catalog returned {len(catalog_ids)} models.\n")
        missing = 0
        for model in default_models():
            ok = model.id.lower() in catalog_ids
            mark = "OK " if ok else "MISSING"
            missing += 0 if ok else 1
            print(f"  [{mark}] {model.id}  ({model.tier})")
        return 1 if missing else 0

    # Copilot CLI provider: ping each model through the CLI.
    print("Verifying registry ids against the Copilot CLI (one ping per model)...\n")
    client = CopilotCliClient(timeout=120.0)
    missing = 0
    for model in default_models():
        ok, detail = verify_model(model, client=client)
        mark = "OK " if ok else "MISSING"
        missing += 0 if ok else 1
        print(f"  [{mark}] {model.id}  ({model.tier})  {detail}")
    return 1 if missing else 0


def cmd_clean_sessions(args: argparse.Namespace) -> int:
    # Wholesale-purge the isolated benchmark home (new runs land here).
    if args.purge_isolated:
        home = _resolve_copilot_home(None)
        removed = purge_home(home) if home else False
        print(f"{'Removed' if removed else 'Nothing to remove at'} isolated home: {home}")
        return 0

    # Surgically prune benchmark sessions from a shared store (e.g. legacy runs
    # that predate isolation, which landed in the real ~/.copilot).
    home_dir = args.home or default_real_home()
    cwd_match = args.cwd or (os.environ.get("TEMP") or os.getcwd())
    result = clean_sessions(
        home_dir=home_dir,
        cwd_match=cwd_match,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )
    print(f"Store: {os.path.join(home_dir, 'session-store.db')}")
    print(f"Match (cwd == {cwd_match}): {result.matched} sessions")
    if args.dry_run:
        print("Dry run -- nothing deleted. Re-run without --dry-run to apply.")
        return 0
    if result.backup_path:
        print(f"Backup: {result.backup_path}")
    print(f"Deleted {result.deleted} sessions and {result.folders_removed} state folders.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GitHub Models benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the benchmark and write CSVs")
    run.add_argument("--live", action="store_true", help="Call real models (not the mock)")
    run.add_argument(
        "--provider",
        choices=["copilot", "github"],
        default="copilot",
        help="Live backend: 'copilot' (Copilot CLI, default) or 'github' (GitHub Models)",
    )
    run.add_argument("--token", default=None, help="GitHub PAT for --provider github (else $GITHUB_TOKEN)")
    run.add_argument("--limit", type=int, default=None, help="Cap number of prompts")
    run.add_argument("--models", default=None, help="Comma-separated id/tier filter")
    run.add_argument("--judge", default=None,
                     help="Comma-separated matchup-judge model id(s). Default: a cheap "
                          "vendor-balanced panel (claude-haiku-4.5, gpt-5.4-mini, gemini-3.5-flash)")
    run.add_argument("--matchup-rounds", type=int, default=None,
                     help="Swiss rounds for subjective ranking (default: auto=ceil(log2 n); "
                          "0 = exhaustive round-robin)")
    run.add_argument("--throttle", type=float, default=1.0,
                     help="Min seconds between Copilot CLI call starts (default: 1.0; "
                          "set 0 to rely purely on --concurrency)")
    run.add_argument("--max-retries", type=int, default=6,
                     help="Retries with exponential backoff on a 429 (default: 6)")
    run.add_argument("--concurrency", type=int, default=1,
                     help="Parallel in-flight calls: answers (phase 1) and per-prompt "
                          "rankings (phase 2). Probed safe to ~20 on the Copilot CLI "
                          "(default: 1 = sequential)")
    run.add_argument("--out", default="results", help="Output directory (default: results)")
    run.add_argument("--verbose", action="store_true", help="Print per-call progress")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify-models", help="Check registry IDs are accepted by the provider")
    verify.add_argument(
        "--provider",
        choices=["copilot", "github"],
        default="copilot",
        help="'copilot' pings each model via the CLI; 'github' checks the live catalog",
    )
    verify.add_argument("--token", default=None, help="GitHub PAT for --provider github (else $GITHUB_TOKEN)")
    verify.set_defaults(func=cmd_verify_models)

    clean = sub.add_parser(
        "clean-sessions",
        help="Remove the throwaway Copilot sessions benchmark runs leave behind",
    )
    clean.add_argument(
        "--purge-isolated", action="store_true",
        help="Delete the whole isolated benchmark home (where new runs persist sessions)",
    )
    clean.add_argument(
        "--home", default=None,
        help="Copilot home holding session-store.db to clean (default: real ~/.copilot)",
    )
    clean.add_argument(
        "--cwd", default=None,
        help="Only delete sessions created from this cwd (default: system temp dir)",
    )
    clean.add_argument("--dry-run", action="store_true", help="Show matches without deleting")
    clean.add_argument("--no-backup", action="store_true", help="Skip the pre-delete DB backup")
    clean.set_defaults(func=cmd_clean_sessions)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
