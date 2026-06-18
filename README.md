# model-judging

Prototype for a model judging and routing pipeline with:

- **Hard-truth checks** for code/math/format validation
- **Subjective ranking** for essays, emails, messages, and advice  
- **Lightweight scoring** for latency, cost, and quality
- **Pairwise aggregation** for multi-judge panels
- **GitHub Models integration** for real LLM judge calls
- **Complexity detection** for request routing (SIMPLE/MEDIUM/COMPLEX/REASONING)
- **Benchmark harness** that ranks the Copilot CLI models per prompt category

## Benchmark harness

The harness calls a panel of models on a shared prompt set, grades hard-truth
answers objectively (run the code / check the number) and ranks subjective
answers via blind pairwise matchups, then writes a CSV per model and per
category. Because every model goes through the same surface, latency and cost
are meaningful as a **relative** scale.

Subjective ranking uses a **Swiss-system tournament** (≈`ceil(log2 n)` rounds)
rather than exhaustive round-robin, so an 8-model panel needs ~12 matchups per
prompt instead of 28 — far fewer Copilot calls (and far less rate-limit
pressure). Small panels (≤4) still use exact round-robin. Pass
`--matchup-rounds 0` to force full round-robin, or a specific round count.

Each matchup is judged by a **cheap, vendor-balanced judge panel** —
`claude-haiku-4.5`, `gpt-5.4-mini`, `gemini-3.1-pro-preview` — and the verdicts
are aggregated. One cheap-to-run model per vendor keeps judging cost low while
balancing per-vendor self-preference bias, so the aggregate verdict is more
neutral than any single judge (including an expensive one). The Google seat uses
Gemini 3.1 Pro rather than Flash because **Flash bills ~14× premium on Copilot**
while Pro bills ~1×. Override with `--judge id1,id2,...`
(e.g. `--judge gpt-5.4-mini` for a single cheap judge).

The default provider is the **GitHub Copilot CLI** (`copilot -p --model <id>`),
which exposes the frontier preview models (`gpt-5.5`, `claude-opus-4.8`,
`gemini-3.1-pro-preview`, …) that the public GitHub Models surface does not.
It reuses your existing `copilot` login — no PAT or token handling — and runs
each prompt with tools disabled so it measures raw model quality. A
`--provider github` path is still available for the GitHub Models API.

### Run it

```bash
# Offline dry-run — no auth, no network, deterministic mock answers:
python run_benchmark.py run

# Live run against the Copilot CLI models (uses your existing `copilot` login):
python run_benchmark.py run --live

# Faster: run several calls in parallel. Keep it modest + stagger spawns so the
# desktop doesn't freeze and latency stays meaningful (see "Concurrency" below):
python run_benchmark.py run --live --concurrency 6 --throttle 1

# Live run against GitHub Models instead (needs a PAT with the 'models: read' scope):
python run_benchmark.py run --live --provider github --token ghp_xxxxxxxx

# Confirm every registry model id is accepted before a paid run:
python run_benchmark.py verify-models                       # pings each model via the CLI
python run_benchmark.py verify-models --provider github --token ghp_xxxxxxxx
```

Useful flags: `--limit N` (cap prompts), `--models claude,openai-low` (filter by
id/tier), `--judge id1,id2,...` (override the matchup-judge panel),
`--concurrency N` (parallel calls), `--out DIR`, `--verbose`.

### Concurrency (and its effect on latency)

By default the harness runs sequentially. `--concurrency N` dispatches up to `N`
calls at once — both the phase-1 answer calls (every model × prompt is
independent) and the phase-2 per-prompt subjective rankings. A full run drops
from ~15 min to a few minutes, and the more prompts you add, the better phase 2
parallelises (one in-flight ranking per prompt). Ranking stays deterministic —
each prompt is judged with its own prompt-seeded RNG.

**Caveat — each call is a separate ~180 MB `copilot` process.** Running many cold
starts at once saturates CPU, which can momentarily **freeze the desktop** *and*
**inflate the reported per-call latency** through contention (the latency comes
from the CLI's own `totalApiDurationMs`, which is clean of process-spawn lag but
not of CPU starvation). Guidance:

- For **trustworthy latency** numbers, run with `--concurrency 1`.
- To go **fast** without freezing, keep `--concurrency` modest (~4–6) and a small
  `--throttle` (e.g. `1`) so spawns are staggered instead of a thundering herd.
- `--concurrency 16 --throttle 0` is fastest but will spike CPU and skew latency.

### Cost and tokens

The Copilot CLI doesn't bill per token, so the report uses the CLI's own metrics:

- `premium_requests` — the multiplier-adjusted Copilot billing unit the CLI
  reports per call (the relative cost metric).
- `est_cost_usd` — a marginal USD estimate, `premium_requests × $0.04` (the
  premium-request overage rate; override via `$COPILOT_PREMIUM_REQUEST_USD`).
- `input_tokens` — **blank**: the CLI does not expose prompt tokens. `output_tokens`
  (including reasoning) *is* reported.

### Throwaway sessions (resume-list hygiene)

Every `copilot -p` call persists a session, so a full run would otherwise dump
~130 throwaway chats into your `copilot --resume` list. The Copilot CLI client
avoids this by running each call under an **isolated `COPILOT_HOME`**
(`<temp>/model-judging-copilot-home`, override with `$COPILOT_BENCH_HOME`) — auth
still works because credentials live in the OS keychain, not in `.copilot`. So
new runs never touch your real resume list.

To clear sessions that earlier runs left in your real `~/.copilot`, or to drop
the isolated home:

```bash
# Preview which sessions would be removed (matched by the cwd the harness uses):
python run_benchmark.py clean-sessions --dry-run

# Delete them (backs up session-store.db first):
python run_benchmark.py clean-sessions

# Or just wipe the isolated benchmark home wholesale:
python run_benchmark.py clean-sessions --purge-isolated
```

### Auth

- **Copilot CLI provider (default):** just run `copilot` once and sign in
  (`/login`). The harness shells out to the installed `copilot` executable; set
  `COPILOT_BIN` if it is not on `PATH`. Latency comes from the CLI's reported
  `totalApiDurationMs`; cost is reported as `premium_requests` plus an
  `est_cost_usd` estimate (see Cost and tokens above).
- **GitHub Models provider:** pass `--token ghp_...` or set `GITHUB_TOKEN`. The
  token needs the **`models: read`** scope (<https://github.com/settings/tokens>).

### Outputs (`results/`)

- `detailed.csv` — one row per (model, prompt): `correct`/`incorrect` for
  hard-truth prompts or the per-prompt `rank` for subjective prompts, plus
  `latency_ms`, `output_tokens`, `premium_requests`, and `est_cost_usd`
  (`input_tokens` is blank for the Copilot provider).
- `summary.csv` — one row per (model, category): `pass_rate` or `avg_rank` with
  `n_ok` (non-errored calls), latency avg/p50/p95, and avg premium-requests/USD.
  This is the table that feeds routing weights.

### Editing the model set and prices

`src/model_judging/registry.py` lists the eight benchmarked models with tier
labels (ids are Copilot CLI model ids). Prompts live in
`src/model_judging/data/prompts.json` — currently 25 subjective prompts across
15 everyday categories (email, advice, essay, message, complaint, apology,
explanation, creative, review, etc.), plus a few hard-truth (code/calculus) and
semantic-truth (proof) prompts kept as scaffolding. For hard objective coding/math
evaluation, defer to an established benchmark such as SWE-bench rather than
hand-rolling graded problems here.

## Judging library quick start

```python
from model_judging import Candidate, CodeExecutionJudge, JudgingPipeline, TaskKind
from model_judging import extract_features, classify_tier, score_complexity

# Complexity-based routing
text = "Write a function that implements a class hierarchy with async/await"
tier = classify_tier(score_complexity(extract_features(text)))
# tier -> ComplexityTier.MEDIUM

# Hard-truth judging
judge = CodeExecutionJudge()
pipeline = JudgingPipeline()

candidates = [
    Candidate(id="gpt4o", name="GPT-4o", quality_factor=0.95, latency_ms=500, cost_per_1k_tokens=0.03),
    Candidate(id="claude", name="Claude", quality_factor=0.92, latency_ms=1200, cost_per_1k_tokens=0.015),
]

decision = pipeline.judge(
    TaskKind.HARD_TRUTH,
    candidates,
    objective_judge=judge,
    artifact={"code": "print(2+2)", "language": "python", "expected_output": "4"}
)

# Subjective utility ranking
decision = pipeline.judge(TaskKind.SUBJECTIVE, candidates)
```

## Modules

- `types.py` — Core data types (Candidate, TaskKind, ObjectiveCheck, PairwiseOutcome)
- `pipeline.py` — JudgingPipeline with hard-truth and subjective paths
- `scoring.py` — Quality/latency/cost utility scoring
- `pairwise.py` — Vote aggregation for judge panels
- `judges.py` — GitHub Models LLM judge
- `executors.py` — Hard-truth evaluators (code, JSON schema)
- `complexity.py` — Feature extraction and complexity tier classification
- `registry.py` — Benchmarked models (Copilot CLI ids), tiers, and prices
- `client.py` — GitHub Models inference client (answers prompts; captures latency/tokens/cost)
- `copilot_client.py` — Copilot CLI client (default; shells out to `copilot -p`, captures latency/tokens/premium-requests)
- `sessions.py` — Cleanup helpers for the throwaway sessions `copilot -p` persists
- `dataset.py` — Prompt dataset loader and answer-format directives
- `assess.py` — Hard-truth grading and subjective matchup ranking
- `harness.py` — Benchmark orchestrator
- `report.py` — CSV reporting
- `mock.py` — Offline deterministic client for dry-runs/tests

## Testing

```bash
python -m unittest discover -s tests
# 26 tests, 100% passing
```

## Examples

```bash
python examples.py
```


