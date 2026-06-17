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

# Live run against GitHub Models instead (needs a PAT with the 'models: read' scope):
python run_benchmark.py run --live --provider github --token ghp_xxxxxxxx

# Confirm every registry model id is accepted before a paid run:
python run_benchmark.py verify-models                       # pings each model via the CLI
python run_benchmark.py verify-models --provider github --token ghp_xxxxxxxx
```

Useful flags: `--limit N` (cap prompts), `--models claude,openai-low` (filter by
id/tier), `--judge <model-id>` (matchup judge), `--out DIR`, `--verbose`.

### Auth

- **Copilot CLI provider (default):** just run `copilot` once and sign in
  (`/login`). The harness shells out to the installed `copilot` executable; set
  `COPILOT_BIN` if it is not on `PATH`. Latency comes from the CLI's reported
  `totalApiDurationMs` and the cost column is the Copilot `premiumRequests`
  billed per call (the relative cost metric).
- **GitHub Models provider:** pass `--token ghp_...` or set `GITHUB_TOKEN`. The
  token needs the **`models: read`** scope (<https://github.com/settings/tokens>).

### Outputs (`results/`)

- `detailed.csv` — one row per (model, prompt): `correct`/`incorrect` for
  hard-truth prompts or the per-prompt `rank` for subjective prompts, plus
  latency, tokens, and cost.
- `summary.csv` — one row per (model, category): `pass_rate` or `avg_rank` with
  latency avg/p50/p95 and average cost. This is the table that feeds routing
  weights.

### Editing the model set and prices

`src/model_judging/registry.py` lists the eight benchmarked models with tier
labels (ids are Copilot CLI model ids). Prompts live in
`src/model_judging/data/prompts.json` — add more (10+ per category) to firm up
the rankings.

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


