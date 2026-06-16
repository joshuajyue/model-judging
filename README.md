# model-judging

Prototype for a model judging and routing pipeline with:

- **Hard-truth checks** for code/math/format validation
- **Subjective ranking** for essays, emails, messages, and advice  
- **Lightweight scoring** for latency, cost, and quality
- **Pairwise aggregation** for multi-judge panels
- **GitHub Models integration** for real LLM judge calls
- **Complexity detection** for request routing (SIMPLE/MEDIUM/COMPLEX/REASONING)

## Quick Start

```python
from model_judging import Candidate, CodeExecutionJudge, JudgingPipeline, TaskKind
from model_judging import extract_features, classify_tier

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

## Testing

```bash
python -m unittest discover -s tests
# 12 tests, 100% passing
```

## Examples

```bash
python examples.py
```


