import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_judging.assess import (
    extract_numeric_answer,
    extract_python_code,
    grade_hard_truth,
    grade_semantic_truth,
    rank_answers,
)
from model_judging.client import CompletionResult
from model_judging.copilot_client import CopilotCliClient
from model_judging.dataset import Prompt, load_prompts
from model_judging.harness import run_benchmark, _percentile, LatencyStats
from model_judging.mock import MockModelClient
from model_judging.registry import DEFAULT_JUDGE_IDS, default_judge_models, default_models
from model_judging.report import write_detailed_csv, write_summary_csv
from run_benchmark import main


class _FakeJudge:
    """Deterministic judge: longer answer wins, equal length -> tie."""

    def judge(self, prompt, rubric, answer_a, answer_b):
        if len(answer_a) > len(answer_b):
            return "A"
        if len(answer_b) > len(answer_a):
            return "B"
        return "tie"


class _CountingJudge(_FakeJudge):
    """Same verdicts as _FakeJudge but counts how many matchups were judged."""

    def __init__(self):
        self.calls = 0

    def judge(self, prompt, rubric, answer_a, answer_b):
        self.calls += 1
        return super().judge(prompt, rubric, answer_a, answer_b)


class CopilotCliParseTests(unittest.TestCase):
    """The CLI parser is pure/offline -- no `copilot` process is spawned."""

    _spec = default_models()[0]

    def _parse(self, stdout, returncode=0, stderr=""):
        return CopilotCliClient._parse(self._spec, stdout, stderr, returncode, wall_ms=1234.0)

    def test_parses_answer_tokens_latency_cost(self):
        stdout = "\n".join([
            '{"type":"session.tools_updated","data":{}}',
            '{"type":"assistant.message","data":{"model":"x","content":"43","outputTokens":76}}',
            '{"type":"assistant.turn_end","data":{"turnId":"0"}}',
            '{"type":"result","timestamp":"t","exitCode":0,'
            '"usage":{"premiumRequests":2.5,"totalApiDurationMs":3826,"sessionDurationMs":8332}}',
        ])
        r = self._parse(stdout)
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "43")
        self.assertEqual(r.output_tokens, 76)
        self.assertEqual(r.latency_ms, 3826.0)   # uses totalApiDurationMs, not wall_ms
        self.assertEqual(r.premium_requests, 2.5)  # Copilot billing unit, as reported
        self.assertAlmostEqual(r.cost_usd, 2.5 * 0.04)  # marginal USD estimate
        self.assertIsNone(r.input_tokens)        # CLI does not expose prompt tokens

    def test_keeps_last_nonempty_message_and_sums_tokens(self):
        stdout = "\n".join([
            '{"type":"assistant.message","data":{"content":"thinking...","outputTokens":10}}',
            '{"type":"assistant.message","data":{"content":"final answer","outputTokens":20}}',
            '{"type":"result","exitCode":0,"usage":{"premiumRequests":0,"totalApiDurationMs":100}}',
        ])
        r = self._parse(stdout)
        self.assertEqual(r.text, "final answer")
        self.assertEqual(r.output_tokens, 30)

    def test_error_when_no_assistant_message(self):
        stdout = '{"type":"result","exitCode":1,"usage":{}}'
        r = self._parse(stdout, returncode=1, stderr="boom")
        self.assertFalse(r.ok)
        self.assertIn("boom", r.error)

    def test_falls_back_to_wall_clock_without_api_duration(self):
        stdout = '{"type":"assistant.message","data":{"content":"hi","outputTokens":1}}'
        r = self._parse(stdout)
        self.assertEqual(r.latency_ms, 1234.0)


class _StubValidityJudge:
    """Returns a fixed verdict regardless of input."""

    def __init__(self, verdict):
        self.verdict = verdict

    def assess(self, prompt, rubric, answer):
        return self.verdict


class SemanticTruthGradingTests(unittest.TestCase):
    _prompt = Prompt(id="pf", category="math_proof", kind="semantic_truth",
                     prompt="prove X", rubric="must show Y")

    def test_majority_valid_is_correct(self):
        judges = [_StubValidityJudge("valid"), _StubValidityJudge("valid"),
                  _StubValidityJudge("invalid")]
        res = grade_semantic_truth(self._prompt, "some proof", judges)
        self.assertTrue(res.correct)
        self.assertIn("valid 2/3", res.detail)

    def test_minority_valid_is_incorrect(self):
        judges = [_StubValidityJudge("valid"), _StubValidityJudge("invalid"),
                  _StubValidityJudge("invalid")]
        res = grade_semantic_truth(self._prompt, "some proof", judges)
        self.assertFalse(res.correct)

    def test_tie_counts_against(self):
        # Even panel, 1-1: strict majority not reached -> incorrect.
        judges = [_StubValidityJudge("valid"), _StubValidityJudge("invalid")]
        res = grade_semantic_truth(self._prompt, "p", judges)
        self.assertFalse(res.correct)

    def test_abstention_counts_against(self):
        judges = [_StubValidityJudge("valid"), _StubValidityJudge("valid"),
                  _StubValidityJudge("unsure")]
        # 2 of 3 valid -> still a strict majority -> correct.
        self.assertTrue(grade_semantic_truth(self._prompt, "p", judges).correct)
        judges = [_StubValidityJudge("valid"), _StubValidityJudge("unsure"),
                  _StubValidityJudge("unsure")]
        # 1 of 3 valid -> not a majority -> incorrect.
        self.assertFalse(grade_semantic_truth(self._prompt, "p", judges).correct)

    def test_semantic_prompts_graded_in_full_run(self):
        prompts = [p for p in load_prompts() if p.kind == "semantic_truth"]
        self.assertTrue(prompts)  # dataset ships proof prompts
        models = default_models()
        result = run_benchmark(prompts, models, MockModelClient(), rng=random.Random(0))
        sem_cells = [c for c in result.cells if c.kind == "semantic_truth"]
        self.assertEqual(len(sem_cells), len(prompts) * len(models))
        # All graded binary, none ranked.
        self.assertTrue(all(c.correct is not None for c in sem_cells))
        self.assertTrue(all(c.rank is None for c in sem_cells))
        # Mock: low-tier proofs are "incomplete" -> invalid; higher tiers valid.
        low = [c for c in sem_cells if c.tier.endswith("low")]
        high = [c for c in sem_cells if c.tier.endswith("high")]
        self.assertTrue(all(not c.correct for c in low))
        self.assertTrue(all(c.correct for c in high))


class CopilotCliRetryTests(unittest.TestCase):
    def test_rate_limit_marker_detection(self):
        from model_judging.copilot_client import _looks_rate_limited
        self.assertTrue(_looks_rate_limited("HTTP 429: Too many requests"))
        self.assertTrue(_looks_rate_limited(None, "secondary rate limit hit"))
        self.assertTrue(_looks_rate_limited("RESOURCE_EXHAUSTED"))
        self.assertFalse(_looks_rate_limited("HTTP 500: server error"))
        self.assertFalse(_looks_rate_limited(None, ""))

    def test_retries_on_rate_limit_then_succeeds(self):
        spec = default_models()[0]
        client = CopilotCliClient(min_interval=0.0, max_retries=3,
                                  backoff_base=0.0, backoff_cap=0.0)
        calls = {"n": 0}

        def fake_run_once(model, full_prompt):
            calls["n"] += 1
            if calls["n"] < 3:
                return CompletionResult(model_id=model.id, text="", latency_ms=1.0,
                                        input_tokens=0, output_tokens=0, cost_usd=0.0,
                                        error="HTTP 429: Too many requests")
            return CompletionResult(model_id=model.id, text="done", latency_ms=1.0,
                                    input_tokens=0, output_tokens=5, cost_usd=0.0)

        client._run_once = fake_run_once
        result = client.complete(spec, "hi")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "done")
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_max_retries(self):
        spec = default_models()[0]
        client = CopilotCliClient(min_interval=0.0, max_retries=2,
                                  backoff_base=0.0, backoff_cap=0.0)
        calls = {"n": 0}

        def always_limited(model, full_prompt):
            calls["n"] += 1
            return CompletionResult(model_id=model.id, text="", latency_ms=1.0,
                                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                                    error="429 too many requests")

        client._run_once = always_limited
        result = client.complete(spec, "hi")
        self.assertFalse(result.ok)
        self.assertEqual(calls["n"], 3)  # 1 initial + 2 retries

    def test_non_rate_limit_error_not_retried(self):
        spec = default_models()[0]
        client = CopilotCliClient(min_interval=0.0, max_retries=5,
                                  backoff_base=0.0, backoff_cap=0.0)
        calls = {"n": 0}

        def server_error(model, full_prompt):
            calls["n"] += 1
            return CompletionResult(model_id=model.id, text="", latency_ms=1.0,
                                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                                    error="HTTP 500: boom")

        client._run_once = server_error
        result = client.complete(spec, "hi")
        self.assertFalse(result.ok)
        self.assertEqual(calls["n"], 1)  # not retried


class DatasetTests(unittest.TestCase):
    def test_load_prompts_has_all_kinds(self):
        prompts = load_prompts()
        self.assertGreaterEqual(len(prompts), 8)
        kinds = {p.kind for p in prompts}
        self.assertEqual(kinds, {"hard_truth", "subjective", "semantic_truth"})

    def test_hard_truth_directive_appended(self):
        p = Prompt(id="x", category="c", kind="hard_truth", prompt="do it",
                   answer_format="code")
        self.assertIn("```python", p.rendered_prompt())
        self.assertTrue(p.rendered_prompt().startswith("do it"))


class ExtractionTests(unittest.TestCase):
    def test_extract_python_block(self):
        text = "sure\n```python\ndef solve():\n    return 1\n```\nthanks"
        self.assertIn("def solve()", extract_python_code(text))

    def test_extract_numeric_final_line(self):
        self.assertEqual(extract_numeric_answer("blah\nFINAL: 12"), 12.0)

    def test_extract_numeric_fallback_last_number(self):
        self.assertEqual(extract_numeric_answer("the answer is 9 i think"), 9.0)


class HardTruthGradingTests(unittest.TestCase):
    def test_code_correct(self):
        prompt = Prompt(
            id="c", category="easy_coding", kind="hard_truth", prompt="sum evens",
            evaluation={"type": "code_exec", "language": "python",
                        "harness": "print(solve(10))", "expected_output": "30"},
        )
        answer = "```python\ndef solve(n):\n    return sum(i for i in range(1, n+1) if i%2==0)\n```"
        self.assertTrue(grade_hard_truth(prompt, answer).correct)

    def test_code_incorrect(self):
        prompt = Prompt(
            id="c", category="easy_coding", kind="hard_truth", prompt="sum evens",
            evaluation={"type": "code_exec", "language": "python",
                        "harness": "print(solve(10))", "expected_output": "30"},
        )
        answer = "```python\ndef solve(n):\n    return 0\n```"
        self.assertFalse(grade_hard_truth(prompt, answer).correct)

    def test_numeric_correct_with_tolerance(self):
        prompt = Prompt(
            id="n", category="calculus", kind="hard_truth", prompt="deriv",
            evaluation={"type": "numeric", "expected": 12, "tolerance": 0.01},
        )
        self.assertTrue(grade_hard_truth(prompt, "FINAL: 12.0").correct)
        self.assertFalse(grade_hard_truth(prompt, "FINAL: 7").correct)


class MatchupRankingTests(unittest.TestCase):
    def test_longer_answer_ranks_first(self):
        prompt = Prompt(id="e", category="email", kind="subjective", prompt="hi")
        answers = {"a": "short", "b": "a much longer answer", "c": "medium one"}
        ranks = rank_answers(prompt, answers, [_FakeJudge()], rng=random.Random(1))
        self.assertEqual(ranks["b"], 1.0)
        self.assertGreater(ranks["a"], ranks["c"])

    def test_single_answer_gets_rank_one(self):
        prompt = Prompt(id="e", category="email", kind="subjective", prompt="hi")
        ranks = rank_answers(prompt, {"only": "x"}, [_FakeJudge()])
        self.assertEqual(ranks, {"only": 1.0})

    def test_swiss_used_for_large_panel_reduces_calls(self):
        # 8 answers of strictly increasing length; "longer wins" is transitive.
        prompt = Prompt(id="e", category="email", kind="subjective", prompt="hi")
        answers = {f"m{i}": "x" * (i + 1) for i in range(8)}
        judge = _CountingJudge()
        ranks = rank_answers(prompt, answers, [judge], rng=random.Random(0))
        # Swiss: ceil(log2 8)=3 rounds * 4 matches = 12 calls (vs 28 round-robin).
        self.assertEqual(judge.calls, 12)
        self.assertEqual(ranks["m7"], 1.0)        # longest is the unique winner
        self.assertEqual(ranks["m0"], 8.0)        # shortest is the unique loser
        self.assertLess(ranks["m7"], ranks["m0"])

    def test_rounds_zero_forces_round_robin(self):
        prompt = Prompt(id="e", category="email", kind="subjective", prompt="hi")
        answers = {f"m{i}": "x" * (i + 1) for i in range(8)}
        judge = _CountingJudge()
        rank_answers(prompt, answers, [judge], rng=random.Random(0), rounds=0)
        self.assertEqual(judge.calls, 28)         # C(8,2)


class PercentileTests(unittest.TestCase):
    def test_percentiles(self):
        values = [100.0, 200.0, 300.0, 400.0]
        stats = LatencyStats.of(values)
        self.assertEqual(stats.avg, 250.0)
        self.assertAlmostEqual(stats.p50, 250.0)
        self.assertGreaterEqual(stats.p95, 300.0)

    def test_empty(self):
        self.assertEqual(_percentile([], 95), 0.0)


class ReportColumnTests(unittest.TestCase):
    def _cell(self, **kw):
        from model_judging.harness import CellResult
        base = dict(prompt_id="p", category="email", kind="subjective",
                    model_id="m", model_name="M", tier="t", latency_ms=100.0,
                    input_tokens=None, output_tokens=5, cost_usd=0.1,
                    premium_requests=2.5, rank=1.0)
        base.update(kw)
        return CellResult(**base)

    def test_detailed_blank_input_tokens_and_premium(self):
        from model_judging.harness import BenchmarkResult
        result = BenchmarkResult(cells=[self._cell()])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_detailed_csv(result, Path(tmp) / "d.csv")
            with path.open(encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
        self.assertIn("premium_requests", row)
        self.assertIn("est_cost_usd", row)
        self.assertEqual(row["input_tokens"], "")        # N/A, not 0
        self.assertEqual(row["premium_requests"], "2.5")
        self.assertEqual(row["est_cost_usd"], "0.1")

    def test_summary_has_premium_and_cost_columns(self):
        from model_judging.harness import BenchmarkResult
        result = BenchmarkResult(cells=[self._cell()])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary_csv(result, Path(tmp) / "s.csv")
            with path.open(encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
        self.assertIn("avg_premium_requests", row)
        self.assertIn("avg_est_cost_usd", row)
        self.assertIn("n_ok", row)


class DefaultJudgePanelTests(unittest.TestCase):
    def test_panel_is_cheap_and_vendor_balanced(self):
        panel = default_judge_models()
        ids = [m.id for m in panel]
        self.assertEqual(ids, list(DEFAULT_JUDGE_IDS))
        # One cheap-to-run model per vendor (Anthropic / OpenAI / Google). Gemini
        # 3.1 Pro is chosen over Flash because Flash bills ~14x premium on Copilot.
        self.assertEqual(
            ids, ["claude-haiku-4.5", "gpt-5.4-mini", "gemini-3.1-pro-preview"]
        )
        tiers = {m.tier.split("-")[0] for m in panel}
        self.assertEqual(tiers, {"claude", "openai", "google"})

    def test_harness_default_uses_full_panel(self):
        # A counting client records how many matchup-judge calls happen, which
        # equals (matchups * panel size). Compare default panel vs a single judge.
        class CountingClient(MockModelClient):
            def __init__(self):
                self.judge_calls = 0

            def complete(self, model, prompt, system=None):
                if "[Answer A]" in prompt and "[Answer B]" in prompt:
                    self.judge_calls += 1
                return super().complete(model, prompt, system)

        prompts = [p for p in load_prompts() if p.category == "email"][:1]
        models = default_models()

        default_client = CountingClient()
        run_benchmark(prompts, models, default_client, rng=random.Random(0))

        single_client = CountingClient()
        run_benchmark(prompts, models, single_client,
                      judge_models=[models[0]], rng=random.Random(0))

        # Same matchups, but the default panel has 3 judges -> 3x the judge calls.
        self.assertEqual(default_client.judge_calls, single_client.judge_calls * 3)


class EstimatorTests(unittest.TestCase):
    def test_estimate_counts_match_actual_run(self):
        # Count actual model calls in a mock run and compare to the estimate.
        from model_judging.estimate import estimate_run

        class CountingClient(MockModelClient):
            def __init__(self):
                self.calls = 0

            def complete(self, model, prompt, system=None):
                self.calls += 1
                return super().complete(model, prompt, system)

        prompts = load_prompts()
        models = default_models()
        panel = default_judge_models()
        est = estimate_run(prompts, models, panel)

        client = CountingClient()
        run_benchmark(prompts, models, client, rng=random.Random(0))
        # All models answer (no errors in the mock), so the estimate is exact.
        self.assertEqual(client.calls, est.total_calls)

    def test_estimate_breakdown(self):
        from model_judging.estimate import estimate_run
        prompts = load_prompts()
        models = default_models()
        panel = default_judge_models()
        est = estimate_run(prompts, models, panel)
        self.assertEqual(est.answer_calls, len(prompts) * len(models))
        self.assertEqual(est.total_calls,
                         est.answer_calls + est.semantic_judge_calls + est.ranking_judge_calls)
        # Premium for an answer phase = n_prompts * sum(model multipliers).
        self.assertAlmostEqual(
            est.answer_premium, len(prompts) * sum(m.premium_per_call for m in models)
        )
        self.assertGreater(est.est_usd, 0)

    def test_estimate_only_does_not_run(self):
        rc = main(["run", "--estimate-only", "--limit", "2"])
        self.assertEqual(rc, 0)


class ProgressClientTests(unittest.TestCase):
    def test_ticks_once_per_call(self):
        from model_judging.estimate import estimate_run
        from model_judging.progress import ProgressBar, ProgressClient
        prompts = [p for p in load_prompts() if p.category == "email"][:1]
        models = default_models()
        panel = default_judge_models()
        est = estimate_run(prompts, models, panel)

        bar = ProgressBar(est.total_calls, enabled=False)
        client = ProgressClient(MockModelClient(), bar)
        run_benchmark(prompts, models, client, rng=random.Random(0))
        # Every underlying call (answers + matchup judges) ticked exactly once.
        self.assertEqual(bar._done, est.total_calls)


class EndToEndTests(unittest.TestCase):
    def test_full_run_with_mock_writes_csv(self):
        prompts = load_prompts()
        models = default_models()
        result = run_benchmark(prompts, models, MockModelClient(), rng=random.Random(0))
        self.assertEqual(len(result.cells), len(prompts) * len(models))
        self.assertTrue(all(c.error is None for c in result.cells))

        # Hard-truth cells are graded; subjective cells are ranked.
        ht = [c for c in result.cells if c.kind == "hard_truth"]
        subj = [c for c in result.cells if c.kind == "subjective"]
        self.assertTrue(all(c.correct is not None for c in ht))
        self.assertTrue(all(c.rank is not None for c in subj))

        with tempfile.TemporaryDirectory() as tmp:
            detailed = write_detailed_csv(result, Path(tmp) / "detailed.csv")
            summary = write_summary_csv(result, Path(tmp) / "summary.csv")
            with detailed.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(result.cells))
            with summary.open(encoding="utf-8") as fh:
                summ = list(csv.DictReader(fh))
            self.assertTrue(any(r["metric"] == "pass_rate" for r in summ))
            self.assertTrue(any(r["metric"] == "avg_rank" for r in summ))

    def test_high_tier_outranks_low_tier_subjective(self):
        prompts = [p for p in load_prompts() if p.category == "email"]
        models = default_models()
        result = run_benchmark(prompts, models, MockModelClient(), rng=random.Random(0))
        ranks = {c.model_id: c.rank for c in result.cells if c.category == "email"
                 and c.prompt_id == prompts[0].id}
        self.assertLess(ranks["claude-opus-4.8"], ranks["claude-haiku-4.5"])

    def test_concurrent_run_matches_sequential_structure(self):
        prompts = load_prompts()
        models = default_models()
        seq = run_benchmark(prompts, models, MockModelClient(), rng=random.Random(0))
        par = run_benchmark(prompts, models, MockModelClient(), concurrency=8)

        # Same cells, same order, no errors.
        self.assertEqual(len(par.cells), len(seq.cells))
        self.assertTrue(all(c.error is None for c in par.cells))
        self.assertEqual(
            [(c.prompt_id, c.model_id) for c in par.cells],
            [(c.prompt_id, c.model_id) for c in seq.cells],
        )
        # Hard-truth grading is deterministic, so it must match exactly.
        seq_ht = {(c.prompt_id, c.model_id): c.correct
                  for c in seq.cells if c.kind == "hard_truth"}
        par_ht = {(c.prompt_id, c.model_id): c.correct
                  for c in par.cells if c.kind == "hard_truth"}
        self.assertEqual(seq_ht, par_ht)
        # Every subjective cell is still ranked.
        self.assertTrue(all(c.rank is not None
                            for c in par.cells if c.kind == "subjective"))

    def test_concurrent_ranking_is_deterministic(self):
        prompts = load_prompts()
        models = default_models()
        a = run_benchmark(prompts, models, MockModelClient(), concurrency=8)
        b = run_benchmark(prompts, models, MockModelClient(), concurrency=8)
        ranks_a = {(c.prompt_id, c.model_id): c.rank for c in a.cells}
        ranks_b = {(c.prompt_id, c.model_id): c.rank for c in b.cells}
        self.assertEqual(ranks_a, ranks_b)


if __name__ == "__main__":
    unittest.main()
