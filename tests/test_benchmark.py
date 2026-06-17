import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_judging.assess import (
    extract_numeric_answer,
    extract_python_code,
    grade_hard_truth,
    rank_answers,
)
from model_judging.dataset import Prompt, load_prompts
from model_judging.harness import run_benchmark, _percentile, LatencyStats
from model_judging.mock import MockModelClient
from model_judging.registry import default_models
from model_judging.report import write_detailed_csv, write_summary_csv


class _FakeJudge:
    """Deterministic judge: longer answer wins, equal length -> tie."""

    def judge(self, prompt, rubric, answer_a, answer_b):
        if len(answer_a) > len(answer_b):
            return "A"
        if len(answer_b) > len(answer_a):
            return "B"
        return "tie"


class DatasetTests(unittest.TestCase):
    def test_load_prompts_has_both_kinds(self):
        prompts = load_prompts()
        self.assertGreaterEqual(len(prompts), 8)
        kinds = {p.kind for p in prompts}
        self.assertEqual(kinds, {"hard_truth", "subjective"})

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


class PercentileTests(unittest.TestCase):
    def test_percentiles(self):
        values = [100.0, 200.0, 300.0, 400.0]
        stats = LatencyStats.of(values)
        self.assertEqual(stats.avg, 250.0)
        self.assertAlmostEqual(stats.p50, 250.0)
        self.assertGreaterEqual(stats.p95, 300.0)

    def test_empty(self):
        self.assertEqual(_percentile([], 95), 0.0)


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
        self.assertLess(ranks["anthropic/claude-opus-4.8"], ranks["anthropic/claude-haiku-4.5"])


if __name__ == "__main__":
    unittest.main()
