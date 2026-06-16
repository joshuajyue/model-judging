from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_judging.executors import CodeExecutionJudge, JSONSchemaValidationJudge
from model_judging.judges import GitHubModelsJudge
from model_judging.pipeline import JudgingPipeline
from model_judging.types import Candidate, ObjectiveCheck, PairwiseOutcome, TaskKind


class CodeExecutionJudgeTests(unittest.TestCase):
    def test_python_code_execution(self) -> None:
        judge = CodeExecutionJudge()
        artifact = {"code": "print('hello')", "language": "python", "expected_output": "hello"}

        checks = judge.evaluate(Candidate(id="a", name="A"), artifact)

        passed_checks = [c for c in checks if c.passed]
        self.assertGreaterEqual(len(passed_checks), 1)
        self.assertTrue(any(c.name == "code_provided" for c in passed_checks))

    def test_python_code_execution_failure(self) -> None:
        judge = CodeExecutionJudge()
        artifact = {
            "code": "print('goodbye')",
            "language": "python",
            "expected_output": "hello",
        }

        checks = judge.evaluate(Candidate(id="a", name="A"), artifact)

        execution_check = next((c for c in checks if c.name == "execution"), None)
        self.assertIsNotNone(execution_check)
        self.assertFalse(execution_check.passed)


class JSONSchemaValidationJudgeTests(unittest.TestCase):
    def test_valid_json(self) -> None:
        judge = JSONSchemaValidationJudge()
        artifact = {"json": '{"name": "John", "age": 30}'}

        checks = judge.evaluate(Candidate(id="a", name="A"), artifact)

        self.assertTrue(any(c.name == "json_valid" and c.passed for c in checks))

    def test_invalid_json(self) -> None:
        judge = JSONSchemaValidationJudge()
        artifact = {"json": '{"name": "John"'}

        checks = judge.evaluate(Candidate(id="a", name="A"), artifact)

        self.assertTrue(any(c.name == "json_valid" and not c.passed for c in checks))


if __name__ == "__main__":
    unittest.main()
