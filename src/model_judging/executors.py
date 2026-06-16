from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Optional

from .types import Candidate, ObjectiveCheck


class CodeExecutionJudge:
    def __init__(self, sandbox_dir: Optional[str] = None):
        self.sandbox_dir = Path(sandbox_dir or "/tmp/model_judging_sandbox")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, candidate: Candidate, artifact: object | None) -> list[ObjectiveCheck]:
        checks: list[ObjectiveCheck] = []

        if not artifact or not isinstance(artifact, dict):
            return [ObjectiveCheck(name="code_provided", passed=False, details="No code provided")]

        code = artifact.get("code", "")
        language = artifact.get("language", "python")
        expected_output = artifact.get("expected_output", "")

        if not code:
            checks.append(ObjectiveCheck(name="code_provided", passed=False, details="Empty code"))
            return checks

        checks.append(ObjectiveCheck(name="code_provided", passed=True))

        if language == "python":
            passed, details = self._run_python(code, expected_output)
            checks.append(ObjectiveCheck(name="execution", passed=passed, details=details))
        elif language == "javascript":
            passed, details = self._run_javascript(code, expected_output)
            checks.append(ObjectiveCheck(name="execution", passed=passed, details=details))
        else:
            checks.append(
                ObjectiveCheck(name="execution", passed=False, details=f"Unsupported language: {language}")
            )

        return checks

    def _run_python(self, code: str, expected_output: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.sandbox_dir,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                return False, f"Exit code {result.returncode}: {result.stderr}"
            if expected_output and output != expected_output:
                return False, f"Expected '{expected_output}', got '{output}'"
            return True, "Execution successful"
        except subprocess.TimeoutExpired:
            return False, "Timeout (>5s)"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def _run_javascript(self, code: str, expected_output: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.sandbox_dir,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                return False, f"Exit code {result.returncode}: {result.stderr}"
            if expected_output and output != expected_output:
                return False, f"Expected '{expected_output}', got '{output}'"
            return True, "Execution successful"
        except subprocess.TimeoutExpired:
            return False, "Timeout (>5s)"
        except Exception as e:
            return False, f"Error: {str(e)}"


class JSONSchemaValidationJudge:
    def evaluate(self, candidate: Candidate, artifact: object | None) -> list[ObjectiveCheck]:
        checks: list[ObjectiveCheck] = []

        if not artifact or not isinstance(artifact, dict):
            return [ObjectiveCheck(name="json_valid", passed=False, details="No artifact provided")]

        json_str = artifact.get("json", "")
        schema = artifact.get("schema", {})

        try:
            parsed = json.loads(json_str)
            checks.append(ObjectiveCheck(name="json_valid", passed=True, details="Valid JSON"))
        except json.JSONDecodeError as e:
            checks.append(ObjectiveCheck(name="json_valid", passed=False, details=f"Invalid JSON: {str(e)}"))
            return checks

        if schema:
            try:
                import jsonschema
                jsonschema.validate(parsed, schema)
                checks.append(ObjectiveCheck(name="schema_valid", passed=True, details="Matches schema"))
            except ImportError:
                checks.append(
                    ObjectiveCheck(name="schema_valid", passed=False, details="jsonschema not installed")
                )
            except jsonschema.ValidationError as e:
                checks.append(
                    ObjectiveCheck(name="schema_valid", passed=False, details=f"Schema violation: {e.message}")
                )

        return checks
