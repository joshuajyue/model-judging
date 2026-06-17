"""Client for requesting completions from the **GitHub Copilot CLI**.

The public GitHub Models surface (``models.github.ai``) does not expose the same
model line-up that the Copilot CLI offers (e.g. ``gpt-5.5``, ``claude-opus-4.8``,
``gemini-3.1-pro-preview``). To benchmark *those* models we shell out to the
already-authenticated ``copilot`` CLI in its non-interactive mode::

    copilot -p "<prompt>" --model <id> --output-format json --available-tools

This reuses the user's existing Copilot login (no PAT/token handling, no internal
endpoints) and every model is driven through the exact same surface, so the
latency/cost numbers stay meaningful as a *relative* scale.

Tools are disabled (``--available-tools`` with no value) so each model answers the
prompt directly from its own knowledge instead of acting as an agent -- that keeps
the benchmark a measure of raw model quality rather than tool use.

What the CLI gives us per call (parsed from the JSONL event stream):

* ``assistant.message.data.content``      -> the answer text
* ``assistant.message.data.outputTokens`` -> output (incl. reasoning) tokens
* ``result.usage.totalApiDurationMs``     -> real model API latency (excludes the
  CLI/MCP session startup overhead, so it is cleaner than wall-clock time)
* ``result.usage.premiumRequests``        -> the Copilot billing unit, used as the
  relative cost metric (input/output token *prices* are not exposed)
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from .client import CompletionResult
from .registry import ModelSpec

COPILOT_BIN = os.environ.get("COPILOT_BIN", "copilot")


class CopilotCliError(RuntimeError):
    """Raised when the ``copilot`` executable cannot be launched at all."""


class CopilotCliClient:
    """Drives completions through ``copilot -p`` (implements the ``ModelClient`` Protocol).

    Parameters
    ----------
    copilot_bin:
        Path/name of the Copilot CLI executable (default ``copilot`` / ``$COPILOT_BIN``).
    timeout:
        Per-call wall-clock timeout in seconds. High-reasoning models on hard
        prompts can be slow, so this defaults generously.
    cwd:
        Working directory for the CLI. Defaults to the system temp dir so no
        repository files or instructions leak into the prompt context.
    disable_builtin_mcps:
        Pass ``--disable-builtin-mcps`` to cut session startup time. This only
        affects wall-clock time, never the reported ``totalApiDurationMs``.
    extra_args:
        Additional raw CLI arguments appended to every invocation.
    """

    def __init__(
        self,
        *,
        copilot_bin: str = COPILOT_BIN,
        timeout: float = 300.0,
        cwd: str | None = None,
        disable_builtin_mcps: bool = True,
        extra_args: list[str] | None = None,
    ) -> None:
        self.copilot_bin = copilot_bin
        self.timeout = timeout
        self.cwd = cwd if cwd is not None else os.environ.get("TEMP") or os.getcwd()
        self.disable_builtin_mcps = disable_builtin_mcps
        self.extra_args = list(extra_args or [])

    def _build_args(self, model: ModelSpec, prompt: str) -> list[str]:
        args = [
            self.copilot_bin,
            "-p",
            prompt,
            "--model",
            model.id,
            "--output-format",
            "json",
            "--no-custom-instructions",
            "--available-tools",  # no value -> disable every tool, answer directly
            "--no-color",
            "--no-auto-update",
        ]
        if self.disable_builtin_mcps:
            args.append("--disable-builtin-mcps")
        if model.reasoning_effort:
            args += ["--reasoning-effort", model.reasoning_effort]
        args += self.extra_args
        return args

    def complete(
        self, model: ModelSpec, prompt: str, system: str | None = None
    ) -> CompletionResult:
        # The CLI has no separate system-prompt flag, so fold any system text in.
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        args = self._build_args(model, full_prompt)

        env = dict(os.environ)
        env.setdefault("COPILOT_DISABLE_UPDATE", "1")

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CopilotCliError(
                f"Could not launch '{self.copilot_bin}'. Is the GitHub Copilot CLI "
                f"installed and on PATH? Original error: {exc}"
            ) from exc
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return CompletionResult(
                model_id=model.id,
                text="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=f"timeout after {self.timeout:.0f}s",
            )

        wall_ms = (time.perf_counter() - started) * 1000.0
        return self._parse(model, proc.stdout, proc.stderr, proc.returncode, wall_ms)

    @staticmethod
    def _parse(
        model: ModelSpec,
        stdout: str,
        stderr: str,
        returncode: int,
        wall_ms: float,
    ) -> CompletionResult:
        text = ""
        output_tokens = 0
        api_ms: float | None = None
        premium_requests = 0.0
        error_detail: str | None = None

        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            data = event.get("data") or {}
            if etype == "assistant.message":
                content = data.get("content")
                if content:
                    text = content  # keep the last non-empty assistant turn
                output_tokens += int(data.get("outputTokens", 0) or 0)
            elif etype == "error":
                error_detail = (
                    data.get("message")
                    or data.get("error")
                    or json.dumps(data)[:300]
                )
            elif etype == "result":
                usage = event.get("usage") or {}
                api_ms = usage.get("totalApiDurationMs")
                premium_requests = float(usage.get("premiumRequests", 0) or 0)
                if event.get("exitCode") not in (0, None):
                    returncode = event.get("exitCode")

        latency_ms = float(api_ms) if api_ms is not None else wall_ms

        error: str | None = None
        if not text:
            error = error_detail or (
                f"copilot exited {returncode} with no assistant message"
                + (f": {stderr.strip()[:300]}" if stderr.strip() else "")
            )
        elif returncode not in (0, None):
            error = error_detail or f"copilot exited {returncode}"

        return CompletionResult(
            model_id=model.id,
            text=text.strip(),
            latency_ms=latency_ms,
            input_tokens=0,  # not exposed by the CLI
            output_tokens=output_tokens,
            cost_usd=premium_requests,  # Copilot premium-request units (relative cost)
            error=error,
            raw={"premium_requests": premium_requests, "returncode": returncode},
        )


def verify_model(model: ModelSpec, *, client: CopilotCliClient | None = None) -> tuple[bool, str]:
    """Ping one model through the CLI; return ``(ok, detail)``.

    Used by ``run_benchmark.py verify-models`` to confirm that each registry id
    is actually accepted by the installed Copilot CLI before a paid run.
    """
    client = client or CopilotCliClient(timeout=120.0)
    result = client.complete(model, "Reply with exactly: ok")
    if result.ok and result.text:
        return True, f"api={result.latency_ms:.0f}ms premium={result.cost_usd:g}"
    return False, result.error or "no response"
