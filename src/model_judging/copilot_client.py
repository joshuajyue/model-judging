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
import random
import subprocess
import sys
import time

from .client import CompletionResult
from .registry import ModelSpec

COPILOT_BIN = os.environ.get("COPILOT_BIN", "copilot")

# Substrings that mark a GitHub/Copilot rate-limit response. The CLI surfaces the
# raw 429 body, which includes GitHub's "Too many requests"/ToS scraping notice;
# Gemini-style backends report "RESOURCE_EXHAUSTED".
_RATE_LIMIT_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "rate-limit",
    "secondary rate",
    "resource_exhausted",
    "quota exceeded",
)


def _looks_rate_limited(*texts: str | None) -> bool:
    for text in texts:
        if not text:
            continue
        low = text.lower()
        if any(marker in low for marker in _RATE_LIMIT_MARKERS):
            return True
    return False


class CopilotCliError(RuntimeError):
    """Raised when the ``copilot`` executable cannot be launched at all."""


def _resolve_copilot_home(copilot_home: str | None) -> str | None:
    """Pick the isolated ``COPILOT_HOME`` for benchmark spawns.

    ``None`` -> ``$COPILOT_BENCH_HOME`` or ``<temp>/model-judging-copilot-home``.
    ``""``   -> opt out (return ``None`` so the real ``~/.copilot`` is used).
    """
    if copilot_home == "":
        return None
    if copilot_home is None:
        copilot_home = os.environ.get("COPILOT_BENCH_HOME") or os.path.join(
            os.environ.get("TEMP") or os.getcwd(), "model-judging-copilot-home"
        )
    os.makedirs(copilot_home, exist_ok=True)
    return copilot_home


class CopilotCliClient:
    """Drives completions through ``copilot -p`` (implements the ``ModelClient`` Protocol).

    A full benchmark spawns a fresh ``copilot`` process per (model, prompt) *and*
    per matchup judge call -- easily ~200 invocations -- which trips GitHub's
    secondary rate limit (HTTP 429). To stay under it this client:

    * **throttles** to at most one spawn every ``min_interval`` seconds (the same
      instance is shared by answer and judge calls, so the cap covers the whole
      run), and
    * **retries with exponential backoff** when a 429 / rate-limit response is
      detected, since those limits are time-windowed and clear on their own.

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
    min_interval:
        Minimum seconds between process spawns (gentle proactive throttle).
    max_retries:
        How many times to retry a single call after a rate-limit response.
    backoff_base / backoff_cap:
        Exponential backoff is ``min(backoff_cap, backoff_base * 2**attempt)``
        seconds plus jitter, per retry.
    copilot_home:
        Value for the ``COPILOT_HOME`` env var of every spawned process. The CLI
        persists one session per ``copilot -p`` call under this directory, so a
        full run would otherwise dump ~130 throwaway sessions into the user's
        real ``~/.copilot`` resume list. Pointing the benchmark at an isolated
        home keeps those sessions out of the way (auth still works -- credentials
        live in the OS keychain/credential manager, not in ``.copilot``). Pass
        ``""`` to opt out and use the real home. Defaults to ``$COPILOT_BENCH_HOME``
        or ``<temp>/model-judging-copilot-home``.
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
        min_interval: float = 1.0,
        max_retries: int = 6,
        backoff_base: float = 10.0,
        backoff_cap: float = 120.0,
        copilot_home: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.copilot_bin = copilot_bin
        self.timeout = timeout
        self.cwd = cwd if cwd is not None else os.environ.get("TEMP") or os.getcwd()
        self.disable_builtin_mcps = disable_builtin_mcps
        self.min_interval = max(0.0, min_interval)
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.copilot_home = _resolve_copilot_home(copilot_home)
        self.extra_args = list(extra_args or [])
        self._last_spawn_at = 0.0
        self._rng = random.Random()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_spawn_at)
        if wait > 0:
            time.sleep(wait)

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
        return base + self._rng.uniform(0, base * 0.25)

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

        result = self._run_once(model, full_prompt)
        attempt = 0
        while (
            attempt < self.max_retries
            and not result.ok
            and _looks_rate_limited(result.error)
        ):
            delay = self._backoff_seconds(attempt)
            print(
                f"[copilot] rate limited on {model.id}; backing off "
                f"{delay:.0f}s (retry {attempt + 1}/{self.max_retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
            result = self._run_once(model, full_prompt)
        return result

    def _run_once(self, model: ModelSpec, full_prompt: str) -> CompletionResult:
        args = self._build_args(model, full_prompt)

        env = dict(os.environ)
        env.setdefault("COPILOT_DISABLE_UPDATE", "1")
        if self.copilot_home:
            # Isolate benchmark sessions so they never pollute the user's real
            # ~/.copilot resume list. Auth still works via the OS credential store.
            env["COPILOT_HOME"] = self.copilot_home

        self._throttle()
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
            self._last_spawn_at = time.monotonic()
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

        self._last_spawn_at = time.monotonic()
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
