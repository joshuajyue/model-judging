"""Client for requesting completions from GitHub Models.

This is the piece that actually calls the underlying model to *answer* a prompt
(as opposed to ``judges.py`` which calls a model to *judge* two answers).

Every model is called through the same GitHub Models surface with identical
context, so the proxy latency/overhead is uniform and the resulting numbers are
meaningful as a *relative* scale.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .registry import ModelSpec

INFERENCE_ENDPOINT = "https://models.github.ai/inference/chat/completions"
CATALOG_ENDPOINT = "https://models.github.ai/catalog/models"


@dataclass(slots=True)
class CompletionResult:
    model_id: str
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelClient(Protocol):
    def complete(self, model: ModelSpec, prompt: str, system: str | None = None) -> CompletionResult:
        ...


class GitHubModelsClient:
    """Calls the GitHub Models inference endpoint over plain ``urllib`` (no deps)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        endpoint: str = INFERENCE_ENDPOINT,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        if not self.token:
            raise ValueError(
                "No GitHub token provided. Pass token=... or set GITHUB_TOKEN "
                "(a PAT with the 'models: read' scope)."
            )

    def complete(self, model: ModelSpec, prompt: str, system: str | None = None) -> CompletionResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": model.id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if model.reasoning_effort:
            payload["reasoning_effort"] = model.reasoning_effort

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            latency_ms = (time.perf_counter() - started) * 1000.0
        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return CompletionResult(
                model_id=model.id,
                text="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=f"HTTP {exc.code}: {detail}",
            )
        except Exception as exc:  # noqa: BLE001 - surface any transport error
            latency_ms = (time.perf_counter() - started) * 1000.0
            return CompletionResult(
                model_id=model.id,
                text="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        text = ""
        choices = raw.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""

        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        return CompletionResult(
            model_id=model.id,
            text=text.strip(),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=model.cost_usd(input_tokens, output_tokens),
            raw=raw,
        )


def fetch_catalog(token: str | None = None, *, timeout: float = 30.0) -> list[dict]:
    """Return the live GitHub Models catalog so registry IDs can be verified."""
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise ValueError("No GitHub token provided for catalog lookup.")
    request = urllib.request.Request(
        CATALOG_ENDPOINT,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if isinstance(data, dict):
        return data.get("models", []) or data.get("data", [])
    return data
