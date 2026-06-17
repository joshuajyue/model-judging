"""Registry of GitHub Models used in the benchmark.

The model IDs use the GitHub Models ``{publisher}/{model_name}`` format and are
passed verbatim to ``https://models.github.ai/inference/chat/completions``.

These IDs and prices are *editable bootstrap values*. Model IDs can drift as the
catalog changes, so run ``python run_benchmark.py verify-models`` (which queries
the live catalog) to confirm them before a paid run.

Prices are USD per 1,000,000 tokens and are only used to produce a *relative*
cost ranking, since GitHub Models itself is quota-based rather than per-token
billed. Adjust them to match published vendor pricing when you have it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    """GitHub Models catalog id, e.g. ``openai/gpt-5.5``."""

    name: str
    """Human-friendly display name."""

    tier: str
    """Coarse tier label used to keep results comparable as frontiers move."""

    input_price_per_1m: float
    """USD per 1,000,000 input (prompt) tokens. Editable bootstrap value."""

    output_price_per_1m: float
    """USD per 1,000,000 output (completion) tokens. Editable bootstrap value."""

    reasoning_effort: str | None = None
    """Optional reasoning effort. ``None`` means use the model default (medium)."""

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_price_per_1m
            + output_tokens * self.output_price_per_1m
        ) / 1_000_000


# Frontier of each current "tier". Reasoning level is left at the model default
# (medium) for every model to keep the prototype cheap, per design.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("anthropic/claude-opus-4.8", "Claude Opus 4.8", "claude-high", 5.0, 25.0),
    ModelSpec("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6", "claude-mid", 3.0, 15.0),
    ModelSpec("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "claude-low", 1.0, 5.0),
    ModelSpec("openai/gpt-5.5", "GPT-5.5", "openai-high", 1.25, 10.0),
    ModelSpec("openai/gpt-5.3-codex", "GPT-5.3 Codex", "openai-coding", 1.25, 10.0),
    ModelSpec("openai/gpt-5.4-mini", "GPT-5.4 mini", "openai-low", 0.25, 2.0),
    ModelSpec("google/gemini-3.1-pro", "Gemini 3.1 Pro", "google-high", 1.25, 10.0),
    ModelSpec("google/gemini-3.5-flash", "Gemini 3.5 Flash", "google-low", 0.30, 2.5),
)


def default_models() -> list[ModelSpec]:
    return list(DEFAULT_MODELS)


def models_by_id() -> dict[str, ModelSpec]:
    return {m.id: m for m in DEFAULT_MODELS}
