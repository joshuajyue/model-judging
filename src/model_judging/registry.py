"""Registry of models used in the benchmark.

The default provider is the **GitHub Copilot CLI** (``copilot -p --model <id>``),
so the IDs below are the bare Copilot CLI model ids (e.g. ``gpt-5.5``,
``claude-opus-4.8``, ``gemini-3.1-pro-preview``) -- *not* the GitHub Models
``{publisher}/{model_name}`` ids. Run ``python run_benchmark.py verify-models``
to confirm every id is accepted by the installed CLI before a paid run.

The ``*_price_per_1m`` fields are *editable bootstrap values* only used by the
optional GitHub Models provider for a relative USD cost ranking. The Copilot CLI
provider ignores them and instead reports the actual ``premiumRequests`` billed
per call, which is the relative cost metric for Copilot runs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    """Copilot CLI model id, e.g. ``gpt-5.5`` (or a GitHub Models id like
    ``openai/gpt-5.5`` when using the GitHub Models provider)."""

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


# Frontier of each current "tier". IDs are Copilot CLI model ids. Reasoning level
# is left at the model default (medium) for every model to keep the prototype
# cheap, per design.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("claude-opus-4.8", "Claude Opus 4.8", "claude-high", 5.0, 25.0),
    ModelSpec("claude-sonnet-4.6", "Claude Sonnet 4.6", "claude-mid", 3.0, 15.0),
    ModelSpec("claude-haiku-4.5", "Claude Haiku 4.5", "claude-low", 1.0, 5.0),
    ModelSpec("gpt-5.5", "GPT-5.5", "openai-high", 1.25, 10.0),
    ModelSpec("gpt-5.3-codex", "GPT-5.3 Codex", "openai-coding", 1.25, 10.0),
    ModelSpec("gpt-5.4-mini", "GPT-5.4 mini", "openai-low", 0.25, 2.0),
    ModelSpec("gemini-3.1-pro-preview", "Gemini 3.1 Pro", "google-high", 1.25, 10.0),
    ModelSpec("gemini-3.5-flash", "Gemini 3.5 Flash", "google-low", 0.30, 2.5),
)


def default_models() -> list[ModelSpec]:
    return list(DEFAULT_MODELS)


# Default matchup-judge panel: the cheapest-to-run model from each vendor *by
# Copilot premium-request cost*. Note this is NOT always the lowest tier -- on the
# Copilot CLI, Gemini 3.5 Flash bills at ~14x premium while Gemini 3.1 Pro bills at
# ~1x, so Pro is the cheaper Google judge. One judge per vendor keeps cost low AND
# balances per-vendor self-preference bias, so the aggregated verdict is more
# neutral than any single judge -- including an expensive one. Override with
# run_benchmark's --judge.
DEFAULT_JUDGE_IDS: tuple[str, ...] = (
    "claude-haiku-4.5",
    "gpt-5.4-mini",
    "gemini-3.1-pro-preview",
)


def default_judge_models() -> list[ModelSpec]:
    by_id = models_by_id()
    return [by_id[mid] for mid in DEFAULT_JUDGE_IDS if mid in by_id]


def models_by_id() -> dict[str, ModelSpec]:
    return {m.id: m for m in DEFAULT_MODELS}
