"""Offline, deterministic mock client for dry-runs and tests.

Lets the full benchmark pipeline run end-to-end with no network or token spend.
Answers are canned per prompt; lower-tier models intentionally get some hard
prompts wrong so the correct/incorrect and ranking machinery is exercised. It
also plays the role of a matchup judge (detected by the ``[Answer A]`` marker).
"""

from __future__ import annotations

import hashlib

from .client import CompletionResult
from .registry import ModelSpec

# Correct, canned answers keyed by a distinctive substring of each prompt.
_CORRECT: list[tuple[str, str]] = [
    ("sum of all even", "```python\ndef solve(n):\n    return sum(i for i in range(1, n + 1) if i % 2 == 0)\n```"),
    ("revereses a string", "```python\ndef solve(s):\n    return s[::-1].upper()\n```"),
    ("longest strictly increasing subsequence",
     "```python\ndef solve(nums):\n    import bisect\n    tails = []\n    for x in nums:\n        i = bisect.bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```"),
    ("parantheses",
     "```python\ndef solve(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n```"),
    ("derivative of x^3", "The derivative is 3x^2, at x=2 that is 12.\nFINAL: 12"),
    ("integral of 2x from 0 to 3", "Integral of 2x is x^2, evaluated 0 to 3 gives 9.\nFINAL: 9"),
]

# Wrong answers used by low-tier models on the harder prompts.
_WRONG: dict[str, str] = {
    "longest strictly increasing subsequence":
        "```python\ndef solve(nums):\n    return len(set(nums))\n```",
    "parantheses":
        "```python\ndef solve(s):\n    return s.count('(') == s.count(')')\n```",
    "integral of 2x from 0 to 3": "FINAL: 6",
}

_SUBJECTIVE = (
    "Subject: {topic}\n\nHi,\n\nThanks for reading. Here is a clear, well-structured "
    "response that addresses your request directly and professionally. {filler}\n\nBest regards"
)


def _hash_jitter(seed: str, lo: int, hi: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return lo + (int(digest[:8], 16) % (hi - lo + 1))


class MockModelClient:
    """Implements the ``ModelClient`` protocol without any network access."""

    def complete(self, model: ModelSpec, prompt: str, system: str | None = None) -> CompletionResult:
        if "[Answer A]" in prompt and "[Answer B]" in prompt:
            return self._judge(model, prompt)
        return self._answer(model, prompt)

    def _answer(self, model: ModelSpec, prompt: str) -> CompletionResult:
        low_tier = model.tier.endswith("low")
        text = None
        for needle, answer in _CORRECT:
            if needle in prompt:
                if low_tier and needle in _WRONG:
                    text = _WRONG[needle]
                else:
                    text = answer
                break
        if text is None:
            # Subjective prompt: vary length by tier so matchups differentiate.
            filler = "Detailed, specific, and considerate. " * (
                4 if model.tier.endswith("high") else 2 if model.tier.endswith("mid") else 1
            )
            text = _SUBJECTIVE.format(topic=prompt[:24].strip(), filler=filler)

        base_latency = {"high": 1800, "mid": 1100, "coding": 1200, "low": 450}
        tier_key = model.tier.split("-")[-1]
        latency = base_latency.get(tier_key, 900) + _hash_jitter(model.id + prompt, 0, 400)
        input_tokens = 40 + _hash_jitter(prompt, 0, 60)
        output_tokens = max(8, len(text) // 4)
        return CompletionResult(
            model_id=model.id,
            text=text,
            latency_ms=float(latency),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=model.cost_usd(input_tokens, output_tokens),
        )

    def _judge(self, model: ModelSpec, prompt: str) -> CompletionResult:
        a = prompt.split("[Answer A]", 1)[1].split("[Answer B]", 1)[0]
        b = prompt.split("[Answer B]", 1)[1].split("Reply with", 1)[0]
        verdict = "A" if len(a) > len(b) else "B" if len(b) > len(a) else "TIE"
        return CompletionResult(
            model_id=model.id,
            text=verdict,
            latency_ms=float(_hash_jitter(prompt, 300, 700)),
            input_tokens=len(prompt) // 4,
            output_tokens=1,
            cost_usd=0.0,
        )
