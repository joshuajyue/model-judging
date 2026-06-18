"""A tiny thread-safe progress bar and a client wrapper that drives it.

``ProgressClient`` wraps any ``ModelClient`` and ticks the bar on every
``complete()`` call. Because *all* model traffic -- answers, validity judges and
matchup judges -- ultimately goes through ``client.complete()``, wrapping the
client captures every underlying call at the finest granularity with no changes
to the harness. The bar is rendered to ``stderr`` so it never pollutes the JSON
or CSV written to stdout/files.
"""

from __future__ import annotations

import sys
import threading
import time

from .client import CompletionResult, ModelClient
from .registry import ModelSpec


class ProgressBar:
    """Single-line, carriage-return progress bar (thread-safe)."""

    def __init__(self, total: int, *, width: int = 28, stream=sys.stderr, enabled: bool = True):
        self.total = max(1, total)
        self.width = width
        self.stream = stream
        self.enabled = enabled and stream.isatty()
        self._done = 0
        self._lock = threading.Lock()
        self._start = time.monotonic()

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self._done += n
            self._render()

    def _render(self) -> None:
        if not self.enabled:
            return
        done = min(self._done, self.total)
        frac = done / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self._start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        self.stream.write(
            f"\r  [{bar}] {done}/{self.total} ({frac * 100:4.1f}%)  "
            f"{rate:4.2f}/s  eta {eta / 60:4.1f}m"
        )
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


class ProgressClient:
    """Wraps a ``ModelClient`` and ticks a :class:`ProgressBar` per completion."""

    def __init__(self, inner: ModelClient, bar: ProgressBar):
        self._inner = inner
        self._bar = bar

    def complete(self, model: ModelSpec, prompt: str, system: str | None = None) -> CompletionResult:
        try:
            return self._inner.complete(model, prompt, system)
        finally:
            self._bar.tick()
