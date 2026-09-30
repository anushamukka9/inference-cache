"""Hit/miss statistics and cost-saved estimates.

:class:`CacheStats` is a small, serializable accumulator kept by
:class:`InferenceCache`. Cost estimates use a configurable price table
(USD per 1K tokens) - pass your provider's actual rates via ``set_price`` for
accurate numbers. Token counts are approximated with a fast heuristic; treat
the estimates as directional, not invoices.

All counters are guarded by a lock, so a stats object shared across threads
stays consistent.
"""

from __future__ import annotations

import re
import threading

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token is common, we use words*1.3)."""
    words = len(_WORD_RE.findall(text))
    return max(1, int(words * 1.3))


class CacheStats:
    """Accumulates hits, misses, evictions, and estimated dollars saved."""

    DEFAULT_PRICE_PER_1K = 0.002  # USD per 1K tokens, conservative default

    def __init__(self, price_per_1k: float | dict[str, float] | None = None) -> None:
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired = 0
        self.tokens_served_from_cache = 0
        self._prices = price_per_1k if price_per_1k is not None else self.DEFAULT_PRICE_PER_1K

    def price_for(self, model: str) -> float:
        with self._lock:
            if isinstance(self._prices, dict):
                return self._prices.get(model, self.DEFAULT_PRICE_PER_1K)
            return self._prices

    def set_price(self, price_per_1k: float | dict[str, float]) -> None:
        with self._lock:
            self._prices = price_per_1k

    def record_hit(self, response: str, model: str = "default") -> None:
        with self._lock:
            self.hits += 1
            self.tokens_served_from_cache += estimate_tokens(response)

    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def record_eviction(self) -> None:
        with self._lock:
            self.evictions += 1

    def record_expired(self) -> None:
        with self._lock:
            self.expired += 1

    @property
    def total_requests(self) -> int:
        with self._lock:
            return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total else 0.0

    @property
    def estimated_cost_saved_usd(self) -> float:
        # Price is per-model in the general case; use the default-rate
        # accumulator here. Per-model accounting is exposed via price_for().
        with self._lock:
            rate = self._prices if isinstance(self._prices, float) else self.DEFAULT_PRICE_PER_1K
            return self.tokens_served_from_cache / 1000 * rate

    def to_dict(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / (self.hits + self.misses), 4)
                if (self.hits + self.misses)
                else 0.0,
                "evictions": self.evictions,
                "expired": self.expired,
                "tokens_served_from_cache": self.tokens_served_from_cache,
                "estimated_cost_saved_usd": round(
                    self.estimated_cost_saved_usd, 4
                ),
            }

    def reset(self) -> None:
        with self._lock:
            self.hits = self.misses = self.evictions = self.expired = 0
            self.tokens_served_from_cache = 0
