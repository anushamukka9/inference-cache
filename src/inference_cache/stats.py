"""Hit/miss statistics and cost-saved estimates.

:class:`CacheStats` is a small, serializable accumulator kept by
:class:`InferenceCache`. Cost estimates use a configurable price table
(USD per 1K tokens) — pass your provider's actual rates via ``set_price`` for
accurate numbers. Token counts are approximated with a fast heuristic; treat
the estimates as directional, not invoices.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token is common, we use words*1.3)."""
    words = len(_WORD_RE.findall(text))
    return max(1, int(words * 1.3))


class CacheStats:
    """Accumulates hits, misses, evictions, and estimated dollars saved."""

    DEFAULT_PRICE_PER_1K = 0.002  # USD per 1K tokens, conservative default

    def __init__(self, price_per_1k: float | dict[str, float] | None = None) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired = 0
        self.tokens_served_from_cache = 0
        self._prices = price_per_1k if price_per_1k is not None else self.DEFAULT_PRICE_PER_1K

    def price_for(self, model: str) -> float:
        if isinstance(self._prices, dict):
            return self._prices.get(model, self.DEFAULT_PRICE_PER_1K)
        return self._prices

    def set_price(self, price_per_1k: float | dict[str, float]) -> None:
        self._prices = price_per_1k

    def record_hit(self, response: str, model: str = "default") -> None:
        self.hits += 1
        self.tokens_served_from_cache += estimate_tokens(response)

    def record_miss(self) -> None:
        self.misses += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    def record_expired(self) -> None:
        self.expired += 1

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self.hits / total if total else 0.0

    @property
    def estimated_cost_saved_usd(self) -> float:
        # Price is per-model in the general case; use the default-rate
        # accumulator here. Per-model accounting is exposed via price_for().
        rate = self._prices if isinstance(self._prices, float) else self.DEFAULT_PRICE_PER_1K
        return self.tokens_served_from_cache / 1000 * rate

    def to_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "expired": self.expired,
            "tokens_served_from_cache": self.tokens_served_from_cache,
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 4),
        }

    def reset(self) -> None:
        self.hits = self.misses = self.evictions = self.expired = 0
        self.tokens_served_from_cache = 0
