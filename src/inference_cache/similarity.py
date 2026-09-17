"""Pluggable semantic-similarity matching for cache lookups.

Exact key matching misses near-duplicates ("What is the capital of France?"
vs "what is the capital of France? "). Similarity modes reuse a cached
response when the incoming prompt is *close enough* to a stored one, trading
a little precision for a much higher hit rate.

A similarity function has the signature::

    similarity(prompt_a: str, prompt_b: str) -> float  # 0.0 .. 1.0

``lexical_similarity`` is the built-in default: dependency-free (stdlib only),
combining token-set Jaccard with a TF cosine over word unigrams and a
sequence-ratio over raw characters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable

SimilarityFn = Callable[[str, str], float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    return len(ta & tb) / len(ta | tb)


def _tf_cosine(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ca or not cb:
        return 1.0 if not ca and not cb else 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _char_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def lexical_similarity(a: str, b: str) -> float:
    """Lightweight lexical similarity in [0, 1]; no third-party dependencies.

    Weighted blend: 50% token-set Jaccard (robust to reordering/verbosity),
    30% TF cosine (robust to repeated words), 20% character ratio (catches
    near-identical strings with punctuation/whitespace differences).
    """
    return 0.5 * _jaccard(a, b) + 0.3 * _tf_cosine(a, b) + 0.2 * _char_ratio(a, b)


def exact_only(a: str, b: str) -> float:
    """Similarity that only ever matches identical strings (1.0 or 0.0)."""
    return 1.0 if a == b else 0.0


def find_best_match(
    prompt: str,
    candidates: list[tuple[str, str]],
    similarity: SimilarityFn,
    *,
    threshold: float,
) -> str | None:
    """Return the cache key of the best candidate above *threshold*, else None.

    *candidates* is a list of ``(cache_key, stored_prompt)`` pairs.
    """
    best_key: str | None = None
    best_score = threshold
    for key, stored_prompt in candidates:
        score = similarity(prompt, stored_prompt)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key
