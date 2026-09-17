"""inference-cache: an LLM inference caching layer.

Drop-in caching for LLM calls — exact-match and semantic-similarity lookup,
TTL + LRU eviction, hit/miss statistics with cost-saved estimates, and
pluggable backends (memory, SQLite, Redis).
"""

from .backends import CacheBackend, MemoryBackend, RedisBackend, SQLiteBackend
from .cache import InferenceCache
from .decorator import cached
from .normalization import make_cache_key, normalize_prompt
from .similarity import exact_only, find_best_match, lexical_similarity
from .stats import CacheStats, estimate_tokens

__all__ = [
    "InferenceCache",
    "cached",
    "CacheBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "RedisBackend",
    "CacheStats",
    "normalize_prompt",
    "make_cache_key",
    "lexical_similarity",
    "exact_only",
    "find_best_match",
    "estimate_tokens",
]

__version__ = "0.1.0"
