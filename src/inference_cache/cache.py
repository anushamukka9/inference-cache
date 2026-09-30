"""The main caching layer: exact + semantic lookup, TTL, and LRU eviction.

:class:`InferenceCache` wraps any callable ``fn(prompt, ...) -> str`` and
serves repeated (or near-duplicate) prompts from storage instead of calling
the model again.

Lookup order for a request:
  1. Exact match - SHA-256 key over normalized prompt + model + params.
  2. Semantic match - optional; scans stored prompts with a pluggable
     similarity function and reuses the best entry above ``threshold``.
  3. Miss - calls the wrapped function, stores the result, and enforces
     ``max_size`` (LRU) and ``ttl_seconds`` expiry.

Thread safety: the cache is safe to share across threads. Internal state
(recency order, statistics) is guarded by a re-entrant lock, and concurrent
``call()`` invocations for the same prompt are serialized on a per-key lock
so the wrapped function runs exactly once per key - every waiter gets the
stored result.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from .backends import CacheBackend, MemoryBackend
from .normalization import make_cache_key, normalize_prompt
from .similarity import SimilarityFn, exact_only, lexical_similarity, find_best_match
from .stats import CacheStats

DEFAULT_SIMILARITY_THRESHOLD = 0.92


class InferenceCache:
    """LLM inference caching layer with exact + semantic matching."""

    def __init__(
        self,
        *,
        backend: Optional[CacheBackend] = None,
        ttl_seconds: Optional[float] = None,
        max_size: Optional[int] = None,
        model: str = "default",
        default_params: Optional[dict] = None,
        similarity: Optional[SimilarityFn] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        case_insensitive: bool = False,
        price_per_1k: float | dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            backend: storage backend; defaults to :class:`MemoryBackend`.
            ttl_seconds: entries older than this are treated as expired.
            max_size: maximum entries; oldest-least-recently-used are evicted.
            model: model identifier baked into cache keys.
            default_params: sampling params baked into cache keys.
            similarity: similarity function for near-duplicate matching.
                ``None`` (default) uses :func:`lexical_similarity`;
                pass :func:`exact_only` to disable semantic matching.
            similarity_threshold: minimum similarity score for a semantic hit.
            case_insensitive: fold prompts to lowercase before keying.
            price_per_1k: USD per 1K tokens (float) or per-model dict, used
                for cost-saved estimates.
        """
        self.backend: CacheBackend = backend or MemoryBackend()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.model = model
        self.default_params = default_params or {}
        self.similarity: SimilarityFn = similarity or lexical_similarity
        self.similarity_threshold = similarity_threshold
        self.case_insensitive = case_insensitive
        self.stats = CacheStats(price_per_1k)
        # Guards _recency and _key_locks; re-entrant so get()/call()/store()
        # can nest without deadlock.
        self._lock = threading.RLock()
        # Per-key locks for exactly-once `call` semantics under concurrency.
        self._key_locks: dict[str, threading.Lock] = {}
        # LRU recency tracker: key insertion order = recency for eviction.
        # Kept alongside the backend so eviction works uniformly across backends.
        self._recency: OrderedDict[str, None] = OrderedDict()
        for key in self.backend.keys():
            self._recency[key] = None

    # ------------------------------------------------------------------ keys
    def _key(self, prompt: str, params: Optional[dict] = None) -> str:
        merged = {**self.default_params, **(params or {})}
        return make_cache_key(
            prompt,
            model=self.model,
            params=merged,
            case_insensitive=self.case_insensitive,
        )

    # -------------------------------------------------------------- lifecycle
    def _is_expired(self, record: dict[str, Any]) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - record.get("created_at", 0.0)) > self.ttl_seconds

    def _touch(self, key: str) -> None:
        with self._lock:
            self._recency[key] = None
            self._recency.move_to_end(key)

    def _evict_if_needed(self) -> None:
        with self._lock:
            while self.max_size is not None and len(self._recency) >= self.max_size:
                oldest, _ = self._recency.popitem(last=False)
                self.backend.delete(oldest)
                self.stats.record_eviction()

    def _key_lock(self, key: str) -> threading.Lock:
        """Return (creating if needed) the lock serializing ``call()`` for *key*."""
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _release_key_lock(self, key: str, lock: threading.Lock) -> None:
        # Drop the dict entry so it cannot grow without bound; threads that
        # already hold this lock object still exclude each other on it, and
        # any latecomer re-checks the cache inside its own critical section.
        with self._lock:
            if self._key_locks.get(key) is lock:
                del self._key_locks[key]
        lock.release()

    def _store(self, key: str, prompt: str, value: str, params: Optional[dict] = None) -> None:
        # One critical section: evict-then-insert stays atomic, so max_size
        # is a hard bound even when threads store concurrently.
        with self._lock:
            self._evict_if_needed()
            merged = {**self.default_params, **(params or {})}
            self.backend.put(
                key,
                {
                    "value": value,
                    "prompt": normalize_prompt(
                        prompt, case_insensitive=self.case_insensitive
                    ),
                    "model": self.model,
                    "params": json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str),
                    "created_at": time.time(),
                },
            )
            self._touch(key)

    # ---------------------------------------------------------------- lookup
    def get(self, prompt: str, params: Optional[dict] = None) -> Optional[str]:
        """Return a cached response for *prompt*, or ``None`` on miss."""
        with self._lock:
            return self._get_locked(prompt, params)

    def _get_locked(self, prompt: str, params: Optional[dict] = None) -> Optional[str]:
        key = self._key(prompt, params)

        record = self.backend.get(key)
        if record is not None:
            if self._is_expired(record):
                self.backend.delete(key)
                self._recency.pop(key, None)
                self.stats.record_expired()
            else:
                self._touch(key)
                self.stats.record_hit(record["value"], self.model)
                return record["value"]

        # Semantic fallback: reuse a near-duplicate's response, but only from
        # entries produced under the same model + params.
        merged = {**self.default_params, **(params or {})}
        canonical_params = json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str)
        candidates = [
            (key, record["prompt"])
            for key, record in ((k, self.backend.get(k)) for k in self._recency)
            if record is not None
            and record.get("model", "default") == self.model
            and record.get("params", "{}") == canonical_params
            and not self._is_expired(record)
        ]
        match_key = find_best_match(
            normalize_prompt(prompt, case_insensitive=self.case_insensitive),
            candidates,
            self.similarity,
            threshold=self.similarity_threshold,
        )
        if match_key is not None:
            record = self.backend.get(match_key)
            if record is not None and not self._is_expired(record):
                self._touch(match_key)
                self.stats.record_hit(record["value"], self.model)
                return record["value"]

        self.stats.record_miss()
        return None

    def call(
        self,
        fn: Callable[..., str],
        prompt: str,
        *args: Any,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Call ``fn(prompt, *args, **kwargs)``, serving from cache when possible.

        Thread-safe: concurrent calls for the same prompt serialize on a
        per-key lock, so ``fn`` runs at most once per key and every thread
        receives the stored result. The function itself runs outside the
        cache-wide lock, so concurrent calls for *different* prompts proceed
        in parallel.
        """
        key = self._key(prompt, params)
        key_lock = self._key_lock(key)
        key_lock.acquire()
        try:
            cached = self.get(prompt, params)
            if cached is not None:
                return cached
            response = fn(prompt, *args, **kwargs)
            self._store(key, prompt, response, params)
            return response
        finally:
            self._release_key_lock(key, key_lock)

    # ------------------------------------------------------------ management
    def invalidate(self, prompt: str, params: Optional[dict] = None) -> None:
        key = self._key(prompt, params)
        with self._lock:
            self.backend.delete(key)
            self._recency.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self.backend.clear()
            self._recency.clear()

    def prune_expired(self) -> int:
        """Delete expired entries; return the number removed."""
        if self.ttl_seconds is None:
            return 0
        removed = 0
        with self._lock:
            for key in list(self._recency.keys()):
                record = self.backend.get(key)
                if record is not None and self._is_expired(record):
                    self.backend.delete(key)
                    self._recency.pop(key, None)
                    self.stats.record_expired()
                    removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._recency)

    def __contains__(self, prompt: str) -> bool:
        return self.get(prompt) is not None
