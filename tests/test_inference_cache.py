"""Test suite for inference-cache."""

from __future__ import annotations

import os
import time

import pytest

from inference_cache import (
    InferenceCache,
    MemoryBackend,
    SQLiteBackend,
    cached,
    estimate_tokens,
    exact_only,
    lexical_similarity,
    make_cache_key,
    normalize_prompt,
)


# ---------------------------------------------------------------- normalization
def test_normalize_collapses_whitespace():
    assert normalize_prompt("  hello\n\n  world\t!  ") == "hello world !"


def test_normalize_case_insensitive():
    assert normalize_prompt("Hello World", case_insensitive=True) == "hello world"
    assert normalize_prompt("Hello World") == "Hello World"


def test_cache_key_deterministic_and_param_sensitive():
    k1 = make_cache_key("hello", model="m", params={"t": 0.7})
    k2 = make_cache_key("  hello  ", model="m", params={"t": 0.7})
    k3 = make_cache_key("hello", model="m", params={"t": 0.0})
    k4 = make_cache_key("hello", model="other", params={"t": 0.7})
    assert k1 == k2  # formatting differences normalize away
    assert len(k1) == 64  # sha256 hex
    assert k1 != k3  # params are part of the key
    assert k1 != k4  # model is part of the key


# ----------------------------------------------------------------- similarity
def test_lexical_similarity_identical_and_unrelated():
    assert lexical_similarity("hello world", "hello world") == pytest.approx(1.0)
    assert lexical_similarity("the capital of France", "quantum chromodynamics") < 0.3


def test_lexical_similarity_near_duplicate():
    score = lexical_similarity(
        "What is the capital of France?", "what is the capital of france?"
    )
    assert score >= 0.92


def test_exact_only():
    assert exact_only("a", "a") == 1.0
    assert exact_only("a", "b") == 0.0


# ---------------------------------------------------------------------- cache
def _counter():
    calls = {"n": 0}

    def fn(prompt: str) -> str:
        calls["n"] += 1
        return f"resp:{prompt}"

    return fn, calls


def test_exact_hit_avoids_recompute():
    fn, calls = _counter()
    cache = InferenceCache()
    assert cache.call(fn, "hello") == "resp:hello"
    assert cache.call(fn, "hello") == "resp:hello"
    assert cache.call(fn, "  hello\n") == "resp:hello"  # normalized exact hit
    assert calls["n"] == 1
    assert cache.stats.hits == 2
    assert cache.stats.misses == 1


def test_semantic_hit():
    fn, calls = _counter()
    cache = InferenceCache(similarity_threshold=0.9)
    cache.call(fn, "What is the capital of France?")
    assert cache.call(fn, "what is the capital of france?") == (
        "resp:What is the capital of France?"
    )
    assert calls["n"] == 1
    assert cache.stats.hits == 1


def test_semantic_disabled_with_exact_only():
    fn, calls = _counter()
    cache = InferenceCache(similarity=exact_only)
    cache.call(fn, "What is the capital of France?")
    cache.call(fn, "what is the capital of france?")
    assert calls["n"] == 2  # near-duplicate is a miss without semantic matching


def test_ttl_expiry():
    fn, calls = _counter()
    cache = InferenceCache(ttl_seconds=0.05)
    cache.call(fn, "hello")
    time.sleep(0.08)
    cache.call(fn, "hello")
    assert calls["n"] == 2
    assert cache.stats.expired >= 1


def test_lru_eviction_max_size():
    fn, calls = _counter()
    cache = InferenceCache(max_size=2)
    cache.call(fn, "a")
    cache.call(fn, "b")
    cache.call(fn, "c")  # evicts "a"
    assert len(cache) == 2
    assert cache.stats.evictions == 1
    cache.call(fn, "a")  # miss again
    assert calls["n"] == 4


def test_invalidate_and_clear():
    fn, calls = _counter()
    cache = InferenceCache()
    cache.call(fn, "hello")
    cache.invalidate("hello")
    cache.call(fn, "hello")
    assert calls["n"] == 2
    cache.call(fn, "world")
    cache.clear()
    assert len(cache) == 0


def test_params_change_key():
    fn, calls = _counter()
    cache = InferenceCache()
    cache.call(fn, "hello", params={"temperature": 0.0})
    cache.call(fn, "hello", params={"temperature": 0.9})
    assert calls["n"] == 2


def test_stats_cost_estimate():
    cache = InferenceCache(price_per_1k=10.0)
    cache.call(lambda p: "word " * 100, "hello")
    cache.call(lambda p: "unused", "hello")  # hit; response is the cached one
    d = cache.stats.to_dict()
    assert d["hit_rate"] == 0.5
    assert d["tokens_served_from_cache"] == estimate_tokens("word " * 100)
    assert d["estimated_cost_saved_usd"] > 0


def test_sqlite_backend_persists(tmp_path):
    db = str(tmp_path / "cache.db")
    fn, calls = _counter()
    c1 = InferenceCache(backend=SQLiteBackend(db))
    c1.call(fn, "hello")
    assert calls["n"] == 1
    c2 = InferenceCache(backend=SQLiteBackend(db))  # new instance, same file
    assert c2.call(fn, "hello") == "resp:hello"
    assert calls["n"] == 1  # served from disk
    assert len(c2) == 1


def test_prune_expired(tmp_path):
    fn, _ = _counter()
    cache = InferenceCache(backend=SQLiteBackend(str(tmp_path / "c.db")), ttl_seconds=0.05)
    cache.call(fn, "a")
    cache.call(fn, "b")
    time.sleep(0.08)
    assert cache.prune_expired() == 2
    assert len(cache) == 0


# ------------------------------------------------------------------ decorator
def test_cached_decorator():
    calls = {"n": 0}

    @cached(max_size=10)
    def ask(prompt: str) -> str:
        calls["n"] += 1
        return "answer"

    assert ask("q") == "answer"
    assert ask("q") == "answer"
    assert calls["n"] == 1
    assert ask.cache.stats.hits == 1


def test_cached_decorator_bare():
    calls = {"n": 0}

    @cached
    def ask(prompt: str) -> str:
        calls["n"] += 1
        return "answer"

    ask("q")
    ask("q")
    assert calls["n"] == 1


def test_memory_backend_thread_sanity():
    backend = MemoryBackend()
    backend.put("k", {"value": "v", "prompt": "p", "created_at": 0.0})
    assert backend.get("k")["value"] == "v"
    assert [k for k, _ in backend.scan()] == ["k"]
    backend.delete("k")
    assert backend.get("k") is None
