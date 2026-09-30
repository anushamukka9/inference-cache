"""Thread-safety tests: concurrent access keeps the cache consistent."""

import threading
import time

import pytest

from inference_cache import InferenceCache, MemoryBackend, SQLiteBackend


def slow_fn_factory(calls: list, delay: float = 0.02):
    def fn(prompt: str) -> str:
        calls.append(prompt)
        time.sleep(delay)
        return f"answer:{prompt}"

    return fn


def hammer(cache, fn, prompts, repeats=4):
    errors = []

    def worker():
        try:
            for _ in range(repeats):
                for prompt in prompts:
                    cache.call(fn, prompt)
        except Exception as exc:  # noqa: BLE001 - fail the test with the error
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"worker threads raised: {errors!r}"


def test_concurrent_same_prompt_calls_fn_once():
    """16 threads racing on one cold key: exactly one model call, all served."""
    cache = InferenceCache(similarity_threshold=2.0)  # exact-only for determinism
    calls: list = []
    fn = slow_fn_factory(calls)

    results = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()  # maximize the race on the cold key
        results.append(cache.call(fn, "shared prompt"))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == ["shared prompt"], f"fn ran {len(calls)} times, expected 1"
    assert results == ["answer:shared prompt"] * 16
    assert cache.stats.hits == 15
    assert cache.stats.misses == 1


def test_concurrent_mixed_prompts_stats_consistent():
    cache = InferenceCache()
    calls: list = []
    prompts = [f"prompt-{i}" for i in range(8)]
    hammer(cache, slow_fn_factory(calls), prompts)

    total = cache.stats.hits + cache.stats.misses
    assert total == cache.stats.total_requests
    assert cache.stats.misses == len(prompts)  # each distinct prompt missed once
    assert cache.stats.hits == total - len(prompts)
    # Every stored key resolves back to its value.
    for prompt in prompts:
        assert cache.get(prompt) == f"answer:{prompt}"


def test_concurrent_inserts_respect_max_size():
    cache = InferenceCache(max_size=50)
    calls: list = []

    def worker(n):
        for i in range(25):
            cache.call(slow_fn_factory(calls, delay=0), f"w{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cache) <= 50
    assert len(cache.backend.keys()) <= 50


def test_concurrent_get_and_invalidate_no_errors():
    cache = InferenceCache()
    cache.call(lambda p: "v", "key")
    errors = []

    def reader():
        try:
            for _ in range(200):
                cache.get("key")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def invalidator():
        try:
            for _ in range(50):
                cache.invalidate("key")
                cache.call(lambda p: "v", "key")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    threads.append(threading.Thread(target=invalidator))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_sqlite_backend_concurrent_same_prompt(tmp_path):
    cache = InferenceCache(backend=SQLiteBackend(str(tmp_path / "cache.db")))
    calls: list = []
    fn = slow_fn_factory(calls)

    def worker():
        for _ in range(5):
            cache.call(fn, "db prompt")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == ["db prompt"]
    assert cache.stats.hits + cache.stats.misses == 40


def test_stats_counters_consistent_under_threads():
    from inference_cache.stats import CacheStats

    stats = CacheStats()

    def worker():
        for _ in range(500):
            stats.record_hit("some response text")
            stats.record_miss()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stats.hits == 4000
    assert stats.misses == 4000
    assert stats.total_requests == 8000
    assert stats.hit_rate == pytest.approx(0.5)
