"""Runnable quickstart for inference-cache.

Wraps a fake (slow, expensive) LLM call with the @cached decorator and shows
exact hits, semantic hits, TTL, and the stats report. Run with:

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from inference_cache import InferenceCache, SQLiteBackend, cached  # noqa: E402


def fake_llm(prompt: str) -> str:
    """Stand-in for a real model call: slow and (pretend) expensive."""
    time.sleep(0.2)
    return f"[{len(prompt)} chars in] The capital of France is Paris."


@cached(ttl_seconds=3600, max_size=100, model="demo-model")
def answer(prompt: str) -> str:
    return fake_llm(prompt)


def main() -> None:
    print("== exact + semantic caching ==")
    for prompt in [
        "What is the capital of France?",
        "What is the capital of France?",  # identical -> exact hit
        "  what is the capital of france? ",  # reworded -> semantic hit
        "Summarize the TCP handshake.",  # new -> miss
    ]:
        t0 = time.perf_counter()
        result = answer(prompt)
        dt = (time.perf_counter() - t0) * 1000
        print(f"{dt:7.1f} ms  {prompt!r}\n           -> {result}")

    print("\n== stats ==")
    print(answer.cache.stats.to_dict())

    print("\n== persistent backend (SQLite) ==")
    persistent = InferenceCache(backend=SQLiteBackend("/tmp/quickstart_demo.db"))
    persistent.call(fake_llm, "What is the capital of France?")
    print("entries:", len(persistent), "| files persist across runs at /tmp/quickstart_demo.db")


if __name__ == "__main__":
    main()
