"""Threaded demo: share one cache across threads with exactly-once calls.

Simulates a web server where many threads ask an LLM the same questions
concurrently. A slow fake model counts its invocations so you can see that
each distinct prompt is computed exactly once no matter how many threads
race on it.

Run from the repo root:

    python examples/threaded_demo.py
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from inference_cache import InferenceCache

MODEL_CALLS = 0
MODEL_CALLS_LOCK = threading.Lock()


def fake_llm(prompt: str) -> str:
    """Stand-in for a slow model call; counts real invocations."""
    global MODEL_CALLS
    with MODEL_CALLS_LOCK:
        MODEL_CALLS += 1
    time.sleep(0.05)  # pretend the model is thinking
    return f"[{prompt}] -> summarized"


def main() -> None:
    cache = InferenceCache(max_size=1000, ttl_seconds=3600)
    prompts = [
        "Summarize the quarterly report.",
        "Draft a launch announcement.",
        "Explain idempotency keys.",
    ]

    def ask(prompt: str) -> str:
        return cache.call(fake_llm, prompt)

    # 24 threads, each asking every prompt 5 times = 360 requests.
    with ThreadPoolExecutor(max_workers=24) as pool:
        answers = list(pool.map(ask, prompts * 5 * 24))

    assert len(set(answers)) == len(prompts), "every prompt got a stable answer"
    print(f"Requests served : {len(answers)}")
    print(f"Model calls     : {MODEL_CALLS} (one per distinct prompt)")
    print(f"Cache stats     : {cache.stats.to_dict()}")
    print(f"Hit rate        : {cache.stats.hit_rate:.1%}")


if __name__ == "__main__":
    main()
