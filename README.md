# inference-cache

A drop-in **LLM inference caching layer**: wrap any `fn(prompt) -> str` and
serve repeated — or near-duplicate — prompts from cache instead of paying for
another model call.

- **Exact + semantic matching** — deterministic SHA-256 keys over normalized
  prompt + model + params, plus pluggable similarity (dependency-free lexical
  default) for near-duplicate prompts.
- **Prompt normalization** — unicode NFC, whitespace collapsing, optional
  case folding, so trivial formatting differences don't bust the cache.
- **TTL + LRU eviction** — time-based expiry and max-size least-recently-used
  eviction, uniform across all backends.
- **Hit/miss statistics & cost-saved estimates** — with configurable
  per-model pricing.
- **Pluggable backends** — in-memory (default), SQLite (persistent, stdlib
  only), Redis (shared across processes/hosts, optional extra).
- **CLI** — demo, put/get, stats, prune, clear, similarity scoring.

## Install

```bash
pip install inference-cache
# with Redis support:
pip install "inference-cache[redis]"
```

Requires Python 3.9+. The core has **zero runtime dependencies**.

## Quickstart

```python
from inference_cache import cached

@cached(ttl_seconds=3600, max_size=10_000, model="gpt-4o-mini")
def ask(prompt: str) -> str:
    return llm_client.complete(prompt)

ask("What is the capital of France?")   # miss -> calls the model
ask("What is the capital of France?")   # hit  -> served from cache
ask("what is the capital of france?")   # semantic hit (similarity >= 0.92)

print(ask.cache.stats.to_dict())
# {'hits': 2, 'misses': 1, 'hit_rate': 0.6667, ...,
#  'estimated_cost_saved_usd': 0.0001}
```

Or manage a cache object directly:

```python
from inference_cache import InferenceCache, SQLiteBackend

cache = InferenceCache(
    backend=SQLiteBackend("cache.db"),
    ttl_seconds=86_400,
    max_size=50_000,
    price_per_1k={"gpt-4o": 0.005, "gpt-4o-mini": 0.0006},
)
answer = cache.call(llm_client.complete, "Explain the TCP handshake.")
cache.prune_expired()
```

Run the runnable example: `python examples/quickstart.py`, or the CLI demo:
`inference-cache demo`. Full walkthrough in [docs/usage.md](docs/usage.md).

## API

| Symbol | Description |
|---|---|
| `InferenceCache(...)` | Main cache: exact + semantic lookup, TTL, LRU, stats |
| `cached(...)` / `@cached` | Decorator adding a shared cache to `fn(prompt, ...) -> str` (`fn.cache`) |
| `MemoryBackend` | Thread-safe in-memory backend (default) |
| `SQLiteBackend(path)` | Persistent single-file backend, stdlib only |
| `RedisBackend(url)` | Shared backend; needs `pip install inference-cache[redis]` |
| `CacheStats` | Hits, misses, evictions, expiry, tokens served, cost saved |
| `normalize_prompt` / `make_cache_key` | Normalization + deterministic key building |
| `lexical_similarity` / `exact_only` / `find_best_match` | Similarity plumbing; bring your own `SimilarityFn` |

## Architecture

```
prompt ──► normalize ──► SHA-256(model, prompt, params)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              exact lookup            semantic scan (similarity ≥ threshold)
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    hit → return cached      miss → call fn → store
                                                         │
                                   TTL expiry + LRU eviction on insert
```

Lookup order is always exact → semantic → miss. Storage backends implement a
small `CacheBackend` protocol (`get`/`put`/`delete`/`clear`/`keys`/`scan`), so
custom backends (DynamoDB, Postgres, …) are a ~30-line class.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — Copyright (c) 2026 Anusha Mukka. See [LICENSE](LICENSE).
