# Usage guide

This guide walks through every major feature of `inference-cache`. See
`examples/quickstart.py` for a runnable version of the basics.

## 1. The decorator (simplest path)

```python
from inference_cache import cached

@cached(ttl_seconds=3600, max_size=10_000, model="gpt-4o-mini")
def ask(prompt: str) -> str:
    return my_llm_client.complete(prompt)  # your real call here

ask("What is the capital of France?")   # miss -> calls the model
ask("What is the capital of France?")   # hit  -> served from cache
ask("what is the capital of france?")   # semantic hit (default threshold 0.92)
```

The wrapped function's cache is available as `ask.cache` — inspect
`ask.cache.stats.to_dict()`, call `ask.cache.clear()`, etc.

The decorator also works bare: `@cached` with no arguments.

## 2. Exact vs semantic matching

Lookup order per request:

1. **Exact** — SHA-256 over normalized prompt + model + params.
2. **Semantic** — scans stored prompts with a similarity function; reuses the
   best entry scoring above `similarity_threshold`.
3. **Miss** — calls the function and stores the result.

Bring your own similarity (e.g. embedding cosine) with the documented
`SimilarityFn` signature `(a, b) -> float`:

```python
from inference_cache import InferenceCache, exact_only

cache = InferenceCache(
    similarity=exact_only,          # disable semantic matching entirely
    # or: similarity=my_embedding_cosine, similarity_threshold=0.97
)
```

The bundled default, `lexical_similarity`, blends token-set Jaccard (50%),
TF cosine (30%) and character ratio (20%). It's dependency-free and good at
catching whitespace, casing, and light rewording — not deep paraphrases.
For production paraphrase detection, plug in an embedding model.

## 3. Cache keys and normalization

Keys are deterministic: same normalized prompt + model + params → same key.
Normalization NFC-normalizes unicode, collapses whitespace, and strips edges.
Options:

- `case_insensitive=True` — fold prompts to lowercase before keying.
- Params matter: `temperature=0.7` and `temperature=0.0` produce different keys,
  because sampling settings change the distribution of valid responses.

## 4. Backends

```python
from inference_cache import InferenceCache, MemoryBackend, SQLiteBackend, RedisBackend

InferenceCache()                                   # in-memory (default)
InferenceCache(backend=SQLiteBackend("cache.db"))   # persistent, stdlib only
InferenceCache(backend=RedisBackend("redis://localhost:6379/0"))  # shared
```

Redis needs `pip install inference-cache[redis]`.

## 5. Eviction: TTL + LRU

```python
cache = InferenceCache(ttl_seconds=86_400, max_size=50_000)
cache.prune_expired()   # eagerly drop expired entries; returns count removed
```

- `ttl_seconds`: entries older than this are expired on read (counted in stats).
- `max_size`: when full, least-recently-used entries are evicted on insert.

Both work uniformly across all backends.

## 6. Statistics and cost estimates

```python
stats = cache.stats.to_dict()
# {'hits': 42, 'misses': 8, 'hit_rate': 0.84, 'evictions': 0, 'expired': 1,
#  'tokens_served_from_cache': 3150, 'estimated_cost_saved_usd': 0.0063}
```

Set real prices for accurate estimates — a float (USD per 1K tokens) or a
per-model dict:

```python
cache = InferenceCache(price_per_1k={"gpt-4o": 0.005, "gpt-4o-mini": 0.0006})
```

Token counts are heuristic (words × 1.3); treat estimates as directional.

## 7. CLI

```bash
inference-cache demo                                   # end-to-end demo
inference-cache put "prompt" "response" --db cache.db  # seed an entry
inference-cache get "prompt" --db cache.db             # fetch (exit 1 on miss)
inference-cache stats --db cache.db                    # JSON stats report
inference-cache prune --db cache.db --ttl 86400        # drop expired entries
inference-cache clear --db cache.db                    # empty the cache
inference-cache similarity "prompt a" "prompt b"       # score similarity
```

The CLI uses a SQLite backend (default `./inference_cache.db`, override with
`INFERENCE_CACHE_DB`).
