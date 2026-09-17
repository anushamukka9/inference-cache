"""Command-line interface for inference-cache.

Commands:
  demo     Run a self-contained demo showing hits, misses, and stats.
  put      Store a response for a prompt.
  get      Fetch a cached response for a prompt.
  stats    Print hit/miss statistics and cost-saved estimates.
  clear    Empty the cache.
  prune    Remove expired entries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from inference_cache import (
    InferenceCache,
    SQLiteBackend,
    cached,
    estimate_tokens,
    lexical_similarity,
    normalize_prompt,
)

DEFAULT_DB = os.environ.get("INFERENCE_CACHE_DB", "inference_cache.db")


def _make_cache(db: str, **kwargs) -> InferenceCache:
    return InferenceCache(backend=SQLiteBackend(db), **kwargs)


def cmd_demo(_: argparse.Namespace) -> int:
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        time.sleep(0.05)  # pretend the model is slow
        return f"[model] Answer to: {prompt[:60]}"

    cache = InferenceCache(ttl_seconds=3600, max_size=1000, similarity_threshold=0.8)

    prompts = [
        "What is the capital of France?",
        "  What is the capital of France?  ",  # whitespace variant -> exact hit
        "what is the capital of france?",  # near-duplicate -> semantic hit
        "Explain TCP three-way handshake.",
        "Explain the TCP three-way handshake",  # near-duplicate -> semantic hit
    ]
    for prompt in prompts:
        t0 = time.perf_counter()
        answer = cache.call(fake_llm, prompt)
        dt = (time.perf_counter() - t0) * 1000
        print(f"prompt: {prompt!r}\n  -> {answer} ({dt:.1f} ms)\n")

    print(f"LLM calls actually made: {calls['n']} (of {len(prompts)} requests)")
    print("stats:", json.dumps(cache.stats.to_dict(), indent=2))
    return 0


def cmd_put(args: argparse.Namespace) -> int:
    cache = _make_cache(args.db)
    # Store via the public API: first call misses and records the response.
    cache.call(lambda _prompt: args.response, args.prompt)
    print(f"stored {len(args.response)} chars for prompt {args.prompt[:48]!r}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    cache = _make_cache(args.db, similarity_threshold=args.threshold)
    value = cache.get(args.prompt)
    if value is None:
        print("MISS: no cached response", file=sys.stderr)
        return 1
    print(value)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cache = _make_cache(args.db)
    report = cache.stats.to_dict()
    report["entries"] = len(cache)
    print(json.dumps(report, indent=2))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    cache = _make_cache(args.db)
    n = len(cache)
    cache.clear()
    print(f"cleared {n} entries")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    cache = _make_cache(args.db, ttl_seconds=args.ttl)
    removed = cache.prune_expired()
    print(f"pruned {removed} expired entries")
    return 0


def cmd_similarity(args: argparse.Namespace) -> int:
    score = lexical_similarity(args.a, args.b)
    print(f"similarity: {score:.4f}")
    print(f"normalized a: {normalize_prompt(args.a)!r}")
    print(f"normalized b: {normalize_prompt(args.b)!r}")
    print(f"tokens a/b: {estimate_tokens(args.a)} / {estimate_tokens(args.b)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-cache",
        description="LLM inference caching layer: cache prompt->response pairs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run a self-contained caching demo")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("put", help="store a response for a prompt")
    p.add_argument("prompt")
    p.add_argument("response")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("get", help="fetch a cached response")
    p.add_argument("prompt")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--threshold", type=float, default=0.92)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("stats", help="show cache statistics")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("clear", help="empty the cache")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser("prune", help="remove expired entries")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--ttl", type=float, default=86400)
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("similarity", help="score the similarity of two prompts")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_similarity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
