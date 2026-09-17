"""Drop-in decorator that adds caching to any ``fn(prompt, ...) -> str``."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from .backends import CacheBackend
from .cache import InferenceCache
from .similarity import SimilarityFn


def cached(
    _fn: Optional[Callable[..., str]] = None,
    *,
    backend: Optional[CacheBackend] = None,
    ttl_seconds: Optional[float] = None,
    max_size: Optional[int] = None,
    model: str = "default",
    default_params: Optional[dict] = None,
    similarity: Optional[SimilarityFn] = None,
    similarity_threshold: float = 0.92,
    case_insensitive: bool = False,
    price_per_1k: float | dict[str, float] | None = None,
) -> Callable[..., str]:
    """Decorate an LLM-calling function with a shared :class:`InferenceCache`.

    Works both as ``@cached`` and ``@cached(ttl_seconds=3600, max_size=1000)``.
    The decorated function exposes its cache as ``fn.cache``.
    """

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        cache = InferenceCache(
            backend=backend,
            ttl_seconds=ttl_seconds,
            max_size=max_size,
            model=model,
            default_params=default_params,
            similarity=similarity,
            similarity_threshold=similarity_threshold,
            case_insensitive=case_insensitive,
            price_per_1k=price_per_1k,
        )

        @functools.wraps(fn)
        def wrapper(prompt: str, *args: Any, **kwargs: Any) -> str:
            return cache.call(fn, prompt, *args, **kwargs)

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator
