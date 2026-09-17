"""Pluggable storage backends for cached inference results.

Every backend stores entries as ``(key, value, prompt, timestamp)`` records and
implements the same small interface, so backends are interchangeable:

- :class:`MemoryBackend` — in-memory, per-process. Fastest; nothing to install.
- :class:`SQLiteBackend` — persistent, single file, stdlib only.
- :class:`RedisBackend` — shared across processes/hosts; needs ``redis-py``.

TTL expiry and max-size LRU eviction are enforced by the :class:`InferenceCache`
wrapper, but :class:`MemoryBackend` also tracks recency internally so direct
use stays consistent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Iterator, Protocol


class StoredEntry(Protocol):
    key: str
    value: str
    prompt: str
    created_at: float


class CacheBackend(Protocol):
    """Storage interface every backend must implement."""

    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, record: dict[str, Any]) -> None: ...
    def delete(self, key: str) -> None: ...
    def clear(self) -> None: ...
    def keys(self) -> list[str]: ...
    def scan(self) -> Iterator[tuple[str, str]]: ...


class MemoryBackend:
    """Thread-safe in-memory backend backed by an ``OrderedDict`` (LRU order)."""

    def __init__(self) -> None:
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._store.get(key)
            if record is not None:
                self._store.move_to_end(key)  # mark as recently used
            return record

    def put(self, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = record
            self._store.move_to_end(key)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def scan(self) -> Iterator[tuple[str, str]]:
        """Yield ``(key, prompt)`` for every entry, oldest first."""
        with self._lock:
            snapshot = list(self._store.items())
        for key, record in snapshot:
            yield key, record.get("prompt", "")


class SQLiteBackend:
    """Persistent backend stored in a SQLite file. Stdlib only."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS inference_cache ("
                " key TEXT PRIMARY KEY,"
                " value TEXT NOT NULL,"
                " prompt TEXT NOT NULL DEFAULT '',"
                " created_at REAL NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT key, value, prompt, created_at FROM inference_cache"
                    " WHERE key = ?",
                    (key,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def put(self, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO inference_cache"
                    " (key, value, prompt, created_at) VALUES (?, ?, ?, ?)",
                    (
                        key,
                        record["value"],
                        record.get("prompt", ""),
                        record.get("created_at", time.time()),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def delete(self, key: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM inference_cache WHERE key = ?", (key,))
                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM inference_cache")
                conn.commit()
            finally:
                conn.close()

    def keys(self) -> list[str]:
        with self._lock:
            conn = self._connect()
            try:
                return [
                    row[0]
                    for row in conn.execute(
                        "SELECT key FROM inference_cache ORDER BY created_at"
                    )
                ]
            finally:
                conn.close()

    def scan(self) -> Iterator[tuple[str, str]]:
        """Yield ``(key, prompt)`` for every entry, oldest first."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT key, prompt FROM inference_cache ORDER BY created_at"
                ).fetchall()
                snapshot = [(row[0], row[1] or "") for row in rows]
            finally:
                conn.close()
        yield from snapshot


class RedisBackend:
    """Shared backend for multi-process / multi-host deployments.

    Requires the ``redis`` package (``pip install inference-cache[redis]``).
    Values are stored as JSON hashes under the key prefix ``ic:``.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "ic:") -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RedisBackend requires the 'redis' package: "
                "pip install inference-cache[redis]"
            ) from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> dict[str, Any] | None:
        raw = self._client.get(self._k(key))
        return json.loads(raw) if raw else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self._client.set(self._k(key), json.dumps(record))

    def delete(self, key: str) -> None:
        self._client.delete(self._k(key))

    def clear(self) -> None:
        for redis_key in self._client.scan_iter(f"{self.prefix}*"):
            self._client.delete(redis_key)

    def keys(self) -> list[str]:
        return [
            k[len(self.prefix):]
            for k in self._client.scan_iter(f"{self.prefix}*")
        ]

    def scan(self) -> Iterator[tuple[str, str]]:
        for redis_key in self._client.scan_iter(f"{self.prefix}*"):
            raw = self._client.get(redis_key)
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:  # pragma: no cover
                continue
            yield redis_key[len(self.prefix):], record.get("prompt", "")
