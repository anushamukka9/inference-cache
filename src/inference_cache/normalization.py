"""Prompt normalization for cache keys.

Normalization ensures that semantically identical prompts that differ only in
trivial formatting (extra whitespace, line endings, unicode quirks) map to the
same cache key, so cache lookups stay stable across callers.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_prompt(
    prompt: str,
    *,
    case_insensitive: bool = False,
    strip_markdown_code_fences: bool = False,
) -> str:
    """Return a canonical form of *prompt*.

    Steps:
      1. Unicode NFC normalization.
      2. Normalize line endings to ``\\n`` and strip surrounding whitespace.
      3. Collapse every run of whitespace (spaces, tabs, newlines) to one space.
      4. Optionally fold to lowercase.
      5. Optionally strip surrounding Markdown code fences (```...```).
    """
    text = unicodedata.normalize("NFC", prompt)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_markdown_code_fences:
        text = re.sub(r"^```[a-zA-Z0-9+-]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if case_insensitive:
        text = text.casefold()
    return text


def make_cache_key(
    prompt: str,
    *,
    model: str = "default",
    params: dict | None = None,
    case_insensitive: bool = False,
) -> str:
    """Build a deterministic SHA-256 cache key from prompt + model + params.

    Two calls with the same normalized prompt, model identifier and effective
    sampling parameters produce the same key. Param dict ordering is ignored.
    """
    normalized = normalize_prompt(prompt, case_insensitive=case_insensitive)
    canonical_params = json.dumps(
        params or {}, sort_keys=True, separators=(",", ":"), default=str
    )
    digest_input = "\n".join([model, normalized, canonical_params]).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()
