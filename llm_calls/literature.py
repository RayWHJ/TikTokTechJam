from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .client import call_model_with_search
from .routing import effort_for, model_for
from .usage import KIND_LITERATURE
from .personas import LITERATURE_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_literature

# File-backed so the cache survives across separate process runs, not just
# within one — useful since the same bottleneck string can plausibly recur
# across many search-tree nodes over the course of a run.
_CACHE_DIR = os.environ.get("LLM_CALLS_CACHE_DIR", ".llm_calls_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "literature_cache.json")

_memory_cache: Optional[Dict[str, Dict]] = None


def _normalize_key(bottleneck: str) -> str:
    return " ".join(bottleneck.strip().lower().split())


def _load_cache() -> Dict[str, Dict]:
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _memory_cache = json.load(f)
                return _memory_cache
        except (json.JSONDecodeError, OSError):
            pass  # corrupted/unreadable cache file — start fresh rather than crash
    _memory_cache = {}
    return _memory_cache


def _save_cache(cache: Dict[str, Dict]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _build_prompt(bottleneck: str) -> str:
    return (
        f"Bottleneck to research: {bottleneck}\n\n"
        "Search for published techniques that address this bottleneck. "
        "Respond with the required JSON."
    )


def ground_in_literature(bottleneck: str) -> Dict:
    """Find published techniques relevant to a given bottleneck, using a
    retrieval-enabled model. Results are cached by (normalized) bottleneck
    string so repeat lookups don't re-call the API.

    Raises:
        LLMSchemaError: if the model can't produce schema-valid JSON within
            the retry budget. Nothing is cached on failure.
    """
    cache = _load_cache()
    key = _normalize_key(bottleneck)
    if key in cache:
        return cache[key]

    prompt = _build_prompt(bottleneck)

    def call_fn(p: str) -> str:
        return call_model_with_search(LITERATURE_SYSTEM_PROMPT, p,
                                      model=model_for(KIND_LITERATURE),
                                      effort=effort_for(KIND_LITERATURE),
                                      kind=KIND_LITERATURE)

    result = call_with_schema_retry(call_fn, prompt, validate_literature)

    cache[key] = result
    _save_cache(cache)
    return result
