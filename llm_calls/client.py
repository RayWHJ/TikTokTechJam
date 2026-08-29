"""
Thin wrapper around the OpenAI API. Kept separate from the 5 contract
functions so the retry/schema logic never has to know about SDK details,
and so this is the only file you need to touch if the team switches
providers or model names later.

Uses OpenAI's Responses API for both plain text and web-search-enabled
calls — it's the unified, current-generation API and supports the
web_search tool cleanly.

Configure via environment variables (all optional, sensible defaults below):
    LLM_CALLS_API_KEY        overrides OPENAI_API_KEY if you want a
                              separate key/budget for this module
    LLM_CALLS_MODEL          model used for diagnose / hypothesis / audit /
                              literature grounding  (default: gpt-4o-mini)
    LLM_CALLS_CHEAP_MODEL    smaller/cheaper model used for dedup escalation
                              (default: gpt-4.1-nano)

Note on temperature: intentionally omitted from the Responses API calls.
Reasoning models (gpt-5, o-series) reject the parameter outright, and for
the non-reasoning defaults the small quality gain from tuning it isn't
worth the portability cost. If a downstream caller needs sampling
variance, add it back guarded by a model-family check.
"""

from __future__ import annotations

import os
from typing import Optional

import openai

DEFAULT_MODEL = os.environ.get("LLM_CALLS_MODEL", "gpt-4o-mini")
DEFAULT_CHEAP_MODEL = os.environ.get("LLM_CALLS_CHEAP_MODEL", "gpt-4.1-nano")

_client: Optional[openai.OpenAI] = None


def get_client() -> openai.OpenAI:
    """Lazily construct and cache the SDK client (avoids paying the
    construction cost, and avoids requiring an API key at import time —
    only needed once you actually call a function)."""
    global _client
    if _client is None:
        api_key = os.environ.get("LLM_CALLS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        _client = openai.OpenAI(api_key=api_key)
    return _client


def call_model_text(system: str, user: str, model: str = DEFAULT_MODEL,
                    max_tokens: int = 2000) -> str:
    """Plain text-in/text-out call via the Responses API, no tools.
    Returns the model's response as a single string."""
    client = get_client()
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
    )
    return resp.output_text


def call_model_with_search(system: str, user: str, model: str = DEFAULT_MODEL,
                            max_tokens: int = 3000) -> str:
    """Retrieval-enabled call using OpenAI's server-executed web_search
    tool via the Responses API. The search happens inside this single API
    call (the server runs the tool itself); we only need to read back the
    final text output.

    Note: `output_text` concatenates only the text output items — any
    intermediate search-tool-call items are excluded automatically, which
    is what we want here."""
    client = get_client()
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
        tools=[{"type": "web_search"}],
    )
    return resp.output_text