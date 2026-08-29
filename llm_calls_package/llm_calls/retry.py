"""
Shared retry/validation wrapper. Every one of the 5 contract functions
routes its model call through call_with_schema_retry so the retry-on-
malformed-JSON behavior is implemented exactly once.

Contract: on malformed output, retry up to `max_retries` times with the
parse/validation error appended to the prompt; if it still fails, raise
LLMSchemaError rather than returning garbage or crashing with a raw
JSONDecodeError/KeyError.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .exceptions import LLMSchemaError

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw: str) -> Any:
    """Parse a model's raw text response into a Python object, tolerating
    the common ways models deviate from "just return JSON":
      - wrapped in a ```json ... ``` fence
      - leading/trailing commentary around a JSON object or array
    Raises ValueError (not JSONDecodeError) so callers have one exception
    type to catch."""
    text = raw.strip()

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to slicing out the outermost {...} or [...] block, in case
    # the model added a sentence of commentary before/after the JSON.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Could not parse any JSON object or array out of the response. "
        f"Raw response started with: {raw[:200]!r}"
    )


def _augment_prompt_with_error(base_prompt: str, raw_response: str, error: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"---\n"
        f"Your previous response could not be used:\n"
        f"{raw_response[:1000]}\n\n"
        f"Validation error: {error}\n\n"
        f"Return ONLY strict JSON matching the required schema exactly. "
        f"No markdown code fences, no commentary before or after the JSON."
    )


def call_with_schema_retry(
    call_fn: Callable[[str], str],
    base_user_prompt: str,
    validate_fn: Callable[[Any], Any],
    max_retries: int = 2,
) -> Any:
    """
    call_fn(prompt: str) -> str          makes the actual model call, returns raw text
    base_user_prompt                     the initial user-turn prompt
    validate_fn(parsed) -> Any           validates + returns a cleaned result,
                                          or raises ValueError on any problem
    max_retries                          number of retries AFTER the first attempt
                                          (so max_retries=2 means up to 3 total calls)

    Returns whatever validate_fn returns on success.
    Raises LLMSchemaError if every attempt fails.
    """
    last_error = None
    last_raw = None
    prompt = base_user_prompt

    for attempt in range(max_retries + 1):
        raw = call_fn(prompt)
        last_raw = raw
        try:
            parsed = extract_json(raw)
            result = validate_fn(parsed)
            return result
        except ValueError as e:
            last_error = str(e)
            prompt = _augment_prompt_with_error(base_user_prompt, raw, last_error)
            continue

    raise LLMSchemaError(
        message=(
            f"Model failed to produce schema-valid JSON after {max_retries + 1} attempt(s). "
            f"Last error: {last_error}"
        ),
        attempts=max_retries + 1,
        last_raw_response=last_raw,
        last_error=last_error,
    )
