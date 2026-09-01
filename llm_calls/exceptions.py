"""Exceptions raised by llm_calls. Teammates' orchestrator code should catch
LLMSchemaError specifically rather than a bare Exception."""

from __future__ import annotations

from typing import Optional


class LLMTruncatedError(Exception):
    """Raised when the model returned no usable text because it ran out of
    output budget (or was otherwise cut off) — a CONFIGURATION failure, not a
    schema failure.

    This exists because the two are indistinguishable downstream and call for
    opposite responses. Reasoning tokens count against `max_output_tokens` in the
    Responses API, so a reasoning model handed a 2000-token ceiling can spend the
    entire budget thinking and return `output_text == ""` with
    `status="incomplete"`. `call_with_schema_retry` then sees unparseable empty
    text, retries with the same ceiling, fails identically, and raises
    LLMSchemaError — which reads as "the model can't follow the schema" when the
    truth is "the ceiling is too low". `_apply_verdict` swallows that silently and
    `diagnose` propagates it as a crash, so the symptom of a wrong constant is a
    model that looks broken.

    Attributes:
        reason: the API's own `incomplete_details.reason` where available
            (e.g. "max_output_tokens", "content_filter"), else a description.
        model: the model id that was called.
        max_tokens: the `max_output_tokens` that was in force.
        status: the response's `status` field, if any.
    """

    def __init__(self, message: str, reason: Optional[str] = None,
                 model: Optional[str] = None, max_tokens: Optional[int] = None,
                 status: Optional[str] = None):
        super().__init__(message)
        self.reason = reason
        self.model = model
        self.max_tokens = max_tokens
        self.status = status


class LLMSchemaError(Exception):
    """Raised when a model response could not be coerced into the required
    JSON schema after all retries were exhausted.

    Attributes:
        attempts: total number of model calls made (including the first).
        last_raw_response: the raw text of the final failed attempt, kept
            for debugging/logging — callers should not try to parse this
            themselves, it is here purely for diagnostics.
        last_error: the validation error message from the final attempt.
    """

    def __init__(self, message: str, attempts: int, last_raw_response: Optional[str] = None,
                 last_error: Optional[str] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_raw_response = last_raw_response
        self.last_error = last_error
