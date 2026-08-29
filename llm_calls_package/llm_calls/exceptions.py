"""Exceptions raised by llm_calls. Teammates' orchestrator code should catch
LLMSchemaError specifically rather than a bare Exception."""

from __future__ import annotations

from typing import Optional


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
