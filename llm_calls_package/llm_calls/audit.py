from __future__ import annotations

import json
from typing import Dict

from .client import DEFAULT_MODEL, call_model_text
from .personas import AUDITOR_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_audit


def _build_prompt(diff: str, checklist: dict) -> str:
    # Deliberately: only diff + checklist go into this prompt. This function's
    # signature has no parameter for hypothesis/rationale, which is what
    # actually enforces "blind" here — there is no field to accidentally
    # pass it through even if a caller had it in scope.
    return (
        "Checklist to check (JSON — only check keys present here):\n"
        f"{json.dumps(checklist, indent=2)}\n\n"
        "Code diff to audit:\n"
        "```diff\n"
        f"{diff}\n"
        "```\n\n"
        "Respond with the required JSON."
    )


def audit(diff: str, checklist: dict) -> Dict:
    """Audit a code diff against a fixed checklist, BLIND to the hypothesis
    or rationale that motivated it. See personas.AUDITOR_SYSTEM_PROMPT.

    Args:
        diff: the raw code diff text, and nothing else about its context.
        checklist: dict of rule-name -> (any truthy marker that it should be
            checked); only keys present here are evaluated.

    Raises:
        LLMSchemaError: if the model can't produce schema-valid JSON within
            the retry budget.
    """
    prompt = _build_prompt(diff, checklist)

    def call_fn(p: str) -> str:
        return call_model_text(AUDITOR_SYSTEM_PROMPT, p, model=DEFAULT_MODEL)

    return call_with_schema_retry(call_fn, prompt, validate_audit)
