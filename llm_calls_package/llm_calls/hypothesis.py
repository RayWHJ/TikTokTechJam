from __future__ import annotations

import json
from functools import partial
from typing import Dict, List

from .client import DEFAULT_MODEL, call_model_text
from .personas import HYPOTHESIS_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_hypothesis_list

# Tunable thresholds for deciding how many hypotheses to request.
LOW_CONFIDENCE_THRESHOLD = 0.5
_PLATEAU_KEYWORDS = ("plateau", "stagnant", "stagnated", "no improvement", "flat")


def _suggests_plateau(diagnosis: dict) -> bool:
    text = f"{diagnosis.get('bottleneck', '')} {diagnosis.get('evidence', '')}".lower()
    return any(kw in text for kw in _PLATEAU_KEYWORDS)


def _decide_count(diagnosis: dict) -> int:
    confidence = diagnosis.get("confidence", 1.0)
    if confidence < LOW_CONFIDENCE_THRESHOLD or _suggests_plateau(diagnosis):
        return 3
    return 1


def _build_prompt(diagnosis: dict, evidence_card: dict, count: int) -> str:
    plural = "hypothesis" if count == 1 else f"exactly {count} hypotheses"
    return (
        "Diagnosis (JSON):\n"
        f"{json.dumps(diagnosis, indent=2)}\n\n"
        "Evidence card from literature grounding (JSON):\n"
        f"{json.dumps(evidence_card, indent=2)}\n\n"
        f"Produce {plural} as a JSON array, matching the required schema exactly. "
        f"The array must contain exactly {count} object(s)."
    )


def generate_hypothesis(diagnosis: dict, evidence_card: dict) -> List[Dict]:
    """Produce 1 hypothesis by default, or up to 3 if diagnosis confidence
    is low or the diagnosis text suggests a plateau. See
    personas.HYPOTHESIS_SYSTEM_PROMPT for the full persona/task description.

    Raises:
        LLMSchemaError: if the model can't produce a schema-valid array of
            exactly the requested length within the retry budget.
    """
    count = _decide_count(diagnosis)
    prompt = _build_prompt(diagnosis, evidence_card, count)

    def call_fn(p: str) -> str:
        return call_model_text(HYPOTHESIS_SYSTEM_PROMPT, p, model=DEFAULT_MODEL)

    validator = partial(validate_hypothesis_list, expected_count=count)
    return call_with_schema_retry(call_fn, prompt, validator)
