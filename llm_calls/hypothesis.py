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


def _build_prompt(diagnosis: dict, evidence_card: dict, count: int,
                  tried: List[Dict] | None = None) -> str:
    plural = "hypothesis" if count == 1 else f"exactly {count} hypotheses"
    tried_block = ""
    if tried:
        tried_block = (
            "ALREADY ATTEMPTED in this run — each entry is a mechanism that was "
            "proposed, the outcome of implementing it, and its measured "
            "candidate-minus-parent delta where one exists. Do NOT re-propose "
            "any mechanism in this list, however differently worded. Treat a "
            "`failed_implementation` outcome as evidence about the WRITER's "
            "budget, not about the idea: if you want that direction, propose a "
            "cheaper implementation of it, and say what makes it cheaper. "
            "Treat a negative `mean_delta_vs_parent` as evidence against the "
            "idea itself.\n"
            "READ `next_action` ON EACH ENTRY — it is a grader's judgement on "
            "that attempt, and it is binding:\n"
            "  - `retry_cheaper`: that mechanism is NOT refuted, it was never "
            "fairly implemented. A cheaper, smaller implementation of the SAME "
            "idea is welcome, and you must say what makes yours cheaper.\n"
            "  - `adjust_magnitude`: the mechanism works directionally but its "
            "strength is wrong. Re-propose it with a different weight or scale, "
            "not a different mechanism.\n"
            "  - `abandon_mechanism`: FORBIDDEN. Do not re-propose that "
            "mechanism or family in any wording.\n"
            "  - `build_on_it`: it worked. Extending it is a strong option.\n"
            "A `verdict` of `missed_but_promising` with "
            "`criterion_was_calibrated: false` means the attempt was judged "
            "against an unreachable bar. Its measured delta may be the best "
            "result this search has, so treat it as a success, not a failure.\n"
            f"{json.dumps(tried, indent=2)}\n\n"
        )
    return (
        "Diagnosis (JSON):\n"
        f"{json.dumps(diagnosis, indent=2)}\n\n"
        "Evidence card from literature grounding (JSON):\n"
        f"{json.dumps(evidence_card, indent=2)}\n\n"
        f"{tried_block}"
        f"Produce {plural} as a JSON array, matching the required schema exactly. "
        f"The array must contain exactly {count} object(s)."
    )


def generate_hypothesis(diagnosis: dict, evidence_card: dict,
                        tried: List[Dict] | None = None) -> List[Dict]:
    """Produce 1 hypothesis by default, or up to 3 if diagnosis confidence
    is low or the diagnosis text suggests a plateau. See
    personas.HYPOTHESIS_SYSTEM_PROMPT for the full persona/task description.

    `tried` is the attempt ledger from orchestrator.driver._attempt_ledger:
    prior mechanisms with their outcomes and measured deltas. It is optional so
    existing callers keep working, but omitting it is what produced 11
    consecutive restatements of "replace the pointwise loss with a pairwise
    one" — the operator had no way to know it was repeating itself.

    Raises:
        LLMSchemaError: if the model can't produce a schema-valid array of
            exactly the requested length within the retry budget.
    """
    count = _decide_count(diagnosis)
    prompt = _build_prompt(diagnosis, evidence_card, count, tried=tried)

    def call_fn(p: str) -> str:
        return call_model_text(HYPOTHESIS_SYSTEM_PROMPT, p, model=DEFAULT_MODEL)

    validator = partial(validate_hypothesis_list, expected_count=count)
    return call_with_schema_retry(call_fn, prompt, validator)
