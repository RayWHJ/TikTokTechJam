from __future__ import annotations

import json
from functools import partial
from typing import Dict, List

from .client import call_model_text
from .families import LEGAL_DECLARATIONS, OTHER
from .routing import effort_for, model_for
from .usage import KIND_HYPOTHESIS
from .personas import HYPOTHESIS_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import hypothesis_json_schema, validate_hypothesis_list

# Tunable thresholds for deciding how many hypotheses to request.
LOW_CONFIDENCE_THRESHOLD = 0.5
_PLATEAU_KEYWORDS = ("plateau", "stagnant", "stagnated", "no improvement", "flat")

#: How many hypotheses to ask for. ALWAYS a batch, never one.
#:
#: This used to return 1 unless confidence < 0.5, so the search was a CHAIN
#: rather than a tree — and with a batch of one, the probability of starving an
#: iteration equals the probability that one family is banned. That is exactly
#: how iteration 4 of the recorded run produced zero candidates.
#:
#: Widening the PROPOSAL distribution is nearly free relative to widening the
#: compute bill. A hypothesis is ~300 tokens of output; a candidate is a writer
#: call (~5.5k in / 4.3k out), an audit call (~4.5k in) and a triage run. Two
#: orders of magnitude. So propose 6-8, filter deterministically, and cap what
#: actually executes (driver.MAX_CANDIDATES_PER_ITER).
HYPOTHESES_MIN = 6
HYPOTHESES_MAX = 8


def _suggests_plateau(diagnosis: dict) -> bool:
    text = f"{diagnosis.get('bottleneck', '')} {diagnosis.get('evidence', '')}".lower()
    return any(kw in text for kw in _PLATEAU_KEYWORDS)


def _decide_count(diagnosis: dict, legal_families: List[str] | None = None
                  ) -> int:
    """How many hypotheses to request: HYPOTHESES_MAX when the diagnosis is
    uncertain or the run is flat, HYPOTHESES_MIN otherwise.

    Clamped by how many DISTINCT families are actually available, because the
    batch is required to span distinct families and asking for more than exist is
    an instruction the model cannot satisfy — it would burn the whole retry
    budget failing the diversity check.
    """
    confidence = diagnosis.get("confidence", 1.0)
    want = (HYPOTHESES_MAX
            if confidence < LOW_CONFIDENCE_THRESHOLD or _suggests_plateau(diagnosis)
            else HYPOTHESES_MIN)
    if legal_families is not None:
        # One slot per NAMED legal family, plus one for `other` — which is free
        # text and may repeat, but allowing it to fill the whole batch would
        # defeat the diversity requirement it is exempt from.
        named = [f for f in legal_families if f != OTHER]
        want = min(want, len(named) + 1)
    return max(want, 1)


def _build_prompt(diagnosis: dict, evidence_card: dict, count: int,
                  tried: List[Dict] | None = None,
                  blocked_families: List[str] | None = None) -> str:
    plural = "hypothesis" if count == 1 else f"exactly {count} hypotheses"

    # The constraint, enumerated. The schema rejects a blocked family and the
    # retry loop re-asks, but a model told the rule up front does not need the
    # retry — and in the recorded run the proposer went 3 for 3 into banned
    # families precisely because nothing in its context named them.
    legal = [f for f in LEGAL_DECLARATIONS
             if f not in set(blocked_families or ())]
    family_block = (
        "MECHANISM FAMILY. Every hypothesis must declare `mechanism_family`, "
        f"chosen from: {legal}. Use {OTHER!r} for a mechanism that genuinely "
        f"fits none of them — that is always legal and is never blocked.\n")
    if blocked_families:
        family_block += (
            f"These families have been REFUTED by measurement in this run and "
            f"are BLOCKED: {sorted(blocked_families)}. A hypothesis declaring "
            f"one will be rejected. Do not propose into them under a different "
            f"wording either — the family is what is blocked, not the phrasing.\n")
    if count > 1:
        family_block += (
            f"The {count} hypotheses must declare {count} DIFFERENT families "
            f"(except {OTHER!r}, which may repeat). A batch that is one family "
            f"wide is one experiment run several times.\n")
    family_block += "\n"

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
        f"{family_block}"
        f"Produce {plural}, matching the required schema exactly: a single "
        f"object with a \"hypotheses\" array holding exactly {count} object(s)."
    )


def generate_hypothesis(diagnosis: dict, evidence_card: dict,
                        tried: List[Dict] | None = None,
                        blocked_families: List[str] | None = None
                        ) -> List[Dict]:
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
    # The legal set goes into the schema ENUM, so the model is
    # structurally unable to declare a refuted family — rather than
    # being told not to and doing it anyway, which is what happened 3
    # for 3 in iteration 4 of the recorded run.
    legal = [f for f in LEGAL_DECLARATIONS
             if f not in set(blocked_families or ())]
    count = _decide_count(diagnosis, legal_families=legal)
    prompt = _build_prompt(diagnosis, evidence_card, count, tried=tried,
                           blocked_families=blocked_families)

    def call_fn(p: str) -> str:
        return call_model_text(HYPOTHESIS_SYSTEM_PROMPT, p,
                               model=model_for(KIND_HYPOTHESIS),
                               effort=effort_for(KIND_HYPOTHESIS),
                               text_format=hypothesis_json_schema(legal),
                               kind=KIND_HYPOTHESIS)

    # `blocked_families` is enforced at the SCHEMA layer, so a proposal into a
    # refuted family bounces back to the model with the reason and the legal set
    # appended, instead of being silently deleted by the driver's dedup filter.
    # Batch diversity is required only when there is a batch to diversify.
    validator = partial(validate_hypothesis_list, expected_count=count,
                        blocked_families=blocked_families,
                        require_distinct_families=count > 1)
    return call_with_schema_retry(call_fn, prompt, validator)
