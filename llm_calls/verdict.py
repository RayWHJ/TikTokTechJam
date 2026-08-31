"""verdict() — grade a measured candidate against the success criterion its own
hypothesis declared, before the node is closed.

Why this exists. Every hypothesis in the 5-iteration run recorded in
orchestrator/_state/ stated a `success_criterion_paired` between +0.004 and
+0.007, and no component ever compared it against the measured mean_delta. The
value was passed to the writer as context and nothing else. Nodes closed on
`evidence_type` alone, so three completely different situations collapsed into
one record:

  * the mechanism was fairly measured and is genuinely bad,
  * the mechanism was never given a working implementation,
  * the criterion was uncalibrated by 8x, and the mechanism actually moved the
    metric by as much as anything in this repo ever has.

They call for opposite responses — abandon it, rewrite it more cheaply, or build
on it — and the search could not tell them apart.

The pattern is deepagents' RubricMiddleware, reduced to one point in the loop:
the caller declares what done looks like up front, and when the agent would
otherwise finish, a separate grader judges the result against that rubric rather
than letting the agent grade itself.

Structure follows llm_calls/diagnose.py exactly — a persona constant in
personas.py, a _build_prompt here, a validator in schemas.py, routed through
retry.call_with_schema_retry.
"""
from __future__ import annotations

import json
from typing import Dict

from .client import DEFAULT_MODEL, call_model_text
from .personas import VERDICT_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_verdict


def _build_prompt(hypothesis: dict, measured: dict, context: dict) -> str:
    return (
        "The hypothesis, as it was declared BEFORE implementation (JSON):\n"
        f"{json.dumps(hypothesis, indent=2)}\n\n"
        "What was actually measured (JSON). `mean_delta`, `p_positive` and "
        "`lower_95` come from a paired per-user bootstrap of this candidate "
        "against its PARENT, not against the baseline:\n"
        f"{json.dumps(measured, indent=2, default=str)}\n\n"
        "Run context (JSON):\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Judge the measurement against the criterion the hypothesis declared, "
        "and respond with the required JSON."
    )


def verdict(hypothesis: dict, measured: dict, context: dict) -> Dict:
    """Grade `measured` against hypothesis['success_criterion_paired'].

    `measured` should carry mean_delta, p_positive, lower_95, per_seed_primary,
    parent_primary and evidence_type. `context` carries whatever else helps —
    baseline_primary, the iteration number, the component.

    Raises:
        LLMSchemaError: if the model can't produce schema-valid JSON within
            the retry budget. The driver catches this and continues with
            verdict=None; a failed grader must never lose a scored candidate.
    """
    prompt = _build_prompt(hypothesis, measured, context)

    def call_fn(p: str) -> str:
        return call_model_text(VERDICT_SYSTEM_PROMPT, p, model=DEFAULT_MODEL)

    return call_with_schema_retry(call_fn, prompt, validate_verdict)
