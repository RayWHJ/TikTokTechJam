"""Per-persona model routing: which model, at what reasoning effort.

WHY A TABLE AND NOT A CONSTANT. `DEFAULT_MODEL` was one shared value read at
import time, so the auditor and the diagnostician were forced onto the same tier.
They are not the same job. The diagnostician chooses WHICH ideas get tried and is
the highest-leverage call in the loop; the auditor is the second-largest input
consumer in the run at ~4,500 tokens per candidate and flagged 5 of 5 candidates
including "y is being used within the step function", which is the training loop.
Paying Sol rates for that is paying for noise.

Reasoning effort is the main cost dial on a reasoning model — effort scales
output tokens, and output is 5x input on Sol ($20 vs $4 per Mtok). Without the
dial the choice is all-or-nothing on price.

ENV OVERRIDES. `LLM_CALLS_MODEL` and `LLM_CALLS_CHEAP_MODEL` still work and still
mean what they meant: they replace the model for the personas whose tier they
name, so the existing deployment story is unchanged. Per-persona overrides use
`LLM_ROUTE_<PERSONA>` / `LLM_EFFORT_<PERSONA>` for when one operator needs to
move on its own.

Everything resolves at CALL time, never at import. `DEFAULT_MODEL` being an
import-time constant is why a typo in an env var used to fail quietly.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from .usage import (KIND_AUDIT, KIND_DIAGNOSE, KIND_HYPOTHESIS,
                    KIND_LITERATURE, KIND_REFINE, KIND_VERDICT, KIND_WRITER,
                    KIND_DEBUG, KIND_REPORT, KIND_SANITY)

# KIND_REFINE is imported and intentionally absent from TABLE below: refine
# is behind REFINE_ENABLED=False (T3.5 kept the subsystem, documented), so it
# has no route. Naming it here keeps the omission deliberate rather than
# looking like an oversight, and _FALLBACK covers it if the flag flips.

#: Reasoning-effort values the gpt-5.6 family accepts. Verified against OpenAI's
#: model docs on 2026-08-31; `medium` is the API default when omitted.
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

#: Which tier a persona belongs to. `strong` follows LLM_CALLS_MODEL, `cheap`
#: follows LLM_CALLS_CHEAP_MODEL — so the two env vars that already exist keep
#: working and keep meaning the same thing.
STRONG, CHEAP = "strong", "cheap"

#: Default tier for anything not named below. Strong, because the failure mode of
#: guessing cheap is a wasted iteration and the failure mode of guessing strong is
#: a few cents.
_FALLBACK = (STRONG, "medium")

#: THE ROUTING TABLE. (tier, effort) per persona.
#:
#: The rationale for each choice, since the numbers are what justify it:
#:   writer/debug  the multi-site coupled edit, reproduced as a whole file. 60%
#:                 of candidates failed to execute on the old model. Highest
#:                 effort that is still affordable per call.
#:   diagnose      the only lever that changes WHICH ideas get tried.
#:   hypothesis    same, one step downstream.
#:   verdict       bounded grading against a stated criterion, with the
#:                 calibration facts handed to it in the prompt. Cheap tier.
#:   audit         advisory only, demonstrated signal-to-noise of zero. Cheap
#:                 tier, and a candidate for deletion (T3.4).
#:   literature    one call per iteration, and the web_search tool is billed per
#:                 call ($10/1k) rather than by thinking. Low effort.
#:   report        one call at the end of the run.
#:   sanity        one call per implausibly-good result; never currently reached.
TABLE: Dict[str, tuple] = {
    KIND_WRITER:     (STRONG, "high"),
    KIND_DEBUG:      (STRONG, "high"),
    KIND_DIAGNOSE:   (STRONG, "medium"),
    KIND_HYPOTHESIS: (STRONG, "medium"),
    KIND_LITERATURE: (STRONG, "low"),
    KIND_VERDICT:    (CHEAP,  "low"),
    KIND_AUDIT:      (CHEAP,  "low"),
    KIND_REPORT:     (CHEAP,  "low"),
    KIND_SANITY:     (CHEAP,  "low"),
}


#: Personas that keep their OWN long-standing env var, which takes precedence
#: over the tier. The writer's model has been configured with
#: `CODEGEN_LLM_MODEL` since before this table existed, it is documented in
#: FIX_PLAN.md, and it is what the deployment `.env` sets — so centralising
#: routing must not silently move the writer onto `LLM_CALLS_MODEL`. The table
#: still owns the writer's TIER and EFFORT; this only preserves the knob.
#: Only the two code-EDITING operators. `report` and `sanity` also run through
#: codegen's client, but CODEGEN_LLM_MODEL has always meant "the model that
#: writes code" — letting it also decide who writes the Devpost markdown would
#: make one knob control two unrelated things, which is the coupling this table
#: exists to remove. They follow their cheap tier instead.
_PERSONA_ENV = {KIND_WRITER: "CODEGEN_LLM_MODEL",
                KIND_DEBUG: "CODEGEN_LLM_MODEL"}


def _tier_model(tier: str) -> str:
    """The model id for a tier, read from the environment at call time."""
    from .client import DEFAULT_CHEAP_MODEL, DEFAULT_MODEL
    if tier == CHEAP:
        return os.environ.get("LLM_CALLS_CHEAP_MODEL") or DEFAULT_CHEAP_MODEL
    return os.environ.get("LLM_CALLS_MODEL") or DEFAULT_MODEL


def _env_key(persona: str) -> str:
    return persona.upper().replace("-", "_")


def model_for(persona: str) -> str:
    """Model id for one persona.

    Precedence, most specific first:
      1. `LLM_ROUTE_<PERSONA>` — this one operator, explicitly.
      2. The persona's own historical env var, if it has one (_PERSONA_ENV).
      3. The tier's env var (`LLM_CALLS_MODEL` / `LLM_CALLS_CHEAP_MODEL`).
      4. The code default in llm_calls/client.py.
    """
    override = os.environ.get(f"LLM_ROUTE_{_env_key(persona)}")
    if override:
        return override
    own = _PERSONA_ENV.get(persona)
    if own and os.environ.get(own):
        return os.environ[own]
    tier, _effort = TABLE.get(persona, _FALLBACK)
    return _tier_model(tier)


def effort_for(persona: str) -> Optional[str]:
    """Reasoning effort for one persona, or None to let the API default apply.

    `LLM_EFFORT_<PERSONA>` overrides. An unrecognised value is ignored rather
    than sent — a rejected parameter fails the whole call, and a typo in an env
    var must not take the run down.
    """
    override = os.environ.get(f"LLM_EFFORT_{_env_key(persona)}")
    if override:
        return override if override in EFFORTS else None
    _tier, effort = TABLE.get(persona, _FALLBACK)
    return effort if effort in EFFORTS else None


def resolved_table() -> Dict[str, dict]:
    """The whole table as resolved right now, for the run log.

    Recorded in progress.json so the Feasibility numbers are interpretable: a
    cost total is meaningless without knowing which model produced it, and a
    reader cannot reconstruct routing that lived only in env vars.
    """
    return {p: {"tier": TABLE.get(p, _FALLBACK)[0],
                "model": model_for(p),
                "effort": effort_for(p),
                "env": _PERSONA_ENV.get(p) or (
                    "LLM_CALLS_CHEAP_MODEL" if TABLE[p][0] == CHEAP
                    else "LLM_CALLS_MODEL")}
            for p in TABLE}
