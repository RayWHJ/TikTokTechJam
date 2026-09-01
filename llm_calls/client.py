"""
Thin wrapper around the OpenAI API. Kept separate from the 5 contract
functions so the retry/schema logic never has to know about SDK details,
and so this is the only file you need to touch if the team switches
providers or model names later.

Uses OpenAI's Responses API for both plain text and web-search-enabled
calls — it's the unified, current-generation API and supports the
web_search tool cleanly.

Configure via environment variables (all optional; the defaults below match the
values the deployment `.env` sets, so a run made without `.env` behaves the same
as one made with it):
    LLM_CALLS_API_KEY        overrides OPENAI_API_KEY if you want a
                              separate key/budget for this module
    LLM_CALLS_MODEL          the STRONG tier: diagnose / hypothesis /
                              literature grounding  (default: gpt-5.6-sol)
    LLM_CALLS_CHEAP_MODEL    the CHEAP tier: verdict / audit / dedup
                              (default: gpt-5.6-luna)

Which persona lands on which tier, and at what reasoning effort, is
llm_calls/routing.py — not these two constants. They only name the models the
tiers resolve to.

Note on temperature: intentionally omitted from the Responses API calls.
Reasoning models (the gpt-5.x families, o-series) reject the parameter
outright, and for a non-reasoning model the small quality gain from tuning it
isn't worth the portability cost. If a downstream caller needs sampling
variance, add it back guarded by a model-family check. Reasoning EFFORT is the
dial that replaces it here — see `effort` on the two call functions below.
"""

from __future__ import annotations

import os
from typing import Optional

import openai

from .exceptions import LLMTruncatedError
from .usage import LEDGER

#: The STRONG tier: diagnose, generate_hypothesis, ground_in_literature. These
#: are the calls that decide WHICH ideas get tried, which is the highest-leverage
#: point in the loop — 61 of 65 stored proposals from earlier runs named a
#: mechanism that cannot be written in numpy at all, a ~94% unimplementable rate
#: WITH the constraint block already in the persona. That is an
#: instruction-following failure under a long constraint list, which is the thing
#: a stronger reasoning model actually fixes.
DEFAULT_MODEL = os.environ.get("LLM_CALLS_MODEL", "gpt-5.6-sol")

#: The CHEAP tier: verdict, audit, dedup. Bounded jobs with the calibration facts
#: handed to them in the prompt. The auditor in particular is the second-largest
#: input consumer in a run at ~4,500 tokens per candidate, and it flagged 5 of 5
#: candidates including "y is being used within the step function" — which is the
#: training loop. Paying strong-tier rates for that is paying for noise.
#:
#: Luna is ~20x cheaper than Sol on output ($1.20 vs $20.00 per Mtok).
DEFAULT_CHEAP_MODEL = os.environ.get("LLM_CALLS_CHEAP_MODEL", "gpt-5.6-luna")

#: Output-token ceiling for every call in this module.
#:
#: Was 2000 for text and 3000 for search, and NO call site overrode either —
#: diagnose, hypothesis, audit and verdict all took the 2000 default, literature
#: took 3000. On a REASONING model that is a trap rather than a saving: reasoning
#: tokens are billed against `max_output_tokens`, so the model can spend the whole
#: budget thinking and return an empty `output_text` with `status="incomplete"`.
#: See LLMTruncatedError for why the resulting failure looks like a bad model
#: instead of a bad constant.
#:
#: 16000 matches what codegen/writer.py and codegen/debug.py already pass, and is
#: well inside the 128,000-completion-token limit of the gpt-5.6 family. It is a
#: CEILING, not a reservation — a call that needs 400 tokens still bills 400.
MAX_OUTPUT_TOKENS = 16000

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


def _text_or_raise(resp, *, model: str, max_tokens: int, kind: str) -> str:
    """Return `resp.output_text`, or raise LLMTruncatedError explaining why not.

    Two distinct failures, both of which previously surfaced as an empty string:

      * `status` is not "completed" — the API says outright that it stopped
        early, and `incomplete_details.reason` says why ("max_output_tokens",
        "content_filter", ...).
      * `status` looks fine but `output_text` is empty. On a reasoning model this
        is the same disease with a quieter symptom: the budget went to reasoning
        tokens and nothing was left to emit.

    Everything is read with getattr, because these fields are absent on the
    stubbed response objects the tests use and on older SDK versions — a status
    check that itself raises AttributeError would be worse than no check.
    """
    status = getattr(resp, "status", None)
    text = getattr(resp, "output_text", None) or ""

    if status is not None and status != "completed":
        details = getattr(resp, "incomplete_details", None)
        reason = getattr(details, "reason", None) or str(details or "unspecified")
        usage = getattr(resp, "usage", None)
        spent = getattr(usage, "output_tokens", None)
        raise LLMTruncatedError(
            f"{kind} call to {model!r} returned status={status!r} "
            f"(reason={reason!r}) with max_output_tokens={max_tokens}"
            + (f"; {spent} output tokens were billed" if spent else "")
            + ". Reasoning tokens count against max_output_tokens, so raise "
              "MAX_OUTPUT_TOKENS or lower the reasoning effort for this "
              "persona. This is a configuration failure, not a schema failure.",
            reason=reason, model=model, max_tokens=max_tokens, status=status)

    if not text.strip():
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        raise LLMTruncatedError(
            f"{kind} call to {model!r} returned status={status!r} but an EMPTY "
            f"output_text with max_output_tokens={max_tokens}"
            + (f"; {reasoning} of the output budget went to reasoning tokens"
               if reasoning else "")
            + ". Retrying cannot help — the retry would spend the same budget "
              "the same way. Raise MAX_OUTPUT_TOKENS or lower this persona's "
              "reasoning effort.",
            reason="empty_output_text", model=model, max_tokens=max_tokens,
            status=status)
    return text


def _reasoning_kwargs(effort: str | None) -> dict:
    """`reasoning={"effort": ...}` if an effort was chosen, else nothing.

    Omitted rather than defaulted, so a non-reasoning model — which rejects the
    parameter outright — still works. Same defensive reasoning as the existing
    decision not to forward `temperature`.
    """
    return {"reasoning": {"effort": effort}} if effort else {}


def call_model_text(system: str, user: str, model: str = DEFAULT_MODEL,
                    max_tokens: int = MAX_OUTPUT_TOKENS,
                    kind: str = "text",
                    effort: str | None = None,
                    text_format: dict | None = None) -> str:
    """Plain text-in/text-out call via the Responses API, no tools.
    Returns the model's response as a single string.

    `kind` names the OPERATOR for token accounting (see llm_calls/usage.py). It
    defaults so no existing caller breaks, but every real call site passes its
    own — a single scalar token total cannot show that the writer and the auditor
    dominate the bill, which is the fact that decides where to spend.

    `effort` is the reasoning-effort dial (see llm_calls/routing.py). Effort
    scales output tokens and output costs 5x input on Sol, so this is the main
    cost control; None lets the API's own default apply.
    """
    client = get_client()
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
        **_reasoning_kwargs(effort),
        # Server-side schema enforcement, when the caller supplies one. Note the
        # Responses API spells this `text={"format": ...}`, NOT `response_format`
        # (which is the Chat Completions spelling).
        **({"text": {"format": text_format}} if text_format else {}),
    )
    # BEFORE the truncation check: a truncated call still burned tokens, and a
    # ledger that only counts successes understates exactly the failure mode
    # that costs the most.
    LEDGER.record_response(kind, resp, model=model)
    return _text_or_raise(resp, model=model, max_tokens=max_tokens, kind=kind)


def call_model_with_search(system: str, user: str, model: str = DEFAULT_MODEL,
                            max_tokens: int = MAX_OUTPUT_TOKENS,
                            kind: str = "web_search",
                            effort: str | None = None) -> str:
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
        **_reasoning_kwargs(effort),
    )
    # web_search=True: the tool is billed per CALL ($10/1k) on top of the model
    # rates, so a token-only ledger misses it entirely.
    LEDGER.record_response(kind, resp, model=model, web_search=True)
    return _text_or_raise(resp, model=model, max_tokens=max_tokens, kind=kind)