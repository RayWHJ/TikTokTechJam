"""Real token accounting, keyed by call kind.

WHY THIS EXISTS. `counters.tokens` was hand-incremented constants — 500 for a
diagnose, 300 per hypothesis, 800 for a writer call, 400 for an audit — and the
persisted total in every `progress.json` is `13200`, a number no API ever
reported. `counters.wallclock_s` is `0.0` in every persisted file because it is
assigned after the loop that writes the file. Those two numbers are exactly what
Feasibility & Practicality is scored on and exactly what Section 2.5 requires the
run log to report.

They are also the only instrument that can say whether a model swap paid for
itself. Guessing costs while reporting fabricated figures on the criterion that
grades cost is the worst of both.

DESIGN. A process-wide ledger rather than threading usage through seven return
values. The five `llm_calls` contract functions return validated dicts and their
schemas are pinned by tests; `codegen`'s writer returns a diff string. Adding a
usage field to each would change every signature, every schema and every mock.
The client layer is the only place that sees a raw API response, so that is where
the recording happens, and `kind` is passed down from the call site so the totals
break down per operator.

This module imports nothing — not even openai — so both packages can depend on it
without either depending on the other.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

#: Call kinds. The point of naming them is that the totals become actionable:
#: the recorded run's spend was dominated by the writer (a whole reproduced file)
#: and the auditor (~4,500 input tokens per candidate for a signal that fired on
#: 5 of 5 candidates, including the training loop's own `y`). A single scalar
#: cannot show either.
KIND_DIAGNOSE = "diagnose"
KIND_HYPOTHESIS = "hypothesis"
KIND_LITERATURE = "literature"
KIND_AUDIT = "audit"
KIND_VERDICT = "verdict"
KIND_REFINE = "refine"
KIND_WRITER = "writer"
KIND_DEBUG = "debug"
KIND_SANITY = "sanity"
KIND_REPORT = "report"

ALL_KINDS = (KIND_DIAGNOSE, KIND_HYPOTHESIS, KIND_LITERATURE, KIND_AUDIT,
             KIND_VERDICT, KIND_REFINE, KIND_WRITER, KIND_DEBUG,
             KIND_SANITY, KIND_REPORT)

#: USD per 1M tokens, verified against OpenAI's pricing page on 2026-08-31.
#: Sol's input/output are PROMOTIONAL ($4/$20, published as running at least
#: through 2026-11-21); standard rates were $5/$30. Cached input is an order of
#: magnitude cheaper than fresh input, which is why it is tracked separately —
#: this repo re-sends the same ~3.5k-token personas every call, so cache hits are
#: most of the input bill.
#:
#: An ESTIMATE, and labelled as one wherever it is reported. The authoritative
#: number is the OpenAI dashboard; this exists so a run log can say roughly what
#: it cost without one.
PRICES_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    # The shipped configuration. `gpt-5.6-sol` is the writer and the strong tier;
    # `gpt-5.6-luna` is the cheap tier. See llm_calls/routing.py::TABLE.
    "gpt-5.6-sol":   {"input": 4.00, "cached": 0.40, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gpt-5.6-luna":  {"input": 0.20, "cached": 0.02, "output": 1.20},

    # LEGACY, not defaults any more. Kept only so that a run which overrides a
    # model back to one of these still reports a real cost instead of $0.00 —
    # `Usage.cost_usd` yields no estimate for a model it has never heard of, and
    # a silent zero on the Feasibility criterion is worse than a stale number.
    # `gpt-4o` is what the recorded 4-iteration run actually used.
    "gpt-4o":        {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-4.1-nano":  {"input": 0.10, "cached": 0.025, "output": 0.40},
}

#: Web search is billed per call, not per token, on top of the model rates.
WEB_SEARCH_USD_PER_CALL = 10.00 / 1000


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    web_searches: int = 0
    #: Which model ids served this kind. A list, not a scalar, because a run that
    #: changes routing mid-flight must not silently report one of them.
    models: Dict[str, int] = field(default_factory=dict)

    def add(self, *, model: Optional[str] = None, tokens_in: int = 0,
            tokens_cached: int = 0, tokens_out: int = 0,
            tokens_reasoning: int = 0, web_search: bool = False) -> None:
        self.calls += 1
        self.tokens_in += tokens_in or 0
        self.tokens_cached += tokens_cached or 0
        self.tokens_out += tokens_out or 0
        self.tokens_reasoning += tokens_reasoning or 0
        if web_search:
            self.web_searches += 1
        if model:
            self.models[model] = self.models.get(model, 0) + 1

    def cost_usd(self) -> float:
        """Estimated spend for this kind, summed over the models it used.

        Attributes tokens to a model in proportion to its share of this kind's
        calls, because the ledger records per-kind totals rather than per-call
        rows. Exact when a kind used one model, which is the normal case.
        """
        total_calls = sum(self.models.values()) or self.calls
        if not total_calls:
            return 0.0
        cost = 0.0
        for model, n in (self.models or {}).items():
            share = n / total_calls
            p = PRICES_USD_PER_MTOK.get(model)
            if p is None:
                continue                    # unknown model: no estimate, not a guess
            fresh_in = max(self.tokens_in - self.tokens_cached, 0) * share
            cost += fresh_in * p["input"] / 1e6
            cost += self.tokens_cached * share * p["cached"] / 1e6
            # Reasoning tokens are billed at the OUTPUT rate and are already
            # included in output_tokens by the API, so they are not added again.
            cost += self.tokens_out * share * p["output"] / 1e6
        cost += self.web_searches * WEB_SEARCH_USD_PER_CALL
        return cost


class UsageLedger:
    """Per-kind usage for one run. Thread-safe because it is a process global."""

    def __init__(self):
        self._lock = threading.Lock()
        self.by_kind: Dict[str, Usage] = {}
        #: Calls that reported no usage block at all. Non-zero means the totals
        #: are an undercount, and a report that cannot say so is not honest.
        self.calls_without_usage = 0

    def reset(self) -> None:
        with self._lock:
            self.by_kind = {}
            self.calls_without_usage = 0

    def record_response(self, kind: str, resp, *, model: str | None = None,
                        web_search: bool = False) -> None:
        """Pull usage off a Responses API response object.

        Everything via getattr: the test suite stubs these responses, older SDK
        versions shape them differently, and accounting must never be the thing
        that crashes a 6-hour run. A response with no usage block increments
        `calls_without_usage` instead of silently contributing zero.
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            with self._lock:
                self.calls_without_usage += 1
                self.by_kind.setdefault(kind, Usage()).add(
                    model=model, web_search=web_search)
            return
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        self.record(
            kind,
            model=model,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_cached=getattr(in_details, "cached_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            tokens_reasoning=getattr(out_details, "reasoning_tokens", 0) or 0,
            web_search=web_search,
        )

    def record(self, kind: str, **kw) -> None:
        with self._lock:
            self.by_kind.setdefault(kind, Usage()).add(**kw)

    def totals(self) -> dict:
        """Run-wide totals plus the per-kind breakdown, JSON-ready."""
        with self._lock:
            kinds = {k: asdict(v) for k, v in self.by_kind.items()}
            costs = {k: round(v.cost_usd(), 4) for k, v in self.by_kind.items()}
            agg = {
                "calls": sum(v.calls for v in self.by_kind.values()),
                "tokens_in": sum(v.tokens_in for v in self.by_kind.values()),
                "tokens_cached": sum(v.tokens_cached
                                     for v in self.by_kind.values()),
                "tokens_out": sum(v.tokens_out for v in self.by_kind.values()),
                "tokens_reasoning": sum(v.tokens_reasoning
                                        for v in self.by_kind.values()),
                "web_searches": sum(v.web_searches
                                    for v in self.by_kind.values()),
                "calls_without_usage": self.calls_without_usage,
            }
        agg["tokens_total"] = agg["tokens_in"] + agg["tokens_out"]
        agg["estimated_cost_usd"] = round(sum(costs.values()), 4)
        for k in kinds:
            kinds[k]["estimated_cost_usd"] = costs[k]
        agg["by_kind"] = kinds
        return agg


#: Process-wide ledger. Reset at the start of each run.
LEDGER = UsageLedger()
