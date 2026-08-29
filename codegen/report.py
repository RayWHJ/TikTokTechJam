"""
codegen.synthesize_report — turn a structured run log into a Devpost-style
markdown project description via the writing model.
"""
from __future__ import annotations
from . import prompts
from .llm_client import LLMClient, get_default_client, KIND_REPORT


def synthesize_report(run_log: dict, *, client: LLMClient | None = None) -> str:
    """Return a markdown Devpost write-up built from `run_log`.

    `run_log` is a free-form dict; useful keys the writer looks for include
    baseline_primary, global_best {primary, mechanism, split}, counters
    {proposals, partial_runs, full_runs, semantic_retries, scorer_queries,
    wallclock_s, tokens}, iterations (list), and evidence (list). Missing keys
    are tolerated. Uses the injected/default client (offline fake backend unless a
    real backend is configured), so it runs end-to-end with no API key.
    """
    client = client or get_default_client()
    user = prompts.build_report_user(run_log)
    return client.complete(prompts.REPORT_SYSTEM, user, kind=KIND_REPORT,
                          max_tokens=3000, temperature=0.3)
