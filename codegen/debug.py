"""
codegen.debug_and_retry — repair a failed candidate, and (separately) sanity-check
a candidate whose score looks too good to be true.

debug_and_retry(code_path, error_context) -> {"code_diff": str, "is_semantic_change": bool}

Two jobs:
  * REPAIR: call the code model with the current file + traceback/divergence
    context to produce the smallest fixing diff, retried up to 2 times.
  * SANITY (optional, triggered by passing observed_score): when a result scores
    implausibly well relative to the run history / oracle ceiling, ask the model
    whether the diff truly implements the stated hypothesis or is a bug/leak, and
    fold the verdict into the return (extra keys; the two contract keys are always
    present).

is_semantic_change is True when the repair changes model/loss/features/schedule,
not merely fixes a crash — the orchestrator counts those as new trials.
"""
from __future__ import annotations
import os, re, json

from . import prompts
from .constants import ORACLE_PRIMARY_CEILING
from .llm_client import LLMClient, get_default_client, KIND_DEBUG, KIND_SANITY

# tokens that mean the diff altered the *science*, not just fixed a crash
_SEMANTIC_TOKENS = re.compile(
    r"(loss|bpr|pairwise|listwise|softmax|sampler|negative|neg_sample|"
    r"\bFIELDS\b|feature|embed|\bk\s*=|lr\s*=|learning_rate|schedule|epochs?\b|"
    r"optimizer|adam|momentum|dropout|regulari|objective|margin|temperature)",
    re.IGNORECASE)
# tokens typical of a pure crash fix
_CRASHFIX_TOKENS = re.compile(
    r"(axis|shape|reshape|dtype|import|indent|typo|NameError|IndexError|"
    r"KeyError|TypeError|None check|off-by-one|parenthes|syntax)", re.IGNORECASE)


def _diff_added_removed(diff: str) -> str:
    return "\n".join(l[1:] for l in diff.splitlines()
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))


def is_semantic_change(code_diff: str) -> bool:
    """Classify a repair diff: True if it changes model/loss/features/schedule."""
    body = _diff_added_removed(code_diff)
    if _SEMANTIC_TOKENS.search(body):
        # a semantic token wins even if crash-fix tokens are also present
        return True
    return False


def _read(code_path: str) -> str:
    with open(code_path, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_diff(text: str) -> str:
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("\n") + "\n"
    return text.strip() + "\n"


def _looks_implausible(observed_score: float, history, threshold) -> bool:
    if observed_score is None:
        return False
    if observed_score > ORACLE_PRIMARY_CEILING:      # physically impossible
        return True
    if threshold is not None and observed_score > threshold:
        return True
    if history:
        try:
            best_prior = max(float(h) for h in history)
            if observed_score - best_prior > 0.02:   # a >0.02 leap in one edit is suspect
                return True
        except (TypeError, ValueError):
            pass
    return False


def debug_and_retry(code_path: str, error_context: str, *,
                    client: LLMClient | None = None, root: str = ".",
                    file_name: str | None = None,
                    observed_score: float | None = None,
                    history: list | None = None,
                    threshold: float | None = None,
                    hypothesis: dict | None = None,
                    max_retries: int = 2) -> dict:
    """Repair a failed candidate and/or sanity-check a suspicious result.

    The frozen 2-argument form debug_and_retry(code_path, error_context) works;
    the keyword args enable the sanity-check path and let the orchestrator inject
    a real client.

    Returns at least {"code_diff": str, "is_semantic_change": bool}. When the
    sanity path runs, also includes {"sanity": {...}, "leak_suspected": bool}.
    """
    client = client or get_default_client()
    fname = file_name or os.path.basename(code_path)
    content = _read(code_path)

    # ---- repair loop (up to max_retries) --------------------------------- #
    code_diff = ""
    last_err = error_context
    for _ in range(max(1, max_retries)):
        user = prompts.build_debug_user(fname, content, last_err)
        raw = client.complete(prompts.DEBUG_SYSTEM, user, kind=KIND_DEBUG,
                              max_tokens=2000, temperature=0.0)
        code_diff = _extract_diff(raw)
        if code_diff.strip():
            break
        last_err = error_context + "\n(previous attempt returned an empty diff)"

    result = {"code_diff": code_diff, "is_semantic_change": is_semantic_change(code_diff)}

    # ---- sanity-check path (only when a score was supplied) -------------- #
    if observed_score is not None and _looks_implausible(observed_score, history, threshold):
        suser = prompts.build_sanity_user(code_diff or "(no diff)",
                                          hypothesis or {}, observed_score,
                                          history, threshold or ORACLE_PRIMARY_CEILING)
        sraw = client.complete(prompts.SANITY_SYSTEM, suser, kind=KIND_SANITY,
                               max_tokens=800, temperature=0.0)
        try:
            verdict = json.loads(sraw)
        except Exception:
            verdict = {"implements_hypothesis": False, "leak_suspected": True,
                       "reasoning": "sanity model returned unparseable output; "
                                    "blocking as a precaution.",
                       "raw": sraw[:500]}
        result["sanity"] = verdict
        result["leak_suspected"] = bool(verdict.get("leak_suspected", True))

    return result
