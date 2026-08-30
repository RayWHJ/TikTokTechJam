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
from . import writer as _writer
from .diffnorm import normalize_unified_diff
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


def _diff_from_raw(raw: str, content: str, fname: str,
                   root: str) -> tuple[str, str]:
    """Turn one model reply into (diff, rejection_reason).

    Mirrors codegen.writer.write_fix's extract-then-diff pipeline, because
    DEBUG_SYSTEM inherits the same _DIFF_FORMAT_RULES: the model is asked for a
    COMPLETE ```python file, which we diff locally so the patch applies by
    construction. The ```diff branch stays as a fallback for a model that emits
    a patch anyway.
    """
    block = _writer._extract_python(raw)
    if block is not None and not block.lstrip().startswith(("--- ", "diff --git")):
        return _writer.rewrite_to_diff(block, content, fname)
    candidate = _extract_diff(raw)
    if candidate.strip() and not _writer.diff_applies(candidate, fname, root)[0]:
        repaired = normalize_unified_diff(candidate, content)
        if _writer.diff_applies(repaired, fname, root)[0]:
            return repaired, ""
    return candidate, ""


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
                    client: LLMClient | None = None, root: str | None = None,
                    file_name: str | None = None,
                    observed_score: float | None = None,
                    history: list | None = None,
                    threshold: float | None = None,
                    hypothesis: dict | None = None,
                    max_retries: int = 2) -> dict:
    """Repair a failed candidate and/or sanity-check a suspicious result.

    The frozen 2-argument form debug_and_retry(code_path, error_context) works;
    the keyword args enable the sanity-check path and let the orchestrator inject
    a real client. `root` is where the repair diff is validated; it defaults to
    code_path's own directory.

    Returns at least {"code_diff": str, "is_semantic_change": bool}. When the
    sanity path runs, also includes {"sanity": {...}, "leak_suspected": bool}.
    """
    client = client or get_default_client()
    fname = file_name or os.path.basename(code_path)
    content = _read(code_path)
    # The diff must apply to the file being REPAIRED, so validation defaults to
    # that file's own directory. A literal "." default validated a staged
    # candidate's repair against the pristine repo copy instead — silently the
    # wrong file whenever code_path lives outside the repo root.
    root = root if root is not None else (os.path.dirname(code_path) or ".")

    # ---- repair loop (up to max_retries) --------------------------------- #
    # A repair diff is only worth returning if it APPLIES — the driver applies it
    # to the staged dir before rerunning, so an unapplyable diff means the rerun
    # would re-execute the same broken code. Each retry carries the concrete
    # rejection reason, which is what makes it worth spending at temperature 0.
    code_diff = ""
    user = prompts.build_debug_user(fname, content, error_context)
    err = ""
    for attempt in range(1, max(1, max_retries) + 1):
        msg = user if attempt == 1 else \
            user + prompts.build_diff_repair_suffix(fname, err or "empty diff")
        raw = client.complete(prompts.DEBUG_SYSTEM, msg, kind=KIND_DEBUG,
                              max_tokens=16000, temperature=0.0)
        diff, err = _diff_from_raw(raw, content, fname, root)
        if diff.strip():
            ok, apply_err = _writer.diff_applies(diff, fname, root)
            if ok:
                code_diff = diff
                break
            err = err or apply_err

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
