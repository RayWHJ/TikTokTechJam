from __future__ import annotations

from typing import List, Tuple

from .client import DEFAULT_CHEAP_MODEL, call_model_text
from .personas import DEDUP_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_dedup

# A near-duplicate is a fingerprint of the same length differing in at most
# this many positions. Tune this if it's over/under-escalating in practice.
_AMBIGUOUS_MAX_DIFF_POSITIONS = 1


def _differing_positions(a: Tuple, b: Tuple) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))  # different shapes are never near-duplicates
    return sum(1 for x, y in zip(a, b) if x != y)


def _find_closest_ambiguous(candidate: Tuple, memory_entries: List[Tuple]) -> Tuple:
    """Returns the closest near-duplicate entry, or None if none are
    ambiguous (either exact match, already handled by the caller, or
    clearly distinct)."""
    best = None
    best_diff = None
    for entry in memory_entries:
        entry_t = tuple(entry)
        diff = _differing_positions(candidate, entry_t)
        if 0 < diff <= _AMBIGUOUS_MAX_DIFF_POSITIONS:
            if best_diff is None or diff < best_diff:
                best, best_diff = entry_t, diff
    return best


def _build_prompt(candidate: Tuple, closest: Tuple) -> str:
    return (
        f"Fingerprint A (candidate): {candidate}\n"
        f"Fingerprint B (existing memory entry): {closest}\n\n"
        "Are these the same underlying experimental idea (duplicate) or "
        "meaningfully different? Respond with the required JSON."
    )


def dedup_fingerprint_match(candidate_fingerprint: tuple, memory_entries: list) -> bool:
    """Check whether candidate_fingerprint duplicates anything already in
    memory_entries. Mostly deterministic: exact tuple matches return True
    immediately, clearly-distinct fingerprints return False immediately, and
    only genuinely ambiguous near-duplicates (differ in a small number of
    positions) get escalated to a cheap model call.

    Raises:
        LLMSchemaError: if escalation is needed and the model can't produce
            schema-valid JSON within the retry budget.
    """
    candidate = tuple(candidate_fingerprint)
    normalized_entries = [tuple(e) for e in memory_entries]

    if candidate in normalized_entries:
        return True

    closest_ambiguous = _find_closest_ambiguous(candidate, normalized_entries)
    if closest_ambiguous is None:
        return False

    prompt = _build_prompt(candidate, closest_ambiguous)

    def call_fn(p: str) -> str:
        return call_model_text(DEDUP_SYSTEM_PROMPT, p, model=DEFAULT_CHEAP_MODEL, max_tokens=300)

    result = call_with_schema_retry(call_fn, prompt, validate_dedup)
    return result["duplicate"]
