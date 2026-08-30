"""
Hand-rolled schema validation — deliberately no jsonschema dependency, so
this module has zero install requirements beyond the anthropic SDK itself.

Every validate_* function takes a parsed dict/list (already JSON-decoded)
and either returns a cleaned copy (numeric coercion, key-stripping) or
raises ValueError with a message specific enough to be useful when it gets
appended back into a retry prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _require_dict(obj: Any) -> Dict:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object, got {type(obj).__name__}.")
    return obj


def _require_keys(d: Dict, keys: List[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required key(s): {missing}. Present keys: {list(d.keys())}.")


def _reject_unknown_keys(d: Dict, allowed: List[str]) -> None:
    extra = [k for k in d.keys() if k not in allowed]
    if extra:
        raise ValueError(f"Unexpected extra key(s): {extra}. Allowed keys: {allowed}.")


def _check_str(d: Dict, key: str) -> str:
    val = d[key]
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Field '{key}' must be a non-empty string, got {val!r}.")
    return val


def _check_float01(d: Dict, key: str) -> float:
    val = d[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Field '{key}' must be a number between 0 and 1, got {val!r}.")
    val = float(val)
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"Field '{key}' must be between 0 and 1 inclusive, got {val}.")
    return val


def _check_bool(d: Dict, key: str) -> bool:
    val = d[key]
    if not isinstance(val, bool):
        raise ValueError(f"Field '{key}' must be a boolean, got {val!r}.")
    return val


def _check_list_of_str(d: Dict, key: str) -> List[str]:
    val = d[key]
    if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
        raise ValueError(f"Field '{key}' must be a list of strings, got {val!r}.")
    return val


def _check_enum(d: Dict, key: str, allowed: List[str]) -> str:
    val = d[key]
    if val not in allowed:
        raise ValueError(f"Field '{key}' must be one of {allowed}, got {val!r}.")
    return val


# ---------------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------------

_DIAGNOSIS_KEYS = [
    "bottleneck", "evidence", "confidence", "component",
    "edit_radius", "expected_cost", "incompatibilities", "uncertainty",
]


def validate_diagnosis(parsed: Any) -> Dict:
    d = _require_dict(parsed)
    _require_keys(d, _DIAGNOSIS_KEYS)
    _reject_unknown_keys(d, _DIAGNOSIS_KEYS)
    return {
        "bottleneck": _check_str(d, "bottleneck"),
        "evidence": _check_str(d, "evidence"),
        "confidence": _check_float01(d, "confidence"),
        "component": _check_str(d, "component"),
        "edit_radius": _check_enum(d, "edit_radius", ["small", "large"]),
        "expected_cost": _check_str(d, "expected_cost"),
        "incompatibilities": _check_list_of_str(d, "incompatibilities"),
        "uncertainty": _check_float01(d, "uncertainty"),
    }


# ---------------------------------------------------------------------------
# ground_in_literature()
# ---------------------------------------------------------------------------

_LITERATURE_KEYS = [
    "mechanism", "assumptions", "contradictory_findings",
    "dataset_compatibility", "implementation_cost", "primary_citation",
]


def validate_literature(parsed: Any) -> Dict:
    d = _require_dict(parsed)
    _require_keys(d, _LITERATURE_KEYS)
    _reject_unknown_keys(d, _LITERATURE_KEYS)
    return {
        "mechanism": _check_str(d, "mechanism"),
        "assumptions": _check_list_of_str(d, "assumptions"),
        "contradictory_findings": _check_list_of_str(d, "contradictory_findings"),
        "dataset_compatibility": _check_list_of_str(d, "dataset_compatibility"),
        "implementation_cost": _check_str(d, "implementation_cost"),
        "primary_citation": _check_str(d, "primary_citation"),
    }


# ---------------------------------------------------------------------------
# generate_hypothesis()
# ---------------------------------------------------------------------------

_HYPOTHESIS_KEYS = ["mechanism", "success_criterion_paired", "implementation_sketch"]

# Heuristic only: flags hypotheses that look like a flat absolute threshold
# rather than a candidate-minus-parent delta on a named tier. This is a
# best-effort lint, not a guarantee — tune the keyword lists if you see
# false positives/negatives in practice.
_DELTA_INDICATORS = ["delta", "improve", "increase", "over the parent", "vs. parent",
                      "vs parent", "relative to parent", "compared to parent", "+", "-"]
_TIER_INDICATORS = ["tier", "val-", "validation", "fold", "split"]


def _check_success_criterion(value: str) -> str:
    lowered = value.lower()
    has_delta_language = any(tok in lowered for tok in _DELTA_INDICATORS)
    has_tier_reference = any(tok in lowered for tok in _TIER_INDICATORS)
    if not has_delta_language:
        raise ValueError(
            f"success_criterion_paired ({value!r}) doesn't read as a candidate-minus-parent "
            f"delta claim (no comparison language found). It must compare against the parent, "
            f"not state a flat absolute threshold."
        )
    if not has_tier_reference:
        raise ValueError(
            f"success_criterion_paired ({value!r}) doesn't name a validation tier/split. "
            f"It must specify which validation tier the comparison is measured on."
        )
    return value


def validate_hypothesis_item(parsed: Any) -> Dict:
    d = _require_dict(parsed)
    _require_keys(d, _HYPOTHESIS_KEYS)
    _reject_unknown_keys(d, _HYPOTHESIS_KEYS)
    return {
        "mechanism": _check_str(d, "mechanism"),
        "success_criterion_paired": _check_success_criterion(_check_str(d, "success_criterion_paired")),
        "implementation_sketch": _check_str(d, "implementation_sketch"),
    }


def validate_hypothesis_list(parsed: Any, expected_count: int) -> List[Dict]:
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array of hypotheses, got {type(parsed).__name__}.")
    if len(parsed) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} hypothesis object(s), got {len(parsed)}.")
    return [validate_hypothesis_item(item) for item in parsed]


# ---------------------------------------------------------------------------
# audit()
# ---------------------------------------------------------------------------

_AUDIT_KEYS = ["pass", "violations", "notes"]


def validate_audit(parsed: Any) -> Dict:
    d = _require_dict(parsed)
    _require_keys(d, _AUDIT_KEYS)
    _reject_unknown_keys(d, _AUDIT_KEYS)
    pass_val = _check_bool(d, "pass")
    violations = _check_list_of_str(d, "violations")
    if pass_val and violations:
        raise ValueError("'pass' is true but 'violations' is non-empty — these are contradictory.")
    if not pass_val and not violations:
        raise ValueError("'pass' is false but 'violations' is empty — must list at least one violated key.")
    notes = d["notes"]
    if not isinstance(notes, str):
        raise ValueError(f"Field 'notes' must be a string (may be empty), got {notes!r}.")
    return {"pass": pass_val, "violations": violations, "notes": notes}


# ---------------------------------------------------------------------------
# dedup_fingerprint_match() escalation
# ---------------------------------------------------------------------------

_DEDUP_KEYS = ["duplicate", "reasoning"]


def validate_dedup(parsed: Any) -> Dict:
    d = _require_dict(parsed)
    _require_keys(d, _DEDUP_KEYS)
    _reject_unknown_keys(d, _DEDUP_KEYS)
    return {
        "duplicate": _check_bool(d, "duplicate"),
        "reasoning": _check_str(d, "reasoning"),
    }
