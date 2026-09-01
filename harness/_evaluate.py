"""validated_evaluate() — a guardrail wrapper around the real, unmodified
evaluate.py:evaluate(). Never returns a number computed from malformed input;
raises ValueError instead.
"""
import numpy as np

from evaluate import evaluate as _real_evaluate

from ._data import get_split_sizes
from ._sizes import SPLIT_NAMES

RESULT_KEYS = ('GAUC', 'nDCG@5', 'primary', 'users', 'rows')


def _as_float_array(values, name):
    """Coerce to a float64 array, reporting a non-numeric element as ValueError.

    numpy raises TypeError (not ValueError) for e.g. a list containing a dict;
    the frozen contract promises ValueError on any malformed input, so we
    normalize here rather than letting a TypeError escape.
    """
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def validated_evaluate(user_ids: list, labels: list, scores: list, split_name: str) -> dict:
    """Validate inputs, then call the real evaluate.py:evaluate().

    Args:
        user_ids: per-row user id. Must be non-null and the same length as the
            other two arrays; values are compared by equality only, so any
            hashable type works (data.py yields strings).
        labels: per-row long_view label, must be binary 0/1 (ints, floats, or
            bools all accepted; anything else rejected).
        scores: per-row model score, must be finite. Any real number — only the
            within-user relative order affects the metrics.
        split_name: one of "train", "valid_search", "valid_confirm", "test" —
            used only to check that len(labels) matches that split's known row
            count; never used to look up labels itself.

    Returns:
        {"GAUC": float, "nDCG@5": float, "primary": float, "users": int, "rows": int}

    Raises:
        ValueError: on unequal lengths, empty input, non-numeric or NaN/Inf
            scores, non-binary or NaN labels, a None/NaN user_id, an unrecognized
            split_name, or a row count that does not match split_name's expected
            size.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(
            f"unknown split_name {split_name!r}, expected one of {SPLIT_NAMES}"
        )

    user_ids = list(user_ids)
    n_u, n_l, n_s = len(user_ids), len(labels), len(scores)
    if not (n_u == n_l == n_s):
        raise ValueError(
            f"unequal lengths: user_ids={n_u}, labels={n_l}, scores={n_s}"
        )
    if n_l == 0:
        raise ValueError("empty input: user_ids/labels/scores have length 0")

    score_arr = _as_float_array(scores, 'scores')
    bad = np.flatnonzero(~np.isfinite(score_arr))
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"scores contains {bad.size} non-finite value(s); first at index {i}: "
            f"{score_arr[i]!r}"
        )

    label_arr = _as_float_array(labels, 'labels')
    bad = np.flatnonzero((label_arr != 0.0) & (label_arr != 1.0))
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"labels contains {bad.size} non-binary value(s); first at index {i}: "
            f"{labels[i]!r} (must be 0 or 1)"
        )

    for i, u in enumerate(user_ids):
        if u is None or (isinstance(u, float) and u != u):
            raise ValueError(f"user_ids[{i}] is null/NaN: {u!r}")

    expected = get_split_sizes()[split_name]
    if n_l != expected:
        raise ValueError(
            f"row count {n_l} does not match split_name={split_name!r}'s "
            f"expected size {expected}"
        )

    # Hand evaluate.py plain ints for labels — its nDCG gain term is 2**rel, and
    # 2**np.float32(1) would silently return a float array element. ints keep the
    # published convention exact.
    result = _real_evaluate(user_ids, [int(v) for v in label_arr], score_arr.tolist())

    missing = [k for k in RESULT_KEYS if k not in result]
    if missing:  # guards against evaluate.py drifting out from under us
        raise ValueError(f"evaluate() returned no {missing}; harness contract broken")
    return result
