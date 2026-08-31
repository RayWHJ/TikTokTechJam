"""The mechanism-family taxonomy, shared by the proposer and the search.

WHY IT LIVES HERE. The table used to be `driver.py::_MECHANISM_FAMILIES`, read
only by `_fingerprint`. T2.6 makes the family a DECLARED schema field, which means
`llm_calls/schemas.py` has to validate against the same list — and having
`llm_calls` import `orchestrator.driver` would invert the dependency and break
this package's standalone test harness. So the taxonomy moves down to the layer
both sides can see, and `driver.py` re-exports `ALL_FAMILIES` for its own callers.

WHY DECLARING BEATS GUESSING. Family assignment was substring matching over the
mechanism prose, checked in hand-ordered table position, first match wins. Two
consequences, both observed:

  * "a pairwise loss over user history" resolves to `generic_pairwise` rather
    than `sequence_features`, purely because the loss families are listed first.
  * `sequence_features` matches the bare token "sequence", so it swallows any
    hypothesis containing that word anywhere — DIN, target attention, history
    pooling, session features — none of which had ever run when the family was
    banned off one categorical-field experiment.

A declared field makes the fingerprint exact and the ban legible in both
directions: the proposer is told which families are refuted, and its answer says
which one it chose.
"""
from __future__ import annotations

from typing import Optional, Tuple

#: The free-text escape hatch. A genuinely novel mechanism declares `other` and
#: is ALWAYS legal — no family entry can block it, because it has no family. That
#: is deliberate: the taxonomy is a dedup aid, not a menu of permitted ideas.
OTHER = "other"

#: Families, in the order the substring FALLBACK checks them. Specific surrogates
#: before the generic bucket they belong to.
#:
#: Deliberately coarse: the point is that two proposals which would produce
#: substantially the same edit collapse to one entry, not that the taxonomy is
#: complete. The token lists now serve only hypotheses that did NOT declare a
#: family (an older mock, a hand-built dict in a test) — production declares.
MECHANISM_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("lambdarank_surrogate", ("lambdarank", "lambda rank", "lambda-weight",
                              "lambda weight")),
    ("ranknet_pairwise", ("ranknet", "rank net")),
    ("bpr_pairwise", ("bpr", "bayesian personalized ranking",
                      "bayesian personalised ranking")),
    ("listwise_softmax", ("listwise", "list-wise", "within-user softmax",
                          "softmax over these scores", "softmax loss")),
    ("generic_pairwise", ("pairwise loss", "pairwise ranking", "pair-wise")),
    ("multitask_auxiliary", ("multi-task", "multitask", "auxiliary task",
                             "auxiliary loss", "esmm")),
    ("sequence_features", ("sequence", "behaviour history", "behavior history",
                           "user history", "din", "target attention")),
    ("watchtime_censored", ("censored", "watch time", "watch-time",
                            "play_time")),
    ("capacity_or_regularization", ("embedding dimension", "embedding dim",
                                    "increase k", "weight decay", "dropout",
                                    "l2 regularization", "l2 regularisation")),
    ("static_feature_domains", ("add feature", "additional feature",
                                "more feature", "feature domain",
                                "extra categorical")),
    ("negative_sampling", ("negative sampling", "sample negatives",
                           "hard negative")),
    ("gbdt_swap", ("lightgbm", "gbdt", "gradient boost")),
    ("ensemble_blend", ("ensemble", "blend", "stack")),
)

#: Every family name, in table order.
ALL_FAMILIES: Tuple[str, ...] = tuple(fam for fam, _tokens in MECHANISM_FAMILIES)

#: What a hypothesis is allowed to declare: a known family, or the escape hatch.
LEGAL_DECLARATIONS: Tuple[str, ...] = ALL_FAMILIES + (OTHER,)


def family_from_text(text: str) -> Optional[str]:
    """The substring FALLBACK, for a hypothesis that declared nothing.

    Returns None when no token matches, which is what sends `_fingerprint` to its
    prose-hash branch — permissive, and correct for a novel idea.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    return next((fam for fam, tokens in MECHANISM_FAMILIES
                 if any(t in lowered for t in tokens)), None)


def normalise_declaration(value) -> Optional[str]:
    """Clean a declared `mechanism_family` value, or None if unusable.

    An unrecognised non-empty string is returned AS ITSELF rather than bucketed
    into `other` or into the nearest known family. Bucketing an unknown value is
    how a novel mechanism inherits a ban it has nothing to do with; keeping it
    distinct means it gets its own fingerprint and its own budget.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    return cleaned or None
