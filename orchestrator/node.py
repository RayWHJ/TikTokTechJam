from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class Node:
    """One point in the search tree — a candidate modification of a parent.
    Fields id/parent_id/code_path/diagnosis/hypothesis/status/local_best_score/
    seeds_run/evidence_type are from the frozen contract. The rest is local
    bookkeeping for selection and triage."""
    id: str
    parent_id: Optional[str]
    code_path: str
    diagnosis: Optional[dict] = None
    hypothesis: Optional[dict] = None
    status: str = "open"                 # "open" | "closed" | "promoted"
    local_best_score: float = float("-inf")
    seeds_run: List[int] = field(default_factory=list)
    evidence_type: Optional[str] = None  # "invariant" | "refuted_under_context"
                                          # | "failed_implementation" | "no_op"
                                          # | "timeout" | "inconclusive"
                                          # "timeout" is deliberately NOT
                                          # "failed_implementation": the code ran,
                                          # it just exceeded
                                          # driver.TRIAGE_WALLCLOCK_CAP_S, so the
                                          # mechanism is unmeasured rather than
                                          # unimplementable.

    # One of: "draft" (a root seed), "improve" (a modification of an ok parent),
    # "fix" (a repair applied to a failed parent), "refine" (Phase 2: a component-
    # scoped improve targeting a specific ablation weakness). Plain string, not
    # typing.Literal, to keep JSON round-trip trivial — matches `status` and
    # `evidence_type` above.
    operation: str = "improve"

    # Number of debug-and-retry attempts spent on this node. Capped by
    # driver.MAX_FIX_ATTEMPTS. Counts even attempts that failed to produce a
    # usable diff, so the ceiling is enforced on LLM calls, not just on successes.
    fix_attempts: int = 0

    # local extras
    partial_scores: List[float] = field(default_factory=list)
    per_user_by_seed: Dict[int, Dict[str, float]] = field(default_factory=dict)
    wallclock_used_s: float = 0.0
    n_visits: int = 0

    # Per-seed primary, so a node's scalar score can be a MEAN over the seeds it
    # actually ran. Comparing max-over-3-seeds (the root) against max-over-1-seed
    # (a triage-only candidate) handed the root a measured +0.00033 head start —
    # see driver._scalar_primary for the arithmetic on the real cached baseline.
    per_seed_primary: Dict[int, float] = field(default_factory=dict)

    # Paired candidate-vs-parent statistics from promotion.bootstrap_delta.
    # Cached on the node so progress.json can report WHY a candidate was kept or
    # dropped instead of only its noisy absolute score.
    mean_delta: Optional[float] = None
    p_positive: Optional[float] = None
    lower_95: Optional[float] = None

    # Primary on the sealed valid_confirm split, set only when a promotion
    # attempt actually spends a confirm query.
    confirm_primary: Optional[float] = None

    # Directory holding this node's staged, already-patched source tree. The
    # root's is the repo itself.
    code_dir: str = "."

    # Tail of the run log from this node's last failed execute(), truncated.
    # The TAIL specifically, because a Python traceback puts its cause last.
    #
    # Exists so driver._ancestor_chain can tell the debug operator not just that
    # an ancestor failed but HOW. Before this the per-attempt failure reason
    # lived only in the run log, never on the Node, so nothing downstream could
    # read it: 6 of the 11 candidates in orchestrator/_state/nodes.jsonl were
    # filed as failed_implementation and 1 as timeout, and every repair attempt
    # on them started from a blank slate.
    last_error_excerpt: Optional[str] = None

    # Verdict on this node's own declared success_criterion_paired, from
    # llm_calls.verdict. Every hypothesis in the recorded run stated a criterion
    # between +0.004 and +0.007 and nothing ever checked it, so "the mechanism
    # was refuted", "the mechanism was never fairly implemented" and "the
    # criterion was uncalibrated by 8x" all collapsed into one record.
    verdict: Optional[str] = None          # met | missed_but_promising
                                           # | refuted | not_tested
    verdict_reason: Optional[str] = None
    next_action: Optional[str] = None      # retry_cheaper | adjust_magnitude
                                           # | abandon_mechanism | build_on_it
    criterion_was_calibrated: Optional[bool] = None