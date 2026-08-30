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
                                          # | "failed_implementation" | "inconclusive"

    # local extras
    partial_scores: List[float] = field(default_factory=list)
    per_user_by_seed: Dict[int, Dict[str, float]] = field(default_factory=dict)
    wallclock_used_s: float = 0.0
    n_visits: int = 0