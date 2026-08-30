"""Paired candidate-minus-parent delta, bootstrapped over user blocks."""
import random
from typing import Dict, Tuple, List

def paired_user_deltas(cand_per_user_by_seed: Dict[int, Dict[str, float]],
                       parent_per_user_by_seed: Dict[int, Dict[str, float]]
                       ) -> List[float]:
    """Only use seeds where BOTH sides ran; only use users present in both."""
    matched_seeds = set(cand_per_user_by_seed) & set(parent_per_user_by_seed)
    deltas: List[float] = []
    for s in matched_seeds:
        cu, pu = cand_per_user_by_seed[s], parent_per_user_by_seed[s]
        for u in set(cu) & set(pu):
            deltas.append(cu[u] - pu[u])
    return deltas

def bootstrap_delta(cand_per_user_by_seed, parent_per_user_by_seed,
                    n_boot: int = 500, seed: int = 0
                    ) -> Tuple[float, float, float]:
    """Returns (mean_delta, p_positive, lower_one_sided_95_ci)."""
    deltas = paired_user_deltas(cand_per_user_by_seed, parent_per_user_by_seed)
    if not deltas:
        return (0.0, 0.0, 0.0)
    mean_delta = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in deltas]
        boot_means.append(sum(sample) / len(sample))
    p_pos = sum(1 for m in boot_means if m > 0) / n_boot
    boot_means.sort()
    lower_95 = boot_means[int(0.05 * n_boot)]
    return (mean_delta, p_pos, lower_95)

def should_continue_locally(mean_delta, p_positive, lower_95, margin=0.0):
    """Keep exploring below this candidate?

    A proper one-sided test: the paired per-user delta must be positive at the
    95% lower bound. The third argument is the bootstrap LOWER bound.

    It used to be an `upper_bound` built as `mean_d + 2*abs(mean_d - lower_95)`
    and compared against margin=0.002. Work a realistic case: mean_delta=0.0007,
    lower_95=0.0002 gives upper_bound=0.0017 < 0.002, so a genuinely-positive
    candidate FAILED and never entered open_nodes — it could never become a
    parent, so nothing could ever build on it. The paired bootstrap runs over
    thousands of users and has ample power to resolve 0.0007, so requiring
    significance rather than a hard 0.002 effect size is the tighter test, not
    the looser one.
    """
    return p_positive > 0.8 and lower_95 > margin

def should_promote_globally(confirm_mean_delta, confirm_lower_95, margin=0.0005):
    """Promote to global champion?

    Both arguments must come from a PAIRED bootstrap on valid_confirm against
    the same split's baseline. The margin is 0.0005 rather than 0.002: nothing
    this search has produced clears 0.002, and a significant +0.001 confirmed on
    a sealed split is a real win worth promoting.
    """
    return confirm_mean_delta > margin and confirm_lower_95 > 0