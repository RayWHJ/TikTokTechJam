"""Mock of llm_calls/. Deterministic canned JSON."""
import hashlib
import json


def _context_digest(node_context) -> str:
    """Stable digest of the whole node context, sorted so dict order can't
    perturb it. Deterministic across processes (no PYTHONHASHSEED
    dependence) and across test orderings (no call counter)."""
    blob = json.dumps(node_context, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def diagnose(node_context):
    # `evidence` carries a digest of the FULL node context, which the real
    # diagnostician also sees in its prompt. Since that context now includes
    # iter_history and improvement_score, it differs on every iteration — so
    # the digest does too, and generate_hypothesis below produces a distinct
    # fingerprint per iteration rather than one frozen mechanism the memory
    # dedup kills at iteration 2.
    return {"bottleneck": "pointwise logloss misaligned with ranking metric",
            "evidence": (f"plateau at ~0.595 across k for parent "
                         f"{node_context.get('parent')} "
                         f"[ctx {_context_digest(node_context)}]"),
            "confidence": 0.75, "component": "loss",
            "edit_radius": "small", "expected_cost": "medium",
            "incompatibilities": [], "uncertainty": 0.25}

def ground_in_literature(bottleneck):
    return {"mechanism": "BPR pairwise loss sampled within user",
            "assumptions": ["at least one positive per user per batch"],
            "contradictory_findings": [],
            "dataset_compatibility": ["~63% discriminative users in KuaiRand-Pure test"],
            "implementation_cost": "small",
            "primary_citation": "Rendle et al. 2009 BPR"}

def generate_hypothesis(diagnosis, evidence_card, tried=None):
    """One hypothesis, varying with the diagnosis it was given.

    `tried` is accepted and folded into the variant tag rather than ignored.
    Ignoring it would let the mock diverge from the real signature, and it is
    also the only way a mocked run can exercise the ledger path at all: the
    driver now passes an attempt ledger on every improve iteration, so a mock
    that dropped it would silently mask a TypeError in production.

    The variation is load-bearing, not cosmetic. This used to return a single
    frozen mechanism, so every iteration produced the same _fingerprint, and
    memory dedup killed iteration 2 onward with zero candidates — which closed
    the parent, emptied open_nodes, and stopped a `--mock --max-iters 8` run at
    iteration 3. Nothing past iteration 3 (the refine cadence, the ε/N plateau
    signal) could be exercised against the mocks at all.

    Derived from the diagnosis rather than a call counter, so a mocked run is
    reproducible regardless of which tests ran first. diagnose() folds a digest
    of its whole node context into `evidence`, and that context now carries
    iter_history — so the tag advances once per iteration.
    """
    tag = hashlib.sha1(
        f"{diagnosis.get('bottleneck', '')}|{diagnosis.get('evidence', '')}"
        f"|{len(tried or [])}"
        .encode("utf-8")).hexdigest()[:8]
    return [{
        "mechanism": ("swap pointwise logloss for BPR pairs sampled within "
                      f"user_id (variant {tag})"),
        "success_criterion_paired": "candidate primary − parent primary > 0.005 on valid_search",
        "implementation_sketch": "in baseline.py FM.step, form (pos, neg) pairs per user",
        # extras the orchestrator uses for dedup fingerprinting
        "loss_type": f"bpr_{tag}", "sampler": "within_user_neg",
        "feature_set": "5field_baseline", "dataset_tier": "pure",
    }]

def refine(component, component_source, ablations, iter_history,
           improvement_score, prior_refines=None):
    """Mock of llm_calls.refine — one component-scoped hypothesis.

    The mechanism varies with the component AND with how many prior refines
    this component already has, so a second refine on the same component
    produces a different fingerprint and a different diff instead of being
    killed by the run-wide dedup.
    """
    attempt = len(prior_refines or []) + 1
    return {
        "mechanism": (f"replace the {component} component "
                      f"(attempt {attempt}): the ablation delta of "
                      f"{ablations.get(component)} says the pipeline barely "
                      f"leans on it"),
        "implementation_sketch": (f"rewrite the {component} block; "
                                  f"attempt {attempt}"),
        "success_criterion_paired": ("primary on val-tier-2 improves by at "
                                     "least +0.003 over the parent"),
        "component": component,
    }


def verdict(hypothesis, measured, context):
    """Mock of llm_calls.verdict — derived from the measurement, not canned.

    Deliberately reproduces the ONE distinction the real persona exists to make:
    a positive-but-small delta against a criterion stating +0.005 is
    `missed_but_promising` with criterion_was_calibrated=False, never `refuted`.
    A mock that always returned "refuted" would make a mocked run close every
    node the same way and hide whether the driver routes the four next_actions
    differently.

    Thresholds are the repo's measured ones: 0.0012 is the paired noise floor,
    0.0008 the baseline's 5-seed std.
    """
    if measured.get("evidence_type") in ("failed_implementation", "timeout",
                                         "no_op"):
        return {"verdict": "not_tested", "criterion_was_calibrated": False,
                "reason": "the mechanism never ran, so it was never measured",
                "next_action": "retry_cheaper"}
    delta = measured.get("mean_delta") or 0.0
    stated = hypothesis.get("success_criterion_paired", "")
    calibrated = not any(t in stated for t in ("0.005", "0.004", "0.006",
                                              "0.007"))
    if delta >= 0.0012:
        return {"verdict": "met", "criterion_was_calibrated": calibrated,
                "reason": "paired delta clears the 0.0012 noise floor",
                "next_action": "build_on_it"}
    if delta > 0:
        return {"verdict": "missed_but_promising",
                "criterion_was_calibrated": calibrated,
                "reason": ("positive but inside the 0.0012 paired noise floor; "
                           "the stated bar was not reachable"),
                "next_action": "adjust_magnitude"}
    return {"verdict": "refuted", "criterion_was_calibrated": True,
            "reason": "paired delta is at or below zero on a fair measurement",
            "next_action": "abandon_mechanism"}


def audit(diff, checklist):
    """Blind — sees only the diff."""
    violations = []
    if "test" in diff.lower() and "load" in diff.lower():
        violations.append("suspected test-split access")
    return {"pass": len(violations) == 0, "violations": violations, "notes": ""}