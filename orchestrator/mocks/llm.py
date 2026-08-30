"""Mock of llm_calls/. Deterministic canned JSON."""

def diagnose(node_context):
    return {"bottleneck": "pointwise logloss misaligned with ranking metric",
            "evidence": f"plateau at ~0.595 across k for parent {node_context.get('parent')}",
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

def generate_hypothesis(diagnosis, evidence_card):
    return [{
        "mechanism": "swap pointwise logloss for BPR pairs sampled within user_id",
        "success_criterion_paired": "candidate primary − parent primary > 0.005 on valid_search",
        "implementation_sketch": "in baseline.py FM.step, form (pos, neg) pairs per user",
        # extras the orchestrator uses for dedup fingerprinting
        "loss_type": "bpr", "sampler": "within_user_neg",
        "feature_set": "5field_baseline", "dataset_tier": "pure",
    }]

def audit(diff, checklist):
    """Blind — sees only the diff."""
    violations = []
    if "test" in diff.lower() and "load" in diff.lower():
        violations.append("suspected test-split access")
    return {"pass": len(violations) == 0, "violations": violations, "notes": ""}