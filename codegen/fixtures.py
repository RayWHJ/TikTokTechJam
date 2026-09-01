"""
Hand-written fake inputs so writer.py / report.py (and the rest) can be exercised
end-to-end with NO teammate module present. These mimic the shapes Person C's
llm.generate_hypothesis and Person B's orchestrator run log would produce.
"""

# Shape matches one element of llm.generate_hypothesis(...) -> list[dict].
FAKE_HYPOTHESIS = {
    "mechanism": (
        "The FM baseline trains a pointwise logloss, but the evaluation metric is a "
        "within-user ranking metric (GAUC + nDCG@5). Replace the pointwise objective "
        "with a within-user pairwise BPR loss: for each user, sample (positive, "
        "negative) impression pairs and optimise sigmoid(score_pos - score_neg). This "
        "aligns the training objective with the ranking metric without changing the "
        "feature set or model capacity."
    ),
    "success_criterion_paired": (
        "candidate-minus-parent delta on valid_search > +0.003 primary on matched "
        "seeds, with a positive one-sided 95% lower bound"
    ),
    "implementation_sketch": (
        "In baseline.py, add FM.bpr_step(Xpos, Xneg) reusing the existing logits() "
        "and Adam update; in run_fm, build per-user positive/negative index pools "
        "from the TRAIN split only, sample B pairs per step, and call bpr_step "
        "instead of the pointwise step. Keep predict() and the valid-based early "
        "stopping unchanged; never touch the test split."
    ),
}

# A feature-side hypothesis, to exercise the data.py routing branch of write_fix.
FAKE_HYPOTHESIS_FEATURE = {
    "mechanism": (
        "Add a point-in-time-safe temporal field (hour-of-day bucket derived from "
        "the row's own timestamp) so the model can capture within-day drift. Uses no "
        "aggregate outcome statistics."
    ),
    "success_criterion_paired": "delta on valid_search > +0.002 primary vs parent",
    "implementation_sketch": (
        "In data.py, append 'hour_bucket' to FIELDS and, in raw(), derive it from the "
        "row timestamp (x[0]); everything else in encode() stays the same."
    ),
}

# Shape matches Person B's orchestrator run log (honest, separated counters).
FAKE_RUN_LOG = {
    "task": "KuaiRand-Pure within-user ranking (Track 2)",
    "baseline_primary": 0.5946,
    "global_best": {
        "node_id": "n7",
        "mechanism": "within-user pairwise BPR loss",
        "primary": 0.6013,
        "split": "valid_confirm",
        "delta_vs_parent": 0.0067,
        "lower_bound_95": 0.0021,
        "seeds": [0, 1, 2, 3, 4],
    },
    "counters": {
        "proposals": 11,
        "partial_runs": 9,
        "full_runs": 5,
        "semantic_retries": 2,
        "scorer_queries": {"valid_search": 23, "valid_confirm": 3, "test": 0},
        "wallclock_s": 12840,
        "tokens": 1_450_000,
    },
    "iterations": [
        {"node": "n1", "component": "loss", "hypothesis": "pointwise->BPR",
         "result": "primary 0.5991 on valid_search", "evidence_type": "invariant"},
        {"node": "n3", "component": "capacity", "hypothesis": "k=16->32",
         "result": "no gain (dedup: known refuted_under_context)",
         "evidence_type": "refuted_under_context"},
        {"node": "n7", "component": "loss", "hypothesis": "BPR + valid-confirm promotion",
         "result": "promoted, primary 0.6013 on valid_confirm",
         "evidence_type": "invariant"},
    ],
    "safety": {
        "gate_blocks": 3,
        "example_block": "auxiliary signal is_like wired in as an input feature",
        "test_split_one_shot_used": False,
    },
}
