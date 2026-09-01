"""Tests for the search-control fixes: parent acceptance, mechanism-family
dedup, the attempt ledger, champion archiving, and frontier pruning.

Each test names the observed failure it guards against. The numbers quoted are
from orchestrator/_state/progress.json and root_baseline.json — the 5-iteration
run whose output was the unmodified baseline.
"""
import json
import random

import pytest

from orchestrator import driver
from orchestrator.node import Node
from orchestrator.promotion import (should_continue_locally,
                                    should_expand_as_parent)


# --------------------------------------------------------------------------- #
#  Parent acceptance                                                           #
# --------------------------------------------------------------------------- #
#: Iteration 1's best candidate, verbatim from progress.json. The only
#: candidate in the whole run with a positive paired delta.
ITER1_WINNER = dict(mean_delta=0.000592797564408543,
                    p_positive=0.776,
                    lower_95=-0.0006730252126552841)


def test_significance_test_rejected_the_only_positive_candidate():
    """The behaviour being fixed, pinned so it can't be reintroduced silently.

    p_positive 0.776 misses the 0.8 bar by 0.024 and lower_95 is negative, so
    the candidate never entered open_nodes — which is why all 11 nodes in
    nodes.jsonl have the root as their parent.
    """
    assert not should_continue_locally(**ITER1_WINNER)


def test_hillclimb_accepts_the_same_candidate_as_a_parent():
    assert should_expand_as_parent(**ITER1_WINNER)


def test_hillclimb_still_refuses_a_negative_step():
    """Iteration 4 and 5's candidates, which really were worse."""
    assert not should_expand_as_parent(mean_delta=-0.0003592755951288171,
                                       p_positive=0.268, lower_95=-0.00136)
    assert not should_expand_as_parent(mean_delta=-0.011524786426779267,
                                       p_positive=0.0, lower_95=-0.01329)


def test_hillclimb_refuses_a_positive_mean_with_a_coin_flip_against_it():
    """A positive mean is not enough on its own — the bootstrap must at least
    not be leaning negative, or the step is noise with a favourable sign."""
    assert not should_expand_as_parent(mean_delta=0.0004, p_positive=0.40,
                                       lower_95=-0.003)


def test_driver_uses_the_hillclimb_bar_not_the_significance_bar():
    """Guards the wiring, not the rule: the constants must be the loose ones."""
    assert driver.HILLCLIMB_MIN_MEAN_DELTA == 0.0
    assert driver.HILLCLIMB_MIN_P_POSITIVE <= 0.5
    # The promotion trigger must stay strict — it spends a sealed-split query.
    assert driver.PROMOTE_TRIGGER_P_POS >= 0.9


# --------------------------------------------------------------------------- #
#  Mechanism-family fingerprinting                                             #
# --------------------------------------------------------------------------- #
#: The mechanism strings of the four loss-swap proposals from iterations 1-5,
#: abridged. Every one is "replace the pointwise loss with a ranking loss";
#: each was worded differently, so each hashed differently, so memory dedup
#: never fired once across 11 candidates.
REWORDED_BPR = [
    "Switching from a pointwise loss to a pairwise BPR-style loss should "
    "better model relative preferences, improving GAUC",
    "Because the pointwise logloss does not align with ranking objectives, "
    "replacing it with a pairwise BPR loss should improve the ranking",
    "By replacing the current pointwise loss with a Bayesian Personalized "
    "Ranking (BPR) loss, the model should better learn to rank",
    "Since the current FM uses a pointwise loss, implementing a pairwise BPR "
    "loss should improve GAUC and nDCG@5",
]


def test_reworded_bpr_proposals_collapse_to_one_fingerprint():
    prints = {driver._fingerprint({"mechanism": m}) for m in REWORDED_BPR}
    assert len(prints) == 1, f"still distinguishable: {prints}"
    assert prints.pop() == ("mechanism_family", "bpr_pairwise", "", "")


def test_prose_hashing_was_the_bug():
    """Same four mechanisms under the old rule: four distinct fingerprints."""
    import hashlib
    old = {("mechanism_hash",
            hashlib.sha1(m.strip().lower().encode()).hexdigest()[:16], "", "")
           for m in REWORDED_BPR}
    assert len(old) == 4


@pytest.mark.parametrize("mechanism,family", [
    ("implement a LambdaRank-style loss weighted by nDCG impact",
     "lambdarank_surrogate"),
    ("apply a RankNet-style pairwise cross-entropy over score differences",
     "ranknet_pairwise"),
    ("compute a listwise softmax over each user's candidate slate",
     "listwise_softmax"),
    ("add a multi-task auxiliary head on is_click and is_like",
     "multitask_auxiliary"),
    ("model the user behaviour history as a sequence with target attention",
     "sequence_features"),
    ("use a censored regression loss on watch time", "watchtime_censored"),
])
def test_distinct_families_stay_distinct(mechanism, family):
    fp = driver._fingerprint({"mechanism": mechanism})
    assert fp == ("mechanism_family", family, "", "")


def test_lambdarank_wins_over_the_generic_pairwise_bucket():
    """Order matters: a LambdaRank proposal mentioning pairs must not land in
    generic_pairwise, or it would dedup against an unrelated BPR attempt."""
    fp = driver._fingerprint({
        "mechanism": "a lambdarank surrogate over pairwise score differences"})
    assert fp == ("mechanism_family", "lambdarank_surrogate", "", "")


def test_unrecognised_mechanism_still_falls_through_to_a_hash():
    """A genuinely novel idea must not be forced into a family it doesn't
    belong to — the taxonomy is coarse on purpose, so the fallback stays."""
    fp = driver._fingerprint({"mechanism": "recalibrate scores per tab using "
                                           "isotonic regression on the "
                                           "randomised-exposure log"})
    assert fp[0] == "mechanism_hash"


def test_structured_fields_still_take_precedence():
    """The hand-authored preseeds in memory.py depend on this path."""
    fp = driver._fingerprint({"loss_type": "bpr", "sampler": "within_user_neg",
                              "feature_set": "5field_baseline",
                              "dataset_tier": "pure",
                              "mechanism": "listwise softmax"})
    assert fp == ("bpr", "within_user_neg", "5field_baseline", "pure")


# --------------------------------------------------------------------------- #
#  Attempt ledger                                                              #
# --------------------------------------------------------------------------- #
def _node(nid, parent_id, mechanism, *, component="loss_function",
          evidence=None, per_seed=None, mean_delta=None):
    n = Node(id=nid, parent_id=parent_id, code_path="baseline.py",
             diagnosis={"component": component},
             hypothesis={"mechanism": mechanism})
    n.evidence_type = evidence
    n.per_seed_primary = per_seed or {}
    n.mean_delta = mean_delta
    return n


def test_ledger_reports_outcome_and_delta_for_each_attempt():
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    a = _node("a", "root", "BPR pairwise", per_seed={0: 0.5949},
              mean_delta=0.00059)
    b = _node("b", "root", "listwise softmax", evidence="failed_implementation")
    led = driver._attempt_ledger([root, a, b], parent=root)

    assert len(led) == 2, "the root draft has no hypothesis and must be skipped"
    assert led[0]["mechanism"] == "BPR pairwise"
    assert led[0]["outcome"] == "scored"
    assert led[0]["mean_delta_vs_parent"] == pytest.approx(0.00059)
    assert led[1]["outcome"] == "failed_implementation"
    assert led[1]["primary"] is None
    assert all(e["sibling_of_selected_parent"] for e in led)


def test_ledger_marks_siblings_of_the_selected_parent():
    """AIRA's scoped memory: siblings are the diversity signal, so the operator
    has to be able to tell them from unrelated branches."""
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    other = Node(id="other", parent_id="root", code_path="baseline.py")
    sib = _node("sib", "other", "sequence features")
    cousin = _node("cousin", "root", "multi-task auxiliary")
    led = {e["id"]: e for e in
           driver._attempt_ledger([root, other, sib, cousin], parent=other)}
    assert led["sib"]["sibling_of_selected_parent"] is True
    assert led["cousin"]["sibling_of_selected_parent"] is False


def test_ledger_is_bounded_and_keeps_the_most_recent():
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    nodes = [root] + [_node(f"n{i}", "root", f"mech {i}") for i in range(40)]
    led = driver._attempt_ledger(nodes, parent=root)
    assert len(led) == driver.LEDGER_MAX_ENTRIES
    assert led[-1]["mechanism"] == "mech 39", "newest must survive truncation"


def test_hypothesis_prompt_contains_the_ledger():
    """The ledger is worthless if it never reaches the prompt string."""
    from llm_calls.hypothesis import _build_prompt
    tried = [{"id": "a", "mechanism": "BPR pairwise",
              "outcome": "scored", "mean_delta_vs_parent": -0.011}]
    prompt = _build_prompt({"bottleneck": "x"}, {"y": 1}, 3, tried=tried)
    assert "ALREADY ATTEMPTED" in prompt
    assert "BPR pairwise" in prompt
    # And absent when there is nothing to report, so iteration 1 is unchanged.
    assert "ALREADY ATTEMPTED" not in _build_prompt({"bottleneck": "x"},
                                                   {"y": 1}, 3, tried=[])


# --------------------------------------------------------------------------- #
#  Champion archiving                                                          #
# --------------------------------------------------------------------------- #
def test_champion_archive_copies_source_out_of_the_temp_dir(tmp_path):
    """The 5-iteration run staged every candidate under /var/folders/... and
    copied none of them out, so even a winner left nothing to submit from."""
    staged = tmp_path / "candidate_abc"
    staged.mkdir()
    (staged / "baseline.py").write_text("# patched model\n")
    (staged / "data.py").write_text("# patched features\n")

    node = _node("abc", "root", "BPR blended at weight 0.2",
                 per_seed={0: 0.5960}, mean_delta=0.0012)
    node.code_dir = str(staged)

    dest = driver._archive_champion(node, 0.5960,
                                   champion_dir=str(tmp_path / "champions"))

    assert dest is not None
    assert (tmp_path / "champions" / "abc" / "baseline.py").read_text() \
        == "# patched model\n"
    manifest = json.loads(
        (tmp_path / "champions" / "abc" / "manifest.json").read_text())
    assert manifest["primary"] == 0.5960
    assert manifest["hypothesis"]["mechanism"].startswith("BPR blended")
    assert manifest["staged_from"] == str(staged)


def test_champion_archive_is_a_noop_without_a_staged_dir(tmp_path):
    node = Node(id="x", parent_id="root", code_path="baseline.py")
    node.code_dir = str(tmp_path / "does-not-exist")
    assert driver._archive_champion(
        node, 0.6, champion_dir=str(tmp_path / "champions")) is None


# --------------------------------------------------------------------------- #
#  Frontier pruning                                                            #
# --------------------------------------------------------------------------- #
def test_frontier_cap_is_small_enough_to_concentrate_visits():
    assert 2 <= driver.MAX_OPEN_NODES <= 6


def test_mocked_run_grows_the_tree_and_reports_a_champion(tmp_path, monkeypatch):
    """End-to-end on the mocks: the tree must leave the root and the run must
    return a champion distinct from global_best when nothing was promoted.

    Also T2.6's acceptance: this passes with a mock that emits NO non-production
    keys, so the end-to-end path finally matches the one production takes.

    Node ids are pinned to a counter. `_new_id()` uses `uuid.uuid4()`, and
    `mocks/codegen.execute` derives a candidate's per-user scores from
    `sha1(code_path|seed)` — so whether a candidate beat its parent, and
    therefore whether the tree grew, depended on a random UUID. Measured at
    roughly 1 failure in 6 runs. `random.seed` cannot fix it because uuid4 reads
    os.urandom, so the id generator itself has to be deterministic.
    """
    import itertools

    from orchestrator.mocks import harness as h, llm as l, codegen as c
    monkeypatch.setattr(driver, "harness", h)
    monkeypatch.setattr(driver, "llm", l)
    monkeypatch.setattr(driver, "codegen", c)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    monkeypatch.setattr(driver, "CHAMPION_DIR", str(tmp_path / "champions"))

    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")
    random.seed(0)                       # mocks.codegen's 5% error injection

    res = driver.run(max_iters=8, verbose=False,
                     progress_path=str(tmp_path / "progress.json"),
                     root_baseline_path=str(tmp_path / "root.json"),
                     confirm_baseline_path=str(tmp_path / "confirm.json"),
                     memory_path=str(tmp_path / "memory.json"))

    assert "champion_primary" in res and "champion_node_id" in res
    prog = json.loads((tmp_path / "progress.json").read_text())
    assert prog["iters_completed"] >= 4, "the run must not stall at iteration 3"
    assert max(r["n_open_nodes"] for r in prog["iterations"]) > 1, \
        "open_nodes never grew past the root — the tree is still flat"
    assert res["champion_primary"] >= prog["baseline_primary"], \
        "the champion must never be worse than the baseline it started from"
