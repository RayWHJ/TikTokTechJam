"""End-to-end smoke test with mocks."""
import os
import pytest
from orchestrator.driver import run
from orchestrator.memory import Memory
from orchestrator.selection import select
from orchestrator.node import Node
from orchestrator.promotion import bootstrap_delta, should_promote_globally
from orchestrator.convergence import local_plateau
from orchestrator.triage import rank


def test_full_loop_runs():
    result = run(max_iters=3, verbose=False)
    assert "global_best" in result
    assert result["counters"].proposals >= 0
    assert result["counters"].scorer_queries["valid_search"] > 0

def test_memory_preseed_blocks_known_dead_ends(tmp_path):
    m = Memory(path=str(tmp_path / "mem.json"))
    dup = m.is_duplicate(("pointwise_logloss", "uniform", "cwm_13field", "pure"))
    assert dup is not None
    assert dup.evidence_type == "refuted_under_context"

def test_selection_prefers_untried():
    fresh = Node(id="a", parent_id=None, code_path="x", n_visits=0)
    stale = Node(id="b", parent_id=None, code_path="x",
                 n_visits=5, local_best_score=0.6)
    assert select([fresh, stale]).id == "a"

def test_bootstrap_delta_paired_over_users():
    cand = {0: {"u1": 0.7, "u2": 0.6}, 1: {"u1": 0.72, "u2": 0.61}}
    par  = {0: {"u1": 0.5, "u2": 0.5}, 1: {"u1": 0.5, "u2": 0.5}}
    mean, p_pos, lower = bootstrap_delta(cand, par, n_boot=200)
    assert mean > 0.1
    assert p_pos == 1.0
    assert lower > 0

def test_bootstrap_no_matched_seeds_returns_zero():
    cand = {0: {"u1": 0.7}}; par = {1: {"u1": 0.5}}
    assert bootstrap_delta(cand, par) == (0.0, 0.0, 0.0)

def test_local_plateau_rule():
    assert local_plateau([0.5, 0.51, 0.511, 0.5115, 0.5116])
    assert not local_plateau([0.5, 0.51, 0.52, 0.55, 0.60])

def test_triage_reserves_wildcard():
    nodes = [Node(id=str(i), parent_id=None, code_path="x",
                  partial_scores=[0.6 - 0.01 * i]) for i in range(5)]
    top = rank(nodes, keep=3, wildcard=True)
    assert len(top) == 3
    assert nodes[0] in top   # highest mean must be included