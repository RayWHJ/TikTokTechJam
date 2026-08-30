"""End-to-end smoke test. Uses mocks even after driver flips to real imports,
because the smoke test verifies orchestrator LOGIC, not real API integration."""
import json
import random
import pytest
from orchestrator import driver
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.memory import Memory
from orchestrator.selection import select
from orchestrator.node import Node
from orchestrator.counters import Counters
from orchestrator.promotion import (bootstrap_delta, paired_user_deltas,
                                    should_promote_globally)
from orchestrator.convergence import local_plateau
from orchestrator.triage import rank


@pytest.fixture
def mocked_driver(monkeypatch, tmp_path):
    """Force driver to use mocks AND an isolated memory file."""
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)

    # isolate Memory to a per-test tmp_path so state doesn't leak between runs
    from orchestrator import memory as memory_mod
    orig_init = memory_mod.Memory.__init__
    def isolated_init(self, path=None):
        orig_init(self, path=str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory_mod.Memory, "__init__", isolated_init)

    # mocks.codegen.execute injects a 5% random error, and the global RNG is
    # seeded from OS entropy per session — pin it so a mocked run's outcome
    # doesn't depend on session entropy or on which tests ran first.
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)

def test_full_loop_runs(mocked_driver, tmp_path):
    result = driver.run(max_iters=3, verbose=False,
                        progress_path=str(tmp_path / "progress.json"))
    assert "global_best" in result
    assert result["counters"].proposals >= 0
    assert result["counters"].scorer_queries["valid_search"] > 0


def test_best_primary_prefers_full_seed_falls_back_to_triage():
    triage_only = Node(id="a", parent_id=None, code_path="x",
                       partial_scores=[0.60])
    assert driver._best_primary(triage_only) == 0.60

    survivor = Node(id="b", parent_id=None, code_path="x",
                    partial_scores=[0.60], local_best_score=0.62)
    assert driver._best_primary(survivor) == 0.62

    unrun = Node(id="c", parent_id=None, code_path="x")
    assert driver._best_primary(unrun) == float("-inf")


def test_progress_file_records_every_iteration(mocked_driver, tmp_path):
    progress = tmp_path / "progress.json"
    result = driver.run(max_iters=3, verbose=False, progress_path=str(progress))

    data = json.loads(progress.read_text())
    iters = data["iterations"]
    assert iters, "expected at least one iteration recorded"
    assert data["iters_completed"] == len(iters)
    assert [r["iter"] for r in iters] == list(range(1, len(iters) + 1))
    assert data["global_best"] == result["global_best"]
    assert data["history"] == result["history"]

    scored = [r for r in iters if r["iter_primary"] is not None]
    assert scored, "mocked run should score at least one candidate"
    for r in scored:
        primaries = [c["primary"] for c in r["candidates"]
                     if c["primary"] is not None]
        assert r["iter_primary"] == max(primaries)
        assert r["delta_vs_global"] == r["iter_primary"] - r["global_best"]

    # Unscored candidates are still recorded, with a null primary — that's what
    # explains a null iter_primary when nothing reaches codegen.execute.
    for r in iters:
        assert len(r["candidates"]) == r["n_candidates"]
        assert sum(c["primary"] is not None for c in r["candidates"]) == r["n_scored"]


def test_unscored_candidates_are_recorded_with_null_primary(mocked_driver, tmp_path,
                                                            monkeypatch):
    """Every candidate dies before execute -> iter_primary null, reason visible."""
    monkeypatch.setattr(mock_codegen, "pre_execution_gate",
                        lambda diff: {"pass": False, "reasons": ["forced"]})
    progress = tmp_path / "progress.json"
    driver.run(max_iters=2, verbose=False, progress_path=str(progress))

    iters = json.loads(progress.read_text())["iterations"]
    blocked = [r for r in iters if r["n_candidates"]]
    assert blocked, "expected at least one iteration with candidates"
    for r in blocked:
        assert r["iter_primary"] is None
        assert r["n_scored"] == 0
        assert all(c["primary"] is None for c in r["candidates"])
        # Gate block -> failed_implementation; a repeat of an already-tried
        # diff is killed earlier by the run-wide diff-hash dedup.
        assert all(c["evidence_type"] in ("failed_implementation",
                                          "refuted_under_context")
                   for c in r["candidates"]), r["candidates"]


def test_progress_path_none_writes_nothing(mocked_driver, tmp_path):
    progress = tmp_path / "progress.json"
    driver.run(max_iters=2, verbose=False, progress_path=None)
    assert not progress.exists()

def test_measure_root_populates_real_per_user(mocked_driver, tmp_path):
    root = driver._new_root()
    assert root.per_user_by_seed == {}, "a fresh root must carry no fabricated data"

    counters = Counters()
    ok = driver._measure_root(root, counters, cache_path=str(tmp_path / "root.json"),
                              verbose=False)
    assert ok
    assert sorted(root.per_user_by_seed) == list(driver.ROOT_SEEDS)
    assert all(v for v in root.per_user_by_seed.values())
    assert root.seeds_run == list(driver.ROOT_SEEDS)
    # Score comes from the measurement, not the published fallback constant.
    assert root.local_best_score != driver.FALLBACK_ROOT_PRIMARY
    assert counters.scorer_queries["valid_search"] == len(driver.ROOT_SEEDS)


def test_measure_root_reuses_cache_without_rerunning(mocked_driver, tmp_path):
    cache = str(tmp_path / "root.json")
    first = driver._new_root()
    driver._measure_root(first, Counters(), cache_path=cache, verbose=False)

    second, counters = driver._new_root(), Counters()
    assert driver._measure_root(second, counters, cache_path=cache, verbose=False)
    assert counters.scorer_queries["valid_search"] == 0, "cache hit must not re-run"
    assert second.per_user_by_seed == first.per_user_by_seed
    assert second.local_best_score == first.local_best_score


def test_measure_root_reports_failure_when_no_per_user(mocked_driver, tmp_path,
                                                      monkeypatch):
    """A candidate that doesn't emit per_user leaves the tree unable to grow."""
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda *a, **kw: {"status": "ok", "logs": "",
                                          "metrics": {"primary": 0.6}})
    root = driver._new_root()
    assert not driver._measure_root(root, Counters(), cache_path=None, verbose=False)


def test_measured_root_pairs_with_candidate_users(mocked_driver, tmp_path):
    """Regression guard for the bug that kept the search tree flat.

    The root used to be seeded with a synthetic {u0..u9: 0.5946}, which shares no
    user ids with a real candidate, so every paired delta list was empty and
    should_continue_locally could never pass.
    """
    root = driver._new_root()
    driver._measure_root(root, Counters(), cache_path=None, verbose=False)

    cand = Node(id="c", parent_id=root.id, code_path="cand/baseline.py")
    for seed in driver.ROOT_SEEDS:
        r = mock_codegen.execute(cand.code_path, seed=seed, split="valid_search",
                                 wallclock_cap_seconds=60)
        cand.per_user_by_seed[seed] = r["metrics"]["per_user"]

    deltas = paired_user_deltas(cand.per_user_by_seed, root.per_user_by_seed)
    assert deltas, "candidate and measured root must share user ids"
    assert len(deltas) == len(driver.ROOT_SEEDS) * len(root.per_user_by_seed[0])

    synthetic = {s: {f"synthetic{i}": 0.5946 for i in range(10)}
                 for s in driver.ROOT_SEEDS}
    assert paired_user_deltas(cand.per_user_by_seed, synthetic) == [], \
        "the old synthetic root shared no users — this is what was broken"


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