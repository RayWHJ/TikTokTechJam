"""Per-iteration schema of progress.json.

The old record named its champion field `global_best` and its delta
`delta_vs_global`, which invited the reading that they move every iteration —
on every run recorded so far they never did, because no candidate cleared the
paired-confirm promotion bar. These tests pin the renamed keys (`baseline`,
`curr_vs_baseline`) plus the two new columns that make a run readable at a
glance: `running_best` (the running max, monotone) and `improvement_score`
(the exact ε/N window delta `local_plateau` compares against).

Fully mocked: no LLM, no FM, no real scorer.
"""
import json
import random

import pytest

from orchestrator import ablation_harness, driver
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm

#: The repo's real cached root primary, so the numbers here read like a run.
BASE = 0.5946
USERS = [f"u{i}" for i in range(20)]

#: Sentinel for "this iteration's candidates all fail to execute".
FAIL = object()


def _uniform(value):
    """Per-user scores that are flat across users AND seeds.

    Flat is load-bearing: _scalar_primary means over seeds and the mock means
    over users, so a flat block makes iter_primary come out exactly `value` and
    the scripted trajectory is the one the log records.
    """
    per_user = {u: value for u in USERS}
    return {"status": "ok", "logs": "",
            "metrics": {"primary": value, "GAUC": value + 0.03,
                        "nDCG@5": value - 0.03, "per_user": per_user}}


class _Script:
    """Scripts iter_primary per iteration for a mocked run.

    The iteration number is tracked off llm.diagnose, which the driver calls
    exactly once at the top of each iteration — there is no other hook that
    tells codegen.execute which iteration it is running under.
    """

    def __init__(self, primaries):
        self.primaries = list(primaries)
        self.it = 0          # 0 while the root baseline is being measured

    def target(self):
        if 1 <= self.it <= len(self.primaries):
            return self.primaries[self.it - 1]
        return BASE


def _install(monkeypatch, tmp_path, script):
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    # nodes.jsonl and ablations.jsonl both default to the live
    # orchestrator/_state/ — a mocked run must never append there.
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    monkeypatch.setattr(ablation_harness, "ABLATIONS_LOG_PATH",
                        str(tmp_path / "ablations.jsonl"))

    # Phase 2's refine scheduler is off for these tests. The scripted iteration
    # counter advances on llm.diagnose, which a refine iteration doesn't call,
    # and a refine iteration replaces the scripted hypothesis entirely — so a
    # refine firing mid-script would silently shift every later iteration's
    # target. Refine's own scheduling is covered in test_mlestar.py; these
    # tests are about the log schema.
    monkeypatch.setattr(driver, "REFINE_EVERY_K_IMPROVES", 10**6)
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))

    from orchestrator import memory as memory_mod
    orig_init = memory_mod.Memory.__init__
    monkeypatch.setattr(memory_mod.Memory, "__init__",
                        lambda self, path=None: orig_init(
                            self, path=str(tmp_path / "memory.json")))

    def diagnose(node_context):
        script.it += 1
        return {"bottleneck": f"bottleneck for iter {script.it}",
                "evidence": "", "confidence": 0.75, "component": "loss",
                "edit_radius": "small", "expected_cost": "medium",
                "incompatibilities": [], "uncertainty": 0.25}

    def generate_hypothesis(diagnosis, evidence_card, tried=None, **kw):
        # loss_type and mechanism both carry the iteration index: the former
        # keeps _fingerprint unique so memory dedup doesn't swallow iteration 2
        # onward, the latter keeps the written diff unique so the run-wide
        # diff-hash dedup doesn't either.
        return [{"mechanism": f"scripted mechanism for iter {script.it}",
                 "success_criterion_paired": "> 0.005 on valid_search",
                 "implementation_sketch": f"sketch {script.it}",
                 "loss_type": f"scripted_{script.it}",
                 "sampler": "within_user_neg",
                 "feature_set": "5field_baseline", "dataset_tier": "pure"}]

    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None, **kw):
        is_candidate = "candidate_" in str(code_path)
        # Candidates score the baseline on valid_confirm, so the paired confirm
        # delta is 0 and nothing promotes. That keeps `baseline` pinned to
        # baseline_primary — which is what every real run has looked like — and
        # keeps the promotion ladder out of these assertions.
        if not is_candidate or split == "valid_confirm":
            return _uniform(BASE)
        target = script.target()
        if target is FAIL:
            return {"status": "error", "metrics": {}, "logs": "scripted failure"}
        return _uniform(target)

    monkeypatch.setattr(mock_llm, "diagnose", diagnose)
    monkeypatch.setattr(mock_llm, "generate_hypothesis", generate_hypothesis)
    monkeypatch.setattr(mock_codegen, "execute", execute)


def _run(monkeypatch, tmp_path, primaries, max_iters=None):
    """Drive run() with a scripted iter_primary per iteration; return the log."""
    script = _Script(primaries)
    _install(monkeypatch, tmp_path, script)
    random.seed(0)   # bootstrap_delta draws; pin it so runs are reproducible
    progress = tmp_path / "progress.json"
    driver.run(max_iters=len(primaries) if max_iters is None else max_iters,
               verbose=False, progress_path=str(progress),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"))
    return json.loads(progress.read_text())


def _iters(data, n):
    iters = data["iterations"]
    assert len(iters) == n, f"expected {n} iterations, got {len(iters)}"
    return iters


# --------------------------------------------------------------------------- #
#  Renames                                                                     #
# --------------------------------------------------------------------------- #
def test_iteration_record_uses_new_key_names(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [0.5960])
    rec = _iters(data, 1)[0]

    assert "baseline" in rec
    assert "curr_vs_baseline" in rec
    # Negative half: a clean cutover, no compatibility aliases.
    assert "global_best" not in rec
    assert "delta_vs_global" not in rec

    assert rec["baseline"] == pytest.approx(BASE, abs=1e-9)
    assert rec["curr_vs_baseline"] == pytest.approx(
        rec["iter_primary"] - BASE, abs=1e-9)


# --------------------------------------------------------------------------- #
#  iter_history                                                                #
# --------------------------------------------------------------------------- #
def test_iter_history_seeded_with_baseline(monkeypatch, tmp_path):
    """Index 0 is the baseline, not the first iteration.

    Asserted off the shortest run that produces a file at all: the progress
    write lives inside _record_iteration, so a literal max_iters=0 run never
    writes one — checked below so the reason is documented rather than
    rediscovered.
    """
    data = _run(monkeypatch, tmp_path, [0.5960])
    assert data["iter_history"][0] == pytest.approx(
        data["baseline_primary"], abs=1e-12)

    progress = tmp_path / "zero.json"
    driver.run(max_iters=0, verbose=False, progress_path=str(progress),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"))
    assert not progress.exists(), \
        "a 0-iteration run records nothing — the write is per-iteration"


def test_running_best_carries_prev_on_none(monkeypatch, tmp_path):
    """An iteration where nothing scored is "no improvement", not "no data"."""
    data = _run(monkeypatch, tmp_path, [FAIL])
    rec = _iters(data, 1)[0]

    assert rec["iter_primary"] is None
    assert rec["n_scored"] == 0
    assert rec["n_candidates"] > 0, "need candidates that failed, not zero candidates"
    assert all(c["evidence_type"] == "failed_implementation"
               for c in rec["candidates"]), rec["candidates"]
    assert rec["running_best"] == pytest.approx(BASE, abs=1e-9)
    assert data["iter_history"] == [pytest.approx(BASE, abs=1e-9)] * 2


def test_running_best_takes_running_max_and_stays_monotone(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [0.60, 0.58])
    first, second = _iters(data, 2)

    assert first["iter_primary"] == pytest.approx(0.60, abs=1e-9)
    assert second["iter_primary"] == pytest.approx(0.58, abs=1e-9)
    assert first["running_best"] == pytest.approx(0.60, abs=1e-9)
    # The point of the column: a worse iteration does not pull the best down.
    assert second["running_best"] == pytest.approx(0.60, abs=1e-9)


def test_iter_history_length_matches_iters_completed_plus_one(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [0.5960, 0.5970, 0.5980])
    _iters(data, 3)
    assert data["iters_completed"] == 3
    assert len(data["iter_history"]) == 4


def test_top_level_iter_history_matches_per_iteration_running_best(monkeypatch,
                                                                  tmp_path):
    data = _run(monkeypatch, tmp_path, [0.5960, 0.5970, 0.5975, 0.5990])
    iters = _iters(data, 4)
    assert [r["running_best"] for r in iters] == data["iter_history"][1:]


# --------------------------------------------------------------------------- #
#  improvement_score — the ε/N window the plateau rule reads                   #
# --------------------------------------------------------------------------- #
def test_improvement_score_null_before_iter_3(monkeypatch, tmp_path):
    """Iterations 1 and 2 have fewer than three prior iterations of history."""
    data = _run(monkeypatch, tmp_path, [0.5960, 0.5970])
    for rec in _iters(data, 2):
        assert rec["improvement_score"] is None


def test_improvement_score_matches_window_math(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [0.5960, 0.5970, 0.5975, 0.5990])
    iters = _iters(data, 4)
    hist = data["iter_history"]

    # Hand-checked: baseline 0.5946, so hist == the scripted trajectory.
    assert hist == pytest.approx([BASE, 0.5960, 0.5970, 0.5975, 0.5990],
                                 abs=1e-9)
    assert iters[0]["improvement_score"] is None
    assert iters[1]["improvement_score"] is None
    assert iters[2]["improvement_score"] == hist[3] - hist[0]
    assert iters[3]["improvement_score"] == hist[4] - hist[1]
    # 0.5975 - 0.5946 and 0.5990 - 0.5960.
    assert iters[2]["improvement_score"] == pytest.approx(0.0029, abs=1e-9)
    assert iters[3]["improvement_score"] == pytest.approx(0.0030, abs=1e-9)
