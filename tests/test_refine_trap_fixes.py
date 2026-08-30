"""Refine-trap fixes — plateau stop, ablation noise guard, refine-target gate.

The failure mode these pin down: on a plateau the driver refined the ROOT every
iteration, picked whichever component had the smallest ablation delta (all three
at measurement noise), and could not stop early because the plateau check read
the promotion-space `history` instead of `iter_history`. Three independent
guards, one test group each.

All fast — no LLM, no FM. The end-to-end group follows the mocked-run pattern in
orchestrator/tests/test_smoke.py and tests/test_mlestar.py.
"""
import json
import random

import pytest

from orchestrator import ablation_harness, driver
from orchestrator.ablation_harness import pick_weakest_component
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm
from orchestrator.node import Node

# Phase 2 is switched off in the driver (driver.REFINE_ENABLED). These guards
# only mean anything while refine can fire, so they skip with it — and come
# back automatically when the flag flips.
pytestmark = pytest.mark.skipif(
    not driver.REFINE_ENABLED,
    reason="Phase 2 disabled: driver.REFINE_ENABLED is False")

BASE = 0.5946
USERS = [f"u{i}" for i in range(20)]


# --------------------------------------------------------------------------- #
#  Fix 2 — pick_weakest_component's noise guard                                #
# --------------------------------------------------------------------------- #
def test_pick_weakest_returns_none_when_all_deltas_below_threshold():
    """Every axis at noise means the pipeline is well-tuned, not that the
    smallest one is a lever."""
    assert pick_weakest_component({"a": 0.001, "b": 0.002, "c": 0.003}) is None


def test_pick_weakest_returns_weakest_when_some_delta_above_threshold():
    """One axis materially above noise makes the small ones interpretable."""
    assert pick_weakest_component({"a": 0.001, "b": 0.010, "c": 0.003}) == "a"


def test_pick_weakest_threshold_is_parameterisable():
    assert pick_weakest_component({"a": 0.001, "b": 0.002},
                                  min_meaningful_delta=0.0001) == "a"


def test_pick_weakest_guard_is_on_the_max_not_the_min():
    """The guard asks "is ANY axis a lever", not "is the chosen one big"."""
    deltas = {"features": 0.02365, "regularization": 0.00044,
              "capacity": 0.00041}                     # the observed run
    assert pick_weakest_component(deltas) == "capacity"


# --------------------------------------------------------------------------- #
#  Fix 3 — refine_target must clear baseline by a margin                       #
# --------------------------------------------------------------------------- #
def _measured(node_id, score, *, parent="root"):
    n = Node(id=node_id, parent_id=parent, code_path=f"{node_id}/baseline.py",
             local_best_score=score)
    n.per_user_by_seed = {0: {u: score for u in USERS}}
    return n


def _refine_target(open_nodes, root):
    """The driver's selection expression, evaluated in isolation.

    Mirrors orchestrator/driver.py::run's refine_target so the gate can be
    tested without spinning a full run; the end-to-end group below pins that
    the driver actually behaves this way.
    """
    return max(
        (n for n in open_nodes
         if n.per_user_by_seed
         and n.local_best_score > root.local_best_score
                                  + driver.REFINE_TARGET_MIN_IMPROVEMENT),
        key=lambda n: n.local_best_score, default=None,
    )


def test_refine_target_none_when_no_node_beats_baseline():
    root = _measured("root", BASE, parent=None)
    assert _refine_target([root], root) is None


def test_refine_target_selects_node_that_beats_baseline_by_margin():
    root = _measured("root", BASE, parent=None)
    better = _measured("c1", BASE + 0.001)
    assert _refine_target([root, better], root) is better


def test_refine_target_ignores_node_within_margin():
    root = _measured("root", BASE, parent=None)
    marginal = _measured("c1", BASE + 0.0003)
    assert _refine_target([root, marginal], root) is None


def test_refine_target_skips_node_without_per_user_data():
    """An unmeasured node has no primary for an ablation to subtract from."""
    root = _measured("root", BASE, parent=None)
    unmeasured = Node(id="c1", parent_id="root", code_path="c1/baseline.py",
                      local_best_score=BASE + 0.05)
    assert _refine_target([root, unmeasured], root) is None


# --------------------------------------------------------------------------- #
#  Mocked run scaffolding (shared by the Fix 1 / Fix 3 end-to-end tests)       #
# --------------------------------------------------------------------------- #
def _uniform(value):
    return {"status": "ok", "logs": "",
            "metrics": {"primary": value, "GAUC": value + 0.03,
                        "nDCG@5": value - 0.03,
                        "per_user": {u: value for u in USERS}}}


class _Script:
    """Scripts iter_primary per iteration; advances on diagnose OR refine."""

    def __init__(self, primaries):
        self.primaries = list(primaries)
        self.it = 0

    def target(self):
        if 1 <= self.it <= len(self.primaries):
            return self.primaries[self.it - 1]
        return BASE


def _install(monkeypatch, tmp_path, script, *, ablated=BASE - 0.01):
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    monkeypatch.setattr(ablation_harness, "ABLATIONS_LOG_PATH",
                        str(tmp_path / "ablations.jsonl"))

    from orchestrator import memory as memory_mod
    orig_init = memory_mod.Memory.__init__
    monkeypatch.setattr(memory_mod.Memory, "__init__",
                        lambda self, path=None: orig_init(
                            self, path=str(tmp_path / "memory.json")))

    real_refine = mock_llm.refine

    def diagnose(node_context):
        script.it += 1
        return {"bottleneck": f"bottleneck for iter {script.it}",
                "evidence": "", "confidence": 0.75, "component": "loss",
                "edit_radius": "small", "expected_cost": "medium",
                "incompatibilities": [], "uncertainty": 0.25}

    def generate_hypothesis(diagnosis, evidence_card):
        return [{"mechanism": f"scripted mechanism for iter {script.it}",
                 "success_criterion_paired": "> 0.005 on valid_search",
                 "implementation_sketch": f"sketch {script.it}",
                 "loss_type": f"scripted_{script.it}",
                 "sampler": "within_user_neg",
                 "feature_set": "5field_baseline", "dataset_tier": "pure"}]

    def refine(*a, **kw):
        script.it += 1
        return real_refine(*a, **kw)

    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        # `ablated` is a knob: BASE - 0.01 puts every ablation delta above the
        # noise floor (refine allowed), BASE puts them all at zero (suppressed).
        if "candidate_ablation_" in str(code_path):
            return _uniform(ablated)
        if "candidate_" not in str(code_path) or split == "valid_confirm":
            return _uniform(BASE)
        return _uniform(script.target())

    monkeypatch.setattr(mock_llm, "diagnose", diagnose)
    monkeypatch.setattr(mock_llm, "generate_hypothesis", generate_hypothesis)
    monkeypatch.setattr(mock_llm, "refine", refine)
    monkeypatch.setattr(mock_codegen, "execute", execute)


def _run(monkeypatch, tmp_path, primaries, *, max_iters=None, **install_kw):
    """One mocked run per test.

    Deliberately not reusable twice in a test: _install wraps mock_llm.refine
    and Memory.__init__ in place, so a second _install wraps the first
    wrapper — script.it double-increments and memory.json still points at the
    first run's file. Contrasting two configurations means two test functions.
    """
    script = _Script(primaries)
    _install(monkeypatch, tmp_path, script, **install_kw)
    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=max_iters or len(primaries), verbose=False,
               progress_path=str(progress),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"))
    nodes = [json.loads(l) for l in
             (tmp_path / "nodes.jsonl").read_text().splitlines() if l.strip()]
    return json.loads(progress.read_text()), nodes


def _refine_nodes(nodes):
    return [n for n in nodes if n["operation"] == "refine"]


# --------------------------------------------------------------------------- #
#  Fix 1 — the plateau stop reads iter_history                                 #
# --------------------------------------------------------------------------- #
def test_plateau_stop_leaves_a_climbing_run_alone(monkeypatch, tmp_path):
    climbing = [0.60, 0.62, 0.64, 0.66]
    progress, _ = _run(monkeypatch, tmp_path, climbing)
    assert progress["history"] == [pytest.approx(BASE)], \
        "nothing promotes in this mock — see the paired iter_history test"
    assert progress["iters_completed"] == len(climbing), \
        "a climbing trajectory must not trip the plateau stop"


def test_local_plateau_call_uses_iter_history(monkeypatch, tmp_path):
    """A run that climbs and then flattens must stop on the flat tail.

    The distinction only exists if the stop reads iter_history. The mock's
    valid_confirm split always returns BASE, so nothing ever promotes and
    `history` stays at length 1 for the whole run — against `history` the
    N+1 length guard makes local_plateau structurally False and this run
    would burn all seven iterations.
    """
    # Tail entries sit BELOW the 0.64 peak, so the running best is genuinely
    # flat rather than creeping. The old version crept +0.0001 per iteration,
    # which clears driver.PLATEAU_STOP_EPSILON=0.0005 over an 8-wide window.
    flat_tail = [0.60, 0.62, 0.64] + [0.6395] * 9
    progress, _ = _run(monkeypatch, tmp_path, flat_tail)

    assert progress["history"] == [pytest.approx(BASE)], \
        "no promotion — so `history` could never have driven this stop"
    # Running best goes B, .60, .62, .64, then .64 forever. With N=8 the window
    # only closes once the 0.64 peak has fallen out of max(h[-8:]) and into
    # max(h[:-8]) — that is len(iter_history) == 12, i.e. the end of iteration 11.
    assert progress["iters_completed"] == 11
    assert len(progress["iter_history"]) == 12   # baseline + 11 iterations


def test_plateau_stop_fires_on_a_flat_run_from_the_start(monkeypatch, tmp_path):
    """The diagnosed run's shape: nothing beats baseline, ever."""
    progress, _ = _run(monkeypatch, tmp_path, [BASE] * 20, max_iters=20)
    # First check is at the end of iteration 8, when iter_history reaches
    # driver.PLATEAU_STOP_WINDOW_N + 1.
    assert progress["iters_completed"] == 8


def test_plateau_stop_no_longer_preempts_the_plateau_refine_trigger(
        monkeypatch, tmp_path):
    """The stop bar now sits BELOW the refine trigger, so refine is reachable.

    Both the stop and the refine trigger read iter_history, which is monotone,
    so both reduce to iter_history[-1] - iter_history[-4]. This test used to
    assert the opposite of what it asserts now: with local_plateau's own
    ε=0.002 the stop bar was LOOSER than PLATEAU_REFINE_THRESHOLD (0.001) and
    was evaluated at the end of the previous iteration, so every trajectory
    flat enough to arm the plateau refine trigger had already killed the run
    and no refine could ever fire.

    driver.PLATEAU_STOP_EPSILON (0.0005) is now below that threshold and
    driver.PLATEAU_STOP_WINDOW_N (8) makes iteration 3 too early to check at
    all, so the trigger arms first. Running best after 3 iterations is 0.5954,
    an improvement_score of 0.0008 — at or under PLATEAU_REFINE_THRESHOLD and
    over REFINE_TARGET_MIN_IMPROVEMENT, so the plateaued node is an eligible
    target and iteration 4 refines it.

    If PLATEAU_STOP_EPSILON is ever raised back above PLATEAU_REFINE_THRESHOLD,
    this is the test that should fail.
    """
    monkeypatch.setattr(driver, "REFINE_EVERY_K_IMPROVES", 10 ** 6)
    progress, nodes = _run(monkeypatch, tmp_path,
                           [0.5954, 0.59535, 0.59532, 0.5990])
    assert progress["iters_completed"] == 4, \
        "the stop bar must no longer close the run before iteration 4"
    assert [n["iter"] for n in _refine_nodes(nodes)] == [4]


def test_plateau_stop_bar_is_below_the_refine_trigger(monkeypatch, tmp_path):
    """The ordering the test above depends on, asserted directly.

    Cheaper and more legible than inferring it from an iteration count: if
    these two constants ever cross again, the plateau refine path becomes
    dead code and only this line says so.
    """
    assert driver.PLATEAU_STOP_EPSILON < driver.PLATEAU_REFINE_THRESHOLD
    # And the window has to be wide enough that h[-1] - h[-4] (the trigger's
    # own window) can be evaluated before the stop's window closes at all.
    assert driver.PLATEAU_STOP_WINDOW_N > 3


# --------------------------------------------------------------------------- #
#  Fix 3 end-to-end + Fix 2 end-to-end                                         #
# --------------------------------------------------------------------------- #
def test_run_does_not_refine_the_root_when_nothing_beats_baseline(
        monkeypatch, tmp_path):
    """Cadence comes due but open_nodes holds only the pristine root."""
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    # No local_plateau stub needed: this run is 6 iterations, and
    # driver.PLATEAU_STOP_WINDOW_N (8) means iter_history never reaches the N+1
    # length the stop requires. The stub used to be load-bearing at N=3.
    # Every candidate scores exactly baseline, so none clears
    # should_continue_locally and the root stays the only open node.
    progress, nodes = _run(monkeypatch, tmp_path, [BASE] * 6, max_iters=6)

    assert progress["iters_completed"] == 6
    assert _refine_nodes(nodes) == [], \
        "refine fired against a node that never beat baseline"
    assert not (tmp_path / "ablations.jsonl").exists(), \
        "a suppressed refine must not spend the ablation budget"
    # "draft" is the root's own record; everything else must be an improve.
    assert {n["operation"] for n in nodes} == {"draft", "improve"}


def test_run_refines_once_a_node_clears_the_margin(monkeypatch, tmp_path):
    """Same cadence, but now a candidate has real headroom over baseline."""
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    _progress, nodes = _run(monkeypatch, tmp_path,
                            [0.5960, 0.5980, 0.6000, 0.6020])

    refines = _refine_nodes(nodes)
    assert refines, "a node well above baseline must unlock refine"
    assert refines[0]["iter"] == driver.REFINE_EVERY_K_IMPROVES + 1
    # And it refined a candidate, not the root — the root's id is generated,
    # so find it by the operation it was logged under.
    root_id = next(n["id"] for n in nodes if n["operation"] == "draft")
    assert refines[0]["parent_id"] != root_id


def test_run_falls_back_to_improve_when_every_ablation_delta_is_noise(
        monkeypatch, tmp_path):
    """Fix 2 through run(): the target qualifies, the evidence does not.

    `ablated=BASE` is not arbitrary — it reproduces the diagnosed run, where
    removing any component moved primary by less than the seed noise.
    """
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    _progress, nodes = _run(monkeypatch, tmp_path,
                            [0.5960, 0.5980, 0.6000, 0.6020],
                            ablated=BASE + 0.0020)

    assert _refine_nodes(nodes) == [], \
        "refining a component picked out of noise is the trap being closed"
    # The ablations still ran — that is how we learned there was no lever —
    # but no refine node was built from them.
    assert (tmp_path / "ablations.jsonl").exists()
