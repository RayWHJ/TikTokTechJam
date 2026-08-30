"""Phase 2 — MLE-STAR ablation-guided component refinement.

Three layers, cheapest first:

1. The registry and the weakest-component pick — pure functions.
2. The ablation harness against a stub codegen — no LLM, no FM, caching and
   the JSONL log.
3. The two refine triggers end-to-end through run(). Each trigger is isolated
   by disabling the other one (a huge K for the plateau test, a -inf threshold
   for the cadence test), so a passing test names which trigger fired rather
   than just "refine happened".

No real FM runs, no real API calls.
"""
import json
import random

import pytest

from llm_calls import personas
from llm_calls.refine import _build_prompt as build_refine_prompt
from orchestrator import ablation_harness, convergence, driver
from orchestrator.ablation_harness import (pick_weakest_component,
                                           run_ablations)
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm
from orchestrator.node import Node
from codegen.ablations import ABLATIONS

# Phase 2 is switched off in the driver (driver.REFINE_ENABLED) because the
# search regressed with refine in the loop. The code under test is still here
# and still correct; these tests come back automatically when the flag flips.
pytestmark = pytest.mark.skipif(
    not driver.REFINE_ENABLED,
    reason="Phase 2 disabled: driver.REFINE_ENABLED is False")

BASE = 0.5946
USERS = [f"u{i}" for i in range(20)]


# --------------------------------------------------------------------------- #
#  1. Registry + weakest-component pick                                        #
# --------------------------------------------------------------------------- #
def test_registry_has_three_components():
    assert {"features", "regularization", "capacity"} <= set(ABLATIONS)
    for name, abl in ABLATIONS.items():
        assert abl.name == name, "registry key must match Ablation.name"
        assert abl.file in ("data.py", "baseline.py")


def test_pick_weakest_by_smallest_delta():
    deltas = {"features": 0.01, "regularization": 0.001, "capacity": 0.005}
    assert pick_weakest_component(deltas) == "regularization"


def test_pick_weakest_deterministic_on_ties():
    # Insertion order deliberately reversed: the tie must break on name, not
    # on dict order, or nodes.jsonl stops being reproducible.
    # Values sit above the 0.005 noise floor on purpose — below it the noise
    # guard returns None and the tie-break never gets a chance to run.
    assert pick_weakest_component({"zebra": 0.010, "capacity": 0.010}) == "capacity"
    assert pick_weakest_component({"capacity": 0.010, "zebra": 0.010}) == "capacity"


def test_pick_weakest_returns_none_on_empty():
    assert pick_weakest_component({}) is None


# --------------------------------------------------------------------------- #
#  2. Ablation harness                                                         #
# --------------------------------------------------------------------------- #
class _StubCodegen:
    """Counts execute calls so caching is observable."""

    def __init__(self, primary=0.5900):
        self.execute_calls = 0
        self.primary = primary

    def write_fix(self, hypothesis, target_component, root="."):
        return f"diff for {target_component}: {hypothesis['mechanism'][:20]}"

    def execute(self, code_path, seed, split, wallclock_cap_seconds,
                root=None, data_dir=None):
        self.execute_calls += 1
        return {"status": "ok", "logs": "",
                "metrics": {"primary": self.primary, "per_user": {}}}


def _stage_fn(tmp_path):
    def stage(diff, root, candidate_id):
        d = tmp_path / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    return stage


def _run_ablations(tmp_path, stub, node=None):
    node = node or Node(id="n1", parent_id=None, code_path="baseline.py",
                        code_dir=".")
    return run_ablations(node, BASE,
                         codegen_mod=stub,
                         stage_fn=_stage_fn(tmp_path),
                         apply_diff_fn=lambda diff, target_dir: True,
                         data_dir=str(tmp_path / "data"),
                         log_path=str(tmp_path / "ablations.jsonl")), node


def test_run_ablations_caches_across_calls(tmp_path):
    stub = _StubCodegen()
    first, node = _run_ablations(tmp_path, stub)
    assert set(first) == set(ABLATIONS)
    assert stub.execute_calls == len(ABLATIONS)

    after_first = stub.execute_calls
    second, _ = _run_ablations(tmp_path, stub, node=node)
    assert stub.execute_calls == after_first, \
        "a second pass over the same node must not re-run a single ablation"
    assert second == first


def test_ablations_jsonl_written_and_parseable(tmp_path):
    stub = _StubCodegen(primary=0.5900)
    deltas, node = _run_ablations(tmp_path, stub)

    log = tmp_path / "ablations.jsonl"
    assert log.exists()
    lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(lines) == len(ABLATIONS)
    for r in lines:
        if "error" in r:
            continue
        assert {"node_id", "component", "delta"} <= set(r)
        assert r["node_id"] == node.id
        # delta = node_primary - ablated_primary
        assert r["delta"] == pytest.approx(BASE - 0.5900, abs=1e-12)
    assert deltas["features"] == pytest.approx(BASE - 0.5900, abs=1e-12)


def test_run_ablations_skips_a_component_that_fails_to_stage(tmp_path):
    """A failed ablation is a bug in the strategy, not evidence — skip it."""
    stub = _StubCodegen()
    node = Node(id="n2", parent_id=None, code_path="baseline.py", code_dir=".")
    deltas = run_ablations(
        node, BASE, codegen_mod=stub,
        stage_fn=lambda diff, root, candidate_id: None,   # staging always fails
        apply_diff_fn=lambda diff, target_dir: True,
        data_dir=str(tmp_path / "data"),
        log_path=str(tmp_path / "ablations.jsonl"))
    assert deltas == {}
    assert stub.execute_calls == 0
    assert pick_weakest_component(deltas) is None


# --------------------------------------------------------------------------- #
#  3. Trigger logic — pure function                                            #
# --------------------------------------------------------------------------- #
def test_triggers_are_both_off_before_four_entries():
    cadence, plateau, recent = driver._refine_triggers([BASE, 0.60, 0.61], 0)
    assert (cadence, plateau, recent) == (False, False, None)


def test_cadence_trigger_needs_k_improves():
    hist = [BASE, 0.60, 0.62, 0.64]
    assert not driver._refine_triggers(hist, driver.REFINE_EVERY_K_IMPROVES - 1)[0]
    assert driver._refine_triggers(hist, driver.REFINE_EVERY_K_IMPROVES)[0]


def test_plateau_trigger_is_independent_of_improve_count():
    # Healthy trajectory: plateau off even at the cadence boundary.
    cadence, plateau, recent = driver._refine_triggers([BASE, 0.60, 0.62, 0.64], 0)
    assert (cadence, plateau) == (False, False)
    assert recent == pytest.approx(0.64 - BASE)

    # Flat trajectory: plateau on with zero improves accumulated.
    hist = [BASE, BASE + 0.0002, BASE + 0.0003, BASE + 0.0005]
    cadence, plateau, recent = driver._refine_triggers(hist, 0)
    assert cadence is False
    assert plateau is True
    assert recent == pytest.approx(0.0005, abs=1e-9)


# --------------------------------------------------------------------------- #
#  3b. Triggers end-to-end through run()                                       #
# --------------------------------------------------------------------------- #
def _uniform(value):
    per_user = {u: value for u in USERS}
    return {"status": "ok", "logs": "",
            "metrics": {"primary": value, "GAUC": value + 0.03,
                        "nDCG@5": value - 0.03, "per_user": per_user}}


class _Script:
    """Scripts iter_primary per iteration for a mocked run.

    The counter advances on BOTH llm.diagnose (improve iterations) and
    llm.refine (refine iterations) — exactly one of the two runs per
    iteration, so the index stays aligned no matter which path fires.
    """

    def __init__(self, primaries):
        self.primaries = list(primaries)
        self.it = 0

    def target(self):
        if 1 <= self.it <= len(self.primaries):
            return self.primaries[self.it - 1]
        return BASE


def _install(monkeypatch, tmp_path, script, spy=None):
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
        if spy is not None:
            spy.append(node_context)
        return {"bottleneck": f"bottleneck for iter {script.it}",
                "evidence": "", "confidence": 0.75, "component": "loss",
                "edit_radius": "small", "expected_cost": "medium",
                "incompatibilities": [], "uncertainty": 0.25}

    def generate_hypothesis(diagnosis, evidence_card):
        # Iteration index in both loss_type and mechanism: the former keeps
        # _fingerprint unique against memory dedup, the latter keeps the
        # written diff unique against the run-wide diff-hash dedup.
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
        is_candidate = "candidate_" in str(code_path)
        # Ablation runs stage under candidate_ablation_*. They all return the
        # same value, so the weakest pick stays a deterministic alphabetical
        # tie rather than a function of mock RNG — but that value has to sit
        # far enough below the node's primary that every delta clears
        # pick_weakest_component's noise floor, or the guard suppresses the
        # refine and these trigger tests stop testing triggers.
        if "candidate_ablation_" in str(code_path):
            return _uniform(BASE - 0.01)
        if not is_candidate or split == "valid_confirm":
            return _uniform(BASE)
        return _uniform(script.target())

    monkeypatch.setattr(mock_llm, "diagnose", diagnose)
    monkeypatch.setattr(mock_llm, "generate_hypothesis", generate_hypothesis)
    monkeypatch.setattr(mock_llm, "refine", refine)
    monkeypatch.setattr(mock_codegen, "execute", execute)


def _run(monkeypatch, tmp_path, primaries, spy=None):
    script = _Script(primaries)
    _install(monkeypatch, tmp_path, script, spy=spy)
    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=len(primaries), verbose=False,
               progress_path=str(progress),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"))
    nodes = [json.loads(l) for l in
             (tmp_path / "nodes.jsonl").read_text().splitlines() if l.strip()]
    return json.loads(progress.read_text()), nodes


def _refine_nodes(nodes):
    return [n for n in nodes if n["operation"] == "refine"]


def _tighter_stop_bar(epsilon):
    """A stricter plateau STOP, so a plateau REFINE trigger can be observed.

    The stop condition and the refine trigger now read the same signal.
    iter_history is monotone (running best), so local_plateau's
    max(h[-3:]) - max(h[:-3]) reduces to exactly h[-1] - h[-4] — the quantity
    _refine_triggers compares against PLATEAU_REFINE_THRESHOLD. Since the stop
    bar (ε=0.002) is looser than that threshold (0.001) AND is evaluated at the
    end of the previous iteration, the stop always pre-empts the trigger in a
    real run — see test_plateau_stop_preempts_plateau_refine_trigger, which
    pins that interaction. Tightening ε isolates the trigger under test.
    """
    return lambda hist: convergence.local_plateau(hist, epsilon=epsilon)


def test_cadence_trigger_fires_after_k_improves(monkeypatch, tmp_path):
    """Healthy improvements, so only the cadence trigger can fire."""
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    # +0.002 per iteration: comfortably above any plateau bar.
    progress, nodes = _run(monkeypatch, tmp_path,
                           [0.5960, 0.5980, 0.6000, 0.6020])

    refines = _refine_nodes(nodes)
    assert refines, ("refine never fired after "
                     f"{driver.REFINE_EVERY_K_IMPROVES} improves; "
                     f"nodes: {[(n['iter'], n['operation']) for n in nodes]}")
    # K improves land in iterations 1..K, so the refine is iteration K+1.
    assert refines[0]["iter"] == driver.REFINE_EVERY_K_IMPROVES + 1
    assert refines[0]["diagnosis"]["component"] in ABLATIONS
    # Cadence, not plateau: compared against the REAL threshold (0.001), not
    # the -inf this test patched in, the trajectory was still climbing — so
    # nothing but the improve count could have triggered this refine.
    assert refines[0]["diagnosis"]["improvement_score"] == pytest.approx(
        0.6000 - BASE, abs=1e-9)
    assert refines[0]["diagnosis"]["improvement_score"] > 0.001


def test_plateau_trigger_fires_below_threshold(monkeypatch, tmp_path):
    """Cadence disabled, so only the plateau trigger can fire."""
    monkeypatch.setattr(driver, "REFINE_EVERY_K_IMPROVES", 10 ** 6)
    monkeypatch.setattr(driver, "local_plateau", _tighter_stop_bar(0.0005))
    # Running best after 3 iterations is 0.5954, so the ε/N window at the top
    # of iteration 4 is 0.5954 - 0.5946 = 0.0008 <= PLATEAU_REFINE_THRESHOLD.
    # 0.0008 is also above REFINE_TARGET_MIN_IMPROVEMENT (0.0005), so the
    # plateaued node is still eligible as a refine target.
    progress, nodes = _run(monkeypatch, tmp_path,
                           [0.5954, 0.59535, 0.59532, 0.5990])

    assert progress["iter_history"][:4] == pytest.approx(
        [BASE, 0.5954, 0.5954, 0.5954], abs=1e-9)
    refines = _refine_nodes(nodes)
    assert refines, ("plateau trigger never fired; iter_history="
                     f"{progress['iter_history']}")
    assert refines[0]["iter"] == 4


def test_refine_node_carries_improvement_score_in_diagnosis(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "REFINE_EVERY_K_IMPROVES", 10 ** 6)
    monkeypatch.setattr(driver, "local_plateau", _tighter_stop_bar(0.0005))
    _progress, nodes = _run(monkeypatch, tmp_path,
                            [0.5954, 0.59535, 0.59532, 0.5990])
    refine_node = _refine_nodes(nodes)[0]
    assert refine_node["diagnosis"]["improvement_score"] == pytest.approx(
        0.0008, abs=1e-9)
    assert refine_node["diagnosis"]["ablation_deltas"], \
        "the refine node must record the evidence that chose its component"


def test_refine_runs_ablations_and_logs_them(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    _run(monkeypatch, tmp_path, [0.5960, 0.5980, 0.6000, 0.6020])

    log = tmp_path / "ablations.jsonl"
    assert log.exists(), "a refine iteration must leave an ablation record"
    recorded = {json.loads(l)["component"]
                for l in log.read_text().splitlines() if l.strip()}
    assert recorded == set(ABLATIONS)


def test_refine_does_not_persist_scheduling_state(monkeypatch, tmp_path):
    """improves_since_refine is private to the run."""
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    progress, nodes = _run(monkeypatch, tmp_path,
                           [0.5960, 0.5980, 0.6000, 0.6020])
    assert "improves_since_refine" not in progress
    for n in nodes:
        assert "improves_since_refine" not in n
    for rec in progress["iterations"]:
        assert "improves_since_refine" not in rec


# --------------------------------------------------------------------------- #
#  4. Trajectory signals reaching the prompts                                  #
# --------------------------------------------------------------------------- #
def test_diagnostician_context_includes_iter_history_and_improvement_score(
        monkeypatch, tmp_path):
    spy = []
    _run(monkeypatch, tmp_path, [0.5960, 0.5970], spy=spy)

    assert spy, "diagnose was never called"
    ctx = spy[-1]
    assert "history" in ctx, "the promotion ladder must stay for compatibility"
    assert "iter_history" in ctx
    assert "improvement_score" in ctx
    assert ctx["iter_history"][0] == pytest.approx(BASE, abs=1e-9)
    # Iteration 2 has fewer than three completed iterations behind it.
    assert ctx["improvement_score"] is None


def test_diagnostician_context_carries_cached_ablations(monkeypatch, tmp_path):
    """Ablations measured during a refine iteration are free to reuse."""
    monkeypatch.setattr(driver, "PLATEAU_REFINE_THRESHOLD", float("-inf"))
    spy = []
    _run(monkeypatch, tmp_path, [0.5960, 0.5980, 0.6000, 0.6020, 0.6040],
         spy=spy)
    # Iteration 4 refines the root and caches its ablations; iteration 5 is an
    # improve whose parent may or may not be that node, so accept either an
    # empty dict or the registry's components — but the key must be present.
    ctx = spy[-1]
    assert "ablations" in ctx
    if ctx["ablations"]:
        assert set(ctx["ablations"]) <= set(ABLATIONS)


def test_diagnostician_prompt_mentions_new_signals():
    assert "iter_history" in personas.DIAGNOSTICIAN_SYSTEM_PROMPT
    assert "improvement_score" in personas.DIAGNOSTICIAN_SYSTEM_PROMPT
    assert "ablations" in personas.DIAGNOSTICIAN_SYSTEM_PROMPT


def test_refiner_prompt_carries_plateau_signals():
    msg = build_refine_prompt(
        "regularization", "def step(self): ...",
        {"features": 0.01, "regularization": 0.0001, "capacity": 0.004},
        [BASE, 0.5951, 0.5951, 0.5951], 0.0005,
        [{"mechanism": "prior try", "mean_delta": -0.001}])
    assert "iter_history" in msg
    assert "improvement_score" in msg
    assert "0.0005" in msg
    assert "regularization" in msg
    assert "prior try" in msg, "prior attempts must reach the refiner"


def test_refiner_persona_exists_and_demands_the_component_key():
    assert "component" in personas.REFINER_SYSTEM_PROMPT
    assert "iter_history" in personas.REFINER_SYSTEM_PROMPT
    assert "improvement_score" in personas.REFINER_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
#  5. Prior-refine lookup + writer routing                                     #
# --------------------------------------------------------------------------- #
def test_prior_refines_reads_nodes_jsonl(tmp_path):
    log = tmp_path / "nodes.jsonl"
    log.write_text("\n".join([
        json.dumps({"operation": "improve", "diagnosis": {"component": "features"},
                    "hypothesis": {"mechanism": "not a refine"},
                    "mean_delta": 0.9}),
        json.dumps({"operation": "refine", "diagnosis": {"component": "features"},
                    "hypothesis": {"mechanism": "cross user_id x tab"},
                    "mean_delta": -0.0004}),
        json.dumps({"operation": "refine", "diagnosis": {"component": "capacity"},
                    "hypothesis": {"mechanism": "different component"},
                    "mean_delta": 0.001}),
        "{ not json at all",
    ]) + "\n")

    got = driver._prior_refines_for_component("features", path=str(log))
    assert got == [{"mechanism": "cross user_id x tab", "mean_delta": -0.0004}]
    assert driver._prior_refines_for_component("capacity", path=str(log)) == [
        {"mechanism": "different component", "mean_delta": 0.001}]


def test_prior_refines_empty_when_log_absent(tmp_path):
    assert driver._prior_refines_for_component(
        "features", path=str(tmp_path / "nope.jsonl")) == []


def test_write_refine_routes_component_to_the_registry_file(monkeypatch):
    """The registry, not write_fix's substring heuristic, decides the file."""
    from codegen import writer
    seen = {}

    def fake_write_fix(hypothesis, target_component, **kw):
        seen["target"] = target_component
        return "diff"

    monkeypatch.setattr(writer, "write_fix", fake_write_fix)
    for component, expected_target in (("features", "features"),
                                       ("regularization", "loss"),
                                       ("capacity", "architecture")):
        writer.write_refine({"mechanism": "m", "implementation_sketch": "s"},
                            component=component)
        assert seen["target"] == expected_target


def test_write_refine_rejects_an_unknown_component():
    from codegen import writer
    with pytest.raises(ValueError, match="unknown component"):
        writer.write_refine({"mechanism": "m"}, component="not_a_component")


def test_refinement_schema_requires_the_component_key():
    from llm_calls.schemas import validate_refinement
    good = {"mechanism": "swap the l2 term for dropout on the embedding table",
            "implementation_sketch": "FM.step: drop the l2 grads, mask V rows",
            "success_criterion_paired": ("primary on val-tier-2 improves by at "
                                         "least +0.003 over the parent"),
            "component": "regularization"}
    assert validate_refinement(good)["component"] == "regularization"

    missing = {k: v for k, v in good.items() if k != "component"}
    with pytest.raises(ValueError, match="component"):
        validate_refinement(missing)
