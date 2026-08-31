"""Tests for the component attempt budget (Tier 4A).

What this guards: nothing in the loop could tell the diagnostician to stop
naming the same bottleneck. It was not wrong that the objective mismatches the
metric — it was wrong to keep re-deriving that after several negative results,
and the trajectory it reads from really does stay flat, so asking it nicely
cannot work. The budget is therefore enforced in the driver, and a prompt
instruction is only the cheap path to the same outcome.

Note on the plan's premise. The claim it was written from — "all 5 iterations
returned component='loss_function'" — lived in orchestrator/_state/nodes.jsonl,
which no longer exists; progress.json kept the per-candidate metrics but not the
per-node diagnosis text. So these tests pin the MECHANISM rather than that count.
The counts they do cite (5 of 11 candidates produced a paired delta) are verified
against progress.json.
"""
import pytest

from orchestrator import driver
from orchestrator.node import Node


def _node(nid, component, *, mean_delta=None, evidence=None, verdict=None):
    n = Node(id=nid, parent_id="root", code_path="baseline.py",
             diagnosis={"component": component},
             hypothesis={"mechanism": f"mech for {component}"})
    n.mean_delta = mean_delta
    n.evidence_type = evidence
    n.verdict = verdict
    return n


ROOT = Node(id="root", parent_id=None, code_path="baseline.py",
            operation="draft")


# --------------------------------------------------------------------------- #
#  Canonicalisation — without it the budget can never fire                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,canon", [
    ("loss", "loss_function"),
    ("loss_function", "loss_function"),
    ("objective mismatch", "loss_function"),
    ("training criterion", "loss_function"),
    ("feature_engineering", "feature_engineering"),
    ("feature encoding", "feature_engineering"),
    ("user behaviour sequence", "sequence_features"),
    ("user history modelling", "sequence_features"),
    ("multi-task auxiliary heads", "auxiliary_targets"),
    ("negative sampling", "sampling"),
    ("regularization", "regularization"),
    ("learning rate schedule", "optimization"),
    ("model capacity", "architecture"),
])
def test_component_names_canonicalise(raw, canon):
    assert driver._canonical_component(raw) == canon


def test_three_spellings_of_one_bottleneck_share_a_budget():
    """The reason canonicalisation exists. The mock returns 'loss', a real
    diagnosis returned 'loss_function', a third phrasing is 'objective
    mismatch'. Counted verbatim these are one attempt each and the budget never
    trips."""
    nodes = [ROOT,
             _node("a", "loss", mean_delta=0.0001),
             _node("b", "loss_function", mean_delta=0.0002),
             _node("c", "objective mismatch", mean_delta=-0.0003)]
    ledger = driver._component_ledger(nodes)
    assert list(ledger) == ["loss_function"]
    assert ledger["loss_function"]["scored"] == 3
    assert "loss_function" in driver._exhausted_components(ledger)


def test_sequence_features_is_not_swallowed_by_feature_engineering():
    """Order matters: 'sequence features' contains 'feature'. If it collapsed
    into feature_engineering, exhausting one would retire the other — and
    feature_engineering is where the refuted static-field work lives while
    sequence_features is the untouched direction."""
    assert driver._canonical_component("sequence features") == \
        "sequence_features"
    assert driver._canonical_component("static feature domains") == \
        "feature_engineering"


def test_an_unrecognised_component_gets_its_own_bucket():
    """A novel bottleneck must not inherit an unrelated one's exhaustion."""
    assert driver._canonical_component("exposure debiasing") == \
        "exposure debiasing"


def test_canonicalisation_handles_missing_components():
    assert driver._canonical_component(None) is None
    assert driver._canonical_component("") is None


# --------------------------------------------------------------------------- #
#  The budget itself                                                          #
# --------------------------------------------------------------------------- #
def test_three_scored_attempts_with_a_tiny_best_delta_exhaust_a_component():
    nodes = [ROOT] + [_node(f"n{i}", "loss_function", mean_delta=0.0002)
                      for i in range(3)]
    ledger = driver._component_ledger(nodes)
    assert ledger["loss_function"]["scored"] == 3
    assert ledger["loss_function"]["best_mean_delta"] == pytest.approx(0.0002)
    assert driver._exhausted_components(ledger) == {"loss_function"}


def test_three_failed_implementations_do_NOT_exhaust_a_component():
    """The distinction the plan calls out and the one that decides whether this
    measures anything: an attempt that never scored is evidence about the
    WRITER, not the component. In the recorded run only 5 of 11 candidates ever
    produced a paired delta, so counting failures would retire bottlenecks that
    were never tested."""
    nodes = [ROOT] + [_node(f"n{i}", "loss_function",
                            evidence="failed_implementation")
                      for i in range(3)]
    ledger = driver._component_ledger(nodes)
    assert ledger["loss_function"]["attempts"] == 3
    assert ledger["loss_function"]["scored"] == 0
    assert driver._exhausted_components(ledger) == set(), \
        "un-implemented is not refuted"


def test_a_timeout_also_does_not_count_toward_the_budget():
    nodes = [ROOT] + [_node(f"n{i}", "loss_function", evidence="timeout")
                      for i in range(4)]
    assert driver._exhausted_components(
        driver._component_ledger(nodes)) == set()


def test_a_component_that_is_paying_off_is_never_exhausted():
    nodes = [ROOT] + [_node(f"n{i}", "loss_function", mean_delta=d)
                      for i, d in enumerate([0.0001, 0.0002, 0.0031])]
    ledger = driver._component_ledger(nodes)
    assert ledger["loss_function"]["best_mean_delta"] == pytest.approx(0.0031)
    assert driver._exhausted_components(ledger) == set()


def test_two_scored_attempts_are_not_yet_enough():
    nodes = [ROOT] + [_node(f"n{i}", "loss_function", mean_delta=0.0)
                      for i in range(2)]
    assert driver._exhausted_components(
        driver._component_ledger(nodes)) == set()


def test_the_budget_constants_sit_below_the_seed_std():
    """0.0005 is deliberately under the baseline's 0.0008 5-seed std, so
    exhaustion reads 'not even noise-sized'."""
    assert driver.COMPONENT_ATTEMPT_BUDGET == 3
    assert driver.COMPONENT_EXHAUSTED_DELTA < 0.0008


def test_ledger_skips_the_root_draft():
    assert driver._component_ledger([ROOT]) == {}


def test_ledger_collects_verdicts_per_component():
    nodes = [ROOT,
             _node("a", "loss", mean_delta=0.0001, verdict="refuted"),
             _node("b", "loss", mean_delta=0.0002,
                   verdict="missed_but_promising")]
    ledger = driver._component_ledger(nodes)
    assert ledger["loss_function"]["verdicts"] == ["refuted",
                                                  "missed_but_promising"]


# --------------------------------------------------------------------------- #
#  Driver-side enforcement, not prompt-side                                    #
# --------------------------------------------------------------------------- #
class _StubLLM:
    """Returns a scripted sequence of components, and records every context it
    was handed so the refusal round-trip is observable."""

    def __init__(self, components):
        self.components = list(components)
        self.contexts = []

    def diagnose(self, ctx):
        self.contexts.append(ctx)
        comp = (self.components.pop(0) if self.components
                else self.components_last)
        self.components_last = comp
        return {"bottleneck": f"bottleneck about {comp}", "evidence": "",
                "confidence": 0.75, "component": comp,
                "edit_radius": "small", "expected_cost": "medium",
                "incompatibilities": [], "uncertainty": 0.25}

    def ground_in_literature(self, bottleneck):
        return {"mechanism": "m", "assumptions": [],
                "contradictory_findings": [], "dataset_compatibility": [],
                "implementation_cost": "small", "primary_citation": "x"}

    def generate_hypothesis(self, diagnosis, evidence_card, tried=None, **kw):
        return [{"mechanism": f"mech targeting {diagnosis['component']}",
                 "success_criterion_paired":
                     "primary on val-tier-1 improves by at least +0.001 "
                     "over the parent",
                 "implementation_sketch": "sketch"}]


class _NullMemory:
    """An evidence store that has never recorded anything.

    Implements the whole surface `_build_improve_candidates` consumes, so these
    tests exercise the exhausted-COMPONENT path with the mechanism-FAMILY path
    held at "nothing is blocked" — the two constraints are independent and this
    module is about the first one.
    """

    def is_duplicate(self, fp, blocking_only=False):
        return None

    def is_blocked(self, fp):
        return False

    def probationary_families(self):
        return []


def _build(llm, ledger):
    return driver._build_improve_candidates(
        ROOT, diag_llm=llm, memory=_NullMemory(),
        counters=driver.Counters(), history=[0.5936],
        iter_history=[0.5936], improvement_score=None,
        component_ledger=ledger, verbose=False)


_EXHAUSTED_LEDGER = {"loss_function": {"attempts": 5, "scored": 3,
                                       "best_mean_delta": 0.0002,
                                       "verdicts": []}}


def test_the_exhausted_set_reaches_the_diagnose_context():
    llm = _StubLLM(["sequence features"])
    _build(llm, _EXHAUSTED_LEDGER)
    ctx = llm.contexts[0]
    assert ctx["exhausted_components"] == ["loss_function"]
    assert ctx["component_ledger"] == _EXHAUSTED_LEDGER


def test_naming_an_exhausted_component_triggers_exactly_one_re_ask():
    llm = _StubLLM(["loss_function", "sequence features"])
    diag, _ = _build(llm, _EXHAUSTED_LEDGER)
    assert len(llm.contexts) == 2, "one re-ask, not a loop"
    assert "refusal" in llm.contexts[1]
    assert "loss_function" in llm.contexts[1]["refusal"]
    assert diag["component"] == "sequence features"
    assert "exhaustion_fallback" not in diag, \
        "it complied on the second ask, so nothing was overridden"


def test_insisting_twice_fires_the_deterministic_fallback_and_records_it():
    """A prompt-only rule would reproduce the original bug, so the driver has
    to be able to override the model outright — and say that it did."""
    llm = _StubLLM(["loss_function", "loss"])
    diag, cands = _build(llm, _EXHAUSTED_LEDGER)
    assert len(llm.contexts) == 2, "no third call is spent"
    assert diag["component"] == "sequence_features", \
        "the next unexhausted priority direction"
    assert diag["exhaustion_fallback"] == {"refused": "loss",
                                           "substituted": "sequence_features"}
    # And the substitution reaches the candidate, so the writer is routed at the
    # replacement rather than the refused component.
    assert cands and "sequence_features" in cands[0].hypothesis["mechanism"]


def test_no_re_ask_when_the_component_is_live():
    llm = _StubLLM(["loss_function"])
    diag, _ = _build(llm, {})
    assert len(llm.contexts) == 1
    assert diag["component"] == "loss_function"


def test_fallback_keeps_the_models_choice_when_everything_is_exhausted():
    """Substituting an exhausted component for another exhausted one would be
    worse than useless — it would hide that the search is out of directions."""
    ledger = {c: {"attempts": 5, "scored": 3, "best_mean_delta": 0.0,
                  "verdicts": []}
              for c in driver.UNEXPLORED_PRIORITY}
    llm = _StubLLM(["loss_function", "loss_function"])
    diag, _ = _build(llm, ledger)
    assert diag["component"] == "loss_function"
    assert "all priority components exhausted" in diag["exhaustion_note"]
    assert "exhaustion_fallback" not in diag


def test_fallback_priority_follows_the_readmes_order():
    assert driver.UNEXPLORED_PRIORITY[0] == "loss_function"
    assert driver.UNEXPLORED_PRIORITY[1] == "sequence_features"
    assert driver.UNEXPLORED_PRIORITY[2] == "auxiliary_targets"


# --------------------------------------------------------------------------- #
#  End to end through driver.run()                                             #
# --------------------------------------------------------------------------- #
def test_a_monopolising_component_is_retired_during_a_real_run(
        tmp_path, monkeypatch):
    """The whole point, exercised through run() rather than the helper.

    The mocks' own diagnose always returns component="loss", so this reproduces
    the monopoly directly. Candidates are forced to score just BELOW the parent
    every time, so after 3 scored attempts loss_function is exhausted and the
    driver must stop letting the diagnostician name it.

    Note the mock's unpatched behaviour is the opposite case and is also correct:
    with its default fluctuating scores loss_function reaches +0.0027, clears
    COMPONENT_EXHAUSTED_DELTA, and is never retired. A component that is paying
    off must not be.
    """
    import json
    from orchestrator.mocks import harness as h, llm as l, codegen as cg

    USERS = [f"u{i}" for i in range(10)]

    def _flat(value):
        return {"status": "ok", "logs": "",
                "metrics": {"primary": value, "GAUC": value + 0.03,
                            "nDCG@5": value - 0.03,
                            "per_user": {u: value for u in USERS}}}

    BASE = 0.6012

    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None, **kw):
        # Baseline for the root and for confirm; every candidate lands a hair
        # below it, so each scored attempt is real evidence against the
        # component rather than a failure to implement it.
        if "candidate_" not in str(code_path) or split == "valid_confirm":
            return _flat(BASE)
        return _flat(BASE - 0.0004)

    monkeypatch.setattr(cg, "execute", execute)
    monkeypatch.setattr(driver, "harness", h)
    monkeypatch.setattr(driver, "llm", l)
    monkeypatch.setattr(driver, "codegen", cg)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    driver.run(max_iters=6, verbose=False,
               progress_path=str(tmp_path / "progress.json"),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"),
               memory_path=str(tmp_path / "memory.json"),
               champion_dir=str(tmp_path / "champions"))

    nodes = [json.loads(line) for line in
             (tmp_path / "nodes.jsonl").read_text().splitlines() if line.strip()]
    scored = [n for n in nodes if n["mean_delta"] is not None]
    assert len(scored) >= 4, \
        f"need >3 scored attempts for the budget to bind, got {len(scored)}"

    components = [driver._canonical_component((n.get("diagnosis") or {})
                                              .get("component"))
                  for n in nodes if n.get("hypothesis")]
    # The first three scored attempts are allowed on loss_function; after that
    # the budget binds and the diagnosis must have moved off it.
    assert components[:3] == ["loss_function"] * 3, components
    assert any(c != "loss_function" for c in components[3:]), (
        "loss_function monopolised the search past its budget: " f"{components}")

    # And the override is on the record, not silent.
    overridden = [n for n in nodes
                  if (n.get("diagnosis") or {}).get("exhaustion_fallback")]
    assert overridden, "a driver-side substitution must be recorded in nodes.jsonl"
    fb = overridden[0]["diagnosis"]["exhaustion_fallback"]
    assert fb["substituted"] in driver.UNEXPLORED_PRIORITY
    assert driver._canonical_component(fb["refused"]) == "loss_function"


def test_diagnostician_prompt_states_the_scored_vs_attempted_distinction():
    """The prompt has to carry the same rule the driver enforces, or the model
    reads a high attempt count as evidence against a component that was simply
    never built."""
    from llm_calls.personas import DIAGNOSTICIAN_SYSTEM_PROMPT
    p = DIAGNOSTICIAN_SYSTEM_PROMPT
    assert "exhausted_components" in p
    assert "component_ledger" in p
    assert "un-implemented" in p
    assert "0.0005" in p
