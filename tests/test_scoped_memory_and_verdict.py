"""Tests for the ancestor-scoped debug memory (2A) and the rubric verdict (2B).

Both close the same gap from opposite ends: the recorded 5-iteration run in
orchestrator/_state/ produced 11 candidates, of which 5 were filed
failed_implementation and 1 timed out — so the repair operator handled the modal
outcome and received nothing but a traceback and a file. The other 5 were scored
against a `success_criterion_paired` that no component ever read, so "refuted",
"never implemented" and "criterion uncalibrated by 8x" all closed a node
identically.

Counts are verified against orchestrator/_state/progress.json. The fix plan this
work follows says "6 of 11 failed_implementation" and "only 2 of 11 ever produced
a paired delta"; the surviving record says 5 and 5. The paired numbers the tests
below pin (+0.000593 at p_pos 0.776, lower_95 -0.000673) do match it exactly.
"""
import json

import pytest

from codegen import prompts
from llm_calls.schemas import validate_verdict
from orchestrator import driver
from orchestrator.memory import EvidenceEntry, Memory
from orchestrator.node import Node


def _node(nid, parent_id, mechanism=None, *, evidence=None, mean_delta=None,
          error=None, operation="improve"):
    n = Node(id=nid, parent_id=parent_id, code_path="baseline.py",
             operation=operation,
             hypothesis={"mechanism": mechanism} if mechanism else None)
    n.evidence_type = evidence
    n.mean_delta = mean_delta
    n.last_error_excerpt = error
    return n


# --------------------------------------------------------------------------- #
#  2A — ancestor chain                                                         #
# --------------------------------------------------------------------------- #
def _lineage():
    """root <- a <- b <- c <- d, so depth truncation is observable."""
    root = _node("root", None, operation="draft")
    a = _node("a", "root", "BPR pairwise", evidence="failed_implementation",
              error="ValueError: operands could not be broadcast")
    b = _node("b", "a", "listwise softmax", evidence=None, mean_delta=-0.0115)
    c = _node("c", "b", "lambdarank surrogate", evidence="timeout")
    d = _node("d", "c", "ranknet pairwise")
    return [root, a, b, c, d], d


def test_chain_is_newest_first_and_excludes_the_node_itself():
    nodes, d = _lineage()
    chain = driver._ancestor_chain(d, nodes)
    assert [e["id"] for e in chain] == ["c", "b", "a", "root"]


def test_chain_stops_at_the_root():
    nodes, d = _lineage()
    chain = driver._ancestor_chain(d, nodes, max_depth=99)
    assert chain[-1]["id"] == "root"
    assert len(chain) == 4, "must not walk past the root"


def test_chain_respects_max_depth():
    nodes, d = _lineage()
    chain = driver._ancestor_chain(d, nodes, max_depth=2)
    assert [e["id"] for e in chain] == ["c", "b"], "keeps the NEAREST ancestors"


def test_chain_carries_the_two_facts_the_debug_branches_need():
    """The instruction splits on failed-to-run vs ran-but-worse, so both the
    evidence type and the paired delta have to survive into the entry."""
    nodes, d = _lineage()
    by_id = {e["id"]: e for e in driver._ancestor_chain(d, nodes)}
    assert by_id["a"]["evidence_type"] == "failed_implementation"
    assert "broadcast" in by_id["a"]["last_error_excerpt"]
    assert by_id["b"]["mean_delta"] == pytest.approx(-0.0115)
    assert by_id["c"]["evidence_type"] == "timeout"


def test_chain_survives_a_parent_id_cycle():
    """A malformed tree must not hang an unattended run."""
    x = _node("x", "y", "mech x")
    y = _node("y", "x", "mech y")
    chain = driver._ancestor_chain(x, [x, y], max_depth=99)
    assert [e["id"] for e in chain] == ["y"]


def test_chain_is_empty_for_a_child_of_the_root_only_when_root_is_missing():
    """A node whose parent is not in all_nodes yields an empty chain rather
    than raising — the driver builds all_nodes incrementally."""
    orphan = _node("o", "gone", "mech")
    assert driver._ancestor_chain(orphan, [orphan]) == []


# --------------------------------------------------------------------------- #
#  2A — the chain has to reach the prompt string                               #
# --------------------------------------------------------------------------- #
def test_ancestor_block_names_both_failure_modes_distinctly():
    nodes, d = _lineage()
    block = prompts.build_ancestor_block(driver._ancestor_chain(d, nodes))
    assert "BPR pairwise" in block
    assert "failed_implementation" in block
    assert "-0.01150" in block, "the paired delta must be rendered"
    assert "broadcast" in block
    # The two branches of the instruction, which is the whole point of the block.
    assert "FAILED TO RUN" in block
    assert "RAN BUT SCORED WORSE" in block


def test_ancestor_block_is_empty_without_ancestors():
    """So a root-level repair prompt is byte-identical to what it was before."""
    assert prompts.build_ancestor_block([]) == ""
    assert prompts.build_ancestor_block(None) == ""


def test_debug_message_carries_hypothesis_and_ancestors():
    nodes, d = _lineage()
    msg = prompts.build_debug_user(
        "baseline.py", "# file\n", "Traceback: boom",
        hypothesis={"mechanism": "blend BPR at weight 0.2",
                    "implementation_sketch": "add a term in FM.step"},
        ancestors=driver._ancestor_chain(d, nodes))
    assert "blend BPR at weight 0.2" in msg
    assert "add a term in FM.step" in msg
    assert "ANCESTRY" in msg
    assert "Traceback: boom" in msg
    # The anti-revert instruction: a repair that drops the mechanism to make the
    # file run scores identically to the parent and is recorded as a real trial.
    assert "reverts the mechanism" in msg


def test_debug_message_keeps_the_frozen_three_positional_form():
    """codegen/tests and debug_and_retry's 2-argument contract depend on it."""
    msg = prompts.build_debug_user("baseline.py", "# file\n", "boom")
    assert "boom" in msg
    assert "ANCESTRY" not in msg
    assert "was trying to implement" not in msg


def test_debug_and_retry_accepts_ancestors_without_breaking_its_contract():
    import inspect
    from codegen.debug import debug_and_retry
    sig = inspect.signature(debug_and_retry)
    assert "ancestors" in sig.parameters
    assert sig.parameters["ancestors"].default is None
    # Still keyword-only, so the frozen positional form cannot be shifted.
    assert sig.parameters["ancestors"].kind is inspect.Parameter.KEYWORD_ONLY


# --------------------------------------------------------------------------- #
#  2B — the verdict schema's cross-field rules                                 #
# --------------------------------------------------------------------------- #
def _v(**kw):
    base = {"verdict": "met", "criterion_was_calibrated": True,
            "reason": "ok", "next_action": "build_on_it"}
    base.update(kw)
    return base


def test_verdict_accepts_the_case_this_step_exists_for():
    """A +0.0006 measured against a stated +0.005: promising, not refuted."""
    out = validate_verdict(_v(verdict="missed_but_promising",
                              criterion_was_calibrated=False,
                              next_action="adjust_magnitude",
                              reason="inside the 0.0012 noise floor"))
    assert out["verdict"] == "missed_but_promising"
    assert out["criterion_was_calibrated"] is False


def test_verdict_rejects_refuting_a_mechanism_for_missing_an_unreachable_bar():
    """The exact failure the step exists to prevent, so the schema blocks it
    rather than trusting the persona to remember."""
    with pytest.raises(ValueError, match="incoherent"):
        validate_verdict(_v(verdict="refuted", criterion_was_calibrated=False,
                            next_action="abandon_mechanism"))


def test_verdict_rejects_abandoning_a_mechanism_that_was_never_measured():
    """abandon_mechanism is a permanent block on the family; not_tested means
    there is no evidence to base it on."""
    with pytest.raises(ValueError, match="incoherent"):
        validate_verdict(_v(verdict="not_tested",
                            criterion_was_calibrated=False,
                            next_action="abandon_mechanism"))


def test_verdict_allows_not_tested_with_a_cheaper_retry():
    out = validate_verdict(_v(verdict="not_tested",
                              criterion_was_calibrated=False,
                              next_action="retry_cheaper",
                              reason="never ran"))
    assert out["next_action"] == "retry_cheaper"


@pytest.mark.parametrize("bad", [
    {"verdict": "inconclusive"},          # not in the enum
    {"next_action": "give_up"},           # not in the enum
    {"criterion_was_calibrated": "yes"},  # not a bool
])
def test_verdict_rejects_off_schema_values(bad):
    with pytest.raises(ValueError):
        validate_verdict(_v(**bad))


def test_verdict_persona_states_the_calibration_numbers():
    """The persona has to carry the measured facts, or it cannot make the one
    judgement it exists to make."""
    from llm_calls.personas import VERDICT_SYSTEM_PROMPT
    for fact in ("0.0008", "0.0012", "+0.0006", "missed_but_promising"):
        assert fact in VERDICT_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
#  2B — driver wiring                                                          #
# --------------------------------------------------------------------------- #
class _FakeVerdictLLM:
    """Injected the way tests/test_mlestar.py injects its fake LLM."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def verdict(self, hypothesis, measured, context):
        self.calls.append((hypothesis, measured, context))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _scored(mean_delta):
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    root.per_seed_primary = {0: 0.5936}
    c = Node(id="c1", parent_id="root", code_path="baseline.py",
             diagnosis={"component": "loss_function"},
             hypothesis={"mechanism": "blend BPR at weight 0.2",
                         "success_criterion_paired":
                             "primary on val-tier-1 improves by at least "
                             "+0.005 over the parent"})
    c.per_seed_primary = {0: 0.5942, 1: 0.5941, 2: 0.5940}
    c.mean_delta = mean_delta
    c.p_positive, c.lower_95 = 0.776, -0.00067
    return root, c


def test_a_small_positive_delta_is_not_recorded_as_refuted():
    """iteration 1's real numbers: mean_delta +0.000593 against a stated
    +0.005. Nothing in the old loop could tell that apart from a refutation."""
    root, c = _scored(0.000593)
    llm = _FakeVerdictLLM({"verdict": "missed_but_promising",
                           "criterion_was_calibrated": False,
                           "reason": "positive but inside the noise floor",
                           "next_action": "adjust_magnitude"})
    driver._apply_verdict(c, root, root, llm, driver.Counters(), verbose=False)

    assert c.verdict == "missed_but_promising"
    assert c.criterion_was_calibrated is False
    assert c.next_action == "adjust_magnitude"
    assert c.verdict != "refuted"
    # The grader must see the stated criterion and the measured delta together.
    hyp, measured, ctx = llm.calls[0]
    assert "+0.005" in hyp["success_criterion_paired"]
    assert measured["mean_delta"] == pytest.approx(0.000593)
    assert ctx["paired_noise_floor"] == 0.0012


def test_the_verdict_never_fabricates_a_token_count():
    """This used to assert `counters.tokens == driver.VERDICT_TOKENS`, i.e. that
    a verdict charged a hard-coded 400 tokens.

    T2.3 removed every such constant. `tokens: 13200` in the persisted
    progress.json was the sum of these guesses and no API ever reported it, while
    the numbers are exactly what Feasibility & Practicality is scored on. Token
    accounting now happens where the raw API response is visible — the client
    layer — so a grader with no API call behind it must charge NOTHING, on
    success as well as on failure.
    """
    root, c = _scored(0.000593)
    counters = driver.Counters()

    driver._apply_verdict(c, root, root, _FakeVerdictLLM(RuntimeError("api down")),
                          counters, verbose=False)
    assert counters.tokens == 0

    driver._apply_verdict(c, root, root,
                          _FakeVerdictLLM({"verdict": "met",
                                           "criterion_was_calibrated": True,
                                           "reason": "ok",
                                           "next_action": "build_on_it"}),
                          counters, verbose=False)
    assert c.verdict == "met", "the verdict itself must still be applied"
    assert counters.tokens == 0, (
        "a fake grader made no API call, so there is nothing to charge; a "
        "non-zero count here would be a guess re-entering the report")
    assert not hasattr(driver, "VERDICT_TOKENS"), \
        "the fabricated per-verdict constant must be gone, not just unused"


def test_no_token_count_in_the_driver_is_hand_incremented():
    """The class fix behind T2.3, pinned so a constant cannot creep back.

    Eight call sites used to bump `tokens` by a guessed constant (500 a
    diagnose, 300 a hypothesis, 800 a writer call, 400 an audit). Real usage now
    comes off `resp.usage` via llm_calls.usage.LEDGER.
    """
    import inspect
    src = inspect.getsource(driver)
    assert 'bump("tokens"' not in src, \
        "a hand-incremented token count is back in driver.py"
    assert "sync_usage" in src, \
        "the driver must pull real usage off the ledger"


def test_a_failed_grader_never_loses_a_scored_candidate():
    """The candidate cost three full training runs; a broken grader is
    commentary, not a measurement."""
    root, c = _scored(0.000593)
    c.per_seed_primary = {0: 0.5942, 1: 0.5941, 2: 0.5940}
    driver._apply_verdict(c, root, root, _FakeVerdictLLM(RuntimeError("boom")),
                          driver.Counters(), verbose=False)
    assert c.verdict is None
    assert c.next_action is None
    assert c.per_seed_primary, "the measurement must survive intact"
    assert c.status == "open", "a grader must not close a node"


def test_verdict_is_skipped_for_a_node_with_no_hypothesis():
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    llm = _FakeVerdictLLM({"verdict": "met", "criterion_was_calibrated": True,
                           "reason": "x", "next_action": "build_on_it"})
    driver._apply_verdict(root, root, root, llm, driver.Counters(), verbose=False)
    assert llm.calls == [], "the root draft has nothing to grade"


# --------------------------------------------------------------------------- #
#  2B — the verdict has to change what gets proposed next                      #
# --------------------------------------------------------------------------- #
def test_abandon_mechanism_blocks_the_family_from_being_reproposed(tmp_path):
    m = Memory(path=str(tmp_path / "memory.json"))
    fp = driver._fingerprint({"mechanism": "swap in a pairwise BPR loss"})
    assert m.is_duplicate(fp, blocking_only=True) is None

    m.record(EvidenceEntry(
        fingerprint=fp, architecture="FM", loss="bpr", sampler="uniform",
        split="valid_search", seed_count=3, confidence_interval=None,
        code_hash="c1", evidence_type="refuted_under_context",
        note="fairly measured and negative"))

    # The next identical proposal, however differently worded, now dedups away.
    reworded = driver._fingerprint(
        {"mechanism": "replace the pointwise logloss with Bayesian "
                      "Personalized Ranking"})
    assert reworded == fp
    assert m.is_duplicate(reworded, blocking_only=True) is not None


def test_retry_cheaper_leaves_the_family_proposable(tmp_path):
    """An inconclusive record is a record, not a verdict. Blocking on it retired
    a family after one indecisive result — including families the grader
    explicitly said to retry or build on."""
    m = Memory(path=str(tmp_path / "memory.json"))
    fp = driver._fingerprint({"mechanism": "swap in a pairwise BPR loss"})
    m.record(EvidenceEntry(
        fingerprint=fp, architecture="FM", loss="bpr", sampler="uniform",
        split="valid_search", seed_count=1, confidence_interval=None,
        code_hash="c1", evidence_type="inconclusive",
        note="implementation timed out; never measured"))

    assert m.is_duplicate(fp, blocking_only=True) is None, \
        "an inconclusive result must not retire the mechanism family"
    # But the evidence is still on record, for the prompts to read back.
    assert m.is_duplicate(fp) is not None


def test_preseeded_dead_ends_still_block(tmp_path):
    """The four hand-authored preseeds are all refuted_under_context, so
    weakening the dedup must not have unblocked them."""
    m = Memory(path=str(tmp_path / "memory.json"))
    cwm = ("pointwise_logloss", "uniform", "cwm_13field", "pure")
    assert m.is_duplicate(cwm, blocking_only=True) is not None


def test_ledger_reports_the_verdict_to_the_proposer():
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    c = _node("c1", "root", "BPR pairwise", mean_delta=0.000593)
    c.per_seed_primary = {0: 0.5942}
    c.verdict = "missed_but_promising"
    c.next_action = "adjust_magnitude"
    entry = driver._attempt_ledger([root, c], parent=root)[0]
    assert entry["verdict"] == "missed_but_promising"
    assert entry["next_action"] == "adjust_magnitude"


def test_hypothesis_prompt_explains_how_to_read_next_action():
    from llm_calls.hypothesis import _build_prompt
    tried = [{"id": "a", "mechanism": "BPR pairwise", "outcome": "scored",
              "mean_delta_vs_parent": 0.000593,
              "verdict": "missed_but_promising",
              "next_action": "retry_cheaper"}]
    prompt = _build_prompt({"bottleneck": "x"}, {"y": 1}, 1, tried=tried)
    assert "next_action" in prompt
    assert "retry_cheaper" in prompt
    assert "FORBIDDEN" in prompt, "abandon_mechanism must read as a hard block"
    assert "criterion_was_calibrated" in prompt


# --------------------------------------------------------------------------- #
#  End to end on the mocks                                                     #
# --------------------------------------------------------------------------- #
def test_mocked_run_writes_verdicts_and_error_excerpts_to_nodes_log(
        tmp_path, monkeypatch):
    from orchestrator.mocks import harness as h, llm as l, codegen as cg
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

    nodes = [json.loads(l) for l in
             (tmp_path / "nodes.jsonl").read_text().splitlines() if l.strip()]
    scored = [n for n in nodes if n["per_seed_primary"]]
    assert scored, "the mocks must produce at least one scored candidate"
    assert all("verdict" in n for n in nodes), \
        "the verdict fields must reach nodes.jsonl for every node"
    assert any(n["verdict"] is not None for n in scored), \
        "at least one scored candidate must carry a verdict"
    # And every verdict written is one the schema would accept.
    for n in scored:
        if n["verdict"] is not None:
            assert n["verdict"] in ("met", "missed_but_promising", "refuted",
                                    "not_tested")
            assert n["next_action"] in ("retry_cheaper", "adjust_magnitude",
                                        "abandon_mechanism", "build_on_it")


# --------------------------------------------------------------------------- #
#  Bans are RUN-SCOPED (T1.4, corrected after the T2.7 interaction)           #
# --------------------------------------------------------------------------- #
#: The measurement that forced this rule. T2.7 widened each iteration from 1
#: proposal to 6-8 with up to 4 executed, which multiplies the number of SCORED
#: candidates per iteration — and every non-positive paired delta earns
#: `abandon_mechanism` -> `refuted_under_context`. Under a bar of "two
#: independent refutations, whenever measured", ONE fresh 8-iteration mocked run
#: wrote 28 entries and left 7 of 13 families hard-blocked for the NEXT run, with
#: 5 more on probation. A third run would start with almost nothing legal.
#:
#: That is the starvation this store was rewritten to prevent, one level up.
#: Liveness held (the frontier floor and the probation node saw to that), but the
#: proposal space collapsed run over run — which makes the search progressively
#: unable to measure anything new.
_FP = ("mechanism_family", "bpr_pairwise", "", "")


def _ref(code_hash, seed_count=3):
    return EvidenceEntry(
        fingerprint=_FP, architecture="FM", loss="bpr", sampler="uniform",
        split="valid_search", seed_count=seed_count, confidence_interval=None,
        code_hash=code_hash, evidence_type="refuted_under_context",
        note="fairly measured and negative")


def test_two_refutations_from_this_run_still_block(tmp_path):
    """The fix must not defang the ban. Within one run, corroborated evidence is
    still a hard block — that is what stops the search re-testing a family it has
    already measured twice."""
    m = Memory(path=str(tmp_path / "m.json"))
    assert not m.is_blocked(_FP)
    m.record(_ref("c1"))
    assert not m.is_blocked(_FP), "one refutation is probation"
    m.record(_ref("c2"))
    assert m.is_blocked(_FP), "two from THIS run must block"


def test_a_ban_inherited_from_a_previous_run_is_probation_not_a_filter(tmp_path):
    """THE fix. A new process gets a new run_id, so evidence written by an earlier
    run no longer blocks on its own — it is rendered into the prompt as a
    discount instead, exactly as a single refutation already was."""
    path = str(tmp_path / "m.json")

    run1 = Memory(path=path)
    run1.record(_ref("c1"))
    run1.record(_ref("c2"))
    assert run1.is_blocked(_FP)

    run2 = Memory(path=path)                 # new process, same file
    assert run2.run_id != run1.run_id
    assert not run2.is_blocked(_FP), \
        "a second run must not start with the first run's families retired"
    assert "bpr_pairwise" in run2.probationary_families(), \
        "the evidence must still be VISIBLE to the proposer, just not enforced"


def test_the_inherited_evidence_is_not_discarded_only_deferred(tmp_path):
    """Prior measurements still count. ONE corroborating result from the current
    run re-establishes the block, so a family that really is dead is re-retired
    after a single cheap negative rather than needing two all over again."""
    path = str(tmp_path / "m.json")
    run1 = Memory(path=path)
    run1.record(_ref("c1"))
    run1.record(_ref("c2"))

    run2 = Memory(path=path)
    assert not run2.is_blocked(_FP)
    run2.record(_ref("c3"))                  # one fresh negative measurement
    assert run2.is_blocked(_FP), \
        "inherited evidence plus one current-run refutation must block"


def test_a_preseed_still_blocks_on_its_own_across_runs(tmp_path):
    """The corroboration bar guards against ONE RUN's noisy verdict. A curated
    measured fact is not that, so it is exempt — otherwise the four hand-authored
    dead ends would silently reopen on every new process."""
    m = Memory(path=str(tmp_path / "m.json"))
    cwm = ("pointwise_logloss", "uniform", "cwm_13field", "pure")
    for fp in [cwm,
               ("lambdarank", "uniform", "lgb_train_aggregates", "pure"),
               ("binary_logloss", "uniform", "lgb_plus_oof_fm_score", "pure")]:
        assert m.is_blocked(fp), f"{fp} must block without corroboration"

    # A brand-new run, sharing no run_id with anything on disk, agrees.
    assert Memory(path=str(tmp_path / "m.json")).is_blocked(cwm) is True

    # Negative control: an unknown fingerprint blocks nothing, so the assertions
    # above are about the preseeds and not about is_blocked returning True.
    assert not m.is_blocked(("pointwise_logloss", "uniform", "no_such_thing",
                             "pure"))


def test_an_unscored_refutation_never_counts_toward_the_bar(tmp_path):
    """seed_count == 0 means no paired delta was ever produced: evidence about
    the WRITER, not about the mechanism. In the recorded run only 2 of 5
    candidates produced one."""
    m = Memory(path=str(tmp_path / "m.json"))
    m.record(_ref("c1", seed_count=0))
    m.record(_ref("c2", seed_count=0))
    assert not m.is_blocked(_FP), "un-implemented is not refuted"
    m.record(_ref("c3", seed_count=3))
    assert not m.is_blocked(_FP), "still only one real measurement"
    m.record(_ref("c4", seed_count=3))
    assert m.is_blocked(_FP)


def test_a_whole_run_of_refutations_leaves_the_next_run_fully_proposable(tmp_path):
    """The regression, end to end, at the scale that exposed it: 13 families each
    refuted twice by one run must leave the NEXT run with nothing blocked and
    everything on probation."""
    from llm_calls.families import ALL_FAMILIES

    path = str(tmp_path / "m.json")
    run1 = Memory(path=path)
    for fam in ALL_FAMILIES:
        fp = ("mechanism_family", fam, "", "")
        for ch in ("a", "b"):
            run1.record(EvidenceEntry(
                fingerprint=fp, architecture="FM", loss="x", sampler="uniform",
                split="valid_search", seed_count=3, confidence_interval=None,
                code_hash=f"{fam}_{ch}", evidence_type="refuted_under_context",
                note="negative"))
    blocked_in_run1 = [f for f in ALL_FAMILIES
                       if run1.is_blocked(("mechanism_family", f, "", ""))]
    assert len(blocked_in_run1) == len(ALL_FAMILIES), \
        "run 1 should block what run 1 measured"

    run2 = Memory(path=path)
    blocked_in_run2 = [f for f in ALL_FAMILIES
                       if run2.is_blocked(("mechanism_family", f, "", ""))]
    assert blocked_in_run2 == [], \
        f"run 2 inherited {len(blocked_in_run2)} bans: {blocked_in_run2}"
    assert set(run2.probationary_families()) == set(ALL_FAMILIES), \
        "every family's evidence must still reach the prompt"
