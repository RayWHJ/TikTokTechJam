"""Tier 1 regression tests: the search must never stop being able to measure.

These are the definition of done for Tier 1. Every one of them fails against the
tree as it was recorded in `orchestrator/_state/progress.json`, and each one pins
a distinct link in the chain that killed that run at iteration 4:

  iter 4: n_candidates=0  n_scored=0  n_open_nodes=0   <- search died here

The chain, in the order the failures compose:

  1. A promotion writes evidence_type="invariant" into memory, and "invariant"
     was a BLOCKING type — so the first mechanism that ever WORKED retired its
     own family. Reachable only on success, i.e. only when it costs the most.
  2. Three hypotheses all fingerprinted into an already-banned family, so
     `_build_improve_candidates` returned [] with no re-ask and no log line.
  3. Zero candidates closed the parent, which was the ROOT, which emptied
     `open_nodes`.
  4. `global_should_stop` reached the empty frontier before `select` could raise
     on it, and reported "global convergence".
  5. A ban generalised from ONE scored refutation, and the store it lived in had
     never merged its own newer preseeds, so the most useful measured fact in the
     repo was on no machine that already had a memory.json.

Why no existing test caught any of this: `llm_calls/schemas.py::_reject_unknown_keys`
forbids `loss_type` / `sampler` / `feature_set` / `dataset_tier` on a hypothesis, so
a REAL hypothesis can never take `_fingerprint`'s structured branch — it always
falls through to the substring-family branch. The mock supplies exactly those
forbidden keys with a per-iteration digest, so the mock's fingerprints are unique
forever and the family collision is unreachable. Several tests below therefore
strip those keys from the mock, which is what makes the mock fingerprint the way
production does.
"""
import json
import random

import pytest

from orchestrator import driver
from orchestrator.memory import EvidenceEntry, Memory
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm
from orchestrator.node import Node


# --------------------------------------------------------------------------- #
#  Fixtures and helpers                                                       #
# --------------------------------------------------------------------------- #
#: The mechanism the mock proposes, and the family production's `_fingerprint`
#: puts it in once the mock's non-production structured keys are stripped.
BANNED_FAMILY = "bpr_pairwise"
BANNED_FP = ("mechanism_family", BANNED_FAMILY, "", "")


def _refutation(fp, code_hash, note="fairly measured and negative"):
    """One scored refutation of `fp`, as the driver's survivors loop writes them."""
    return EvidenceEntry(
        fingerprint=fp, architecture="FM", loss="bpr", sampler="uniform",
        split="valid_search", seed_count=3, confidence_interval=(-0.007, -0.006),
        code_hash=code_hash, evidence_type="refuted_under_context", note=note)


def _hard_ban(mem, fp=BANNED_FP):
    """Two independent scored refutations — the bar a HARD block requires.

    Two, not one, because a single refutation is probation: the two bans in the
    recorded run measured -0.0065 and -0.0050 against a paired noise floor of
    0.0012, so they are sound evidence about the IMPLEMENTATIONS they tested and
    unsound as a verdict on a whole family.
    """
    mem.record(_refutation(fp, "cand_a"))
    mem.record(_refutation(fp, "cand_b"))


class _FakeDiagLLM:
    """Stands in for llm_calls'. Records every diagnose() context it is handed.

    `hypotheses` is a list of hypothesis dicts returned on EVERY call, so a test
    can make the proposer insist on a banned family no matter how many times it
    is re-asked — which is exactly what iteration 4 of the recorded run did: the
    ledger already carried `next_action: abandon_mechanism` for both families and
    the prompt already said `abandon_mechanism: FORBIDDEN`, and the model proposed
    into them 3 for 3 anyway.
    """

    def __init__(self, hypotheses, component="loss"):
        self.hypotheses = hypotheses
        self.component = component
        self.contexts = []

    def diagnose(self, ctx):
        self.contexts.append(ctx)
        return {"bottleneck": "pointwise logloss misaligned with ranking metric",
                "evidence": "plateau at ~0.595", "confidence": 0.75,
                "component": self.component, "edit_radius": "small",
                "expected_cost": "medium", "incompatibilities": [],
                "uncertainty": 0.25}

    def ground_in_literature(self, bottleneck):
        return {"mechanism": "BPR pairwise loss sampled within user",
                "assumptions": [], "contradictory_findings": [],
                "dataset_compatibility": [], "implementation_cost": "small",
                "primary_citation": "Rendle et al. 2009"}

    def generate_hypothesis(self, diagnosis, evidence_card, tried=None, **kw):
        return [dict(h) for h in self.hypotheses]


def _bpr_hypothesis(n=1):
    """n hypotheses that all land in BANNED_FAMILY however they are worded.

    Deliberately three different phrasings, because that is the shape of the
    real failure: dedup on a prose hash never fired once across 11 candidates,
    and family matching is what makes the collision visible.
    """
    wordings = [
        "swap the pointwise logloss for a within-user BPR pairwise loss",
        "replace the objective with Bayesian Personalized Ranking over pairs",
        "optimise BPR on (positive, negative) impression pairs per user",
    ]
    return [{"mechanism": wordings[i % len(wordings)],
             "success_criterion_paired": "paired delta > +0.003 on valid_search",
             "implementation_sketch": "in baseline.py FM.step, form pairs per user"}
            for i in range(n)]


def _build(parent, memory, diag_llm, counters=None):
    """_build_improve_candidates with the boring arguments filled in."""
    counters = counters or driver.Counters()
    diag, cands = driver._build_improve_candidates(
        parent, diag_llm=diag_llm, memory=memory, counters=counters,
        history=[0.5946], iter_history=[0.5946], improvement_score=None,
        tried=[], component_ledger={}, verbose=False)
    return diag, cands, counters


def _isolated_caches(tmp_path):
    """Per-test paths for both baseline caches.

    Load-bearing, not hygiene: with the default paths a mocked run reads the REAL
    orchestrator/_state/root_baseline.json, whose measured user ids share nothing
    with the mock's u0..u9 — so every paired delta list comes back empty and the
    tree cannot grow past the root.
    """
    return {"root_baseline_path": str(tmp_path / "root_baseline.json"),
            "confirm_baseline_path": str(tmp_path / "confirm_baseline.json")}


@pytest.fixture
def production_shaped_mock(monkeypatch, tmp_path):
    """The mocks, wired into the driver, emitting PRODUCTION-shaped hypotheses.

    The strip is the whole point. `mocks/llm.py::generate_hypothesis` returns
    loss_type / sampler / feature_set / dataset_tier with a per-iteration digest
    folded into loss_type, so the mock takes `_fingerprint`'s structured branch
    and its fingerprints are unique forever. Schemas forbid all four keys on a
    real hypothesis, so production ALWAYS takes the family branch. Without this
    strip the family collision is unreachable in every test in the repo.
    """
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    # Since T2.6 the mock emits schema-shaped output natively — a declared
    # `mechanism_family` and none of the four forbidden keys — so the strip that
    # used to be necessary here is now a belt-and-braces assertion instead.
    #
    # It also HONOURS blocked_families, so this wrapper forces it to ignore them.
    # That is the point of this fixture: it recreates a proposer that insists on
    # a refuted family, which is what the real model did 3 for 3 in iteration 4
    # despite the prompt saying `abandon_mechanism: FORBIDDEN`.
    _NON_PRODUCTION = ("loss_type", "sampler", "feature_set", "dataset_tier")
    _orig = mock_llm.generate_hypothesis

    def _schema_shaped(diagnosis, evidence_card, tried=None, **kw):
        out = _orig(diagnosis, evidence_card, tried=tried)   # blocked_families dropped
        for h in out:
            assert not any(k in h for k in _NON_PRODUCTION), (
                "the mock is emitting keys production forbids, so its "
                "fingerprints are unique forever and the family collision is "
                "unreachable")
            # Pin every hypothesis to the banned family, whatever the mock chose.
            h["mechanism_family"] = BANNED_FAMILY
        return out

    monkeypatch.setattr(mock_llm, "generate_hypothesis", _schema_shaped)

    # mocks.codegen.execute injects a 5% random error off the global RNG, which
    # is seeded from OS entropy per session — pin it so liveness is not a
    # coin flip on which tests ran first.
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


# --------------------------------------------------------------------------- #
#  T1.1 — success must not ban itself                                          #
# --------------------------------------------------------------------------- #
def test_promotion_bans_the_family_it_just_proved(tmp_path):
    """A candidate that clears the SEALED-split promotion gate gets
    evidence_type="invariant" (driver.py, promotion branch) and that value is
    written straight into memory by the survivors loop below it.

    While "invariant" was in Memory.BLOCKING_EVIDENCE, that record retired the
    family permanently — so the ONE thing the search is looking for was also the
    thing that stopped it looking. It never fired in the recorded run only
    because nothing ever promoted.
    """
    assert "invariant" not in Memory.BLOCKING_EVIDENCE, (
        "a promotion writes evidence_type='invariant'; only a refutation may "
        "block a family from being proposed again")

    mem = Memory(path=str(tmp_path / "memory.json"))
    h = _bpr_hypothesis(1)[0]
    fp = driver._fingerprint(h)
    assert fp == BANNED_FP, "the family branch is the one production takes"

    # Exactly what driver.py records for a promoted node.
    mem.record(EvidenceEntry(
        fingerprint=fp, architecture="FM", loss=h["mechanism"], sampler="uniform",
        split="valid_search", seed_count=3, confidence_interval=(0.60, 0.61),
        code_hash="promoted_node", evidence_type="invariant",
        note="cleared the sealed valid_confirm gate"))

    # The verdict on a promotion is build_on_it, and build_on_it must leave the
    # family proposable — so the very next iteration can build on what worked.
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    llm = _FakeDiagLLM(_bpr_hypothesis(1))
    _diag, cands, _counters = _build(root, mem, llm)
    assert cands, ("the family that just PROVED itself must stay proposable; "
                   "it was retired by its own success")


# --------------------------------------------------------------------------- #
#  T1.2 — the root never leaves the frontier                                   #
# --------------------------------------------------------------------------- #
def test_a_run_that_starts_with_bans_still_completes_iterations(
        production_shaped_mock, tmp_path):
    """The measured consequence of the starvation bug, as a test.

    Mocked driver, mock's non-production keys stripped so its fingerprints match
    production's:

        pre-existing bans = none:             iters_completed=2  n_candidates=[1, 0]
        pre-existing bans = ['bpr_pairwise']: iters_completed=1  n_candidates=[0]

    A second run on a machine that already has a memory.json died in iteration 1
    having measured nothing, and reported the unmodified baseline as its champion.
    """
    memory_path = tmp_path / "memory.json"
    mem = Memory(path=str(memory_path))
    _hard_ban(mem)

    progress = tmp_path / "progress.json"
    result = driver.run(max_iters=5, verbose=False,
                        progress_path=str(progress),
                        memory_path=str(memory_path),
                        champion_dir=str(tmp_path / "champions"),
                        **_isolated_caches(tmp_path))

    iters = json.loads(progress.read_text())["iterations"]
    assert result["iters_completed"] >= 4, (
        f"a run that starts from a populated memory.json must still complete "
        f"iterations; got {result['iters_completed']} "
        f"(n_candidates={[r['n_candidates'] for r in iters]})")

    for r in iters:
        assert r["n_open_nodes"] >= 1, (
            f"iter {r['iter']} emptied the frontier. selection.select raises "
            f"RuntimeError on an empty list, so the only thing between this and "
            f"a crash is global_should_stop calling it convergence first.")

    # And an iteration that measured nothing has to say WHY, in the file.
    for r in iters:
        if r["n_candidates"] == 0:
            assert r.get("dedup_starved") or r.get("dropped_by_dedup"), (
                f"iter {r['iter']} produced no candidates and left no reason; "
                f"the only surviving evidence of the drop in the recorded run "
                f"was the arithmetic gap between counters.proposals and "
                f"sum(n_candidates)")


# --------------------------------------------------------------------------- #
#  T1.3 — re-ask, then substitute; never return zero                           #
# --------------------------------------------------------------------------- #
def test_every_proposal_in_a_banned_family_starves_the_iteration(tmp_path):
    """Iteration 4: three hypotheses, all in one banned family, all dropped
    silently, zero candidates returned.

    The re-ask alone is not sufficient and the test says so by making the
    proposer insist — iteration 4's ledger already carried
    `next_action: abandon_mechanism` for both families and
    `llm_calls/hypothesis.py`'s prompt already said `abandon_mechanism:
    FORBIDDEN`, and the model proposed into them 3 for 3 anyway. So the
    last-resort branch is what actually guarantees liveness.

    The pattern already exists and already works ~60 lines above for exhausted
    components: enumerate the constraint into the context, re-ask once, then
    substitute deterministically.
    """
    mem = Memory(path=str(tmp_path / "memory.json"))
    _hard_ban(mem)

    root = Node(id="root", parent_id=None, code_path="baseline.py")
    llm = _FakeDiagLLM(_bpr_hypothesis(3))      # insists, every time
    diag, cands, counters = _build(root, mem, llm)

    assert len(llm.contexts) >= 2, (
        "the dedup filter emptied the candidate list and the proposer was never "
        "re-asked")
    assert cands, (
        "an iteration whose every proposal was dropped must still emit a "
        "probation node — zero candidates closes the parent, and in the "
        "recorded run the parent was the root")

    # The drop was invisible: no node, no ledger entry, no log line.
    dropped = diag.get("dropped_by_dedup")
    assert dropped, "every dropped hypothesis must be recorded"
    assert len(dropped) >= 3
    assert all(d.get("mechanism") for d in dropped)
    assert {d.get("family") for d in dropped} == {BANNED_FAMILY}

    # Section 2.5 of the problem statement requires per-iteration error and
    # recovery events, so an unlogged widening event is a missing deliverable.
    assert counters.dedup_starved == 1


def test_the_banned_families_are_not_in_the_diagnosis_context(tmp_path):
    """The diagnostician was asked to avoid a constraint it could not see.

    `_ask`'s context carried parent / history / iter_history / improvement_score
    / ablations / tried / component_ledger / exhausted_components — and nothing
    at all about which mechanism families memory had already retired. So the
    model proposed into a banned family, the filter silently deleted the result,
    and neither side ever learned anything.
    """
    mem = Memory(path=str(tmp_path / "memory.json"))
    _hard_ban(mem)

    root = Node(id="root", parent_id=None, code_path="baseline.py")
    llm = _FakeDiagLLM(_bpr_hypothesis(3))
    _build(root, mem, llm)

    first = llm.contexts[0]
    assert BANNED_FAMILY in (first.get("refuted_families") or []), (
        "the blocked families must be named in the FIRST diagnose context, not "
        "only in the re-ask")
    legal = first.get("legal_families") or []
    assert legal, "the legal set has to be enumerated, not implied"
    assert BANNED_FAMILY not in legal

    # The re-ask has to name the constraint explicitly, the way the exhausted
    # component path does.
    refusal = llm.contexts[-1].get("refusal") or ""
    assert BANNED_FAMILY in refusal
    assert any(f in refusal for f in legal), \
        "the refusal must name the legal set, not just the ban"


# --------------------------------------------------------------------------- #
#  T1.4 — probationary, run-scoped bans over a repaired evidence store         #
# --------------------------------------------------------------------------- #
def test_preseeded_dead_ends_reach_an_existing_memory_file(tmp_path):
    """`_preseed` ran only `if not self.entries`, so the two LightGBM
    refutations added after the store was created exist on NO machine that
    already has a memory.json.

    Grep the live file for `lgb` and you get zero hits out of 65 entries — which
    means the single most useful measured fact in the repo (a user x author pair
    occurs 1.07 times in train, so count features cannot work here) is missing
    from the thing built to remember it.

    Meanwhile 61 of those 65 entries are prose hashes from a RETIRED fingerprint
    scheme: unmatchable forever, never collected, linearly scanned on every
    lookup.
    """
    path = tmp_path / "memory.json"

    # A store as it exists in the wild: created before the lgb preseeds landed,
    # and full of dead prose hashes from the retired scheme.
    stale = [
        {"fingerprint": ["pointwise_logloss", "uniform", "cwm_13field", "pure"],
         "architecture": "FM", "loss": "pointwise_logloss", "sampler": "uniform",
         "split": "valid+test", "seed_count": 3,
         "confidence_interval": [0.593, 0.595], "code_hash": "preseed_cwm13",
         "evidence_type": "refuted_under_context", "note": "CWM 13-field"},
    ] + [
        {"fingerprint": ["mechanism_hash", f"{i:016x}", "", ""],
         "architecture": "FM", "loss": "x", "sampler": "uniform",
         "split": "valid_search", "seed_count": 0, "confidence_interval": None,
         "code_hash": f"n{i}", "evidence_type": "inconclusive", "note": "prose"}
        for i in range(61)
    ]
    path.write_text(json.dumps(stale))

    m = Memory(path=str(path))
    blob = json.dumps([e.fingerprint for e in m.entries])

    assert "lgb_train_aggregates" in blob, (
        "the LightGBM refutations must be merged into an EXISTING store, not "
        "only written into an empty one")
    assert "lgb_plus_oof_fm_score" in blob
    # The measured fact itself, not just the fingerprint.
    assert any("1.07 times" in e.note for e in m.entries)

    # All four preseeds, exactly once each — a merge, not an append.
    for fp in [("pointwise_logloss", "uniform", "cwm_13field", "pure"),
               ("pointwise_logloss", "uniform", "5field_baseline", "pure_k_sweep"),
               ("lambdarank", "uniform", "lgb_train_aggregates", "pure"),
               ("binary_logloss", "uniform", "lgb_plus_oof_fm_score", "pure")]:
        n = sum(1 for e in m.entries if tuple(e.fingerprint) == fp)
        assert n == 1, f"{fp} appears {n} times; the merge must be by fingerprint"

    # The retired scheme is collected, not scanned forever.
    assert not [e for e in m.entries if tuple(e.fingerprint)[0] == "mechanism_hash"], \
        "61 unmatchable prose hashes were linearly scanned on every lookup"

    # Every surviving entry declares the scheme it was fingerprinted under, so
    # the next scheme change can collect these the same way.
    assert all(e.scheme for e in m.entries)


def test_a_single_refutation_is_probation_not_a_ban(tmp_path):
    """One categorical-field experiment must not retire a whole family.

    The two bans in the recorded run measured -0.0065 and -0.0050 against a
    paired noise floor of 0.0012, so they are sound evidence about the
    IMPLEMENTATIONS they tested. The unsound step is generalising one of them to
    a family whose token list includes the bare word `sequence` — which also
    covers DIN, target attention, history pooling and session features, none of
    which ran.
    """
    mem = Memory(path=str(tmp_path / "memory.json"))
    mem.record(_refutation(BANNED_FP, "cand_a"))

    assert not mem.is_blocked(BANNED_FP), \
        "one scored refutation is probation, not a hard block"
    assert BANNED_FAMILY in mem.probationary_families(), \
        "probation has to be legible to the prompt, since it is not a filter"

    # A store containing one family refutation still yields candidates.
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    llm = _FakeDiagLLM(_bpr_hypothesis(2))
    diag, cands, counters = _build(root, mem, llm)
    # ONE candidate, not two: both proposals are in the same family, and T2.7's
    # diversity gate allows one candidate per family per iteration. 4 of the 5
    # candidates in the recorded run were the same idea, so an iteration that
    # runs two members of one family measures one thing twice. The point of this
    # test is that a PROBATIONARY family is still proposable at all.
    assert len(cands) == 1, "a probationary family is still proposable"
    assert cands[0].hypothesis["mechanism"] in [
        h["mechanism"] for h in _bpr_hypothesis(2)]
    assert counters.dedup_starved == 0, "nothing was starved, so nothing widened"
    # The second proposal WAS recorded as dropped — but for batch diversity, not
    # for the refutation. The distinction is the point: a probationary family
    # must not be filtered out as refuted.
    dropped = diag.get("dropped_by_dedup") or []
    assert all(d["reason"] == "duplicate_family_in_batch" for d in dropped), \
        f"a probationary family must not be dropped as refuted: {dropped}"

    # A second INDEPENDENT scored refutation is what makes it a block.
    mem.record(_refutation(BANNED_FP, "cand_b"))
    assert mem.is_blocked(BANNED_FP)
    assert BANNED_FAMILY not in mem.probationary_families()

    # Two records of the SAME measurement are not two refutations.
    mem2 = Memory(path=str(tmp_path / "memory2.json"))
    mem2.record(_refutation(BANNED_FP, "cand_a"))
    mem2.record(_refutation(BANNED_FP, "cand_a"))
    assert not mem2.is_blocked(BANNED_FP), \
        "independence is by measurement, not by row count"
