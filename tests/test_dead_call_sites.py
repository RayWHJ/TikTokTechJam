"""T3.5 — the four LLM call sites that never executed, and what was decided.

Four unreachable API paths in a submission judged on technical execution read as
scaffolding rather than as a system. Each decision is recorded here as an
executable assertion, so "we decided X" cannot drift from what the code does.

  1. llm_calls/dedup.py::dedup_fingerprint_match  -> DELETED
  2. codegen/debug.py's KIND_SANITY branch        -> WIRED (extracted)
  3. codegen/report.py::synthesize_report         -> WIRED
  4. llm_calls/refine.py (REFINE_ENABLED=False)   -> KEPT, documented
"""
import json

import pytest

from orchestrator import driver


# --------------------------------------------------------------------------- #
#  1. dedup — DELETED                                                         #
# --------------------------------------------------------------------------- #
def test_the_dedup_escalation_is_gone():
    """Deleted rather than wired because wiring it would have been ACTIVELY
    WRONG, not merely wasteful — see the next test."""
    with pytest.raises(ImportError):
        import llm_calls.dedup            # noqa: F401
    import llm_calls
    assert not hasattr(llm_calls, "dedup_fingerprint_match")


def test_wiring_dedup_in_would_have_escalated_every_distinct_family():
    """The measured reason for deleting it. With family fingerprints of the form
    ("mechanism_family", fam, "", ""), any two DISTINCT families differ in
    exactly one position — and the module's ambiguity threshold was one position.
    So it would have called a model for every family pair and asked whether two
    unrelated mechanisms were the same experiment."""
    from llm_calls.families import ALL_FAMILIES

    fps = [("mechanism_family", f, "", "") for f in ALL_FAMILIES]
    diffs = set()
    for i, a in enumerate(fps):
        for b in fps[i + 1:]:
            diffs.add(sum(1 for x, y in zip(a, b) if x != y))
    assert diffs == {1}, (
        f"every distinct family differs in exactly 1 position; got {diffs}. "
        f"The retired module treated <=1 differing position as 'ambiguous, ask "
        f"the model'.")


def test_nothing_orphaned_by_the_deletion_remains():
    """A deletion that leaves its prompt, validator and routing entry behind is
    not a deletion."""
    from llm_calls import personas, routing, schemas, usage
    assert not hasattr(personas, "DEDUP_SYSTEM_PROMPT")
    assert not hasattr(schemas, "validate_dedup")
    assert not hasattr(usage, "KIND_DEDUP")
    assert "dedup" not in routing.TABLE
    assert "dedup" not in usage.ALL_KINDS


# --------------------------------------------------------------------------- #
#  2. the sanity branch — WIRED                                               #
# --------------------------------------------------------------------------- #
def test_sanity_check_is_callable_without_the_repair_loop():
    """THE reason it was unreachable. `debug_and_retry` runs its repair loop
    FIRST, so calling it on a candidate that succeeded would spend a writer call
    trying to repair working code — the driver could not use it, so it never
    passed `observed_score`, so ORACLE_PRIMARY_CEILING was checkable only from
    there and checked nowhere."""
    import codegen
    assert callable(codegen.sanity_check)


def test_sanity_check_makes_no_model_call_on_a_plausible_score():
    """It has to be free on a normal run, or wiring it in is a tax on every
    promotion."""
    import codegen

    class _Forbid:
        def complete(self, *a, **k):
            raise AssertionError("no model call should be made")

    assert codegen.sanity_check("+ x = 1", hypothesis={"mechanism": "m"},
                                observed_score=0.60, history=[0.5946],
                                client=_Forbid()) is None
    # And nothing at all when no score was supplied.
    assert codegen.sanity_check("+ x = 1", client=_Forbid()) is None


def test_sanity_check_fires_above_the_oracle_ceiling():
    import codegen
    from codegen.constants import ORACLE_PRIMARY_CEILING

    class _Fake:
        def complete(self, *a, **k):
            return json.dumps({"implements_hypothesis": False,
                               "leak_suspected": True, "reasoning": "impossible"})

    out = codegen.sanity_check("+ x = 1", hypothesis={"mechanism": "m"},
                               observed_score=ORACLE_PRIMARY_CEILING + 0.01,
                               history=[0.5946], client=_Fake())
    assert out and out["leak_suspected"] is True


def test_unparseable_sanity_output_is_not_read_as_approval():
    import codegen

    class _Junk:
        def complete(self, *a, **k):
            return "I think it's probably fine?"

    out = codegen.sanity_check("+ x = 1", hypothesis={"mechanism": "m"},
                               observed_score=0.99, history=[0.5946],
                               client=_Junk())
    assert out["leak_suspected"] is True, \
        "a leak check with no opinion must not read as a pass"


def test_the_driver_reaches_it_on_the_promotion_path(monkeypatch):
    """Wired, i.e. actually called — which is the whole point of T3.5."""
    from orchestrator.counters import Counters
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.node import Node

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda *a, **k: {"status": "ok", "logs": "",
                                         "metrics": {"primary": 0.4840}})
    called = []
    monkeypatch.setattr(mock_codegen, "sanity_check",
                        lambda *a, **k: called.append(k) or None,
                        raising=False)

    c = Node(id="c1", parent_id="root", code_path="/tmp/c/baseline.py",
             hypothesis={"mechanism": "m"})
    assert driver._leak_check(c, "/tmp/c", Counters(), confirm_primary=0.61,
                              history=[0.5946], verbose=False) is None
    assert called, "sanity_check was not reached from the promotion path"
    assert called[0]["observed_score"] == 0.61
    assert called[0]["history"] == [0.5946]


def test_the_sanity_opinion_is_advisory_not_a_veto(monkeypatch):
    """The LLM auditor flagged 5 of 5 candidates in the recorded run, including
    the training loop's own `y`. An opinion with that track record is recorded,
    not obeyed — the deterministic controls are the gatekeepers."""
    from orchestrator.counters import Counters
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.node import Node

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda *a, **k: {"status": "ok", "logs": "",
                                         "metrics": {"primary": 0.4840}})
    monkeypatch.setattr(mock_codegen, "sanity_check",
                        lambda *a, **k: {"leak_suspected": True,
                                         "reasoning": "vibes"},
                        raising=False)

    c = Node(id="c1", parent_id="root", code_path="/tmp/c/baseline.py",
             hypothesis={"mechanism": "m"})
    reason = driver._leak_check(c, "/tmp/c", Counters(), confirm_primary=0.61,
                                history=[0.5946], verbose=False)
    assert reason is None, "an advisory opinion must not block a promotion"
    assert c.diagnosis["sanity"]["leak_suspected"] is True, \
        "but it must be recorded"


# --------------------------------------------------------------------------- #
#  3. synthesize_report — WIRED                                               #
# --------------------------------------------------------------------------- #
def test_a_run_writes_the_report_beside_its_progress_file(monkeypatch, tmp_path):
    """It produces a REQUIRED deliverable and was wired into nothing, so the
    write-up was being assembled by hand from a file the search wrote."""
    import itertools
    import random

    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    seen = {}

    def _report(log, **k):
        seen["log"] = log
        return "# Report\n\nSome markdown.\n"

    monkeypatch.setattr(mock_codegen, "synthesize_report", _report)

    random.seed(0)
    driver.run(max_iters=1, verbose=False,
               progress_path=str(tmp_path / "progress.json"),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    report = tmp_path / "report.md"
    assert report.exists(), "the report was not written"
    assert "Some markdown" in report.read_text()

    # The log it was handed has to carry the facts a write-up needs.
    log = seen["log"]
    for key in ("baseline_primary", "global_best", "champion", "counters",
                "models", "iterations", "iters_completed", "promotions",
                "repeated_failures", "metric_ceilings"):
        assert key in log, f"the report log is missing {key}"
    assert log["champion"]["is_baseline"] in (True, False)


def test_the_report_never_fails_the_run(monkeypatch, tmp_path):
    """It is the last thing that happens; losing it must not lose the numbers."""
    import itertools
    import random

    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")
    monkeypatch.setattr(mock_codegen, "synthesize_report",
                        lambda log, **k: (_ for _ in ()).throw(
                            RuntimeError("report model down")),
                        raising=False)

    random.seed(0)
    res = driver.run(max_iters=1, verbose=False,
                     progress_path=str(tmp_path / "progress.json"),
                     memory_path=str(tmp_path / "m.json"),
                     champion_dir=str(tmp_path / "ch"),
                     root_baseline_path=str(tmp_path / "rb.json"),
                     confirm_baseline_path=str(tmp_path / "cb.json"))
    assert "global_best" in res
    assert (tmp_path / "progress.json").exists()
    assert not (tmp_path / "report.md").exists()


def test_a_run_that_persists_nothing_writes_no_report(monkeypatch, tmp_path):
    import random

    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    random.seed(0)
    driver.run(max_iters=1, verbose=False, progress_path=None,
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))
    assert list(tmp_path.glob("report.md")) == []


# --------------------------------------------------------------------------- #
#  4. refine — KEPT, and the decision is recorded                             #
# --------------------------------------------------------------------------- #
def test_refine_is_still_off_and_still_intact():
    """KEPT rather than deleted. The plan recommends deleting it, and the case is
    real — it is an unreachable subsystem. But it is a DOCUMENTED kill switch
    with a recorded measurement behind it (the search regressed with refine in
    the loop), flipping one flag restores it exactly, and its two test modules
    skip themselves while it is off. Deleting `llm_calls/refine.py`,
    `codegen/ablations.py`, `orchestrator/ablation_harness.py`, `write_refine`
    and two test modules the day before submission is a large irreversible change
    with no score upside, so the decision is recorded here instead.
    """
    assert driver.REFINE_ENABLED is False
    # Intact, so the flag really is all that stands between off and on.
    import codegen
    from llm_calls import refine  # noqa: F401
    from orchestrator import ablation_harness  # noqa: F401
    assert callable(codegen.write_refine)
    assert codegen.ABLATIONS


def test_the_refine_tests_skip_themselves_rather_than_failing():
    """Which is what makes the kill switch honest: nothing pretends to pass."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_mlestar.py", "-q", "--no-header"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:]
    assert "skipped" in r.stdout, r.stdout[-400:]
