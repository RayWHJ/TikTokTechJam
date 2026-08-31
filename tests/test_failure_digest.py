"""T3.2 — the run-wide failure digest, keyed on (file, exception, line).

WHY IT IS NOT THE ANCESTOR CHAIN. `driver._ancestor_chain` walks ancestors only,
and correctly so: a descendant inherits its parent's staged code. But the two
identical `encode` crashes in the recorded run were SIBLINGS — both children of
the root in iteration 2, two candidates that extended `raw(x)` without extending
`FIELDS` and died with the same `IndexError` at the same line. Neither is an
ancestor of the other, so no ancestor walk could ever have shown either one the
other's traceback. The cheapest learning signal a search can get was discarded.
"""
import json

import pytest

from codegen import prompts
from orchestrator import driver
from orchestrator.driver import FailureDigest, _failure_signature


def _tb(file="data.py", line=193, exc="IndexError", cand="ab12"):
    return (f"Traceback (most recent call last):\n"
            f'  File "/tmp/candidate_{cand}/baseline.py", line 300, in <module>\n'
            f"    main()\n"
            f'  File "/tmp/candidate_{cand}/{file}", line {line}, in encode\n'
            f"    X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]\n"
            f"{exc}: index 5 is out of bounds for axis 1 with size 5\n")


# --------------------------------------------------------------------------- #
#  The signature                                                              #
# --------------------------------------------------------------------------- #
def test_the_signature_is_the_deepest_frame_not_the_outermost():
    """The deepest frame says what to FIX; the outer frames are the call path and
    are identical for every candidate, so keying on them would collapse every
    failure in the run into one bucket."""
    assert _failure_signature(_tb()) == ("data.py", "IndexError", 193)


def test_the_signature_ignores_the_staging_directory():
    """Every candidate is staged under its own temp dir, so keying on the full
    path would make two identical failures look distinct — which is the whole
    thing this fixes."""
    a = _failure_signature(_tb(cand="aaaa"))
    b = _failure_signature(_tb(cand="bbbb"))
    assert a == b == ("data.py", "IndexError", 193)


@pytest.mark.parametrize("log", [
    "", "killed after 240s", "no traceback here at all",
    "Traceback (most recent call last):\n  no frames\n",
])
def test_a_log_without_a_traceback_has_no_signature(log):
    """A timeout has no signature and must not be filed under a fake one — it
    would pollute a real signature's bucket with an unrelated failure."""
    assert _failure_signature(log) is None


def test_the_smoke_stages_assertion_is_a_recognised_signature():
    """After T1.6 the digest fills from smoke rejections, which cost ~0.06s
    instead of 240s — so the signal accumulates at near-zero price."""
    log = ('  File "/tmp/cand/data.py", line 271, in encode\n'
           "    assert n_raw <= len(fields), (\n"
           "AssertionError: raw(x) returns 6 values but there are 5 fields\n")
    assert _failure_signature(log) == ("data.py", "AssertionError", 271)


def test_different_lines_in_the_same_file_are_different_signatures():
    d = FailureDigest()
    d.record(_tb(line=193), node_id="n1", mechanism="m1")
    d.record(_tb(line=250), node_id="n2", mechanism="m2")
    assert len(d.by_signature) == 2
    assert d.matching(_tb(line=193), exclude_node="zz") == \
        [e for e in d.by_signature[("data.py", "IndexError", 193)]]


# --------------------------------------------------------------------------- #
#  THE case it exists for: siblings                                           #
# --------------------------------------------------------------------------- #
def test_a_sibling_failure_is_visible_where_the_ancestor_chain_shows_nothing():
    """The recorded run's two identical encode crashes, reconstructed."""
    from orchestrator.node import Node

    root = Node(id="root", parent_id=None, code_path="baseline.py",
                operation="draft")
    a = Node(id="a", parent_id="root", code_path="baseline.py",
             hypothesis={"mechanism": "add prev_video_id to FIELDS"})
    b = Node(id="b", parent_id="root", code_path="baseline.py",
             hypothesis={"mechanism": "add prev_author_id to FIELDS"})

    # The ancestor chain of b contains only the root — a is invisible to it.
    chain = driver._ancestor_chain(b, [root, a, b])
    assert [e["id"] for e in chain] == ["root"]
    assert "a" not in [e["id"] for e in chain]

    # The digest sees it.
    d = FailureDigest()
    d.record(_tb(cand="a"), node_id="a",
             mechanism=a.hypothesis["mechanism"], stage="execute")
    prior = d.matching(_tb(cand="b"), exclude_node="b")
    assert [e["node_id"] for e in prior] == ["a"]
    assert "prev_video_id" in prior[0]["mechanism"]


def test_a_node_never_sees_its_own_failure_as_prior_art():
    d = FailureDigest()
    d.record(_tb(), node_id="n1", mechanism="m")
    assert d.matching(_tb(), exclude_node="n1") == []
    assert len(d.matching(_tb(), exclude_node="other")) == 1


def test_a_new_signature_yields_no_prior_failures():
    """Early in a run there is nothing useful to say, and the prompt must be
    unchanged rather than carrying an empty section."""
    d = FailureDigest()
    assert d.matching(_tb()) == []
    assert prompts.build_digest_block([]) == ""
    assert prompts.build_digest_block(None) == ""


# --------------------------------------------------------------------------- #
#  Repair outcomes — what makes it more than a list of tracebacks             #
# --------------------------------------------------------------------------- #
def test_whether_a_repair_worked_is_carried_and_rendered():
    d = FailureDigest()
    sig = d.record(_tb(), node_id="n1", mechanism="lag field")
    d.note_repair(sig, node_id="n1", diff="-  return [x[1]]\n+  return [x[1], x[2]]")
    d.resolve(sig, node_id="n1", repaired=False)

    prior = d.matching(_tb(), exclude_node="n2")
    assert prior[0]["repaired"] is False
    assert "return [x[1], x[2]]" in prior[0]["repair_attempted"]

    block = prompts.build_digest_block(prior)
    assert "did NOT fix it" in block
    assert "do something different" in block


def test_a_successful_repair_is_rendered_as_the_cheapest_answer():
    d = FailureDigest()
    sig = d.record(_tb(), node_id="n1", mechanism="lag field")
    d.resolve(sig, node_id="n1", repaired=True)
    block = prompts.build_digest_block(d.matching(_tb(), exclude_node="n2"))
    assert "FIXED it" in block
    assert "cheapest correct answer" in block


def test_an_unresolved_repair_says_so_rather_than_guessing():
    d = FailureDigest()
    d.record(_tb(), node_id="n1", mechanism="m")
    block = prompts.build_digest_block(d.matching(_tb(), exclude_node="n2"))
    assert "outcome unknown" in block


def test_the_digest_is_capped_like_the_attempt_ledger():
    """This text goes into a repair prompt that already carries a whole source
    file, which is the same reason LEDGER_MAX_ENTRIES exists."""
    d = FailureDigest(max_per_signature=3)
    for i in range(10):
        d.record(_tb(), node_id=f"n{i}", mechanism="m")
    entries = d.by_signature[("data.py", "IndexError", 193)]
    assert len(entries) == 3
    # The NEWEST are kept: the most recent repairs are the relevant ones.
    assert [e["node_id"] for e in entries] == ["n7", "n8", "n9"]
    assert driver.DIGEST_MAX_PER_SIGNATURE == 4


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #
def test_the_summary_reports_only_repeated_signatures():
    """A signature seen once is a bug; seen three times it is a systematic gap in
    what the writer is being told."""
    d = FailureDigest()
    d.record(_tb(line=193), node_id="n1", mechanism="m")
    d.record(_tb(line=193), node_id="n2", mechanism="m")
    d.record(_tb(line=999), node_id="n3", mechanism="m")   # seen once

    summary = d.summary()
    assert len(summary) == 1
    assert summary[0]["line"] == 193 and summary[0]["count"] == 2
    assert summary[0]["nodes"] == ["n1", "n2"]
    assert summary[0]["repairs_that_worked"] == 0


def test_the_summary_is_ordered_by_how_often_a_signature_recurs():
    d = FailureDigest()
    for i in range(2):
        d.record(_tb(line=10), node_id=f"a{i}", mechanism="m")
    for i in range(4):
        d.record(_tb(line=20), node_id=f"b{i}", mechanism="m")
    assert [r["line"] for r in d.summary()] == [20, 10]


def test_progress_json_carries_the_repeated_failures(monkeypatch, tmp_path):
    """A digest that never reaches the run log cannot be read after the fact."""
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

    # Every candidate fails with the SAME traceback, so the digest must fill and
    # the repair prompt must start carrying prior art.
    seen_priors = []
    real_debug = mock_codegen.debug_and_retry

    def _debug(code_path, error_context, root=".", **kw):
        seen_priors.append(kw.get("prior_failures") or [])
        return real_debug(code_path, error_context, root=root, **kw)

    monkeypatch.setattr(mock_codegen, "debug_and_retry", _debug)
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda code_path, seed, split, wallclock_cap_seconds,
                        root=None, data_dir=None:
                        {"status": "ok",
                         "metrics": {"primary": 0.6, "GAUC": 0.63,
                                     "nDCG@5": 0.57, "per_user": {"u0": 0.6}},
                         "logs": ""}
                        if code_path == "baseline.py" else
                        {"status": "error", "metrics": {}, "logs": _tb()})

    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=2, verbose=False, progress_path=str(progress),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    blob = json.loads(progress.read_text())
    repeats = blob["repeated_failures"]
    assert repeats, "identical failures across candidates were not recorded"
    assert repeats[0]["file"] == "data.py"
    assert repeats[0]["exception"] == "IndexError"
    assert repeats[0]["line"] == 193
    assert repeats[0]["count"] >= 2

    # And the later repair calls actually received the prior art.
    assert any(p for p in seen_priors), \
        "debug_and_retry never received a non-empty prior_failures list"
    assert seen_priors[0] == [], "the FIRST failure has no prior art"


def test_debug_and_retry_accepts_prior_failures_without_breaking_its_contract():
    import inspect

    from codegen.debug import debug_and_retry
    sig = inspect.signature(debug_and_retry)
    assert "prior_failures" in sig.parameters
    assert sig.parameters["prior_failures"].default is None
    assert sig.parameters["prior_failures"].kind is inspect.Parameter.KEYWORD_ONLY
    # The frozen 3-positional form still works and carries no digest section.
    msg = prompts.build_debug_user("baseline.py", "# file\n", "boom")
    assert "SEEN BEFORE" not in msg
