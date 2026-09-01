"""T3.4 — the leak gap the auditor cannot see.

`codegen/constants.py::AUXILIARY_SIGNALS` lists is_click, is_like and the rest but
OMITTED `long_view` — the actual label. So `prev_long_view`, the one genuinely
risky construction in the recorded run, passed the deterministic gate cleanly.

Meanwhile the advisory LLM auditor flagged 5 of 5 candidates, including "y is
being used within the step function", which is the training loop. Its
signal-to-noise is zero at ~4,500 input tokens per candidate. A permutation
control is deterministic, costs one run per promotion candidate, and catches leaks
the auditor cannot articulate.
"""
import os

import pytest

from codegen import sandbox
from codegen.constants import (AUXILIARY_SIGNALS, LABEL_COLUMNS,
                               NON_CAUSAL_COLUMNS, ORACLE_PRIMARY_CEILING)
from codegen.gate import pre_execution_gate


# --------------------------------------------------------------------------- #
#  The constant that was missing                                              #
# --------------------------------------------------------------------------- #
def test_the_label_is_now_named_somewhere():
    """It was in neither list, which is why nothing could check it."""
    assert "long_view" in LABEL_COLUMNS
    assert "long_view" not in AUXILIARY_SIGNALS, \
        "the label is not an auxiliary signal — it is the target"
    assert "long_view" not in NON_CAUSAL_COLUMNS


# --------------------------------------------------------------------------- #
#  Rule 2b — the label flowing into features                                  #
# --------------------------------------------------------------------------- #
def test_the_recorded_runs_risky_construction_is_now_blocked():
    """`prev_long_view` appended to FIELDS with no marker."""
    diff = ("+FIELDS.append('prev_long_view')\n"
            "+    return [x[1], x[2], x[3], x[4], str(x[6])]\n")
    r = pre_execution_gate(diff)
    assert not r["pass"]
    assert any("long_view" in x for x in r["reasons"])


def test_a_word_boundary_would_have_missed_it():
    """`\\blong_view\\b` cannot match inside `prev_long_view` — the preceding `_`
    is a word character. Matching the substring is what catches the family of
    label-derived names."""
    import re
    assert not re.search(r"\blong_view\b", "prev_long_view")
    from codegen.gate import _LABEL_RE
    assert _LABEL_RE["long_view"].search("prev_long_view")


@pytest.mark.parametrize("line", [
    "+    X = np.column_stack([X, user_long_view_rate])",
    "+FIELDS = FIELDS + ['long_view_lag']",
    "+    vocabs[5][str(x[6])] = long_view_bucket",
    "+EXTRA_FIELDS.append(('long_view_prev', my_own_lag_fn))",
])
def test_label_derived_features_are_blocked_without_a_marker(line):
    r = pre_execution_gate(line + "\n")
    assert not r["pass"], f"not blocked: {line}"


@pytest.mark.parametrize("diff", [
    # The tested primitive, whose whole contract is that it never reads the row
    # it describes.
    "+EXTRA_FIELDS.append(('prev_long_view', "
    "lambda rows: prev_value_within_user(rows, key=6)))\n",
    # The explicit marker, same rule as the non-causal columns.
    "+    # point_in_time=True\n+    X[n, 5] = prev_long_view_id\n",
])
def test_a_properly_lagged_label_is_allowed(diff):
    """A LAGGED label is legitimate and is a direction the prompts actively
    recommend, so the rule demands a marker rather than forbidding the name."""
    r = pre_execution_gate(diff)
    assert r["pass"], r["reasons"]


@pytest.mark.parametrize("line", [
    "+            y[n] = x[6]",
    "+    ytr = np.array([r[6] for r in rows])",
    "+        loss = self.step(X, y)",
    "+    A = data.aux_targets(splits)",
    "+from codegen.constants import LABEL_COLUMNS",
    "+    k = 32",
])
def test_the_label_used_as_the_target_is_not_a_leak(line):
    """`y[n] = x[6]` in encode() and `y` in a training step ARE the target, which
    is what the label is for. A rule that blocked those would block the baseline
    itself."""
    r = pre_execution_gate(line + "\n")
    assert r["pass"], r["reasons"]


def test_the_new_rule_fires_on_no_line_of_the_real_repo_files():
    """The strongest false-positive check available for rule 2b: every line of
    baseline.py, data.py and evaluate.py, treated as added.

    Scoped to `_check_label_as_feature` rather than the whole gate on purpose.
    The full gate DOES flag these files when the entire body is presented as
    added lines — baseline.py legitimately contains `if split == 'test':` (rule 1)
    and `rng.normal(...)` in FM.__init__ (rule 5). That is a property of feeding a
    whole file as a diff, not a T3.4 regression: a real unified diff marks only
    CHANGED lines with `+`, and T3.1's scoped edits mark fewer still.
    """
    from codegen.gate import _check_label_as_feature

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for name in ("baseline.py", "data.py", "evaluate.py"):
        with open(os.path.join(root, name)) as fh:
            added = list(enumerate(fh.read().splitlines()))
        reasons = _check_label_as_feature(added)
        assert reasons == [], f"{name}: rule 2b false positives: {reasons[:3]}"


def test_the_gate_still_blocks_the_label_leak_inside_a_realistic_diff():
    """A real unified diff, with context lines, hunk headers and the `+` markers
    the gate actually parses."""
    diff = (
        "--- a/data.py\n"
        "+++ b/data.py\n"
        "@@ -14,6 +14,7 @@\n"
        " LABEL = 'long_view'\n"
        " BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']\n"
        "+FIELDS = BASE_FIELDS + ['long_view_rate']\n"
        " \n"
        "@@ -180,6 +181,7 @@\n"
        "     def raw(x):\n"
        "+        return [x[1], x[2], x[3], x[4], '0', str(x[6])]\n"
    )
    r = pre_execution_gate(diff)
    assert not r["pass"]
    assert any("long_view" in x for x in r["reasons"])


# --------------------------------------------------------------------------- #
#  The label-permutation control                                              #
# --------------------------------------------------------------------------- #
def test_the_shim_is_valid_python_and_only_activates_on_the_env_var():
    compile(sandbox._PERMUTE_SHIM, "<shim>", "exec")
    assert "CODEGEN_PERMUTE_LABELS" in sandbox._PERMUTE_SHIM
    # Guarded, so a normal run is byte-identical to one without the shim.
    assert sandbox._PERMUTE_SHIM.index("if _pc_os.environ.get") < \
        sandbox._PERMUTE_SHIM.index("def evaluate")


def test_the_shim_shuffles_within_a_user_and_preserves_positive_counts():
    """Within-user is what makes it the right null for THIS metric: every user's
    positive count is preserved, so GAUC's `0 < npos < len(labs)` filter selects
    the same users and nDCG's ideal DCG is unchanged, while the label-to-score
    association is destroyed."""
    ns = {}
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "evaluate.py")).read()
    os.environ["CODEGEN_PERMUTE_LABELS"] = "1"
    try:
        exec(compile(src + "\n\n" + sandbox._PERMUTE_SHIM, "<ev>", "exec"), ns)
        users = ["u1"] * 6 + ["u2"] * 4
        labels = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        scores = list(range(10))
        captured = {}
        real = ns["_pc_real_evaluate"]

        def _spy(uu, ll, ss, k=5):
            captured["labels"] = list(ll)
            return real(uu, ll, ss, k=k)

        ns["_pc_real_evaluate"] = _spy
        ns["evaluate"](users, labels, scores)
    finally:
        os.environ.pop("CODEGEN_PERMUTE_LABELS", None)

    got = captured["labels"]
    assert got != labels, "the labels were not shuffled at all"
    # Per-user positive counts preserved exactly.
    for lo, hi in ((0, 6), (6, 10)):
        assert sum(got[lo:hi]) == sum(labels[lo:hi]), \
            "a user's positive count changed — the null is wrong"


def test_execute_exposes_the_control_and_leaves_the_repo_evaluate_untouched():
    import inspect
    assert "permute_labels" in inspect.signature(sandbox.execute).parameters
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    body = open(os.path.join(root, "evaluate.py")).read()
    assert "CODEGEN_PERMUTE_LABELS" not in body, \
        "evaluate.py is a frozen deliverable; the shim goes in the WORKDIR copy"


def test_the_shim_is_appended_after_the_candidate_is_staged(tmp_path):
    """A candidate must not be able to disable its own leak control."""
    import shutil
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    stage = tmp_path / "cand"
    stage.mkdir()
    for n in ("data.py", "evaluate.py", "baseline.py", "submit.py"):
        shutil.copy2(os.path.join(root, n), stage / n)
    # A candidate that tried to neuter the control by rewriting evaluate.py.
    (stage / "evaluate.py").write_text(
        "def evaluate(user_ids, labels, scores, k=5):\n"
        "    return {'primary': 0.99}\n")

    work, _base = sandbox._prepare_workdir(str(stage / "baseline.py"),
                                            str(stage), permute_labels=True)
    got = open(os.path.join(work, "evaluate.py")).read()
    assert "CODEGEN_PERMUTE_LABELS" in got, "the shim was not appended"
    assert got.index("def evaluate") < got.index("CODEGEN_PERMUTE_LABELS"), \
        "the shim must WRAP whatever evaluate the candidate left behind"


def test_the_control_is_off_by_default(tmp_path):
    import shutil
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    stage = tmp_path / "cand"
    stage.mkdir()
    for n in ("data.py", "evaluate.py", "baseline.py", "submit.py"):
        shutil.copy2(os.path.join(root, n), stage / n)
    work, _ = sandbox._prepare_workdir(str(stage / "baseline.py"), str(stage))
    assert "CODEGEN_PERMUTE_LABELS" not in \
        open(os.path.join(work, "evaluate.py")).read()


# --------------------------------------------------------------------------- #
#  The driver's promotion gate                                                #
# --------------------------------------------------------------------------- #
def _node():
    from orchestrator.node import Node
    return Node(id="c1", parent_id="root", code_path="/tmp/cand/baseline.py",
                hypothesis={"mechanism": "m"})


def _driver_with_execute(monkeypatch, fn):
    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(mock_codegen, "execute", fn)
    return driver


def test_an_impossible_score_is_refused_without_even_running_the_control(
        monkeypatch):
    """`ORACLE_PRIMARY_CEILING` already existed and was reachable ONLY through
    debug.py's sanity branch, which the driver never triggers because it never
    passes `observed_score`. So nothing in the loop ever checked it."""
    from orchestrator.counters import Counters
    calls = []
    driver = _driver_with_execute(
        monkeypatch,
        lambda *a, **k: calls.append(k) or {"status": "ok", "logs": "",
                                            "metrics": {"primary": 0.4}})
    reason = driver._leak_check(_node(), "/tmp/cand", Counters(),
                                confirm_primary=0.95, verbose=False)
    assert reason and "oracle ceiling" in reason
    assert calls == [], "no run should be spent on a physically impossible score"
    assert ORACLE_PRIMARY_CEILING == 0.8645


def test_a_score_that_survives_label_shuffling_is_refused(monkeypatch):
    from orchestrator.counters import Counters
    driver = _driver_with_execute(
        monkeypatch,
        lambda *a, **k: {"status": "ok", "logs": "",
                         "metrics": {"primary": 0.61}})   # unchanged by shuffle
    c = _node()
    reason = driver._leak_check(c, "/tmp/cand", Counters(),
                                confirm_primary=0.61, verbose=False)
    assert reason and "SHUFFLED" in reason
    assert c.permuted_primary == 0.61


def test_a_clean_candidate_passes_the_control(monkeypatch):
    from orchestrator.counters import Counters
    driver = _driver_with_execute(
        monkeypatch,
        lambda *a, **k: {"status": "ok", "logs": "",
                         "metrics": {"primary": 0.4840}})
    c = _node()
    counters = Counters()
    assert driver._leak_check(c, "/tmp/cand", counters,
                              confirm_primary=0.61, verbose=False) is None
    assert c.permuted_primary == 0.4840
    assert counters.permutation_runs == 1


def test_a_control_that_could_not_run_is_inconclusive_not_a_pass(monkeypatch):
    """Promoting on "the check crashed" is how a leak ships."""
    from orchestrator.counters import Counters
    driver = _driver_with_execute(
        monkeypatch,
        lambda *a, **k: {"status": "timeout", "logs": "", "metrics": {}})
    reason = driver._leak_check(_node(), "/tmp/cand", Counters(),
                                confirm_primary=0.61, verbose=False)
    assert reason and "unverified" in reason


def test_the_bar_sits_between_the_measured_null_and_the_real_baseline():
    """Calibrated by measurement, not guessed. Running the unmodified baseline on
    valid_search with labels shuffled within each user gives primary 0.4840 and
    GAUC 0.4998 against a theoretical 0.5."""
    from orchestrator import driver
    assert driver.PERMUTED_BASELINE_PRIMARY == 0.4840
    assert driver.PERMUTATION_MAX_PRIMARY == 0.55
    assert driver.PERMUTED_BASELINE_PRIMARY < driver.PERMUTATION_MAX_PRIMARY
    # And clearly below the real baseline (~0.5936), so a genuine score is never
    # mistaken for a leak.
    assert driver.PERMUTATION_MAX_PRIMARY < 0.59


def test_the_control_runs_on_the_promotion_path_only(monkeypatch, tmp_path):
    """One extra run per PROMOTION candidate, not per candidate. The whole point
    is that certainty is worth paying for exactly where a submission comes from."""
    import itertools
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    seen = []
    real = mock_codegen.execute

    def _spy(code_path, seed, split, wallclock_cap_seconds, root=None,
             data_dir=None, **kw):
        seen.append((split, bool(kw.get("permute_labels"))))
        return real(code_path, seed, split, wallclock_cap_seconds, root=root,
                    data_dir=data_dir, **kw)

    monkeypatch.setattr(mock_codegen, "execute", _spy)
    random.seed(0)
    res = driver.run(max_iters=3, verbose=False,
                     progress_path=str(tmp_path / "p.json"),
                     memory_path=str(tmp_path / "m.json"),
                     champion_dir=str(tmp_path / "ch"),
                     root_baseline_path=str(tmp_path / "rb.json"),
                     confirm_baseline_path=str(tmp_path / "cb.json"))

    permuted = [s for s in seen if s[1]]
    # Never on valid_search — a control there would double the triage bill for
    # candidates that will never be promoted.
    assert all(split == "valid_confirm" for split, _ in permuted), permuted
    assert res["counters"].permutation_runs == len(permuted)
    assert res["counters"].permutation_runs <= res["counters"].scorer_queries[
        "valid_confirm"]
