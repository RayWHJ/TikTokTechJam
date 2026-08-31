"""T1.6 — the sub-second smoke stage, and the driver wiring that spends the
cheap budget instead of the expensive one.

Every execution failure in the recorded run was detectable on 200 rows: a shape
mismatch, an `IndexError` in `data.py::encode`, and a mechanism whose per-row
cost was obvious immediately. Each cost a 240 s triage run plus up to two further
240 s repair runs. These tests pin that the check catches those failures, that it
is fast enough to be free, and that a candidate rejected by it never reaches
`codegen.execute`.
"""
import os
import shutil
import time

import pytest

from codegen import smoke as smoke_mod
from codegen.smoke import smoke_check

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAGED = ("data.py", "evaluate.py", "baseline.py", "submit.py")

#: The line in data.py::encode that a candidate edits to add a field.
_RAW_LINE = ("        return [x[1], x[2], x[3], x[4], "
             "str(int(np.searchsorted(edges, x[5])))]")


@pytest.fixture
def candidate(tmp_path):
    """A staged candidate tree: the four root modules, copied, unpatched.

    Mirrors driver._apply_diff_and_stage, which is the state the smoke stage is
    handed.
    """
    d = tmp_path / "candidate"
    d.mkdir()
    for name in _STAGED:
        shutil.copy2(os.path.join(ROOT, name), d / name)
    return d


def _patch(path, old, new):
    src = path.read_text()
    assert old in src, f"anchor not found in {path.name}; update the test"
    path.write_text(src.replace(old, new, 1))


# --------------------------------------------------------------------------- #
#  The happy path, and the budget claim                                       #
# --------------------------------------------------------------------------- #
def test_the_unmodified_baseline_passes_smoke(candidate):
    r = smoke_check(str(candidate))
    assert r["ok"], r["error"]
    assert r["error"] == ""


def test_it_is_cheap_enough_to_be_free(candidate):
    """The whole argument for the stage is the price. A triage run is 240 s;
    this has to be nearer 1 s than 10, or moving the repair loop onto it stops
    being a saving."""
    t0 = time.time()
    smoke_check(str(candidate))
    assert time.time() - t0 < 2.0


def test_it_needs_no_dataset(candidate, monkeypatch):
    """The fixture is synthetic on purpose, so the stage works when
    KuaiRand-Pure/data/ is absent — and so it costs no CSV parsing."""
    monkeypatch.setenv("CODEGEN_DATA_DIR", str(candidate / "nonexistent"))
    monkeypatch.setenv("HARNESS_DATA_DIR", str(candidate / "nonexistent"))
    assert smoke_check(str(candidate))["ok"]


def test_it_exercises_the_real_orchestrator_entry_point(candidate):
    """A proxy check would not catch a candidate whose custom loss crashes. The
    probe calls baseline.run_for_orchestrator, which is exactly what
    codegen.sandbox.execute reaches, so the training path really runs."""
    _patch(candidate / "baseline.py",
           "def run_for_orchestrator(a, split):",
           "def run_for_orchestrator(a, split):\n"
           "    raise RuntimeError('custom loss exploded')")
    r = smoke_check(str(candidate))
    assert not r["ok"]
    assert "custom loss exploded" in r["error"]


# --------------------------------------------------------------------------- #
#  The recorded failures, on 200 rows                                         #
# --------------------------------------------------------------------------- #
def test_extending_raw_without_extending_fields_is_rejected_with_both_lengths(
        candidate):
    """THE acceptance criterion. Two candidates in the recorded run died with
    this identical IndexError at the same line of data.py, 240 s apiece."""
    _patch(candidate / "data.py", _RAW_LINE,
           _RAW_LINE[:-1] + ", x[2]]")     # one extra value out of raw()

    t0 = time.time()
    r = smoke_check(str(candidate))
    elapsed = time.time() - t0

    assert not r["ok"]
    assert elapsed < 2.0, f"took {elapsed:.2f}s; must be a cheap check"
    # Both lengths, so the repair operator knows which side to change.
    assert "6" in r["error"] and "5" in r["error"]
    # And both lists.
    assert "dur_bucket" in r["error"], "the FIELDS list must be in the message"
    assert "raw(x)" in r["error"]
    assert r["stage"] == "encode", \
        "the stage has to name the encoder, not just 'it broke'"


def test_a_shape_mismatch_in_encode_is_rejected(candidate):
    """X wider than the field list: the FM would index its embedding table out
    of range, which on the real split is an IndexError 240 s in."""
    _patch(candidate / "data.py",
           "        X = np.empty((len(rws), len(fields)), dtype=np.int32)",
           "        X = np.empty((len(rws), len(fields) + 2), dtype=np.int32)")
    r = smoke_check(str(candidate))
    assert not r["ok"]
    assert r["stage"] == "encode"
    assert "shape" in r["error"].lower()


def test_a_feature_id_outside_the_embedding_table_is_rejected(candidate):
    """The offsets/field_dims arithmetic is the easiest thing to get subtly
    wrong, and it fails as an IndexError deep in the training loop."""
    _patch(candidate / "data.py",
           "    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)",
           "    offsets = (np.cumsum([0] + field_dims[:-1]) + 1000).astype(np.int32)")
    r = smoke_check(str(candidate))
    assert not r["ok"]
    assert "outside" in r["error"] or "IndexError" in r["error"]


def test_a_syntax_error_is_rejected_at_the_import_stage(candidate):
    _patch(candidate / "data.py", "def encode(splits):", "def encode(splits) ::::")
    r = smoke_check(str(candidate))
    assert not r["ok"]
    assert r["stage"] == "import"
    assert "SyntaxError" in r["error"]


def test_a_candidate_that_stops_emitting_the_metrics_marker_is_rejected(candidate):
    """`codegen.sandbox` parses that line to read the score, so a candidate that
    drops it runs for 240 s and is then unscoreable."""
    _patch(candidate / "baseline.py",
           "    print(f\"{METRICS_MARK} {json.dumps(payload)}\")",
           "    pass")
    r = smoke_check(str(candidate))
    assert not r["ok"]
    assert r["stage"] == "metrics"


def test_a_hang_is_killed_rather_than_inherited(candidate):
    """An unattended run must not be stopped by a candidate that never returns.
    A mechanism that cannot do 200 rows inside the cap cannot do 1.4M either."""
    _patch(candidate / "data.py", "def encode(splits):",
           "def encode(splits):\n    import time as _t\n    _t.sleep(600)")
    t0 = time.time()
    r = smoke_check(str(candidate), timeout_s=2)
    elapsed = time.time() - t0
    assert not r["ok"]
    assert r["stage"] == "timeout"
    assert elapsed < 20, f"the timeout did not take effect ({elapsed:.1f}s)"
    assert "vectorised" in r["error"], \
        "a timeout calls for a cheaper implementation, not a longer cap"


# --------------------------------------------------------------------------- #
#  The registry seam still passes                                             #
# --------------------------------------------------------------------------- #
def test_a_correct_extra_field_registration_passes_smoke(candidate):
    """The other half of T1.5: the seam the prompts now recommend has to be one
    the smoke stage accepts, or the advice is a trap."""
    _patch(candidate / "data.py",
           "EXTRA_FIELDS: List[Tuple[str, Callable]] = []",
           "def _prev_video(rows):\n"
           "    last, out = {}, []\n"
           "    for x in rows:\n"
           "        out.append(last.get(x[1], 'NONE'))\n"
           "        last[x[1]] = x[2]\n"
           "    return out\n\n"
           "EXTRA_FIELDS: List[Tuple[str, Callable]] = "
           "[('prev_video_id', _prev_video)]")
    r = smoke_check(str(candidate))
    assert r["ok"], r["error"]


# --------------------------------------------------------------------------- #
#  It must never be the thing that breaks the run                             #
# --------------------------------------------------------------------------- #
def test_a_missing_candidate_directory_is_skipped_not_fatal():
    """A smoke stage that can itself kill the driver is worse than none. An
    unexpected failure of the probe machinery lets the candidate through to the
    real run, exactly as before the stage existed."""
    r = smoke_check("/nonexistent/candidate/dir")
    assert r["ok"] is True
    assert "skipped" in r["stage"]


def test_an_unlaunchable_interpreter_is_skipped_not_fatal(candidate):
    r = smoke_check(str(candidate), python="/nonexistent/python")
    assert r["ok"] is True
    assert "skipped" in r["stage"]


def test_the_fixture_covers_every_split_the_orchestrator_path_needs(candidate):
    """train_fit / train_es / valid_search / valid_confirm all have to be
    non-empty AND carry both label classes, or the probe fails on the fixture
    rather than on the candidate — a smoke stage that reports its own fixture's
    shortcomings as candidate bugs is worse than no smoke stage.

    Re-derived here by asking the probe to dump its split sizes, rather than
    grepping the probe source for split names.
    """
    assert smoke_mod.FIXTURE_ROWS == 200
    assert smoke_mod.FIXTURE_USERS * smoke_mod.ROWS_PER_USER == 200

    # Have the candidate's own cut functions slice the probe's fixture, and
    # report what came out.
    _patch(candidate / "baseline.py",
           "    print(f\"{METRICS_MARK} {json.dumps(payload)}\")",
           "    print(f\"{METRICS_MARK} {json.dumps(payload)}\")\n"
           "    import collections as _c\n"
           "    _s = cut_train_subsplits(cut_valid_subsplits(load(None)))\n"
           "    for _n in ('train_fit','train_es','valid_search','valid_confirm'):\n"
           "        _rows = _s[_n]\n"
           "        _labs = _c.Counter(x[6] for x in _rows)\n"
           "        print('SPLITSIZE', _n, len(_rows), dict(_labs))")
    r = smoke_check(str(candidate))
    assert r["ok"], r["error"]

    # smoke_check does not return stdout, so re-run the probe to read it.
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-c", smoke_mod._build_probe()],
                       cwd=str(candidate), capture_output=True, text=True)
    sizes = {}
    for line in p.stdout.splitlines():
        if line.startswith("SPLITSIZE"):
            _, name, n, labs = line.split(" ", 3)
            sizes[name] = (int(n), labs)
    assert set(sizes) == {"train_fit", "train_es", "valid_search",
                          "valid_confirm"}, p.stdout
    for name, (n, labs) in sizes.items():
        assert n > 0, f"{name} is empty in the fixture"
        # Both classes, so GAUC's 0 < positives < impressions filter has users
        # to score on the split that early-stops and the split that is reported.
        if name in ("train_es", "valid_search"):
            assert "0: " in labs and "1: " in labs, \
                f"{name} needs both label classes, got {labs}"


# --------------------------------------------------------------------------- #
#  Driver wiring                                                              #
# --------------------------------------------------------------------------- #
def test_a_smoke_failure_never_reaches_execute(monkeypatch, tmp_path):
    """The saving, stated as a test: a candidate rejected by smoke costs zero
    triage runs. In the recorded run each of these cost 240 s, plus up to two
    further 240 s repair runs."""
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    executed = []
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda *a, **k: executed.append(a) or
                        {"status": "ok", "metrics": {"primary": 0.6,
                                                     "per_user": {"u0": 0.6}},
                         "logs": ""})
    monkeypatch.setattr(mock_codegen, "smoke_check",
                        lambda root, **kw: {"ok": False, "seconds": 0.05,
                                            "stage": "encode",
                                            "error": "AssertionError: raw(x)"})
    # A repair that never fixes anything, so the budget is spent in full.
    monkeypatch.setattr(mock_codegen, "debug_and_retry",
                        lambda *a, **k: {"code_diff": "",
                                         "is_semantic_change": False})

    random.seed(0)
    result = driver.run(max_iters=2, verbose=False,
                        progress_path=str(tmp_path / "progress.json"),
                        memory_path=str(tmp_path / "memory.json"),
                        champion_dir=str(tmp_path / "champions"),
                        root_baseline_path=str(tmp_path / "rb.json"),
                        confirm_baseline_path=str(tmp_path / "cb.json"))

    # The root measurement runs execute; no CANDIDATE triage run does.
    assert result["counters"].triage_runs == 0, \
        "a candidate rejected by smoke must not spend a triage run"


def test_the_smoke_budget_is_larger_than_the_execution_budget():
    """The point of the cheap signal is that you can afford more attempts at
    it. If these were equal the stage would only be a filter, not a budget
    reallocation."""
    from orchestrator import driver
    assert driver.MAX_SMOKE_FIX_ATTEMPTS == 5
    assert driver.MAX_FIX_ATTEMPTS == 2
    assert driver.MAX_SMOKE_FIX_ATTEMPTS > driver.MAX_FIX_ATTEMPTS


def test_smoke_repairs_do_not_consume_the_execution_repair_budget(monkeypatch,
                                                                  tmp_path):
    """c.fix_attempts records TOTAL repair spend, so the execution loop has to
    measure its own 2 attempts from wherever the smoke loop left it. Sharing one
    counter would silently give a smoke-repaired candidate zero execution
    repairs."""
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    # Smoke fails twice then passes; execute then always fails, so the
    # execution loop must still get its full MAX_FIX_ATTEMPTS.
    smoke_calls = {"n": 0}

    def _smoke(root, **kw):
        smoke_calls["n"] += 1
        ok = smoke_calls["n"] > 2
        return {"ok": ok, "seconds": 0.05, "stage": "ok" if ok else "encode",
                "error": "" if ok else "AssertionError: raw(x)"}

    exec_calls = {"n": 0}

    _ROOT_OK = {"status": "ok",
                "metrics": {"primary": 0.6, "per_user": {"u0": 0.6}},
                "logs": ""}

    def _execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                 data_dir=None):
        # The ROOT baseline measurement must still succeed, or the tree cannot
        # grow and no candidate is ever attempted. It is the only call whose
        # code_path is the bare "baseline.py" rather than a staged temp path.
        if code_path == "baseline.py":
            return _ROOT_OK
        exec_calls["n"] += 1
        return {"status": "error", "metrics": {}, "logs": "boom"}

    monkeypatch.setattr(mock_codegen, "smoke_check", _smoke)
    monkeypatch.setattr(mock_codegen, "execute", _execute)

    random.seed(0)
    driver.run(max_iters=1, verbose=False,
               progress_path=str(tmp_path / "progress.json"),
               memory_path=str(tmp_path / "memory.json"),
               champion_dir=str(tmp_path / "champions"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    # Per CANDIDATE: 1 initial triage run + MAX_FIX_ATTEMPTS repair runs. The
    # iteration proposes a batch (T2.7), so divide by however many candidates
    # reached _attempt rather than assuming one.
    per_candidate = 1 + driver.MAX_FIX_ATTEMPTS
    assert exec_calls["n"] % per_candidate == 0, (
        f"{exec_calls['n']} execute calls is not a whole number of "
        f"{per_candidate}-call budgets — a smoke repair ate an execution one")
    n_candidates = exec_calls["n"] // per_candidate
    assert 1 <= n_candidates <= driver.MAX_CANDIDATES_PER_ITER
    # Only the FIRST candidate's smoke failed twice (the stub passes from call 3
    # on), so this also pins that a spent smoke budget does not carry over.
    assert exec_calls["n"] == n_candidates * per_candidate
