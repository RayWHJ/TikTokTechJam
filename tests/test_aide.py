"""Phase 1 (AIDE): operation-typed nodes, capped fix attempts, nodes.jsonl.

All fast — the mock llm/codegen/harness modules are patched into the driver the
same way orchestrator/tests/test_smoke.py wires them, so no real FM run happens.
NODES_LOG_PATH is redirected per test: the driver's default points at the live
orchestrator/_state/, which a test must never write to.
"""
import json
import os
import random

import pytest

from orchestrator import driver
from orchestrator.node import Node
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm
from orchestrator.mocks import codegen as mock_codegen


def _isolated_caches(tmp_path):
    """Per-test baseline caches — see the same helper in test_smoke.py for why
    the real caches make a mocked run exercise only the degenerate path."""
    return {"root_baseline_path": str(tmp_path / "root_baseline.json"),
            "confirm_baseline_path": str(tmp_path / "confirm_baseline.json")}


@pytest.fixture
def mocked_driver(monkeypatch, tmp_path):
    """Driver wired to mocks, with memory and the nodes log isolated to tmp."""
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    from orchestrator import memory as memory_mod
    orig_init = memory_mod.Memory.__init__

    def isolated_init(self, path=None):
        orig_init(self, path=str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory_mod.Memory, "__init__", isolated_init)

    # mocks.codegen.execute injects a 5% random error; pin the global RNG so an
    # outcome doesn't depend on session entropy or test ordering.
    state = random.getstate()
    random.seed(0)
    yield str(tmp_path / "nodes.jsonl")
    random.setstate(state)


def _run(tmp_path, iters=2, **kw):
    return driver.run(max_iters=iters, verbose=False,
                      progress_path=str(tmp_path / "progress.json"),
                      **_isolated_caches(tmp_path), **kw)


def _lines(log_path):
    with open(log_path, "r", encoding="utf-8") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


# ── Fix 1 — Node fields ────────────────────────────────────────────────── #

def test_node_operation_defaults_to_improve():
    assert Node(id="x", parent_id=None,
                code_path="baseline.py").operation == "improve"


def test_node_fix_attempts_defaults_to_zero():
    assert Node(id="x", parent_id=None,
                code_path="baseline.py").fix_attempts == 0


# ── Fix 2 — the root is a draft ────────────────────────────────────────── #

def test_root_labeled_draft():
    assert driver._new_root().operation == "draft"


# ── Fix 3 — improve children, capped fixes ─────────────────────────────── #

def test_improve_child_construction_labeled_improve():
    parent = Node(id="p", parent_id=None, code_path="baseline.py",
                  code_dir="/tmp/parent_xyz", operation="draft")
    child = Node(id="c", parent_id=parent.id, code_path=parent.code_path,
                 code_dir=parent.code_dir,
                 operation="improve",
                 diagnosis={"component": "loss"}, hypothesis={"mechanism": "m"})
    assert child.operation == "improve"


def test_fix_attempts_capped_at_two(mocked_driver, tmp_path, monkeypatch):
    """Every triage run fails and every repair applies cleanly but never helps.

    Without the cap this is the infinite loop MAX_FIX_ATTEMPTS exists to stop:
    the repair diff changes a file each time, so the hash guard keeps letting it
    through. The mock's debug_and_retry already returns a real applyable diff
    (it prepends a comment to the staged baseline.py).
    """
    monkeypatch.setattr(mock_codegen, "execute",
                        lambda *a, **kw: {"status": "error", "metrics": {},
                                          "logs": "fake traceback"})
    log_path = mocked_driver
    _run(tmp_path, iters=1)

    records = [json.loads(ln) for ln in _lines(log_path)]
    children = [r for r in records if r["parent_id"] is not None]
    assert children, "expected at least one candidate to be logged"
    for r in children:
        assert r["fix_attempts"] == driver.MAX_FIX_ATTEMPTS, r
        # Repair loop exhausted -> _attempt returned "exec".
        assert r["evidence_type"] == "failed_implementation", r


def test_fix_attempts_stay_zero_when_the_run_is_clean(mocked_driver, tmp_path):
    """The counter is a diagnostic signal, so a clean candidate must read 0."""
    log_path = mocked_driver
    _run(tmp_path, iters=2)

    records = [json.loads(ln) for ln in _lines(log_path)]
    scored = [r for r in records
              if r["parent_id"] is not None and r["per_seed_primary"]]
    assert scored, "mocked run should score at least one candidate"
    assert all(r["fix_attempts"] == 0 for r in scored)


# ── Fix 4 — nodes.jsonl ────────────────────────────────────────────────── #

def test_nodes_jsonl_written_and_parseable(mocked_driver, tmp_path):
    log_path = mocked_driver
    _run(tmp_path, iters=2)

    assert os.path.exists(log_path)
    lines = _lines(log_path)
    assert lines, "expected at least the root record"
    for ln in lines:
        rec = json.loads(ln)
        for key in ("id", "parent_id", "operation", "status", "iter"):
            assert key in rec, (key, rec)


def test_nodes_jsonl_starts_with_root_at_iter_zero(mocked_driver, tmp_path):
    log_path = mocked_driver
    _run(tmp_path, iters=2)

    first = json.loads(_lines(log_path)[0])
    assert first["iter"] == 0
    assert first["operation"] == "draft"
    assert first["parent_id"] is None


def test_nodes_jsonl_cleared_on_fresh_run(mocked_driver, tmp_path):
    log_path = mocked_driver
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("}{ not json at all\ngarbage\n")

    _run(tmp_path, iters=1)

    lines = _lines(log_path)
    assert lines
    for ln in lines:
        json.loads(ln)      # raises if the garbage survived
    assert json.loads(lines[0])["iter"] == 0


def test_nodes_jsonl_untouched_when_progress_path_is_none(mocked_driver, tmp_path):
    """progress_path=None means write nothing — the log included."""
    log_path = mocked_driver
    driver.run(max_iters=1, verbose=False, progress_path=None,
               **_isolated_caches(tmp_path))
    assert not os.path.exists(log_path)
