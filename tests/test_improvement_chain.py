"""Does a real improvement survive the pipeline?

The first overnight run produced 50 iterations in which `iter_primary` never
once rose above the baseline. The cause was not weak hypotheses — it was that
nothing in the repo asserted the chain

    write_fix -> stage -> execute -> paired delta -> open_nodes -> promote

could carry a genuine win from one end to the other. Every link was unit-tested;
the chain was not. These tests inject a candidate that is *known* to be better
and assert it arrives, plus regression guards pinned to the exact numbers that
used to silently drop it.

The fast tests here are fully mocked and run in milliseconds. The `slow` test at
the bottom does the same thing through real staging and a real FM run.
"""
import json
import random

import pytest

import codegen
from codegen import writer
from orchestrator import driver
from orchestrator.counters import Counters
from orchestrator.memory import Memory
from orchestrator.node import Node
from orchestrator.promotion import (bootstrap_delta, should_continue_locally,
                                    should_promote_globally)
from orchestrator.mocks import codegen as mock_codegen
from orchestrator.mocks import harness as mock_harness
from orchestrator.mocks import llm as mock_llm

USERS = [f"u{i}" for i in range(40)]
BASE_PRIMARY = 0.5946
#: Big enough to be unambiguous, small enough to be realistic for this dataset.
KNOWN_LIFT = 0.004


def _baseline_per_user(seed):
    """Deterministic per-user baseline scores. Varies by seed, not by user."""
    rng = random.Random(f"base|{seed}")
    return {u: BASE_PRIMARY + rng.gauss(0, 0.02) for u in USERS}


def _metrics(per_user):
    primary = sum(per_user.values()) / len(per_user)
    return {"status": "ok", "logs": "",
            "metrics": {"primary": primary, "GAUC": primary + 0.03,
                        "nDCG@5": primary - 0.03, "per_user": dict(per_user)}}


def _is_candidate(code_path):
    """The driver stages candidates into .../candidate_<id>/baseline.py."""
    return "candidate_" in str(code_path)


def _install(monkeypatch, tmp_path, execute_fn):
    """Point the driver at the mocks, an isolated memory, and `execute_fn`."""
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(mock_codegen, "execute", execute_fn)
    # nodes.jsonl defaults to the live orchestrator/_state/ — same rule as the
    # baseline caches in _run: a mocked run must never write there.
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    from orchestrator import memory as memory_mod
    orig_init = memory_mod.Memory.__init__
    monkeypatch.setattr(memory_mod.Memory, "__init__",
                        lambda self, path=None: orig_init(
                            self, path=str(tmp_path / "memory.json")))


def _run(tmp_path, max_iters=2):
    """Every cache path is per-test. The two baseline caches live under
    orchestrator/_state by default, and a mocked run must never write there."""
    progress = tmp_path / "progress.json"
    result = driver.run(max_iters=max_iters, verbose=False,
                        progress_path=str(progress),
                        root_baseline_path=str(tmp_path / "root.json"),
                        confirm_baseline_path=str(tmp_path / "confirm.json"))
    return result, json.loads(progress.read_text())


# --------------------------------------------------------------------------- #
#  The chain: a known-good candidate must reach promotion                      #
# --------------------------------------------------------------------------- #
def test_a_known_good_candidate_promotes_and_lifts_global_best(monkeypatch,
                                                               tmp_path):
    """Every candidate is +0.004 on every user. This MUST promote.

    If this test fails, the search cannot climb no matter how good its
    hypotheses are, and any number of iterations will report a flat
    iter_primary — which is exactly what 50 iterations did.
    """
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        per_user = _baseline_per_user(seed)
        if _is_candidate(code_path):
            per_user = {u: v + KNOWN_LIFT for u, v in per_user.items()}
        return _metrics(per_user)

    _install(monkeypatch, tmp_path, execute)
    result, progress = _run(tmp_path)

    baseline = progress["baseline_primary"]
    assert result["global_best"] > baseline, (
        "a uniform +0.004 candidate did not lift global_best; the "
        "write->stage->execute->delta->promote chain is broken")
    assert len(result["history"]) > 1, "promotion must extend the history ladder"
    assert any(r["promoted"] for r in progress["iterations"])

    # And it must be visible in the progress file, which is the only artifact an
    # overnight run leaves behind.
    scored = [r for r in progress["iterations"] if r["iter_primary"] is not None]
    assert scored, "expected at least one scored iteration"
    assert max(r["iter_primary"] for r in scored) > baseline
    # best_mean_delta is null on an iteration whose only candidate ran but was
    # rejected as a no-op: it scored, so iter_primary is set, but it never
    # reached the paired bootstrap. Every candidate here returns the same
    # baseline+KNOWN_LIFT, so a candidate of a candidate IS bit-identical to
    # its parent — filter the nulls rather than asserting they can't happen.
    paired = [r["best_mean_delta"] for r in scored
              if r["best_mean_delta"] is not None]
    assert paired, "expected at least one iteration with a paired delta"
    assert max(paired) == pytest.approx(KNOWN_LIFT, abs=1e-6)


def test_promotion_pairs_against_the_same_split_it_scores(monkeypatch, tmp_path):
    """valid_confirm is scored against a valid_confirm baseline, not valid_search.

    valid_confirm sits on a later date range and runs at a different absolute
    level. Promotion used to compute `confirm_primary - global_best` with
    global_best carried on valid_search, so the "delta" was mostly the gap
    between two splits. Here valid_confirm is shifted DOWN by 0.05 while the
    candidate keeps its genuine +0.004: a same-split paired test still promotes,
    a cross-split subtraction cannot.
    """
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        offset = -0.05 if split == "valid_confirm" else 0.0
        per_user = {u: v + offset for u, v in _baseline_per_user(seed).items()}
        if _is_candidate(code_path):
            per_user = {u: v + KNOWN_LIFT for u, v in per_user.items()}
        return _metrics(per_user)

    _install(monkeypatch, tmp_path, execute)
    result, progress = _run(tmp_path)

    assert result["global_best"] > progress["baseline_primary"]
    promoted = [c for r in progress["iterations"] for c in r["candidates"]
                if c["status"] == "promoted"]
    assert promoted, "a +0.004 candidate must survive a same-split confirm test"
    # global_best stays the valid_search number, so the reported scoreboard is
    # not silently rebased onto a lower-level split.
    assert result["global_best"] == pytest.approx(
        max(c["primary"] for c in promoted))
    assert promoted[0]["confirm_primary"] < result["global_best"]


def test_a_flat_candidate_does_not_promote(monkeypatch, tmp_path):
    """The mirror of the above: no lift must NOT promote.

    Guards against having loosened the gates into rubber-stamping noise.
    """
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        rng = random.Random(f"{code_path}|{seed}")
        return _metrics({u: v + rng.gauss(0, 0.02)
                         for u, v in _baseline_per_user(seed).items()})

    _install(monkeypatch, tmp_path, execute)
    result, progress = _run(tmp_path, max_iters=3)

    assert result["history"] == [progress["baseline_primary"]]
    assert not any(r["promoted"] for r in progress["iterations"])


# --------------------------------------------------------------------------- #
#  Tier 1: the bar itself                                                      #
# --------------------------------------------------------------------------- #
def test_node_score_is_the_mean_over_seeds_not_the_max():
    """Pinned to the bias that hid every real gain.

    The root runs 3 seeds; most candidates finish only the 1-seed triage run.
    Under max-vs-max the root's bar was 0.5960 while an equally-good 1-seed
    candidate offered 0.5955 — a loss on paper, from seed count alone.
    """
    root = Node(id="r", parent_id=None, code_path="baseline.py",
                per_seed_primary={0: 0.5940, 1: 0.5960, 2: 0.5950})
    cand = Node(id="c", parent_id="r", code_path="cand/baseline.py",
                per_seed_primary={0: 0.5955})

    assert driver._scalar_primary(root) == pytest.approx(0.5950)
    assert driver._scalar_primary(cand) == pytest.approx(0.5955)
    assert driver._scalar_primary(cand) > driver._scalar_primary(root)
    # The old max-based scalar is what inverted the comparison.
    assert driver._best_primary(cand) < max(root.per_seed_primary.values())


def test_measure_root_stores_per_seed_and_uses_the_mean(monkeypatch, tmp_path):
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        return _metrics(_baseline_per_user(seed))

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(mock_codegen, "execute", execute)

    root = driver._new_root()
    assert driver._measure_root(root, Counters(), cache_path=None, verbose=False)
    assert sorted(root.per_seed_primary) == list(driver.ROOT_SEEDS)
    per_seed = list(root.per_seed_primary.values())
    assert root.local_best_score == pytest.approx(sum(per_seed) / len(per_seed))
    assert root.local_best_score < max(per_seed), \
        "the root's bar must not be its luckiest seed"


def test_should_continue_locally_admits_a_real_small_win():
    """The exact case the old rule rejected.

    mean_delta=0.0007, lower_95=0.0002 built upper_bound=0.0017, which lost to
    margin=0.002 — so a significantly-positive candidate never entered
    open_nodes, never became a parent, and nothing could build on it.
    """
    assert should_continue_locally(0.0007, 0.95, 0.0002)
    # Still rejects what it should: positive mean, but not significant.
    assert not should_continue_locally(0.0007, 0.60, -0.0004)
    assert not should_continue_locally(-0.002, 0.10, -0.005)


def test_should_promote_globally_margin_matches_achievable_deltas():
    assert should_promote_globally(0.0010, 0.0003)
    assert not should_promote_globally(0.0002, 0.0001), "below the 0.0005 margin"
    assert not should_promote_globally(0.0010, -0.0001), "lower bound must clear 0"


def test_progress_file_reports_paired_stats_per_candidate(monkeypatch, tmp_path):
    """A flat absolute score is not enough to tell a win from noise."""
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        per_user = _baseline_per_user(seed)
        if _is_candidate(code_path):
            per_user = {u: v + KNOWN_LIFT for u, v in per_user.items()}
        return _metrics(per_user)

    _install(monkeypatch, tmp_path, execute)
    _, progress = _run(tmp_path)

    survivors = [c for r in progress["iterations"] for c in r["candidates"]
                 if c["mean_delta"] is not None]
    assert survivors, "expected at least one candidate with paired statistics"
    for c in survivors:
        assert c["mean_delta"] == pytest.approx(KNOWN_LIFT, abs=1e-6)
        assert c["p_positive"] == 1.0
        assert c["lower_95"] > 0
        assert c["n_seeds"] >= 1


# --------------------------------------------------------------------------- #
#  Tier 2: candidates that change nothing                                      #
# --------------------------------------------------------------------------- #
def test_is_no_op_detects_a_bit_identical_candidate():
    parent = Node(id="p", parent_id=None, code_path="baseline.py",
                  per_user_by_seed={0: {"u1": 0.6, "u2": 0.5}})
    twin = Node(id="t", parent_id="p", code_path="c/baseline.py",
                per_user_by_seed={0: {"u1": 0.6, "u2": 0.5}})
    changed = Node(id="d", parent_id="p", code_path="c/baseline.py",
                   per_user_by_seed={0: {"u1": 0.6, "u2": 0.5001}})
    disjoint = Node(id="x", parent_id="p", code_path="c/baseline.py",
                    per_user_by_seed={0: {"u9": 0.6}})

    assert driver._is_no_op(twin, parent, seed=0)
    assert not driver._is_no_op(changed, parent, seed=0)
    # No shared users and no measurement are unknowns, not no-ops.
    assert not driver._is_no_op(disjoint, parent, seed=0)
    assert not driver._is_no_op(Node(id="e", parent_id="p", code_path="c"),
                                parent, seed=0)


def test_a_no_op_candidate_is_named_not_recorded_as_refuted(monkeypatch, tmp_path):
    """Candidates that reproduce the parent exactly get their own evidence type.

    19 of the 33 candidates that scored in the first overnight run returned the
    baseline's primary to the last bit — the diff applied but never touched code
    that runs. Filing those as `refuted_under_context` would poison memory
    against mechanisms that were never actually tried.
    """
    def execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                data_dir=None):
        return _metrics(_baseline_per_user(seed))   # candidate == parent, always

    _install(monkeypatch, tmp_path, execute)
    result, progress = _run(tmp_path, max_iters=2)

    evidence = [c["evidence_type"] for r in progress["iterations"]
                for c in r["candidates"]]
    assert "no_op" in evidence, evidence
    assert result["counters"].no_op_rewrites > 0, \
        "a no-op must trigger exactly one re-write with the reason fed back"

    mem = Memory(path=str(tmp_path / "memory.json"))
    hashes = {e.code_hash for e in mem.by_type("refuted_under_context")}
    no_op_ids = {c["id"] for r in progress["iterations"]
                 for c in r["candidates"] if c["evidence_type"] == "no_op"}
    assert not (hashes & no_op_ids), \
        "a mechanism that was never implemented must not be filed as refuted"


def test_writer_rejects_a_comment_only_rewrite():
    original = "def f(x):\n    return x + 1\n"
    annotated = ('"""Docstring added."""\n'
                 "def f(x):\n"
                 "    # explain the increment\n"
                 "    return x + 1\n")
    real = "def f(x):\n    return x + 2\n"

    assert not writer.changes_executable_code(original, annotated)
    assert writer.changes_executable_code(original, real)

    diff, err = writer.rewrite_to_diff(annotated, original, "m.py")
    assert diff == ""
    assert err == writer.NO_SEMANTIC_CHANGE

    diff, err = writer.rewrite_to_diff(real, original, "m.py")
    assert err == ""
    assert "+    return x + 2" in diff


def test_writer_noop_guard_does_not_swallow_syntax_errors():
    """Unparseable output is the sandbox's problem, not the no-op guard's."""
    original = "def f(x):\n    return x\n"
    assert writer.changes_executable_code(original, "def f(x:\n  return")


def test_write_fix_survives_the_semantic_feedback_path(tmp_path):
    """The retry prompt must not break diff generation."""
    hypothesis = {"mechanism": "m", "implementation_sketch": "s"}
    diff = codegen.write_fix(hypothesis, "loss", root=".",
                             semantic_feedback=codegen.NO_SEMANTIC_CHANGE)
    assert diff.strip(), "semantic feedback must still yield a usable diff"
    assert writer.diff_applies(diff, "baseline.py", root=".")[0]


def test_fake_backend_makes_a_real_edit_to_the_actual_baseline():
    """Guards the fake's anchors against drift in baseline.py / data.py.

    The fake used to echo the file back with a comment prepended, which is
    precisely the no-op the new guard rejects. If an anchor in
    FakeBackend._SEMANTIC_EDITS stops matching, this is the test that says so
    rather than four unrelated ones failing.
    """
    for file_name, anchors in codegen.FakeBackend._SEMANTIC_EDITS.items():
        with open(file_name, encoding="utf-8") as fh:
            content = fh.read()
        assert any(old in content for old, _ in anchors), (
            f"no FakeBackend anchor still matches {file_name}: {anchors}")
        edited = codegen.FakeBackend._apply_semantic_edit(content, file_name)
        assert writer.changes_executable_code(content, edited)


# --------------------------------------------------------------------------- #
#  The same chain, through real staging and a real FM run                      #
# --------------------------------------------------------------------------- #
#: epochs 40 -> 2 keeps the real runs short. Applied to every variant below, so
#: it is held constant and never the thing under test.
_FAST = ("ap.add_argument('--epochs', type=int, default=40)",
         "ap.add_argument('--epochs', type=int, default=2)")

#: The rank change under test, taken at the argparse default because that is the
#: value that actually reaches the model. Editing `k=16` in FM.__init__ instead
#: does NOTHING: run_fm calls FM(dim, k=k, ...) and always passes k explicitly,
#: so the class default is dead. The first run of this test used that anchor and
#: produced a bit-identical score — which is the whole no-op failure mode in
#: miniature, and a live trap for any candidate that "tunes k" in the class.
_RANK = ("ap.add_argument('--k', type=int, default=16)",
         "ap.add_argument('--k', type=int, default=8)")


def _full_file_diff(subs):
    """A unified diff applying `subs` to baseline.py, via the writer's own path."""
    with open("baseline.py", encoding="utf-8") as fh:
        original = fh.read()
    rewritten = original
    for old, new in subs:
        assert old in rewritten, f"anchor drifted out of baseline.py: {old!r}"
        rewritten = rewritten.replace(old, new, 1)
    diff, err = writer.rewrite_to_diff(rewritten, original, "baseline.py")
    assert err == "", err
    return diff


def _stage_and_run(diff, candidate_id):
    cand_dir = driver._apply_diff_and_stage(diff, root=".",
                                            candidate_id=candidate_id)
    assert cand_dir is not None, "hand-written diff failed to apply"
    code_path = f"{cand_dir}/baseline.py"
    res = codegen.execute(code_path, seed=0, split="valid_search",
                          wallclock_cap_seconds=900, root=cand_dir,
                          data_dir=driver.DATA_DIR)
    assert res["status"] == "ok", res["logs"][-2000:]
    assert res["metrics"].get("per_user"), \
        "no per_user block: paired promotion is impossible without it"
    return res["metrics"]


@pytest.mark.slow
def test_real_diff_changes_the_score_and_a_comment_does_not():
    """The chain on real data: staging carries a semantic change, and only that.

    Three real FM runs on valid_search:
      control        epochs=2
      annotated      epochs=2 + a comment        -> must score IDENTICALLY
      real change    epochs=2 + --k default 16->8 -> must score DIFFERENTLY

    Together these prove the staged patch is what executes (a real edit moves
    the number) and that the no-op detector fires on exactly the case that
    produced 19 bit-identical scores overnight.
    """
    control = _stage_and_run(_full_file_diff([_FAST]), "chain_control")

    annotated = _stage_and_run(_full_file_diff(
        [_FAST, ("def sigmoid(x):", "# annotated, changes nothing\ndef sigmoid(x):")]),
        "chain_annotated")
    assert annotated["primary"] == control["primary"], \
        "a comment must not move the metric"

    changed = _stage_and_run(_full_file_diff([_FAST, _RANK]), "chain_real")
    assert changed["primary"] != control["primary"], \
        "--k 16 -> 8 did not move the metric: the staged patch is not what ran"

    parent = Node(id="p", parent_id=None, code_path="baseline.py",
                  per_user_by_seed={0: control["per_user"]})
    twin = Node(id="t", parent_id="p", code_path="c/baseline.py",
                per_user_by_seed={0: annotated["per_user"]})
    real = Node(id="r", parent_id="p", code_path="c/baseline.py",
                per_user_by_seed={0: changed["per_user"]})

    assert driver._is_no_op(twin, parent, seed=0)
    assert not driver._is_no_op(real, parent, seed=0)

    mean_d, p_pos, lower = bootstrap_delta({0: changed["per_user"]},
                                           {0: control["per_user"]})
    assert (mean_d, p_pos, lower) != (0.0, 0.0, 0.0), \
        "real candidate and baseline must share user ids to pair on"
