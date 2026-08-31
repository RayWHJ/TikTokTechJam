"""Mock of codegen/. Simulates gates, sandboxed runs, per-user results."""
import difflib
import hashlib
import os
import random

#: Mirrors codegen.NO_SEMANTIC_CHANGE so the driver's no-op retry path is
#: exercisable against the mocks.
NO_SEMANTIC_CHANGE = ("the rewrite changed only comments, docstrings or "
                      "formatting — the executable code is byte-identical, so "
                      "this candidate would score exactly the same as its parent")


def write_fix(hypothesis, target_component, root=".", semantic_feedback=None):
    """A real unified diff, so driver._apply_diff_and_stage actually patches.

    Deliberately a new-file diff against /dev/null: it applies under `patch -p1`
    without depending on baseline.py's exact contents. The per-hypothesis body
    keeps the driver's diff-hash dedup meaningful.

    `semantic_feedback` changes the body, so a no-op retry produces a different
    diff hash and is not immediately killed by the driver's run-wide dedup —
    matching the real writer, whose prompt changes on the retry.
    """
    body = [f"# fake diff for {target_component}",
            f"# mechanism: {hypothesis['mechanism']}",
            f"# sketch: {hypothesis.get('implementation_sketch', '')}"]
    if semantic_feedback:
        body.append(f"# retry after: {semantic_feedback}")
    return ("diff --git a/_mock_fix.py b/_mock_fix.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/_mock_fix.py\n"
            f"@@ -0,0 +1,{len(body)} @@\n"
            + "".join(f"+{line}\n" for line in body))

def write_refine(hypothesis, component, root=".", semantic_feedback=None):
    """Mock of codegen.write_refine — resolves the component via the real
    registry, then delegates to write_fix exactly as the real one does."""
    from codegen.ablations import ABLATIONS
    abl = ABLATIONS.get(component)
    if abl is None:
        raise ValueError(f"unknown component {component!r}")
    return write_fix(hypothesis, target_component=abl.target, root=root,
                     semantic_feedback=semantic_feedback)


def pre_execution_gate(code_diff):
    if "cancel_like_cnt" in code_diff and "point_in_time=True" not in code_diff:
        return {"pass": False, "reasons": ["non-causal column without PIT marker"]}
    if "test" in code_diff.lower() and "read" in code_diff.lower():
        return {"pass": False, "reasons": ["suspected test-split file read"]}
    return {"pass": True, "reasons": []}

def smoke_check(root, **kw):
    """Mock of codegen.smoke_check — the 200-row pre-flight check.

    Passes by default, because the mock's diffs are comment-only new files that
    would genuinely survive a real smoke run. Tests that want to exercise the
    driver's smoke-repair path monkeypatch this to fail, the same way they
    already monkeypatch pre_execution_gate.

    Present rather than omitted so a mocked run exercises the same call sequence
    production does: a mock missing a function the driver calls turns an
    interface change into a passing test suite and a broken real run.
    """
    return {"ok": True, "error": "", "seconds": 0.0, "stage": "ok"}


#: What a CLEAN candidate scores when its labels are shuffled within each user.
#: Measured on the real baseline (GAUC lands on 0.4998 against a theoretical 0.5).
#: A stub that ignored `permute_labels` and returned its normal score would be
#: claiming "my score survives label shuffling", which IS the leak signature — so
#: the driver's control would correctly refuse every mocked promotion.
PERMUTED_PRIMARY = 0.4840


def execute(code_path, seed, split, wallclock_cap_seconds, root=None, data_dir=None, **kw):
    if random.random() < 0.05:
        return {"status": "error", "metrics": {}, "logs": "fake traceback"}
    if kw.get("permute_labels"):
        # The label-permutation control (T3.4). The mock models an HONEST
        # candidate, so its score collapses to chance.
        users = [f"u{i}" for i in range(10)]
        return {"status": "ok", "logs": "",
                "metrics": {"primary": PERMUTED_PRIMARY,
                            "GAUC": 0.4998, "nDCG@5": 0.4682,
                            "per_user": {u: PERMUTED_PRIMARY for u in users}}}
    # Stable per (candidate, seed): seeding on `seed` alone made every candidate
    # score identically to the baseline, so all paired per-user deltas were
    # exactly 0.0 and should_continue_locally could never pass — the mock could
    # not exercise the search tree growing at all. Hash the path rather than
    # using hash() so it is stable across processes (PYTHONHASHSEED).
    digest = hashlib.sha1(f"{code_path}|{seed}".encode("utf-8")).hexdigest()
    rng = random.Random(digest)
    users = [f"u{i}" for i in range(10)]
    per_user = {u: 0.5946 + rng.gauss(0.005, 0.02) for u in users}
    primary = sum(per_user.values()) / len(per_user)
    return {"status": "ok",
            "metrics": {"primary": primary,
                        "GAUC": primary + 0.03,
                        "nDCG@5": primary - 0.03,
                        "per_user": per_user},
            "logs": ""}

def debug_and_retry(code_path, error_context, root=".", **kw):
    """A real unified diff against the staged baseline.py, so the driver can
    actually APPLY the repair before rerunning.

    The driver now requires three things of a repair: a non-empty diff, that it
    applies to the staged dir, and that applying it changes one of the four
    staged modules (it re-hashes them). A `# fake repair` string satisfied none
    of those, so the mocked repair path would always dead-end at "exec".
    """
    src = os.path.join(root, "baseline.py")
    if not os.path.exists(src):
        return {"code_diff": "", "is_semantic_change": False}
    with open(src, "r", encoding="utf-8") as fh:
        original = fh.readlines()
    # COLLAPSED to one line. `error_context` is a real traceback in a real run,
    # so slicing it raw embedded its newlines inside what was meant to be a single
    # `+` comment line: patch then read lines 2-3 of the traceback as CONTEXT that
    # did not match the file, the hunk failed, and the repair silently never
    # applied. The mocked execution-repair loop therefore never iterated — it
    # broke on "no file changed" after one attempt, in every mocked run.
    # Single-line error strings (which is what the tests happened to use) hid it.
    summary = " ".join((error_context or "").split())[:60]
    repaired = [f"# fake repair for: {summary}\n"] + original
    diff = "".join(difflib.unified_diff(original, repaired,
                                        "a/baseline.py", "b/baseline.py", n=3))
    # semantic change only when we actually rewrite the mechanism
    return {"code_diff": diff, "is_semantic_change": False}

def check_submission(path, split):
    return True


def synthesize_report(run_log, **kw):
    """Mock of codegen.synthesize_report — a deterministic markdown stub.

    Present rather than omitted so a mocked run exercises the same call sequence
    production does. The driver swallows exceptions from the report path (it is
    the last thing that happens and must not lose the numbers), so a MISSING
    function here would look exactly like a working one — the report would simply
    never appear and nothing would say why.
    """
    best = (run_log.get("global_best") or {})
    champ = (run_log.get("champion") or {})
    return (
        "# Mock run report\n\n"
        f"baseline primary: {run_log.get('baseline_primary')}\n\n"
        f"best promoted: {best.get('primary')} (node {best.get('node_id')})\n\n"
        f"champion: {champ.get('primary')} "
        f"(is_baseline={champ.get('is_baseline')})\n\n"
        f"iterations: {run_log.get('iters_completed')} | "
        f"promotions: {run_log.get('promotions')}\n"
    )


def sanity_check(code_diff, **kw):
    """Mock of codegen.sanity_check — never suspicious, and never called on a
    plausible score (the real one returns None without a model call)."""
    if kw.get("observed_score") is None:
        return None
    return None
