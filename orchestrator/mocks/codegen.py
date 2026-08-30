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

def execute(code_path, seed, split, wallclock_cap_seconds, root=None, data_dir=None):
    if random.random() < 0.05:
        return {"status": "error", "metrics": {}, "logs": "fake traceback"}
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
    repaired = [f"# fake repair for: {error_context[:60]}\n"] + original
    diff = "".join(difflib.unified_diff(original, repaired,
                                        "a/baseline.py", "b/baseline.py", n=3))
    # semantic change only when we actually rewrite the mechanism
    return {"code_diff": diff, "is_semantic_change": False}

def check_submission(path, split):
    return True