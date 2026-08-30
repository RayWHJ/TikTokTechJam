"""Phase 0 correctness fixes: prompt metric description, parent code_dir
inheritance, and actually applying the repair diff.

All fast by construction — tempdirs plus a stub client, no real FM run. The
end-to-end backstop lives in tests/test_improvement_chain.py (marked slow).
"""
import os
import shutil

import codegen
from codegen import writer
from llm_calls import personas
from orchestrator import driver
from orchestrator.node import Node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGED = ("data.py", "evaluate.py", "baseline.py", "submit.py")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _copy_root_modules(dest):
    for name in STAGED:
        shutil.copy2(os.path.join(ROOT, name), os.path.join(dest, name))


def _marker_diff(path, marker="_PHASE0_MARKER = 1"):
    """A real, applyable diff adding one EXECUTABLE line to the file at `path`.

    Deliberately not a comment: writer.rewrite_to_diff runs the file through
    changes_executable_code, which compares docstring-stripped ASTs — so a
    comment-only rewrite is rejected as a no-op and yields an empty diff.
    """
    original = _read(path)
    rewritten = original.rstrip("\n") + f"\n{marker}\n"
    diff, err = writer.rewrite_to_diff(rewritten, original,
                                       os.path.basename(path))
    assert diff, f"fixture diff should be valid, got error: {err!r}"
    return diff, marker


# ── Fix 1 — the LLM prompt's metric description ────────────────────────── #

def test_dataset_context_uses_primary_mean():
    assert "primary = (GAUC + nDCG@5) / 2" in personas._DATASET_CONTEXT


def test_dataset_context_has_real_baseline_numbers():
    for number in ("0.6610", "0.5282", "0.5946"):
        assert number in personas._DATASET_CONTEXT, number


def test_hypothesis_prompt_uses_primary_not_gauc_only():
    paragraphs = personas.HYPOTHESIS_SYSTEM_PROMPT.split("\n\n")
    matching = [p for p in paragraphs if "success_criterion_paired" in p
                and "DELTA" in p]
    assert matching, "criterion-constraint paragraph not found"
    assert "primary" in matching[0].lower()


# ── Fix 2 — children inherit parent.code_dir ───────────────────────────── #

def test_child_inherits_parent_code_dir_field():
    parent = Node(id="p", parent_id=None, code_path="baseline.py",
                  code_dir="/tmp/parent_xyz")
    child = Node(id="c", parent_id=parent.id, code_path=parent.code_path,
                 code_dir=parent.code_dir,
                 diagnosis={"component": "loss"}, hypothesis={})
    assert child.code_dir == "/tmp/parent_xyz"


def test_stage_reads_from_parent_dir(tmp_path):
    parent_dir = str(tmp_path / "parent")
    os.makedirs(parent_dir)
    _copy_root_modules(parent_dir)

    parent_baseline = os.path.join(parent_dir, "baseline.py")
    _write(parent_baseline, _read(parent_baseline) + "# PARENT_MARKER\n")

    diff, marker = _marker_diff(parent_baseline)
    cand_dir = driver._apply_diff_and_stage(diff, root=parent_dir,
                                            candidate_id="t2")
    assert cand_dir is not None and os.path.isdir(cand_dir)
    try:
        staged = _read(os.path.join(cand_dir, "baseline.py"))
        # The parent's edit survived (staging read from the parent, not the
        # pristine repo) AND this child's edit landed on top of it.
        assert "# PARENT_MARKER" in staged
        assert marker in staged
    finally:
        shutil.rmtree(cand_dir, ignore_errors=True)


# ── Fix 3 — apply the repair diff before rerunning ─────────────────────── #

def test_apply_diff_to_dir_succeeds_and_changes_file(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    target = os.path.join(work, "baseline.py")
    _write(target, "import os\n\nX = 1\n\n\ndef f():\n    return X\n")

    diff, marker = _marker_diff(target)
    before = _read(target)
    assert driver._apply_diff_to_dir(diff, work) is True
    after = _read(target)
    assert after != before
    assert marker in after


def test_apply_diff_to_dir_returns_false_on_malformed(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    target = os.path.join(work, "baseline.py")
    _write(target, "X = 1\n")

    assert driver._apply_diff_to_dir("not a diff at all", work) is False
    assert _read(target) == "X = 1\n"


def test_dir_sha256_changes_when_a_root_file_changes(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    _copy_root_modules(work)

    before = driver._dir_sha256(work)
    assert before == driver._dir_sha256(work), "hash must be stable"

    # A repair may edit data.py rather than baseline.py — the hash covers all
    # four staged modules so that change is still detected.
    data_py = os.path.join(work, "data.py")
    _write(data_py, _read(data_py) + "\n# mutated\n")
    assert driver._dir_sha256(work) != before


class _StubClient:
    """Returns a canned full-file rewrite, mirroring what DEBUG_SYSTEM asks for."""

    def __init__(self, fixed_source):
        self.fixed_source = fixed_source
        self.calls = 0

    def complete(self, system, user, kind, **kw):
        self.calls += 1
        return ("Here is the corrected file.\n\n```python\n"
                + self.fixed_source + "```\n")


def test_debug_returns_applyable_diff(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    _copy_root_modules(work)

    target = os.path.join(work, "baseline.py")
    fixed = _read(target)
    # Break a symbol that later code calls, so the file raises NameError.
    broken = fixed.replace("def sigmoid(x)", "def sigmoidd(x)", 1)
    assert broken != fixed, "breakage anchor 'def sigmoid(x)' not found"
    _write(target, broken)

    stub = _StubClient(fixed)
    result = codegen.debug_and_retry(target, "NameError: name 'sigmoid' is not defined",
                                     client=stub, root=work)

    assert result["code_diff"].strip(), "repair diff must not be empty"
    assert writer.diff_applies(result["code_diff"], "baseline.py",
                               root=work) == (True, "")


def test_debug_returns_empty_diff_when_model_never_produces_one(tmp_path):
    """The driver's guard depends on this: unusable repair -> empty code_diff,
    not a malformed one it would try to apply."""
    work = str(tmp_path / "work")
    os.makedirs(work)
    _copy_root_modules(work)
    target = os.path.join(work, "baseline.py")

    class _Useless:
        def complete(self, system, user, kind, **kw):
            return "I cannot fix this."

    result = codegen.debug_and_retry(target, "NameError: boom",
                                     client=_Useless(), root=work)
    assert result["code_diff"] == ""
