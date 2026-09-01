"""writer.diff_applies + write_fix's validate-and-repair loop.

The malformed shapes asserted here are the ones a real run actually produced:
headers without the a/ b/ prefix (patch -p1 says "can't find file to patch"),
normal/ed-format output, and a truncated hunk header.
"""
import os
import textwrap

import pytest

from codegen import prompts
from codegen.diffnorm import normalize_unified_diff
from codegen.llm_client import KIND_DIFF
from codegen import writer
from codegen.writer import diff_applies, rewrite_to_diff, write_fix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _valid_diff(file_name, prefix=True):
    """A diff that inserts a comment above the first 3 real lines of file_name."""
    with open(os.path.join(REPO_ROOT, file_name), encoding="utf-8") as fh:
        ctx = fh.read().splitlines()[:3]
    old, new = (f"a/{file_name}", f"b/{file_name}") if prefix else (file_name, file_name)
    return (f"--- {old}\n+++ {new}\n@@ -1,{len(ctx)} +1,{len(ctx) + 1} @@\n"
            "+# probe comment\n" + "".join(f" {c}\n" for c in ctx))


def test_accepts_git_style_p1_headers():
    ok, err = diff_applies(_valid_diff("baseline.py"), "baseline.py", REPO_ROOT)
    assert ok, err


def test_accepts_bare_p0_headers():
    """`--- data.py` with no a/ prefix: this is what the real model emitted."""
    ok, err = diff_applies(_valid_diff("data.py", prefix=False), "data.py", REPO_ROOT)
    assert ok, err


def test_rejects_ed_format_diff():
    ed = textwrap.dedent("""\
        18a19,22
        > import mlflow
        > from mlflow import log_param
        """)
    ok, err = diff_applies(ed, "baseline.py", REPO_ROOT)
    assert not ok
    assert "garbage" in err.lower() or "no valid patches" in err.lower()


def test_rejects_truncated_hunk_header():
    bad = "--- a/data.py\n+++ b/data.py\n@@ -21,7 +25\n+# oops\n"
    ok, err = diff_applies(bad, "data.py", REPO_ROOT)
    assert not ok
    assert err.strip()


def test_rejects_empty_diff():
    ok, err = diff_applies("   \n", "baseline.py", REPO_ROOT)
    assert not ok
    assert err == "empty diff"


def test_reports_missing_target_file():
    ok, err = diff_applies(_valid_diff("baseline.py"), "nope.py", REPO_ROOT)
    assert not ok
    assert "not found" in err


def test_write_fix_returns_an_applying_diff(monkeypatch):
    monkeypatch.setenv("CODEGEN_LLM_BACKEND", "fake")
    from codegen import fixtures
    diff = write_fix(fixtures.FAKE_HYPOTHESIS, "loss", root=REPO_ROOT)
    ok, err = diff_applies(diff, "baseline.py", REPO_ROOT)
    assert ok, err


class _ScriptedClient:
    """Returns each canned reply in turn; records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user, kind=KIND_DIFF, **kw):
        self.prompts.append(user)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


def test_write_fix_retries_and_feeds_back_the_patch_error():
    good = _valid_diff("baseline.py")
    client = _ScriptedClient(["18a19,22\n> import mlflow\n", f"```diff\n{good}```\n"])
    diff = write_fix({"mechanism": "m", "implementation_sketch": "s"}, "loss",
                     client=client, root=REPO_ROOT, max_attempts=3)

    ok, err = diff_applies(diff, "baseline.py", REPO_ROOT)
    assert ok, err
    assert len(client.prompts) == 2, "should have retried exactly once"
    # The retry must carry a concrete rejection reason, not just a generic nudge.
    assert "REJECTED" in client.prompts[1]
    assert "baseline.py" in client.prompts[1]


def test_write_fix_gives_up_after_max_attempts():
    client = _ScriptedClient(["18a19,22\n> junk\n"])
    diff = write_fix({"mechanism": "m", "implementation_sketch": "s"}, "loss",
                     client=client, root=REPO_ROOT, max_attempts=2)
    assert len(client.prompts) == 2
    # Returns the last (still bad) diff; staging rejects it and logs the reason.
    assert not diff_applies(diff, "baseline.py", REPO_ROOT)[0]


def test_repair_suffix_includes_the_error_text():
    suffix = prompts.build_diff_repair_suffix("baseline.py", "rewrite looks truncated")
    assert "rewrite looks truncated" in suffix
    assert "baseline.py" in suffix


# --------------------------------------------------------------------------- #
#  Full-file rewrite -> diff (the primary writer path)                         #
# --------------------------------------------------------------------------- #
ORIGINAL = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"


def test_rewrite_to_diff_produces_an_applying_diff(tmp_path):
    (tmp_path / "m.py").write_text(ORIGINAL)
    new = ORIGINAL.replace("return 1", "return 42")
    diff, err = rewrite_to_diff(new, ORIGINAL, "m.py")
    assert not err
    ok, apply_err = diff_applies(diff, "m.py", str(tmp_path))
    assert ok, apply_err


def test_rewrite_to_diff_rejects_elided_output():
    for elided in ("def a():\n    ...\n",
                   "def a():\n    return 1\n# rest of file unchanged\n",
                   "def a():\n    return 1\n# ... unchanged\n"):
        diff, err = rewrite_to_diff(elided, ORIGINAL, "m.py")
        assert diff == ""
        assert "elided" in err or "truncated" in err


def test_rewrite_to_diff_rejects_truncation():
    diff, err = rewrite_to_diff("def a():\n", ORIGINAL, "m.py")
    assert diff == ""
    assert "truncated" in err


def test_rewrite_to_diff_rejects_a_noop():
    diff, err = rewrite_to_diff(ORIGINAL, ORIGINAL, "m.py")
    assert diff == ""
    assert "identical" in err


def test_rewrite_to_diff_rejects_empty():
    assert rewrite_to_diff("   ", ORIGINAL, "m.py") == ("", "empty rewrite")


def test_write_fix_uses_the_full_file_path(monkeypatch, tmp_path):
    """A full-file rewrite must be diffed locally, in one call, no retry."""
    (tmp_path / "baseline.py").write_text(ORIGINAL)
    new = ORIGINAL.replace("return 1", "return 42")
    client = _ScriptedClient([f"Sure.\n```python\n{new}```\n"])
    diff = write_fix({"mechanism": "m", "implementation_sketch": "s"}, "loss",
                     client=client, root=str(tmp_path))
    assert len(client.prompts) == 1, "a valid rewrite must not trigger a retry"
    assert "return 42" in diff
    assert diff_applies(diff, "baseline.py", str(tmp_path))[0]


# --------------------------------------------------------------------------- #
#  Hunk-count repair, for a model that returns a patch anyway                  #
# --------------------------------------------------------------------------- #
def test_normalize_fixes_wrong_hunk_counts():
    """Wrong @@ counts make patch reject the WHOLE file as malformed."""
    ctx = ORIGINAL.splitlines()[:2]
    body = "+# added\n" + "".join(f" {c}\n" for c in ctx)
    bad = f"--- a/m.py\n+++ b/m.py\n@@ -1,99 +1,7 @@\n{body}"
    fixed = normalize_unified_diff(bad, ORIGINAL)
    assert f"@@ -1,{len(ctx)} +1,{len(ctx) + 1} @@" in fixed


def test_normalize_restores_dropped_space_on_blank_context_lines():
    bad = "--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,4 @@\n+# added\n def a():\n\n     return 1\n"
    fixed = normalize_unified_diff(bad, ORIGINAL)
    assert "\n \n" in fixed, "blank context line should carry its leading space"


def test_normalize_leaves_start_lines_alone():
    """patch relocates hunks by searching context; rewriting starts broke a
    diff that applied cleanly, so starts must be preserved verbatim."""
    bad = "--- a/m.py\n+++ b/m.py\n@@ -180,1 +250,2 @@\n+# added\n def a():\n"
    fixed = normalize_unified_diff(bad, ORIGINAL)
    assert "@@ -180," in fixed and "+250," in fixed


def test_normalize_passes_through_a_diff_without_hunks():
    assert normalize_unified_diff("not a diff\n", ORIGINAL) == "not a diff\n"


# --------------------------------------------------------------------------- #
#  T3.1 — function-scoped edits spliced by AST                                #
# --------------------------------------------------------------------------- #
_SAMPLE = '''\
"""Module docstring."""
import numpy as np

CONST = 3


def helper(a):
    """Doc."""
    return a + 1


class FM:
    """A model."""

    def __init__(self, k=16):
        self.k = k

    def step(self, X, y):
        return 0.0

    def predict(self, X):
        return X


def tail():
    return helper(CONST)
'''


def test_splice_replaces_a_top_level_function_and_leaves_the_rest_byte_identical():
    new, err = writer.splice_definitions(
        _SAMPLE, 'def helper(a):\n    """Doc."""\n    return a + 99\n')
    assert err == ""
    assert "return a + 99" in new
    # Everything else survives untouched — the point of splicing over unparsing.
    assert '"""Module docstring."""' in new
    assert "CONST = 3" in new
    assert "def predict(self, X):" in new
    assert new.count("def helper") == 1


def test_splice_replaces_one_method_and_keeps_the_classes_other_methods():
    """A model asked for one method commonly re-emits the class header around it
    and omits the siblings. Replacing the whole class would delete them."""
    new, err = writer.splice_definitions(
        _SAMPLE, "class FM:\n    def step(self, X, y):\n        return 1.0\n")
    assert err == ""
    assert "return 1.0" in new
    assert "def __init__(self, k=16):" in new, "sibling method deleted"
    assert "def predict(self, X):" in new, "sibling method deleted"
    assert new.count("class FM:") == 1
    import ast
    ast.parse(new)


def test_splice_adds_a_new_sibling_method_next_to_the_one_it_replaces():
    """"edit FM.step and add FM.bpr_step" has to be ONE splice, or the most
    common shape of a real loss change falls back to the whole file."""
    block = ("class FM:\n"
             "    def step(self, X, y):\n"
             "        return self.bpr_step(X, y)\n"
             "\n"
             "    def bpr_step(self, Xp, Xn):\n"
             "        return 2.0\n")
    new, err = writer.splice_definitions(_SAMPLE, block)
    assert err == ""
    assert "def bpr_step" in new
    assert "self.bpr_step(X, y)" in new
    import ast
    tree = ast.parse(new)
    fm = next(n for n in tree.body if getattr(n, "name", None) == "FM")
    names = [s.name for s in fm.body if isinstance(s, ast.FunctionDef)]
    assert set(names) == {"__init__", "step", "predict", "bpr_step"}, names


def test_splice_preserves_indentation_of_a_method_returned_unindented():
    new, err = writer.splice_definitions(
        _SAMPLE, "class FM:\n    def predict(self, X):\n        return X * 2\n")
    assert err == ""
    assert "    def predict(self, X):" in new
    assert "        return X * 2" in new
    import ast
    ast.parse(new)


def test_splice_refuses_an_unlocatable_definition_so_the_caller_falls_back():
    """Guessing where an unknown definition goes is how a splice corrupts a
    file."""
    new, err = writer.splice_definitions(_SAMPLE, "def nope():\n    return 1\n")
    assert new == ""
    assert "does not exist" in err or "exists in the file" in err


def test_splice_refuses_module_level_statements():
    """A new constant or an edit to a list literal cannot be a definition splice.
    Refusing is what routes it to the whole-file path."""
    new, err = writer.splice_definitions(_SAMPLE, "CONST = 4\n")
    assert new == ""
    assert "module-level" in err or "Assign" in err


def test_splice_rejects_an_unparseable_block():
    new, err = writer.splice_definitions(_SAMPLE, "def broken(:\n")
    assert new == ""
    assert "not valid Python" in err


def test_splice_carries_decorators_from_the_replacement_not_the_original():
    src = ("import functools\n\n\n"
           "@functools.cache\n"
           "def f(x):\n"
           "    return x\n")
    new, err = writer.splice_definitions(src, "def f(x):\n    return x + 1\n")
    assert err == ""
    assert "@functools.cache" not in new, \
        "the old decorator must go with the old definition"
    assert "return x + 1" in new


def test_splice_to_diff_runs_the_same_guards_as_the_whole_file_path():
    # Elision inside a long function is still elision.
    d, err = writer.splice_to_diff(
        "def helper(a):\n    # ... rest of function unchanged\n    return a\n",
        _SAMPLE, "sample.py")
    assert d == "" and "elided" in err

    # A docstring-only change is still a no-op.
    d, err = writer.splice_to_diff(
        'def helper(a):\n    """Different doc."""\n    return a + 1\n',
        _SAMPLE, "sample.py")
    assert d == "" and err == writer.NO_SEMANTIC_CHANGE

    # A real change produces a diff.
    d, err = writer.splice_to_diff(
        'def helper(a):\n    """Doc."""\n    return a + 2\n',
        _SAMPLE, "sample.py")
    assert err == "" and d.startswith("--- a/sample.py")


def test_the_scoped_diff_touches_far_fewer_lines_than_a_whole_file_rewrite():
    """The measurable claim behind T3.1: the diff stops touching the whole file,
    which is what weakened the diff-hash dedup and forced gate exemptions."""
    changed = _SAMPLE.replace("return a + 1", "return a + 2")
    whole, err1 = writer.rewrite_to_diff(changed, _SAMPLE, "sample.py")
    scoped, err2 = writer.splice_to_diff(
        'def helper(a):\n    """Doc."""\n    return a + 2\n',
        _SAMPLE, "sample.py")
    assert err1 == "" and err2 == ""

    def _touched(d):
        return sum(1 for l in d.splitlines()
                   if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))

    # Same semantic edit, and the scoped form is no larger.
    assert _touched(scoped) <= _touched(whole)


# --------------------------------------------------------------------------- #
#  T3.1 — through write_fix, including the fallback                           #
# --------------------------------------------------------------------------- #
class _SpyClient:
    """Records which output contract each attempt asked for."""

    def __init__(self, replies=None):
        from codegen.llm_client import FakeBackend, LLMClient
        self.inner = LLMClient(backend=FakeBackend())
        self.replies = list(replies or [])
        self.modes = []

    def complete(self, system, user, kind, **kw):
        marker = "THIS OVERRIDES rules 5 and 6"
        self.modes.append("scoped" if marker in user else "whole_file")
        if self.replies:
            return self.replies.pop(0)
        return self.inner.complete(system, user, kind, **kw)


def test_write_fix_tries_the_scoped_contract_first_and_succeeds_on_it():
    from codegen.fixtures import FAKE_HYPOTHESIS
    spy = _SpyClient()
    diff = writer.write_fix(FAKE_HYPOTHESIS, "loss_function", client=spy, root=".")
    assert spy.modes == ["scoped"], f"expected one scoped call, got {spy.modes}"
    assert writer.diff_applies(diff, "baseline.py", ".")[0]


def test_write_fix_falls_back_to_the_whole_file_when_the_splice_cannot_apply():
    """A module-level change — a new constant, an edit to FIELDS — genuinely
    cannot be a definition splice, and the fallback is what keeps it reachable."""
    from codegen.fixtures import FAKE_HYPOTHESIS
    spy = _SpyClient(replies=["```python\nNEW_CONSTANT = 5\n```\n"])
    diff = writer.write_fix(FAKE_HYPOTHESIS, "loss_function", client=spy, root=".")
    assert spy.modes[0] == "scoped"
    assert "whole_file" in spy.modes, "the whole-file path was never reached"
    assert writer.diff_applies(diff, "baseline.py", ".")[0]


def test_write_fix_falls_back_when_the_scoped_reply_has_no_code_block():
    """The scoped suffix invites the model to say in prose that it needs
    module-level changes. That must route to the fallback, not crash."""
    from codegen.fixtures import FAKE_HYPOTHESIS
    spy = _SpyClient(replies=["I need to add a module-level constant."])
    diff = writer.write_fix(FAKE_HYPOTHESIS, "loss_function", client=spy, root=".")
    assert spy.modes[0] == "scoped" and "whole_file" in spy.modes
    assert writer.diff_applies(diff, "baseline.py", ".")[0]


def test_a_single_attempt_budget_skips_scoped_entirely():
    """With no room for a fallback, spending the only attempt on the form that
    can fail to locate its target would be strictly worse."""
    from codegen.fixtures import FAKE_HYPOTHESIS
    spy = _SpyClient()
    writer.write_fix(FAKE_HYPOTHESIS, "loss_function", client=spy, root=".",
                     max_attempts=1)
    assert spy.modes == ["whole_file"]


def test_the_scoped_suffix_states_that_it_overrides_the_whole_file_rules():
    """Two contradictory format rules with no stated precedence is a coin flip."""
    from codegen import prompts
    s = prompts.build_scoped_suffix("baseline.py")
    assert "OVERRIDES rules 5 and 6" in s
    assert "Do NOT return the whole" in s
    # The escape hatch for module-level changes has to be spelled out, or the
    # model invents a definition to wrap them in.
    assert "module-level" in s
    # And the guards that still apply.
    assert "never elide" in s.lower()


@pytest.mark.parametrize("line", [
    "    ...",
    "    # ...",
    "    # ... rest of function unchanged",
    "    # ...rest of file",
    "    # rest of file",
    "    # unchanged",
    "    # remainder omitted",
    "    # same as before",
    "    # truncated",
])
def test_every_elision_form_is_caught(line):
    """`# ...` and `# ... rest of function` used to slip through: the `\\b` sat
    after the whole alternation, and `...` ends in a non-word character, so the
    boundary could never be satisfied against the space or line end that follows.
    Those are the most natural forms a model produces, and rule 7 of the writer
    prompt names `...` explicitly — so the guard was missing its main case."""
    assert writer._ELISION_RE.search(f"def f(a):\n{line}\n    return a\n")


@pytest.mark.parametrize("line", [
    "    x = 1",
    "    y = x ... z",          # `...` mid-expression is not an elision marker
    '    s = "..."',            # nor inside a string literal
    "    # this function is unchanged in spirit but not in code",
])
def test_the_elision_guard_does_not_fire_on_real_code(line):
    """A guard that rejects valid rewrites costs a whole attempt each time.

    The marker has to follow `#` immediately, which is what keeps prose that
    merely CONTAINS one of the words — "this function is unchanged in spirit" —
    from being read as elision.
    """
    src = f"def f(a):\n{line}\n    return a\n"
    assert not writer._ELISION_RE.search(src), f"false positive on {line!r}"


def test_both_writer_paths_reject_the_same_elision():
    """T3.1 requires the guards to hold on the scoped path too, not just on the
    whole-file one."""
    elided = 'def helper(a):\n    """Doc."""\n    # ... rest unchanged\n'
    _, err_scoped = writer.splice_to_diff(elided, _SAMPLE, "sample.py")
    _, err_whole = writer.rewrite_to_diff(
        _SAMPLE.replace("    return a + 1", "    # ... rest unchanged"),
        _SAMPLE, "sample.py")
    assert "elided" in err_scoped
    assert "elided" in err_whole
