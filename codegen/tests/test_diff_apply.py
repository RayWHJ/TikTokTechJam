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
