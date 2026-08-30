"""
codegen.write_fix — generate a code diff implementing a hypothesis.

Routing (per the frozen contract): if target_component is about
features / history / auxiliary signals, frame the task as "extend data.py's
feature encoding"; otherwise frame it as "modify baseline.py's model / loss /
training loop." The relevant existing file's content is passed to the model as
context, together with hypothesis['mechanism'] and hypothesis['implementation_sketch'].
"""
from __future__ import annotations
import ast, difflib, os, re, shutil, subprocess, tempfile

from .diffnorm import normalize_unified_diff

from .constants import FEATURE_COMPONENTS
from . import prompts
from .llm_client import LLMClient, get_default_client, KIND_DIFF


def _is_feature_component(target_component: str) -> bool:
    """True -> edit data.py (features); False -> edit baseline.py (model/loss)."""
    tc = (target_component or "").strip().lower()
    if tc in FEATURE_COMPONENTS:
        return True
    # substring signals so callers can pass e.g. "add_user_history_feature"
    return any(tok in tc for tok in
               ("feature", "history", "sequence", "auxiliary", "aux", "encoding",
                "field", "embedding_input"))


def _read_root_file(name: str, root: str) -> str:
    path = os.path.join(root, name)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_diff(text: str) -> str:
    """Return the unified diff. Prefer a ```diff fenced block; else the first
    ``` block; else the raw text (already a diff)."""
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("\n") + "\n"
    m = re.search(r"```(?:patch|python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("\n") + "\n"
    return text.strip() + "\n"


#: Strip levels and tools tried, in order, when checking whether a diff applies.
#: Both -p1 (`a/baseline.py`) and -p0 (`baseline.py`) headers occur in practice.
#: stdin is closed because `patch` prompts "File to patch:" when no level fits.
_APPLY_CHECKS = (
    ["patch", "-p1", "--dry-run", "-i", "_patch.diff"],
    ["patch", "-p0", "--dry-run", "-i", "_patch.diff"],
    ["git", "apply", "--check", "--unsafe-paths", "_patch.diff"],
    ["git", "apply", "--check", "--unsafe-paths", "-p0", "_patch.diff"],
)


#: Rewrites shorter than this fraction of the original are treated as truncated.
_MIN_REWRITE_RATIO = 0.6
#: Elision markers that mean the model summarised instead of reproducing code.
_ELISION_RE = re.compile(
    r"^\s*(?:\.\.\.|#\s*(?:\.\.\.|rest of|remainder|unchanged|as before|"
    r"same as|omitted|truncated)\b)", re.IGNORECASE | re.MULTILINE)


def _extract_python(text: str) -> str | None:
    """Return the largest ```python (or bare ```) fenced block, if any."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else None


#: Message fed back to the model when its rewrite changed no executable code.
NO_SEMANTIC_CHANGE = ("the rewrite changed only comments, docstrings or "
                      "formatting — the executable code is byte-identical, so "
                      "this candidate would score exactly the same as its parent")


#: AST nodes whose leading string literal is a docstring, not a statement.
_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop every docstring in place, so prose changes read as no change."""
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str) and len(node.body) > 1):
            del node.body[0]
    return tree


def changes_executable_code(original: str, rewritten: str) -> bool:
    """True if `rewritten` differs from `original` in code that actually runs.

    Compares ASTs with docstrings stripped, so added or edited comments,
    docstrings, blank lines and reformatting all read as "no change". This is the
    cheap half of the no-op guard: it catches the "helpfully annotated the file
    and changed nothing" rewrite for the price of two ast.parse calls, before a
    gate + audit + sandbox run that can only reproduce the parent's score.

    It cannot catch a rewrite that adds a helper function and never calls it —
    the AST does change there. The driver's empirical check (candidate per-user
    scores identical to the parent's) is what catches that half.

    Unparseable input returns True: deciding no-op-ness is not this function's
    job, and a syntax error should surface from the sandbox with a traceback.
    """
    try:
        old_tree = _strip_docstrings(ast.parse(original))
        new_tree = _strip_docstrings(ast.parse(rewritten))
    except SyntaxError:
        return True
    return ast.dump(old_tree) != ast.dump(new_tree)


def rewrite_to_diff(rewritten: str, original: str, file_name: str) -> tuple[str, str]:
    """Turn a full-file rewrite into a unified diff. Returns (diff, error).

    Asking the model for the whole file and diffing locally removes every
    patch-format failure mode at once — no hunk headers to miscount, no context
    lines to retype, no strip level to guess. A difflib-generated diff applies by
    construction. The cost is guarding against the one new failure mode: a model
    that abbreviates the file instead of reproducing it.
    """
    if not rewritten.strip():
        return "", "empty rewrite"
    if _ELISION_RE.search(rewritten):
        return "", ("the file was elided (`...` / `# rest of file`) instead of "
                    "reproduced in full")
    old_lines = original.splitlines(keepends=True)
    new_lines = rewritten.splitlines(keepends=True)
    if not new_lines or len(new_lines) < _MIN_REWRITE_RATIO * len(old_lines):
        return "", (f"rewrite looks truncated: {len(new_lines)} lines vs "
                    f"{len(old_lines)} in the original")
    if not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    diff = "".join(difflib.unified_diff(
        old_lines, new_lines, f"a/{file_name}", f"b/{file_name}", n=3))
    if not diff.strip():
        return "", "rewrite is identical to the original (no change made)"
    if not changes_executable_code(original, rewritten):
        return "", NO_SEMANTIC_CHANGE
    return diff, ""


def diff_applies(diff: str, file_name: str, root: str = ".") -> tuple[bool, str]:
    """Dry-run `diff` against a pristine copy of file_name.

    Returns (ok, error_text). Checking here rather than at execute time means a
    malformed patch costs one cheap retry instead of a gate + audit + sandbox
    run that can only fail.
    """
    if not diff.strip():
        return False, "empty diff"
    work = tempfile.mkdtemp(prefix="codegen_diffcheck_")
    try:
        src = os.path.join(root, file_name)
        if not os.path.exists(src):
            return False, f"{file_name} not found under root {root!r}"
        shutil.copy2(src, os.path.join(work, file_name))
        with open(os.path.join(work, "_patch.diff"), "w", encoding="utf-8") as fh:
            fh.write(diff)
        errors = []
        for cmd in _APPLY_CHECKS:
            proc = subprocess.run(cmd, cwd=work, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                return True, ""
            errors.append(f"$ {' '.join(cmd)}\n"
                          f"{(proc.stdout + proc.stderr).strip()}")
        return False, "\n\n".join(errors)
    except FileNotFoundError as exc:          # no patch/git on PATH at all
        return False, f"no patch tool available: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def write_fix(hypothesis: dict, target_component: str, *,
              client: LLMClient | None = None, root: str = ".",
              max_attempts: int = 3,
              semantic_feedback: str | None = None) -> str:
    """Generate a code diff/patch (as text) implementing `hypothesis`.

    Parameters
    ----------
    hypothesis : dict
        Must contain 'mechanism' and 'implementation_sketch'
        (as produced by llm.generate_hypothesis). 'success_criterion_paired'
        is used as context only.
    target_component : str
        Routes the prompt. Feature/history/auxiliary -> data.py; anything else
        (model/loss/training/schedule) -> baseline.py.
    client : LLMClient, optional
        Model client. Defaults to the process client (offline fake backend unless
        a real backend is configured). Inject the real one from the orchestrator.
    root : str
        Repo root containing baseline.py / data.py (default: current dir). Pass
        the PARENT node's staged directory to build on its modifications rather
        than re-editing the pristine baseline.
    semantic_feedback : str, optional
        Why the caller rejected a previous attempt on semantic (not syntactic)
        grounds — e.g. it ran but scored bit-identically to the parent. Included
        in the prompt so the retry is aimed at the executed code path.

    Returns
    -------
    str
        A unified-diff patch as text (the ```diff fence stripped).
    """
    client = client or get_default_client()
    if _is_feature_component(target_component):
        file_name, system = "data.py", prompts.WRITER_SYSTEM_DATA
    else:
        file_name, system = "baseline.py", prompts.WRITER_SYSTEM_MODEL

    content = _read_root_file(file_name, root)
    user = prompts.build_writer_user(file_name, content, hypothesis, target_component)
    if semantic_feedback:
        user += prompts.build_semantic_repair_suffix(file_name, semantic_feedback)

    # Validate-and-repair. The prompt asks for a full-file rewrite, which we diff
    # locally; the retry is only worth spending because the message changes (the
    # client runs at temperature 0) and carries the concrete rejection reason.
    diff, err = "", ""
    for attempt in range(1, max(1, max_attempts) + 1):
        msg = user if attempt == 1 else \
            user + prompts.build_diff_repair_suffix(file_name, err)
        raw = client.complete(system, msg, kind=KIND_DIFF,
                              max_tokens=16000, temperature=0.0)

        block = _extract_python(raw)
        if block is not None and not block.lstrip().startswith(("--- ", "diff --git")):
            diff, err = rewrite_to_diff(block, content, file_name)
        else:
            # The model returned a patch anyway. Take it, but repair the hunk
            # counts first — that alone is what turns "malformed patch" (whole
            # patch rejected) into something applyable.
            candidate = _extract_diff(raw)
            diff, err = candidate, ""
            if not diff_applies(candidate, file_name, root)[0]:
                repaired = normalize_unified_diff(candidate, content)
                if diff_applies(repaired, file_name, root)[0]:
                    diff = repaired

        if diff:
            ok, apply_err = diff_applies(diff, file_name, root)
            if ok:
                return diff
            err = err or apply_err
    # Give up and return the last attempt: the caller's staging step will reject
    # it and log the reason, which keeps the failure visible in progress.json.
    return diff


def write_refine(hypothesis: dict, component: str, *,
                 client: LLMClient | None = None, root: str = ".",
                 max_attempts: int = 3,
                 semantic_feedback: str | None = None) -> str:
    """A refine is a write_fix scoped to one component.

    We delegate to write_fix after resolving the component to the writer's
    existing target_component vocabulary. Going through the registry rather
    than passing the component name straight to _is_feature_component makes
    the file routing explicit: "regularization" and "capacity" both live in
    baseline.py, which the substring heuristic gets right only by accident.

    The mechanism/implementation_sketch fields in `hypothesis` come from the
    refiner persona (see personas.py). `semantic_feedback` is threaded through
    so the driver's no-op rewrite retry works for refine candidates too.
    """
    from .ablations import ABLATIONS
    abl = ABLATIONS.get(component)
    if abl is None:
        raise ValueError(f"unknown component {component!r}")
    return write_fix(hypothesis, target_component=abl.target,
                     client=client, root=root, max_attempts=max_attempts,
                     semantic_feedback=semantic_feedback)
