"""
codegen.write_fix — generate a code diff implementing a hypothesis.

Routing (per the frozen contract): if target_component is about
features / history / auxiliary signals, frame the task as "extend data.py's
feature encoding"; otherwise frame it as "modify baseline.py's model / loss /
training loop." The relevant existing file's content is passed to the model as
context, together with hypothesis['mechanism'] and hypothesis['implementation_sketch'].
"""
from __future__ import annotations
import ast, difflib, os, re, shutil, subprocess, tempfile, textwrap

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
#:
#: The `\b` applies only to the WORD alternatives. It used to sit after the whole
#: inner alternation, which made `# ...` unmatchable: `...` ends in a non-word
#: character, so `\b` could never be satisfied against the space or line end that
#: follows. Measured before the fix — `# ...` and `# ... rest of function
#: unchanged` both passed the guard, while `# rest of file` and `# unchanged`
#: were caught. `# ...` is the most natural form a model produces, and rule 7 of
#: the writer prompt names it explicitly, so the guard was missing its main case.
#:
#: `[ \t]*` rather than `\s*` at the start so a match cannot begin by consuming
#: the previous line's newline.
_ELISION_RE = re.compile(
    r"^[ \t]*(?:\.\.\.|#\s*(?:\.\.\.|(?:rest of|remainder|unchanged|as before|"
    r"same as|omitted|truncated)\b))", re.IGNORECASE | re.MULTILINE)


def _extract_python(text: str) -> str | None:
    """Return the largest ```python (or bare ```) fenced block, if any."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else None


#: Message fed back to the model when its rewrite changed no executable code.
NO_SEMANTIC_CHANGE = ("the rewrite changed only comments, docstrings or "
                      "formatting — the executable code is byte-identical, so "
                      "this candidate would score exactly the same as its parent")

#: How many of `write_fix`'s attempts use the function-SCOPED contract before
#: falling back to the whole-file rewrite.
#:
#: One. The scoped form is strictly better when it works, but the cases it cannot
#: express — a new module-level constant, an edit to the FIELDS list literal —
#: are common enough here that burning two of three attempts on it would be worse
#: than trying it once and moving on. Set to 0 to disable scoped edits entirely;
#: `write_fix` also skips them when max_attempts leaves no room for a fallback,
#: so a caller asking for a single attempt still gets the whole-file path.
SCOPED_ATTEMPTS = 1


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


# --------------------------------------------------------------------------- #
#  Function-scoped edits (T3.1)                                                #
# --------------------------------------------------------------------------- #
# WHY. The whole-file rewrite removed every patch-FORMAT failure mode, but it
# replaced them with a reproduction problem: the model has to re-emit ~500 lines
# correctly to change five. Three consequences, all visible in this repo:
#
#   * `codegen/gate.py::_removed_lines` needs an exemption for lines that "merely
#     moved", and the RNG rule has to distinguish a new draw from a shifted hunk —
#     both are artefacts of a diff that touches the whole file.
#   * the run-wide diff-hash dedup weakens, because an unrelated formatting change
#     makes a semantically identical edit hash differently.
#   * it is the dominant token cost after the model swap — ~4,300 output tokens
#     per writer call at $20 per Mtok.
#
# So ask for only the definitions being changed, locate them by AST, and splice
# them into the original text. The whole-file path stays as the fallback for when
# a definition cannot be located, and both paths run the same elision and
# no-op guards afterwards.
#
# Line-based splicing, deliberately, NOT ast.unparse: unparsing would reformat
# and re-emit the entire file, producing exactly the huge diff this task exists to
# avoid, and would discard every comment in it.

#: A located definition in a source file: the line span it occupies (1-based,
#: inclusive) and the indentation its body sits at.
class _Span:
    __slots__ = ("start", "end", "indent")

    def __init__(self, start: int, end: int, indent: str):
        self.start, self.end, self.indent = start, end, indent


def _def_start(node) -> int:
    """First line of a definition, decorators included.

    `node.lineno` points at the `def`, so splicing from there would leave the old
    decorators attached to the new function.
    """
    lines = [node.lineno]
    lines += [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines)


_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _index_definitions(source: str) -> dict:
    """Map every top-level function and method to its line span.

    Keys are `"name"` for a module-level function and `"Class.name"` for a
    method, which is the vocabulary the writer prompt uses ("FM.step").
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    src_lines = source.splitlines()
    out: dict = {}

    def _indent_of(lineno: int) -> str:
        line = src_lines[lineno - 1] if 0 < lineno <= len(src_lines) else ""
        return line[:len(line) - len(line.lstrip())]

    for node in tree.body:
        if isinstance(node, _DEF_TYPES):
            s = _def_start(node)
            out[node.name] = _Span(s, node.end_lineno, _indent_of(s))
        elif isinstance(node, ast.ClassDef):
            cls_span = _Span(_def_start(node), node.end_lineno,
                             _indent_of(_def_start(node)))
            out[f"class {node.name}"] = cls_span
            for sub in node.body:
                if isinstance(sub, _DEF_TYPES):
                    s = _def_start(sub)
                    out[f"{node.name}.{sub.name}"] = _Span(
                        s, sub.end_lineno, _indent_of(s))
    return out


def _parse_replacements(block: str) -> tuple[dict, str]:
    """Definitions the model's block provides, keyed as `_index_definitions` is.

    Accepts a bare function, several functions, or a `class X:` wrapper holding
    methods. A class wrapper contributes its METHODS rather than the class itself:
    a model asked for one method commonly re-emits the class header around it and
    omits the other methods, and replacing the whole class would silently delete
    them.

    Returns ({key: source_text}, error).
    """
    block = textwrap.dedent(block).strip("\n")
    if not block.strip():
        return {}, "empty replacement block"
    try:
        tree = ast.parse(block)
    except SyntaxError as e:
        return {}, f"replacement block is not valid Python: {e}"

    lines = block.splitlines()

    def _segment(node) -> str:
        return "\n".join(lines[_def_start(node) - 1:node.end_lineno])

    out: dict = {}
    for node in tree.body:
        if isinstance(node, _DEF_TYPES):
            out[node.name] = _segment(node)
        elif isinstance(node, ast.ClassDef):
            methods = [s for s in node.body if isinstance(s, _DEF_TYPES)]
            if not methods:
                return {}, (f"class {node.name} in the replacement block has no "
                            f"methods to splice")
            for sub in methods:
                out[f"{node.name}.{sub.name}"] = textwrap.dedent(_segment(sub))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.setdefault("__imports__", "")
            out["__imports__"] += _segment(node) + "\n"
        else:
            return {}, (
                f"the replacement block contains a top-level "
                f"{type(node).__name__} statement. Return ONLY the complete "
                f"function or method definitions you are changing (plus any new "
                f"import), not module-level code — module-level changes need the "
                f"whole-file form.")
    if not any(k != "__imports__" for k in out):
        return {}, "the replacement block defines no function or method"
    return out, ""


def splice_definitions(original: str, block: str) -> tuple[str, str]:
    """Replace the named definitions in `original` with those in `block`.

    Returns (new_source, error). A definition present in `block` but ABSENT from
    `original` is inserted immediately before the first definition being
    replaced, so "edit FM.step and add FM.bpr_step" is one splice rather than a
    fallback to the whole file.

    Fails (so the caller falls back) when no definition in `block` matches
    anything in `original` — that means the model did not scope its answer to
    something locatable, and guessing where to put it is how a splice corrupts a
    file.
    """
    replacements, err = _parse_replacements(block)
    if err:
        return "", err
    imports = replacements.pop("__imports__", "")

    index = _index_definitions(original)
    if not index:
        return "", "could not parse the original file to locate definitions"

    known = {k: v for k, v in replacements.items() if k in index}
    unknown = {k: v for k, v in replacements.items() if k not in index}
    if not known:
        return "", (f"none of {sorted(replacements)} exists in the file "
                    f"(top-level definitions are {sorted(k for k in index if not k.startswith('class '))}); "
                    f"cannot splice")

    lines = original.splitlines()
    # Apply spans BOTTOM-UP so earlier line numbers stay valid as we edit.
    edits = sorted(((index[k], text, k) for k, text in known.items()),
                   key=lambda t: t[0].start, reverse=True)
    # Captured BEFORE editing: where new definitions go, and at what indentation.
    # A method's new sibling belongs inside the class, at the class body's indent,
    # which is the indent of the definition it sits next to.
    first = min((sp for sp, _t, _k in edits), key=lambda sp: sp.start)
    insert_at, insert_indent = first.start, first.indent

    for span, text, _key in edits:
        body = textwrap.indent(textwrap.dedent(text).strip("\n"), span.indent)
        lines[span.start - 1:span.end] = body.splitlines()

    if unknown:
        block_lines: list = []
        for _key, text in unknown.items():
            block_lines += textwrap.indent(
                textwrap.dedent(text).strip("\n"), insert_indent).splitlines()
            block_lines.append("")
        lines[insert_at - 1:insert_at - 1] = block_lines

    if imports.strip():
        # After the module docstring and any existing imports, which is the only
        # placement that cannot shadow a name the file already binds.
        lines[0:0] = [l for l in imports.strip().splitlines()]

    new_source = "\n".join(lines)
    if not new_source.endswith("\n"):
        new_source += "\n"
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        return "", f"the spliced file does not parse: {e}"
    return new_source, ""


def splice_to_diff(block: str, original: str, file_name: str
                   ) -> tuple[str, str]:
    """Turn a function-scoped replacement into a unified diff. (diff, error).

    Runs the SAME guards the whole-file path runs — elision on the block, and
    `changes_executable_code` on the resulting file — because both failure modes
    are still reachable here: a model can elide the middle of a long function, and
    it can return the function with only its docstring changed.
    """
    if not block.strip():
        return "", "empty replacement block"
    if _ELISION_RE.search(block):
        return "", ("the replacement was elided (`...` / `# rest of function`) "
                    "instead of reproduced in full")
    new_source, err = splice_definitions(original, block)
    if err:
        return "", err
    if not changes_executable_code(original, new_source):
        return "", NO_SEMANTIC_CHANGE
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), new_source.splitlines(keepends=True),
        f"a/{file_name}", f"b/{file_name}", n=3))
    if not diff.strip():
        return "", "the splice produced no change"
    return diff, ""


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

    # Validate-and-repair, over TWO output contracts.
    #
    # Attempt 1 asks for only the definitions being changed and splices them in
    # (T3.1): a ~40-line answer instead of a ~500-line one, so there are ~460
    # fewer lines the model can get wrong, the diff stops touching the whole file,
    # and the output token cost drops by roughly an order of magnitude.
    #
    # Later attempts fall back to the whole-file rewrite. That is not a
    # concession — a change to module-level code (a new constant, an edit to the
    # FIELDS list literal) genuinely cannot be expressed as a definition splice,
    # and `splice_definitions` refuses to guess. The fallback is REACHED by the
    # same error-feedback mechanism as any other rejection, so it costs no extra
    # call: max_attempts is unchanged.
    diff, err = "", ""
    total = max(1, max_attempts)
    for attempt in range(1, total + 1):
        scoped = attempt <= SCOPED_ATTEMPTS and total > SCOPED_ATTEMPTS
        msg = user
        if scoped:
            msg += prompts.build_scoped_suffix(file_name)
        if attempt > 1:
            msg += prompts.build_diff_repair_suffix(file_name, err)
        raw = client.complete(system, msg, kind=KIND_DIFF,
                              max_tokens=16000, temperature=0.0)

        block = _extract_python(raw)
        if scoped:
            if block is None:
                # No code block at all: the model said it needs module-level
                # changes (which the scoped suffix invites it to say in prose),
                # or it ignored the format. Either way the next attempt is
                # whole-file, which is what it needs.
                diff, err = "", (
                    "no ```python block in the scoped reply: "
                    f"{raw.strip()[:300]!r}")
            else:
                diff, err = splice_to_diff(block, content, file_name)
        elif block is not None and not block.lstrip().startswith(("--- ", "diff --git")):
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
