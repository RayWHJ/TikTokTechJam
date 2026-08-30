"""Repair the @@ hunk headers of a model-generated unified diff.

Observed failure mode, after the writer prompt already got the model to emit
well-formed `a/`/`b/` file headers: the @@ line COUNTS are wrong, so patch(1)
runs out of hunk mid-body and rejects the WHOLE patch:

    patch: **** malformed patch at line 52: +

Asking the model to count more carefully is not a fix — LLMs are unreliable at
arithmetic over long bodies, and each retry costs a call. The counts are fully
derivable from the hunk body, so derive them here.

Deliberately conservative: only the two counts are rewritten. Start lines are
left exactly as the model wrote them, because patch already relocates a hunk by
searching for its context (reporting "succeeded at N (offset M lines)") and is
better at it than we are — an earlier version that recomputed starts broke a
diff that had applied cleanly. Body lines are never reordered or invented, so a
hunk whose context genuinely mismatches the file still gets rejected by the
caller's dry-run.
"""
from __future__ import annotations

import re

_HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@(.*)$")


def _split_header_and_hunks(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return (preamble, hunks). Preamble is everything before the first @@."""
    first = next((i for i, l in enumerate(lines) if l.startswith("@@")), None)
    if first is None:
        return lines, []
    hunks: list[list[str]] = []
    for line in lines[first:]:
        if line.startswith("@@"):
            hunks.append([line])
        else:
            hunks[-1].append(line)
    return lines[:first], hunks


def _normalize_body(body: list[str]) -> tuple[list[str], int, int]:
    """Normalize body lines and count the old and new sides."""
    out: list[str] = []
    n_old = n_new = 0
    for line in body:
        if line.startswith("\\"):            # "\ No newline at end of file"
            out.append(line)
            continue
        if line == "":
            # A blank context line with its leading space dropped. Models do this
            # constantly and some patch builds reject it, so restore the space.
            line = " "
        elif not line.startswith((" ", "-", "+")):
            # An unprefixed line: read it as context, the interpretation that
            # preserves file content. The dry-run is still the arbiter.
            line = " " + line
        out.append(line)
        if line.startswith("-"):
            n_old += 1
        elif line.startswith("+"):
            n_new += 1
        else:
            n_old += 1
            n_new += 1
    return out, n_old, n_new


def normalize_unified_diff(diff: str, file_text: str | None = None) -> str:
    """Rewrite each @@ header so its counts match its body.

    `file_text` is accepted for call-site symmetry but is not needed: nothing
    here depends on the target file.
    """
    lines = diff.splitlines()
    preamble, hunks = _split_header_and_hunks(lines)
    if not hunks:
        return diff

    out = list(preamble)
    for hunk in hunks:
        m = _HUNK_RE.match(hunk[0])
        if not m:
            out.extend(hunk)
            continue
        body, n_old, n_new = _normalize_body(hunk[1:])
        out.append(f"@@ -{m.group(1)},{n_old} +{m.group(3)},{n_new} @@{m.group(5) or ''}")
        out.extend(body)

    return "\n".join(out) + "\n"
