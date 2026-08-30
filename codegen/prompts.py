"""
System prompts + user-message builders for codegen's model calls.

The writer prompts are deliberately *gate-aware*: they tell the model the same
hard rules the static gate enforces, so generated diffs tend to pass the gate
and we don't waste a round-trip. The rules are stated as constraints, not as the
scanner's internals.
"""
from __future__ import annotations
import json

# The rules the pre-execution gate enforces, in plain language for the model.
_SAFETY_RULES = """\
HARD RULES (a static gate rejects any diff that violates these, before it runs):
1. Never read, load, open, or reference the TEST split or any test-named file.
   Model selection uses the validation splits only.
2. Never use any non-causal aggregate statistic column (show_cnt, play_cnt,
   like_cnt, ... — the video_features_statistic_pure.csv columns) as a model
   INPUT feature. They are outcomes, not point-in-time-safe inputs. If and only
   if a value is genuinely known at serving time, mark it `point_in_time=True`
   right where it is used.
3. Never import external pretrained weights or external datasets (no
   from_pretrained, transformers, torch.hub, load_dataset, network downloads).
4. Same-row auxiliary signals (is_click, is_like, is_follow, is_comment,
   is_forward, play_time_ms) may be used ONLY as auxiliary LOSS TARGETS, never
   fed into the model as an input feature array.
Output ONLY a unified diff in a ```diff fenced block. No prose outside it."""

# Diff-format rules. A patch that does not apply is a wasted iteration, and the
# two failure modes below are the ones actually observed from the model: headers
# without the a/ b/ prefix (so `patch -p1` reports "can't find file to patch"),
# and normal/ed-format output (`18a19,22`, `> line`) which is not a patch at all.
_DIFF_FORMAT_RULES = """\
OUTPUT FORMAT — return the COMPLETE updated file, not a diff:
5. Output the entire file after your change, inside one ```python fenced block.
   The caller computes the diff itself, so you never write @@ headers.
6. Reproduce every unchanged line exactly as given, including comments,
   docstrings and non-ASCII text. Do NOT reformat or reindent untouched code.
7. NEVER elide anything. No `...`, no `# unchanged`, no `# rest of file`. A
   truncated file is rejected outright and the attempt is wasted.
8. Keep the change minimal: the output should differ from the input only where
   the mechanism requires it."""

WRITER_SYSTEM_MODEL = f"""\
You are a precise ML systems engineer editing a numpy Factorization Machine.
You modify ONLY baseline.py's model / loss / training loop. You do not touch
data.py's feature encoding, evaluate.py, or submit.py. Keep the change minimal
and self-contained; preserve the public run/predict interface so the harness can
still score outputs.
{_SAFETY_RULES}
{_DIFF_FORMAT_RULES}"""

WRITER_SYSTEM_DATA = f"""\
You are a precise ML feature engineer editing data.py's feature encoding for a
numpy Factorization Machine. You modify ONLY data.py (the FIELDS list and the
encode()/raw() logic). You do not change the model or loss. Any new feature must
be derivable from columns known at serving time for that row.
{_SAFETY_RULES}
{_DIFF_FORMAT_RULES}"""

DEBUG_SYSTEM = f"""\
You are debugging a failed candidate for a numpy FM ranking pipeline. Given the
current file and a traceback (or divergence report), produce the SMALLEST diff
that makes it run correctly. Do not opportunistically change the model, loss, or
features unless the fix genuinely requires it.
{_SAFETY_RULES}
{_DIFF_FORMAT_RULES}"""

SANITY_SYSTEM = """\
You are a skeptical ML reviewer. You are shown a code diff, the stated
hypothesis it was meant to implement, and a validation score that looks
suspiciously good relative to the run's history and the task's oracle ceiling
(primary <= 0.8645 is the theoretical maximum on this dataset). Decide whether
the diff actually implements the stated mechanism, or whether the number is more
likely produced by a bug or a data/label leak (e.g. reading the test split,
using an outcome column as an input, or peeking at labels).
Respond with STRICT JSON only:
{"implements_hypothesis": bool, "leak_suspected": bool, "reasoning": str}"""

REPORT_SYSTEM = """\
You are a technical writer producing a Devpost-style project description for a
hackathon submission (TikTok TechJam 2026, Track 2: Autonomous ML Research
Agent). Turn the structured run log into clear, honest markdown with these
sections: Inspiration, What it does, How we built it, Challenges we ran into,
Accomplishments, What we learned, What's next. Be specific and do not inflate
numbers beyond what the run log states. Output markdown only."""


# --------------------------------------------------------------------------- #
#  User-message builders                                                       #
# --------------------------------------------------------------------------- #
def build_writer_user(file_name: str, file_content: str,
                      hypothesis: dict, target_component: str) -> str:
    return f"""\
Target component: {target_component}
File to edit: {file_name}

Hypothesis mechanism:
{hypothesis.get('mechanism', '(none provided)')}

Implementation sketch:
{hypothesis.get('implementation_sketch', '(none provided)')}

Paired success criterion (for context, do not encode a threshold in the code):
{hypothesis.get('success_criterion_paired', '(none provided)')}

Current {file_name} content:
```python
{file_content}
```

Return the COMPLETE updated {file_name} in one ```python block, implementing the
mechanism above. Reproduce all untouched lines verbatim and elide nothing."""


def build_diff_repair_suffix(file_name: str, apply_error: str) -> str:
    """Appended to the writer message when the previous attempt was unusable.

    Feeding back the concrete reason is what makes the retry worth spending —
    the model sees whether it truncated the file, elided code, or produced an
    unapplyable patch, instead of guessing.
    """
    return f"""

Your previous answer was REJECTED: {apply_error.strip()}

Return the COMPLETE updated {file_name} again, in one ```python block. Every
line of the original must be present unless your change removes it. Do not
abbreviate, do not use `...`, and do not emit a diff."""


def build_semantic_repair_suffix(file_name: str, reason: str) -> str:
    """Appended when a previous attempt was rejected for changing nothing that RUNS.

    A distinct message from build_diff_repair_suffix: there the patch was
    unusable, here it applied cleanly and produced a candidate that scored
    bit-identically to its parent. Saying "aim at the executed code path" is the
    only instruction that changes the outcome, since the model is at
    temperature 0 and would otherwise re-emit the same annotated file.
    """
    return f"""

Your previous answer was REJECTED as a NO-OP: {reason.strip()}

Adding comments, docstrings, type hints, or an unreferenced helper function does
NOT count as implementing the mechanism. Change code that {file_name} actually
executes on the training path — a value that is read, a term in the loss, a
field in the feature list, a call site. Name the function you are modifying in a
comment on the changed line, then return the COMPLETE updated {file_name} in one
```python block."""


def build_debug_user(file_name: str, file_content: str, error_context: str) -> str:
    return f"""\
File: {file_name}
The candidate failed. Error / divergence context:
```
{error_context}
```

Current {file_name} content:
```python
{file_content}
```

Produce the smallest unified diff that fixes the failure."""


def build_sanity_user(code_diff: str, hypothesis: dict,
                      observed_score: float, history: list | None,
                      threshold: float | None) -> str:
    return f"""\
Stated hypothesis mechanism:
{hypothesis.get('mechanism', '(unknown)')}

Observed validation primary: {observed_score}
Prior history of validation primary: {history}
Practical/oracle context threshold: {threshold}

The code diff under review:
```diff
{code_diff}
```

Does this diff actually implement the stated mechanism, or is the score more
likely a bug/leak? Respond with the strict JSON schema."""


def build_report_user(run_log: dict) -> str:
    return ("Here is the structured run log as JSON. Write the Devpost markdown.\n\n"
            + json.dumps(run_log, indent=2, default=str))
