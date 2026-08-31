"""
System prompts + user-message builders for codegen's model calls.

The writer prompts are deliberately *gate-aware*: they tell the model the same
hard rules the static gate enforces, so generated diffs tend to pass the gate
and we don't waste a round-trip. The rules are stated as constraints, not as the
scanner's internals.
"""
from __future__ import annotations

import json
import textwrap

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
   fed into the model as an input feature array. They are already plumbed:
   data.aux_targets(splits) returns, per split, a float32
   (N, len(AUX_SIGNALS)) array aligned row-for-row with encode()'s X, so a
   multi-task head is a baseline.py-only change. Putting that array into X, a
   column_stack, or FIELDS is the leak the gate blocks; passing it as the
   target of an auxiliary loss term is the intended use.
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

# The runtime the generated code has to survive. Stated to the writer because
# half of the first run's candidates never produced runnable code — they were
# written against libraries that are not installed, or a cost budget the sandbox
# kills. Kept in sync with llm_calls/personas.py's _DATASET_CONTEXT.
_RUNTIME_RULES = """\
RUNTIME — code that violates this is killed before it scores:
- Importable: numpy and lightgbm. NOTHING else. No torch, tensorflow, sklearn,
  pandas or transformers — they are not installed and an ImportError wastes the
  whole candidate.
- One CPU core, single-threaded. The unmodified baseline runs in 18s; the
  sandbox kills a candidate at 240s. Vectorise with numpy — a per-row Python
  loop over the 1.14M training rows will not finish.
- Determinism matters: the search compares candidates at ~0.001, so seed
  anything stochastic from the --seed argument.
- ANY new randomness (negative sampling, dropout masks, row subsampling,
  tie-breaking) needs its OWN generator:
      neg_rng = np.random.default_rng(seed + 1000)
  NEVER draw from the existing `rng` that run_fm uses for its
  epoch permutation, and NEVER call np.random.* directly. The search scores this
  candidate by pairing its per-user results against its parent's at the same
  seed, which only cancels noise while both runs consume the same draws in the
  code they share; one extra draw from `rng` shifts every later epoch shuffle
  and the inflated variance reads as a real effect. The static gate rejects
  either form."""

WRITER_SYSTEM_MODEL = f"""\
You are a precise ML systems engineer editing baseline.py, which holds TWO
models: run_fm (a numpy Factorization Machine, the current champion at 0.5953
test primary) and run_lgb (a LightGBM ranker over dense train-only aggregate
features). Edit whichever the mechanism concerns; run_fm unless told otherwise.
You modify ONLY baseline.py's model / loss / training loop. You do not touch
data.py's feature encoding, evaluate.py, or submit.py. Keep the change minimal
and self-contained; preserve the public run/predict interface and the
`##CODEGEN_METRICS##` output line so the harness can still score outputs.
{_RUNTIME_RULES}
{_SAFETY_RULES}
{_DIFF_FORMAT_RULES}"""

WRITER_SYSTEM_DATA = f"""\
You are a precise ML feature engineer editing data.py's feature encoding. It
holds TWO encoders: encode() (the FIELDS list plus encode()/raw(), feeding the
numpy FM) and encode_lgb() (dense train-only aggregate features feeding the
LightGBM ranker). Edit whichever the mechanism concerns. You modify ONLY
data.py; you do not change the model or loss.

Any new feature must be derivable from columns known at serving time for that
row, and every aggregate statistic must be computed on the TRAIN split only and
then looked up for valid/test — train is strictly earlier in time, which is what
makes that safe.

A feature that is CONSTANT WITHIN A USER contributes exactly zero to this
metric, because the ranking is done inside each user. Pure user-side features
are only useful through an interaction with an item-side value. A LAG feature
(the user's previous video_id / author_id, whether their previous impression was
a long_view) is NOT user-constant — it varies row to row inside one user — which
is precisely why it can move this metric where a user-side aggregate cannot.

Adding a fixed-width categorical field is a legal, self-contained change, and
data.py has a REGISTRY for exactly it. Append one entry to EXTRA_FIELDS:

    EXTRA_FIELDS.append(('prev_video_id', prev_video_within_user))

where the function takes that split's `rows` and returns one value per row, in
row order. That single append is the WHOLE edit: encode() derives X's width, the
vocabularies, the UNK slots, field_dims and offsets from it, FIELDS follows, and
the FM picks up the new embedding because it indexes X. encode() returns X as
int32 (N, len(FIELDS)).

DO NOT append to FIELDS and return one more value from encode()'s inner raw(x).
raw(x) covers the five BASE_FIELDS and only those; returning an extra value from
it writes past X's last column, and that is the single most common way a
candidate has failed here — two candidates died with the identical IndexError
inside `for i, v in enumerate(raw(x))`. Use EXTRA_FIELDS and there is nothing to
keep in sync.

The registry function is handed ONE SPLIT's rows at a time, which is what makes
the two rules for a lag feature easy to satisfy. First: compute it using only
rows at or before the current row — never reaching forward, and never across the
split boundary. Second: leave the vocabulary train-only so an unseen value lands
in that field's UNK slot; encode() does the train-only part for you. The function
is also called once per split rather than once per row, so write it as a single
pass and never as a per-row Python callback over 1.4M rows.

DO NOT AUTHOR YOUR OWN VERSION OF THESE. data.py already has them, they are
tested for causality and split-locality, and re-implementing one is how a
candidate times out at 240s or leaks a label:

  data.prev_value_within_user(rows, key=2)   -> the user's previous video_id
  data.prev_value_within_user(rows, key=3)   -> previous author_id
  data.prev_value_within_user(rows, key=6)   -> whether the previous impression
                                                was a long_view
  data.prior_count_within_user(rows, key=3)  -> how many times this user has
                                                already seen this author,
                                                bucketed
  data.position_within_user(rows)            -> position in the user's log,
                                                bucketed
  data.within_user_pairs(users, y, rng)      -> vectorised (pos, neg) index
                                                pairs for a pairwise loss

So the whole edit for a lag field is one line:

    EXTRA_FIELDS.append(('prev_author_id',
                         lambda rows: prev_value_within_user(rows, key=3)))

`within_user_pairs` takes its generator as an ARGUMENT — call it with
np.random.default_rng(seed + 1000) and never with run_fm's `rng`, for the reason
given in the RNG rule below.
{_RUNTIME_RULES}
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


def build_scoped_suffix(file_name: str) -> str:
    """Override the whole-file output contract with a function-scoped one.

    Appended to the writer message rather than baked into the system prompt so
    the SAME system prompt serves both paths — the fallback needs the whole-file
    rules intact, and maintaining two copies of them is how they drift.

    The instruction here is the opposite of _DIFF_FORMAT_RULES rules 5-6, so it
    says so explicitly. A model handed two contradictory format rules with no
    precedence picks one at random.
    """
    return f"""

OUTPUT FORMAT — THIS OVERRIDES rules 5 and 6 above. Do NOT return the whole
file. Return ONLY the complete definitions you are changing, inside one
```python fenced block:

- One or more complete `def` blocks, reproduced in FULL from `def` to the last
  line of the body. Include decorators if the original has them.
- To change a METHOD, wrap it in its class so the target is unambiguous:

    class FM:
        def step(self, X, y):
            ...the complete new body...

  Only the methods you include are replaced; the class's other methods are left
  exactly as they are, so you must not reproduce them.
- A definition you include that does NOT already exist in {file_name} is treated
  as new and inserted next to the one you are replacing. So "edit FM.step and add
  FM.bpr_step" is one block containing both.
- A new `import` line at the top of the block is allowed.
- NOTHING ELSE. No module-level statements, no assignments outside a def, no
  prose. If your change genuinely needs module-level code — a new constant, an
  edit to a list literal like FIELDS — say exactly that in one line instead of a
  code block, and you will be re-asked for the whole file.
- Rules 7 and 8 still apply: never elide, and keep the change minimal.

The caller locates your definitions by name and splices them in, so you do not
reproduce {file_name}'s other ~400 lines and cannot introduce an error in them."""


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


def build_ancestor_block(ancestors: list | None) -> str:
    """Render a node's ancestor chain for the debug operator.

    This is AIRA's "ancestral memories for Debug" (arXiv:2507.02554) — their
    scoped-memory change to the Debug operator is what made advanced search
    policies pay off at all. The repair model used to see a traceback and a
    file, and nothing else: not what the edit was trying to do, and not that its
    two ancestors had already failed the same way. That operator handled 7 of
    the 11 candidates in the recorded run.

    Returns "" for an empty chain, so a root-level repair prompt is unchanged.
    """
    if not ancestors:
        return ""
    lines = ["", "ANCESTRY of this candidate — newest first. Read it before you",
             "choose a fix; it tells you which failures are already ruled out.",
             ""]
    for a in ancestors:
        lines.append(f"- {a.get('id', '?')} ({a.get('operation', 'improve')})")
        if a.get("mechanism"):
            lines.append(f"    mechanism: {a['mechanism']}")
        if a.get("evidence_type"):
            lines.append(f"    outcome:   {a['evidence_type']}")
        if a.get("mean_delta") is not None:
            lines.append(f"    paired delta vs its parent: {a['mean_delta']:+.5f}")
        if a.get("last_error_excerpt"):
            excerpt = " ".join(str(a["last_error_excerpt"]).split())
            lines.append(f"    failed with: {excerpt}")
    lines += [
        "",
        "How to use this, and the distinction matters:",
        "- An ancestor that FAILED TO RUN (failed_implementation, timeout) means",
        "  the mechanism was never measured. Do not abandon the idea — write a",
        "  DIFFERENT, cheaper implementation of the same mechanism, and do not",
        "  reproduce the construct that broke.",
        "- An ancestor that RAN BUT SCORED WORSE (a negative paired delta) means",
        "  the mechanism itself is suspect. A repair must NOT resurrect it: fix",
        "  only what crashes, and leave the failing mechanism no larger than the",
        "  ancestor already made it.",
        "",
    ]
    return "\n".join(lines)


def build_digest_block(prior_failures: list | None) -> str:
    """Prior failures elsewhere in the run with the SAME signature (T3.2).

    Distinct from `build_ancestor_block`, which walks the tree upward. The two
    identical `encode` crashes in the recorded run were SIBLINGS, so no ancestor
    walk could have shown either one the other's traceback. Keyed on
    (file, exception, line) instead, because that is how the failures cluster.

    Empty when the signature is new, which keeps the prompt unchanged in the
    common case.
    """
    if not prior_failures:
        return ""
    lines = ["", "SEEN BEFORE IN THIS RUN — the same file, exception and line "
                 "have already failed here:"]
    for e in prior_failures:
        outcome = ("the repair below FIXED it" if e.get("repaired") is True
                   else "the repair below did NOT fix it"
                   if e.get("repaired") is False
                   else "outcome unknown")
        lines.append(f"- candidate {e.get('node_id')} "
                     f"({e.get('stage', 'execute')}), {outcome}")
        if e.get("mechanism"):
            lines.append(f"  it was implementing: {e['mechanism']}")
        if e.get("repair_attempted"):
            lines.append(f"  repair attempted:\n"
                         f"{textwrap.indent(e['repair_attempted'], '    ')}")
    lines += [
        "",
        "Read this before repeating a repair that already failed. If a listed "
        "repair did NOT fix the failure, do something different — the same diff "
        "will fail the same way. If one DID fix it, the equivalent change here "
        "is the cheapest correct answer.",
        "Two candidates failing identically usually means the ORIGINAL file's "
        "contract was misread, not that both edits were careless: prefer a fix "
        "that makes the contract hard to get wrong over one that patches the "
        "symptom.",
        "",
    ]
    return "\n".join(lines)


def build_debug_user(file_name: str, file_content: str, error_context: str,
                     hypothesis: dict | None = None,
                     ancestors: list | None = None,
                     prior_failures: list | None = None) -> str:
    """`hypothesis` and `ancestors` are optional so the frozen 3-positional-arg
    form keeps working; codegen/tests and debug_and_retry's 2-argument contract
    both depend on it."""
    intent = ""
    if hypothesis:
        intent = f"""
This candidate was trying to implement:
{hypothesis.get('mechanism', '(none provided)')}

Implementation sketch it was written from:
{hypothesis.get('implementation_sketch', '(none provided)')}

Fix the failure while KEEPING that mechanism in place. A repair that quietly
reverts the mechanism to make the file run is worse than the crash: it scores
identically to the parent and is recorded as the mechanism having been tried.
"""
    return f"""\
File: {file_name}
The candidate failed. Error / divergence context:
```
{error_context}
```
{intent}{build_digest_block(prior_failures)}{build_ancestor_block(ancestors)}
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
