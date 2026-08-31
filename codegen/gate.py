"""
codegen.pre_execution_gate — STATIC, deterministic safety scanner.

Runs BEFORE any compute is spent. No LLM. Parses the *added* lines of a unified
diff (the lines a candidate would introduce) and blocks on any of four rules.
The scanner biases toward blocking on ambiguity — a false block costs one cheap
re-generation; a false pass can burn a full run on a leaking candidate or, worse,
promote a cheating result.

Rules (block if the added code):
  1. references the TEST split / a test-named file in a data-loading context;
  2. uses any NON_CAUSAL_COLUMNS name without an explicit `point_in_time=True`
     marker nearby;
  3. imports external pretrained weights or external datasets / downloads;
  4. feeds a same-row AUXILIARY_SIGNALS value into the model as an INPUT feature
     array rather than only as a loss target (ambiguous usage also blocks);
  5. draws randomness from the legacy global `np.random.*` API, or from the
     generator named `rng` that carries run_fm's epoch permutation — either one
     breaks the common-random-numbers pairing the orchestrator's paired
     bootstrap rests on.

Returns {"pass": bool, "reasons": list[str]}.
"""
from __future__ import annotations
import re
from .constants import NON_CAUSAL_COLUMNS, AUXILIARY_SIGNALS


# --------------------------------------------------------------------------- #
#  Diff parsing                                                                #
# --------------------------------------------------------------------------- #
def _added_lines(code_diff: str) -> list[tuple[int, str]]:
    """Return (added_index, code_text) for each ADDED line in the diff.

    Handles both real unified diffs ('+' prefix, ignoring '+++' headers) and a
    bare code snippet (no diff markers -> treat every line as added, so the gate
    still works if a caller passes raw code)."""
    lines = code_diff.splitlines()
    has_markers = any(l.startswith(("+", "-", "@@", "diff ", "--- ", "+++ ")) for l in lines)
    out = []
    if not has_markers:
        for i, l in enumerate(lines):
            out.append((i, l))
        return out
    for i, l in enumerate(lines):
        if l.startswith("+++"):
            continue
        if l.startswith("+"):
            out.append((i, l[1:]))
    return out


def _removed_lines(code_diff: str) -> set[str]:
    """The stripped text of every REMOVED line in the diff.

    Used only by rule 5, to tell a genuinely new random draw from one of the
    baseline's own lines that merely moved. The driver builds its diffs by
    rewriting a whole file and diffing the result, so a line the writer did not
    touch can still surface as a `+` when the hunk around it shifts. Rules 1-4
    deliberately do NOT get this exemption: a leaking line is a leaking line
    whether or not it also appears as removed.
    """
    out = set()
    for l in code_diff.splitlines():
        if l.startswith("---"):
            continue
        if l.startswith("-"):
            out.add(l[1:].strip())
    return out


def _strip_inline_comment(code: str) -> str:
    """Remove a trailing # comment when it's clearly not inside a string.
    Conservative: if the line contains a quote we keep it whole (bias to block —
    we would rather scan too much than miss a string key like df['like_cnt'])."""
    if '"' in code or "'" in code:
        return code
    return code.split("#", 1)[0]


# --------------------------------------------------------------------------- #
#  Rule 1 — test-split access                                                  #
# --------------------------------------------------------------------------- #
_DATA_LOAD_HINTS = re.compile(
    r"(get_split|load\s*\(|encode\s*\(|read_csv|\.csv|open\s*\(|splits\s*\[|"
    r"--split|split\s*=|data_dir|np\.load|pd\.read|glob\s*\()", re.IGNORECASE)
_TEST_TOKEN = re.compile(r"\btest\b", re.IGNORECASE)
# explicit high-signal test-access forms
_TEST_EXPLICIT = re.compile(
    r"""(get_split\s*\(\s*['"]test['"]|splits\s*\[\s*['"]test['"]\s*\]|"""
    r"""--split\s+test|split\s*=\s*['"]test['"]|[\w]*_?test_?[\w]*\.csv|"""
    r"""\bX_?te\b|\bXte\b|\by_?te\b|\byte\b|\btest_split\b|\btest_data\b|\btest_set\b)""",
    re.IGNORECASE)

def _check_test_access(added: list[tuple[int, str]]) -> list[str]:
    reasons = []
    for _, raw in added:
        code = _strip_inline_comment(raw)
        if _TEST_EXPLICIT.search(code):
            reasons.append(f"test-split access: `{raw.strip()}`")
            continue
        if _TEST_TOKEN.search(code) and _DATA_LOAD_HINTS.search(code):
            reasons.append(f"possible test-split access (‘test’ near a data-load): `{raw.strip()}`")
    return reasons


# --------------------------------------------------------------------------- #
#  Rule 2 — non-causal statistic columns without point_in_time marker          #
# --------------------------------------------------------------------------- #
_PIT_MARKER = re.compile(r"point_in_time\s*=\s*True", re.IGNORECASE)
_NONCAUSAL_RE = {c: re.compile(r"\b" + re.escape(c) + r"\b") for c in NON_CAUSAL_COLUMNS}

def _check_non_causal(added: list[tuple[int, str]]) -> list[str]:
    reasons = []
    # a point_in_time=True anywhere in the added block, or within +/-3 added lines,
    # counts as "nearby". Build the set of added-line indices carrying the marker.
    marker_idx = {idx for idx, raw in added if _PIT_MARKER.search(raw)}
    idx_list = [idx for idx, _ in added]

    def marker_nearby(idx: int) -> bool:
        return any(abs(idx - m) <= 3 for m in marker_idx)

    for idx, raw in added:
        code = _strip_inline_comment(raw)
        for col, rx in _NONCAUSAL_RE.items():
            if rx.search(code):
                if not marker_nearby(idx):
                    reasons.append(
                        f"non-causal column ‘{col}’ used without a nearby "
                        f"point_in_time=True marker: `{raw.strip()}`")
    return reasons


# --------------------------------------------------------------------------- #
#  Rule 3 — external pretrained weights / datasets / downloads                 #
# --------------------------------------------------------------------------- #
_EXTERNAL_RE = re.compile(
    r"(from_pretrained|transformers|torchvision\.models|\btimm\b|torch\.hub|"
    r"load_dataset\s*\(|huggingface|hf_hub_download|snapshot_download|"
    r"\bwget\b|urllib\.request|requests\.get|\bgdown\b|kagglehub|kaggle\s|"
    r"https?://|s3://|gs://|\.pt\b|\.pth\b|\.ckpt\b|\.safetensors\b|\.bin\b|"
    r"openai|onnxruntime|\bfasttext\b|gensim\.downloader)", re.IGNORECASE)

def _check_external(added: list[tuple[int, str]]) -> list[str]:
    reasons = []
    for _, raw in added:
        code = _strip_inline_comment(raw)
        m = _EXTERNAL_RE.search(code)
        if m:
            reasons.append(f"external pretrained weights/dataset/download ‘{m.group(0)}’: `{raw.strip()}`")
    return reasons


# --------------------------------------------------------------------------- #
#  Rule 4 — same-row auxiliary signal used as an INPUT feature                 #
# --------------------------------------------------------------------------- #
_AUX_RE = {a: re.compile(r"\b" + re.escape(a) + r"\b") for a in AUXILIARY_SIGNALS}
# tokens that indicate the aux value is being wired in as an INPUT.
# Split by case-sensitivity: the feature-matrix names (X, X[, X_train) must be
# matched case-SENSITIVELY, otherwise "aux_target"/"max_" etc. false-trigger the
# `X_` pattern. The wordy indicators are matched case-insensitively.
_INPUT_TOKENS_CS = re.compile(r"(\bX\b|X\[|\bX_\w+)")
_INPUT_TOKENS_CI = re.compile(
    r"(FIELDS|features?\b|feat\b|\binput|encode|\braw\b|raw\(|"
    r"np\.column_stack|np\.concatenate|np\.stack|np\.hstack|self\.V\[|"
    r"embed|\bemb\b|offsets?)", re.IGNORECASE)

def _is_input_use(code: str) -> bool:
    return bool(_INPUT_TOKENS_CS.search(code) or _INPUT_TOKENS_CI.search(code))
# tokens that indicate it is only a TARGET / label / auxiliary loss
_TARGET_TOKENS = re.compile(
    r"(target|label|\by_|_y\b|loss|aux_loss|bce|logloss|criterion|_head\b|"
    r"multitask|multi_task|task_weight)", re.IGNORECASE)

# data.py's aux seam, by NAME. Rule 4 above matches individual signal names, and
# `X = np.column_stack([X, aux_targets(splits)['train']])` contains none of them
# — so without this the one legitimate accessor is also the one way to leak
# through it undetected.
#
# Matched NARROWLY, against feature-matrix construction only, rather than through
# _is_input_use. "aux_targets" contains the substring "target", and the natural
# multi-task loss line mentions X:
#     aux_loss = bce(m.aux_head(X), aux_targets(splits)['train'])
# _is_input_use fires on the bare `X` there, so reusing it would block the exact
# legitimate usage this seam exists to enable.
_AUX_SEAM = re.compile(r"\b(aux_targets|AUX_SIGNALS)\b")
_FEATURE_MATRIX_BUILD = re.compile(
    r"(np\.(column_stack|concatenate|hstack|vstack|stack|append)\s*\(|"
    r"^\s*X\w*\s*=|\bX\w*\s*=\s*np\.|X\[[^\]]*\]\s*=|"
    r"FIELDS\s*(\.append|\+|\[)|"
    r"^\s*(features?|inputs?|feats?)\s*=)")


def _check_aux_seam(added: list[tuple[int, str]]) -> list[str]:
    reasons = []
    for _, raw in added:
        code = _strip_inline_comment(raw)
        if _AUX_SEAM.search(code) and _FEATURE_MATRIX_BUILD.search(code):
            reasons.append(
                f"data.aux_targets/AUX_SIGNALS values built into a FEATURE "
                f"MATRIX. They are the same-row outcome for the row being "
                f"scored, so they are permitted only as auxiliary LOSS TARGETS: "
                f"`{raw.strip()}`")
    return reasons


def _check_auxiliary(added: list[tuple[int, str]]) -> list[str]:
    reasons = []
    for _, raw in added:
        code = _strip_inline_comment(raw)
        for aux, rx in _AUX_RE.items():
            if not rx.search(code):
                continue
            is_input = _is_input_use(code)
            is_target = bool(_TARGET_TOKENS.search(code))
            if is_input:
                reasons.append(
                    f"auxiliary signal ‘{aux}’ wired in as an INPUT feature "
                    f"(allowed only as a loss target): `{raw.strip()}`")
            elif not is_target:
                # ambiguous: aux referenced with neither clear target nor input
                # framing -> bias to block.
                reasons.append(
                    f"auxiliary signal ‘{aux}’ used ambiguously (no explicit "
                    f"loss-target framing; bias-to-block): `{raw.strip()}`")
    return reasons


# --------------------------------------------------------------------------- #
#  Rule 5 — randomness that breaks common-random-numbers pairing               #
# --------------------------------------------------------------------------- #
# orchestrator/promotion.py::bootstrap_delta pairs a candidate's per-user scores
# against its PARENT's, seed by seed. That pairing only reduces variance if the
# two runs consume the same random draws wherever they share code. In
# baseline.py::run_fm the model init takes its own generator
# (FM(..., seed=seed)) and the epoch shuffle takes another
# (rng = np.random.default_rng(seed)), whose sole consumer is
# `idx = rng.permutation(len(ytr))`. A candidate that adds negative sampling and
# draws from that same `rng` advances the stream, so from the first extra draw
# onward parent and candidate see different epoch shuffles at the same seed.
# The whole training trajectory decorrelates, paired variance inflates, and the
# inflation looks exactly like a real effect.
#
# The legacy global API is worse still: np.random.* draws from process-global
# state, so the sequence depends on import order and on how many draws anything
# else happened to make.
#
# Legal form: a NEW named generator, seeded off the run seed but offset —
#     neg_rng = np.random.default_rng(seed + 1000)
_NP_RANDOM_LEGACY = re.compile(r"\bnp(?:\.|umpy\.)random\s*\.\s*(\w+)")
_RNG_DRAW = re.compile(
    r"\brng\s*\.\s*(integers|choice|permutation|permuted|shuffle|random|"
    r"normal|uniform|standard_normal|binomial|poisson|beta|gamma|"
    r"exponential|bytes|spawn)\b")
# run_fm's own epoch shuffle, the one legitimate consumer of the `rng` stream.
# Exempted because a candidate that changes the loss almost always rewrites the
# epoch loop, which puts this untouched baseline line into the diff as an added
# line — blocking it would reject exactly the class of candidate this rule
# exists to make measurable.
_EPOCH_PERMUTATION = re.compile(r"\brng\s*\.\s*permutation\s*\(\s*len\s*\(")


def _check_rng_isolation(added: list[tuple[int, str]],
                         removed: set[str]) -> list[str]:
    reasons = []
    for _, raw in added:
        code = _strip_inline_comment(raw)
        if raw.strip() in removed:
            continue        # not new code — the same line is also removed
        m = _NP_RANDOM_LEGACY.search(code)
        if m and m.group(1) != "default_rng":
            reasons.append(
                f"legacy global randomness `np.random.{m.group(1)}` — draws "
                f"from process-global state, so it is not reproducible at a "
                f"fixed seed. Use a named generator, e.g. "
                f"`neg_rng = np.random.default_rng(seed + 1000)`: "
                f"`{raw.strip()}`")
            continue
        m = _RNG_DRAW.search(code)
        if m and not _EPOCH_PERMUTATION.search(code):
            reasons.append(
                f"draws `rng.{m.group(1)}` from run_fm's epoch-permutation "
                f"generator, which shifts that stream and breaks the "
                f"candidate-vs-parent common-random-numbers pairing the paired "
                f"bootstrap needs. Use a separate generator, e.g. "
                f"`neg_rng = np.random.default_rng(seed + 1000)`: "
                f"`{raw.strip()}`")
    return reasons


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #
def pre_execution_gate(code_diff: str) -> dict:
    """Static safety scan of a code diff. Returns {"pass": bool, "reasons": [...]}.

    `pass` is True only when no rule fired. Deterministic; no model call. Runs in
    microseconds so it can gate every candidate before any compute."""
    if not isinstance(code_diff, str) or not code_diff.strip():
        return {"pass": False, "reasons": ["empty or non-string diff"]}

    added = _added_lines(code_diff)
    if not added:
        return {"pass": False, "reasons": ["diff introduces no added lines to inspect"]}

    reasons: list[str] = []
    reasons += _check_test_access(added)
    reasons += _check_non_causal(added)
    reasons += _check_external(added)
    reasons += _check_auxiliary(added)
    reasons += _check_aux_seam(added)
    reasons += _check_rng_isolation(added, _removed_lines(code_diff))

    # de-duplicate while preserving order
    seen, uniq = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r); uniq.append(r)
    return {"pass": len(uniq) == 0, "reasons": uniq}
