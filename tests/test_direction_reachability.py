"""Tests that the untouched research directions are actually REACHABLE — that
the prompts state the facts a proposer needs to believe a proposal is
implementable, and that those facts are true of the code.

Tier 4B is a prompt-only change, so most of these assert on prompt strings. That
is the point: the proposer had no reason to believe a sequence feature fit the
one-file budget, so it never proposed one. 0 of 11 candidates in the recorded run
attacked direction 2.

The last group is the important one — it checks the prompts against data.py
rather than just for the presence of words, so a future change to FIELDS or
encode() makes these fail instead of leaving the prompt quietly lying.
"""
import inspect

import pytest

import data
from codegen import prompts
from llm_calls import personas


# --------------------------------------------------------------------------- #
#  The shape fact has to reach both the proposer and the writer                #
# --------------------------------------------------------------------------- #
def test_proposer_is_told_appending_to_fields_is_a_legal_one_file_change():
    """Without this the proposer treats a new categorical field as out of
    scope, because the one-file rule reads like it forbids touching data.py."""
    ctx = personas._DATASET_CONTEXT
    assert "FIELDS" in ctx
    assert "LEGAL SINGLE-FILE CHANGE" in ctx.upper()
    assert "int32 (N, len(FIELDS))" in ctx


def test_writer_is_told_the_same_shape_fact():
    """The proposer proposing it is useless if the writer refuses to write it."""
    assert "append" in prompts.WRITER_SYSTEM_DATA.lower()
    assert "FIELDS" in prompts.WRITER_SYSTEM_DATA
    assert "int32 (N, len(FIELDS))" in prompts.WRITER_SYSTEM_DATA


def test_variable_length_sequences_are_ruled_out_explicitly():
    """A proposal needing a ragged tensor cannot be built here, and saying so is
    what keeps the sequence direction from being proposed in an unusable form."""
    assert "VARIABLE-LENGTH" in personas._DATASET_CONTEXT


# --------------------------------------------------------------------------- #
#  The concrete lag forms                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form", [
    "PREVIOUS video_id",
    "PREVIOUS author_id",
    "long_view",
    "bucketed",
    "position of the row within the user's log",
])
def test_named_lag_forms_are_present(form):
    assert form in personas._DATASET_CONTEXT


def test_both_causality_constraints_are_stated():
    """A lag that reaches forward is leakage; a vocabulary built off train is a
    different leak. Both have to be said, in both prompts."""
    for text in (personas._DATASET_CONTEXT, prompts.WRITER_SYSTEM_DATA):
        assert "at or before the current row" in text
        assert "train only" in text.lower() or "train-only" in text.lower()


def test_the_user_constant_distinction_is_drawn_in_both_prompts():
    """This is the load-bearing insight: README records that a user-side feature
    contributes exactly zero because ranking is within-user, and the whole case
    for a lag field is that it is NOT user-constant. A prompt that states the
    dead end without the exception reads as forbidding the direction."""
    for text in (personas._DATASET_CONTEXT, prompts.WRITER_SYSTEM_DATA):
        assert "CONSTANT WITHIN A USER" in text
        assert "NOT user-constant" in text


def test_the_prompt_does_not_promise_a_timestamp_that_is_not_loaded():
    """data.load() reads `date` and no finer temporal column. Telling the
    proposer to sort by hourmin would produce code that cannot run, which is the
    exact failure mode this tier is trying to reduce."""
    ctx = personas._DATASET_CONTEXT
    assert "hourmin" in ctx, "the limitation must be stated, not omitted"
    assert "are not loaded" in ctx or "not loaded" in ctx
    src = inspect.getsource(data.load)
    assert "hourmin" not in src, \
        "if load() gains hourmin, update the prompt — it currently says it has none"


# --------------------------------------------------------------------------- #
#  Against the real budget                                                     #
# --------------------------------------------------------------------------- #
def test_the_unbuildable_families_are_named_concretely():
    """61 recorded proposals from earlier runs named MAML, NAS, DeepFM,
    contrastive learning, state-space models or an LLM — none writable in numpy
    on one core. Naming them beats restating 'numpy only'."""
    ctx = personas._DATASET_CONTEXT
    for family in ("MAML", "NAS", "DeepFM", "contrastive", "state-space", "LLM"):
        assert family in ctx, family
    # And the constructive requirement that replaces the ban.
    assert "name the numpy operations" in ctx


# --------------------------------------------------------------------------- #
#  The prompts must not lie about data.py                                      #
# --------------------------------------------------------------------------- #
def test_fields_list_in_the_prompt_matches_data_py():
    assert data.FIELDS == ['user_id', 'video_id', 'author_id', 'tab',
                           'dur_bucket']
    rendered = str(data.FIELDS)
    assert rendered in personas._DATASET_CONTEXT, \
        "the prompt quotes FIELDS verbatim; regenerate it if data.py changed"


def test_encode_really_returns_the_shape_the_prompt_claims():
    """The claim 'X is int32 (N, len(FIELDS))' is what makes a proposer believe
    appending a field is cheap. Verify it against the source, not the docstring."""
    src = inspect.getsource(data.encode)
    assert "len(FIELDS)" in src and "np.int32" in src
    assert "for i, v in enumerate(raw(x))" in src, \
        "the prompt tells the writer to return one more value from raw(x)"


def test_x_width_tracks_fields_and_raw_must_be_extended_with_it():
    """Exercises the exact two-part edit the prompt describes, on a synthetic
    split so it needs no CSVs.

    X's width is driven by len(FIELDS), so appending a name is what makes room
    for the feature. But raw(x) is a separate list literal inside encode(), and
    the enumerate(raw(x)) loop only writes as many columns as raw returns — so
    appending to FIELDS ALONE leaves the new column unwritten by np.empty and
    its vocabulary empty, contributing only a UNK slot. That is why the prompt
    names both halves of the edit; this test is what makes the claim checkable.
    """
    rows = [
        # (date, user_id, video_id, author_id, tab, duration_ms, long_view)
        (20220408, 'u1', 'v1', 'a1', '1', 1000.0, 1),
        (20220409, 'u1', 'v2', 'a2', '1', 2000.0, 0),
        (20220410, 'u2', 'v1', 'a1', '0', 3000.0, 1),
    ]
    splits = {'train': rows, 'valid': rows[:1]}
    enc_before, dim_before = data.encode(splits)
    assert enc_before['train'][0].shape == (3, len(data.FIELDS))

    original = list(data.FIELDS)
    try:
        data.FIELDS.append('prev_author_id')
        enc_after, dim_after = data.encode(splits)
        # Width follows FIELDS: the append is what creates the slot.
        assert enc_after['train'][0].shape == (3, len(original) + 1)
        # And exactly +1, not more: the new field's vocabulary stayed EMPTY
        # because raw(x) was not extended, so it contributed only its UNK slot.
        # An extended raw(x) would have added one entry per distinct value.
        assert dim_after == dim_before + 1, (
            "a field with no raw(x) value must contribute only a UNK slot; "
            "if this grew further, encode() changed and the prompt's "
            "description of the edit needs updating")
    finally:
        data.FIELDS[:] = original
    # The module is left exactly as it was, so test order cannot matter.
    assert data.FIELDS == original


def test_raw_returns_exactly_one_value_per_field_today():
    """The invariant the two-part edit has to preserve. Pinned so a candidate
    that appends to FIELDS without extending raw(x) is a visible bug rather
    than a silently uninitialized column."""
    rows = [(20220408, 'u1', 'v1', 'a1', '1', 1000.0, 1)]
    enc, _ = data.encode({'train': rows})
    X = enc['train'][0]
    assert X.shape[1] == len(data.FIELDS)
    # Every column was actually written: with one row and one distinct value per
    # field, each entry equals that field's offset (vocab index 0 + offset).
    assert (X[0] >= 0).all() and (X[0] < 10).all(), \
        f"a column looks uninitialized: {X[0]}"
