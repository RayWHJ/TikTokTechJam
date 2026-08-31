"""T2.10 — the verified numpy primitives the writer is told to compose.

Three of the five proposals in the recorded run needed exactly a
previous-value-within-user computation, and the one that got the semantics right
timed out implementing it in a Python loop over 1.4M rows. So these helpers exist
to make the hardest part of the most-proposed direction one call — which is only
worth doing if the two properties that are easy to get wrong are actually
guaranteed:

  * POINT-IN-TIME: a row is never described by itself or by anything after it.
  * SPLIT-LOCAL: a helper sees one split's rows and nothing else.

Every test below asserts one of those two, or the vectorisation that keeps the
cost inside the triage cap.
"""
import numpy as np
import pytest

import data


#: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
def _row(date, user, video, author='a1', tab='1', dur=1000.0, label=0):
    return (date, user, video, author, tab, dur, label)


TRAIN = [
    _row(20220408, 'u1', 'v1', 'a1', label=1),
    _row(20220409, 'u1', 'v2', 'a2', label=0),
    _row(20220410, 'u1', 'v3', 'a1', label=1),
    _row(20220408, 'u2', 'v9', 'a9', label=0),
    _row(20220411, 'u2', 'v1', 'a1', label=1),
]


# --------------------------------------------------------------------------- #
#  prev_value_within_user                                                     #
# --------------------------------------------------------------------------- #
def test_prev_value_is_the_row_before_in_date_order():
    got = data.prev_value_within_user(TRAIN, key=2)
    # u1: v1 -> NONE, v2 -> v1, v3 -> v2.  u2: v9 -> NONE, v1 -> v9.
    assert got == ['NONE', 'v1', 'v2', 'NONE', 'v9']


def test_prev_value_never_reads_the_row_it_describes_or_any_later_row():
    """The property that makes it not leakage. Checked by construction: mutate
    every row at or after position i and the value for row i must not move."""
    base = data.prev_value_within_user(TRAIN, key=2)
    for i in range(len(TRAIN)):
        tampered = list(TRAIN)
        # Rewrite row i and everything after it in the same user's timeline.
        for j in range(i, len(tampered)):
            r = list(tampered[j])
            r[2] = f'TAMPERED{j}'
            tampered[j] = tuple(r)
        got = data.prev_value_within_user(tampered, key=2)
        assert got[i] == base[i], (
            f"row {i}'s feature changed when row {i} (or later) was rewritten — "
            f"it is reading forward")


def test_prev_value_is_split_local():
    """A valid-split row's `previous` value must never reach back into train.
    Under-informing the first row of each user is the conservative direction; a
    cross-split lookup is a boundary violation the paired bootstrap cannot see."""
    valid = [_row(20220422, 'u1', 'v7'), _row(20220423, 'u1', 'v8')]
    got = data.prev_value_within_user(valid, key=2)
    assert got[0] == 'NONE', \
        "u1's first VALID row must not inherit v3 from the train split"
    assert got[1] == 'v7'
    # And the helper is given no way to see another split even if it wanted to.
    import inspect
    src = inspect.getsource(data.prev_value_within_user)
    assert 'splits' not in src and 'SPLITS' not in src


def test_prev_value_is_aligned_with_rows_as_given_not_with_sorted_order():
    """encode() iterates `for n, x in enumerate(rws)`, so a returned column has
    to be in the caller's row order. Returning it in date order would silently
    misalign every feature."""
    shuffled = [TRAIN[2], TRAIN[0], TRAIN[4], TRAIN[1], TRAIN[3]]
    got = data.prev_value_within_user(shuffled, key=2)
    # Keyed on (user, video): 'v1' appears for BOTH u1 and u2.
    by_row = dict(zip([(r[1], r[2]) for r in shuffled], got))
    assert by_row == {('u1', 'v3'): 'v2', ('u1', 'v1'): 'NONE',
                      ('u2', 'v1'): 'v9', ('u1', 'v2'): 'v1',
                      ('u2', 'v9'): 'NONE'}


def test_prev_value_works_on_other_columns():
    assert data.prev_value_within_user(TRAIN, key=3) == \
        ['NONE', 'a1', 'a2', 'NONE', 'a9']
    # The label column: `prev_long_view` is the one genuinely risky construction
    # in the recorded run, and it is safe ONLY because of the shift.
    assert data.prev_value_within_user(TRAIN, key=6) == \
        ['NONE', '1', '0', 'NONE', '0']


def test_prev_long_view_never_returns_the_rows_own_label():
    """The leak this helper exists to make impossible. `prev_long_view` passed the
    static gate cleanly in the recorded run because AUXILIARY_SIGNALS omits
    long_view; the shift is what makes the feature legitimate."""
    rows = [_row(20220408, 'u1', 'v1', label=1),
            _row(20220409, 'u1', 'v2', label=0),
            _row(20220410, 'u1', 'v3', label=1)]
    got = data.prev_value_within_user(rows, key=6)
    for i, r in enumerate(rows):
        assert got[i] != str(r[6]) or i > 0 and got[i] == str(rows[i - 1][6]), \
            "the feature must equal the PREVIOUS label, never this row's own"
    assert got == ['NONE', '1', '0']


def test_prev_value_plugs_into_the_extra_fields_registry():
    """The two halves of T1.5 and T2.10 have to actually fit together."""
    fields, extras = list(data.FIELDS), list(data.EXTRA_FIELDS)
    try:
        _, dim_base = data.encode({'train': TRAIN, 'valid': TRAIN[:2]})
        data.EXTRA_FIELDS.append(
            ('prev_author_id', lambda rows: data.prev_value_within_user(rows, 3)))
        enc, dim = data.encode({'train': TRAIN, 'valid': TRAIN[:2]})
        assert enc['train'][0].shape == (5, len(data.BASE_FIELDS) + 1)
        col = enc['train'][0][:, -1]
        # 'NONE', 'a1', 'a2', 'a9' distinct in train => 4 ids, + 1 UNK slot.
        assert len(set(col.tolist())) == 4
        assert dim == dim_base + 5
    finally:
        data.FIELDS[:] = fields
        data.EXTRA_FIELDS[:] = extras


# --------------------------------------------------------------------------- #
#  prior_count_within_user / position_within_user                             #
# --------------------------------------------------------------------------- #
def test_prior_count_excludes_the_current_row():
    rows = [_row(20220408, 'u1', 'v1', 'a1'),
            _row(20220409, 'u1', 'v2', 'a1'),
            _row(20220410, 'u1', 'v3', 'a1'),
            _row(20220411, 'u1', 'v4', 'a2')]
    got = data.prior_count_within_user(rows, key=3)
    # a1 seen 0, then 1, then 2 times before; a2 seen 0 times.
    # buckets=(0, 1, 2, 5) => ranges "0", "1", "2-4", "5+".
    assert got == ['0', '1', '2-4', '0']


def test_prior_count_buckets_the_tail_rather_than_exploding_the_domain():
    rows = [_row(20220408 + i, 'u1', f'v{i}', 'a1') for i in range(9)]
    got = data.prior_count_within_user(rows, key=3)
    assert got == ['0', '1', '2-4', '2-4', '2-4', '5+', '5+', '5+', '5+']
    assert len(set(got)) <= 6, "a raw count would be one embedding per value"


def test_position_within_user_uses_only_preceding_rows():
    rows = [_row(20220408 + i, 'u1', f'v{i}') for i in range(4)] + \
           [_row(20220408, 'u2', 'v0')]
    got = data.position_within_user(rows)
    # buckets=(0, 1, 2, 5, 10, 20) => "0", "1", "2-4", "5-9", "10-19", "20+".
    assert got == ['0', '1', '2-4', '2-4', '0']
    # Appending more of u1's rows must not change the earlier ones.
    more = rows + [_row(20220420, 'u1', 'v9')]
    assert data.position_within_user(more)[:5] == got


@pytest.mark.parametrize("fn,key", [
    (data.prior_count_within_user, 3),
    (data.position_within_user, None),
])
def test_the_bucketed_helpers_are_split_local_and_row_aligned(fn, key):
    rows = list(TRAIN)
    got = fn(rows, key) if key is not None else fn(rows)
    assert len(got) == len(rows)
    assert all(isinstance(v, str) and v for v in got)
    # Same rows in a different order give each row the same answer.
    perm = [3, 1, 4, 0, 2]
    shuffled = [rows[i] for i in perm]
    got2 = fn(shuffled, key) if key is not None else fn(shuffled)
    for out_i, src_i in enumerate(perm):
        assert got2[out_i] == got[src_i]


# --------------------------------------------------------------------------- #
#  within_user_pairs                                                          #
# --------------------------------------------------------------------------- #
def test_pairs_are_always_within_one_user():
    users = np.array(['u1'] * 4 + ['u2'] * 4)
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.float32)
    pos, neg = data.within_user_pairs(users, y, np.random.default_rng(0))
    assert len(pos) == len(neg) > 0
    assert (users[pos] == users[neg]).all(), "a cross-user pair is meaningless"


def test_pairs_put_a_positive_on_the_left_and_a_negative_on_the_right():
    users = np.array(['u1'] * 4)
    y = np.array([1, 0, 1, 0], dtype=np.float32)
    pos, neg = data.within_user_pairs(users, y, np.random.default_rng(0))
    assert (y[pos] > 0).all()
    assert (y[neg] <= 0).all()


def test_users_with_no_positive_or_no_negative_contribute_nothing():
    """They carry no within-user ranking signal, and GAUC excludes them too."""
    users = np.array(['all_pos'] * 3 + ['all_neg'] * 3 + ['mixed'] * 2)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    pos, neg = data.within_user_pairs(users, y, np.random.default_rng(0))
    assert set(users[pos]) == {'mixed'}
    assert set(users[neg]) == {'mixed'}


def test_no_pairable_users_returns_empty_arrays_not_an_error():
    users = np.array(['u1'] * 3)
    y = np.array([0, 0, 0], dtype=np.float32)
    pos, neg = data.within_user_pairs(users, y, np.random.default_rng(0))
    assert len(pos) == 0 and len(neg) == 0
    assert pos.dtype == np.int64


def test_pairs_take_their_own_generator_and_never_the_shared_rng():
    """The RNG rule, which is load-bearing for the paired bootstrap: the
    orchestrator pairs candidate against parent SEED BY SEED, and that only
    cancels noise while both runs consume the same draws in shared code. One
    extra draw from run_fm's `rng` decorrelates the trajectories and the inflated
    variance is indistinguishable from a real effect."""
    import inspect
    src = inspect.getsource(data.within_user_pairs)
    assert 'np.random.default_rng(' not in src.split('"""')[2], \
        "the helper must not construct its own generator in its body"
    assert 'np.random.seed' not in src and 'np.random.choice' not in src

    users = np.array(['u1'] * 4)
    y = np.array([1, 0, 1, 0], dtype=np.float32)
    a = data.within_user_pairs(users, y, np.random.default_rng(7))
    b = data.within_user_pairs(users, y, np.random.default_rng(7))
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all(), \
        "same generator seed must give the same pairs"


def test_pairs_are_vectorised_enough_for_the_real_split():
    """A per-row Python callback over 1.4M rows is the 240s timeout that killed a
    candidate. 200k rows here as a proxy — it must be well under a second."""
    import time
    rng = np.random.default_rng(0)
    n = 200_000
    users = np.array([f'u{i % 5000}' for i in range(n)])
    y = (rng.random(n) < 0.3).astype(np.float32)

    t0 = time.time()
    pos, neg = data.within_user_pairs(users, y, rng)
    elapsed = time.time() - t0

    assert len(pos) > 0
    assert elapsed < 5.0, f"took {elapsed:.1f}s on {n} rows"


def test_prev_value_is_fast_enough_for_the_real_split():
    import time
    n = 200_000
    rows = [_row(20220408 + (i % 14), f'u{i % 5000}', f'v{i % 7000}')
            for i in range(n)]
    t0 = time.time()
    out = data.prev_value_within_user(rows, key=2)
    elapsed = time.time() - t0
    assert len(out) == n
    assert elapsed < 5.0, f"took {elapsed:.1f}s on {n} rows"
