"""T1.5 — data.py's categorical-field registry is a genuine single point of change.

The claim under test: appending one entry to `EXTRA_FIELDS` widens X by exactly
one column, gives the new field its own train-only vocabulary and UNK slot, and
requires NO other edit. Before the registry existed, adding a categorical field
was a coupled four-place edit — FIELDS, raw(), the vocab list and the X
allocation — with no single point of change, and two candidates in the recorded
run died with the identical `IndexError` inside
`for i, v in enumerate(raw(x))` having got that coupling wrong.

`tests/test_direction_reachability.py` covers the other half: that the prompts do
not lie about any of this.
"""
import numpy as np
import pytest

import data


#: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
ROWS = [
    (20220408, 'u1', 'v1', 'a1', '1', 1000.0, 1),
    (20220409, 'u1', 'v2', 'a2', '1', 2000.0, 0),
    (20220410, 'u1', 'v3', 'a1', '0', 3000.0, 1),
    (20220411, 'u2', 'v1', 'a1', '0', 4000.0, 0),
]


@pytest.fixture
def registry_restored():
    """Leave data.py exactly as it was, so test order cannot matter."""
    fields, extras = list(data.FIELDS), list(data.EXTRA_FIELDS)
    yield
    data.FIELDS[:] = fields
    data.EXTRA_FIELDS[:] = extras


def _prev_video_within_user(rows):
    """The user's PREVIOUS video_id, in row order. Point-in-time by construction
    — it only ever reads rows strictly before the one it describes — and
    split-local, because it only ever sees one split's rows."""
    last, out = {}, []
    for x in rows:
        out.append(last.get(x[1], 'NONE'))
        last[x[1]] = x[2]
    return out


# --------------------------------------------------------------------------- #
#  The single append                                                          #
# --------------------------------------------------------------------------- #
def test_appending_to_extra_fields_widens_x_by_exactly_one_column(registry_restored):
    """The acceptance criterion, stated literally: one append, one column."""
    splits = {'train': ROWS, 'valid': ROWS[:2]}
    enc_before, dim_before = data.encode(splits)
    w_before = enc_before['train'][0].shape[1]
    assert w_before == len(data.BASE_FIELDS)

    data.EXTRA_FIELDS.append(('prev_video_id', _prev_video_within_user))

    enc_after, dim_after = data.encode(splits)
    X = enc_after['train'][0]
    assert X.shape == (len(ROWS), w_before + 1), \
        "the append alone must create the column"
    assert enc_after['valid'][0].shape == (2, w_before + 1), \
        "every split widens, not just train"

    # Derived, not hand-maintained: the new field got its own train-only
    # vocabulary ('NONE', 'v1', 'v2') plus one UNK slot.
    assert dim_after == dim_before + 4, \
        f"expected 3 distinct values + 1 UNK slot, got {dim_after - dim_before}"

    # And FIELDS reports the truth to anything that reads it (baseline.py
    # imports it; the prompts quote it).
    assert data.active_fields() == data.BASE_FIELDS + ['prev_video_id']


def test_the_new_column_is_actually_written_and_not_uninitialized(registry_restored):
    """X's new column must hold the encoded lag value, inside the field's own
    id range — not np.empty garbage, and not another field's ids."""
    data.EXTRA_FIELDS.append(('prev_video_id', _prev_video_within_user))
    enc, total_dim = data.encode({'train': ROWS})
    X = enc['train'][0]
    col = X[:, -1]

    assert (col >= 0).all() and (col < total_dim).all()
    # u1's first row has no predecessor and u2's only row has none either, so
    # those two share the 'NONE' id; u1's second row's predecessor is v1.
    assert col[0] == col[3], "both first-impression rows encode 'NONE'"
    assert col[1] != col[0], "a row with a real predecessor differs from 'NONE'"
    # Distinct from every base field's id range: offsets keep the domains apart.
    base_max = int(X[:, :len(data.BASE_FIELDS)].max())
    assert int(col.min()) > base_max, \
        "the new field must occupy its own offset block, not overlap a base field"


def test_add_categorical_field_keeps_fields_and_extra_fields_in_sync(
        registry_restored):
    data.add_categorical_field('prev_video_id', _prev_video_within_user)
    assert 'prev_video_id' in data.FIELDS
    assert 'prev_video_id' in [n for n, _ in data.EXTRA_FIELDS]
    # Registering the same name twice is a mistake, not a silent duplicate
    # column, because a duplicate would double-count the domain in field_dims.
    with pytest.raises(ValueError):
        data.add_categorical_field('prev_video_id', _prev_video_within_user)
    enc, _ = data.encode({'train': ROWS})
    assert enc['train'][0].shape[1] == len(data.BASE_FIELDS) + 1


def test_an_unseen_value_lands_in_the_new_fields_unk_slot(registry_restored):
    """The vocabulary stays TRAIN-ONLY, so a valid-split value train never saw
    must not silently collide with a train id."""
    data.EXTRA_FIELDS.append(('prev_video_id', _prev_video_within_user))
    valid = [(20220422, 'u9', 'v9', 'a9', '1', 1500.0, 1),
             (20220423, 'u9', 'v8', 'a9', '1', 1500.0, 0)]
    enc, _ = data.encode({'train': ROWS, 'valid': valid})

    train_ids = set(enc['train'][0][:, -1].tolist())
    # u9's second row has predecessor 'v9', which train never saw.
    unseen = int(enc['valid'][0][1, -1])
    assert unseen not in train_ids, \
        "an unseen lag value must land in the UNK slot, not on a train id"


# --------------------------------------------------------------------------- #
#  The crash class the seam exists to remove                                  #
# --------------------------------------------------------------------------- #
def test_raw_returning_more_values_than_fields_fails_loudly(registry_restored,
                                                            monkeypatch):
    """The exact failure that killed two candidates, now a named error naming
    both lengths and both lists instead of a bare IndexError on a numpy index.

    Simulated by shrinking `fields` rather than by rewriting encode's inner
    raw(), which is what a candidate's bad edit does to the same effect.
    """
    monkeypatch.setattr(data, 'FIELDS', data.BASE_FIELDS[:-1])
    with pytest.raises(AssertionError) as e:
        data.encode({'train': ROWS})
    msg = str(e.value)
    assert 'raw(x)' in msg and 'fields' in msg
    assert '(5)' in msg and '(4)' in msg, "both lengths must be in the message"
    # Both LISTS, so the reader can see which value has no field to go in.
    assert "['user_id', 'video_id', 'author_id', 'tab']" in msg, \
        "the field list must be in the message"
    assert "['u1', 'v1', 'a1', '1', '0']" in msg, \
        "raw(x)'s own values must be in the message"
    assert 'EXTRA_FIELDS' in msg, "the message must name the supported seam"


def test_an_extra_field_returning_the_wrong_row_count_fails_loudly(
        registry_restored):
    """A misaligned column would silently describe row n with row m's feature,
    which is a leak in one direction and noise in the other."""
    data.EXTRA_FIELDS.append(('broken', lambda rows: ['x'] * (len(rows) - 1)))
    with pytest.raises(AssertionError) as e:
        data.encode({'train': ROWS})
    assert 'one value per row' in str(e.value)


# --------------------------------------------------------------------------- #
#  Nothing changed for the default configuration                              #
# --------------------------------------------------------------------------- #
def test_the_registry_is_empty_and_fields_unchanged_at_import():
    """The prompts quote FIELDS verbatim and baseline.py imports it, so the
    default configuration has to be byte-identical to the 5-field baseline."""
    assert data.EXTRA_FIELDS == []
    assert data.FIELDS == ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
    assert data.FIELDS == data.BASE_FIELDS
    assert data.FIELDS is not data.BASE_FIELDS, \
        "FIELDS must be its own list, or mutating it would edit BASE_FIELDS"


def test_encode_is_unchanged_with_no_extra_fields():
    """Same ids, same field_dims as the pre-registry encode()."""
    enc, total = data.encode({'train': ROWS, 'valid': ROWS[:1]})
    X, y, users = enc['train']
    assert X.shape == (4, 5) and X.dtype == np.int32
    assert users == ['u1', 'u1', 'u1', 'u2']
    assert list(y) == [1.0, 0.0, 1.0, 0.0]
    # 2 users + 3 videos + 2 authors + 2 tabs + n dur buckets, each with a UNK.
    assert total == sum(len(set(vals)) + 1 for vals in (
        ['u1', 'u1', 'u1', 'u2'],
        ['v1', 'v2', 'v3', 'v1'],
        ['a1', 'a2', 'a1', 'a1'],
        ['1', '1', '0', '0'],
        [X[n, 4] for n in range(4)]))
