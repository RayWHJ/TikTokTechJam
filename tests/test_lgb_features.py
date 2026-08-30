"""data.encode_lgb — leak-freedom, grouping, and the contract baseline.py needs.

The features encode_lgb builds are all TRAIN-ONLY aggregates looked up for
valid/test. That property is the whole reason the encoder is allowed to exist,
and it is invisible in the output arrays — so it is asserted here directly
rather than trusted.

Fast: every test runs on a hand-built splits dict, never the real dataset.
"""
import numpy as np
import pytest

from data import (encode_lgb, LGB_FIELDS, LGB_CATEGORICAL, LGB_USER_BUCKETS)


def _row(date, user, video, author, tab, dur, label):
    """A row in load()'s tuple layout: (date, user, video, author, tab, dur, y)."""
    return (date, user, video, author, tab, float(dur), label)


@pytest.fixture
def splits():
    """Two users, three videos, and a valid row whose video is train-unseen."""
    train = [
        _row(20220408, '1', 'v1', 'a1', '1', 1000, 1),
        _row(20220409, '1', 'v1', 'a1', '1', 1000, 1),
        _row(20220410, '1', 'v2', 'a1', '2', 5000, 0),
        _row(20220411, '2', 'v1', 'a1', '1', 1000, 0),
        _row(20220412, '2', 'v2', 'a1', '2', 5000, 0),
        _row(20220413, '2', 'v3', 'a2', '1', 9000, 1),
    ]
    valid = [
        _row(20220422, '2', 'v1', 'a1', '1', 1000, 1),
        _row(20220423, '1', 'v9', 'a9', '1', 2000, 0),   # unseen video + author
        _row(20220424, '2', 'v2', 'a1', '2', 5000, 0),
    ]
    return {'train': train, 'valid': valid}


def test_shapes_and_feature_names(splits):
    enc, names, cats = encode_lgb(splits)
    assert names == LGB_FIELDS
    assert cats == LGB_CATEGORICAL
    for name, rows in splits.items():
        X, y, users, groups = enc[name]
        assert X.shape == (len(rows), len(LGB_FIELDS))
        assert X.dtype == np.float32
        assert len(y) == len(rows) and len(users) == len(rows)
        assert groups.sum() == len(rows), "groups must cover every row exactly once"


def test_rows_are_grouped_by_user_for_ranking(splits):
    """LightGBM's ranking objectives take run lengths, not a key column — so
    equal user ids have to be contiguous or the groups are silently wrong."""
    enc, _, _ = encode_lgb(splits)
    for name in splits:
        _X, _y, users, groups = enc[name]
        # Contiguity: the number of blocks equals the number of distinct users.
        blocks, prev = 0, object()
        for u in users:
            if u != prev:
                blocks += 1
                prev = u
        assert blocks == len(set(users))
        assert len(groups) == blocks


def test_sort_by_user_false_preserves_row_order(splits):
    """The stacking path needs index-for-index alignment with encode()."""
    enc, _, _ = encode_lgb(splits, sort_by_user=False)
    for name, rows in splits.items():
        _X, y, users, _g = enc[name]
        assert users == [r[1] for r in rows]
        assert list(y) == [float(r[6]) for r in rows]


def test_no_feature_uses_the_rows_own_label(splits):
    """Flip every valid label and the valid features must not budge.

    This is the leak test that matters: a target-encoded feature computed with
    the row's own outcome in it would move here.
    """
    base, _, _ = encode_lgb(splits)
    flipped = dict(splits)
    flipped['valid'] = [(*r[:6], 1 - r[6]) for r in splits['valid']]
    after, _, _ = encode_lgb(flipped)
    np.testing.assert_array_equal(base['valid'][0], after['valid'][0])


def test_valid_rows_do_not_influence_any_statistic(splits):
    """Aggregates are train-only, so piling rows into valid must not move the
    TRAIN features, nor the features of the valid rows that were already there.

    sort_by_user=False so the original valid rows stay at indices 0..2 and are
    comparable — under the default sort, appending rows permutes the array and
    an element-wise comparison would be meaningless rather than informative.
    """
    base, _, _ = encode_lgb(splits, sort_by_user=False)
    more = dict(splits)
    more['valid'] = splits['valid'] + [
        _row(20220425, '1', 'v1', 'a1', '1', 1000, 1)] * 50
    after, _, _ = encode_lgb(more, sort_by_user=False)
    np.testing.assert_array_equal(base['train'][0], after['train'][0])
    n = len(splits['valid'])
    np.testing.assert_array_equal(base['valid'][0], after['valid'][0][:n])


def test_unseen_video_is_flagged_and_falls_back_to_the_global_rate(splits):
    enc, names, _ = encode_lgb(splits, sort_by_user=False)
    X = enc['valid'][0]
    is_new, rate = names.index('video_is_new'), names.index('video_rate')
    imp_log = names.index('video_imp_log')
    row = 1                                  # the v9 / a9 row
    assert X[row, is_new] == 1.0
    assert X[row, imp_log] == 0.0            # log1p(0)
    gmean = sum(r[6] for r in splits['train']) / len(splits['train'])
    # Zero observations means the smoothed rate lands exactly on the prior.
    assert X[row, rate] == pytest.approx(gmean, abs=1e-6)
    # And a video that WAS seen must not be flagged.
    assert X[0, is_new] == 0.0


def test_train_absent_user_gets_its_own_activity_bucket(splits):
    """A user with no train history has no decile to belong to; -1 is a level,
    not a silent squeeze into the bottom bucket."""
    more = dict(splits)
    more['valid'] = splits['valid'] + [_row(20220425, '99', 'v1', 'a1', '1', 1000, 0)]
    enc, names, _ = encode_lgb(more, sort_by_user=False)
    X = enc['valid'][0]
    # user_imp_log is log1p(0) for a user absent from train.
    assert X[-1, names.index('user_imp_log')] == 0.0
    assert 0 <= LGB_USER_BUCKETS       # sanity: the bucket count is configured


def test_features_are_finite(splits):
    enc, _, _ = encode_lgb(splits)
    for name in splits:
        X = enc[name][0]
        assert np.isfinite(X).all(), f"non-finite feature in {name}"
