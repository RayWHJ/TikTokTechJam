"""Golden fixture tests for harness.validated_evaluate and harness.check_provenance.

These use small hand-built arrays, not the real dataset, except where noted.
"""
import subprocess
import sys

import numpy as np
import pytest

import harness


# ---------------------------------------------------------------------------
# validated_evaluate: correctness on small fixtures (split_name row-count check
# bypassed by monkeypatching get_encoded's cached sizes to match our fixture).
# ---------------------------------------------------------------------------

@pytest.fixture
def bypass_size_check(monkeypatch):
    """Make the row-count check accept any length, for tests using tiny fixtures
    that aren't meant to line up with the real dataset's split sizes."""
    import harness._evaluate as ev_mod

    monkeypatch.setattr(ev_mod, 'get_split_sizes', lambda data_dir=None: _AnyLen())
    yield


class _AnyLen(dict):
    def __getitem__(self, key):
        return _EqualsAnything()


class _EqualsAnything:
    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


def test_tied_scores_auc(bypass_size_check):
    # 2 users, scores tied within a user -> AUC falls back to the tie-corrected rank formula.
    user_ids = ['u1', 'u1', 'u1', 'u1']
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    r = harness.validated_evaluate(user_ids, labels, scores, 'valid_search')
    assert r['GAUC'] == pytest.approx(0.5)
    assert r['users'] == 1
    assert r['rows'] == 4


def test_all_positive_user(bypass_size_check):
    user_ids = ['u1', 'u1', 'u1']
    labels = [1, 1, 1]
    scores = [0.1, 0.9, 0.5]
    r = harness.validated_evaluate(user_ids, labels, scores, 'valid_search')
    # all-positive user contributes nDCG=1.0, is excluded from GAUC (npos == len(labs))
    assert r['nDCG@5'] == pytest.approx(1.0)
    assert r['GAUC'] == pytest.approx(0.5)  # gden == 0 -> default 0.5


def test_all_negative_user(bypass_size_check):
    user_ids = ['u1', 'u1', 'u1']
    labels = [0, 0, 0]
    scores = [0.1, 0.9, 0.5]
    r = harness.validated_evaluate(user_ids, labels, scores, 'valid_search')
    assert r['nDCG@5'] == pytest.approx(0.0)
    assert r['GAUC'] == pytest.approx(0.5)


def test_mismatched_length_raises(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u2'], [0, 1, 0], [0.1, 0.2], 'valid_search')


def test_nan_score_raises(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u1'], [0, 1], [0.1, float('nan')], 'valid_search')


def test_inf_score_raises(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u1'], [0, 1], [0.1, float('inf')], 'valid_search')


def test_non_binary_label_raises(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u1'], [0, 2], [0.1, 0.2], 'valid_search')


def test_empty_input_raises(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate([], [], [], 'valid_search')


def test_nan_label_raises(bypass_size_check):
    with pytest.raises(ValueError, match='non-binary'):
        harness.validated_evaluate(['u1', 'u1'], [0, float('nan')], [0.1, 0.2], 'valid_search')


def test_non_numeric_score_raises_valueerror_not_typeerror(bypass_size_check):
    # numpy would raise TypeError here; the contract promises ValueError.
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u1'], [0, 1], [0.1, {'a': 1}], 'valid_search')


def test_non_numeric_label_raises_valueerror(bypass_size_check):
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1', 'u1'], [0, 'yes'], [0.1, 0.2], 'valid_search')


def test_null_user_id_raises(bypass_size_check):
    with pytest.raises(ValueError, match='user_ids'):
        harness.validated_evaluate(['u1', None], [0, 1], [0.1, 0.2], 'valid_search')


def test_bool_labels_accepted(bypass_size_check):
    # bools are a legitimate 0/1 encoding and must not be rejected.
    r = harness.validated_evaluate(['u1', 'u1'], [False, True], [0.1, 0.9], 'valid_search')
    assert r['GAUC'] == pytest.approx(1.0)


def test_numpy_inputs_accepted(bypass_size_check):
    # data.py/baseline.py hand back float32 arrays; those must pass unchanged.
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    s = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    r = harness.validated_evaluate(['u1'] * 4, y, s, 'valid_search')
    assert r['GAUC'] == pytest.approx(1.0)
    assert r['rows'] == 4


def test_2d_scores_raises(bypass_size_check):
    with pytest.raises(ValueError, match='1-D'):
        harness.validated_evaluate(['u1', 'u1'], [0, 1], [[0.1], [0.2]], 'valid_search')


def test_primary_is_mean_of_both_metrics(bypass_size_check):
    r = harness.validated_evaluate(
        ['u1', 'u1', 'u2', 'u2'], [0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8], 'valid_search'
    )
    assert r['primary'] == pytest.approx((r['GAUC'] + r['nDCG@5']) / 2.0)


def test_unknown_split_name_raises():
    with pytest.raises(ValueError):
        harness.validated_evaluate(['u1'], [1], [0.1], 'not_a_real_split')


def test_row_count_mismatch_raises_against_real_split():
    # Real dataset must be present under ./KuaiRand-Pure/data for this one.
    with pytest.raises(ValueError, match='row count'):
        harness.validated_evaluate(['u1'], [1], [0.1], 'valid_confirm')


# ---------------------------------------------------------------------------
# check_provenance
# ---------------------------------------------------------------------------

def test_check_provenance_blocks_non_causal_column():
    with pytest.raises(ValueError):
        harness.check_provenance(['user_id', 'like_cnt'])


def test_check_provenance_allows_causal_columns():
    harness.check_provenance(['user_id', 'video_id', 'tab'])  # no raise


def test_check_provenance_point_in_time_override():
    harness.check_provenance(['like_cnt'], point_in_time=True)  # no raise


# ---------------------------------------------------------------------------
# get_split: one-shot test gate (run in a subprocess to isolate from other tests
# touching the same module-level flag).
# ---------------------------------------------------------------------------

def test_test_split_is_one_shot():
    code = (
        "import harness\n"
        "harness.get_split('test')\n"
        "try:\n"
        "    harness.get_split('test')\n"
        "    raise SystemExit(1)\n"
        "except RuntimeError:\n"
        "    raise SystemExit(0)\n"
    )
    result = subprocess.run([sys.executable, '-c', code], cwd='.', timeout=120)
    assert result.returncode == 0


def test_get_split_unknown_name_raises():
    with pytest.raises(ValueError):
        harness.get_split('valid')  # not a recognized split name in this contract
