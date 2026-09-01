"""Tests for train_fit / train_es — the split that decouples early stopping
from scoring.

The failure being guarded: run_fm early-stops on `select_on` and reports on
`report_on`, and run_for_orchestrator passed valid_search for BOTH. The stopping
epoch is chosen by maximising primary on those exact rows (patience=4,
epochs=40), so a candidate that trains more epochs collects more draws at the
maximum on the split it is scored on. It scores higher for a reason that has
nothing to do with its mechanism, and the paired bootstrap cannot see it because
it compares only final per-user scores. Against a 0.0012 paired noise floor and
a largest-ever measured candidate delta of +0.0006 (iteration 1 of the run in
orchestrator/_state/progress.json), that bias is a large fraction of the signal.

The range checks here need no CSVs, so they run in the default suite. The
row-level partition check pays a real data.load() and is marked slow, matching
tests/test_splits.py.
"""
import inspect

import pytest

import baseline
from data import SPLITS, TRAIN_SUBSPLITS, cut_train_subsplits


def _dates(lo_hi):
    lo, hi = lo_hi
    return set(range(lo, hi + 1))


# --------------------------------------------------------------------------- #
#  Range arithmetic                                                            #
# --------------------------------------------------------------------------- #
def test_train_fit_and_train_es_are_disjoint():
    assert not _dates(TRAIN_SUBSPLITS['train_fit']) & \
               _dates(TRAIN_SUBSPLITS['train_es'])


def test_union_is_exactly_the_official_train_range():
    """'train' must keep its full range: encode() builds its vocabularies and
    _bucket_edges from splits['train'], and building them from train_fit alone
    would push two days' worth of ids into each field's UNK slot."""
    union = (_dates(TRAIN_SUBSPLITS['train_fit'])
             | _dates(TRAIN_SUBSPLITS['train_es']))
    assert union == _dates(SPLITS['train'])


def test_the_two_train_tiers_are_contiguous():
    lo_fit, hi_fit = TRAIN_SUBSPLITS['train_fit']
    lo_es, hi_es = TRAIN_SUBSPLITS['train_es']
    assert lo_fit == SPLITS['train'][0]
    assert hi_es == SPLITS['train'][1]
    assert lo_es == hi_fit + 1, 'no gap and no overlap'


@pytest.mark.parametrize("tier", ["train_fit", "train_es"])
def test_no_train_tier_shares_a_date_with_any_validation_tier(tier):
    """The whole point is three DISJOINT date ranges: fit, stop, score. A single
    shared date would put the stopping signal back on the scored rows."""
    tier_dates = _dates(TRAIN_SUBSPLITS[tier])
    for name, rng in baseline.VALID_SUBSPLITS.items():
        assert not tier_dates & _dates(rng), f'{tier} overlaps {name}'
    for name in ('valid', 'test'):
        assert not tier_dates & _dates(SPLITS[name]), f'{tier} overlaps {name}'


def test_train_es_is_the_two_days_immediately_before_valid():
    """train_es has to be temporally adjacent to the fitted rows to be a fair
    stopping signal for a split that comes later still."""
    assert TRAIN_SUBSPLITS['train_es'][1] + 1 == SPLITS['valid'][0]


# --------------------------------------------------------------------------- #
#  Wiring                                                                      #
# --------------------------------------------------------------------------- #
def test_run_fm_interactive_defaults_are_unchanged():
    """`python3 baseline.py --model fm` must keep training on all 14 days, or
    the ladder in README.md and baseline_scores.json stops reproducing. This is
    the guard that lets the orchestrator path change on its own."""
    sig = inspect.signature(baseline.run_fm)
    assert sig.parameters['train_on'].default == 'train'
    assert sig.parameters['select_on'].default == 'valid'
    assert sig.parameters['report_on'].default == ('valid', 'test')


def test_orchestrator_path_fits_and_stops_on_different_splits():
    """Reads the source of run_for_orchestrator rather than running it, so the
    assertion holds without a 20s FM train. What matters is that the fm branch
    never passes the reported split as select_on again."""
    src = inspect.getsource(baseline.run_for_orchestrator)
    assert "'train_on': 'train_fit'" in src
    assert "'select_on': 'train_es'" in src
    assert "cut_train_subsplits" in src


# --------------------------------------------------------------------------- #
#  Row-level partition, against the real dataset                               #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_cut_train_subsplits_partitions_train_row_for_row():
    from data import load
    from harness._data import DEFAULT_DATA_DIR

    splits = cut_train_subsplits(load(DEFAULT_DATA_DIR))
    fit, es = splits['train_fit'], splits['train_es']

    assert len(fit) + len(es) == len(splits['train']), \
        'train_fit + train_es must account for every train row'
    assert fit and es, 'neither tier may be empty'
    # 'train' itself must survive untouched — encode() reads it.
    assert len(splits['train']) == len(load(DEFAULT_DATA_DIR)['train'])

    for name in ('train_fit', 'train_es'):
        lo, hi = TRAIN_SUBSPLITS[name]
        dates = {x[0] for x in splits[name]}
        assert min(dates) >= lo and max(dates) <= hi, name
