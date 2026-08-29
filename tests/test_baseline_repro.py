"""Baseline reproduction: does this checkout still produce the organizer's
published numbers, and does the harness wrapper report them undistorted?

Everything here goes through baseline.py's own functions on raw data.load()
output, deliberately NOT through harness.get_split('test'). get_split('test') is
one-shot per process, and burning that budget inside a test run would be exactly
the mistake the gate exists to prevent. These tests read test labels only to
reproduce a published number, never to select anything.

All slow. Run with: pytest -m slow
The FM case alone is ~4 minutes (5 seeds x train + a re-encode per call).
"""
import json
import os

import pytest

from baseline import run_fm, run_pop, run_random
from data import load
from evaluate import evaluate
from harness._data import DEFAULT_DATA_DIR, partition

import harness

pytestmark = pytest.mark.slow

SEEDS = (0, 1, 2, 3, 4)

# baseline_scores.json reports std 0.0008 over 5 seeds. A 5-seed mean has
# standard error std/sqrt(5), and 2.5 sigma is the same multiple the repo's own
# convergence rule uses (epsilon 0.002 ~ 2.5 * 0.0008).
STD_OVER_SEEDS = 0.0008
SIGMAS = 2.5
MEAN_TOLERANCE = SIGMAS * STD_OVER_SEEDS / len(SEEDS) ** 0.5

# For the deterministic baselines there is no seed noise, so the only slack
# needed is the 4-decimal rounding in the published table.
ROUNDING_TOLERANCE = 0.001


@pytest.fixture(scope='module')
def splits():
    return load(DEFAULT_DATA_DIR)


@pytest.fixture(scope='module')
def published():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'baseline_scores.json'), encoding='utf-8') as fh:
        return json.load(fh)['scores']


# ---------------------------------------------------------------------------
# The README's first-line check: "if --model random doesn't give primary ~0.475,
# something is wrong with the harness -- fix that first."
# ---------------------------------------------------------------------------

def test_random_baseline_matches_published(splits, published):
    """Random scoring must land on the published lower bound.

    This is the cheapest possible end-to-end proof that the metric wiring is
    intact: random scores have no signal, so any deviation from 0.4753 means the
    evaluation path itself is broken, not the model. Per the README this is the
    first thing to check and everything else is meaningless until it passes.
    """
    expected = published['random']['test']['primary']
    primaries = [run_random(splits, seed=s)['test']['primary'] for s in SEEDS]
    mean = sum(primaries) / len(primaries)
    assert mean == pytest.approx(expected, abs=MEAN_TOLERANCE), (
        f'random 5-seed mean test primary {mean:.4f} != published {expected:.4f}; '
        f'the evaluation path is broken -- fix this before trusting any other number'
    )


def test_random_baseline_gauc_is_chance(splits):
    # A sharper form of the same check: GAUC on signal-free scores is 0.5 by
    # construction, independent of the dataset, so this catches a broken AUC
    # implementation even if the published table were itself wrong.
    gauc = run_random(splits, seed=0)['test']['GAUC']
    assert gauc == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Deterministic and official baselines
# ---------------------------------------------------------------------------

def test_item_popularity_matches_published(splits, published):
    # run_pop takes no seed, so this is exactly reproducible.
    res = run_pop(splits)
    for split in ('valid', 'test'):
        for metric in ('GAUC', 'nDCG@5', 'primary'):
            assert res[split][metric] == pytest.approx(
                published['item_popularity'][split][metric], abs=ROUNDING_TOLERANCE
            ), f'item_popularity {split} {metric} drifted'


def test_fm_five_seed_mean_matches_published(splits, published):
    """Compare the 5-seed MEAN against the published 5-seed mean.

    Deliberately not a single-seed comparison: with std 0.0008 a single seed can
    sit ~2 sigma off the mean and still be perfectly healthy, so a per-seed
    assertion would flake. The mean's standard error is 5x tighter than the
    per-seed spread, which is what makes this a real check rather than a loose one.
    """
    expected = published['fm_official']['test']['primary']
    primaries = []
    for seed in SEEDS:
        p = run_fm(splits, seed=seed, verbose=False)['test']['primary']
        primaries.append(p)
        print(f'  seed {seed}: test primary = {p:.4f}')

    mean = sum(primaries) / len(primaries)
    spread = max(primaries) - min(primaries)
    print(f'  5-seed mean = {mean:.4f} (published {expected:.4f}), spread = {spread:.4f}')

    assert mean == pytest.approx(expected, abs=MEAN_TOLERANCE), (
        f'FM 5-seed mean test primary {mean:.4f} != published {expected:.4f} '
        f'(tolerance {MEAN_TOLERANCE:.4f}); baseline.py or data.py has drifted'
    )


def test_fm_beats_pop_beats_random(splits):
    """The baseline ladder must stay ordered.

    Cheap structural check that survives any future re-tuning of the absolute
    numbers: whatever the exact values, FM > item popularity > random. If this
    inverts, something is wrong with the model or the metric direction, and the
    absolute-value assertions above would not necessarily catch a sign flip.
    """
    rnd = run_random(splits, seed=0)['test']['primary']
    pop = run_pop(splits)['test']['primary']
    fm = run_fm(splits, seed=0, verbose=False)['test']['primary']
    assert rnd < pop < fm, f'ladder inverted: random={rnd:.4f} pop={pop:.4f} fm={fm:.4f}'


# ---------------------------------------------------------------------------
# The harness wrapper must not distort what evaluate.py reports
# ---------------------------------------------------------------------------

def test_validated_evaluate_matches_raw_evaluate_exactly():
    """validated_evaluate is a guardrail, not a reimplementation.

    Scores a real split with item popularity and asserts the wrapper returns
    bit-identical metrics to calling evaluate.py directly. This is what rules out
    the wrapper's input coercion -- float32 to float64, labels to int -- silently
    shifting a metric in the fourth decimal, which is the decimal the entire
    promotion rule (epsilon 0.002) turns on.
    """
    import collections

    parts = partition(DEFAULT_DATA_DIR)
    rows = parts['valid_search']

    pos, imp = collections.Counter(), collections.Counter()
    for x in parts['train']:
        imp[x[2]] += 1
        pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())

    user_ids = [x[1] for x in rows]
    labels = [x[6] for x in rows]
    scores = [
        (pos[x[2]] + 20.0 * gmean) / (imp[x[2]] + 20.0) if imp[x[2]] else gmean
        for x in rows
    ]

    direct = evaluate(user_ids, labels, scores)
    wrapped = harness.validated_evaluate(user_ids, labels, scores, 'valid_search')

    assert wrapped == direct, 'the harness wrapper changed the reported metrics'
    assert wrapped['rows'] == len(rows)


def test_validated_evaluate_accepts_float32_arrays_from_encode():
    """The arrays get_split hands out are int32/float32; feeding those straight
    back into validated_evaluate must work and must agree with the float path."""
    import numpy as np

    parts = partition(DEFAULT_DATA_DIR)
    rows = parts['valid_confirm']
    user_ids = [x[1] for x in rows]
    labels32 = np.array([x[6] for x in rows], dtype=np.float32)
    scores32 = np.arange(len(rows), dtype=np.float32)

    wrapped = harness.validated_evaluate(user_ids, labels32, scores32, 'valid_confirm')
    direct = evaluate(user_ids, [int(v) for v in labels32], scores32.tolist())
    assert wrapped == direct
