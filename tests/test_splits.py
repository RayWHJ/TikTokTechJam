"""Split-integrity tests: the partition must be airtight, or every score the
agent loop produces is measuring the wrong thing.

Two of these are load-bearing beyond ordinary coverage:

  * test_cheap_sizes_match_load_derived_sizes is what licenses _sizes.py's
    date-column shortcut. If that shortcut ever disagrees with data.load(),
    validated_evaluate would accept or reject arrays on a wrong row count.
  * test_no_row_appears_in_two_splits is the leak check. A single row shared
    between train and any valid tier would silently inflate every measured
    delta.

They pay one real data.load() (~14s), which is why this module is marked slow and
kept out of the default run. Invoke with: pytest -m slow
"""
import pytest

from data import load, SPLITS
from harness._data import DEFAULT_DATA_DIR, partition
from harness._sizes import SPLIT_NAMES, SPLIT_RANGES, get_sizes

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def raw():
    return load(DEFAULT_DATA_DIR)


@pytest.fixture(scope='module')
def parts():
    return partition(DEFAULT_DATA_DIR)


# ---------------------------------------------------------------------------
# The two valid tiers must exactly partition data.py's 'valid'
# ---------------------------------------------------------------------------

def test_valid_tiers_partition_official_valid(raw, parts):
    total = len(parts['valid_search']) + len(parts['valid_confirm'])
    assert total == len(raw['valid']), (
        'valid_search + valid_confirm must account for every official valid row'
    )


def test_valid_tier_ranges_are_contiguous_and_cover_official_valid():
    lo_official, hi_official = SPLITS['valid']
    lo_s, hi_s = SPLIT_RANGES['valid_search']
    lo_c, hi_c = SPLIT_RANGES['valid_confirm']
    assert lo_s == lo_official
    assert hi_c == hi_official
    assert lo_c == hi_s + 1, 'no gap and no overlap between the two valid tiers'


def test_train_and_test_ranges_match_data_py_exactly():
    # If data.py's official split ever moves, this harness must not silently
    # keep using a stale range.
    assert SPLIT_RANGES['train'] == SPLITS['train']
    assert SPLIT_RANGES['test'] == SPLITS['test']


def test_every_row_falls_in_its_declared_date_range(parts):
    for name in SPLIT_NAMES:
        lo, hi = SPLIT_RANGES[name]
        dates = {x[0] for x in parts[name]}
        assert dates, f'{name} is empty'
        assert min(dates) >= lo and max(dates) <= hi, (
            f'{name} contains dates outside {lo}-{hi}: '
            f'{sorted(d for d in dates if not lo <= d <= hi)}'
        )


def test_valid_search_covers_five_days_and_confirm_two(parts):
    assert len({x[0] for x in parts['valid_search']}) == 5
    assert len({x[0] for x in parts['valid_confirm']}) == 2


# ---------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------

def test_no_row_appears_in_two_splits(parts):
    """Splits must be mutually exclusive as sets of rows.

    Rows are compared as whole tuples including date, which is what makes this a
    real check: (user_id, video_id) is deliberately non-unique in this dataset
    (3.06% of test pairs repeat, up to 12 times), so a pair-level check would
    report false leaks.
    """
    seen = {}
    for name in SPLIT_NAMES:
        for x in parts[name]:
            prev = seen.setdefault(x, name)
            assert prev == name, f'row {x} appears in both {prev} and {name}'


def test_split_date_ranges_are_mutually_exclusive():
    spans = [(name, *SPLIT_RANGES[name]) for name in SPLIT_NAMES]
    for i, (n1, lo1, hi1) in enumerate(spans):
        for n2, lo2, hi2 in spans[i + 1:]:
            assert hi1 < lo2 or hi2 < lo1, f'{n1} and {n2} date ranges overlap'


def test_partition_leaves_train_and_test_untouched(raw, parts):
    assert parts['train'] is raw['train'] or parts['train'] == raw['train']
    assert parts['test'] is raw['test'] or parts['test'] == raw['test']


# ---------------------------------------------------------------------------
# The shortcut that validated_evaluate depends on
# ---------------------------------------------------------------------------

def test_cheap_sizes_match_load_derived_sizes(parts):
    """_sizes.py counts rows by scanning only the date column, skipping the video
    feature join and tuple construction that load() does. That is a ~17x speedup
    on a code path validated_evaluate hits constantly, and this test is the only
    thing keeping the two implementations honest with each other."""
    cheap = get_sizes(DEFAULT_DATA_DIR)
    expensive = {name: len(parts[name]) for name in SPLIT_NAMES}
    assert cheap == expensive


def test_no_log_row_is_dropped_by_the_partition(parts):
    """Every row of both log files lands in exactly one split.

    The four contract ranges cover 20220408-20220508 with no gaps, so a
    discrepancy here means either a stray date in the logs or a range typo.
    """
    from harness._sizes import LOG_FILES
    import csv
    import os

    total_log_rows = 0
    for fname in LOG_FILES:
        with open(os.path.join(DEFAULT_DATA_DIR, fname), newline='') as fh:
            total_log_rows += sum(1 for _ in csv.reader(fh)) - 1  # minus header

    assert sum(len(parts[n]) for n in SPLIT_NAMES) == total_log_rows
