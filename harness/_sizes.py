"""Cheap, disk-cached row counts per split.

Why this exists: validated_evaluate() only needs to know how many rows a split
has in order to reject a mis-sized array. Getting that from _data.get_encoded()
costs a full load() + encode() (~34s cold on this dataset) — unacceptable when a
caller is validating a 3-element fixture, and paid again in every subprocess that
codegen.execute() spawns.

So we count rows directly off the two log CSVs, reading only the `date` column,
and memoize the result to disk keyed on a signature of the input files. Cold cost
is ~2s, warm cost is ~0ms.

The counts this module produces are asserted equal to load()-derived counts in
tests/test_splits.py — that test is what licenses the shortcut.
"""
import csv
import json
import os

# The official date ranges. train/test match data.py's SPLITS exactly; the two
# valid_* ranges partition data.py's 'valid' (20220422-20220428) per the frozen
# interface contract.
SPLIT_RANGES = {
    'train':         (20220408, 20220421),
    'valid_search':  (20220422, 20220426),
    'valid_confirm': (20220427, 20220428),
    'test':          (20220429, 20220508),
}

SPLIT_NAMES = ('train', 'valid_search', 'valid_confirm', 'test')

# Same two files, same order, as data.py:load(). Order does not matter for
# counting but is kept identical so the signature covers exactly what load() reads.
LOG_FILES = (
    'log_standard_4_08_to_4_21_pure.csv',
    'log_standard_4_22_to_5_08_pure.csv',
)

CACHE_DIRNAME = '.harness_cache'
CACHE_FILENAME = 'split_sizes.json'

_memo = {}  # data_dir -> {name: int}


def _signature(data_dir):
    """A cheap fingerprint of the log files, so a changed dataset busts the cache."""
    parts = []
    for fname in LOG_FILES:
        st = os.stat(os.path.join(data_dir, fname))
        parts.append(f"{fname}:{st.st_size}:{int(st.st_mtime)}")
    return '|'.join(parts)


def _cache_path(data_dir):
    return os.path.join(data_dir, CACHE_DIRNAME, CACHE_FILENAME)


def _count(data_dir):
    """Scan the `date` column of both log files, bucketing rows into splits."""
    counts = {name: 0 for name in SPLIT_NAMES}
    for fname in LOG_FILES:
        with open(os.path.join(data_dir, fname), newline='') as fh:
            reader = csv.reader(fh)
            header = next(reader)
            di = header.index('date')
            for rec in reader:
                d = int(rec[di])
                for name, (lo, hi) in SPLIT_RANGES.items():
                    if lo <= d <= hi:
                        counts[name] += 1
                        break
    return counts


def get_sizes(data_dir):
    """Return {split_name: row_count} for the four contract splits.

    Memoized in-process, and cached on disk under
    <data_dir>/.harness_cache/split_sizes.json keyed on the log files' size+mtime.
    A cache-write failure (read-only data dir) is non-fatal.

    Args:
        data_dir: the KuaiRand-Pure/data directory.

    Returns:
        dict mapping each of "train", "valid_search", "valid_confirm", "test" to
        its row count.
    """
    if data_dir in _memo:
        return _memo[data_dir]

    sig = _signature(data_dir)
    path = _cache_path(data_dir)
    try:
        with open(path, encoding='utf-8') as fh:
            blob = json.load(fh)
        if blob.get('signature') == sig:
            counts = {name: int(blob['counts'][name]) for name in SPLIT_NAMES}
            _memo[data_dir] = counts
            return counts
    except (OSError, ValueError, KeyError):
        pass  # no cache, unreadable cache, or stale schema -> recount

    counts = _count(data_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'signature': sig, 'counts': counts}, fh, indent=2)
    except OSError:
        pass  # caching is an optimization, never a requirement

    _memo[data_dir] = counts
    return counts
