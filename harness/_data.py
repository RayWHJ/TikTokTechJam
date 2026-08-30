"""Internal helper: builds on data.py's load()/encode() to add the
valid_search / valid_confirm split, without modifying data.py itself.

data.load() already returns a 'valid' split covering 20220422-20220428 with each
row tuple carrying its date as element 0 ([0]=date, [1]=user_id, [2]=video_id,
[3]=author_id, [4]=tab, [5]=duration_ms, [6]=label). We re-filter that split by
date into the two sealed sub-splits, then hand the resulting 4-way dict to
encode() as-is — encode() is generic over whatever split names it's given, and
derives its vocabularies and duration-bucket edges from splits['train'] only, so
partitioning valid cannot alter the encoding.
"""
import os
import threading

from data import load, encode

from ._sizes import SPLIT_NAMES, SPLIT_RANGES, get_sizes

VALID_SEARCH_RANGE = SPLIT_RANGES['valid_search']
VALID_CONFIRM_RANGE = SPLIT_RANGES['valid_confirm']

DEFAULT_DATA_DIR = os.environ.get('HARNESS_DATA_DIR', './KuaiRand-Pure/data')

_lock = threading.Lock()
_cache = {}  # data_dir -> {'enc': {...}, 'dim': int, 'sizes': {name: int}}


def partition(data_dir):
    """load() the raw splits and re-cut 'valid' into valid_search / valid_confirm.

    Returns a dict keyed by the four contract split names, values being lists of
    raw data.py row tuples.
    """
    raw = load(data_dir)
    lo_s, hi_s = VALID_SEARCH_RANGE
    lo_c, hi_c = VALID_CONFIRM_RANGE
    return {
        'train': raw['train'],
        'valid_search': [x for x in raw['valid'] if lo_s <= x[0] <= hi_s],
        'valid_confirm': [x for x in raw['valid'] if lo_c <= x[0] <= hi_c],
        'test': raw['test'],
    }


def get_encoded(data_dir=None):
    """Load + partition + encode once per data_dir, cached in-process.

    Expensive (~34s cold). Only call this when you actually need feature arrays;
    for row counts alone use get_split_sizes(), which is ~2s cold / free warm.

    Returns {'enc': {name: (X, y, user_ids)}, 'dim': int, 'sizes': {name: int}}.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    with _lock:
        if data_dir not in _cache:
            splits = partition(data_dir)
            enc, dim = encode(splits)
            sizes = {name: len(splits[name]) for name in SPLIT_NAMES}
            _cache[data_dir] = {'enc': enc, 'dim': dim, 'sizes': sizes}
        return _cache[data_dir]


def get_split_sizes(data_dir=None):
    """Row count per contract split, without paying for load()/encode().

    Prefers the in-process encoded cache if it already exists (exact by
    construction); otherwise falls back to the cheap disk-cached column scan.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    with _lock:
        cached = _cache.get(data_dir)
    if cached is not None:
        return cached['sizes']
    return get_sizes(data_dir)
