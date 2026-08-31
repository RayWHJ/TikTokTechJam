"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。

encode()     : FM 的 5 个类别域 -> 连续 id。只依赖标准库和 numpy。
encode_lgb() : GBDT 的稠密数值特征（train-only 统计量）。同样只依赖 numpy。
"""
import csv, os, collections, itertools
from typing import Callable, List, Tuple
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}

#: The 5 categorical domains encode()'s inner raw(x) reads off a row directly.
#: These are positional and pinned — raw(x) returns them in this order.
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

#: THE FRONT DOOR FOR A NEW CATEGORICAL FIELD. Append `(name, fn)` and you are
#: done: X's width, the vocabulary list, the UNK slots, field_dims, offsets and
#: the X allocation all follow from this one list.
#:
#: `fn(rows) -> sequence of len(rows) values`, computed ONCE PER SPLIT and
#: VECTORISED, not once per row. Two reasons that shape and not `fn(x)`:
#:
#:  1. It is what a LAG feature needs. "The user's previous video_id" cannot be
#:     computed from a row in isolation, and it is the most-proposed direction in
#:     this repo. A per-split callable sees exactly the rows of one split, which
#:     makes split-locality structural rather than something the author has to
#:     remember.
#:  2. A per-row Python callable over 1.4M rows is the timeout that killed a
#:     candidate in the recorded run. Handing the whole split over at once lets
#:     the implementation be numpy.
#:
#: Values are coerced with str() and mapped through a TRAIN-ONLY vocabulary, so
#: an unseen value lands in that field's UNK slot automatically.
#:
#: WHY THIS EXISTS. Adding one categorical field used to be a coupled four-place
#: edit — FIELDS, raw(), the vocab list and the X allocation — with no single
#: point of change. Two candidates in the recorded run died with the IDENTICAL
#: IndexError inside `for i, v in enumerate(raw(x))`, having extended raw(x) and
#: not FIELDS. Three of five candidates never produced a paired delta, so that
#: one coupling consumed more search budget than every measurement combined.
EXTRA_FIELDS: List[Tuple[str, Callable]] = []

#: All categorical domains, base plus registered extras. Derived, never edited by
#: hand. Kept a module-level list with this name because baseline.py imports it
#: and the proposer/writer prompts quote it verbatim.
FIELDS = BASE_FIELDS + [n for n, _ in EXTRA_FIELDS]


def active_fields() -> List[str]:
    """The field list in effect for an encode() call, right now.

    Reconciles the module-level FIELDS with EXTRA_FIELDS rather than trusting
    either alone, so BOTH ways in work: `add_categorical_field(...)` (which keeps
    the two in sync) and a bare `EXTRA_FIELDS.append(...)` by a candidate that
    never touched FIELDS.
    """
    names = list(FIELDS)
    for name, _fn in EXTRA_FIELDS:
        if name not in names:
            names.append(name)
    return names


def add_categorical_field(name: str, fn: Callable) -> None:
    """Register one new fixed-width categorical field. The whole edit.

    `fn(rows)` returns one value per row of the split it is handed — see
    EXTRA_FIELDS for the contract and why it is per-split rather than per-row.
    """
    if name in active_fields():
        raise ValueError(f"field {name!r} is already registered: {active_fields()}")
    EXTRA_FIELDS.append((name, fn))
    if name not in FIELDS:
        FIELDS.append(name)

#: 'train' re-cut by date into the rows a model FITS on and the rows it
#: EARLY-STOPS on. These two partition SPLITS['train'] exactly; 'train' itself
#: stays defined over the full range, because encode()'s vocabularies and
#: _bucket_edges are built from splits['train'] and building them from train_fit
#: alone would push two days' worth of ids into the per-field UNK slot.
#:
#: Why this exists. run_fm early-stops on `select_on` and then reports the
#: metric on `report_on`, and the orchestrator passed valid_search for both. The
#: stopping epoch is chosen by maximising primary on those exact rows
#: (patience=4, epochs=40), so a candidate that happens to train more epochs
#: gets more draws at the maximum on the split it is scored on, and scores
#: higher for a reason that has nothing to do with its mechanism. The paired
#: bootstrap cannot see it, because it compares only final per-user scores.
#: MEASURED, 3 seeds each, root primary on valid_search:
#:   fit 14d, stop on valid_search, report valid_search   0.5946398  (old)
#:   fit 12d, stop on valid_search, report valid_search   0.5940682
#:   fit 12d, stop on train_es,     report valid_search   0.5935765  (new)
#: So the two effects separate cleanly: -0.00057 is the real cost of fitting on
#: 12 days instead of 14, and -0.00049 is the optimistic bias this change
#: removes. That second number is 0.83x the largest candidate-minus-parent delta
#: the search has ever measured (+0.000593, iteration 1 of the run in
#: orchestrator/_state/progress.json) — so most of that "gain" was plausibly the
#: candidate training more epochs, not its mechanism. The new number is lower
#: and comparable across candidates; the old one was higher and was not.
#:
#: train_es is the two days immediately before 'valid' starts, so it is genuinely
#: held out from fitting while staying temporally adjacent to the rows the model
#: trained on — which is what makes it a fair stopping signal for a split that
#: comes later still.
TRAIN_SUBSPLITS = {'train_fit': (20220408, 20220419),
                   'train_es':  (20220420, 20220421)}


def cut_train_subsplits(splits):
    """Add train_fit / train_es, re-cut from 'train' by date.

    Mirrors baseline.py::cut_valid_subsplits. Returns a new dict; 'train' is
    left in place untouched, since encode() reads it for the vocabulary.
    """
    out = dict(splits)
    for name, (lo, hi) in TRAIN_SUBSPLITS.items():
        out[name] = [x for x in splits['train'] if lo <= x[0] <= hi]
    return out

#: Auxiliary feedback signals, loaded onto each row tuple at index 7 and beyond
#: in exactly this order. They are permitted as auxiliary LOSS TARGETS and are
#: NEVER model inputs — the value for the row being scored is the same-row
#: outcome, so feeding it in is leakage. codegen/gate.py rule 4 blocks that
#: statically and codegen/constants.py::AUXILIARY_SIGNALS must list every name
#: here (tests/test_aux_targets.py asserts it does).
#:
#: Why they are loaded at all: multi-task auxiliary targets are the starter kit's
#: own direction 3, and it was structurally unreachable. load() read only date /
#: user_id / video_id / author_id / tab / duration_ms / long_view, so no
#: candidate could use these as targets however it was written — and the one-file
#: rule forbade a candidate from adding them to data.py and consuming them in
#: baseline.py at the same time. Plumbing them here, plus aux_targets() below,
#: turns the direction into a baseline.py-only edit.
#:
#: APPEND ONLY. Indices 0..6 keep their current meaning: data.py, baseline.py and
#: harness/_data.py all index these tuples POSITIONALLY, and evaluate.py's
#: callers depend on x[6] being the label.
AUX_SIGNALS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
               'play_time_ms']

#: Where AUX_SIGNALS starts in a row tuple: after the original 7 fields.
AUX_OFFSET = 7


def _aux_value(r, col):
    """One auxiliary signal off a CSV row, tolerating a missing column.

    Fills 0 rather than raising, so the loader still works on a truncated or
    hand-built dataset — the sandbox fixtures and several tests use cut-down
    CSVs, and a KeyError here would break loading entirely for a signal a
    candidate may not even use. Same '!= "0"' coercion as LABEL for the binary
    signals; play_time_ms is continuous.
    """
    v = r.get(col)
    if v is None or v == '':
        return 0.0 if col == 'play_time_ms' else 0
    if col == 'play_time_ms':
        try:
            return float(v)
        except ValueError:
            return 0.0
    return 1 if v != '0' else 0


def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。

    Row tuple layout: [0]=date, [1]=user_id, [2]=video_id, [3]=author_id,
    [4]=tab, [5]=duration_ms, [6]=long_view label, then AUX_SIGNALS from
    index AUX_OFFSET (=7) onward, in AUX_SIGNALS order.
    """
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0)
                            + tuple(_aux_value(r, c) for c in AUX_SIGNALS))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def aux_targets(splits):
    """Auxiliary TARGETS per split, aligned row-for-row with encode()'s X.

    Returns {split_name: float32 (N, len(AUX_SIGNALS))}, column order matching
    AUX_SIGNALS.

    This is the single seam a baseline.py-only candidate needs, which is what
    makes the multi-task direction reachable under the one-file rule: a
    candidate adds an auxiliary head and a weighted term to the loss in
    baseline.py and calls this for the labels, touching no other file.

    ALIGNMENT. encode() iterates `for n, x in enumerate(rws)` over
    splits[name], and so does this — so row n here is row n of X, for the SAME
    splits dict. Pass the same object to both. Slicing or reordering a split
    between the two calls silently misaligns targets against features.

    THESE ARE TARGETS, NEVER INPUTS. The value for the row being scored is that
    row's own outcome; putting it in the feature matrix leaks the label.
    codegen/gate.py blocks it statically, both by signal name and by this
    function's name.

    Rows shorter than AUX_OFFSET + len(AUX_SIGNALS) yield zeros rather than
    raising, so hand-built 7-tuples in tests still work.
    """
    n_aux = len(AUX_SIGNALS)
    out = {}
    for name, rws in splits.items():
        A = np.zeros((len(rws), n_aux), dtype=np.float32)
        for n, x in enumerate(rws):
            vals = x[AUX_OFFSET:AUX_OFFSET + n_aux]
            if len(vals) == n_aux:
                A[n] = vals
        out[name] = A
    return out


# --------------------------------------------------------------------------- #
#  Verified primitives the writer composes (T2.10)                             #
# --------------------------------------------------------------------------- #
# WHY THESE EXIST. Three of the five proposals in the recorded run needed exactly
# a previous-value-within-user computation, and the one that got the semantics
# right timed out implementing it in a Python loop over 1.4M rows. These turn the
# hardest part of the most-proposed direction into one call, with causality and
# split boundaries already correct — and they narrow what the static gate and the
# auditor have to reason about, because a known-safe primitive replaces novel
# indexing code on every attempt.
#
# Each one is SPLIT-LOCAL by construction: it is handed one split's rows and can
# see nothing else. And each is POINT-IN-TIME safe: it reads only rows strictly
# before the row it describes. tests/test_primitives.py asserts both.

def prev_value_within_user(rows, key=2, missing='NONE'):
    """The user's PREVIOUS value of column `key`, in date order. One per row.

    `key` is an index into the row tuple — 2 is video_id, 3 is author_id,
    6 is the long_view label, 4 is tab. Returns a list of strings aligned
    row-for-row with `rows` AS GIVEN, so it plugs straight into EXTRA_FIELDS.

    CAUSALITY. Rows are ranked by date within each user and each row is described
    by the value from the row before it, so the first impression a user has in
    this split gets `missing`. A row is never described by itself or by anything
    after it. Ties on date keep the order they appear in `rows`, which is the
    logged order — `sorted` is stable.

    SPLIT-LOCALITY. Only `rows` is visible, so a valid-split row's "previous"
    value never reaches back into train. That is deliberate and it is the
    conservative direction: it under-informs the first row of each user per
    split rather than leaking across the boundary.

    Cost is one sort plus one pass — no per-row Python callback into numpy, which
    is what timed out at 240s over 1.4M rows.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    out = [missing] * len(rows)
    last = {}
    for i in order:
        u = rows[i][1]
        if u in last:
            out[i] = last[u]
        last[u] = str(rows[i][key])
    return out


def prior_count_within_user(rows, key=3, buckets=(0, 1, 2, 5)):
    """How many times this user has ALREADY seen `rows[i][key]`, bucketed.

    Returns a list of bucket labels as strings, aligned with `rows`. The count is
    strictly prior: the current row is not counted in its own feature.

    Bucketed rather than raw because a raw count is a high-cardinality integer
    domain in a fixed-width categorical field, and the FM would spend an
    embedding per distinct count. `buckets` are lower bounds, so the default
    gives 0 / 1 / 2 / 3-5 / 6+.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    out = [''] * len(rows)
    seen = collections.Counter()
    for i in order:
        k = (rows[i][1], rows[i][key])
        n = seen[k]
        label = str(buckets[-1]) + '+'
        for b_i, b in enumerate(buckets):
            nxt = buckets[b_i + 1] if b_i + 1 < len(buckets) else None
            if nxt is None:
                if n >= b:
                    label = f'{b}+'
                break
            if b <= n < nxt:
                label = str(b) if nxt == b + 1 else f'{b}-{nxt - 1}'
                break
        out[i] = label
        seen[k] = n + 1
    return out


def position_within_user(rows, buckets=(0, 1, 2, 5, 10, 20)):
    """The row's 0-based position in this user's log for this split, bucketed.

    A coarse "how deep into the session are we" field. Uses only rows at or
    before the current row — the position of row i depends on how many of this
    user's rows precede it, never on how many follow.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    out = [''] * len(rows)
    seen = collections.Counter()
    for i in order:
        u = rows[i][1]
        n = seen[u]
        label = f'{buckets[-1]}+'
        for b_i, b in enumerate(buckets):
            nxt = buckets[b_i + 1] if b_i + 1 < len(buckets) else None
            if nxt is None:
                if n >= b:
                    label = f'{b}+'
                break
            if b <= n < nxt:
                label = str(b) if nxt == b + 1 else f'{b}-{nxt - 1}'
                break
        out[i] = label
        seen[u] = n + 1
    return out


def within_user_pairs(users, y, rng, max_pairs_per_user=8):
    """Vectorised (positive_row, negative_row) index pairs, sampled within user.

    Returns (pos_idx, neg_idx) as int64 arrays into the SAME rows `users`/`y`
    describe, for a pairwise loss (BPR, RankNet) or a within-user softmax.

    This is the other computation the recorded run kept needing and kept getting
    wrong. It is fully vectorised: one lexsort, then numpy indexing. A per-user
    Python loop over 1.4M rows is the 240s timeout.

    RNG. Takes its generator as an ARGUMENT and never touches np.random or
    run_fm's own `rng`. That is a hard rule here, not hygiene: the orchestrator
    pairs a candidate against its parent SEED BY SEED, which only cancels noise
    while both runs consume the same draws in the code they share. One extra draw
    from the shared `rng` shifts every later epoch shuffle, the trajectories
    decorrelate, and the inflated paired variance is indistinguishable from a real
    effect. codegen/gate.py rejects a diff that draws from the wrong generator.
    Call it as `within_user_pairs(users, y, np.random.default_rng(seed + 1000))`.

    Users with no positive or no negative contribute no pairs, which is correct:
    they carry no within-user ranking signal and GAUC excludes them too.
    """
    users = np.asarray(users)
    y = np.asarray(y)
    # Group rows by user without a Python loop: sort by user, then slice runs.
    order = np.argsort(users, kind='stable')
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(su)]

    pos_out, neg_out = [], []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        yy = y[idx]
        p = idx[yy > 0]
        n = idx[yy <= 0]
        if not len(p) or not len(n):
            continue
        k = min(max_pairs_per_user, max(len(p), len(n)))
        pos_out.append(rng.choice(p, size=k, replace=len(p) < k))
        neg_out.append(rng.choice(n, size=k, replace=len(n) < k))
    if not pos_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (np.concatenate(pos_out).astype(np.int64),
            np.concatenate(neg_out).astype(np.int64))


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。

    To add a categorical field, append to EXTRA_FIELDS (or call
    add_categorical_field). Everything below — the vocabularies, the UNK slots,
    field_dims, offsets and X's width — is derived from `fields`, so that append
    is the entire change. Do not extend raw(x) by hand: raw(x) covers
    BASE_FIELDS and only BASE_FIELDS, and returning an extra value from it is
    what produced the two identical IndexErrors in the recorded run.
    """
    fields = active_fields()
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    # Where each registered extra lands in `fields`. By NAME, not by position:
    # a candidate may have appended to FIELDS as well, and a shifted index would
    # write the new feature into an existing field's column.
    extra_at = {name: fields.index(name) for name, _fn in EXTRA_FIELDS}

    def extra_columns(rws):
        """EXTRA_FIELDS as string columns for one split. One call per field per
        split, so the implementation can be vectorised."""
        cols = {}
        for name, fn in EXTRA_FIELDS:
            vals = list(fn(rws))
            assert len(vals) == len(rws), (
                f"EXTRA_FIELDS[{name!r}] returned {len(vals)} values for a "
                f"split of {len(rws)} rows; it must return exactly one value "
                f"per row, in row order")
            cols[extra_at[name]] = [str(v) for v in vals]
        return cols

    if tr:
        n_raw = len(raw(tr[0]))
        assert n_raw <= len(fields), (
            f"encode()'s raw(x) returns {n_raw} values but there are "
            f"{len(fields)} fields, so {n_raw - len(fields)} value(s) would be "
            f"written past X's last column — this is the IndexError in "
            f"`for i, v in enumerate(raw(x))` that killed two candidates in the "
            f"recorded run. Register the new field in EXTRA_FIELDS instead of "
            f"extending raw(x).\n"
            f"  fields ({len(fields)}): {fields}\n"
            f"  raw(x) ({n_raw}): {raw(tr[0])}")

    vocabs = [dict() for _ in fields]
    tr_extra = extra_columns(tr)
    for n, x in enumerate(tr):
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
        for i, col in tr_extra.items():
            if col[n] not in vocabs[i]:
                vocabs[i][col[n]] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        cols = tr_extra if rws is tr else extra_columns(rws)
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        # Pre-fill every column with its own UNK id. Only matters for a field
        # that has NO value source — a name appended to FIELDS with no matching
        # EXTRA_FIELDS entry — which previously left that column as np.empty's
        # uninitialized memory and fed the FM garbage indices. Now it degrades to
        # a constant UNK embedding: still useless, but deterministic and
        # harmless instead of undefined.
        for i in range(len(fields)):
            X[:, i] = unk[i] + offsets[i]
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            for i, col in cols.items():
                X[n, i] = vocabs[i].get(col[n], unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


# --------------------------------------------------------------------------- #
#  GBDT feature encoding (used by baseline.py --model lgb)                     #
# --------------------------------------------------------------------------- #
# Dense features for a tree model. Every statistic below is computed on the
# TRAIN split ONLY and then looked up for valid/test, which is what makes them
# point-in-time safe: train is 20220408-0421, strictly earlier than valid
# (0422-0428) and test (0429-0508), so no row is ever described by information
# that did not exist when it was logged. A row's own label is never an input.
#
# Why these and not "target-encode every id": measured on this dataset, a
# user x author pair occurs 1.07 times in train on average (1,070,326 pairs
# over 1,141,112 rows) and user x video is sparser still. A per-pair rate is
# therefore one observation of noise, and smoothing collapses it back to the
# author rate. That sparsity is exactly why the FM's *embeddings* beat the
# popularity baseline (0.5946 vs 0.5715) — SGD shares strength across users,
# count features cannot. `video_rate_by_user_bucket` is the compromise: a
# COARSE collaborative cross (video x user-activity-decile, ~7.5k x 10 cells
# over 1.14M rows ≈ 15 observations each) that carries a personalisation signal
# without dissolving into per-pair noise.
LGB_FIELDS = [
    'video_rate',               # item quality: the `pop` baseline's whole signal
    'video_imp_log',            # item exposure mass
    'author_rate',              # generalises to videos with thin history
    'author_imp_log',
    'user_rate',                # constant within a user, but lets the trees
    'user_imp_log',             # modulate item features per user type
    'duration_log',             # item-side, varies within user; long_view is a
                                # watch-time threshold so duration is direct
    'tab',                      # 15 levels, varies within user
    'video_rate_by_user_bucket',  # the coarse collaborative cross
    'video_is_new',             # 1.0 if the video is unseen in train
]
#: Indices into LGB_FIELDS that LightGBM should treat as categorical.
LGB_CATEGORICAL = ['tab']

#: Smoothing counts. `prior` pulls a thin per-id rate toward the global rate;
#: `cross_prior` pulls a thin (video, bucket) cell toward that video's own rate,
#: which is a far better fallback than the global mean.
LGB_PRIOR = 20.0
LGB_CROSS_PRIOR = 10.0
LGB_USER_BUCKETS = 10


def _rate_maps(rows, key_fn):
    """(positives, impressions) counters keyed by key_fn(row)."""
    pos, imp = collections.Counter(), collections.Counter()
    for x in rows:
        k = key_fn(x)
        imp[k] += 1
        pos[k] += x[6]
    return pos, imp


def encode_lgb(splits, prior=LGB_PRIOR, cross_prior=LGB_CROSS_PRIOR,
               n_user_buckets=LGB_USER_BUCKETS, sort_by_user=True):
    """Dense train-only features for a GBDT, grouped by user for ranking.

    Returns (enc, feature_names, categorical_names) where
    enc[split] = (X float32 (N, F), y float32 (N,), users list, groups int32).

    Rows are SORTED BY USER inside each split, because LightGBM's ranking
    objectives take a `group` array of contiguous run lengths rather than a key
    column. This is safe for scoring: evaluate() buckets by user_id itself and
    is order-independent. It does mean these arrays are NOT in submission row
    order — submit.py's row_id contract is served by the FM path.
    """
    tr = splits['train']
    gmean = sum(x[6] for x in tr) / len(tr)

    vid_pos, vid_imp = _rate_maps(tr, lambda x: x[2])
    aut_pos, aut_imp = _rate_maps(tr, lambda x: x[3])
    usr_pos, usr_imp = _rate_maps(tr, lambda x: x[1])

    def smoothed(pos, imp, k, p, fallback):
        """Empty-count keys land exactly on `fallback` (p*fallback / p)."""
        return (pos.get(k, 0) + p * fallback) / (imp.get(k, 0) + p)

    # User activity deciles, cut on TRAIN impression counts. A user absent from
    # train gets bucket -1 (its own level) rather than being forced into a
    # decile it has no evidence for.
    counts = np.array(sorted(usr_imp.values()), dtype=np.float64)
    bucket_edges = np.quantile(counts, np.linspace(0, 1, n_user_buckets + 1)[1:-1])

    def user_bucket(u):
        n = usr_imp.get(u)
        return -1 if n is None else int(np.searchsorted(bucket_edges, n))

    # The coarse collaborative cross, also train-only.
    cross_pos, cross_imp = _rate_maps(tr, lambda x: (x[2], user_bucket(x[1])))

    tabs = sorted({x[4] for x in tr})
    tab_code = {t: i for i, t in enumerate(tabs)}   # unseen tab -> -1

    def featurise(x):
        vid, aut, usr = x[2], x[3], x[1]
        v_rate = smoothed(vid_pos, vid_imp, vid, prior, gmean)
        b = user_bucket(usr)
        return (
            v_rate,
            float(np.log1p(vid_imp.get(vid, 0))),
            smoothed(aut_pos, aut_imp, aut, prior, gmean),
            float(np.log1p(aut_imp.get(aut, 0))),
            smoothed(usr_pos, usr_imp, usr, prior, gmean),
            float(np.log1p(usr_imp.get(usr, 0))),
            float(np.log1p(x[5])),
            float(tab_code.get(x[4], -1)),
            smoothed(cross_pos, cross_imp, (vid, b), cross_prior, v_rate),
            0.0 if vid in vid_imp else 1.0,
        )

    enc = {}
    for name, rws in splits.items():
        # sort_by_user=False keeps splits[name]'s own row order, so these arrays
        # line up index-for-index with encode()'s — needed to stack an FM score
        # (or any other per-row model output) in as a feature. Only the ranking
        # objectives need the contiguous-user layout.
        if sort_by_user:
            rws = sorted(rws, key=lambda x: x[1])   # contiguous user blocks
        X = np.empty((len(rws), len(LGB_FIELDS)), dtype=np.float32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            X[n, :] = featurise(x)
            y[n] = x[6]
            users.append(x[1])
        groups = np.array([len(list(g)) for _, g in itertools.groupby(users)],
                          dtype=np.int32)
        enc[name] = (X, y, users, groups)
    return enc, list(LGB_FIELDS), list(LGB_CATEGORICAL)
