"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。

encode()     : FM 的 5 个类别域 -> 连续 id。只依赖标准库和 numpy。
encode_lgb() : GBDT 的稠密数值特征（train-only 统计量）。同样只依赖 numpy。
"""
import csv, os, collections, itertools
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

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


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
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
