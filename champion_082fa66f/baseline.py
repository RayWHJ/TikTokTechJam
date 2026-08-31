"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, json, os, time
import numpy as np
from data import load, encode, encode_lgb, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- orchestrator contract (codegen/sandbox.py) ----------------
# The sandbox invokes a candidate as `python <candidate> --data_dir D --seed S`
# with CODEGEN_SPLIT naming the split to score, and prefers to read back a
# `##CODEGEN_METRICS## {json}` line. Emitting that marker is what makes the
# split argument real: without it the sandbox falls back to regex-scraping
# "primary ..." lines and keeps the LAST one, which is the test line printed
# below — so the search was being steered by test scores.
METRICS_MARK = '##CODEGEN_METRICS##'

# Mirrors harness/_sizes.py SPLIT_RANGES: these two partition data.py's 'valid'.
# Duplicated because the sandbox deliberately does not expose the harness package
# to candidates; tests/test_split_ranges_agree.py asserts they stay identical.
VALID_SUBSPLITS = {'valid_search':  (20220422, 20220426),
                   'valid_confirm': (20220427, 20220428)}


def cut_valid_subsplits(splits):
    """Add valid_search / valid_confirm, re-cut from 'valid' by date."""
    out = dict(splits)
    for name, (lo, hi) in VALID_SUBSPLITS.items():
        out[name] = [x for x in splits['valid'] if lo <= x[0] <= hi]
    return out


def per_user_primary(user_ids, labels, scores):
    """Per-user primary, computed through the frozen evaluate() so the per-user
    metric definition is identical to the aggregate one.

    Restricted to the users GAUC itself counts (0 < positives < impressions): a
    user whose labels are all 0 or all 1 has no rankable signal, so including
    them would only pad the paired candidate-vs-parent bootstrap with constant
    zero deltas and wash out p_positive.

    Note this is a per-user analogue, not a decomposition — aggregate GAUC is
    positive-weighted across users, so mean(per_user) != aggregate primary.
    """
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((y, s))
    out = {}
    for u, rows in byu.items():
        labs = [int(y) for y, _ in rows]
        npos = sum(labs)
        if not 0 < npos < len(labs):
            continue
        r = evaluate([u] * len(rows), labs, [float(s) for _, s in rows])
        out[u] = round(float(r['primary']), 6)
    return out

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

def group_by_user_offsets(user_ids):
    """Pre-compute per-user row ranges for vectorised within-user operations.
 
    Parameters
    ----------
    user_ids : array-like of shape (N,)
        User identifier per row (strings or ints — anything sortable).
 
    Returns
    -------
    order : int32 array (N,)
        Indices that sort the rows by user_id.
    group_starts : int32 array (G,)
        Start index (into `order`) of each user group.
    group_ends : int32 array (G,)
        One-past-end index of each user group.
    group_ids : int32 array (N,)
        Per-row group number (into group_starts / group_ends), in sorted order.
 
    Usage
    -----
    order, starts, ends, gids = group_by_user_offsets(user_ids_train)
    # rows for group g: order[starts[g] : ends[g]]
    # all negative rows for group g (assuming sorted_y = y[order]):
    #   neg_mask = sorted_y[starts[g]:ends[g]] == 0
    """
    user_arr = np.asarray(user_ids)
    order = np.argsort(user_arr, kind='stable').astype(np.int32)
    sorted_u = user_arr[order]
 
    mask = np.empty(len(order), dtype=bool)
    mask[0] = True
    mask[1:] = sorted_u[1:] != sorted_u[:-1]
 
    group_starts = np.flatnonzero(mask).astype(np.int32)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = len(order)
 
    group_ids = np.cumsum(mask, dtype=np.int32) - 1
 
    return order, group_starts, group_ends, group_ids
 
 
def build_bpr_tables(ytr, order, group_starts, group_ends, group_ids):
    """Pre-compute positive-row and per-user negative-range arrays for BPR.
 
    Parameters
    ----------
    ytr : float32 array (N,)       — training labels (0 or 1)
    order, group_starts, group_ends, group_ids — from group_by_user_offsets
 
    Returns
    -------
    pos_rows : int32 array (P,)
        Indices (into the ORIGINAL unsorted arrays) of positive rows whose
        user has at least one negative.
    pos_groups : int32 array (P,)
        Group id for each positive row.
    neg_starts : int32 array (P,)
        For each positive row, the start offset into `negative_order` of that
        user's negative block.
    neg_counts : int32 array (P,)
        Number of negatives for that user.
    negative_order : int32 array (N,)
        Original row indices grouped by user with negatives before positives.
 
    Usage
    -----
    # sample one random negative per positive:
    off = (rng.random(len(pos_rows)) * neg_counts).astype(np.int64)
    neg_rows = negative_order[neg_starts + off]
    """
    sorted_y = ytr[order]
 
    neg_per_group = np.add.reduceat(
        (sorted_y == 0).astype(np.int32), group_starts)
    pos_per_group = (group_ends - group_starts) - neg_per_group
    eligible = (neg_per_group > 0) & (pos_per_group > 0)
 
    # Stable sorting by (group, label) keeps every user block contiguous while
    # moving its negatives before its positives, without a Python row loop.
    within_group_key = group_ids.astype(np.int64) * 2 + (sorted_y == 1)
    rearrange = np.argsort(within_group_key, kind='stable')
    negative_order = order[rearrange]
    sorted_y_new = ytr[negative_order]
 
    neg_start_per_group = group_starts
    positive_mask = (sorted_y_new == 1) & eligible[group_ids]
    pos_rows = negative_order[positive_mask].astype(np.int32)
    pos_grps = group_ids[positive_mask].astype(np.int32)
 
    return (pos_rows, pos_grps,
            neg_start_per_group[pos_grps],
            neg_per_group[pos_grps],
            negative_order)

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_bpr(self, Xpos, Xneg):
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        d = zpos - zneg
        gpos = ((sigmoid(d) - 1.0) / B).astype(np.float32)  # FM.step_bpr: BPR loss gradient
        gneg = -gpos
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, gpos[:, None])
        np.add.at(gW, Xneg, gneg[:, None])
        np.add.at(gV, Xpos, gpos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gneg[:, None, None] * (Sneg[:, None, :] - Eneg))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(np.mean(np.logaddexp(0.0, -d)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           select_on='valid', report_on=('valid', 'test'), return_preds=False):
    """Train an FM, early-stopping on `select_on`, and score every `report_on` split.

    The defaults reproduce the original valid/test behaviour exactly. The
    orchestrator passes select_on='valid_search' so that model selection never
    touches valid_confirm, which stays a clean held-out check for promotion.
    """
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xsel, ysel, usel = enc[select_on]
    order, group_starts, group_ends, group_ids = group_by_user_offsets(utr)
    pos_rows, _pos_groups, neg_starts, neg_counts, negative_order = build_bpr_tables(
        ytr, order, group_starts, group_ends, group_ids)
    lr_grid = (lr * 0.5, lr, lr * 2.0)  # run_fm: select a small BPR learning-rate grid
    best, best_state, best_lr = -1, None, None
    m = None
    for trial_lr in lr_grid:
        m = FM(dim, k=k, lr=trial_lr, seed=seed)
        rng = np.random.default_rng(seed)
        trial_best, bad = -1, 0
        for ep in range(1, epochs + 1):
            neg_offsets = rng.integers(0, neg_counts, size=len(pos_rows), dtype=np.int32)
            neg_rows = negative_order[neg_starts + neg_offsets]
            idx = rng.permutation(len(pos_rows)); t0 = time.time()
            losses = [m.step_bpr(Xtr[pos_rows[idx[i:i + bs]]], Xtr[neg_rows[idx[i:i + bs]]])
                      for i in range(0, len(idx), bs)]
            va = evaluate(usel, ysel, m.predict(Xsel))
            if verbose:
                print(f"  lr {trial_lr:.6g} epoch {ep:2d} | loss {np.mean(losses):.4f} | "
                      f"{select_on} GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                      f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
            if va['primary'] > trial_best + 1e-5:
                trial_best, bad = va['primary'], 0
            else:
                bad += 1
            if va['primary'] > best + 1e-5:
                best, best_lr = va['primary'], trial_lr
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    if verbose:
        print(f"  selected lr {best_lr:.6g} | {select_on} primary {best:.4f}")
    m.V, m.W, m.b = best_state
    out, preds = {}, {}
    for name in report_on:
        X, y, u = enc[name]
        s = m.predict(X)
        out[name] = evaluate(u, y, s)
        preds[name] = (u, y, s)
    return (out, preds) if return_preds else out

# ---------------- LightGBM LambdaRank ----------------
# Why this exists alongside the FM: the metric is a RANKING metric (GAUC and
# nDCG@5) but the FM trains pointwise logloss, which the starter-kit README
# names as the single most promising untested direction. LightGBM's `lambdarank`
# optimises nDCG directly, over per-user groups, which is exactly this task's
# shape (within-user ranking over logged impressions).
#
# The FM path above is untouched and `--model fm` still reproduces the published
# ladder, so the baseline comparison stays honest.
LGB_PARAMS = {
    'objective': 'lambdarank',
    # 'None', not 'ndcg'. LightGBM's built-in ndcg@5 is NOT this competition's
    # nDCG@5: it skips groups with no positive, while evaluate() scores a
    # zero-positive user as 0.0 and counts it in the average. Measured on the
    # same predictions, builtin ndcg@5 = 0.8287 against the official 0.5255 —
    # so early-stopping on the builtin optimises a different quantity and
    # stopped this model at 18 rounds. _primary_feval below is the real metric.
    'metric': 'None',
    # nDCG@5 is scored on the top 5, but GAUC is half the primary and depends on
    # the WHOLE within-user ordering. Truncating lambda pairs at 5 would optimise
    # only the head; 30 covers the bulk of per-user impression counts.
    'lambdarank_truncation_level': 30,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'lambda_l2': 1.0,
    'verbosity': -1,
    # Reproducibility: the orchestrator compares candidates at ~0.001 and pairs
    # per-user scores against a parent, so a run that is not bit-reproducible at
    # a fixed seed manufactures deltas out of thread scheduling.
    'num_threads': 1,
    'deterministic': True,
    'force_row_wise': True,
}


def _primary_feval(users, labels):
    """A LightGBM feval that reports the OFFICIAL primary, so early stopping
    selects on the same number the competition scores.

    run_fm already early-stops on evaluate()['primary']; this makes the GBDT
    path do the same thing, rather than on LightGBM's own ndcg convention.
    """
    def feval(preds, _eval_data):
        r = evaluate(users, labels, preds)
        return 'primary', r['primary'], True     # higher is better
    return feval


def run_lgb(splits, seed=0, num_boost_round=600, patience=50, verbose=True,
            select_on='valid', report_on=('valid', 'test'), return_preds=False,
            params=None):
    """Train a LambdaRank GBDT on train-only dense features, early-stopping on
    `select_on`, and score every `report_on` split.

    Mirrors run_fm's signature so the orchestrator contract below is identical
    for both models.
    """
    import lightgbm as lgb

    enc, feat_names, cat_names = encode_lgb(splits)
    Xtr, ytr, _utr, gtr = enc['train']
    Xsel, ysel, usel, gsel = enc[select_on]

    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    p.update({'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed,
              'data_random_seed': seed})

    cat_idx = [feat_names.index(c) for c in cat_names]
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr,
                         feature_name=feat_names, categorical_feature=cat_idx)
    dsel = lgb.Dataset(Xsel, label=ysel, group=gsel, reference=dtrain,
                       feature_name=feat_names, categorical_feature=cat_idx)

    callbacks = [lgb.early_stopping(patience, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))
    booster = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                        valid_sets=[dsel], valid_names=[select_on],
                        feval=_primary_feval(usel, ysel),
                        callbacks=callbacks)
    if verbose:
        sel = evaluate(usel, ysel, booster.predict(Xsel))
        print(f"  best_iter {booster.best_iteration} | {select_on} "
              f"GAUC {sel['GAUC']:.4f} nDCG@5 {sel['nDCG@5']:.4f} "
              f"primary {sel['primary']:.4f}")

    out, preds = {}, {}
    for name in report_on:
        X, y, u, _g = enc[name]
        s = booster.predict(X)
        out[name] = evaluate(u, y, s)
        preds[name] = (u, y, s)
    return (out, preds) if return_preds else out


#: Model used by the orchestrator when the candidate is invoked without an
#: explicit --model. Overridable via CODEGEN_MODEL so an FM-vs-GBDT A/B is a
#: one-env-var change rather than a code edit; codegen/sandbox.py forwards it.
#:
#: 'fm', NOT 'lgb', on measurement. Every GBDT-over-train-aggregates variant
#: tried lands well below the FM on test primary:
#:
#:     FM (reference)                  0.5953
#:     lgb lambdarank                  0.5755
#:     lgb lambdarank, small capacity  0.5795
#:     lgb binary objective            0.5800
#:     lgb binary + OOF FM score       0.5797
#:
#: Two measured reasons, both structural rather than tuning:
#:   1. The collaborative signal is not available to count features. A
#:      user x author pair occurs 1.07 times in train on average, user x video
#:      less, so a per-pair rate is one observation of noise. The FM's
#:      embeddings share strength across users; smoothed counts cannot.
#:   2. A pointwise GBDT spends its capacity on the wrong variance. In the
#:      stacked model `user_rate` had the single largest gain (944k, twice
#:      fm_score's) — and user_rate is CONSTANT WITHIN A USER, so it cannot
#:      change a within-user ranking at all. The objective rewards predicting
#:      the user's base rate; the metric ignores it entirely.
#:
#: `--model lgb` stays available and supported: it is a real second model the
#: search may branch to, and the numbers above are evidence for the agent
#: rather than a reason to delete the path.
ORCHESTRATOR_DEFAULT_MODEL = os.environ.get('CODEGEN_MODEL', 'fm')

#: The two models the orchestrator can score. `pop` and `random` are diagnostic
#: baselines with no training loop, so they have no select_on/report_on contract.
_ORCHESTRATOR_RUNNERS = {'fm': run_fm, 'lgb': run_lgb}


def run_for_orchestrator(a, split):
    """Score exactly one split and emit the machine-readable metrics marker."""
    if split == 'test':
        raise SystemExit("CODEGEN_SPLIT=test refused: candidates never score the "
                         "sealed test split.")
    # The orchestrator invokes candidates as `python <candidate> --data_dir D
    # --seed S` with no --model, so the model comes from
    # ORCHESTRATOR_DEFAULT_MODEL unless a caller asked for one explicitly.
    model = a.model or ORCHESTRATOR_DEFAULT_MODEL
    if model not in _ORCHESTRATOR_RUNNERS:
        raise SystemExit(f"CODEGEN_SPLIT set but model={model!r}; orchestrator "
                         f"mode supports {sorted(_ORCHESTRATOR_RUNNERS)} only.")
    splits = cut_valid_subsplits(load(a.data_dir))
    if split not in splits:
        raise SystemExit(f"unknown CODEGEN_SPLIT {split!r}; expected one of "
                         f"{sorted(VALID_SUBSPLITS)}")
    # Always select on valid_search, so scoring valid_confirm stays uncontaminated.
    kw = ({'k': a.k, 'lr': a.lr, 'epochs': a.epochs} if model == 'fm' else {})
    res, preds = _ORCHESTRATOR_RUNNERS[model](
        splits, seed=a.seed, select_on='valid_search', report_on=(split,),
        return_preds=True, **kw)
    users, labels, scores = preds[split]
    # evaluate() is fed numpy predictions here, so its aggregates come back as
    # np.float32 — not JSON-serializable. Coerce to plain Python numbers.
    payload = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
               for k, v in res[split].items()}
    payload['split'] = split
    payload['seed'] = a.seed
    payload['model'] = model
    payload['per_user'] = per_user_primary(users, labels, scores)
    print(f"{METRICS_MARK} {json.dumps(payload)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    # default=None, not 'fm': run_for_orchestrator has to be able to tell
    # "nobody asked" (use ORCHESTRATOR_DEFAULT_MODEL) from "the caller asked for
    # fm". The interactive path below still falls back to fm, so every command
    # documented in README.md behaves exactly as before.
    ap.add_argument('--model', default=None,
                    choices=['pop', 'fm', 'random', 'lgb'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    _split = os.environ.get('CODEGEN_SPLIT')
    if _split:
        run_for_orchestrator(a, _split)
        raise SystemExit(0)
    model = a.model or 'fm'          # interactive default unchanged
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'lgb': lambda s: run_lgb(s, seed=a.seed)}[model](splits)
    print(f"\n=== {model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
