"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, json, os, time
import numpy as np
from data import load, encode, FIELDS
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
    Xtr, ytr, _ = enc['train']; Xsel, ysel, usel = enc[select_on]
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(usel, ysel, m.predict(Xsel))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | {select_on} GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    out, preds = {}, {}
    for name in report_on:
        X, y, u = enc[name]
        s = m.predict(X)
        out[name] = evaluate(u, y, s)
        preds[name] = (u, y, s)
    return (out, preds) if return_preds else out

def run_for_orchestrator(a, split):
    """Score exactly one split and emit the machine-readable metrics marker."""
    if split == 'test':
        raise SystemExit("CODEGEN_SPLIT=test refused: candidates never score the "
                         "sealed test split.")
    if a.model != 'fm':
        raise SystemExit(f"CODEGEN_SPLIT set but --model={a.model}; orchestrator "
                         "mode supports fm only.")
    splits = cut_valid_subsplits(load(a.data_dir))
    if split not in splits:
        raise SystemExit(f"unknown CODEGEN_SPLIT {split!r}; expected one of "
                         f"{sorted(VALID_SUBSPLITS)}")
    # Always select on valid_search, so scoring valid_confirm stays uncontaminated.
    res, preds = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                        select_on='valid_search', report_on=(split,),
                        return_preds=True)
    users, labels, scores = preds[split]
    # evaluate() is fed numpy predictions here, so its aggregates come back as
    # np.float32 — not JSON-serializable. Coerce to plain Python numbers.
    payload = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
               for k, v in res[split].items()}
    payload['split'] = split
    payload['seed'] = a.seed
    payload['per_user'] = per_user_primary(users, labels, scores)
    print(f"{METRICS_MARK} {json.dumps(payload)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    _split = os.environ.get('CODEGEN_SPLIT')
    if _split:
        run_for_orchestrator(a, _split)
        raise SystemExit(0)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
