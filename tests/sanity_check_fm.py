"""Baseline sanity check: run baseline.py --model fm for 5 seeds, average the
result, and compare that MEAN against baseline_scores.json's 5-seed mean within a
tolerance derived from the reported std of 0.0008.

Slow (~40s x 5 seeds). Not part of the default pytest run — invoke directly:
    python tests/sanity_check_fm.py
"""
import json
import sys

from data import load
from baseline import run_fm

SEEDS = (0, 1, 2, 3, 4)
STD = 0.0008
TOLERANCE_SIGMAS = 2.5  # matches the repo's own convergence-rule sigma multiple


def main():
    with open('baseline_scores.json', encoding='utf-8') as fh:
        expected = json.load(fh)['scores']['fm_official']['test']['primary']

    splits = load('./KuaiRand-Pure/data')
    primaries = []
    for seed in SEEDS:
        res = run_fm(splits, seed=seed, verbose=False)
        p = res['test']['primary']
        primaries.append(p)
        print(f"  seed {seed}: test primary = {p:.4f}")

    mean = sum(primaries) / len(primaries)
    tol = TOLERANCE_SIGMAS * STD / (len(SEEDS) ** 0.5)
    diff = abs(mean - expected)
    print(f"\n5-seed mean = {mean:.4f} | expected = {expected:.4f} | "
          f"diff = {diff:.4f} | tolerance = {tol:.4f}")

    if diff > tol:
        print("FAIL: baseline.py's FM output has drifted from baseline_scores.json")
        sys.exit(1)
    print("OK: baseline.py's FM output matches baseline_scores.json within tolerance")


if __name__ == '__main__':
    main()
