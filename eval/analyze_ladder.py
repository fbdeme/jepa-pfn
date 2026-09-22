"""Scale-ladder table: base (6.6M) vs big (35M) on the SAME datasets/splits.

Merges results_suite_full.json (base arms + baselines) with results_suite_big.json
(big arms + baselines). Selection + splits are deterministic in suite_bench, so the
two runs share dataset keys and identical baseline columns -> directly comparable.
Reports per-kind mean rank over {base_ds,dual,lat_s, big_ds,dual,lat_s, linear, gbm}
and the headline: does the data-space >= latent gap WIDEN from base to big?

Run: python -m eval.analyze_ladder
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
base = json.load(open(ROOT / "eval/results_suite_full.json"))
big = json.load(open(ROOT / "eval/results_suite_big.json"))

BASE = ["base_ds", "base_dual", "base_lat_s"]
BIG = ["big_ds", "big_dual", "big_lat_s"]
LIN = {"clf": "logreg", "reg": "ridge"}
GBM = {"clf": "histgb", "reg": "histgbr"}


def merged(kind, metric):
    """Yield per-dataset {model: value} for datasets present (with all models) in both."""
    for k, bv in base.items():
        if k.split("|")[2] != kind or k not in big:
            continue
        gv = big[k]
        row = {}
        ok = True
        for m in BASE + [LIN[kind], GBM[kind]]:
            if m not in bv or bv[m].get(metric) is None:
                ok = False; break
            row[m] = bv[m][metric]
        for m in BIG:
            if m not in gv or gv[m].get(metric) is None:
                ok = False; break
            row[m] = gv[m][metric]
        if ok:
            yield row


def report(kind, metric):
    models = BASE + BIG + [LIN[kind], GBM[kind]]
    per = collections.defaultdict(list)
    ranks = collections.defaultdict(list)
    ds_ge_dual_base = ds_ge_dual_big = 0
    gap_base, gap_big = [], []          # ds_acc - dual_acc, per dataset
    n = 0
    for row in merged(kind, metric):
        n += 1
        for m in models:
            per[m].append(row[m])
        for i, m in enumerate(sorted(models, key=lambda m: row[m], reverse=True)):
            ranks[m].append(i + 1)
        ds_ge_dual_base += row["base_ds"] >= row["base_dual"]
        ds_ge_dual_big += row["big_ds"] >= row["big_dual"]
        gap_base.append(row["base_ds"] - row["base_dual"])
        gap_big.append(row["big_ds"] - row["big_dual"])
    print(f"\n### {kind}  (n={n} shared datasets, metric={metric})")
    print(f"{'model':12} {'mean':>8} {'median':>8} {'mean_rank':>10}")
    for m in models:
        a = np.array(per[m], float)
        print(f"{m:12} {a.mean():8.3f} {np.median(a):8.3f} {np.mean(ranks[m]):10.2f}")
    print(f"  ds >= dual   base: {ds_ge_dual_base}/{n} = {ds_ge_dual_base/n:.2f}"
          f"   big: {ds_ge_dual_big}/{n} = {ds_ge_dual_big/n:.2f}")
    print(f"  mean(ds - dual)  base: {np.mean(gap_base):+.3f}   big: {np.mean(gap_big):+.3f}"
          f"   -> gap {'WIDENS' if np.mean(gap_big) > np.mean(gap_base) else 'shrinks'} with scale")


report("clf", "acc")
report("reg", "r2")
