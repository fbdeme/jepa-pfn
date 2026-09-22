"""How hard are the tables a prior generates? (a pre-GPU diagnostic)

A prior is a *teacher*: if its tables are trivially solvable, an ICL model never has to
learn hard in-context reasoning and transfers poorly to real data. This scores prior
difficulty the cheap way -- fit real learners (GBM / ridge / trivial mean) on a within-table
holdout and report the R^2 distribution. Signals of a good teacher, mirroring real tabular data:

  - GBM R^2 spread widely and NOT pinned at ~1.0 (irreducible noise + genuine difficulty)
  - a real GBM-over-linear headroom (nonlinear structure worth learning)
  - a real GBM-over-trivial signal (not pure noise)

Run:  uv run python -m eval.prior_difficulty [config.yaml ...]
      (no arg -> the base_ds prior; extra args = configs whose `prior:` block is scored)
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parents[1]))
from prior.scm import PriorConfig, SCMPrior


def _score_table(x, rng):
    """One regression task per table: last observed column ~ the rest, GBM/ridge/trivial R^2."""
    R, D = x.shape
    if D < 2 or R < 32:
        return None
    y = x[:, -1]
    X = x[:, :-1]
    ok = ~np.isnan(y)
    X, y = np.nan_to_num(X[ok]), y[ok]
    if len(y) < 32 or y.std() < 1e-6:
        return None
    n_tr = int(len(y) * 0.7)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    var = ((y[te] - y[tr].mean()) ** 2).mean()  # trivial-predictor MSE (=Var on test vs train mean)
    if var < 1e-8:
        return None
    r2 = lambda p: 1.0 - ((y[te] - p) ** 2).mean() / var
    gbm = HistGradientBoostingRegressor(max_iter=200, max_depth=4).fit(X[tr], y[tr])
    rdg = Ridge().fit(X[tr], y[tr])
    return dict(gbm=r2(gbm.predict(X[te])), ridge=r2(rdg.predict(X[te])))


def score_prior(prior_cfg, n_tables=200, seed=0):
    prior = SCMPrior(PriorConfig(**prior_cfg), seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n_tables:
        b = prior.sample_batch(8)
        for x in b.x:
            s = _score_table(x, rng)
            if s is not None:
                rows.append(s)
    gbm = np.array([r["gbm"] for r in rows])
    rdg = np.array([r["ridge"] for r in rows])
    return dict(n=len(rows), gbm=gbm, ridge=rdg)


def _report(name, s):
    g, r = np.clip(s["gbm"], -1, 1), np.clip(s["ridge"], -1, 1)
    print(f"\n=== {name}  (n={s['n']} tables) ===")
    print(f"  GBM   R^2   median {np.median(g):+.3f}   mean {g.mean():+.3f}   "
          f"[p10 {np.percentile(g,10):+.3f}, p90 {np.percentile(g,90):+.3f}]")
    print(f"  Ridge R^2   median {np.median(r):+.3f}   mean {r.mean():+.3f}")
    print(f"  GBM-over-ridge headroom (nonlinearity worth learning):  median {np.median(g-r):+.3f}")
    print(f"  fraction of tables GBM 'solves' (R^2>0.95):  {(g>0.95).mean():.2f}   "
          f"(want << 1.0; real tabular data is rarely this easy)")


def main():
    cfgs = sys.argv[1:]
    if not cfgs:
        base = yaml.safe_load(open(Path(__file__).parents[1] / "configs/base_ds.yaml"))
        _report("base_ds prior (current 'toy')", score_prior(base.get("prior", {})))
        return
    for cfg_path in cfgs:
        cfg = yaml.safe_load(open(cfg_path))
        _report(Path(cfg_path).stem, score_prior(cfg.get("prior", {})))


if __name__ == "__main__":
    main()
