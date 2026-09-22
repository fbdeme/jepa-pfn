"""Measure the GBM-over-linear headroom of TabICL's OPEN prior, with our methodology, so we
can compare its richness to our own prior deterministically (regenerates the numbers in
docs/current_status.md's TabICL section).

TabICL's prior (soda-inria/tabicl) is open source and generates on CPU. It is the prior that
produced the nanoTabPFN dump we tested earlier (150x5 binary is only a small slice of it).

SETUP (vendor/tabicl is a gitignored throwaway clone; deps are pulled per-run by uv):
    git clone --depth 1 https://github.com/soda-inria/tabicl vendor/tabicl
    echo '# neutered for prior-only import' > vendor/tabicl/src/tabicl/__init__.py   # skip model deps
RUN (n_jobs=1 is REQUIRED -- TabICL's HpSampler closures can't be pickled for multiprocessing):
    uv run --with psutil --with einops --with xgboost --with networkx \
        python -m eval.tabicl_headroom

KEY FINDING (seed 0, 56 tables, 2000 rows, mlp_scm, regression): TabICL's prior is BIMODAL --
only ~29% of tables are learnable (GBM R^2>0.1); the rest are near-noise BY DESIGN. Among the
learnable ones the nonlinearity headroom is ~+0.21 (vs our nonlinear prior's +0.12, toy +0.02),
and the signal only emerges at large row counts (2000, not 300). So TabICL is richer in three
ways at once: more nonlinear where learnable, a much wider difficulty range, and large-data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "vendor/tabicl/src"))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from tabicl.prior import PriorDataset


def _r2(y_tr, y_te, pred):
    var = ((y_te - y_tr.mean()) ** 2).mean()
    return None if var < 1e-8 else 1 - ((y_te - pred) ** 2).mean() / var


def measure(prior_type="mlp_scm", n_tables=56, max_seq_len=2000, seed=0):
    ds = PriorDataset(regression=True, batch_size=8, min_features=4, max_features=20,
                      max_seq_len=max_seq_len, prior_type=prior_type, device="cpu", n_jobs=1)
    G, R, n = [], [], 0
    while n < n_tables:
        X, y, d, sl, _ = ds.get_batch(8)
        X, y = X.cpu().numpy(), y.cpu().numpy()
        d, sl = d.cpu().numpy().ravel(), sl.cpu().numpy().ravel()
        for i in range(X.shape[0]):
            Xi, yi = X[i, :sl[i], :d[i]], y[i, :sl[i]]
            k = int(len(yi) * 0.7)
            if len(yi) < 60 or yi.std() < 1e-6:
                continue
            g = _r2(yi[:k], yi[k:], HistGradientBoostingRegressor(max_iter=300).fit(Xi[:k], yi[:k]).predict(Xi[k:]))
            r = _r2(yi[:k], yi[k:], Ridge().fit(Xi[:k], yi[:k]).predict(Xi[k:]))
            if g is None or r is None:
                continue
            G.append(g); R.append(r); n += 1
    G, R = np.clip(np.array(G), -1, 1), np.clip(np.array(R), -1, 1)
    learn = G > 0.1
    print(f"\n=== TabICL {prior_type}, {max_seq_len} rows, {len(G)} tables ===")
    print(f"  fraction learnable (GBM R2>0.1): {learn.mean():.2f}   (bimodal: rest near-noise by design)")
    print(f"  ALL tables:      median GBM {np.median(G):+.2f}  headroom {np.median(G - R):+.3f}")
    if learn.any():
        print(f"  LEARNABLE only:  median GBM {np.median(G[learn]):+.2f}  headroom {np.median((G - R)[learn]):+.3f}  (n={int(learn.sum())})")
    print("  compare our prior: toy +0.02, nonlinear +0.12 (mostly-learnable, 300 rows)")


if __name__ == "__main__":
    measure("mlp_scm")
