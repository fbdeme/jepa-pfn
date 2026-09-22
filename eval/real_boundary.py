"""$0 oracle map for the real-data-matched prior (docs/real_prior_plan.md 10.3), run BEFORE training.

Per table: latent-advantage index = median_d (R2_f[d] - R2_x[d]) with the SAME cross-column ridge as
eval/factor_oracle.py (clean f_d vs noisy x_d recovered from the other columns' x, first-half fit /
second-half score). Stored next to the table's generator truth (hyper: rho, sigma, cat_frac, D, R, ...)
so the expected latent-favourable region is pinned down before any checkpoint exists. Also stores the
designated target column's own (R2_f - R2_x), since that column is what the suite predicts.

Deterministic: seed 777 (training uses 0/1/2), batch 1 per table so every table draws its own (D, R).
Run: OMP_NUM_THREADS=1 uv run --with scikit-learn python -m eval.real_boundary
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from prior.real import RealConfig, RealPrior

ROOT = Path(__file__).parents[1]
N_TABLES = int(os.environ.get("N_TABLES", 2000))
SEED = 777
AXES = ["rho", "sigma_mean", "cat_frac", "D", "R"]   # the five stratification axes of the plan


def table_index(X, F):
    """Per-column (R2_f - R2_x) list for one table; NaN cells filled with the fit-half column mean."""
    R, D = X.shape
    half = R // 2
    X = X.copy()
    mu = np.nanmean(X[:half], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X = np.where(np.isnan(X), mu, X)
    out = []
    for d in range(D):
        others = [j for j in range(D) if j != d]
        Xtr, Xte = X[:half][:, others], X[half:][:, others]
        r2 = []
        for tgt in (F, X):
            m = Ridge(alpha=1.0).fit(Xtr, tgt[:half, d])
            r2.append(r2_score(tgt[half:, d], m.predict(Xte)))
        out.append((r2[0], r2[1]))
    return out


def strata(rows, axis, key="index"):
    """Median index per quintile of `axis` (quintile edges from the tables themselves)."""
    v = np.array([r[axis] for r in rows], float)
    idx = np.array([r[key] for r in rows], float)
    edges = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, v, side="right")
    return [{"q": int(i), "lo": float(v[q == i].min()), "hi": float(v[q == i].max()),
             "n": int((q == i).sum()), "median_index": round(float(np.median(idx[q == i])), 4)}
            for i in range(5) if (q == i).any()]


def main():
    prior = RealPrior(RealConfig(), seed=SEED)
    rows = []
    for t in range(N_TABLES):
        bt = prior.sample_batch(1)
        h = bt.hyper[0]
        per_col = table_index(bt.x[0].astype(np.float64), bt.f[0].astype(np.float64))
        diffs = [a - b for a, b in per_col]
        rec = {k: h[k] for k in ("rho", "K", "D", "R", "sigma_table", "sigma_mean", "cat_frac",
                                 "edge_density_obs", "target_col", "target_sigma")}
        rec.update(index=round(float(np.median(diffs)), 4),
                   r2_f=round(float(np.median([a for a, _ in per_col])), 4),
                   r2_x=round(float(np.median([b for _, b in per_col])), 4))
        if h["target_col"] is not None:
            a, b = per_col[h["target_col"]]
            rec.update(target_r2_f=round(float(a), 4), target_r2_x=round(float(b), 4), target_index=round(float(a - b), 4))
        rows.append(rec)
        if (t + 1) % 200 == 0:
            print(f"{t + 1}/{N_TABLES} median index so far {np.median([r['index'] for r in rows]):.3f}", flush=True)

    idx = np.array([r["index"] for r in rows])
    tidx = np.array([r["target_index"] for r in rows if "target_index" in r])
    out = {
        "n_tables": N_TABLES, "seed": SEED, "prior": "RealConfig() defaults (= gate-passing settings)",
        "note": "index = median_d(R2_f - R2_x), cross-column ridge alpha=1, first-half fit / second-half score (eval/factor_oracle.py definition, per table)",
        "summary": {"median_index": round(float(np.median(idx)), 4), "mean_index": round(float(idx.mean()), 4),
                    "frac_index_gt_0.15": round(float((idx > 0.15).mean()), 4),
                    "median_r2_f": round(float(np.median([r["r2_f"] for r in rows])), 4),
                    "median_r2_x": round(float(np.median([r["r2_x"] for r in rows])), 4),
                    "n_with_target": int(len(tidx)), "median_target_index": round(float(np.median(tidx)), 4) if len(tidx) else None},
        "strata": {ax: strata(rows, ax) for ax in AXES},
        "strata_target": {ax: strata([r for r in rows if "target_index" in r], ax, "target_index") for ax in AXES},
        "tables": rows,
    }
    p = ROOT / "eval" / "results_real_boundary.json"
    p.write_text(json.dumps(out, indent=1))
    print("summary", json.dumps(out["summary"]))
    for ax in AXES:
        print(ax, [(s["q"], round(s["lo"], 2), round(s["hi"], 2), s["median_index"]) for s in out["strata"][ax]])
    print("wrote", p)


if __name__ == "__main__":
    main()
