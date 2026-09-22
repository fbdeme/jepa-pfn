"""Does the real-data-matched prior generate tables that look like the real suite? (gate before training)

Generates N tables from prior/real.py at the row cap and measures them with the SAME code the census
used on the real datasets (eval/realdata_shape_census.structure: K90/D, effective rank, category
counts) plus the suite's noise protocol (HistGB on a random continuous non-root column, 50/50 split,
noise fraction = 1 - R^2). Compares deciles axis by axis against eval/results_realdata_shapes.json
within stated tolerances; exit 1 on any FAIL. Also tabulates the generative rho against the measured
K90/D (the calibration the plan asks for).

Run: OMP_NUM_THREADS=1 N_TABLES=600 uv run --with scikit-learn --with pandas python -m eval.real_prior_check   (OMP=1: HistGB 0.1 s/fit; with default threads on a loaded box it is 100x slower)
Writes eval/results_real_prior_check.json
"""
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.realdata_shape_census import structure          # noqa: E402
from prior.real import RealConfig, RealPrior              # noqa: E402
from prior.scm import FuncFamily                          # noqa: E402

OUT = ROOT / "eval/results_real_prior_check.json"
N_TABLES = int(os.environ.get("N_TABLES", "200"))
SEED = 123
TOL = {"n_features": ("ratio", 1.3), "categorical_fraction": ("abs", 0.10), "cat_n_categories": ("abs", 1.0),
       "k90_over_d": ("abs", 0.10), "erank": ("ratio", 1.3), "noise_fraction_reg": ("abs", 0.10)}
DECILES = ("p10", "p30", "p50", "p70", "p90")


def deciles(xs):
    q = st.quantiles(xs, n=10)
    return {"p10": q[0], "p30": q[2], "p50": q[4], "p70": q[6], "p90": q[8], "n": len(xs)}


def noise_fraction(x, cat, fam_obs, rng, prefer=None):
    """Suite protocol on a generated table: HistGB regression of one continuous non-root column on the
    rest, half/half split, 1 - R^2 on the held-out half (clipped to [0, 1]). `prefer` (bool mask over
    columns) restricts the candidates when any qualify -- used for the OUTCOME protocol (a column with
    at least one observed parent), the closer analogue of a benchmark's curated target column; the
    random-column protocol is reported alongside."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cand = np.flatnonzero((cat == 0) & (fam_obs != int(FuncFamily.ROOT)))
    if prefer is not None:                       # strict: no silent fallback to a random column
        cand = np.flatnonzero(prefer & (cat == 0) & (fam_obs != int(FuncFamily.ROOT)))
    if len(cand) == 0 or x.shape[1] < 2:
        return None
    t = int(rng.choice(cand))
    y = x[:, t]; X = np.delete(x, t, 1)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    n = len(y) // 2
    m = HistGradientBoostingRegressor(random_state=0).fit(X[:n], y[:n])
    var = ((y[n:] - y[:n].mean()) ** 2).mean()
    if var < 1e-8:
        return None
    r2 = 1 - ((y[n:] - m.predict(X[n:])) ** 2).mean() / var
    return float(min(max(1 - r2, 0.0), 1.0))


def main():
    real = json.loads((ROOT / "eval/results_realdata_shapes.json").read_text())
    cfg = RealConfig(n_rows=(2048, 2048), max_cells=1 << 22)   # distribution check: 2048 rows (HistGB's own fit error
                                                                 # then matches the census, which trained on thousands
                                                                 # of rows); the training cell budget is irrelevant here
    prior = RealPrior(cfg, seed=SEED)
    rng = np.random.default_rng(SEED)
    gen = []
    for i in range(N_TABLES):
        b = prior.sample_batch(1)
        x, cat, h = b.x[0], b.categorical[0], b.hyper[0]
        fam_obs = b.func_family[0][b.observed_idx[0]]
        tc = h.get("target_col")
        feat = [d for d in range(x.shape[1]) if d != tc]        # the census measures FEATURES (target excluded)
        cols = [f"c{d}" for d in feat]
        df = pd.DataFrame(x[:, feat], columns=cols)
        rec = {"D": len(feat), "rho": h["rho"], "K": h["K"], "sigma_table": h["sigma_table"],
               "cat_frac": float((cat[feat] > 0).mean()), "source": h["source"]}
        rec.update(structure(df, [(c, bool(cat[d] > 0)) for d, c in zip(feat, cols)]))
        rec["cat_n_categories"] = [int(cat[d]) for d in feat if cat[d] > 0]
        K = h["K"]
        obs_parents = b.adjacency[0][K:, :][:, b.observed_idx[0]].sum(0)      # observed parents per column
        tmask = np.zeros(x.shape[1], bool)
        if tc is not None:
            tmask[tc] = True
        nf = noise_fraction(x, cat, fam_obs, rng, prefer=tmask) if tmask.any() else None
        nr = noise_fraction(x, cat, fam_obs, rng)
        if nf is not None:
            rec["noise_fraction"] = round(nf, 4)          # designated-target protocol (verdict), like the census
            if h.get("target_sigma") is not None:
                ts = h["target_sigma"]
                rec["noise_fraction_theory"] = round(ts * ts / (1 + ts * ts), 4)   # sigma^2/(1+sigma^2), Var f = 1
        if nr is not None:
            rec["noise_fraction_random"] = round(nr, 4)
        gen.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_TABLES}", flush=True)
    # real-side deciles under the same caps the prior applies
    rows = real["rows"]
    real_d = {"n_features": deciles([min(r["n_features"], cfg.max_cols) for r in rows]),
              "categorical_fraction": deciles([r["n_categorical"] / max(r["n_features"], 1) for r in rows]),
              "cat_n_categories": deciles([min(n, cfg.max_categories) for r in rows for n in r["cat_n_categories"] if n >= 2]),
              "k90_over_d": deciles([r["k90_over_d"] for r in rows if "k90_over_d" in r]),
              "erank": deciles([r["erank"] for r in rows if "erank" in r]),
              "noise_fraction_reg": deciles([r["noise_fraction_histgb"] for r in rows if "noise_fraction_histgb" in r])}
    gen_d = {"n_features": deciles([g["D"] for g in gen]),
             "categorical_fraction": deciles([g["cat_frac"] for g in gen]),
             "cat_n_categories": deciles([n for g in gen for n in g["cat_n_categories"]] or [2, 2, 2]),
             "k90_over_d": deciles([g["k90_over_d"] for g in gen if "k90_over_d" in g]),
             "erank": deciles([g["erank"] for g in gen if "erank" in g]),
             "noise_fraction_reg": deciles([g["noise_fraction"] for g in gen if "noise_fraction" in g])}
    gen_d["noise_fraction_random_column"] = deciles([g["noise_fraction_random"] for g in gen if "noise_fraction_random" in g])
    verdict, fails = {}, []
    for axis, (kind, tol) in TOL.items():
        bad = []
        for q in DECILES:
            a, b = real_d[axis][q], gen_d[axis][q]
            ok = (abs(a - b) <= tol) if kind == "abs" else (max(a, b) / max(min(a, b), 1e-9) <= tol)
            if not ok:
                bad.append(q)
        verdict[axis] = {"pass": not bad, "failed_deciles": bad, "real": {q: round(real_d[axis][q], 3) for q in DECILES},
                         "generated": {q: round(gen_d[axis][q], 3) for q in DECILES}, "tolerance": f"{kind} {tol}"}
        if bad:
            fails.append(axis)
    # calibration: generative rho vs measured K90/D, by rho quintile
    rk = [(g["rho"], g["k90_over_d"]) for g in gen if "k90_over_d" in g]
    edges = st.quantiles([r for r, _ in rk], n=5)
    cal = []
    for lo, hi in zip([0] + edges, edges + [1.01]):
        sel = [k for r, k in rk if lo <= r < hi]
        rs = [r for r, _ in rk if lo <= r < hi]
        if sel:
            cal.append({"rho_lo": round(lo, 3), "rho_hi": round(hi, 3), "rho_mean": round(st.mean(rs), 3),
                        "k90_over_d_mean": round(st.mean(sel), 3), "n": len(sel)})
    # coverage: generated mass per REAL quintile (the stratified analysis needs every real bin populated)
    coverage = {}
    for axis, rk, gk in (("k90_over_d", "k90_over_d", "k90_over_d"), ("noise_fraction_reg", "noise_fraction_histgb", "noise_fraction")):
        rv = sorted(r[rk] for r in rows if rk in r); gv = [g[gk] for g in gen if gk in g]
        edges = st.quantiles(rv, n=5)
        bins = [0] + edges + [float("inf")]
        coverage[axis] = {"real_quintile_edges": [round(e, 3) for e in edges],
                          "generated_mass": [round(sum(lo <= v < hi for v in gv) / len(gv), 3) for lo, hi in zip(bins[:-1], bins[1:])]}
    gaps = [g["noise_fraction"] - g["noise_fraction_theory"] for g in gen if "noise_fraction_theory" in g]
    noise_model_error = ({"median_measured_minus_theory": round(st.median(gaps), 4), "p10": round(st.quantiles(gaps, n=10)[0], 4),
                          "p90": round(st.quantiles(gaps, n=10)[8], 4), "n": len(gaps)} if gaps else None)
    if noise_model_error:
        print("noise: measured - theory (HistGB fit error) median/p10/p90:", noise_model_error)
    out = {"n_tables": N_TABLES, "seed": SEED, "config": cfg.__dict__, "verdict": verdict, "pass": not fails,
           "coverage": coverage, "noise_model_error": noise_model_error,
           "noise_fraction_random_column": {q: round(gen_d["noise_fraction_random_column"][q], 3) for q in DECILES},
           "calibration_rho_vs_k90_over_d": cal, "generated": gen}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'axis':22} {'tol':10} " + " ".join(f"{q:>13}" for q in DECILES))
    for axis, v in verdict.items():
        print(f"{axis:22} {v['tolerance']:10} " + " ".join(f"{v['real'][q]:6.3f}/{v['generated'][q]:6.3f}" for q in DECILES)
              + ("   PASS" if v["pass"] else f"   FAIL {v['failed_deciles']}"))
    print("noise fraction, random-column protocol:", {q: round(gen_d["noise_fraction_random_column"][q], 3) for q in DECILES})
    print("calibration rho -> K90/D:", [(c["rho_mean"], c["k90_over_d_mean"]) for c in cal])
    print("coverage (generated mass per real quintile, 0.2 each ideal):", {k: v["generated_mass"] for k, v in coverage.items()})
    print("PASS" if not fails else f"FAIL: {fails}", "-> wrote", OUT.relative_to(ROOT))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
