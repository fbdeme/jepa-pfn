"""Shape and noise census of the real-data suite, for matching the prior to real tables.

Reads the OpenML datasets already cached on this machine (~/.cache/openml, populated by
eval/suite_bench.py) and restricts to the datasets in eval/results_suite_full.json. Per dataset:
rows, features (target excluded), categorical/numeric split, and, for regression, the noise
fraction 1 - R^2(HistGB) from the suite results with its equivalent sigma under the prior's
convention x = f + sigma * eps with Var f = 1 (noise fraction = sigma^2 / (1 + sigma^2)).

Run: uv run --with openml --with pyarrow python -m eval.realdata_shape_census
Writes eval/results_realdata_shapes.json
"""
import glob
import json
import math
import os
import pickle
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_realdata_shapes.json"
CACHE = os.path.expanduser("~/.cache/openml/org/openml/www/datasets")


def deciles(xs):
    q = st.quantiles(xs, n=10)
    return {"p10": q[0], "p30": q[2], "p50": q[4], "p70": q[6], "p90": q[8], "min": min(xs), "max": max(xs), "n": len(xs)}


def structure(df, feats, max_rows=4000, seed=0):
    """Categorical axis: category counts per categorical column. K axis: intrinsic dimension of the
    standardised numeric block -- PCA components for 90% variance (K90), K90 / D, and the effective
    rank exp(H(eigenvalues)) of the correlation matrix. Rows subsampled for speed, NaNs mean-imputed."""
    import numpy as np
    cat_cols = [a for a, c in feats if c]
    num_cols = [a for a, c in feats if not c]
    out = {"cat_n_categories": [int(df[a].nunique(dropna=True)) for a in cat_cols]}
    X = df[num_cols].apply(lambda col: __import__("pandas").to_numeric(col, errors="coerce")).to_numpy(dtype=float)
    if X.shape[1] >= 2:
        rng = np.random.default_rng(seed)
        if X.shape[0] > max_rows:
            X = X[rng.choice(X.shape[0], max_rows, replace=False)]
        mu = np.nanmean(X, 0)
        X = np.where(np.isnan(X), mu, X)
        sd = X.std(0)
        keep = sd > 1e-9
        X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
        if X.shape[1] >= 2:
            ev = np.linalg.svd(X, compute_uv=False) ** 2
            ev = ev / ev.sum()
            k90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
            p = ev[ev > 0]
            out.update({"k90": k90, "k90_over_d": round(k90 / X.shape[1], 3),
                        "erank": round(float(np.exp(-(p * np.log(p)).sum())), 2), "d_numeric_used": int(X.shape[1])})
    return out


def main():
    suite = json.loads((ROOT / "eval/results_suite_full.json").read_text())
    by_name = {}
    for k, v in suite.items():
        bench, name, task = k.split("|")
        by_name[name] = (bench, task, v)
    rows = []
    for d in sorted(glob.glob(CACHE + "/*/")):
        desc = Path(d, "description.xml")
        pk = glob.glob(d + "dataset_*.pkl.py3")
        if not desc.exists() or not pk:
            continue
        meta = dict(re.findall(r"<oml:(name|default_target_attribute|id)>([^<]*)<", desc.read_text()))
        name = meta.get("name")
        if name not in by_name:
            continue
        df, cat, attrs = pickle.load(open(pk[0], "rb"))
        target = meta.get("default_target_attribute", "")
        feats = [(a, c) for a, c in zip(attrs, cat) if a != target]
        bench, task, res = by_name[name]
        rec = {"id": int(meta["id"]), "name": name, "bench": bench, "task": task,
               "n_rows": int(df.shape[0]), "n_features": len(feats),
               "n_categorical": int(sum(c for _, c in feats)), "n_numeric": int(sum(not c for _, c in feats))}
        rec.update(structure(df, feats))
        if task == "reg" and "histgbr" in res:
            nf = min(max(1 - res["histgbr"]["r2"], 0.0), 0.999)
            rec["noise_fraction_histgb"] = round(nf, 4)
            rec["sigma_equiv"] = round(math.sqrt(nf / (1 - nf)), 3)
        if task == "clf" and "histgb" in res:
            rec["error_histgb"] = round(1 - res["histgb"]["acc"], 4)
        rows.append(rec)
    out = {"n_datasets": len(rows), "missing_from_cache": sorted(set(by_name) - {r["name"] for r in rows}),
           "rows": rows,
           "summary": {"n_rows": deciles([r["n_rows"] for r in rows]),
                       "n_features": deciles([r["n_features"] for r in rows]),
                       "categorical_fraction": deciles([r["n_categorical"] / max(r["n_features"], 1) for r in rows]),
                       "frac_features_le_32": round(sum(r["n_features"] <= 32 for r in rows) / len(rows), 3),
                       "frac_features_le_8": round(sum(r["n_features"] <= 8 for r in rows) / len(rows), 3),
                       "frac_rows_le_1024": round(sum(r["n_rows"] <= 1024 for r in rows) / len(rows), 3),
                       "frac_rows_le_384": round(sum(r["n_rows"] <= 384 for r in rows) / len(rows), 3),
                       "noise_fraction_reg": deciles([r["noise_fraction_histgb"] for r in rows if "noise_fraction_histgb" in r]),
                       "sigma_equiv_reg": deciles([r["sigma_equiv"] for r in rows if "sigma_equiv" in r]),
                       "error_clf": deciles([r["error_histgb"] for r in rows if "error_histgb" in r]),
                       "k90": deciles([r["k90"] for r in rows if "k90" in r]),
                       "k90_over_d": deciles([r["k90_over_d"] for r in rows if "k90_over_d" in r]),
                       "erank": deciles([r["erank"] for r in rows if "erank" in r]),
                       "cat_n_categories": deciles([n for r in rows for n in r["cat_n_categories"]]),
                       "frac_datasets_with_categorical": round(sum(r["n_categorical"] > 0 for r in rows) / len(rows), 3)},
           "prior_reference": {"scm_base": {"n_cols": [4, 32], "n_rows": [128, 384], "sigma_per_node": "log-uniform [0.01, 0.3] x noise_scale",
                                            "noise_fraction_max_at_scale1": round(0.3 ** 2 / (1 + 0.3 ** 2), 4),
                                            "p_categorize": 0.2, "n_categories": [2, 5], "p_missing_table": 0.3},
                               "factor": {"n_cols": [12, 12], "n_rows": [128, 256], "sigma": 1.0, "noise_fraction": 0.5}}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    print("missing from cache:", out["missing_from_cache"])


if __name__ == "__main__":
    main()
