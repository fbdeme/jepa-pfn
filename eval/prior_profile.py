"""Prior profile: the same task-level statistics for every prior we can train on, so a prior change is measured before
a rental (docs/prior_v2_plan.md). Mirrors the summary vector of "Towards Evaluating Data Priors" (arXiv 2606.29241,
Appx C) and the GBM-over-linear headroom of eval/tabicl_headroom.py.

Per table (target = the prior's designated target column, or a random column when the prior has none):
  regression targets  : ridge R2, HistGB R2 (70/30 row split), headroom = GBM - ridge, learnable = GBM R2 > 0.1
  classification      : majority / logreg / HistGB accuracy, n_classes
  table geometry      : feature count, rows, categorical fraction, mean |corr|, effective-rank ratio, missing fraction
Aggregates per prior = medians / fractions. Writes eval/results_prior_profile.json.

Run:  uv run python -m eval.prior_profile [--n 40] [--rows 512] [--priors paper_scm,real,tabicl1,tabicl2]
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).parents[1]
warnings.filterwarnings("ignore")


def build(name, rows, seed):
    if name == "paper_scm":                                   # base_ds / paper prior at the profiling row count
        from prior.scm import PriorConfig, SCMPrior
        return SCMPrior(PriorConfig(n_rows=(rows, rows), n_cols=(4, 32)), seed=seed)
    if name == "real":
        from prior.real import RealConfig, RealPrior
        return RealPrior(RealConfig(n_rows=(rows, rows), max_cells=10 ** 9), seed=seed)
    if name in ("tabicl1", "tabicl2"):
        from prior.tabicl import TabICLConfig, TabICLPrior
        return TabICLPrior(TabICLConfig(prior_type="mix_scm" if name == "tabicl1" else "graph_scm", task="mixed", p_cls=0.5,
                                        n_rows=(rows, rows), max_cells=10 ** 9), seed=seed)
    raise ValueError(name)


def _r2(y_tr, y_te, pred):
    var = ((y_te - y_tr.mean()) ** 2).mean()
    return None if var < 1e-8 else float(1 - ((y_te - pred) ** 2).mean() / var)


def profile_table(x, target_col, n_classes, rng):
    feats = np.delete(x, target_col, axis=1)
    y = x[:, target_col]
    keep = ~np.isnan(y)
    feats, y = feats[keep], y[keep]
    med = np.nanmedian(feats, 0)
    X = np.where(np.isnan(feats), np.nan_to_num(med), feats)           # median-impute for the probes only
    R, Dm = X.shape
    k = int(R * 0.7)
    out = dict(n_feat=int(Dm), rows=int(R), missing_frac=float(np.isnan(feats).mean()))
    with np.errstate(all="ignore"):
        c = np.corrcoef(X.T) if Dm > 1 else np.ones((1, 1))
        c = np.nan_to_num(c)
        off = np.abs(c[~np.eye(Dm, dtype=bool)]) if Dm > 1 else np.array([0.0])
        ev = np.clip(np.linalg.eigvalsh(c), 0, None); p = ev / max(ev.sum(), 1e-12)
        out["mean_abs_corr"] = float(off.mean())
        out["erank_ratio"] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) / Dm)
    if n_classes:
        yi = y.astype(int)
        out["task"] = "cls"; out["n_classes"] = int(n_classes)
        out["majority_acc"] = float(np.bincount(yi[k:], minlength=1).max() / max(R - k, 1)) if R > k else None
        if len(np.unique(yi[:k])) >= 2 and R - k >= 5:
            out["logreg_acc"] = float((LogisticRegression(max_iter=300).fit(X[:k], yi[:k]).predict(X[k:]) == yi[k:]).mean())
            out["gbm_acc"] = float((HistGradientBoostingClassifier(max_iter=200).fit(X[:k], yi[:k]).predict(X[k:]) == yi[k:]).mean())
    else:
        out["task"] = "reg"
        if y[:k].std() > 1e-6 and R - k >= 5:
            g = _r2(y[:k], y[k:], HistGradientBoostingRegressor(max_iter=200).fit(X[:k], y[:k]).predict(X[k:]))
            r = _r2(y[:k], y[k:], Ridge().fit(X[:k], y[:k]).predict(X[k:]))
            if g is not None and r is not None:
                out.update(gbm_r2=float(np.clip(g, -1, 1)), ridge_r2=float(np.clip(r, -1, 1)))
    return out


def run(names, n_tables, rows, seed):
    res = {}
    for name in names:
        t0 = time.time()
        prior, rng = build(name, rows, seed), np.random.default_rng(seed)
        tables, cat_fracs = [], []
        while len(tables) < n_tables:
            b = prior.sample_batch(2)
            for i in range(b.x.shape[0]):
                h = (b.hyper or [None] * b.x.shape[0])[i] or {}
                t = h.get("target_col") if h.get("target_col") is not None else int(rng.integers(b.x.shape[2]))
                k = int(h.get("n_classes") or 0)
                if not k and b.categorical[i, t]:                    # a categorical column drawn as target counts as a class task
                    k = int(b.categorical[i, t])
                    col = b.x[i, :, t]; codes = {v: j for j, v in enumerate(np.unique(col[~np.isnan(col)]))}
                    b.x[i, :, t] = np.array([codes.get(v, np.nan) for v in col])
                tables.append(profile_table(b.x[i], t, k, rng))
                cat_fracs.append(float((np.delete(b.categorical[i], t) > 0).mean()))
                if len(tables) >= n_tables: break
        reg = [x for x in tables if x["task"] == "reg" and "gbm_r2" in x]
        cls = [x for x in tables if x["task"] == "cls" and "gbm_acc" in x]
        med = lambda xs, key: float(np.median([x[key] for x in xs])) if xs else None
        agg = dict(n_tables=len(tables), n_reg=len(reg), n_cls=len(cls), rows=rows, seconds=round(time.time() - t0, 1),
                   n_feat_median=med(tables, "n_feat"), cat_frac_mean=float(np.mean(cat_fracs)), missing_frac_mean=float(np.mean([x["missing_frac"] for x in tables])),
                   mean_abs_corr_median=med(tables, "mean_abs_corr"), erank_ratio_median=med(tables, "erank_ratio"),
                   reg_gbm_r2_median=med(reg, "gbm_r2"), reg_ridge_r2_median=med(reg, "ridge_r2"),
                   reg_headroom_median=float(np.median([x["gbm_r2"] - x["ridge_r2"] for x in reg])) if reg else None,
                   reg_learnable_frac=float(np.mean([x["gbm_r2"] > 0.1 for x in reg])) if reg else None,
                   cls_n_classes_hist={str(k): int(sum(x["n_classes"] == k for x in cls)) for k in sorted({x["n_classes"] for x in cls})},
                   cls_majority_acc_median=med(cls, "majority_acc"), cls_logreg_acc_median=med(cls, "logreg_acc"), cls_gbm_acc_median=med(cls, "gbm_acc"),
                   cls_gbm_over_logreg_median=float(np.median([x["gbm_acc"] - x["logreg_acc"] for x in cls])) if cls else None)
        res[name] = dict(aggregate=agg, tables=tables)
        print(f"{name:10s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in agg.items() if k != "cls_n_classes_hist"), flush=True)
    return res


COLS = [("n_feat_median", "features (median)"), ("cat_frac_mean", "categorical frac"), ("missing_frac_mean", "missing frac"),
        ("mean_abs_corr_median", "mean abs corr"), ("erank_ratio_median", "erank ratio"), ("reg_learnable_frac", "reg learnable (GBM R2>.1)"),
        ("reg_gbm_r2_median", "reg GBM R2"), ("reg_headroom_median", "reg headroom GBM-ridge"), ("cls_majority_acc_median", "cls majority acc"),
        ("cls_logreg_acc_median", "cls logreg acc"), ("cls_gbm_acc_median", "cls GBM acc"), ("cls_gbm_over_logreg_median", "cls GBM-logreg")]


def render_md(js):
    """Markdown table from the results JSON (the only way numbers reach docs: source-backed)."""
    pr = js["priors"]; p = js["protocol"]
    f = lambda v: "·" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = [f"<!-- generated by eval/prior_profile.py from eval/results_prior_profile.json: {p['n_tables']} tables/prior, {p['rows']} rows, seed {p['seed']} -->",
             "| statistic | " + " | ".join(pr) + " |", "|---|" + "---|" * len(pr)]
    lines.append("| tables (reg / cls) | " + " | ".join(f"{a['aggregate']['n_reg']} / {a['aggregate']['n_cls']}" for a in pr.values()) + " |")
    for key, label in COLS:
        lines.append(f"| {label} | " + " | ".join(f(a["aggregate"].get(key)) for a in pr.values()) + " |")
    lines.append("| cls class-count histogram | " + " | ".join(str(a["aggregate"]["cls_n_classes_hist"]).replace("'", "") for a in pr.values()) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--rows", type=int, default=512); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--priors", default="paper_scm,real,tabicl1,tabicl2"); ap.add_argument("--out", default=str(ROOT / "eval/results_prior_profile.json"))
    ap.add_argument("--md", default=str(ROOT / "docs/prior_census/profile.md")); ap.add_argument("--render", action="store_true", help="only re-render --md from --out")
    a = ap.parse_args()
    if not a.render:
        res = run(a.priors.split(","), a.n, a.rows, a.seed)
        Path(a.out).write_text(json.dumps(dict(protocol=dict(n_tables=a.n, rows=a.rows, seed=a.seed, split="first 70 % rows fit / last 30 % test, median-imputed probes"),
                                               priors=res), indent=1) + "\n")
        print("wrote", a.out)
    Path(a.md).write_text(render_md(json.loads(Path(a.out).read_text())))
    print("wrote", a.md)
