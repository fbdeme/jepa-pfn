"""Standard-benchmark eval for the released small models: OpenML-CC18 +
Grinsztajn-2022 + TabArena-v0.1, classification AND regression.

Our pilot models were trained at D in [3,8], R in [64,150] (small regime), so
every dataset is cropped to <=8 features (SelectKBest) and context is subsampled
to 128 rows - applied to ALL methods equally, so the comparison is fair, it just
measures the SMALL regime. Test rows are capped too (suite datasets reach 10^5-10^6
rows; an in-context model only needs a context + a test batch). Regression reuses
the same bar-distribution head via point_pred (that IS the wrapper the model card
should disclose); scaling up (#2) removes the crop by training on wider D from the
start.

Run (pilot, ~5 datasets/benchmark):
  uv run --with openml python -m eval.suite_bench --pilot
Full sweep:
  uv run --with openml python -m eval.suite_bench
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parents[1]))
import eval.realdata_bench as _rb
from eval.realdata_bench import DEVICE, N_ENSEMBLE, PFNClassifier, load_pfn

ROOT = Path(__file__).parents[1]
K_FEAT = int(os.environ.get("K_FEAT", 32))   # base FM trained wide-D (n_cols<=32); real-prior arms: K_FEAT=64
CTX = int(os.environ.get("CTX", 512))        # eval context; real-prior arms trained to 1024 rows: CTX=1024
TASKS = os.environ.get("SUITE_TASKS", "clf,reg").split(",")   # SUITE_TASKS=clf for class-head arms (no trained bar head)
_rb.MAX_CONTEXT = CTX   # PFNClassifier (defined in realdata_bench) reads that global...
MAX_CONTEXT = CTX       # ...PFNRegressor (defined here) reads this name
TEST_CAP = 512      # forward on a bounded query batch, not the whole test half
N_SPLITS = 3
PFN_RUNS = os.environ.get("PFN_RUNS", "base_ds,base_dual,base_lat_s").split(",")

# 3 benchmarks -> OpenML suite ids. TabArena mixes clf+reg (split by n_classes).
SUITES = {"CC18": [99], "Grinsztajn": [337, 334, 336, 335], "TabArena": [457]}


class PFNRegressor:
    """Same forward as PFNClassifier, but read the continuous point_pred at the
    masked target cell and un-standardize it. Target is standardized with context
    stats (like every other column)."""

    def __init__(self, model, seed=0):
        self.model, self.rng = model, np.random.default_rng(seed)

    def fit(self, X, y):
        if len(X) > MAX_CONTEXT:
            idx = self.rng.choice(len(X), MAX_CONTEXT, replace=False)
            X, y = X[idx], y[idx]
        self.X, self.y = X.astype(np.float64), y.astype(np.float64)
        return self

    @torch.no_grad()
    def predict(self, X_test):
        n_ctx = len(self.X)
        tab = np.vstack([np.column_stack([self.X, self.y]),
                         np.column_stack([X_test, np.zeros(len(X_test))])])
        mean = tab[:n_ctx].mean(0, keepdims=True)
        std = tab[:n_ctx].std(0, keepdims=True) + 1e-8
        z = np.clip((tab - mean) / std, -100, 100)
        imask = np.zeros_like(z, bool)
        imask[n_ctx:, -1] = True
        zt = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)[None]
        im = torch.as_tensor(imask, device=DEVICE)[None]
        logits = 0
        for _ in range(N_ENSEMBLE):
            logits = logits + self.model(zt, im, n_ctx)[0, n_ctx:, -1, :]
        yhat_z = self.model.point_pred(logits / N_ENSEMBLE).cpu().numpy()
        return yhat_z * std[0, -1] + mean[0, -1]


def load_openml(did, download=True):
    """Return (X float matrix, y, is_regression). Categoricals ordinal-encoded,
    NaNs median-filled - the same matrix for every method (fair). ponytail:
    ordinal-encode is crude for high-cardinality cats but identical across methods."""
    import openml
    ds = openml.datasets.get_dataset(did, download_data=download,
                                     download_qualities=False,
                                     download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute,
                             dataset_format="dataframe")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]
    X = X.apply(lambda c: c.fillna(c.median() if np.isfinite(c.median()) else 0.0))
    Xa = X.to_numpy(dtype=float)
    is_reg = not (pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y)
                  or y.nunique() <= 10)
    ya = y.to_numpy(dtype=float) if is_reg else pd.factorize(y)[0]
    return Xa, ya, is_reg, ds.name


def crop(X, y, is_reg):
    k = min(K_FEAT, X.shape[1])
    if X.shape[1] > k:
        try:
            X = SelectKBest(f_regression if is_reg else f_classif, k=k).fit_transform(X, y)
        except Exception:
            X = X[:, :k]
    return X


def pick_pilot(per_bench=5):
    """Smallest datasets per benchmark (fast smoke test), mixing clf + reg where
    available: up to 3 clf (2..10 classes) + 2 reg."""
    import openml
    picks = {}
    for bench, sids in SUITES.items():
        ids = []
        for sid in sids:
            ids += list(openml.study.get_suite(sid).data)
        df = openml.datasets.list_datasets(data_id=sorted(set(ids)),
                                           output_format="dataframe")
        df = df.sort_values("NumberOfInstances")
        clf = df[(df.NumberOfClasses >= 2) & (df.NumberOfClasses <= 10)]
        reg = df[df.NumberOfClasses == 0]
        chosen = list(clf.did[:3]) + list(reg.did[:2])
        picks[bench] = [int(d) for d in chosen][:per_bench] or [int(d) for d in df.did[:per_bench]]
    return picks


def pick_full(cap_rows=200_000, cap_feat=500):
    """All datasets per benchmark within our regime; skip (and LOG) the ones too
    big to download cheaply or outside the head's reach. rows cap only bounds the
    DOWNLOAD (we subsample context anyway); feat/class caps are the model regime."""
    import openml
    picks, skipped = {}, []
    for bench, sids in SUITES.items():
        ids = []
        for sid in sids:
            ids += list(openml.study.get_suite(sid).data)
        df = openml.datasets.list_datasets(data_id=sorted(set(ids)),
                                           output_format="dataframe")
        keep = []
        for _, row in df.iterrows():
            did, n, f, c = (int(row["did"]), row["NumberOfInstances"],
                            row["NumberOfFeatures"], row["NumberOfClasses"])
            nm = str(row.get("name", ""))[:24]
            if n > cap_rows or f > cap_feat:
                skipped.append((bench, did, nm, f"size n={n:.0f} f={f:.0f}"))
            elif not (c == 0 or 2 <= c <= 10):
                skipped.append((bench, did, nm, f"classes={c:.0f}"))
            else:
                keep.append(did)
        picks[bench] = keep
    for b, did, nm, why in skipped:
        print(f"  SKIP[{b}] did={did} {nm}: {why}", flush=True)
    print(f"  -> kept { {b: len(v) for b, v in picks.items()} }, "
          f"skipped {len(skipped)}", flush=True)
    return picks


def clf_scores(y, p):
    acc = accuracy_score(y, p.argmax(1))
    auc = None
    try:
        auc = (roc_auc_score(y, p[:, 1]) if p.shape[1] == 2
               else roc_auc_score(y, p, multi_class="ovr"))
    except Exception:
        pass
    return acc, auc


def eval_dataset(did, models_clf, models_reg):
    X, y, is_reg, name = load_openml(did)
    if ("reg" if is_reg else "clf") not in TASKS:
        raise RuntimeError(f"task filtered by SUITE_TASKS={TASKS}")
    X = crop(X, y, is_reg)
    rows = {}
    for split in range(N_SPLITS):
        strat = None if is_reg else y
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                              random_state=split, stratify=strat)
        if len(Xte) > TEST_CAP:
            r = np.random.default_rng(split)
            sel = r.choice(len(Xte), TEST_CAP, replace=False)
            Xte, yte = Xte[sel], yte[sel]
        for mname, make in ((models_reg if is_reg else models_clf)).items():
            clf = make()
            clf.fit(Xtr, ytr)
            rows.setdefault(mname, [])
            if is_reg:
                yhat = clf.predict(Xte)
                rows[mname].append((r2_score(yte, yhat),
                                    float(np.sqrt(np.mean((yhat - yte) ** 2)))))
            else:
                acc, auc = clf_scores(yte, clf.predict_proba(Xte))
                rows[mname].append((acc, auc))
    out = {}
    for m, vals in rows.items():
        a = np.array([v[0] for v in vals], float)
        b = [v[1] for v in vals if v[1] is not None]
        key = "r2" if is_reg else "acc"
        out[m] = {key: round(float(a.mean()), 4),
                  ("rmse" if is_reg else "auc"): (round(float(np.mean([v[1] for v in vals])), 4)
                                                  if is_reg else (round(float(np.mean(b)), 4) if b else None))}
    return name, ("reg" if is_reg else "clf"), out


def main():
    warnings.filterwarnings("ignore")
    pilot = "--pilot" in sys.argv
    pfns = {r: load_pfn(r) for r in PFN_RUNS
            if (ROOT / "runs" / r / "ckpt.pt").exists()}
    models_clf = {r: (lambda m=m: PFNClassifier(m)) for r, m in pfns.items()}
    models_clf["logreg"] = lambda: LogisticRegression(max_iter=2000)
    models_clf["histgb"] = lambda: HistGradientBoostingClassifier(random_state=0)
    models_reg = {r: (lambda m=m: PFNRegressor(m)) for r, m in pfns.items()}
    models_reg["ridge"] = lambda: Ridge()
    models_reg["histgbr"] = lambda: HistGradientBoostingRegressor(random_state=0)

    picks = pick_pilot() if pilot else pick_full()
    tag = os.environ.get("SUITE_TAG", "pilot" if pilot else "full")
    outpath = ROOT / f"eval/results_suite_{tag}.json"
    results = {}
    for bench, dids in (picks or {}).items():
        print(f"\n=== {bench} ({len(dids)} datasets) ===", flush=True)
        for did in dids:
            try:
                name, kind, out = eval_dataset(did, models_clf, models_reg)
            except Exception as e:
                print(f"  did={did}: SKIP {type(e).__name__}: {e}", flush=True)
                continue
            metric = "r2" if kind == "reg" else "acc"
            line = " ".join(f"{m}={out[m][metric]}" for m in out)
            print(f"  [{kind}] {name[:28]:28} did={did}  {line}", flush=True)
            results[f"{bench}|{name}|{kind}"] = out
            outpath.write_text(json.dumps(results, indent=1))  # incremental: survive a crash
    print(f"\nwrote {outpath}  ({len(results)} datasets)")


if __name__ == "__main__":
    main()
