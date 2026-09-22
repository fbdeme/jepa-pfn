"""Phase 5: small real-data sanity benchmark (SSOT R3: auxiliary, not the
main stage) + missing-value imputation demo (C2 byproduct).

Positioning: external validity of the synthetic findings at MATCHED training
budget - p2_yonly (y-only PFN, our nanoTabPFN-equivalent) vs p2_anycell
(single model that also imputes), against LogReg / HistGB. TabPFN v2 is a
scale reference, not a fair comparison (72x params, 156x data).

Caveats stated up front: our prior trained D in [3,8] and R in [64,150], so
datasets are reduced to <=8 features (SelectKBest, applied to ALL methods
equally) and context is subsampled to 128 rows.

Run: uv run python -m eval.realdata_bench   (CPU, forward-only)
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.pfn import CellPFN

ROOT = Path(__file__).parents[1]
MAX_FEATURES = 8
MAX_CONTEXT = 128  # training saw R in [64,150]
DEVICE = os.environ.get("DEVICE", "cpu")   # DEVICE=cuda on a GPU box: the wide real-prior suite (CTX 1024 x 64 feats) is ~4 min/dataset on CPU
N_ENSEMBLE = 4     # marginalize over random column identities r_j
N_SPLITS = 5


def load_pfn(run_name):
    ck = torch.load(ROOT / "runs" / run_name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    c = ck["cfg"]
    if c.get("head_on_pred") or c.get("ctx_only"):   # I-JEPA-split arms: value head on the predictor (x-encoder + predictor +
        # head) or on the ctx_only encoder (headenc: x-encoder + head); a bare CellPFN read would run the encoder without ctx_only
        from eval.latent_probe import load_jepa
        from model.jepa import ValueModel
        return ValueModel(load_jepa(run_name)).eval().to(DEVICE)
    # rope + CLS arms (p6rc) need the scheme/n_cls or the state_dict won't map;
    # old p2/p3 ckpts lack these keys -> None -> harmless defaults.
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample", n_classes=c.get("n_classes") or 0)
    sd = ck["model"]
    if any(k.startswith("online.") for k in sd):
        # JEPA arms wrap a CellPFN as .online (same 63 keys). Take online, not
        # target: lambda_ppd's data-space CE trains the online head, and target
        # is only its EMA mirror. Pure-latent arms (lambda_ppd=0) load fine here
        # but their head never got a gradient - that is the point, not a bug.
        sd = {k[len("online."):]: v for k, v in sd.items()
              if k.startswith("online.")}
    m.load_state_dict(sd)
    return m.eval().to(DEVICE)


class PFNClassifier:
    """sklearn-like wrapper: y as an extra column, query-row y cells masked,
    bar-distribution mass assigned to the nearest class code."""

    def __init__(self, model, seed=0):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        if len(X) > MAX_CONTEXT:
            idx = self.rng.choice(len(X), MAX_CONTEXT, replace=False)
            X, y = X[idx], y[idx]
        self.X, self.y = X.astype(np.float64), y.astype(np.float64)
        self.classes = np.unique(y)
        return self

    @torch.no_grad()
    def predict_proba(self, X_test):
        n_ctx = len(self.X)
        tab = np.vstack([np.column_stack([self.X, self.y]),
                         np.column_stack([X_test, np.zeros(len(X_test))])])
        mean = tab[:n_ctx].mean(0, keepdims=True)
        std = tab[:n_ctx].std(0, keepdims=True) + 1e-8
        z = np.clip((tab - mean) / std, -100, 100)
        imask = np.zeros_like(z, bool)
        imask[n_ctx:, -1] = True  # hide test labels
        zt = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)[None]
        im = torch.as_tensor(imask, device=DEVICE)[None]
        if getattr(self.model, "n_classes", 0):        # class-head arm: direct class probabilities (codes = column order)
            logits = 0
            for _ in range(N_ENSEMBLE):
                h = self.model.encode(zt, im, n_ctx)[0, n_ctx:, -1]
                logits = logits + self.model.cls_head(h)[:, :len(self.classes)]
            return torch.softmax(logits / N_ENSEMBLE, -1).cpu().numpy()
        logits = 0
        for _ in range(N_ENSEMBLE):
            logits = logits + self.model(zt, im, n_ctx)[0, n_ctx:, -1, :]
        probs_bins = torch.softmax(logits / N_ENSEMBLE, -1).cpu()
        z_codes = (self.classes - mean[0, -1]) / std[0, -1]
        owner = np.abs(self.model.bin_centers.cpu().numpy()[:, None]
                       - z_codes[None]).argmin(1)  # each bin -> nearest class
        cls = np.zeros((len(X_test), len(self.classes)))
        for b, c in enumerate(owner):
            cls[:, c] += probs_bins[:, b].numpy()
        return cls / cls.sum(1, keepdims=True)


def datasets():
    warnings.filterwarnings("ignore")
    out = {"iris": load_iris(return_X_y=True),
           "wine": load_wine(return_X_y=True),
           "breast_cancer": load_breast_cancer(return_X_y=True)}
    for name in ("diabetes", "blood-transfusion-service-center"):
        try:
            d = fetch_openml(name=name, version=1, as_frame=False,
                             parser="liac-arff")
            y = np.unique(d.target, return_inverse=True)[1]
            out[name.split("-")[0]] = (d.data.astype(float), y)
        except Exception as exc:  # offline etc. - benchmark the rest
            print(f"skip {name}: {exc}", flush=True)
    return out


def bench_classification(models):
    results = {}
    for dname, (X, y) in datasets().items():
        if X.shape[1] > MAX_FEATURES:
            X = SelectKBest(f_classif, k=MAX_FEATURES).fit_transform(X, y)
        accs, aucs = {m: [] for m in models}, {m: [] for m in models}
        for split in range(N_SPLITS):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                                  random_state=split, stratify=y)
            for mname, make in models.items():
                clf = make()
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xte)
                accs[mname].append(accuracy_score(yte, p.argmax(1)))
                if len(np.unique(y)) == 2:
                    aucs[mname].append(roc_auc_score(yte, p[:, 1]))
        for mname in models:
            results[f"{dname}|{mname}"] = dict(
                acc=round(float(np.mean(accs[mname])), 4),
                acc_std=round(float(np.std(accs[mname])), 4),
                auc=round(float(np.mean(aucs[mname])), 4) if aucs[mname] else None)
            print(f"{dname:16} {mname:12} acc={results[f'{dname}|{mname}']['acc']:.3f}"
                  f"±{results[f'{dname}|{mname}']['acc_std']:.3f}"
                  f" auc={results[f'{dname}|{mname}']['auc']}", flush=True)
    return results


@torch.no_grad()
def pfn_impute(model, X_nan, rng):
    """Impute NaNs: rows with NaNs are appended as query copies; missing cells
    are input-masked everywhere, so the true value never leaks."""
    n = len(X_nan)
    ctx = X_nan[:MAX_CONTEXT] if n > MAX_CONTEXT else X_nan
    holes = np.isnan(X_nan).any(1)
    tab = np.vstack([ctx, X_nan[holes]])
    n_ctx = len(ctx)
    mean = np.nanmean(tab[:n_ctx], 0, keepdims=True)
    std = np.nanstd(tab[:n_ctx], 0, keepdims=True) + 1e-8
    z = np.nan_to_num(np.clip((tab - mean) / std, -100, 100))
    imask = np.isnan(tab)
    zt = torch.as_tensor(z, dtype=torch.float32)[None]
    im = torch.as_tensor(imask)[None]
    logits = 0
    for _ in range(N_ENSEMBLE):
        logits = logits + model(zt, im, n_ctx)[0]
    zhat = model.point_pred(logits / N_ENSEMBLE).numpy()
    out = X_nan.copy()
    qrows = np.flatnonzero(holes)
    for qi, row in enumerate(qrows):
        for d in np.flatnonzero(np.isnan(X_nan[row])):
            out[row, d] = zhat[n_ctx + qi, d] * std[0, d] + mean[0, d]
    return out


def bench_imputation(pfn_model):
    rng = np.random.default_rng(0)
    results = {}
    for dname, (X, y) in datasets().items():
        if X.shape[1] > MAX_FEATURES:
            X = SelectKBest(f_classif, k=MAX_FEATURES).fit_transform(X, y)
        X = X[:500]
        scale = X.std(0, keepdims=True) + 1e-8
        rmses = {"pfn_anycell": [], "column_mean": [], "knn": []}
        for split in range(N_SPLITS):
            m = rng.random(X.shape) < 0.2
            X_nan = X.copy()
            X_nan[m] = np.nan
            fill_mean = np.where(m, np.nanmean(X_nan, 0, keepdims=True), X)
            fill_knn = KNNImputer(n_neighbors=5).fit_transform(X_nan)
            fill_pfn = pfn_impute(pfn_model, X_nan, rng)
            for name, F in [("pfn_anycell", fill_pfn), ("column_mean", fill_mean),
                            ("knn", fill_knn)]:
                rmses[name].append(float(np.sqrt(np.mean(((F - X) / scale)[m] ** 2))))
        for name in rmses:
            results[f"{dname}|{name}"] = round(float(np.mean(rmses[name])), 4)
        print(f"impute {dname:16} " + " ".join(
            f"{k}={results[f'{dname}|{k}']:.3f}" for k in rmses), flush=True)
    return results


def main():
    # Released headline (grid rope+CLS) arms first, then the original p2/p3 set
    # for context. p6rc_lat is pure-latent (random-init head) -> its score IS the
    # measurement, like pure_latent. Skip any run that isn't on disk.
    pfn_runs = ["p6rc_ds", "p6rc_dual", "p6rc_lat",
                "p2_yonly", "p2_anycell", "p3d_ppd_01", "p3d_ppd_10", "p3b_mixed_tw2"]
    loaded = {r: load_pfn(r) for r in pfn_runs
              if (ROOT / "runs" / r / "ckpt.pt").exists()}
    models = {r: (lambda m=m: PFNClassifier(m)) for r, m in loaded.items()}
    models["logreg"] = lambda: LogisticRegression(max_iter=2000)
    models["histgb"] = lambda: HistGradientBoostingClassifier(random_state=0)
    imp_model = loaded.get("p2_anycell") or loaded.get("p6rc_ds")
    results = dict(classification=bench_classification(models),
                   imputation=bench_imputation(imp_model))
    (ROOT / "eval/results_p5_realdata.json").write_text(json.dumps(results, indent=1))
    print("wrote eval/results_p5_realdata.json")


if __name__ == "__main__":
    main()
