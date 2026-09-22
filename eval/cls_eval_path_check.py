"""Eval-path check for the class-head arms (docs/real_prior_plan.md 11; cited by cls_head_2026-09-04/verdict.md).
Same in-prior tables, three read-outs: (a) training path (NaN masked, label in place), (b) label in place but NaN mean-imputed,
(c) PFNClassifier (label appended last, NaN mean-imputed). If the three agree, the real-suite numbers are not an evaluation-path artefact.
Output eval/results_cls_eval_path_check.json. Run: uv run python -m eval.cls_eval_path_check [RUN ...] (default cls_dual_lr5e-4_s1 cls_ds_lr1.7e-4_s0)"""
import json, sys, yaml, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from prior.real import RealPrior, RealConfig
from eval.realdata_bench import load_pfn, PFNClassifier
cfg = yaml.safe_load((ROOT / "configs/cls_ds_lr5e-4_s0.yaml").read_text())
RUNS = sys.argv[1:] or ["cls_dual_lr5e-4_s1", "cls_ds_lr1.7e-4_s0"]; S = 512
OUT = ROOT / "eval/results_cls_eval_path_check.json"

@torch.no_grad()
def readout(m, x, t, K, mask_missing):
    """label in place; x raw with NaN. mask_missing: True = training path, False = mean-impute."""
    x = x.copy()
    ctx = x[:S]; mean = np.nanmean(ctx, 0, keepdims=True); std = np.nanstd(ctx, 0, keepdims=True) + 1e-8
    missing = np.isnan(x)
    z = np.nan_to_num(np.clip((x - np.nan_to_num(mean)) / np.nan_to_num(std, nan=1.0), -100, 100))
    if not mask_missing:
        z = np.where(missing, 0.0, z); missing = np.zeros_like(missing)
    tmask = np.zeros_like(missing); tmask[S:, t] = True
    im = torch.as_tensor(tmask | missing)[None]
    zt = torch.as_tensor(z, dtype=torch.float32)[None]
    h = m.encode(zt, im, S)[0, S:, t]
    lg = m.cls_head(h)[:, :K]
    return lg.argmax(-1).numpy()

def run_one(run):
    m = load_pfn(run); pr = RealPrior(RealConfig(n_rows=(640, 640), target_classes=tuple(cfg["prior"]["target_classes"])), seed=777)
    res = {"a_train_path": [], "b_impute_inplace": [], "c_classifier": [], "maj": []}; n = 0; tables = []
    for _ in range(16):
        b = pr.sample_batch(1); h = b.hyper[0]
        if h["target_col"] is None or not h["n_classes"]: continue
        x = b.x[0]; t = h["target_col"]; K = h["n_classes"]
        y = np.nan_to_num(x[:, t], nan=-1).astype(int)
        if h["D"] < 3 or len(x) < 560 or len(np.unique(y[:S][y[:S] >= 0])) < 2: continue
        ok = y[S:] >= 0; yt = y[S:][ok]
        vals, cnt = np.unique(y[:S][y[:S] >= 0], return_counts=True); res["maj"].append((yt == vals[cnt.argmax()]).mean())
        res["a_train_path"].append((readout(m, x, t, K, True)[ok] == yt).mean())
        res["b_impute_inplace"].append((readout(m, x, t, K, False)[ok] == yt).mean())
        xi = x.copy(); cm = np.nanmean(xi[:S], 0); xi = np.where(np.isnan(xi), cm, xi)
        X = np.delete(xi, t, 1); yc = y[:S]; keep = yc >= 0
        clf = PFNClassifier(m).fit(X[:S][keep], yc[keep]); p = clf.predict_proba(X[S:][ok])
        res["c_classifier"].append((clf.classes[p.argmax(1)] == yt).mean())
        n += 1
        tables.append({"D": int(h["D"]), "K": int(K), "missing": round(float(np.isnan(x).mean()), 3), **{k: round(float(v[-1]), 4) for k, v in res.items()}})
    means = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    print(run, "MEAN", means)
    return {"tables": tables, "mean": means}


if __name__ == "__main__":
    out = {"note": "640-row in-prior tables, 512 context rows; a = training path (NaN masked, label in place), b = NaN mean-imputed in place, c = PFNClassifier (label appended last)", "runs": {r: run_one(r) for r in RUNS}}
    OUT.write_text(json.dumps(out, indent=1) + "\n"); print("wrote", OUT.relative_to(ROOT))
