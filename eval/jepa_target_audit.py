"""JEPA target audit (checks 1 and 2 of the 2026-09-04 "are we misusing JEPA?" review).

Check 1 - is the JEPA target just the value re-encoded?  For each trained arm, take the actual JEPA targets z_tgt
(target-encoder embeddings at the target cells / CLS row summaries, LayerNorm-ed as in the loss) on fresh in-prior
tables and regress them on the raw standardised values: X_full = the row's values as the TARGET encoder sees them
(hidden cells included), X_vis = the row's values as the PREDICTOR sees them (hidden cells zeroed) [+ column one-hot
for cell targets]. Ridge (linear) and ridge on random Fourier features (nonlinear); explained variance on a held-out half.
Check 2 - what did the predictor learn beyond trivial predictors?  MSE(pred, z_tgt) vs zeros (=1 for LN targets),
the global mean target, the per-column mean target, and the RFF regressions above (X_vis = same information as the
predictor; X_full = sees the hidden values, an upper reference).
Output eval/results_jepa_target_audit.json.  Run: uv run --with scikit-learn python -m eval.jepa_target_audit [RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.latent_probe import load_jepa          # noqa: E402
from model.jepa import JEPA                      # noqa: E402
from train.data import make_batch                # noqa: E402
from train.train import _make_prior              # noqa: E402

OUT = ROOT / "eval/results_jepa_target_audit.json"
DEFAULT = ["base_dual_s1", "base_dual_s2", "base_ema_nohead_s1", "real_dual_lr5e-4_s0", "cls_dual_lr5e-4_s1", "base_dual_s0", "base_lat_s_s0"]
N_BATCHES, MAX_T, N_RFF = 6, 12000, 2048


def rff(X, rng, n=N_RFF):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    W = rng.normal(size=(X.shape[1], n)) / np.sqrt(X.shape[1]); b = rng.uniform(0, 2 * np.pi, n)
    return np.cos(X @ W + b)


def ev(X, Y, rng, alpha=1.0):
    """Held-out explained-variance fraction of ridge X->Y (pooled over dims) and the held-out MSE."""
    n = len(X); p = rng.permutation(n); tr, te = p[: n // 2], p[n // 2:]
    m = Ridge(alpha=alpha).fit(X[tr], Y[tr]); P = m.predict(X[te])
    resid = ((Y[te] - P) ** 2).sum(); tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1 - resid / tot), float(((Y[te] - P) ** 2).mean())


@torch.no_grad()
def audit(model, cfg, seed_offset=20_000):
    seed = seed_offset + int(cfg.get("seed", 0))
    prior = _make_prior(cfg, seed, "val"); rng = np.random.default_rng(seed)
    policy = "y_target" if cfg.get("n_classes") else cfg.get("policy_main", "mixed")
    k = model.online.n_cls; keep = model.cls_target
    preds, tgts, Xf, Xv, cols, tids, isq, raws = [], [], [], [], [], [], [], []
    for bi in range(N_BATCHES):
        bt = make_batch(prior, cfg["batch_size"], policy, rng, "cpu")
        _, _, pred, tgt = model(bt["z"], bt["input_mask"], bt["target_mask"], bt["split"],
                                y_ids=bt.get("y_ids"), n_classes=bt.get("n_classes"))
        tm = model._cls_mask(bt["target_mask"], k) if keep else bt["target_mask"]
        if model.mode == "ema":                                 # raw target-encoder output before the LayerNorm of the loss
            B_, _, D_ = bt["z"].shape; r_ = model.online.sample_r(B_, D_, bt["z"].device)
            raw = model.target.encode(bt["z"], torch.zeros_like(bt["input_mask"]), bt["split"], r=r_, keep_cls=keep)[tm]
            raws.append(raw.numpy())
        idx = tm.nonzero()                                     # (N_t, 3) in the same order as tgt / pred
        z = bt["z"]; vis = torch.where(bt["input_mask"], torch.zeros_like(z), z)
        D = z.shape[2]
        for (b, r, c), p_, t_ in zip(idx.tolist(), pred, tgt):
            preds.append(p_.numpy()); tgts.append(t_.numpy()); tids.append(bi * cfg["batch_size"] + b); isq.append(int(r >= bt["split"]))
            row_full, row_vis = z[b, r].numpy(), vis[b, r].numpy()
            if keep:                                           # CLS row-summary target: the row's D values
                Xf.append(row_full); Xv.append(row_vis); cols.append(c)
            else:                                              # cell target: the cell's value + its row + column one-hot
                oh = np.zeros(D); oh[c] = 1
                Xf.append(np.concatenate([[row_full[c]], row_full, oh])); Xv.append(np.concatenate([[0.0], row_vis, oh])); cols.append(c)
    Dm = max(len(x) for x in Xf)                              # ragged D across batches -> pad to max D
    Xf = np.stack([np.pad(x, (0, Dm - len(x))) for x in Xf]); Xv = np.stack([np.pad(x, (0, Dm - len(x))) for x in Xv])
    P, T, cols, tids, isq = np.asarray(preds), np.asarray(tgts), np.asarray(cols), np.asarray(tids), np.asarray(isq)
    R = np.concatenate(raws) if raws else None
    if len(T) > MAX_T:
        sel = rng.choice(len(T), MAX_T, replace=False); P, T, Xf, Xv, cols, tids, isq = P[sel], T[sel], Xf[sel], Xv[sel], cols[sel], tids[sel], isq[sel]
        R = R[sel] if R is not None else None
    r = np.random.default_rng(0)
    out = {"n_targets": int(len(T)), "target": "cls_row_summary" if keep else "cell", "policy": policy, "mode": cfg.get("mode", "ema"),
           "emb_dim": int(T.shape[1])}
    out["r2_linear_full"], out["mse_ridge_full"] = ev(Xf, T, r)
    out["r2_rff_full"], out["mse_rff_full"] = ev(rff(Xf, r), T, r, alpha=10.0)
    out["r2_linear_vis"], out["mse_ridge_vis"] = ev(Xv, T, r)
    out["r2_rff_vis"], out["mse_rff_vis"] = ev(rff(Xv, r), T, r, alpha=10.0)
    out["mse_pred"] = float(((P - T) ** 2).mean())
    out["mse_zeros"] = float((T ** 2).mean())
    out["mse_global_mean"] = float(((T - T.mean(0)) ** 2).mean())
    cm = np.stack([T[cols == c].mean(0) if (cols == c).any() else T.mean(0) for c in range(cols.max() + 1)])
    out["mse_col_mean"] = float(((T - cm[cols]) ** 2).mean())
    # within-table structure: is the target a per-TABLE constant?  (batch-level collapse instruments cannot see this)
    tm_ = np.stack([T[tids == t].mean(0) for t in np.unique(tids)]); tmap = {t: i for i, t in enumerate(np.unique(tids))}
    Tt = tm_[[tmap[t] for t in tids]]
    tot = ((T - T.mean(0)) ** 2).sum(); between = ((Tt - T.mean(0)) ** 2).sum()
    out["n_tables"] = int(len(tm_)); out["tgt_between_table_var_frac"] = float(between / tot) if tot > 0 else None
    out["mse_table_mean"] = float(((T - Tt) ** 2).mean())            # predict every row by its table's mean target (in-sample)
    out["mse_pred_vs_table_mean"] = float(((P - Tt) ** 2).mean())    # how far the predictor's output is from the table mean
    # per-table regression (a context-dependent encoding of the values defeats the pooled ridge above but not this one)
    def pertable(X, feat=None):
        resid = tot_ = 0.0; n_used = 0
        for t in np.unique(tids):
            sel = np.flatnonzero(tids == t)
            if len(sel) < 120:
                continue
            Xt_, Yt_ = X[sel], T[sel]
            if feat is not None:
                Xt_ = feat(Xt_)
            pr = r.permutation(len(sel)); a, b_ = pr[: len(sel) // 2], pr[len(sel) // 2:]
            m = Ridge(alpha=1.0).fit(Xt_[a], Yt_[a]); Pp = m.predict(Xt_[b_])
            resid += ((Yt_[b_] - Pp) ** 2).sum(); tot_ += ((Yt_[b_] - Yt_[a].mean(0)) ** 2).sum(); n_used += 1
        return (float(1 - resid / tot_) if tot_ > 0 else None), n_used
    def pertable_knn(X, k=10):
        resid = tot_ = 0.0
        for t in np.unique(tids):
            sel = np.flatnonzero(tids == t)
            if len(sel) < 120:
                continue
            Xt_ = X[sel]; Xt_ = (Xt_ - Xt_.mean(0)) / (Xt_.std(0) + 1e-6); Yt_ = T[sel]
            pr = r.permutation(len(sel)); a, b_ = pr[: len(sel) // 2], pr[len(sel) // 2:]
            Pp = KNeighborsRegressor(n_neighbors=k).fit(Xt_[a], Yt_[a]).predict(Xt_[b_])
            resid += ((Yt_[b_] - Pp) ** 2).sum(); tot_ += ((Yt_[b_] - Yt_[a].mean(0)) ** 2).sum()
        return float(1 - resid / tot_) if tot_ > 0 else None
    def pertable_hgb(X, n_pc=3, max_tables=16):
        """per-table gradient boosting on the top-n_pc principal directions of the targets (sharp / column-selective maps)."""
        resid = tot_ = 0.0; used = 0
        for t in np.unique(tids):
            sel = np.flatnonzero(tids == t)
            if len(sel) < 160 or used >= max_tables:
                continue
            used += 1; Xt_, Yt_ = X[sel], T[sel]; Yc = Yt_ - Yt_.mean(0)
            _, _, Vt = np.linalg.svd(Yc, full_matrices=False); Z = Yc @ Vt[:n_pc].T
            pr = r.permutation(len(sel)); a, b_ = pr[: len(sel) // 2], pr[len(sel) // 2:]
            for j in range(n_pc):
                m = HistGradientBoostingRegressor(max_iter=60, max_depth=3, random_state=0).fit(Xt_[a], Z[a, j])
                resid += ((Z[b_, j] - m.predict(Xt_[b_])) ** 2).sum(); tot_ += ((Z[b_, j] - Z[a, j].mean()) ** 2).sum()
        return float(1 - resid / tot_) if tot_ > 0 else None
    out["r2_hgb_top3pc_full_pertable"] = pertable_hgb(Xf)
    out["r2_knn10_full_pertable"] = pertable_knn(Xf)      # non-smooth but local dependence on the row's values
    out["r2_knn10_vis_pertable"] = pertable_knn(Xv)
    # context/query membership (the only row-position signal the encoders can see)
    q = isq[:, None].astype(float); qm = np.stack([T[isq == v].mean(0) if (isq == v).any() else T.mean(0) for v in (0, 1)])[isq]
    out["tgt_var_frac_context_vs_query"] = float(((qm - T.mean(0)) ** 2).sum() / tot) if tot > 0 else None
    out["frac_query_targets"] = float(isq.mean())
    out["r2_linear_full_pertable"], out["n_tables_pertable"] = pertable(Xf)
    out["r2_rff_full_pertable"], _ = pertable(Xf, lambda X: rff(X, r, n=256))
    out["r2_linear_vis_pertable"], _ = pertable(Xv)
    out["r2_rff_vis_pertable"], _ = pertable(Xv, lambda X: rff(X, r, n=256))
    out["perturb"] = perturb_sensitivity(model, cfg, prior, keep, k, rng)
    if R is not None:                                        # norm-collapse diagnostics on the RAW target (what LayerNorm hides)
        mu = R.mean(0); dev = R - mu
        out["raw_rel_spread"] = float((dev ** 2).sum(1).mean() / (R ** 2).sum(1).mean())          # ||raw - mean||^2 / ||raw||^2
        sub = R[r.choice(len(R), min(512, len(R)), replace=False)]; nrm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        cs = nrm @ nrm.T; out["raw_mean_pairwise_cos"] = float((cs.sum() - np.trace(cs)) / (len(sub) * (len(sub) - 1)))
        out["raw_crossdim_std_mean"] = float(R.std(1).mean()); out["raw_norm_mean"] = float(np.linalg.norm(R, axis=1).mean())
        sv = np.linalg.svd(dev - dev.mean(0), compute_uv=False); out["raw_top1_pc_var_frac"] = float(sv[0] ** 2 / (sv ** 2).sum())
        svT = np.linalg.svd(T - T.mean(0), compute_uv=False); out["ln_top1_pc_var_frac"] = float(svT[0] ** 2 / (svT ** 2).sum())
    ptm = np.stack([P[tids == t].mean(0) for t in np.unique(tids)])[[tmap[t] for t in tids]]
    ptot = ((P - P.mean(0)) ** 2).sum(); out["pred_between_table_var_frac"] = float(((ptm - P.mean(0)) ** 2).sum() / ptot) if ptot > 0 else None
    return out


@torch.no_grad()
def perturb_sensitivity(model, cfg, prior, keep, k, rng, eps_list=(0.01, 0.1, 1.0), n_rows=24):
    """Smoothness of the target map g_table(x_row): add N(0, eps^2) noise (in standardised units) to ONE query row's values,
    recompute the target-encoder targets, report ||dT|| / ||T|| for that row (LN'd targets) vs the untouched rows' change (=0)."""
    import torch.nn.functional as F
    bt = make_batch(prior, cfg["batch_size"], cfg.get("policy_main", "mixed") if not cfg.get("n_classes") else "y_target", rng, "cpu")
    z, im, S = bt["z"], bt["input_mask"], bt["split"]; B, R_, D = z.shape
    enc = model.target if model.mode == "ema" else model.online
    r_ = model.online.sample_r(B, D, z.device); nm = torch.zeros_like(im)
    base = enc.encode(z, nm, S, r=r_, keep_cls=keep)
    base = F.layer_norm(base, base.shape[-1:]) if model.mode == "ema" else base
    res = {}
    g = torch.Generator().manual_seed(0)
    for eps in eps_list:
        ratios = []
        for _ in range(n_rows):
            b = int(rng.integers(B)); row = int(rng.integers(S, R_))
            z2 = z[b:b + 1].clone(); z2[0, row] += eps * torch.randn(D, generator=g)
            h2 = enc.encode(z2, nm[b:b + 1], S, r=r_[b:b + 1] if r_ is not None else None, keep_cls=keep)
            h2 = F.layer_norm(h2, h2.shape[-1:]) if model.mode == "ema" else h2
            t0 = base[b, row, :k] if keep else base[b, row]; t1 = h2[0, row, :k] if keep else h2[0, row]
            ratios.append(float((t1 - t0).norm() / (t0.norm() + 1e-9)))
        res[str(eps)] = {"rel_change_mean": float(np.mean(ratios)), "rel_change_median": float(np.median(ratios))}
    return res


def table():
    """Markdown table from the JSON (no recomputation): uv run python -m eval.jepa_target_audit --table"""
    d = json.loads(OUT.read_text())["runs"]
    g = lambda v, k, f="{:.3f}": ("-" if v.get(k) is None else f.format(v[k]))
    print("| run | target | n | LN'd var (global) | between-table frac | raw pairwise cos | per-table R2 lin / kNN10 / HGB | perturb rel.change @eps .01 / .1 / 1 | MSE pred / table-mean / global-mean / zeros |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run, v in d.items():
        pj = v.get("perturb") or {}
        pert = " / ".join(g(pj.get(e, {}), "rel_change_median") for e in ("0.01", "0.1", "1.0"))
        print(f"| {run} | {v['target']} | {v['n_targets']} | {g(v,'mse_global_mean')} | {g(v,'tgt_between_table_var_frac')} | {g(v,'raw_mean_pairwise_cos')} | "
              f"{g(v,'r2_linear_full_pertable')} / {g(v,'r2_knn10_full_pertable')} / {g(v,'r2_hgb_top3pc_full_pertable')} | {pert} | "
              f"{g(v,'mse_pred')} / {g(v,'mse_table_mean')} / {g(v,'mse_global_mean')} / {g(v,'mse_zeros')} |")


def main():
    if sys.argv[1:] == ["--table"]:
        return table()
    runs = sys.argv[1:] or DEFAULT
    res = {}
    for run in runs:
        m = load_jepa(run); cfg = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)["cfg"]
        res[run] = audit(m, cfg); print(run, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[run].items()}))
        if run == runs[0]:                                     # untrained reference with the same architecture / prior
            c = cfg
            rnd = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"], c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
                       c.get("predictor", "mlp"), c.get("mode", "ema"), c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
                       c.get("r_invariant", False), c.get("n_cls", 0), c.get("cls_target", False), c.get("r_scheme", "resample"),
                       n_classes=c.get("n_classes", 0)).eval()
            torch.manual_seed(0); res[f"random_init({run})"] = audit(rnd, cfg)
            print(f"random_init({run})", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[f'random_init({run})'].items()}))
    prev = json.loads(OUT.read_text())["runs"] if OUT.exists() else {}
    prev.update(res)
    OUT.write_text(json.dumps({"note": __doc__.strip().split("\n")[0], "n_batches": N_BATCHES, "max_targets": MAX_T, "n_rff": N_RFF, "runs": prev}, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
