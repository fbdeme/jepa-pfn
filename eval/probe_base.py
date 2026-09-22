"""(A) Mechanism-recovery probe on the base FM (grid axis, at scale).

Freeze each base encoder, then linearly probe row-averaged column embeddings for
(C5) edge presence + function family and (C4) noise-free `f` recovery, on the base
wide-D SCM prior. Answers the grid-axis question at scale: does the pure-latent arm
(base_lat_s) encode *less* mechanism than the data-space arm (base_ds)?

Reuses the truth_eval probe math; only the encoders and the (wide-D) prior differ.
Embeddings are collected ONCE per encoder and shared by C4 + C5 (both at noise 1.0).
Run: uv run python -m eval.probe_base   (CPU, forward-only)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import load_jepa
from eval.truth_eval import ridge_mse
from model.pfn import CellPFN
from prior.scm import FuncFamily, PriorConfig, SCMPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
CTX, N_QUERY = 128, 50          # 178 rows, inside the base prior's n_rows [128,384]
N_BATCHES, BATCH = 8, 8         # CPU forward ~2.4s each; embed once, probe twice
EVAL_SEED = 30_000
SIGMAS = (1.0, 2.0, 4.0, 8.0)   # mse_f is only discriminative once noise >> signal
#   (at sigma 1 z~=f, so any value-preserving encoder recovers f trivially)
MAX_PAIRS, MAX_COLS = 24000, 12000   # cap probe rows so LogisticRegression converges
NCOLS = (4, 32)                 # base prior wide-D
# Eval-prior mechanism nonlinearity. Default (1, 1.0) = toy/base prior (unchanged). Set
# PROBE_MLP_DEPTH/PROBE_MLP_GAIN to probe on the SAME richer prior an arm was trained on
# (e.g. the L4 nonlinear prior for the matched_L4_* arms), so the probe tests recovery of the
# mechanism the encoder actually saw, not a toy one.
MLP_DEPTH = int(os.environ.get("PROBE_MLP_DEPTH", "1"))
MLP_GAIN = float(os.environ.get("PROBE_MLP_GAIN", "1.0"))
# PROBE_PRIOR=factor swaps the eval substrate to the C4 latent-factor prior (K=4, D=12, sigma=1.0):
# same mse_f probe (rep -> clean signal f), but the edge/family probes are SCM-specific and skipped.
PROBE_PRIOR = os.environ.get("PROBE_PRIOR", "scm")
CLEAN = dict(p_categorize=0.0, p_missing_table=0.0)
# swap to the big arms with PROBE_RUNS=big_ds,big_dual,big_lat_s PROBE_TAG=big
DS, DUAL, LAT = os.environ.get("PROBE_RUNS", "base_ds,base_dual,base_lat_s").split(",")


def load_cellpfn(run_name):
    ck = torch.load(ROOT / "runs" / run_name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    c = ck["cfg"]
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample")
    m.load_state_dict(ck["model"])
    return m.eval()


def encoders():
    c = torch.load(ROOT / "runs" / DS / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]
    torch.manual_seed(999)
    rand = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                   n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample")
    return {DS: load_cellpfn(DS),
            DUAL: load_jepa(DUAL).encoder().eval(),
            LAT: load_jepa(LAT).encoder().eval(),
            "random_init": rand.eval()}


@torch.no_grad()
def collect(enc, noise=1.0, ctx=CTX):
    """Encode N_BATCHES holdout tables once; keep (context-cell embeddings, truth)."""
    torch.manual_seed(EVAL_SEED)   # paired: same tables + column identities per arm
    if PROBE_PRIOR == "factor":    # C4 latent-favorable substrate (noise is fixed by sigma, not scaled)
        from prior.factor import FactorConfig, FactorPrior
        nfac = int(os.environ.get("PROBE_N_FACTORS", "4"))   # boundary sweep: match the trained K
        prior = FactorPrior(FactorConfig(n_rows=(ctx + N_QUERY, ctx + N_QUERY),
                                         n_cols=(12, 12), n_factors=nfac, sigma=1.0), seed=EVAL_SEED)
    else:
        prior = SCMPrior(PriorConfig(n_rows=(ctx + N_QUERY, ctx + N_QUERY), n_cols=NCOLS,
                                     noise_scale=noise, mlp_depth=(MLP_DEPTH, MLP_DEPTH),
                                     mlp_gain=MLP_GAIN, **CLEAN), seed=EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    out = []
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, "y_only", rng, split=ctx, return_truth=True)
        no_mask = torch.zeros_like(bt["input_mask"])
        out.append((enc.encode(bt["z"], no_mask, bt["split"]), bt))
    return out


def c4_mse_f(batches, ctx=CTX, max_n=24000):
    """Noise alignment: recover noise-free f from context-cell embeddings (non-root)."""
    H, F_ = [], []
    for h, bt in batches:
        keep = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand(-1, ctx, -1)
        H.append(h[:, :ctx][keep])
        F_.append(bt["z_f"][:, :ctx][keep].clamp(-3.05, 3.05))
    return round(ridge_mse(torch.cat(H).numpy()[:max_n],
                           torch.cat(F_).numpy()[:max_n]), 4)


def c5_scores(batches, ctx=CTX):
    """Mechanism probing from row-averaged column embeddings: edge presence (AUC),
    function family (acc)."""
    cols, fams, pairs, edges = [], [], [], []
    for h, bt in batches:
        e = h[:, :ctx].mean(1)                       # (B,D,E)
        adj = bt["adjacency"].float()
        for i in range(e.shape[0]):
            obs = bt["observed_idx"][i]
            D = len(obs)
            cols.append(e[i])
            fams.append(bt["cell_family"][i])
            a = (adj[i][obs][:, obs] + adj[i][obs][:, obs].T).bool()
            ii, jj = torch.triu_indices(D, D, offset=1)
            pairs.append(torch.cat([e[i][ii] + e[i][jj], (e[i][ii] - e[i][jj]).abs(),
                                    e[i][ii] * e[i][jj]], -1))
            edges.append(a[ii, jj])
    cols, fams = torch.cat(cols).numpy(), torch.cat(fams).numpy()
    pairs, edges = torch.cat(pairs).numpy(), torch.cat(edges).numpy()
    rng = np.random.default_rng(0)   # fixed subsample so LR converges + arms paired
    if len(pairs) > MAX_PAIRS:
        s = rng.choice(len(pairs), MAX_PAIRS, replace=False); pairs, edges = pairs[s], edges[s]
    if len(cols) > MAX_COLS:
        s = rng.choice(len(cols), MAX_COLS, replace=False); cols, fams = cols[s], fams[s]
    nc, npr = len(cols) // 2, len(pairs) // 2
    lr = LogisticRegression(max_iter=1000).fit(pairs[:npr], edges[:npr])
    edge_auc = roc_auc_score(edges[npr:], lr.predict_proba(pairs[npr:])[:, 1])
    fam_lr = LogisticRegression(max_iter=1000).fit(cols[:nc], fams[:nc])
    fam_acc = float((fam_lr.predict(cols[nc:]) == fams[nc:]).mean())
    fam_maj = float(np.bincount(fams[nc:].astype(int)).max() / len(fams[nc:]))
    return round(float(edge_auc), 4), round(fam_acc, 4), round(fam_maj, 4)


def main():
    res = {}
    factor = PROBE_PRIOR == "factor"
    for name, enc in encoders().items():
        t = time.time()
        b1 = collect(enc, 1.0)
        mse_f = {"1.0": c4_mse_f(b1)}
        if factor:   # headline metric only; edge/family + noise sweep are SCM-specific
            res[name] = dict(mse_f=mse_f)
            print(f"{name:16} mse_f={mse_f}  [{time.time()-t:.0f}s]", flush=True)
            continue
        edge_auc, fam_acc, fam_maj = c5_scores(b1)
        for s in SIGMAS[1:]:
            mse_f[str(s)] = c4_mse_f(collect(enc, s))
        res[name] = dict(edge_auc=edge_auc, fam_acc=fam_acc, fam_majority=fam_maj,
                         mse_f=mse_f)
        print(f"{name:12} edge_auc={edge_auc} fam_acc={fam_acc}(maj{fam_maj}) "
              f"mse_f={mse_f}  [{time.time()-t:.0f}s]", flush=True)
    tag = os.environ.get("PROBE_TAG", "base")
    (ROOT / f"eval/results_probe_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote eval/results_probe_{tag}.json")


if __name__ == "__main__":
    main()
