"""Phase 4: ground-truth evaluation suite (SSOT section 4 Phase 4, section 5 track 1).

C4 noise alignment (headline): probes on CONTEXT-cell embeddings recover the
mechanism value f vs the observed value x = f + sigma*eps, swept over the
prior's global noise_scale. Rationale (docs/history.md): for masked-cell
predictions eps is unpredictable for every model, so f-vs-x can only
discriminate on cells whose value is in the input. Root cells (f := x) and
categorized/missing cells are excluded by config.

C5 mechanism probing: from row-averaged column embeddings recover (a) edge
presence between observed column pairs, (b) node function family, (c) node
degree. Probes are fit on one half of the tables and scored on the other.

Also: ICL curves (context size vs C4/C5 metrics) and a secondary
masked-cell f-recovery sweep.

Representations: EMA arms = target encoder, SIGReg arms = online encoder (they
have no EMA copy; JEPA.encoder() picks); p2_anycell = trunk encoder;
random_init = untrained encoder (SSOT baseline iii). All are the same cell
encoder at the same point in the stack - the SIGReg arm's projector is
objective machinery, not representation (LeJEPA probes the frozen backbone
too), so the comparison stays apples-to-apples.

Run: uv run python -m eval.truth_eval   (CPU, forward-only)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import load_auto, load_cfg, load_jepa, probe_prior, truth_cells
from model.pfn import CellPFN
from prior.scm import FuncFamily
from train.data import make_batch

ROOT = Path(__file__).parents[1]
CTX, N_QUERY = 100, 50
SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]  # global noise_scale; per-node sigma is
# LogU(0.01,0.3) x scale, so scale 8 => mean sigma ~0.65 vs unit signal -
# below ~2 the f/x gap sits under the probe's resolution
ICL_CTX = [10, 25, 50, 100]
N_BATCHES, BATCH = 24, 16
EVAL_SEED = 30_000
RIDGE_ALPHA = 1.0
CLEAN = dict(p_categorize=0.0, p_missing_table=0.0)


def encoders():
    encs = {}
    for d in sorted((ROOT / "runs").glob("p3*")):  # all JEPA arms present
        if (d / "ckpt.pt").exists():
            encs[d.name] = load_jepa(d.name).encoder().eval()
    p2 = torch.load(ROOT / "runs/p2_anycell/ckpt.pt", map_location="cpu",
                    weights_only=False)
    c = p2["cfg"]
    p2_enc = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"])
    p2_enc.load_state_dict(p2["model"])
    torch.manual_seed(999)
    rand = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"])
    encs["p2_anycell"] = p2_enc.eval()
    encs["random_init"] = rand.eval()
    return encs


@torch.no_grad()
def embed_batches(enc, noise_scale, ctx, seed_off=0, n_batches=N_BATCHES, cfg=None):
    """Yield (h, bt) per holdout batch: unmasked-table embeddings + truth. cfg = the run's ckpt cfg
    (C3: probe on the prior the run trained on); None = the paper prior at this noise_scale.

    The torch seed is reset per call: encode() draws r_j from the global torch
    RNG, so without this every arm's numbers depend on how many arms precede it
    in the grid (adding the Issue #8 arms moved p2_anycell's edge AUC .6607 ->
    .6649). Reset makes each arm see the same tables AND the same column
    identities - a paired comparison, and stable across eval runs.
    """
    torch.manual_seed(EVAL_SEED + seed_off)
    prior = probe_prior(cfg or {}, ctx, EVAL_SEED + seed_off, noise_scale=noise_scale, **CLEAN)
    rng = np.random.default_rng(EVAL_SEED + seed_off)
    for _ in range(n_batches):
        bt = make_batch(prior, BATCH, "y_only", rng, split=ctx, return_truth=True)
        no_mask = torch.zeros_like(bt["input_mask"])
        h = enc.encode(bt["z"], no_mask, bt["split"])
        yield h, bt


def ridge_mse(X, y):
    n = len(X) // 2
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb[:n].T @ Xb[:n] + RIDGE_ALPHA * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[:n].T @ y[:n])
    return float(np.mean((Xb[n:] @ w - y[n:]) ** 2))


def c4_collect(enc, noise_scale, ctx=CTX, max_n=24000, cfg=None):
    """Context-cell embeddings + paired (z_f, z_x) targets, non-root continuous cells only
    (C4: a categorised column's x is a code, its f is not - never scored against truth)."""
    H, F_, X = [], [], []
    for h, bt in embed_batches(enc, noise_scale, ctx, cfg=cfg):
        keep = truth_cells(bt)
        keep = keep.expand(-1, ctx, -1)  # context rows only
        H.append(h[:, :ctx][keep])
        F_.append(bt["z_f"][:, :ctx][keep].clamp(-3.05, 3.05))
        X.append(bt["z"][:, :ctx][keep].clamp(-3.05, 3.05))
    H = torch.cat(H).numpy()[:max_n]
    return H, torch.cat(F_).numpy()[:max_n], torch.cat(X).numpy()[:max_n]


def c5_collect(enc, ctx=CTX, noise_scale=1.0, cfg=None):
    """Row-averaged column embeddings + column-pair edge labels + node labels."""
    cols, fams, degs, pairs, edges = [], [], [], [], []
    for h, bt in embed_batches(enc, noise_scale, ctx, cfg=cfg):
        e = h[:, :ctx].mean(1)  # (B,D,E)
        adj = bt["adjacency"].float()
        deg = (adj.sum(1) + adj.sum(2))  # (B,N) total degree
        for i in range(e.shape[0]):
            obs = bt["observed_idx"][i]
            D = len(obs)
            cols.append(e[i])
            fams.append(bt["cell_family"][i])
            degs.append(deg[i][obs])
            a = (adj[i][obs][:, obs] + adj[i][obs][:, obs].T).bool()
            ii, jj = torch.triu_indices(D, D, offset=1)
            f_sym = torch.cat([e[i][ii] + e[i][jj], (e[i][ii] - e[i][jj]).abs(),
                               e[i][ii] * e[i][jj]], -1)
            pairs.append(f_sym)
            edges.append(a[ii, jj])
    return (torch.cat(cols).numpy(), torch.cat(fams).numpy(),
            torch.cat(degs).numpy(), torch.cat(pairs).numpy(),
            torch.cat(edges).numpy())


def c5_scores(enc, ctx=CTX, noise_scale=1.0, cfg=None):
    cols, fams, degs, pairs, edges = c5_collect(enc, ctx, noise_scale, cfg=cfg)
    nc, np_ = len(cols) // 2, len(pairs) // 2
    lr = LogisticRegression(max_iter=2000).fit(pairs[:np_], edges[:np_])
    edge_auc = roc_auc_score(edges[np_:], lr.predict_proba(pairs[np_:])[:, 1])
    fam_lr = LogisticRegression(max_iter=2000).fit(cols[:nc], fams[:nc])
    fam_acc = float((fam_lr.predict(cols[nc:]) == fams[nc:]).mean())
    fam_maj = float(np.bincount(fams[nc:].astype(int)).max() / len(fams[nc:]))
    deg_mse = ridge_mse(cols, degs)
    deg_base = float(((degs[len(degs)//2:] - degs[:len(degs)//2].mean()) ** 2).mean())
    return dict(edge_auc=round(float(edge_auc), 4), fam_acc=round(fam_acc, 4),
                fam_majority=round(fam_maj, 4), deg_mse=round(deg_mse, 4),
                deg_mse_meanpred=round(deg_base, 4))


@torch.no_grad()
def pred_f_recovery(noise_scale, run="p3_jepa", cfg=None):
    """Secondary: masked-cell predicted-latent probe -> f, per noise scale."""
    jepa = load_jepa(run)
    prior = probe_prior(cfg or {}, CTX, EVAL_SEED + 7, noise_scale=noise_scale, **CLEAN)
    rng = np.random.default_rng(EVAL_SEED + 7)
    X, y = [], []
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, "any_cell", rng, split=CTX, return_truth=True)
        m = bt["target_mask"] & truth_cells(bt)
        X.append(jepa.predict_latent(bt["z"], bt["input_mask"], m, bt["split"]))
        y.append(bt["z_f"][m].clamp(-3.05, 3.05))
    return ridge_mse(torch.cat(X).numpy(), torch.cat(y).numpy())


RUNS_OUT = ROOT / "eval/results_truth_eval_runs.json"


def score_runs(runs):
    """C3 run mode: structure (C5) + value/f (C4) read-outs of named runs on the prior each run trained on,
    merged by run into eval/results_truth_eval_runs.json.
    Run: uv run --with pyarrow python -m eval.truth_eval RUN [RUN ...]"""
    res = json.loads(RUNS_OUT.read_text()) if RUNS_OUT.exists() else {}
    for run in runs:
        enc, cfg = load_auto(run), load_cfg(run)
        H, F_, X = c4_collect(enc, 1.0, cfg=cfg)
        res[run] = dict(prior_source=cfg.get("prior_source") or "paper", c5=c5_scores(enc, cfg=cfg),
                        c4=dict(mse_f=round(ridge_mse(H, F_), 4), mse_x=round(ridge_mse(H, X), 4), n=int(len(H))),
                        protocol=dict(ctx=CTX, n_query=N_QUERY, n_batches=N_BATCHES, batch=BATCH, eval_seed=EVAL_SEED))
        print(run, res[run], flush=True)
    RUNS_OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", RUNS_OUT.relative_to(ROOT))


def main():
    if len(sys.argv) > 1:
        return score_runs(sys.argv[1:])
    encs = encoders()
    results = {"c4": {}, "c5": {}, "icl": {}, "pred_f": {}}

    for s in SIGMAS:
        for name, enc in encs.items():
            H, F_, X = c4_collect(enc, s)
            results["c4"][f"{name}|{s}"] = dict(mse_f=round(ridge_mse(H, F_), 4),
                                                mse_x=round(ridge_mse(H, X), 4))
            print(f"C4 sigma={s:4} {name:12} {results['c4'][f'{name}|{s}']}", flush=True)

    for name, enc in encs.items():
        results["c5"][name] = c5_scores(enc)
        print(f"C5 {name:12} {results['c5'][name]}", flush=True)

    for ctx in ICL_CTX:
        for name, enc in encs.items():
            H, F_, X = c4_collect(enc, 1.0, ctx=ctx, max_n=16000)
            results["icl"][f"{name}|{ctx}"] = dict(
                mse_f=round(ridge_mse(H, F_), 4),
                edge_auc=c5_scores(enc, ctx=ctx)["edge_auc"])
            print(f"ICL ctx={ctx:3} {name:12} {results['icl'][f'{name}|{ctx}']}", flush=True)

    for s in SIGMAS:
        results["pred_f"][str(s)] = round(pred_f_recovery(s), 4)
    print("pred_f:", results["pred_f"], flush=True)

    (ROOT / "eval/results_p4.json").write_text(json.dumps(results, indent=1))
    print("wrote eval/results_p4.json")


if __name__ == "__main__":
    main()
