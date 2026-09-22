"""Phase 3 first comparison (SSOT section 4): linear probe on predicted latents.

For masked holdout cells, fit a ridge probe z_hat_latent -> value and compare
test MSE against (i) the same probe on a RANDOM-INIT encoder (SSOT baseline
iii) and (ii) p2_anycell's direct bar-dist prediction (data-space control,
identical masking) read from eval/results_p2.json.

Same eval seeds as eval/context_curve.py -> identical holdout tables/masks.

Run: uv run python -m eval.latent_probe
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.jepa import JEPA
from model.pfn import CellPFN
from prior.scm import FuncFamily
from train.data import make_batch
from train.train import _make_prior

ROOT = Path(__file__).parents[1]
CTX_SIZES = [10, 25, 50, 100]
N_QUERY = 50
N_BATCHES = 20
BATCH = 16
EVAL_SEED = 20_000  # keep aligned with eval/context_curve.py
RIDGE_ALPHA = 1.0


def load_cfg(run):
    return torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)["cfg"]


def load_cellpfn(run):
    ck = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample", n_classes=c.get("n_classes") or 0)
    m.load_state_dict(ck["model"])
    return m.eval()


def load_auto(run):
    """Data-space arms are a bare CellPFN; latent arms are a JEPA whose .encoder() is what
    every probe in this repo reads."""
    ck = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    if any(k.startswith("online.") for k in ck["model"]):
        m = load_jepa(run)
        enc = m.encoder().eval()
        if m.ctx_only:   # the encoder's output AT a hidden cell is untrained under ctx_only; the predictor's is the read-out
            enc.jepa = m
        return enc
    return load_cellpfn(run)


def random_init(ref_run, seed):
    c = load_cfg(ref_run)
    torch.manual_seed(seed)
    return CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                   n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample").eval()


def truth_cells(bt):
    """(B, 1, D) bool: cells whose f is a truth to score against - non-root AND continuous columns (C4: a
    categorised column's x is a code, its f is not; the paper prior's CLEAN protocol has no such columns,
    the real prior does). Broadcasts against (B, R, D) masks."""
    return ((bt["cell_family"] != int(FuncFamily.ROOT)) & (bt["categorical"] == 0)).unsqueeze(1)


def probe_prior(cfg, ctx, seed, n_query=N_QUERY, **paper_only):
    """The prior a run was trained on (train.train._make_prior on its ckpt cfg), rows pinned to ctx + n_query
    (C3). cfg {} = the paper prior with the probe protocol's own knobs (noise_scale, CLEAN) - exactly the
    pre-C3 construction; those knobs exist only on PriorConfig and are dropped for the real / factor priors,
    whose truth (SCMBatch: f, adjacency, family, sigma, hyper) the probes read unchanged."""
    n = ctx + n_query
    prior = {**cfg.get("prior", {}), "n_rows": (n, n)}
    if not cfg.get("prior_source"):
        prior.update(paper_only)
    return _make_prior({**cfg, "prior": prior}, seed, None)


def load_jepa(run_name):
    ck = torch.load(ROOT / "runs" / run_name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    c = ck["cfg"]
    m = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
             c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
             c.get("predictor", "mlp"), c.get("mode", "ema"),
             c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
             c.get("r_invariant", False), c.get("n_cls", 0),
             c.get("cls_target", False), c.get("r_scheme", "resample"), n_classes=c.get("n_classes", 0),
             ctx_only=c.get("ctx_only", False), fresh_target_r=c.get("fresh_target_r", False),
             diff_target=c.get("diff_target", False), head_on_pred=c.get("head_on_pred", False),
             lambda_jepa=c.get("lambda_jepa", 1.0), dae_sigma=c.get("dae_sigma", 0.5),
             sigreg_detach=c.get("sigreg_detach", True), sigreg_additive=c.get("sigreg_additive", False),
             ckpt_pass2=c.get("ckpt_pass2", False))
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def collect(model, ctx, cfg=None):
    """(features z_hat, targets z_true-on-head-support) over the holdout grid cell."""
    prior = probe_prior(cfg or {}, ctx, EVAL_SEED + ctx)
    rng = np.random.default_rng(EVAL_SEED + ctx)
    X, y = [], []
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, "any_cell", rng, split=ctx)
        m = bt["target_mask"]
        X.append(model.predict_latent(bt["z"], bt["input_mask"], m, bt["split"]))
        y.append(bt["z"][m].clamp(-3.05, 3.05))
    return torch.cat(X).numpy(), torch.cat(y).numpy()


def ridge_mse(X, y):
    n = len(X) // 2  # first half fits the probe, second half scores it
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb[:n].T @ Xb[:n] + RIDGE_ALPHA * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[:n].T @ y[:n])
    return float(np.mean((Xb[n:] @ w - y[n:]) ** 2))


def main():
    torch.manual_seed(999)
    cfg = torch.load(ROOT / "runs/p3_jepa/ckpt.pt", map_location="cpu",
                     weights_only=False)["cfg"]
    models = {"p3_jepa": load_jepa("p3_jepa"),
              "random_init": JEPA(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"],
                                  cfg["n_bins"]).eval()}
    ref = json.loads((ROOT / "eval/results_p2.json").read_text())

    results = {}
    for ctx in CTX_SIZES:
        for name, model in models.items():
            X, y = collect(model, ctx)
            results[f"{name}|{ctx}"] = round(ridge_mse(X, y), 4)
        results[f"p2_anycell_direct|{ctx}"] = ref[f"p2_anycell|any_cell|{ctx}"]["mse"]
        print(f"ctx={ctx:3}  jepa_probe={results[f'p3_jepa|{ctx}']:.4f}  "
              f"random_probe={results[f'random_init|{ctx}']:.4f}  "
              f"p2_anycell_direct={results[f'p2_anycell_direct|{ctx}']:.4f}", flush=True)
    (ROOT / "eval/results_p3_probe.json").write_text(json.dumps(results, indent=1))
    print("wrote eval/results_p3_probe.json")


if __name__ == "__main__":
    main()
