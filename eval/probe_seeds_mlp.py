"""Nonlinear (MLP) counterpart of the linear f-MSE probe, at the headline sigma=1.0.

Defends the load-bearing f-MSE claim (§Mechanism) against the standard "a linear probe
underestimates" objection: fit a shallow MLP (not ridge) to recover noise-free f from the
frozen representation, across the SAME seeded arms + 30-draw floor as results_probe_seeds.
Reproduces the linear f-MSE alongside (mse_f_ridge_check) as a consistency check -- it must
match results_probe_seeds.json mse_f["1.0"] since collect() is deterministic.

Run: uv run python -m eval.probe_seeds_mlp   (CPU, ~20min)
Output: eval/results_probe_seeds_mlp.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.probe_base import ROOT, collect, load_cellpfn
from eval.truth_eval import ridge_mse
from eval.latent_probe import load_jepa
from prior.scm import FuncFamily
from model.pfn import CellPFN

SEEDS = range(5)
N_FLOOR = 30
CTX = 128
MAX_N = 24000


def c4_HF(batches, ctx=CTX, max_n=MAX_N):
    """Non-root context-cell embeddings H + noise-free target f (== c4_mse_f's inputs)."""
    H, F_ = [], []
    for h, bt in batches:
        keep = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand(-1, ctx, -1)
        H.append(h[:, :ctx][keep])
        F_.append(bt["z_f"][:, :ctx][keep].clamp(-3.05, 3.05))
    return torch.cat(H).numpy()[:max_n], torch.cat(F_).numpy()[:max_n]


def mlp_mse(X, y, hidden=(256,), seed=0):
    """Shallow MLP probe, same half-split protocol as ridge_mse (fit [:n], MSE on [n:]).
    Fixed random_state so the only cross-arm variation is the encoder, not the probe init.
    A fair (not crippled) nonlinear probe: if f is recoverable nonlinearly, this finds it."""
    n = len(X) // 2
    sc = StandardScaler().fit(X[:n])
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=seed,
                     max_iter=500, early_stopping=True, n_iter_no_change=15)
    m.fit(sc.transform(X[:n]), y[:n])
    return float(np.mean((m.predict(sc.transform(X[n:])) - y[n:]) ** 2))


def metrics(enc):
    H, F_ = c4_HF(collect(enc, 1.0))
    return {"mse_f_mlp": round(mlp_mse(H, F_), 4),
            "mse_f_ridge_check": round(ridge_mse(H, F_), 4)}


def rand_encoder(cfg, seed):
    torch.manual_seed(seed)   # same draw sequence as probe_seeds.rand_encoder
    return CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"],
                   n_cls=cfg.get("n_cls") or 0,
                   r_scheme=cfg.get("r_scheme") or "resample").eval()


def main():
    cfg = torch.load(ROOT / "runs" / "base_ds_s0" / "ckpt.pt", map_location="cpu",
                     weights_only=False)["cfg"]
    res = {"trained": {"ds": [], "dual": [], "lat_s": []}, "floor": [],
           "probe": {"kind": "mlp", "hidden": [256], "sigma": 1.0}}
    loaders = {"ds": lambda n: load_cellpfn(n),
               "dual": lambda n: load_jepa(n).encoder().eval(),
               "lat_s": lambda n: load_jepa(n).encoder().eval()}
    for k in SEEDS:
        for arm, load in loaders.items():
            n = f"base_{arm}_s{k}"
            t = time.time()
            m = metrics(load(n)); m["seed"] = k
            res["trained"][arm].append(m)
            print(f"{n:16} mlp={m['mse_f_mlp']} ridge_check={m['mse_f_ridge_check']} "
                  f"[{time.time()-t:.0f}s]", flush=True)
    for d in range(N_FLOOR):
        t = time.time()
        m = metrics(rand_encoder(cfg, 10_000 + d)); m["draw"] = d
        res["floor"].append(m)
        print(f"floor_{d:<3} mlp={m['mse_f_mlp']} ridge_check={m['mse_f_ridge_check']} "
              f"[{time.time()-t:.0f}s]", flush=True)
    out = ROOT / "eval/results_probe_seeds_mlp.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
