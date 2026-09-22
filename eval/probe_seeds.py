"""Seed-CI mechanism probe: run the base probe on ALL seeded arms + a multi-draw
random-init noise floor, saving per-seed / per-draw records (source-backed-numbers).

Resolves the Table-2 "n=1" objection (review A1/C-3): trained arms get a seed
distribution and the noise floor becomes a band instead of a single draw.

- Trained arms: base_{ds,dual,lat_s}_s{0..4} -> edge_auc, fam_acc, mse_f per seed.
- Noise floor: N_FLOOR random-init CellPFN draws (varying init seed) -> floor dist.
Eval tables are identical across every encoder (collect() fixes EVAL_SEED), so the
only variation is the trained-seed / init-draw -> a clean paired comparison.

Run: uv run python -m eval.probe_seeds   (CPU, forward-only, ~1h)
Output: eval/results_probe_seeds.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.probe_base import ROOT, SIGMAS, c4_mse_f, c5_scores, collect, load_cellpfn
from eval.latent_probe import load_jepa
from model.pfn import CellPFN

SEEDS = range(5)
N_FLOOR = 30


def arm_metrics(enc):
    """Same protocol as probe_base.main(): one collect at noise 1.0 shared by C4+C5,
    then C4 f-MSE re-collected at the higher sigmas."""
    b1 = collect(enc, 1.0)
    edge_auc, fam_acc, fam_maj = c5_scores(b1)
    mse_f = {"1.0": c4_mse_f(b1)}
    for s in SIGMAS[1:]:
        mse_f[str(s)] = c4_mse_f(collect(enc, s))
    return dict(edge_auc=edge_auc, fam_acc=fam_acc, fam_majority=fam_maj, mse_f=mse_f)


def rand_encoder(cfg, seed):
    torch.manual_seed(seed)
    return CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"],
                   n_cls=cfg.get("n_cls") or 0,
                   r_scheme=cfg.get("r_scheme") or "resample").eval()


def main():
    cfg = torch.load(ROOT / "runs" / "base_ds_s0" / "ckpt.pt", map_location="cpu",
                     weights_only=False)["cfg"]
    res = {"trained": {"ds": [], "dual": [], "lat_s": []}, "floor": [],
           "arch": {k: cfg[k] for k in ("emb", "heads", "mlp", "layers", "n_bins",
                                        "n_cls", "r_scheme")},
           "protocol": {"seeds": list(SEEDS), "n_floor": N_FLOOR}}
    loaders = {"ds": lambda n: load_cellpfn(n),
               "dual": lambda n: load_jepa(n).encoder().eval(),
               "lat_s": lambda n: load_jepa(n).encoder().eval()}
    for k in SEEDS:
        for arm, load in loaders.items():
            n = f"base_{arm}_s{k}"
            t = time.time()
            m = arm_metrics(load(n))
            m["seed"] = k
            res["trained"][arm].append(m)
            print(f"{n:16} edge={m['edge_auc']} fam={m['fam_acc']} "
                  f"mse_f1={m['mse_f']['1.0']} [{time.time()-t:.0f}s]", flush=True)
    for d in range(N_FLOOR):
        t = time.time()
        m = arm_metrics(rand_encoder(cfg, 10_000 + d))
        m["draw"] = d
        res["floor"].append(m)
        print(f"floor_draw_{d:<7} edge={m['edge_auc']} fam={m['fam_acc']} "
              f"mse_f1={m['mse_f']['1.0']} [{time.time()-t:.0f}s]", flush=True)
    out = ROOT / "eval/results_probe_seeds.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
