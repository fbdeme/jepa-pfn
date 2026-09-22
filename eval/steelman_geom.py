"""Recompute the steelman collapse geometry (dim_std / erank / cos) from the checkpoints.

The big latent arms collapse late in training, so the paper probes a pre-collapse *steelman*
checkpoint and must report its collapse geometry. Those numbers were originally read off the
(now-lost) training logs; this script regenerates them deterministically from the checkpoints
using the SAME definition the training loop logs (model.jepa.collapse_stats on the target
encoder's masked-cell representations, via train.train_jepa.validate).

erank is entropy of the singular-value spectrum over a 512-row subsample, so it fluctuates a
little with the subsample; we average over a few torch seeds and record the range. Run (slow,
CPU, a few minutes per model):

    uv run python -m eval.steelman_geom

Writes eval/results_steelman_geom.json, which paper/build_tables.py turns into \num macros.
"""
import json
import statistics as st
from pathlib import Path

import torch

from eval.latent_probe import load_jepa
from train.train_jepa import validate

ROOT = Path(__file__).parents[1]
ARMS = ["big_lat_s_h", "big_dual_lewm_h"]
N_SEEDS = 5
VAL_BATCHES = 16


def main():
    out = {}
    for name in ARMS:
        ck = torch.load(ROOT / "runs" / name / "ckpt.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        cfg["val_batches"] = VAL_BATCHES
        m = load_jepa(name)
        ds, er, co = [], [], []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            tgt = validate(m, cfg, "cpu")["val_tgt"]
            ds.append(tgt["dim_std"]); er.append(tgt["erank"]); co.append(tgt["cos"])
        out[name] = {
            "dim_std": round(st.mean(ds), 3),
            "dim_std_range": [min(ds), max(ds)],
            "erank": round(st.mean(er), 1),
            "erank_range": [min(er), max(er)],
            "cos": round(st.mean(co), 3),
            "n_seeds": N_SEEDS, "val_batches": VAL_BATCHES,
        }
        print(name, out[name])
    with open(ROOT / "eval" / "results_steelman_geom.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote eval/results_steelman_geom.json")


if __name__ == "__main__":
    main()
