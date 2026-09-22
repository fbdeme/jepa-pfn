"""Fixed-R value MSE per real-prior run + the pre-registered lr selection (docs/real_prior_plan.md 10.2).

eval/rows_control.py generalised: run list from the configs (real_{ds,dual}_lr*_s*), the prior built by
train.train._make_prior from the run's own cfg with n_rows pinned to R (so the real prior's cell budget,
categoricals and designated target are all in force, identically for every arm), each arm scored under
its own training policy, const-map anchor = predicting 0 on context-standardised cells.

lr selection (pre-registered): per arm, mean of val_mse over R in SELECT_ROWS, lowest mean wins; a win at
either grid end is flagged (edge rule: add one point x3 in that direction, re-select once).

Run: uv run python -m eval.real_readout [RUN ...]   -> eval/results_real_readout.json
     (DEVICE=cuda to score on a GPU box: R=1024 x D<=65 tables are slow on a loaded CPU; results are
      device-independent up to float noise, the const-map anchor is exact either way)
"""
import os
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
DEVICE = os.environ.get("DEVICE", "cpu")
from eval.rows_control import _model                            # noqa: E402
from train.data import make_batch                               # noqa: E402
from train.train import _make_prior                             # noqa: E402

# READOUT_OUT=eval/results_real_readout_v4.json keeps the v4 grid out of the POST lr-selection artifact (its
# "selection" is a pre-registered verdict; v4 runs at the same lr would silently overwrite those candidates).
OUT = ROOT / os.environ.get("READOUT_OUT", "eval/results_real_readout.json")
ROWS = (16, 64, 128, 384, 1024)
SELECT_ROWS = (64, 128, 384, 1024)
LEARNED = 0.1
# real_{arm}_lr{lr}_s{seed}  and the v4 rule  v4_{arm}_{mech}_lam{l}_lr{lr}_{policy}_{target}[_{steps}k]_s{seed}  (C13)
PAT = re.compile(r"^(?:real|v4)_(ds|dual)_(?:[^_].*?_)?lr([0-9.e-]+)_(?:.*?_)?s(\d+)$")


@torch.no_grad()
def mse_at(enc, cfg, policy, R, batches=8, batch=8):
    val_seed = 10_000 + cfg["seed"]
    prior = _make_prior({**cfg, "prior": {**cfg.get("prior", {}), "n_rows": [R, R]}}, val_seed, None)
    rng = np.random.default_rng(val_seed)
    k = getattr(enc, "n_cls", 0) or 0
    se = se_const = 0.0
    n = 0
    for _ in range(batches):
        bt = make_batch(prior, batch, policy, rng, device=DEVICE)
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        logits = enc.head(h[:, :, k:] if k else h)
        m = bt["target_mask"]
        z_sup = bt["z"][m].clamp(enc.bin_centers[0], enc.bin_centers[-1])
        se += ((enc.point_pred(logits)[m] - z_sup) ** 2).sum().item()
        se_const += (z_sup ** 2).sum().item()
        n += int(m.sum())
    return round(se / n, 4), round(se_const / n, 4), n


def select(runs):
    """Per arm: best lr by mean val_mse over SELECT_ROWS (seed-0 runs only), edge flag."""
    sel = {}
    for arm in ("ds", "dual"):
        cands = {}
        for run, rec in runs.items():
            mt = PAT.match(run)
            if mt and mt.group(1) == arm and mt.group(3) == "0":
                cands[mt.group(2)] = round(float(np.mean([rec["by_R"][str(R)]["val_mse"] for R in SELECT_ROWS])), 4)
        if not cands:
            continue
        best = min(cands, key=cands.get)
        lrs = {k: float(k) for k in cands}                       # the grid is whatever has been run, edge points included
        hi, lo = max(lrs, key=lrs.get), min(lrs, key=lrs.get)
        edge = best in (hi, lo)
        sel[arm] = {"candidates": cands, "best_lr": best, "at_grid_edge": edge,
                    "edge_rule": ("add lr x3 above" if best == hi else "add lr /3 below") if edge else None}
    return sel


def main():
    runs_arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    RUNS = runs_arg or sorted(p.name for p in (ROOT / "runs").glob("real_*_lr*_s*") if (p / "ckpt.pt").exists())
    out = json.loads(OUT.read_text()) if OUT.exists() and runs_arg else \
        {"protocol": {"rows": list(ROWS), "select_rows": list(SELECT_ROWS), "val_batches": 8, "batch": 8,
                      "val_seed": "10000+seed", "policy": "each arm under its own training policy",
                      "const_map": "MSE of predicting 0 (context-standardised cells)",
                      "prior": "run cfg prior_source/prior with n_rows pinned to R (cell budget in force)",
                      "learned_rule": f"val_mse < const_map - {LEARNED}"}, "runs": {}}
    for run in RUNS:
        if not (ROOT / "runs" / run / "ckpt.pt").exists():
            print(f"  skip {run}: no ckpt", flush=True)
            continue
        enc, cfg, policy = _model(run)
        enc = enc.to(DEVICE)
        rec = {"lr": cfg["lr"], "seed": cfg["seed"], "batch_size": cfg["batch_size"], "policy": policy, "by_R": {}}
        for R in ROWS:
            m, c, n = mse_at(enc, cfg, policy, R)
            rec["by_R"][str(R)] = {"val_mse": m, "const_map": c, "n_cells": n, "learned": m < c - LEARNED}
        rec["select_mean"] = round(float(np.mean([rec["by_R"][str(R)]["val_mse"] for R in SELECT_ROWS])), 4)
        out["runs"][run] = rec
        print(f"  {run:24} lr={cfg['lr']:<8} " + "  ".join(f"R{R}={rec['by_R'][str(R)]['val_mse']:.3f}" for R in ROWS)
              + f"  mean4={rec['select_mean']:.3f}", flush=True)
    out["selection"] = select(out["runs"])
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("selection", json.dumps(out["selection"]))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
