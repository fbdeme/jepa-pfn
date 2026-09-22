"""Held-out curves of the open-horizon pair (paper_v5 Figure 2) from the fetched runs/<run>/metrics.jsonl, plus the
constant-map MSE of each arm's held-out stream, recomputed from the prior exactly as the trainers' validate() draws it
(val seed 10_000 + seed, the run's val policy and batch count). Constant map = the best constant prediction, the mean of
the held-out target cells (clamped to the value head's support), i.e. their variance; a collapsed encoder can do no better.
zero_map_mse (predicting 0, eval/rows_control.py's convention) is kept for reference. Writes curves.json next to this file.
Run: .venv/bin/python eval/open_horizon_2026-09-14/extract_curves.py"""
import json, sys
from pathlib import Path

import numpy as np, torch, yaml

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
sys.path.insert(0, str(R))
RUNS = {"PFN_tabicl2_ds_conv_s0": "ds", "TabularJEPA_v3_tabicl2_headenc_conv_s0": "jepa",
        # the fixed-budget pair behind "collapsed in our earlier runs": same recipe, one key apart (head_on_pred)
        "TabularJEPA_v3_tabicl2_s0": "pred", "TabularJEPA_v3_tabicl2_headenc_s0": "enc"}
PAIR = ("PFN_tabicl2_ds_conv_s0", "TabularJEPA_v3_tabicl2_headenc_conv_s0")


def held_out_cells(cfg, arm):
    """The target cells of the run's held-out stream, in validation order (train/train.py:validate, train_jepa.py:validate)."""
    from train.data import make_batch
    from model.pfn import CellPFN
    mp = __import__("train.train" if arm == "ds" else "train.train_jepa", fromlist=["_make_prior"])._make_prior
    seed = 10_000 + cfg["seed"]
    prior, rng = mp(cfg, seed, "val"), np.random.default_rng(seed)
    policy = cfg["policy"] if arm == "ds" else cfg["val_policy"]
    lo, hi = (c := CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], n_bins=cfg["n_bins"]).bin_centers)[0], c[-1]
    cells = []
    for _ in range(cfg["val_batches"]):
        bt = make_batch(prior, cfg["batch_size"], policy, rng, "cpu")
        z = bt["z"][bt["target_mask"]]
        assert not torch.isnan(z).any()
        cells.append(z.clamp(lo, hi))
    return cells, dict(val_seed=seed, val_policy=policy, val_batches=cfg["val_batches"], batch_size=cfg["batch_size"],
                       val_tables=cfg["val_batches"] * cfg["batch_size"], support=[round(float(lo), 2), round(float(hi), 2)])


def main():
    out = {}
    for r, arm in RUNS.items():
        p = R / "runs" / r / "metrics.jsonl"
        if not p.exists():
            print("skip (not fetched)", r); continue
        cfg = yaml.safe_load((R / "runs" / r / "config.yaml").read_text())
        L = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        va = [x for x in L if "val_mse" in x]; stop = [x for x in L if "stop" in x]
        cells, meta = held_out_cells(cfg, arm)
        allz = torch.cat(cells)
        const, zero = float(allz.var(unbiased=False)), float(allz.pow(2).mean())
        below = [x["step"] for x in va if x["val_mse"] < const]
        first = below[0] if below else None
        lo_v = min(va, key=lambda x: x["val_mse"])
        s = dict(arm=arm, stop_step=stop[-1]["step"] if stop else None, **meta, n_cells=int(sum(len(c) for c in cells)),
                 min_val_mse=lo_v["val_mse"], min_step=lo_v["step"], final_val_mse=va[-1]["val_mse"], final_step=va[-1]["step"],
                 const_map_mse=round(const, 4), cell_mean=round(float(allz.mean()), 4), zero_map_mse=round(zero, 4),
                 per_batch_zero=[round(float(c.pow(2).mean()), 4) for c in cells],
                 first_below_const_map_step=first,
                 n_above_after_first_below=sum(1 for x in va if first is not None and x["step"] > first and x["val_mse"] >= const),
                 val=[dict(step=x["step"], val_mse=x["val_mse"], **({"val_jepa": x["val_jepa"], "tgt_cos": x["val_tgt"]["cos"],
                           "tgt_erank": x["val_tgt"]["erank"]} if "val_jepa" in x else {})) for x in va])
        out[r] = s
        print(r, arm, "const", s["const_map_mse"], "cells", s["n_cells"], "first below", first, "above after", s["n_above_after_first_below"], "stop", s["stop_step"])
    if all(r in out for r in PAIR):   # both trainers draw the same held-out stream from the same seed: the ds set is a prefix of the jepa set
        a, b = (out[r]["per_batch_zero"] for r in PAIR)
        out["_note"] = ("ds held-out set == first %d batches of the jepa set (per-batch E[z^2] identical)" % len(a)) if a == b[:len(a)] \
            else "WARNING: ds and jepa held-out streams differ"
        print(out["_note"])
    (D / "curves.json").write_text(json.dumps(out, indent=1) + "\n"); print("wrote", D / "curves.json")


if __name__ == "__main__":
    main()
