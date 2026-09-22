"""Row-count control (todo section 30 / the large-N question, 2026-09-03).

8/24 concluded "R=1024 is a collapse threshold, prior-independent" from runs at a single learning
rate. This session showed the value arm sits at the constant map at lr 1e-3 on one substrate and
trains at lr/10, so the rows question is re-asked with lr as an arm: base recipe, n_rows [10, 1024]
(the code floor is 10 = 8 context + 2 query rows), ds and dual, lr = ratio (1e-3 @ batch 8) and
ratio/10, 5k steps.

Two read-outs per run, both from frozen checkpoints on CPU:
  curve   the logged val_mse (train.validate: mixed-R holdout) per step, from metrics.jsonl
  by_R    value MSE at FIXED row counts R in ROWS, train.validate's protocol otherwise
          (same val seed, same point_pred on head support), so "learns at 1024" is read at
          1024 and not averaged over the range. The constant map's MSE at each R is the anchor.
The base arms (trained on [128, 384]) are scored on the same grid as the in-range reference.

Run: uv run python -m eval.rows_control            -> eval/results_rows_control.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.factor_value_mse import _load_dual                    # noqa: E402
from eval.collapse import is_collapsed                          # noqa: E402
from eval.probe_masked_all import load_cellpfn                  # noqa: E402
from prior.scm import PriorConfig, SCMPrior                     # noqa: E402
from train.data import make_batch                               # noqa: E402

OUT = ROOT / "eval/results_rows_control.json"
ROWS = (16, 64, 128, 384, 1024)
RUNS = [f"rows_{a}_lr{lr}_s0" for a in ("ds", "dual") for lr in ("1e-3", "1e-4")] + ["base_ds_s0", "base_dual_s0", "base_dual_s1"]   # dual s0 is a collapsed seed; s1 is admitted
LEARNED = 0.1        # val_mse must fall below (constant map - LEARNED) to count as learning


def _model(run):
    ck = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    if any(k.startswith("online.") for k in ck["model"]):
        if ck["cfg"].get("head_on_pred") or ck["cfg"].get("ctx_only"):   # I-JEPA-split arms: head on the predictor / on the ctx_only encoder (model.jepa.ValueModel)
            from eval.latent_probe import load_jepa
            from model.jepa import ValueModel
            return ValueModel(load_jepa(run)).eval(), ck["cfg"], ck["cfg"].get("policy_main", "mixed")
        jepa, cfg = _load_dual(run)
        return jepa.online, cfg, cfg.get("policy_main", "mixed")
    enc = load_cellpfn(run)
    return enc, ck["cfg"], ck["cfg"]["policy"]


@torch.no_grad()
def mse_at(enc, cfg, policy, R, batches=8, batch=8):
    val_seed = 10_000 + cfg["seed"]
    prior = SCMPrior(PriorConfig(**{**cfg["prior"], "n_rows": (R, R)}), seed=val_seed)
    rng = np.random.default_rng(val_seed)
    k = getattr(enc, "n_cls", 0) or 0
    se = se_const = 0.0
    n = 0
    for _ in range(batches):
        bt = make_batch(prior, batch, policy, rng)
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        logits = enc.head(h[:, :, k:] if k else h)
        m = bt["target_mask"]
        z_sup = bt["z"][m].clamp(enc.bin_centers[0], enc.bin_centers[-1])
        se += ((enc.point_pred(logits)[m] - z_sup) ** 2).sum().item()
        se_const += (z_sup ** 2).sum().item()      # context-standardised cells: constant map = 0
        n += int(m.sum())
    return round(se / n, 4), round(se_const / n, 4), n


def curve(run):
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    v = [(r["step"], r["val_mse"]) for r in rows if "val_mse" in r]
    if v:
        return {"steps": [s for s, _ in v], "val_mse": [m for _, m in v]}
    v = [r for r in rows if "val_err" in r]          # class-head ds arm (train.train with n_classes)
    if v:
        return {"steps": [r["step"] for r in v], "val_nll": [r["val_nll"] for r in v], "val_err": [r["val_err"] for r in v]}
    # dual arms log the JEPA loss and the target's collapse instruments instead of a value MSE
    v = [r for r in rows if "val_jepa" in r]
    if not v:
        return None
    cv = {"steps": [r["step"] for r in v], "val_jepa": [r["val_jepa"] for r in v],
          "tgt_erank": [r["val_tgt"]["erank"] for r in v], "tgt_dim_std": [r["val_tgt"]["dim_std"] for r in v],
          "tgt_cos": [r["val_tgt"]["cos"] for r in v], "tgt_within": [r["val_tgt"].get("within_table_var_frac") for r in v]}
    if all("cls_nll" in r["val_tgt"] for r in v):     # class-head dual arm: the online encoder's own class NLL / error
        cv["cls_nll"] = [r["val_tgt"]["cls_nll"] for r in v]; cv["cls_err"] = [r["val_tgt"]["cls_err"] for r in v]
    return cv


def _curve_summary(cv):
    if not cv:
        return None
    if "val_mse" in cv:
        return {"first": cv["val_mse"][0], "last": cv["val_mse"][-1], "best": min(cv["val_mse"]), "steps": cv["steps"][-1]}
    if "val_err" in cv:
        return {"nll_first": cv["val_nll"][0], "nll_last": cv["val_nll"][-1], "err_last": cv["val_err"][-1], "steps": cv["steps"][-1]}
    out = {"val_jepa_last": cv["val_jepa"][-1], "tgt_erank_last": cv["tgt_erank"][-1],
           "tgt_dim_std_last": cv["tgt_dim_std"][-1], "tgt_cos_last": cv["tgt_cos"][-1],
           "collapsed": is_collapsed({"dim_std": cv["tgt_dim_std"][-1], "within_table_var_frac": cv["tgt_within"][-1]}), "steps": cv["steps"][-1]}
    if "cls_nll" in cv:
        out["cls_nll_last"] = cv["cls_nll"][-1]; out["cls_err_last"] = cv["cls_err"][-1]
    return out


def main():
    if "--curves-only" in sys.argv:       # refresh the logged curves in an existing artifact, no re-scoring
        out = json.loads(OUT.read_text())
        for run, rec in out["runs"].items():
            rec["curve"] = curve(run)
            rec["curve_summary"] = _curve_summary(rec["curve"])
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        print({r: rec["curve_summary"] for r, rec in out["runs"].items()})
        return
    out = {"protocol": {"rows": list(ROWS), "val_batches": 8, "batch": 8, "val_seed": "10000+seed",
                        "policy": "each arm under its own training policy",
                        "const_map": "MSE of predicting 0 (context-standardised cells)",
                        "learned_rule": f"val_mse < const_map - {LEARNED}"}, "runs": {}}
    for run in RUNS:
        if not (ROOT / "runs" / run / "ckpt.pt").exists():
            print(f"  skip {run}: no ckpt", flush=True)
            continue
        enc, cfg, policy = _model(run)
        rec = {"lr": cfg["lr"], "batch_size": cfg["batch_size"], "n_rows": list(cfg["prior"]["n_rows"]),
               "policy": policy, "curve": curve(run), "by_R": {}}
        for R in ROWS:
            m, c, n = mse_at(enc, cfg, policy, R)
            rec["by_R"][str(R)] = {"val_mse": m, "const_map": c, "n_cells": n, "learned": m < c - LEARNED}
        rec["curve_summary"] = _curve_summary(rec["curve"])
        out["runs"][run] = rec
        print(f"  {run:20} lr={cfg['lr']:<7} " + "  ".join(f"R{R}={rec['by_R'][str(R)]['val_mse']:.3f}" for R in ROWS)
              + (f"  curve {rec['curve_summary']}" if rec["curve_summary"] else ""), flush=True)
    out["_verdict_inputs"] = {r: {R: out["runs"][r]["by_R"][str(R)]["learned"] for R in ROWS}
                              for r in out["runs"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
