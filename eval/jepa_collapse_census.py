"""Collapse census over every local JEPA run (runs/*/metrics.jsonl with val_jepa), derived, never hand-typed.

collapsed / onset / recovered = eval/collapse.py (the single criterion: last-validation dim_std < THR_DIM_STD or, when logged,
within_table_var_frac < THR_WITHIN_TABLE). Cells = (prior, anti-collapse mode, value-head weight lambda_ppd, width emb, lr, policy).
Output eval/results_jepa_collapse_census.json.  Run: uv run python -m eval.jepa_collapse_census [--table]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.collapse import THR_DIM_STD, THR_WITHIN_TABLE, trajectory   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eval/results_jepa_collapse_census.json")


def census():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs/*/"))):
        m, c = os.path.join(d, "metrics.jsonl"), os.path.join(d, "config.yaml")
        if not (os.path.exists(m) and os.path.exists(c)):
            continue
        cfg = yaml.safe_load(open(c))
        if "policy_main" not in cfg and "lambda_ppd" not in cfg:
            continue
        vals = [r for r in map(json.loads, open(m)) if "val_jepa" in r]
        if not vals:
            continue
        last, tj = vals[-1], trajectory(vals)
        rows.append({"run": os.path.basename(d.rstrip("/")), "prior": cfg.get("prior_source", "paper"), "mode": cfg.get("mode", "ema"),
                     "lambda_ppd": cfg.get("lambda_ppd"), "lambda_sig": cfg.get("lambda_sig") if cfg.get("mode") == "sigreg" else None,
                     "policy": cfg.get("policy_main"), "lr": cfg.get("lr"), "emb": cfg.get("emb"), "steps": cfg.get("steps"), "last_step": last["step"],
                     "shuffle_target": bool(cfg.get("shuffle_target", False)), "dim_std": last["val_tgt"]["dim_std"], "cos": last["val_tgt"]["cos"],
                     "erank": last["val_tgt"]["erank"], "within_table_var_frac": last["val_tgt"].get("within_table_var_frac"),
                     "collapsed": tj["collapsed"], "onset": tj["onset"], "recovered": tj["recovered"],
                     "complete": bool(last["step"] >= (cfg.get("steps") or 0))})
    return rows


def cells(rows):
    g = defaultdict(list)
    for r in rows:
        if not r["complete"] or r["shuffle_target"]:
            continue
        key = (r["prior"], r["mode"], "head" if (r["lambda_ppd"] or 0) > 0 else "nohead", r["emb"], r["lr"])
        g[key].append(r)
    out = []
    for k, rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0, kv[0][4] or 0)):
        out.append({"prior": k[0], "mode": k[1], "head": k[2], "emb": k[3], "lr": k[4], "n": len(rs), "n_collapsed": sum(r["collapsed"] for r in rs),
                    "runs": [r["run"] for r in rs]})
    return out


def main():
    rows = census(); cl = cells(rows)
    if "--table" in sys.argv:
        print("| prior | mode | value head | emb | lr | runs | collapsed |\n|---|---|---|---|---|---|---|")
        for c in cl:
            print(f"| {c['prior']} | {c['mode']} | {c['head']} | {c['emb']} | {c['lr']} | {c['n']} | {c['n_collapsed']} |")
        return
    json.dump({"threshold_dim_std": THR_DIM_STD, "threshold_within_table": THR_WITHIN_TABLE, "n_runs": len(rows), "cells": cl, "runs": rows}, open(OUT, "w"), indent=1)
    print("runs", len(rows), "cells", len(cl), "-> wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
