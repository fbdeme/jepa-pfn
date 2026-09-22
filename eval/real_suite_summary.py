"""Merge the real-prior suite passes and aggregate them by benchmark, task and census stratum
(docs/real_prior_plan.md 10.3 / 10.4 rule 2, real-data side).

Inputs  eval/results_suite_real_ds.json, eval/results_suite_real_dual_*.json  (same protocol: CTX 1024, K_FEAT 64,
        each pass = some PFN runs + the classical baselines; baselines are deterministic and must agree across passes)
        eval/results_realdata_shapes.json (census; join on bench|name -> rho=k90_over_d, noise=error_histgb,
        cat_frac, D=n_features, R=n_rows)
Output  eval/results_suite_real.json: merged per-dataset scores + summary (per bench/task means over seeds,
        paired per-dataset sign tests dual vs ds, 5 census axes x quintiles with Holm, counts vs HistGB / linear)

Run: uv run python -m eval.real_suite_summary
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_real_prior import holm, sign_test                 # noqa: E402

OUT = ROOT / "eval/results_suite_real.json"
AXES = {"rho": "k90_over_d", "noise": "error_histgb", "cat_frac": None, "D": "n_features", "R": "n_rows"}


def metric_of(kind):
    return "r2" if kind == "reg" else "acc"


def census():
    rows = json.loads((ROOT / "eval/results_realdata_shapes.json").read_text())["rows"]
    out = {}
    for r in rows:
        out[(r["bench"], r["name"])] = {"rho": r.get("k90_over_d"), "noise": r.get("error_histgb"),
                                        "cat_frac": r["n_categorical"] / max(r["n_features"], 1),
                                        "D": r["n_features"], "R": r["n_rows"], "task": r["task"]}   # rho/noise None where the census could not measure them
    return out


BASELINE_DRIFT = {"max": 0.0, "where": None}


def merge(paths):
    """Union of the passes; a baseline seen twice keeps the first pass's value and the largest disagreement is
    recorded (the classical fits are not bit-reproducible across boxes: logreg lbfgs differs in the 3rd decimal)."""
    merged = {}
    for p in paths:
        for key, models in json.loads(p.read_text()).items():
            slot = merged.setdefault(key, {})
            for m, v in models.items():
                if m in slot:
                    for mk, mv in v.items():
                        d = abs(slot[m][mk] - mv)
                        if d > BASELINE_DRIFT["max"]:
                            BASELINE_DRIFT.update(max=round(d, 4), where=f"{key}:{m}:{mk}")
                    continue
                slot[m] = v
    return merged


def seeds(models, arm):
    return sorted(m for m in models if m.startswith(f"real_{arm}_"))


def strata(rows, axis):
    rows = [r for r in rows if r.get(axis) is not None]
    if not rows:
        return []
    v = np.array([r[axis] for r in rows], float)
    edges = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, v, side="right")
    cells = []
    for i in range(5):
        m = q == i
        if not m.any():
            continue
        g = np.array([r["gap"] for r in rows])[m]
        neg, pos = int((g < 0).sum()), int((g > 0).sum())            # gap = dual - ds on the metric (higher better)
        cells.append({"axis": axis, "q": i, "lo": float(v[m].min()), "hi": float(v[m].max()), "n": int(m.sum()),
                      "dual_wins": pos, "ds_wins": neg, "median_gap": round(float(np.median(g)), 4), "p": sign_test(neg, pos)})
    return cells


def main():
    paths = sorted((ROOT / "eval").glob("results_suite_real_d*.json"))
    paths = [p for p in paths if p.name != OUT.name]
    merged = merge(paths)
    cen = census()
    per = []
    for key, models in merged.items():
        bench, name, kind = key.split("|")
        met = metric_of(kind)
        ds, du = seeds(models, "ds"), seeds(models, "dual")
        rec = {"key": key, "bench": bench, "task": kind, "metric": met,
               "ds_seeds": {m: models[m][met] for m in ds}, "dual_seeds": {m: models[m][met] for m in du},
               "ds_mean": float(np.mean([models[m][met] for m in ds])) if ds else None,
               "dual_mean": float(np.mean([models[m][met] for m in du])) if du else None,
               "linear": models.get("logreg", models.get("ridge", {})).get(met),
               "histgb": models.get("histgb", models.get("histgbr", {})).get(met)}
        if ds and du:
            rec["gap"] = rec["dual_mean"] - rec["ds_mean"]
        rec.update(cen.get((bench, name), {}))
        per.append(rec)
    paired = [r for r in per if "gap" in r]
    summary = {"inputs": [p.name for p in paths], "baseline_drift_across_passes": BASELINE_DRIFT,
               "n_datasets": len(per), "n_paired": len(paired),
               "n_with_census": sum(1 for r in paired if "rho" in r)}
    # per bench / task means and paired sign tests
    for grp_name, grp_fn in (("by_bench", lambda r: r["bench"]), ("by_task", lambda r: r["task"]), ("all", lambda r: "all")):
        d = {}
        for r in paired:
            d.setdefault(grp_fn(r), []).append(r)
        summary[grp_name] = {g: {"n": len(rs), "ds_mean": round(float(np.mean([r["ds_mean"] for r in rs])), 4),
                                 "dual_mean": round(float(np.mean([r["dual_mean"] for r in rs])), 4),
                                 "dual_wins": int(sum(r["gap"] > 0 for r in rs)), "ds_wins": int(sum(r["gap"] < 0 for r in rs)),
                                 "p": sign_test(int(sum(r["gap"] < 0 for r in rs)), int(sum(r["gap"] > 0 for r in rs))),
                                 "ds_ge_histgb": int(sum(r["ds_mean"] >= r["histgb"] for r in rs if r["histgb"] is not None)),
                                 "dual_ge_histgb": int(sum(r["dual_mean"] >= r["histgb"] for r in rs if r["histgb"] is not None)),
                                 "ds_ge_linear": int(sum(r["ds_mean"] >= r["linear"] for r in rs if r["linear"] is not None)),
                                 "dual_ge_linear": int(sum(r["dual_mean"] >= r["linear"] for r in rs if r["linear"] is not None))}
                             for g, rs in d.items()}
    # census strata (paired + census-joined only)
    joined = [r for r in paired if "D" in r]
    cells = [c for ax in AXES for c in strata(joined, ax)]
    rej = holm([c["p"] for c in cells])
    for j, c in enumerate(cells):
        c["holm_significant"] = j in rej
        c["direction"] = "dual" if c["median_gap"] > 0 else "ds"
    summary["strata"] = {"n": len(joined), "cells": cells,
                         "significant": [{"axis": c["axis"], "q": c["q"], "direction": c["direction"]} for c in cells if c["holm_significant"]]}
    out = {"protocol": "CTX 1024, K_FEAT 64, N_ENSEMBLE 4, each arm = mean over its 3 seeds; gap = dual - ds on acc (clf) / r2 (reg); "
                       "paired sign test over datasets, Holm over the 25 census cells", "summary": summary, "datasets": per}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_datasets", "n_paired", "n_with_census")}))
    for g, v in summary.get("all", {}).items():
        print("all:", v)
    for g, v in summary.get("by_bench", {}).items():
        print(f"{g}: ds {v['ds_mean']} dual {v['dual_mean']} dual_wins {v['dual_wins']} ds_wins {v['ds_wins']} p {v['p']:.3g}")
    print("strata significant:", summary["strata"]["significant"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
