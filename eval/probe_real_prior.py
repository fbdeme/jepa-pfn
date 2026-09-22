"""Per-table masked mechanism read-out on the real-data-matched prior + the pre-registered verdict
(docs/real_prior_plan.md 10.3 / 10.4).

Protocol (fixed before any number is seen):
  tables    N_TABLES tables from RealPrior(RealConfig()) at EVAL_SEED (not a training seed, not the val
            seed), batch 1 each so every table carries its own (D, R) and generator truth `hyper`.
            Even-indexed tables FIT the linear probe, odd-indexed tables are SCORED per table.
  read-out  eval.probe_masked_all.masked_read's measurement per table: hidden target cells (policy
            any_cell AND mixed, the real input_mask), f read at those cells, non-root CONTINUOUS cells
            only (a categorised column's x is the category code while its f is the pre-binning continuous
            value: f is undefined in the code frame, so those cells are excluded from every read-out),
            at most CELLS_PER_TABLE cells per table. torch is re-seeded per table before the batch and
            before each encode, so masks and random column identities are identical for every encoder. One ridge (truth_eval.RIDGE_ALPHA) fit on the FIT
            cells, scored per SCORE table -> mse_f (mechanism), mse_x (value-retention control),
            cls_mean read-out where the encoder has CLS columns, and the head's value MSE.
  anchors   const map per policy = the FIT cells' mean f scored on the SCORE cells (what a collapsed
            encoder gets); random-init floor = N_FLOOR seeded untrained encoders under the same read.
  admission masked_all_summary's rule: pooled mse_f < min(DEGENERATE x const, floor) under BOTH
            policies. A seed is admitted only if it passes both.
  verdict   1. overall: every admitted dual seed's pooled mse_f < the admitted ds seeds' mean, under
               both policies, with >= 2 admitted dual seeds  -> latent clue.
            2. strata: 5 axes (rho, sigma_mean, cat_frac, D, R) x 5 quintiles over SCORE tables; per
               cell a paired sign test (tables where seed-mean dual mse_f < seed-mean ds mse_f), Holm
               over the 25 cells per policy; a cell counts only if significant in the same direction
               under both policies.
Run (GPU box, DEVICE=cuda; CPU is hours):
  DEVICE=cuda DS_RUNS=a,b,c DUAL_RUNS=d,e,f uv run python -m eval.probe_real_prior
  uv run python -m eval.probe_real_prior --selftest
Writes eval/results_probe_real_prior.json
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import random_init                   # noqa: E402
from eval.rows_control import _model                            # noqa: E402
from eval.truth_eval import RIDGE_ALPHA                         # noqa: E402
from prior.real import RealConfig, RealPrior                    # noqa: E402
from prior.scm import FuncFamily                                # noqa: E402
from train.data import make_batch                               # noqa: E402

OUT = ROOT / "eval/results_probe_real_prior.json"
EVAL_SEED = 40_000
N_TABLES = int(os.environ.get("N_TABLES", 2000))
CELLS_PER_TABLE = 64
POLICIES = ("any_cell", "mixed")
AXES = ("rho", "sigma_mean", "cat_frac", "D", "R")
N_FLOOR = int(os.environ.get("N_FLOOR", 30))   # C9: the paper's draw count; the 9/3-9/4 probes ran at 3
DEGENERATE = 0.98
ALPHA = 0.05
DEVICE = os.environ.get("DEVICE", "cpu")


# ---------------------------------------------------------------- measurement

@torch.no_grad()
def collect(enc, policy, n_tables=N_TABLES):
    """Per table: hidden-cell features + truth. Same tables, masks and cell subset for every encoder."""
    prior = RealPrior(RealConfig(), seed=EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    K = getattr(enc, "n_cls", 0) or 0
    recs = []
    for t in range(n_tables):
        torch.manual_seed(EVAL_SEED + t)                        # masks: identical across encoders
        bt = make_batch(prior, 1, policy, rng, device=DEVICE, return_truth=True)
        tm = bt["target_mask"]
        nonroot = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand_as(tm)
        contin = (bt["categorical"] == 0).unsqueeze(1).expand_as(tm)
        sel = (tm & nonroot & contin)[0]
        idx = sel.nonzero(as_tuple=False)
        if len(idx) == 0:
            recs.append(None)
            continue
        if len(idx) > CELLS_PER_TABLE:
            idx = idx[torch.as_tensor(rng.choice(len(idx), CELLS_PER_TABLE, replace=False), device=idx.device)]
        r, d = idx[:, 0], idx[:, 1]
        torch.manual_seed(EVAL_SEED + t)                        # column identities: identical across encoders
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        hc = h[:, :, K:] if K else h
        logits = enc.head(hc)
        rec = {"cell": hc[0, r, d].cpu().numpy(),
               "f": bt["z_f"][0, r, d].clamp(-3.05, 3.05).cpu().numpy(),
               "x": bt["z"][0, r, d].clamp(-3.05, 3.05).cpu().numpy(),
               "pred": enc.point_pred(logits)[0, r, d].cpu().numpy(),
               "hyper": bt["hyper"][0]}
        if K:
            rec["cls_mean"] = h[0, :, :K][r].mean(1).cpu().numpy()
        recs.append(rec)
    return recs


def ridge_fit(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + RIDGE_ALPHA * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_apply(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def score(recs, feat="cell"):
    """Fit on even tables, score odd tables per table. Returns (pooled, per-table dict)."""
    fit = [r for i, r in enumerate(recs) if i % 2 == 0 and r is not None and feat in r]
    Xf = np.concatenate([r[feat] for r in fit]); ff = np.concatenate([r["f"] for r in fit]); xf = np.concatenate([r["x"] for r in fit])
    wf, wx = ridge_fit(Xf, ff), ridge_fit(Xf, xf)
    f_mean = float(ff.mean())
    per, sf, sx, sc, sv, n = {}, 0.0, 0.0, 0.0, 0.0, 0
    for i, r in enumerate(recs):
        if i % 2 == 0 or r is None or feat not in r:
            continue
        ef = (ridge_apply(wf, r[feat]) - r["f"]) ** 2
        ex = (ridge_apply(wx, r[feat]) - r["x"]) ** 2
        ec = (r["f"] - f_mean) ** 2
        ev = (r["pred"] - r["x"]) ** 2
        per[i] = {"mse_f": round(float(ef.mean()), 4), "mse_x": round(float(ex.mean()), 4),
                  "const": round(float(ec.mean()), 4), "value_mse": round(float(ev.mean()), 4), "n": int(len(ef))}
        sf += ef.sum(); sx += ex.sum(); sc += ec.sum(); sv += ev.sum(); n += len(ef)
    pooled = {"mse_f": round(float(sf / n), 4), "mse_x": round(float(sx / n), 4), "const": round(float(sc / n), 4),
              "value_mse": round(float(sv / n), 4), "n_cells": int(n), "n_tables": len(per)}
    return pooled, per


# ---------------------------------------------------------------- statistics

def sign_test(neg, pos):
    """Two-sided exact binomial p for `neg` wins out of neg+pos (ties dropped)."""
    n = neg + pos
    if n == 0:
        return 1.0
    k = min(neg, pos)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals, alpha=ALPHA):
    """Holm step-down; returns the list of rejected indices."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rejected = []
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            rejected.append(i)
        else:
            break
    return rejected


def quintiles(values):
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, values, side="right"), edges


def strata_test(per_ds, per_dual, hyper):
    """per_*: {table -> seed-mean mse_f}. Paired sign test per (axis, quintile) cell, Holm over 25/policy."""
    tables = sorted(set(per_ds) & set(per_dual))
    gaps = np.array([per_dual[t] - per_ds[t] for t in tables])
    out = {"n_tables": len(tables), "overall": {"dual_wins": int((gaps < 0).sum()), "ds_wins": int((gaps > 0).sum()),
                                                 "p": sign_test(int((gaps < 0).sum()), int((gaps > 0).sum())),
                                                 "median_gap": round(float(np.median(gaps)), 4)}, "cells": []}
    for ax in AXES:
        v = np.array([hyper[t][ax] for t in tables], float)
        q, edges = quintiles(v)
        for i in range(5):
            m = q == i
            if not m.any():
                continue
            g = gaps[m]
            neg, pos = int((g < 0).sum()), int((g > 0).sum())
            out["cells"].append({"axis": ax, "q": i, "lo": float(v[m].min()), "hi": float(v[m].max()), "n": int(m.sum()),
                                 "dual_wins": neg, "ds_wins": pos, "median_gap": round(float(np.median(g)), 4),
                                 "p": sign_test(neg, pos)})
    rej = holm([c["p"] for c in out["cells"]])
    for j, c in enumerate(out["cells"]):
        c["holm_significant"] = j in rej
        c["direction"] = "dual" if c["median_gap"] < 0 else "ds"
    return out


# ---------------------------------------------------------------- main

def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


CACHE = ROOT / "eval/cache_probe_real"


def scored(name, enc, policy):
    """(pooled, per, hyper_by_table, has_cls) for one encoder+policy, cached as JSON under CACHE (keyed by name,
    policy, N_TABLES, seed) so the 2,000 forwards are done once per box."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}_{policy}_n{N_TABLES}_s{EVAL_SEED}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["pooled"], {int(k): v for k, v in d["per"].items()}, {int(k): v for k, v in d["hyper"].items()}, d.get("cls")
    recs = collect(enc, policy)
    pooled, per = score(recs)
    cls = score(recs, "cls_mean")[0] if any(r is not None and "cls_mean" in r for r in recs) else None
    hyp = {t: recs[t]["hyper"] for t in per}
    f.write_text(json.dumps({"pooled": pooled, "per": per, "hyper": hyp, "cls": cls}, default=_np))
    return pooled, per, hyp, cls


def admitted_rule(pooled, const, floor):
    cut = {p: round(min(DEGENERATE * const[p], floor[p]), 4) for p in POLICIES}
    ok = {p: pooled[p]["mse_f"] < cut[p] for p in POLICIES}
    return cut, ok, all(ok.values())


def main():
    ds_runs = [r for r in os.environ.get("DS_RUNS", "").split(",") if r]
    dual_runs = [r for r in os.environ.get("DUAL_RUNS", "").split(",") if r]
    assert ds_runs and dual_runs, "set DS_RUNS and DUAL_RUNS (comma-separated run names, seed-aligned)"
    out = {"protocol": {"eval_seed": EVAL_SEED, "n_tables": N_TABLES, "cells_per_table": CELLS_PER_TABLE,
                        "fit_tables": "even index", "score_tables": "odd index", "policies": list(POLICIES),
                        "cells": "hidden, non-root, continuous (categorical columns excluded: f undefined in the code frame)",
                        "seeding": "torch re-seeded per table before the batch and before encode",
                        "ridge_alpha": RIDGE_ALPHA, "admission": f"pooled mse_f < min({DEGENERATE} x const, floor) under both policies",
                        "n_floor": N_FLOOR, "axes": list(AXES), "holm_alpha": ALPHA, "device": DEVICE},
           "runs": {}, "floor": {}, "const": {}}
    per_table = {}                                                  # run -> policy -> {table -> mse_f}
    hyper = {}
    # floors + const
    fl = {p: [] for p in POLICIES}
    for d in range(N_FLOOR):
        enc = random_init(ds_runs[0], 10_000 + d).to(DEVICE)
        for p in POLICIES:
            pooled, per, _, _ = scored(f"floor{d}", enc, p)
            fl[p].append(pooled["mse_f"])
            out["const"][p] = pooled["const"]
        print(f"  floor draw {d}: " + " ".join(f"{p}={fl[p][-1]:.4f}" for p in POLICIES), flush=True)
    out["floor"] = {p: {"mean": round(float(np.mean(fl[p])), 4), "draws": fl[p]} for p in POLICIES}
    floor_mean = {p: out["floor"][p]["mean"] for p in POLICIES}
    # arms
    for arm, runs in (("ds", ds_runs), ("dual", dual_runs)):
        for run in runs:
            enc, cfg, _ = _model(run)
            enc = enc.to(DEVICE)
            rec = {"arm": arm, "seed": cfg["seed"], "lr": cfg["lr"]}
            per_table[run] = {}
            for p in POLICIES:
                pooled, per, hyp, cls = scored(run, enc, p)
                rec[p] = {"cell": pooled}
                if cls is not None:
                    rec[p]["cls_mean"] = cls
                per_table[run][p] = {t: v["mse_f"] for t, v in per.items()}
                hyper.update(hyp)                                          # union: the policies hide different cells
            rec["cutoff"], rec["admitted_by_policy"], rec["admitted"] = admitted_rule(
                {p: rec[p]["cell"] for p in POLICIES}, out["const"], floor_mean)
            out["runs"][run] = rec
            print(f"  {run:24} " + "  ".join(f"{p}: f={rec[p]['cell']['mse_f']:.4f} x={rec[p]['cell']['mse_x']:.4f} v={rec[p]['cell']['value_mse']:.4f}" for p in POLICIES)
                  + f"  admitted={rec['admitted']}", flush=True)
    # verdict 1: overall
    adm = {arm: [r for r in runs if out["runs"][r]["admitted"]] for arm, runs in (("ds", ds_runs), ("dual", dual_runs))}
    v1 = {"admitted": adm, "by_policy": {}}
    for p in POLICIES:
        ds_mean = float(np.mean([out["runs"][r][p]["cell"]["mse_f"] for r in adm["ds"]])) if adm["ds"] else None
        wins = [out["runs"][r][p]["cell"]["mse_f"] < ds_mean for r in adm["dual"]] if ds_mean is not None else []
        v1["by_policy"][p] = {"ds_admitted_mean": ds_mean, "dual_wins": wins}
    v1["clue"] = bool(len(adm["dual"]) >= 2 and adm["ds"] and all(all(v1["by_policy"][p]["dual_wins"]) for p in POLICIES))
    # verdict 2: strata, on admitted seeds (seed-mean per table), both policies
    v2 = {}
    if adm["ds"] and adm["dual"]:
        for p in POLICIES:
            mean_of = lambda runs: {t: float(np.mean([per_table[r][p][t] for r in runs])) for t in per_table[runs[0]][p]}
            v2[p] = strata_test(mean_of(adm["ds"]), mean_of(adm["dual"]), hyper)
        both = {(c["axis"], c["q"]) for c in v2[POLICIES[0]]["cells"] if c["holm_significant"]} & \
               {(c["axis"], c["q"]) for c in v2[POLICIES[1]]["cells"] if c["holm_significant"]}
        agree = []
        for ax, q in sorted(both):
            dirs = {p: next(c["direction"] for c in v2[p]["cells"] if (c["axis"], c["q"]) == (ax, q)) for p in POLICIES}
            if len(set(dirs.values())) == 1:
                agree.append({"axis": ax, "q": q, "direction": dirs[POLICIES[0]]})
        v2["significant_both_policies"] = agree
        v2["latent_cells"] = [c for c in agree if c["direction"] == "dual"]
    out["verdict"] = {"overall": v1, "strata": v2,
                      "rule": "clue = every admitted dual seed (>=2) beats the admitted ds mean under both policies; "
                              "strata cell = Holm-significant paired sign test in the same direction under both policies"}
    out["tables"] = {str(t): {**hyper[t], "mse_f": {r: {p: per_table[r][p].get(t) for p in POLICIES} for r in per_table}} for t in sorted(hyper)}
    OUT.write_text(json.dumps(out, indent=1, default=_np) + "\n")   # numpy scalars from hyper / sums
    print("verdict overall:", json.dumps(v1, default=_np))
    print("verdict strata latent cells:", v2.get("latent_cells"))
    print("wrote", OUT.relative_to(ROOT))


# ---------------------------------------------------------------- selftest

def selftest():
    """The guards must fire: an oracle feature is admitted, a constant feature is not, the sign test
    and Holm reject exactly when they should."""
    rng = np.random.default_rng(0)
    recs = []
    for t in range(200):
        n = 40
        f = rng.normal(size=n); x = f + rng.normal(size=n) * 0.5
        recs.append({"cell": np.stack([f, rng.normal(size=n)], 1), "f": f, "x": x, "pred": x * 0 + 0.0,
                     "hyper": {"rho": rng.uniform(), "sigma_mean": rng.uniform(), "cat_frac": rng.uniform(), "D": rng.integers(4, 65), "R": rng.integers(16, 1024)}})
    pooled, per = score(recs)
    assert pooled["mse_f"] < 0.05 < pooled["const"], pooled                       # oracle read-out learns f
    const_recs = [{**r, "cell": np.ones((len(r["f"]), 2))} for r in recs]
    pc, _ = score(const_recs)
    assert abs(pc["mse_f"] - pc["const"]) < 0.02, pc                              # constant features = const map
    floor = {p: pc["const"] * 1.05 for p in POLICIES}
    cut, ok, adm = admitted_rule({p: pooled for p in POLICIES}, {p: pooled["const"] for p in POLICIES}, floor)
    assert adm, (cut, ok)
    cut, ok, adm = admitted_rule({p: pc for p in POLICIES}, {p: pc["const"] for p in POLICIES}, floor)
    assert not adm, (cut, ok)                                                      # constant map is NOT admitted
    assert sign_test(0, 20) < 1e-4 and sign_test(10, 10) == 1.0
    assert holm([0.001, 0.5, 0.002] + [0.9] * 22) == [0, 2]
    assert holm([0.01] * 25) == []                                                 # 0.01 > 0.05/25
    hyper = {i: r["hyper"] for i, r in enumerate(recs)}
    ds = {t: 1.0 for t in per}; dual = {t: 0.9 for t in per}                       # dual wins every table
    st = strata_test(ds, dual, hyper)
    assert st["overall"]["p"] < 1e-6 and all(c["holm_significant"] and c["direction"] == "dual" for c in st["cells"]), st["overall"]
    null = {t: 1.0 + (0.1 if t % 4 < 2 else -0.1) for t in per}                   # balanced gaps -> nothing significant
    st = strata_test(ds, null, hyper)
    assert not any(c["holm_significant"] for c in st["cells"]), [c for c in st["cells"] if c["holm_significant"]]
    print("selftest OK: oracle admitted, constant rejected, sign/Holm fire and stay silent as they should")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
