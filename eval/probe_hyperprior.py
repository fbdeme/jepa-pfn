"""Table-level hyperprior read-out (docs/todo.md section 30, step 1; pre-registered 2026-09-03).

Question: does a latent objective leave the table's GENERATOR-LEVEL variables more readable
than the value objective does? The cell-level read-outs in this repo all target f, a
conditional of observed cells, which the value objective is by definition trained to
represent. The targets here are properties of the sampled SCM that no single cell's
conditional determines: the table's noise level, its graph density, the number of hidden
confounders, and the mechanism-family mix.

Protocol: the recipe family's prior (eval/probe_masked_all.scm_prior: sigma 1.0, ctx 128,
nq 50, 4-32 cols, EVAL_SEED), ONE table per batch so every table has its own (D, H, p_edge)
draw, N_TABLES tables. Features per table: the mean of the CONTEXT-row cell embeddings and
the mean of the context-row CLS columns. Context rows attend only to context rows
(model/pfn.py:109-120), so these features do not depend on the masking policy; the two
policies of the cell-level protocol collapse to one here, and --selftest proves it.

Read-out: ridge (eval/truth_eval.RIDGE_ALPHA) with K_FOLD-fold cross-validation over tables,
scored as R^2 on the held-out folds. The constant map scores R^2 = 0 by construction; the
random-init floor is measured. Verdict rule (pre-registered in todo section 30, coded in
`verdict()` below BEFORE any number was read): a latent arm group wins a target when EVERY
admitted seed of that group beats the mean of the admitted data-space seeds on the best of
its read-outs AND that read-out beats the constant map; a "clue" is >= 2 wins among
(sigma_log_mean, edge_density, n_hidden). Family fractions are reported, not judged.

Anchor: `anchor_d_only` = ridge on (D, D^2, 1/D) alone; a target it already reads is a proxy of
the visible column count, not a generator variable.

Run: uv run python -m eval.probe_hyperprior              (CPU, frozen ckpts, ~30 min)
     HP_TABLES=16 uv run python -m eval.probe_hyperprior --selftest
     uv run python -m eval.probe_hyperprior --anchor-only   (merge the anchor into an existing artifact)
Writes eval/results_hyperprior_probe.json
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import (EVAL_SEED, N_FLOOR, RECIPE_ARMS, load_auto,   # noqa: E402
                                   random_init, scm_prior)
from eval.probe_lr_control import RUNS as LR_RUNS                                # noqa: E402
from eval.truth_eval import RIDGE_ALPHA                                          # noqa: E402
from prior.scm import FuncFamily                                                 # noqa: E402
from train.data import make_batch                                                # noqa: E402

OUT = ROOT / "eval/results_hyperprior_probe.json"
CTX, NQ, NCOLS = 128, 50, (4, 32)
N_TABLES = int(os.environ.get("HP_TABLES", "512"))
K_FOLD = 8
POLICY = "any_cell"      # irrelevant for context-row features; kept explicit
TARGETS = ("sigma_log_mean", "edge_density", "n_hidden", "fam_mlp", "fam_tree", "fam_disc")
VERDICT_TARGETS = TARGETS[:3]
FLOOR_REF = "base_ds_s0"


# ---------------------------------------------------------------- truth per table

def truth_rows(bt):
    fam, sig, adj = bt["cell_family"], bt["cell_sigma"], bt["adjacency"]
    B, _, D = bt["z"].shape
    N = adj.shape[1]
    rows = []
    for b in range(B):
        nr = fam[b] != int(FuncFamily.ROOT)
        n_nr = int(nr.sum())
        frac = lambda fm: float((fam[b] == int(fm)).sum()) / n_nr if n_nr else float("nan")
        rows.append({
            "sigma_log_mean": float(torch.log(sig[b][nr]).mean()) if n_nr else float("nan"),
            "edge_density": float(adj[b].sum()) / (N * (N - 1) / 2),
            "n_hidden": float(N - D),
            "fam_mlp": frac(FuncFamily.MLP), "fam_tree": frac(FuncFamily.TREE),
            "fam_disc": frac(FuncFamily.DISC)})
    return rows


# ---------------------------------------------------------------- features per table

@torch.no_grad()
def collect(enc, n_tables, policy=POLICY):
    torch.manual_seed(EVAL_SEED)          # same tables and column codes for every arm
    prior = scm_prior(1.0, CTX, NQ, NCOLS)
    rng = np.random.default_rng(EVAL_SEED)
    K = getattr(enc, "n_cls", 0) or 0
    feats = {"cell_mean": []}
    if K:
        feats["cls_mean"] = []
    ys = []
    for _ in range(n_tables):
        bt = make_batch(prior, 1, policy, rng, split=CTX, return_truth=True)
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)[0, :CTX]
        feats["cell_mean"].append(h[:, K:].mean((0, 1)))
        if K:
            feats["cls_mean"].append(h[:, :K].mean((0, 1)))
        ys += truth_rows(bt)
    X = {k: torch.stack(v).numpy() for k, v in feats.items()}
    Y = {t: np.array([y[t] for y in ys]) for t in TARGETS}
    return X, Y


def ridge_r2_cv(X, y, k=K_FOLD):
    """Same solve as eval/truth_eval.ridge_mse, k interleaved folds over tables."""
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    n = len(y)
    folds = np.arange(n) % k
    Xb = np.hstack([X, np.ones((n, 1))])
    sse = sst = 0.0
    for f in range(k):
        tr, te = folds != f, folds == f
        A = Xb[tr].T @ Xb[tr] + RIDGE_ALPHA * np.eye(Xb.shape[1])
        w = np.linalg.solve(A, Xb[tr].T @ y[tr])
        sse += float(((Xb[te] @ w - y[te]) ** 2).sum())
        sst += float(((y[te] - y[tr].mean()) ** 2).sum())
    return {"r2": round(1 - sse / sst, 4), "mse": round(sse / n, 4), "n": int(n)}


def d_only_anchor(n_tables):
    """How much of each target the visible column count D alone explains (features D, D^2, 1/D).
    A target this anchor already reads is a D-proxy, not a generator variable."""
    torch.manual_seed(EVAL_SEED)
    prior = scm_prior(1.0, CTX, NQ, NCOLS)
    rng = np.random.default_rng(EVAL_SEED)
    ys, ds = [], []
    for _ in range(n_tables):
        bt = make_batch(prior, 1, POLICY, rng, split=CTX, return_truth=True)
        ys += truth_rows(bt)
        ds.append(bt["z"].shape[2])
    D = np.array(ds, float)
    X = np.stack([D, D ** 2, 1 / D], 1)
    return {t: ridge_r2_cv(X, np.array([y[t] for y in ys])) for t in TARGETS}


def score(enc, n_tables):
    X, Y = collect(enc, n_tables)
    return {rd: {t: ridge_r2_cv(X[rd], Y[t]) for t in TARGETS} for rd in X}


# ---------------------------------------------------------------- verdict (pre-registered)

def best_r2(arm, t):
    return max(arm[rd][t]["r2"] for rd in arm if rd in ("cell_mean", "cls_mean"))


def verdict(arms, ds_runs, latent_groups):
    """arms: run -> {readout -> {target -> {r2}}}; latent_groups: group -> admitted runs."""
    ds_mean = {t: round(st.mean(best_r2(arms[r], t) for r in ds_runs), 4) for t in VERDICT_TARGETS}
    groups = {}
    for g, runs in latent_groups.items():
        per_t = {}
        for t in VERDICT_TARGETS:
            r2 = {r: best_r2(arms[r], t) for r in runs}
            per_t[t] = {"latent_r2": r2, "ds_mean_r2": ds_mean[t],
                        "wins": bool(runs) and all(v > ds_mean[t] and v > 0 for v in r2.values())}
        n_wins = sum(per_t[t]["wins"] for t in VERDICT_TARGETS)
        groups[g] = {"admitted": runs, "per_target": per_t, "n_wins": n_wins, "clue": n_wins >= 2}
    return {"rule": ("a latent group wins a target when every admitted seed's best read-out R^2 "
                     "exceeds the admitted data-space seeds' mean AND 0 (constant map); "
                     "clue = >= 2 wins among " + ", ".join(VERDICT_TARGETS)),
            "ds_runs": ds_runs, "ds_mean_r2": ds_mean, "groups": groups,
            "clue": any(v["clue"] for v in groups.values())}


def admitted():
    """Admitted seeds come from the cell-level verdict artifacts, not from this file."""
    summ = json.loads((ROOT / "eval/results_probe_masked_all_summary.json").read_text())["recipe"]["per_group"]
    both = lambda g: sorted(set(summ[g]["any_cell"]["alive_runs"]) & set(summ[g]["mixed"]["alive_runs"]))
    lr = json.loads((ROOT / "eval/results_lr_control_probe.json").read_text())["summary"]
    lr_alive = [a for a in LR_RUNS if a in lr and all(lr[a][p]["alive"] for p in ("any_cell", "mixed"))]
    groups = {g: both(g) for g in ("dual", "ema_nohead", "lat", "dual_lewm")}
    groups["dual_lr1e-4"] = [a for a in lr_alive if a.startswith("base_dual")]
    return both("ds"), groups, [a for a in lr_alive if a.startswith("base_ds")]


# ---------------------------------------------------------------- selftest

def selftest():
    enc = random_init(FLOOR_REF, 1)
    Xa, Ya = collect(enc, 4, "any_cell")
    Xm, Ym = collect(enc, 4, "mixed")
    for rd in Xa:
        assert np.allclose(Xa[rd], Xm[rd], atol=1e-6), f"{rd}: context features depend on the policy"
    assert all(np.array_equal(Ya[t], Ym[t], equal_nan=True) for t in TARGETS)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    lin = ridge_r2_cv(X, X @ rng.normal(size=16))["r2"]
    noise = ridge_r2_cv(X, rng.normal(size=400))["r2"]
    assert lin > 0.95 and noise < 0.1, (lin, noise)
    mk = lambda a, b, c: {"cell_mean": {t: {"r2": v} for t, v in zip(VERDICT_TARGETS, (a, b, c))}}
    arms = {"ds0": mk(.2, .1, .0), "ds1": mk(.3, .1, .0),
            "win0": mk(.4, .2, .0), "win1": mk(.35, .15, .0),
            "mixed": mk(.4, .2, -.1), "low": mk(.1, .2, .0), "neg": mk(.4, .2, .0)}
    v = verdict(arms, ["ds0", "ds1"], {"A": ["win0", "win1"], "B": ["win0", "low"],
                                       "C": ["mixed"], "D": ["neg"], "E": []})
    assert v["groups"]["A"]["clue"] and v["groups"]["A"]["n_wins"] == 2
    assert not v["groups"]["B"]["clue"], "one seed below ds mean must not win"
    assert v["groups"]["C"]["n_wins"] == 2 and v["groups"]["C"]["clue"]
    assert not v["groups"]["E"]["clue"] and v["clue"]
    arms["neg"] = mk(.4, .0, .0)      # a win at R^2 <= 0 does not count
    assert verdict(arms, ["ds0", "ds1"], {"D": ["neg"]})["groups"]["D"]["n_wins"] == 1
    print("selftest ok: policy-invariant context features (4 tables), ridge R2 "
          f"lin={lin:.3f} noise={noise:.3f}, verdict rule 6/6")


# ---------------------------------------------------------------- main

def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--anchor-only" in sys.argv:       # merge the D-only anchor into an existing artifact
        res = json.loads(OUT.read_text())
        res["anchor_d_only"] = d_only_anchor(res["protocol"]["n_tables"])
        OUT.write_text(json.dumps(res, indent=1) + "\n")
        print("anchor_d_only", {t: v["r2"] for t, v in res["anchor_d_only"].items()})
        return
    ds_runs, latent_groups, ds_lr = admitted()
    arms_meta = {a: {"group": g, "seed": s} for a, g, s in RECIPE_ARMS}
    arms_meta.update({a: {"group": a.split("_")[1] + "_lr" + a.split("_lr")[1].split("_")[0], "seed": 0}
                      for a in LR_RUNS})
    res = {"protocol": {"prior": "scm_prior(1.0, 128, 50, (4, 32))", "ctx": CTX, "n_tables": N_TABLES,
                        "batch": 1, "eval_seed": EVAL_SEED, "policy": POLICY, "k_fold": K_FOLD,
                        "ridge_alpha": RIDGE_ALPHA, "features": "context-row means (cell, cls)",
                        "targets": list(TARGETS), "const_map_r2": 0.0, "n_floor": N_FLOOR,
                        "floor_ref_arch": FLOOR_REF},
           "arms": {}}
    t0 = time.time()
    for a, meta in arms_meta.items():
        if not (ROOT / "runs" / a / "ckpt.pt").exists():
            print(f"  skip {a}: no ckpt", flush=True)
            continue
        t = time.time()
        r = score(load_auto(a), N_TABLES)
        r["meta"] = meta
        res["arms"][a] = r
        print(f"  {a:24} " + "  ".join(f"{t_[:5]}={best_r2(r, t_):+.3f}" for t_ in TARGETS)
              + f"   [{time.time()-t:.0f}s]", flush=True)
    draws = []
    for d in range(N_FLOOR):
        draws.append(score(random_init(FLOOR_REF, 10_000 + d), N_TABLES))
    res["floor"] = {rd: {t: {"mean": round(st.mean(x[rd][t]["r2"] for x in draws), 4),
                             "draws": [x[rd][t]["r2"] for x in draws]} for t in TARGETS}
                    for rd in draws[0]}
    print("  floor " + "  ".join(f"{t[:5]}={res['floor']['cell_mean'][t]['mean']:+.3f}" for t in TARGETS), flush=True)
    res["anchor_d_only"] = d_only_anchor(N_TABLES)
    res["verdict"] = verdict(res["arms"], ds_runs, latent_groups)
    res["verdict"]["ds_lr_controls"] = {a: {t: best_r2(res["arms"][a], t) for t in VERDICT_TARGETS}
                                        for a in ds_lr if a in res["arms"]}
    res["_note"] = __doc__.split("\n\n")[1]
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  clue={res['verdict']['clue']}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
