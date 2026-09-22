"""Per-term convergence read-out for the I-JEPA-split runs, with E7's pre-registered rule (eval.v4_readout.convergence_quarters:
last quarter vs previous quarter, converged iff drop <= max(1% of Q3 mean, 2 sd of Q4)). Train series are smoothed over
val_every-sized windows (the raw per-25-step loss is noisy); val series are used as logged. Writes results_convergence.json."""
import json, sys
from pathlib import Path
from statistics import mean
D = Path(__file__).parent; ROOT = D.parents[1]; sys.path.insert(0, str(ROOT))
from eval.v4_readout import convergence_quarters
RUNS = {"dual_split": D / "metrics_dual_split_s0", "cell_ctx_diff": D / "metrics_cell_ctx_diff_s0",
        "ds_40k (ref)": ROOT / "runs/v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0/metrics.jsonl"}
CHECK = (1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)

def load(p):
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    tr = [r for r in rows if "loss" in r]; va = [r for r in rows if any(k.startswith("val_") for k in r)]
    return tr, va

def smooth(tr, key, window=250):
    """mean of a train key over [step-window, step] at every multiple of window."""
    out, T = [], tr[-1]["step"]
    for s in range(window, T + 1, window):
        xs = [r[key] for r in tr if s - window < r["step"] <= s and key in r]
        if xs: out.append({"step": s, key: mean(xs)})
    return out

def series(tr, va):
    S = {}
    if tr and "pred_loss" in tr[0]:
        for r in tr: r["ce_loss"] = r["loss"] - r["pred_loss"]
        S["train jepa (pred_loss)"] = smooth(tr, "pred_loss"); S["train bar CE (loss - pred_loss)"] = smooth(tr, "ce_loss"); S["train total"] = smooth(tr, "loss")
    else:
        S["train CE"] = smooth(tr, "loss")
    for k in ("val_jepa", "val_ppd", "val_nll", "val_mse"):
        if any(k in r for r in va): S[k] = [{"step": r["step"], k: r[k]} for r in va if k in r]
    for k, sub in (("val_tgt", ("erank", "dim_std", "dim_std_min", "cos", "within_table_var_frac")), ("val_pred", ("erank", "cos"))):
        if any(k in r for r in va):
            for s in sub:
                name = f"{k}.{s}"
                S[name] = [{"step": r["step"], name: r[k][s]} for r in va if k in r and r[k].get(s) is not None]
    return S

out = {}
for name, p in RUNS.items():
    tr, va = load(p); S = series(tr, va); rec = {}
    print(f"\n=== {name}  (train records {len(tr)}, val records {len(va)}, last step {va[-1]['step']})")
    print(f"{'series':34} " + " ".join(f"{c//1000:>5}k" for c in CHECK) + "   Q3->Q4 drop  tol   converged")
    for sname, rows in S.items():
        key = [k for k in rows[0] if k != "step"][0]
        if all(abs(r[key]) < 1e-12 for r in rows):      # a term that is switched off (lambda 0) is not a series
            continue
        by = {r["step"]: r[key] for r in rows}
        traj = [by.get(c) for c in CHECK]
        cv = convergence_quarters(rows, key=key)
        rec[sname] = {"trajectory": {str(c): (None if v is None else round(v, 4)) for c, v in zip(CHECK, traj)}, "convergence": cv}
        cells = " ".join("     -" if v is None else f"{v:6.3f}" for v in traj)
        tail = "   n/a" if cv is None else f"   {cv['drop']:+.4f} ({(cv['rel_drop'] or 0):+.1%}) {cv['tolerance']:.4f}  {'YES' if cv['converged'] else 'no'}"
        print(f"{sname:34} {cells}{tail}")
    out[name] = rec
(D / "results_convergence.json").write_text(json.dumps(out, indent=1))
print("\nwrote", D / "results_convergence.json")
