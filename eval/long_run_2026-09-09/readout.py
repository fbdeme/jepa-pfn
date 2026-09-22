"""Deterministic read-out of the single-seed 200k long run (dual_split, schedule-free constant lr, 2026-09-08/09).

Inputs: metrics_200k_s0.jsonl (copied from runs/.../metrics.jsonl; the same file is on HF fbdeme/jepa-pfn-ckpt).
Writes results_long_run.json: per-20k val trajectory, best val_mse step, the project collapse trajectory
(eval.collapse), and E7's quarter rule (eval.v4_readout.convergence_quarters) on the pre-collapse window
[0, last snapshot before onset] and on the full run. Run: uv run python eval/long_run_2026-09-09/readout.py
"""
import json
from pathlib import Path
import sys

D = Path(__file__).parent; ROOT = D.parents[1]; sys.path.insert(0, str(ROOT))
from eval.collapse import trajectory                      # noqa: E402
from eval.v4_readout import convergence_quarters          # noqa: E402

rows = [json.loads(l) for l in (D / "metrics_200k_s0.jsonl").read_text().splitlines() if l.strip()]
va = [r for r in rows if "val_mse" in r]
T = va[-1]["step"]
SERIES = {"val_mse": lambda r: r["val_mse"], "val_ppd": lambda r: r["val_ppd"], "val_jepa": lambda r: r["val_jepa"],
          "val_tgt.erank": lambda r: r["val_tgt"]["erank"], "val_tgt.dim_std": lambda r: r["val_tgt"]["dim_std"],
          "val_tgt.dim_std_min": lambda r: r["val_tgt"]["dim_std_min"], "val_pred.erank": lambda r: r["val_pred"]["erank"]}
snaps = list(range(20000, T + 1, 20000))
by_step = {r["step"]: r for r in va}
traj = {name: {str(s): round(get(by_step[s]), 4) for s in snaps} for name, get in SERIES.items()}
col = trajectory(va)
best = min(va, key=lambda r: r["val_mse"])
pre_end = max(s for s in snaps if col["onset"] is None or s < col["onset"])


def quarters(upto):
    sub = [r for r in va if r["step"] <= upto]
    return {name: convergence_quarters([{"step": r["step"], name: get(r)} for r in sub], key=name)
            for name, get in SERIES.items() if name in ("val_mse", "val_ppd", "val_jepa")}


out = {"run": "v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_200k_s0", "steps": T, "val_records": len(va),
       "snapshots": snaps, "trajectory": traj,
       "best_val_mse": {"step": best["step"], "val_mse": best["val_mse"], "val_ppd": best["val_ppd"]},
       "collapse": {**col, "criterion": "eval.collapse.is_collapsed (dim_std < 1e-3 or within_table_var_frac < 0.1)",
                    "last_pre_collapse_snapshot": pre_end},
       "e7_quarters_pre_collapse": {"window": [0, pre_end], **quarters(pre_end)},
       "e7_quarters_full": {"window": [0, T], **quarters(T)}}
(D / "results_long_run.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"{'series':22} " + " ".join(f"{s//1000:>5}k" for s in snaps))
for name in SERIES:
    print(f"{name:22} " + " ".join(f"{traj[name][str(s)]:6.3f}" for s in snaps))
print("best val_mse", out["best_val_mse"]); print("collapse", out["collapse"])
for w in ("e7_quarters_pre_collapse", "e7_quarters_full"):
    print(w, {k: (v["rel_drop"], v["converged"]) for k, v in out[w].items() if isinstance(v, dict)})
print("wrote", (D / "results_long_run.json").relative_to(ROOT))
