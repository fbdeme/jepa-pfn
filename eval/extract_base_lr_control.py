"""Training-log endpoints for the headline-scale learning-rate controls (with the seed-0 baselines
at the shared lr), so the appendix can state what each control run did during training: the value
arm's held-out value MSE, the latent arm's target-representation health (per-dimension std, erank).
The verdict metric itself (masked-probe f-MSE) is in eval/results_lr_control_probe.json.

Run: uv run python -m eval.extract_base_lr_control
Writes eval/results_base_lr_control_curves.json
"""
import json
from pathlib import Path

import yaml

from eval.extract_ds_training_curve import DIVERGED_TOL, LEARNED_TOL

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_base_lr_control_curves.json"
RUNS = ["base_ds_s0", "base_ds_lr1e-4_s0", "base_ds_lr4e-3_s0",
        "base_dual_s0", "base_dual_lr1e-4_s0", "base_dual_lr4e-3_s0"]


def _row(name):
    d = ROOT / "runs" / name
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    row = {"run": name, "arm": "ds" if "_ds" in name else "dual", "lr": float(cfg["lr"]),
           "control": "_lr" in name, "last_step": recs[-1]["step"], "steps_configured": cfg["steps"]}
    if row["arm"] == "ds":
        vals = [(r["step"], r["val_mse"]) for r in recs if "val_mse" in r]
        best_step, best = min(vals, key=lambda t: t[1])
        row.update({"val_mse_first": round(vals[0][1], 4), "val_mse_last": round(vals[-1][1], 4),
                    "val_mse_best": round(best, 4), "best_step": best_step,
                    "learned": bool(vals[0][1] - best > LEARNED_TOL),
                    "diverged": bool(vals[-1][1] - best > DIVERGED_TOL)})
    else:
        last = [r for r in recs if "val_tgt" in r][-1]
        row.update({"val_jepa_last": last["val_jepa"],
                    "tgt_dim_std_last": last["val_tgt"]["dim_std"], "tgt_erank_last": last["val_tgt"]["erank"],
                    "tgt_cos_last": last["val_tgt"]["cos"],
                    "collapsed": last["val_tgt"]["dim_std"] < 1e-3})
    return row


def main():
    rows = [_row(n) for n in RUNS if (ROOT / "runs" / n / "metrics.jsonl").exists()]
    assert len(rows) == len(RUNS), f"missing runs: {set(RUNS) - {r['run'] for r in rows}}"
    assert all(r["last_step"] == r["steps_configured"] for r in rows), "a run stopped short"
    out = {"note": "seed-0 headline arms at the shared lr (1e-3) beside one control run per alternative lr; "
                   "val_mse is train.validate's deterministic holdout (unit-variance targets, 1.0 = predict the mean); "
                   "dual rows report the latent target's health at the last validation",
           "runs": rows,
           "summary": {"ds": {str(r["lr"]): r["val_mse_last"] for r in rows if r["arm"] == "ds"},
                       "ds_learned": {str(r["lr"]): r["learned"] for r in rows if r["arm"] == "ds"},
                       "dual_collapsed": {str(r["lr"]): r["collapsed"] for r in rows if r["arm"] == "dual"},
                       "dual_tgt_erank": {str(r["lr"]): r["tgt_erank_last"] for r in rows if r["arm"] == "dual"}}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT)); print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
