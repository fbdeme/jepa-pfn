"""C1 (round-13 R5 ISSUE-06): does the value arm's stall on the shared-factor substrate survive a
learning-rate change? The K sweep ran every ds arm at one unswept lr (1e-3). This extracts the control
runs at other learning rates next to the baseline seeds of the same rank, with the same learned/diverged
rule as eval/extract_ds_training_curve.py, into a committed artifact. Rank K=4 is the headline substrate
and its baseline runs carry no K in the name.

Run: uv run python -m eval.extract_factor_lr_control
Writes eval/results_factor_lr_control.json
"""
import json
import re
from pathlib import Path

import yaml

from eval.extract_ds_training_curve import DIVERGED_TOL, LEARNED_TOL

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_factor_lr_control.json"


def _row(d):
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    vals = [(r["step"], r["val_mse"]) for r in
            (json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip())
            if "val_mse" in r]
    if len(vals) < 2:
        return None          # a run that died before its first validation (e.g. an OOM'd launch)
    best_step, best_val = min(vals, key=lambda t: t[1])
    return {"run": d.name, "k": int(cfg["prior"]["n_factors"]), "lr": float(cfg["lr"]),
            "control": "_lr" in d.name,
            "n_val": len(vals), "last_step": vals[-1][0], "steps_configured": cfg["steps"],
            "val_mse_first": round(vals[0][1], 4), "val_mse_last": round(vals[-1][1], 4),
            "val_mse_best": round(best_val, 4), "best_step": best_step,
            "learned": bool(vals[0][1] - best_val > LEARNED_TOL),
            "diverged": bool(vals[-1][1] - best_val > DIVERGED_TOL)}


def main():
    runs = [r for r in (_row(d) for d in sorted((ROOT / "runs").glob("c4_factor_ds*"))
                        if (d / "metrics.jsonl").exists()) if r]
    ctrl = [r for r in runs if r["control"]]
    assert ctrl, "no lr-control runs in runs/"
    short = [r["run"] for r in ctrl if r["last_step"] != r["steps_configured"]]
    assert not short, f"control runs stopped short of their configured steps: {short}"
    sweep = json.loads((ROOT / "eval/results_ds_training_curve.json").read_text())["by_k"]
    by_k = {}
    for k in sorted({r["k"] for r in ctrl}):
        base = [r for r in runs if r["k"] == k and not r["control"]]
        cs = [r for r in ctrl if r["k"] == k]
        # self-check: the baseline seeds must reproduce the committed K sweep artifact exactly
        assert {r["run"]: r["val_mse_last"] for r in base} == \
               {r["run"]: r["val_mse_last"] for r in sweep[str(k)]["runs"]}, f"K={k} baseline drifted"
        by_k[str(k)] = {"baseline_lr": sorted({r["lr"] for r in base}),
                        "baseline_val_mse_last_mean": round(sum(r["val_mse_last"] for r in base) / len(base), 4),
                        "baseline_n_learned": sum(r["learned"] for r in base), "baseline_n": len(base),
                        "controls": {str(r["lr"]): {"val_mse_last": r["val_mse_last"], "val_mse_best": r["val_mse_best"],
                                                     "best_step": r["best_step"], "learned": r["learned"],
                                                     "diverged": r["diverged"]} for r in cs},
                        "runs": base + cs}
    lo = {k: v["controls"].get("0.0001") for k, v in by_k.items()}
    out = {"metric": "val_mse -- held-out value MSE; targets are unit variance, so 1.0 is the "
                     "score of predicting the mean",
           "arm": "c4_factor_ds (data-space) on the shared-factor substrate at sigma=1.0; baseline seeds "
                  "at the sweep's single lr plus one control run per alternative lr and rank",
           "learned_rule": f"val_mse_best < val_mse_first - {LEARNED_TOL}",
           "by_k": by_k,
           "summary": {"k_values_controlled": [int(k) for k in by_k],
                       "control_lrs": sorted({r["lr"] for r in ctrl}),
                       "n_control": len(ctrl), "n_control_learned": sum(r["learned"] for r in ctrl),
                       "n_control_diverged": sum(r["diverged"] for r in ctrl),
                       "lr1e-4_learned_by_k": {k: (v["learned"] if v else None) for k, v in lo.items()},
                       "lr1e-4_val_mse_last_by_k": {k: (v["val_mse_last"] if v else None) for k, v in lo.items()},
                       "all_lr1e-4_learned": all(v and v["learned"] for v in lo.values())}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    for k, v in by_k.items():
        print(f"  K={k:>2} baseline lr={v['baseline_lr']} last={v['baseline_val_mse_last_mean']} learned {v['baseline_n_learned']}/{v['baseline_n']} | "
              + "  ".join(f"lr={lr}: last={c['val_mse_last']} learned={c['learned']} div={c['diverged']}" for lr, c in v["controls"].items()))


if __name__ == "__main__":
    main()
