"""Pure-lat pretraining runs (E1 best_cell_lat cell): is the *representation* still changing at the end of the run?
The JEPA loss is non-stationary (EMA target: it reaches its minimum in the near-collapse start and rises), so convergence is
read from the target-encoder geometry logged at every validation (erank, dim_std, dim_std_min, within_table_var_frac).
Writes eval/results_lat_pretrain_trajectory.json: per run, the metrics at 12.5/25/50/75/100 percent of steps, the last-quarter
relative change of each, and the step of the JEPA-loss minimum.  Run: uv run python -m eval.lat_pretrain_trajectory [RUN ...]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_lat_pretrain_trajectory.json"
FRACS = (0.125, 0.25, 0.5, 0.75, 1.0)
GEOM = ("erank", "dim_std", "dim_std_min", "within_table_var_frac")


def trajectory(run):
    rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
    v = [r for r in rows if "val_jepa" in r]
    at = {f: v[int(len(v) * f) - 1] for f in FRACS}
    out = {"steps": v[-1]["step"], "n_val": len(v),
           "val_jepa": {str(f): round(at[f]["val_jepa"], 4) for f in FRACS},
           "val_jepa_min": round(min(r["val_jepa"] for r in v), 4), "val_jepa_min_step": min(v, key=lambda r: r["val_jepa"])["step"]}
    for k in GEOM:
        if all(isinstance(at[f]["val_tgt"].get(k), (int, float)) for f in FRACS):
            out[k] = {str(f): round(at[f]["val_tgt"][k], 4) for f in FRACS}
            a, b = at[0.75]["val_tgt"][k], at[1.0]["val_tgt"][k]
            out[k + "_last_quarter_rel_change"] = round((b - a) / a, 4) if a else None
    return out


def main():
    runs = sys.argv[1:] or [f"v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s{s}" for s in range(3)]
    res = {r: trajectory(r) for r in runs}
    OUT.write_text(json.dumps({"fracs": list(FRACS), "runs": res}, indent=1) + "\n")
    for r, t in res.items():
        print(f"{r}: erank {list(t['erank'].values())} (last-quarter {t['erank_last_quarter_rel_change']:+.3f}) | val_jepa min {t['val_jepa_min']}@{t['val_jepa_min_step']} -> end {t['val_jepa']['1.0']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
