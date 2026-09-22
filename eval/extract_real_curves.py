"""Logged val curves + dual collapse instruments for every real-prior run (docs/real_prior_plan.md 10.3),
and the pre-registered convergence rule: if the val metric (ds val_mse / dual val_jepa) still falls by
more than CONV_DROP over the last CONV_WINDOW steps, the arm's best-lr run is extended to 40k.

Run: uv run python -m eval.extract_real_curves [PREFIX]   -> eval/results_<PREFIX>train_curves.json
     (default PREFIX real_ -> results_real_train_curves.json; PREFIX cls_ -> results_cls_train_curves.json)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.rows_control import _curve_summary, curve             # noqa: E402

PREFIX = next((a for a in sys.argv[1:] if not a.startswith("--")), "real_")
OUT = ROOT / f"eval/results_{PREFIX}train_curves.json"
CONV_WINDOW = 1000
CONV_DROP = 0.01


def convergence(cv):
    if not cv:
        return None
    key = "val_mse" if "val_mse" in cv else ("val_nll" if "val_nll" in cv else "val_jepa")
    steps, vals = cv["steps"], cv[key]
    last = steps[-1]
    before = [v for s, v in zip(steps, vals) if s <= last - CONV_WINDOW]
    if not before:
        return None
    drop = (before[-1] - vals[-1]) / abs(before[-1]) if before[-1] else 0.0
    return {"metric": key, "last_step": last, "value_before": before[-1], "value_last": vals[-1],
            "rel_drop": round(float(drop), 4), "still_falling": bool(drop > CONV_DROP)}


def main():
    runs = sorted(p.name for p in (ROOT / "runs").glob(f"{PREFIX}*_lr*_s*") if (p / "metrics.jsonl").exists())
    out = {"rule": f"still_falling = val metric drops > {CONV_DROP:.0%} over the last {CONV_WINDOW} steps -> extend to 40k", "runs": {}}
    for run in runs:
        cv = curve(run)
        out["runs"][run] = {"curve": cv, "summary": _curve_summary(cv), "convergence": convergence(cv)}
        print(f"  {run:24} {out['runs'][run]['summary']}  conv={out['runs'][run]['convergence']}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
