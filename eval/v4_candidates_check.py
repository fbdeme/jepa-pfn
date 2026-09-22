"""C18: the candidate-experiment checks, derived from artifacts instead of argued in prose.

E5  - the pre-registered extension rule (docs/real_prior_plan.md 10.2: still falling if the val metric drops by more than
      CONV_DROP over the last CONV_WINDOW steps) applied to the 20k curves of the E1 best cells and the POST ds pair.
E1b - detectability of the registered contrast (sigreg vs ema at the registered lr): how many EMA runs collapsed at that lr
      in E1 (0 => the Fisher test can only detect sigreg being WORSE = one-sided), and the alternative lr where EMA collapses.
lat - the pure-latent (lambda 0) cells' head-free probes next to the head cells, ds and the random-init floor.
Run: uv run python -m eval.v4_candidates_check   -> eval/results_v4_candidates_check.json
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.extract_real_curves import CONV_DROP, CONV_WINDOW, convergence   # noqa: E402
from eval.rows_control import curve                                        # noqa: E402

OUT = ROOT / "eval/results_v4_candidates_check.json"
E1 = ROOT / "eval/results_v4_e1_collapse_grid.json"
SPEC = ROOT / "paper_v4/experiments.yaml"


def val_ppd_drop(run):
    rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl") if '"val_ppd"' in l]
    if len(rows) < 2:
        return None
    last = rows[-1]; before = [r for r in rows if r["step"] <= last["step"] - CONV_WINDOW]
    if not before or not before[-1]["val_ppd"]:
        return None
    d = (before[-1]["val_ppd"] - last["val_ppd"]) / abs(before[-1]["val_ppd"])
    return dict(metric="val_ppd", last_step=last["step"], value_before=before[-1]["val_ppd"], value_last=last["val_ppd"],
                rel_drop=round(float(d), 4), still_falling=bool(d > CONV_DROP))


def main():
    e1 = json.loads(E1.read_text()); spec = yaml.safe_load(SPEC.read_text()); ex = {e["id"]: e for e in spec["experiments"]}
    cells = e1["cells"]
    # ---- E5: extension rule on the existing 20k curves
    e5_runs = []
    for key in ("best_cell", "best_cell_lat"):
        bc = e1.get(key)
        if bc:
            e5_runs += [r for k, c in cells.items() if c["lambda_ppd"] == bc["lambda_ppd"] and c["lr"] == bc["lr"] for r in c["runs"]]
    e5_runs += [r for r in ex["E3"]["reuse_existing"] if "5e-4" in r]
    e5 = {}
    for r in e5_runs:
        cv = curve(r); c = convergence(cv) if cv else None
        e5[r] = {"logged_metric": c, "val_ppd": val_ppd_drop(r) if cv and "val_jepa" in cv else None}
    any_falling = any((v["logged_metric"] or {}).get("still_falling") or (v["val_ppd"] or {}).get("still_falling") for v in e5.values())
    # ---- E1b: detectability at the registered lr
    lr_reg = ex["E1b"]["axes"]["lr"][0]
    ema = [c for c in cells.values() if c["lr"] == lr_reg and c["lambda_ppd"] in ex["E1b"]["axes"]["lambda_ppd"]]
    n_ema, c_ema = sum(c["n_complete"] for c in ema), sum(c["n_collapsed"] for c in ema)
    by_lr = {}
    for c in cells.values():
        d = by_lr.setdefault(c["lr"], [0, 0]); d[0] += c["n_collapsed"]; d[1] += c["n_complete"]
    alt = [lr for lr, (k, n) in by_lr.items() if n and k == n]
    e1b = dict(registered_lr=lr_reg, ema_collapsed_at_registered_lr=[c_ema, n_ema],
               detectability="one-sided (EMA never collapses here: only 'sigreg worse' is detectable)" if c_ema == 0 else "two-sided",
               ema_collapse_by_lr={lr: f"{k}/{n}" for lr, (k, n) in sorted(by_lr.items(), key=lambda kv: float(kv[0]))},
               informative_lr=alt)
    # ---- pure lat vs head cells on head-free probes
    lat = {k: dict(lambda_ppd=c["lambda_ppd"], lr=c["lr"], collapsed=f"{c['n_collapsed']}/{c['n_complete']}", **c["probe"]) for k, c in cells.items()}
    out = dict(rule_e5=f"still falling = val metric drops > {CONV_DROP:.0%} over the last {CONV_WINDOW} steps",
               e5=dict(runs=e5, any_still_falling=any_falling, extension_needed=any_falling),
               e1b=e1b, lat_probe=dict(cells=lat, floor=e1.get("probe_floor"), best_cell_lat=e1.get("best_cell_lat")))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("E5 extension needed:", any_falling, "| E1b:", e1b["detectability"], "informative lr", alt, "| best_cell_lat", e1.get("best_cell_lat"))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
