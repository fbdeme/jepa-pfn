"""Training-curve summary of the architecture-ladder-2 runs (docs/prior_v2_plan.md 5.2) from the fetched runs/<run>/metrics.jsonl:
last val record, min val_mse and its step, and the pre-registered survival check (val_mse never returns to the constant map .936
after its first drop). Writes train_summary.json next to this file. Run: uv run python eval/arch_ladder_2026-09-11/summarize_runs.py"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
RUNS = ["TabularJEPA_v3_tabicl2_headenc_s0", "TabularJEPA_v3_tabicl2_dae_headenc_s0", "TabularJEPA_v3_tabicl2_sigreg_headenc_s0"]
CONST_MAP = 0.936   # v3's constant-map val_mse (docs/prior_v2_plan.md 5.2 survival rule)
out = {}
for r in RUNS:
    p = R / "runs" / r / "metrics.jsonl"
    if not p.exists():
        continue
    L = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    va = [v for v in L if "val_mse" in v]
    cfg = json.loads(json.dumps(__import__("yaml").safe_load(open(R / "runs" / r / "config.yaml"))))
    lo = min(va, key=lambda v: v["val_mse"])
    below = [v for v in va if v["val_mse"] < CONST_MAP]
    relapse = [v["step"] for v in va if below and v["step"] > below[0]["step"] and v["val_mse"] >= CONST_MAP]
    last = va[-1]
    out[r] = dict(steps_configured=cfg["steps"], last_step=last["step"], complete=last["step"] >= cfg["steps"],
                  last=dict(val_mse=last["val_mse"], val_ppd=last.get("val_ppd"), val_jepa=last["val_jepa"],
                            tgt_cos=last["val_tgt"]["cos"], pred_std=last["val_pred"]["dim_std"], tgt_erank=last["val_tgt"].get("erank")),
                  min_val_mse=lo["val_mse"], min_val_mse_step=lo["step"], first_below_const_map_step=below[0]["step"] if below else None,
                  relapse_steps=relapse, survived=bool(below) and not relapse and last["step"] >= cfg["steps"],
                  n_val=len(va), sec_total=[x for x in L if "sec" in x][-1]["sec"])
(D / "train_summary.json").write_text(json.dumps(out, indent=1) + "\n")
for r, s in out.items():
    print(f"{r}: step {s['last_step']}/{s['steps_configured']} val_mse last {s['last']['val_mse']} min {s['min_val_mse']}@{s['min_val_mse_step']} "
          f"first<{CONST_MAP} @{s['first_below_const_map_step']} relapse {s['relapse_steps']} survived {s['survived']} ({s['sec_total']/3600:.2f} h)")
