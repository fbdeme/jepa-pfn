"""Training-curve summary of the open-horizon pair (docs/prior_v2_plan.md 5.3) from the fetched runs/<run>/metrics.jsonl:
plateau stop record, best val_mse and its step, final val, hours, val count, snapshots, val_mse at every 20k, and (JEPA run)
the latent stats at the end and the constant map (from curves.json). Runs not fetched yet are skipped. Writes conv_summary.json next to this file.
Run: .venv/bin/python eval/open_horizon_2026-09-14/summarize_conv.py"""
import glob, json
from pathlib import Path

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
RUNS = ["PFN_tabicl2_ds_conv_s0", "TabularJEPA_v3_tabicl2_headenc_conv_s0"]
out = {}
for r in RUNS:
    p = R / "runs" / r / "metrics.jsonl"
    if not p.exists():
        continue
    L = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    tr = [x for x in L if "loss" in x]; va = [x for x in L if "val_mse" in x]; stop = [x for x in L if "stop" in x]
    best = min(va, key=lambda x: x["val_mse"])
    s = dict(stop=stop[-1] if stop else None, last_step=tr[-1]["step"], hours=round(tr[-1]["sec"] / 3600, 2), n_val=len(va),
             best_val_mse=best["val_mse"], best_step=best["step"], final_val_mse=va[-1]["val_mse"], final_step=va[-1]["step"],
             snapshots=sorted(int(f.split("snap_step")[1][:-3]) for f in glob.glob(str(R / "runs" / r / "snap_step*.pt"))),
             val_mse_20k={x["step"]: x["val_mse"] for x in va if x["step"] % 20000 == 0})
    if "val_jepa" in va[-1]:
        v = va[-1]; s["latent_last"] = dict(val_jepa=v["val_jepa"], tgt_cos=v["val_tgt"]["cos"], pred_std=v["val_pred"]["dim_std"], tgt_erank=v["val_tgt"]["erank"])
        cu = json.loads((D / "curves.json").read_text())[r]          # extract_curves.py: constant map of this run's held-out set
        s["const_map_mse"] = cu["const_map_mse"]
        s["first_below_const_map_step"] = next((x["step"] for x in va if x["val_mse"] < cu["const_map_mse"]), None)
        s["n_above_after_first_below"] = cu["n_above_after_first_below"]   # 0 = "never returned"
    out[r] = s
    print(r, "stop", s["stop"], "best", s["best_val_mse"], "@", s["best_step"], "final", s["final_val_mse"], "@", s["final_step"], f"{s['hours']} h", len(s["snapshots"]), "snaps")
(D / "conv_summary.json").write_text(json.dumps(out, indent=1) + "\n"); print("wrote", D / "conv_summary.json")
