"""Headline-scale learning-rate control (round-13 R5 ISSUE-06 follow-up).

The headline verdict compares base_ds and base_dual trained at one shared lr (1e-3). The
factor-substrate control showed a shared lr is not a matched optimization: the value arm
there trained at lr/10 and not at 1e-3. This scores the four base-scale control runs
(ds and dual at lr 1e-4 and 4e-3, seed 0) with EXACTLY the recipe family's masked-probe
protocol (eval/probe_masked_all.py: same prior, ctx, batches, eval seed, policies, floor
reference) so they sit on the verdict's own yardstick.

Run: uv run python -m eval.probe_lr_control            (CPU, frozen ckpts)
Writes eval/results_lr_control_probe.json
"""
import json
import re
from pathlib import Path

from eval.probe_masked_all import PROTOCOL, ROOT, load_auto, run_family

OUT = ROOT / "eval/results_lr_control_probe.json"
RUNS = [f"base_{arm}_lr{lr}_s0" for arm in ("ds", "dual") for lr in ("1e-4", "4e-3")]


def main():
    jobs = [(a, (lambda a=a: load_auto(a)),
             {"group": a.split("_")[1], "lr": float(re.search(r"_lr([0-9e.-]+)_", a).group(1)), "seed": 0})
            for a in RUNS if (ROOT / "runs" / a / "ckpt.pt").exists()]
    assert len(jobs) == 4, f"expected 4 control checkpoints, found {[j[0] for j in jobs]}"
    rec = run_family("lr_control", jobs, *PROTOCOL["recipe"], "base_ds_s0", "recipe")
    # the verdict's own numbers, for the side-by-side (alive-seed means at the shared lr)
    summ = json.loads((ROOT / "eval/results_probe_masked_all_summary.json").read_text())["recipe"]
    rec["reference_shared_lr"] = {
        "lr": 0.001, "cutoff": summ["cutoff"],
        "ds": {p: summ["per_group"]["ds"][p]["alive"] for p in ("any_cell", "mixed")},
        "dual": {p: summ["per_group"]["dual"][p]["alive"] for p in ("any_cell", "mixed")}}
    rec["summary"] = {}
    for a, r in rec["arms"].items():
        g = r["meta"]["group"]
        rec["summary"][a] = {p: {"mse_f": r[p]["cell"]["mse_f"],
                                 "alive": r[p]["cell"]["mse_f"] < summ["cutoff"][p],
                                 "vs_own_group_shared_lr": round(r[p]["cell"]["mse_f"] - summ["per_group"][g][p]["alive"]["mean"], 4),
                                 "vs_ds_shared_lr": round(r[p]["cell"]["mse_f"] - summ["per_group"]["ds"][p]["alive"]["mean"], 4)}
                             for p in ("any_cell", "mixed")}
    best_dual = {p: min(rec["summary"][a][p]["mse_f"] for a in rec["summary"] if a.startswith("base_dual"))
                 for p in ("any_cell", "mixed")}
    rec["summary"]["_sign"] = {p: {"best_dual_control": best_dual[p],
                                   "ds_shared_lr": summ["per_group"]["ds"][p]["alive"]["mean"],
                                   "ds_still_leads": best_dual[p] > summ["per_group"]["ds"][p]["alive"]["mean"]}
                               for p in ("any_cell", "mixed")}
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    print(json.dumps(rec["summary"], indent=1))


if __name__ == "__main__":
    main()
