"""Markdown tables for the classification-head verdict (paper_v3/review/round13/cls_head_2026-09-04/verdict.md), derived from
eval/results_suite_cls.json, eval/results_cls_train_curves.json and eval/results_cls_val_baselines.json. No number is typed by hand:
the verdict file embeds this script's output verbatim. Run: uv run python -m eval.cls_verdict_tables"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
S = json.loads((ROOT / "eval/results_suite_cls.json").read_text())
C = json.loads((ROOT / "eval/results_cls_train_curves.json").read_text())["runs"]
B = json.loads((ROOT / "eval/results_cls_val_baselines.json").read_text())["seeds"]

f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"


def pair_row(name, p):
    return f"| {name} | {p['n']} | {p['wins']} : {p['losses']} | {p['p']:.1e} | {f(p['mean_gap'])} |"


print("### Real suite (clf datasets, CTX 512 / K_FEAT 32), paired sign tests (arm = mean over seeds)\n")
print("| comparison | n | wins : losses | p | mean gap |\n|---|---|---|---|---|")
for k, p in S["paired"].items():
    if p["n"]:
        print(pair_row(k, p))
for k, p in S["rematch_11_2"]["paired"].items():
    if p["n"]:
        print(pair_row(k + " (11.2)", p))
print("\n### Accuracy by class-count bucket\n")
cols = ["cls_ds", "cls_dual", "real_ds", "base_ds", "linear", "histgb"]
print("| bucket (n) | " + " | ".join(cols) + " | ctrl_ds (1.7e-4) |\n|---|" + "---|" * (len(cols) + 1))
for b, g in S["by_class_bucket"].items():
    rows = [r for r in S["datasets"] if r["bucket"] == b and r.get("ctrl_ds") is not None]
    ctrl = sum(r["ctrl_ds"] for r in rows) / len(rows) if rows else None
    print(f"| {b} ({g['n']}) | " + " | ".join(f(g[c]) for c in cols) + f" | {f(ctrl)} |")
print("\n### Synthetic held-out (same val stream per seed; error at label cells; NLL of the class head)\n")
print("| seed | majority err | logreg err | ds 5e-4 nll / err | dual 5e-4 nll / err | ds 1.7e-4 nll / err | dual 1.7e-4 nll / err | ds 5e-5 nll / err |\n|---|---|---|---|---|---|---|---|")
def cell(run, kind):
    s = C.get(run, {}).get("summary")
    if not s or s.get("steps") != 20000:
        return "-"
    return f"{s['nll_last']:.3f} / {s['err_last']:.3f}" if kind == "ds" else f"{s['cls_nll_last']:.3f} / {s['cls_err_last']:.3f}"
for seed in range(3):
    b = B[str(seed)]
    print(f"| {seed} | {b['majority_err']:.3f} | {b['logreg_err']:.3f} | {cell(f'cls_ds_lr5e-4_s{seed}', 'ds')} | {cell(f'cls_dual_lr5e-4_s{seed}', 'dual')} | "
          f"{cell(f'cls_ds_lr1.7e-4_s{seed}', 'ds')} | {cell(f'cls_dual_lr1.7e-4_s{seed}', 'dual')} | {cell(f'cls_ds_lr5e-5_s{seed}', 'ds')} |")
print("\n### Pre-registered verdicts\n")
print("| rule | 11 (lr 5e-4, registered) | 11.2 (ds re-matched to 1.7e-4) |\n|---|---|---|")
for k in S["verdict"]:
    print(f"| {k} | {S['verdict'][k]} | {S['rematch_11_2']['verdict'].get(k)} |")
