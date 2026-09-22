"""Equal-total-step comparison of every two-stage arm (pretraining + fine-tune) against single-stage scratch at the nearest
available total budget, from the value read-outs only (2026-09-07: the 'is it just more steps?' question).
Run: uv run python -m eval.v4_equal_steps  -> eval/results_v4_equal_steps.json"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_v4_equal_steps.json"
SCRATCH = {5000: "v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s{s}", 20000: "real_ds_lr5e-4_s{s}",
           40000: "v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s{s}", 60000: "v4_ds_none_lam0_lr5e-4_any_cell_bar32_60k_s{s}"}
ARMS = [  # label, pretraining objective, pretrain steps, fine-tune steps, run pattern
    ("lat20k+ft5k", "latent", 20000, 5000, "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_5k_s{s}"),
    ("dsmix20k+ft5k", "ds-mixed", 20000, 5000, "v4_ds_dsinit20000_lam0_lr5e-4_any_cell_bar32_5k_s{s}"),
    ("lat20k+ft20k", "latent", 20000, 20000, "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_20k_s{s}"),
    ("lat40k+ft5k", "latent", 40000, 5000, "v4_ds_lat40000_lam0_lr5e-4_any_cell_bar32_5k_s{s}"),
    ("dsmix40k+ft5k", "ds-mixed", 40000, 5000, "v4_ds_dsinit40000_lam0_lr5e-4_any_cell_bar32_5k_s{s}"),
    ("lat20k+ft40k", "latent", 20000, 40000, "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s{s}"),
    ("dsmix20k+ft40k", "ds-mixed", 20000, 40000, "v4_ds_dsinit20000_lam0_lr5e-4_any_cell_bar32_40k_s{s}"),   # E10b, if run
    ("lat20k+ft60k", "latent", 20000, 60000, "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_60k_s{s}"),
]


def main():
    R = {}
    for p in sorted(glob.glob(str(ROOT / "eval/results_real_readout*.json"))):
        R.update(json.load(open(p))["runs"])
    v = lambda n: R[n]["select_mean"] if n in R else None
    rows = []
    for lab, obj, P, F, pat in ARMS:
        total = P + F; near = max((T for T in SCRATCH if T <= total), default=min(SCRATCH))
        a = [v(pat.format(s=s)) for s in range(3)]; b = [v(SCRATCH[near].format(s=s)) for s in range(3)]
        if any(x is None for x in a):
            continue
        gaps = [round(x - y, 4) for x, y in zip(a, b)]
        rows.append(dict(arm=lab, objective=obj, pretrain_steps=P, finetune_steps=F, total_steps=total, scratch_steps=near, values=a, scratch_values=b,
                         gap=gaps, mean_gap=round(sum(gaps) / 3, 4), two_stage_better_all_seeds=all(g < 0 for g in gaps), scratch_better_all_seeds=all(g > 0 for g in gaps),
                         exact_match=(near == total)))
    OUT.write_text(json.dumps(dict(protocol="fixed-R value MSE (select rows mean) from eval/results_real_readout*.json; scratch at the largest available total <= the two-stage total",
                                   rows=rows), indent=1) + "\n")
    for r in rows:
        print(f"{r['arm']:15} {r['objective']:9} total {r['total_steps']//1000:>2}k vs scratch {r['scratch_steps']//1000:>2}k{'' if r['exact_match'] else '*'}: gap {r['gap']} mean {r['mean_gap']:+.4f} two-stage-better-all {r['two_stage_better_all_seeds']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
