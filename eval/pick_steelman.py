"""Pick the pre-collapse steelman ckpt for each base lat arm (bsw_lat_*).

Base-scale SIGReg latent collapses (~step 5500: dim_std->0, cos->1). Matching the paper's
base_lat_s steelman methodology, we probe the pre-collapse peak, not the collapsed final.
For each arm we scan the step-tagged ckpts (saved by train_jepa save_steps) and choose the
step with the highest target dim_std among the non-collapsed vals (cos < COS_MAX), then copy
that ckpt_<step>.pt -> ckpt.pt so eval/sigma_sweep_probe.py (PREFIX=bsw) loads the steelman.

Run ON THE BOX after training:  uv run python -m eval.pick_steelman
Writes eval/results_bsw_steelman_pick.json (which step chosen per arm + its geometry).
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]
COS_MAX = 0.95   # a val with batch-cosine >= this is treated as collapsed
GLOB = os.environ.get("STEELMAN_GLOB", "bsw_lat_*")   # which runs/<glob> to steelman-pick
TAG = os.environ.get("STEELMAN_TAG", "bsw")           # output results_<TAG>_steelman_pick.json


def steelman_step(run_dir):
    """From metrics.jsonl, return (step, geometry) of the healthiest pre-collapse val
    that also has a saved ckpt_<step>.pt. Healthiest = max target dim_std with cos<COS_MAX."""
    best = None
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "val_tgt" not in r:
            continue
        g = r["val_tgt"]
        if g["cos"] >= COS_MAX or g["dim_std"] <= 0:
            continue
        if not (run_dir / f"ckpt_{r['step']}.pt").exists():
            continue
        if best is None or g["dim_std"] > best[1]["dim_std"]:
            best = (r["step"], g)
    return best


def main():
    picks = {}
    for run_dir in sorted(ROOT.glob(f"runs/{GLOB}")):
        arm = run_dir.name
        pick = steelman_step(run_dir)
        if pick is None:
            print(f"{arm}: NO non-collapsed ckpt found (collapsed from the start?)")
            picks[arm] = {"step": None}
            continue
        step, g = pick
        shutil.copy(run_dir / f"ckpt_{step}.pt", run_dir / "ckpt.pt")   # steelman -> ckpt.pt
        picks[arm] = {"step": step, "dim_std": g["dim_std"], "erank": g["erank"], "cos": g["cos"]}
        print(f"{arm}: steelman @ step {step}  dim_std={g['dim_std']:.3f} "
              f"erank={g['erank']:.1f} cos={g['cos']:.3f} -> ckpt.pt", flush=True)
    (ROOT / f"eval/results_{TAG}_steelman_pick.json").write_text(json.dumps(picks, indent=1))
    print(f"wrote eval/results_{TAG}_steelman_pick.json")


if __name__ == "__main__":
    main()
