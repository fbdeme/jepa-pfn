"""Address share, and the geometry that fails to detect it, for every arm the paper reports.

Round 4 measured the column-address share of the sigma-sweep pilot latent arms for the first
time and found them address-dominated -- the same arms Sections 4.3.1 and 5.3 certify as the
paper's clean, non-collapsed evidence. That measurement existed only as a review re-run; this
script fixes it in a committed artifact and extends it to every reported arm.

It records two things per arm, deliberately together:
  slot_share -- fraction of the representation recoverable from column identity alone
                (eval/address_leak.slot_share, column-permutation metric)
  dim_std / erank / cos -- the geometric health checks, measured HERE on the probed backbone
                (enc.encode of the encoder the probes read), not on a training-time target or
                projector output. The paper's steelman geometry macros come from the training
                loop's target-side statistic, which is a different tensor; keeping both makes
                the mismatch visible instead of implicit.

The point of the pair: an arm can be address-dominated while every geometric certificate reads
healthy, which is what makes rank and per-dimension spread unfit to certify content.

Run: uv run python -m eval.slot_share_all      (CPU, forward-only, frozen checkpoints)
Writes eval/results_slot_share_all.json
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.address_leak import slot_share          # noqa: E402
from eval.noise_grid import load_any              # noqa: E402
from model.jepa import collapse_stats             # noqa: E402
from prior.scm import PriorConfig, SCMPrior       # noqa: E402
from train.data import make_batch                 # noqa: E402

OUT = ROOT / "eval" / "results_slot_share_all.json"
EVAL_SEED = 41_000
N_BATCHES, BATCH = 12, 8

# arm -> (group, note). Groups mirror how the paper cites them.
ARMS = [
    ("base_ds_s0", "base", "data-space"),
    ("base_dual_s0", "base", "EMA latent + head"),
    ("base_lat_s_s0", "base", "SIGReg latent (collapsed)"),
    ("base_ema_nohead_s1", "base", "EMA, no head (the surviving seed)"),
    ("sw_ds_n1_s0", "sigma-sweep", "data-space, noise 1"),
    ("sw_lat_n1_s0", "sigma-sweep", "SIGReg latent, noise 1"),
    ("sw_ds_n10_s0", "sigma-sweep", "data-space, noise 10"),
    ("sw_lat_n10_s0", "sigma-sweep", "SIGReg latent, noise 10"),
    ("big_ds", "35M", "data-space"),
    ("big_dual", "35M", "EMA latent + head"),
    ("big_lat_s_h", "35M", "SIGReg latent, steelman"),
    ("big_dual_lewm_h", "35M", "SIGReg + head, steelman"),
]


@torch.no_grad()
def measure(enc, batches):
    # slot_share draws its k_perm column permutations from the global torch RNG at call
    # time, so seeding once before the arm loop would give each arm a different
    # permutation sequence. Re-seed here to make the across-arm comparison actually paired.
    torch.manual_seed(EVAL_SEED)
    slots, ds_, er_, co_ = [], [], [], []
    for b in batches:
        slots.append(slot_share(enc, b))
        h = enc.encode(b["z"], b["input_mask"], b["split"])   # the tensor the PROBES read
        g = collapse_stats(h[b["target_mask"]])
        ds_.append(g["dim_std"]); er_.append(g["erank"]); co_.append(g["cos"])

    def agg(v):
        return {"mean": round(st.mean(v), 4),
                "sd": round(st.stdev(v), 4) if len(v) > 1 else 0.0}
    return {"slot_share": agg(slots), "dim_std": agg(ds_),
            "erank": agg(er_), "cos": agg(co_), "n_batches": len(batches)}


def main():
    torch.manual_seed(EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    prior = SCMPrior(PriorConfig())
    batches = [make_batch(prior, BATCH, "any_cell", rng) for _ in range(N_BATCHES)]

    res, t0 = {}, time.time()
    for name, group, note in ARMS:
        if not (ROOT / "runs" / name / "ckpt.pt").exists():
            print(f"{name:20} SKIP (no checkpoint)", flush=True)
            continue
        t = time.time()
        r = measure(load_any(name).eval(), batches)
        r.update(group=group, note=note)
        res[name] = r
        print(f"{name:20} slot={r['slot_share']['mean']:.4f}"
              f"±{r['slot_share']['sd']:.4f}  dim_std={r['dim_std']['mean']:.3f}"
              f"  erank={r['erank']['mean']:.1f}  cos={r['cos']['mean']:.3f}"
              f"   [{time.time()-t:.0f}s]", flush=True)
    res["_protocol"] = {
        "slot_share": "eval/address_leak.slot_share -- permute column slots, un-permute the "
                      "output, attribute what fails to return to the address",
        "geometry": "model.jepa.collapse_stats on enc.encode(...) at the target cells, i.e. on "
                    "the backbone the probes read (NOT the training-time target/projector "
                    "output that the steelman geometry macros come from)",
        "batches": N_BATCHES, "batch": BATCH, "eval_seed": EVAL_SEED,
        "note": "arms are frozen; the same batches AND the same column permutations are "
                "reused across arms (the RNG is re-seeded per arm), so the comparison is "
                "paired",
        "k_perm": "slot_share's default 8. Whether the share is protocol-free is MEASURED in "
                   "eval/slot_share_kperm.py (results_slot_share_kperm.json), not asserted "
                   "here: this field once carried three hand-typed values, all of them "
                   "wrong, and paper/build_tables.py regex-scraped them into macros.",
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
