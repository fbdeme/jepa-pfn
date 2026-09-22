"""Measure the address slot_share of the 35M arms (review C5 / H3 measurement half).

The 35M steelman is the paper's only non-collapsed evidence that the negative is a
property of the objective, but its address share was never measured, leaving the
address-domination alternative open (§8.4, H3). This closes the measurement half.

Recipe: the SAME scheme-agnostic column-permutation metric as eval/addr_grid.py
(slot_share from eval/address_leak.py; default PriorConfig, batch 8, 12 batches,
seed 41_000). Module under test: the representation the mechanism probes actually
evaluate — load_any(), i.e. JEPA.encoder() (EMA arms: target copy; SIGReg arms:
online) or the bare CellPFN for data-space. Reference points
(results_addr_grid.json, online-encoder recipe): base resample lat .336, rope+cls .012.

    uv run python -m eval.steelman_slot

Writes eval/results_steelman_slot.json.
"""
import json
from pathlib import Path

import numpy as np

from eval.address_leak import slot_share
from eval.noise_grid import load_any
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
# steelman'd latent arms + same-scale controls
ARMS = {"big_lat_s_h": "steelman (pure latent, SIGReg; encoder()==online)",
        "big_dual_lewm_h": "steelman'd dual+LeWM (EMA; encoder()==target copy)",
        "big_ds": "control (data-space CellPFN)",
        "big_dual": "control (main dual, EMA; encoder()==target copy)"}
SEED = 41_000       # identical recipe to addr_grid.latent_slot_share
N_BATCHES = 12
BATCH = 8


def main():
    out = {}
    for name, note in ARMS.items():
        enc = load_any(name)  # the module the mechanism probes evaluate
        rng = np.random.default_rng(SEED)
        prior = SCMPrior(PriorConfig(), seed=SEED)
        vals = [slot_share(enc, make_batch(prior, BATCH, "any_cell", rng))
                for _ in range(N_BATCHES)]
        out[name] = dict(slot_share=round(float(np.mean(vals)), 4),
                         slot_share_sd=round(float(np.std(vals)), 4),
                         n_batches=N_BATCHES, batch=BATCH, seed=SEED, note=note,
                         recipe="addr_grid.latent_slot_share (column-permutation, "
                                "address_leak.slot_share) on load_any()")
        print(name, out[name], flush=True)
    (ROOT / "eval/results_steelman_slot.json").write_text(json.dumps(out, indent=1))
    print("wrote eval/results_steelman_slot.json")


if __name__ == "__main__":
    main()
