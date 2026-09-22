"""Issue #20 — does removing the resample address tax structurally close the gap?

#15 removed the column code by LINEAR projection after training (partial: 25%
residual, mse_f gap only 8% closed). This removes it at the source instead, by
training under a scheme that never pays it:

    resample (banked) -> fixed r (TabPFN v2) -> RoPE (v3)

Per-scheme we report the gap = data-space - latent inside a MATCHED pair (never
across schemes: both arms change, so absolutes are incomparable - §20 issue).
Plus slot_share (eval/address_leak.py) on the latent arm: the scheme-agnostic
address residual, the one ruler defined for all three.

The bar (SSOT V3.4) is C4 + 2-of-3 C5 vs the matched control, above the noise
floor (noise_grid.md §6). The prediction from #15: the address tax bites fam_acc
(37%) and edge (22%) far more than mse_f (8%) - exactly the two C5 metrics that
blocked #19 - so if any scheme helps, watch those.

Run: uv run python -m eval.addr_grid  ->  eval/results_addr_grid.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from prior.scm import PriorConfig, SCMPrior

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.address_leak import slot_share
from eval.cls_grid import NOISE_FLOOR, beats_floor, gap
from eval.latent_probe import load_jepa
from eval.noise_grid import ROOT, load_any, present, score
from train.data import make_batch

# scheme -> (data-space arm, latent arm). resample is banked (= the #15 pair).
# The last two (follow-up A) STACK #20's rope onto #19's CLS: same rope+CLS
# data-space control, latent = CLS target (pure) or CLS+cell CE (dual).
PAIRS = {"resample": ("p2_anycell", "p3b_mixed_tw2"),
         "fixed": ("p6adr_ds_fixed", "p6adr_lat_fixed"),
         "rope": ("p6adr_ds_rope", "p6adr_lat_rope"),
         "rope+cls": ("p6rc_ds", "p6rc_lat"),
         "rope+cls_dual": ("p6rc_ds", "p6rc_dual"),
         # V3.4 confirmation + a 3rd seed. Verdict fixed BEFORE seeing s2: the
         # mean mse_f gap across s0/s1/s2 with its spread, NOT best-of-3. s0 -.045
         # (win), s1 +.001 (tie) already span 6.5x the .007 floor.
         "rope+cls_s1": ("p6rc_ds_s1", "p6rc_lat_s1"),
         "rope+cls_s2": ("p6rc_ds_s2", "p6rc_lat_s2")}
EVAL_SCALE = 1.0
SEED = 41_000


def latent_slot_share(name, n_batches=12, batch=8):
    """Scheme-agnostic address residual on the latent arm's encoder."""
    m = load_jepa(name).eval()
    rng = np.random.default_rng(SEED)
    prior = SCMPrior(PriorConfig(), seed=SEED)
    vals = [slot_share(m.online, make_batch(prior, batch, "any_cell", rng))
            for _ in range(n_batches)]
    return round(float(np.mean(vals)), 4)


def main():
    out, sc = {}, {}
    for scheme, (ds, lat) in PAIRS.items():
        if not (present(ds) and present(lat)):
            print(f"{scheme}: missing arm(s), skipped", flush=True)
            continue
        for n in (ds, lat):
            sc.setdefault(n, score(load_any(n), EVAL_SCALE))
        g = gap(sc[ds], sc[lat])
        out[scheme] = dict(ds=sc[ds], lat=sc[lat], gap=g,
                           latent_wins=beats_floor(g),
                           lat_slot_share=latent_slot_share(lat))
        print(f"{scheme:9} {ds} vs {lat}\n  gap {g}\n  latent clears floor: "
              f"{beats_floor(g)}  slot_share(lat)={out[scheme]['lat_slot_share']}",
              flush=True)
    (ROOT / "eval/results_addr_grid.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote eval/results_addr_grid.json ({len(out)} schemes). "
          f"floor: {NOISE_FLOOR}")


if __name__ == "__main__":
    main()
