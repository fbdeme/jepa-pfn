"""Seeded random-init floor for the C4 factor substrate (review C9).

The floor in results_probe_c4_s{s}.json is ONE deterministic draw (torch seed 999)
copied into every per-seed file, so the reported n=4/sd=0 floor is effectively n=1.
This produces a real floor band: N_DRAWS independently initialized encoders (same
architecture as the c4 ds arm), each scored with the SAME probe recipe as the
per-seed files (probe_base PROBE_PRIOR=factor path: c4_mse_f on FactorConfig
K=4, D=12, sigma=1.0, paired eval tables).

    uv run python -m eval.c4_floor_seeds

Writes eval/results_c4_floor_seeds.json (draws, mean, sd, min, max, n).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PROBE_PRIOR", "factor")   # must precede probe_base import

import numpy as np
import torch

from eval import probe_base as pb
from model.pfn import CellPFN

ROOT = Path(__file__).parents[1]
N_DRAWS = 30
CFG_RUN = "c4_factor_ds_s0"   # architecture/config source, matching the legacy floor


def main():
    c = torch.load(ROOT / "runs" / CFG_RUN / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]
    draws = []
    for d in range(N_DRAWS):
        torch.manual_seed(d)
        rand = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                       n_cls=c.get("n_cls") or 0,
                       r_scheme=c.get("r_scheme") or "resample").eval()
        mse = pb.c4_mse_f(pb.collect(rand, 1.0))
        draws.append(mse)
        print(f"draw {d:2d}: mse_f={mse}", flush=True)
    a = np.array(draws, float)
    out = dict(draws=draws, mean=round(float(a.mean()), 4), sd=round(float(a.std()), 4),
               min=round(float(a.min()), 4), max=round(float(a.max()), 4), n=N_DRAWS,
               note="seeded random-init floor, factor substrate (K=4, D=12, sigma=1.0); "
                    "same probe recipe as results_probe_c4_s*.json; replaces the single "
                    "seed-999 draw (.2426) replicated across per-seed files")
    (ROOT / "eval/results_c4_floor_seeds.json").write_text(json.dumps(out, indent=1))
    print(f"wrote eval/results_c4_floor_seeds.json  mean={out['mean']} sd={out['sd']} "
          f"range=[{out['min']}, {out['max']}]")


if __name__ == "__main__":
    main()
