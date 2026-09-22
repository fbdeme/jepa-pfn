"""Nuisance fraction vs. noise_scale sweep (C-1a: is the 1.3% a property of the data
or of our sigma choice?).

Reviewer critique (Devil's Advocate): the headline "mechanism cell is 98.7% structural"
is a readout of the prior's sigma ~ LogUniform(0.01, 0.3), not an empirical property of
tabular data -- nuisance = sigma^2/(1+sigma^2) is fixed by construction. This sweeps the
built-in noise_scale knob (prior/scm.py:46, sigma *= noise_scale) and records the nuisance
fraction at each level, so we can (a) show the 1.3% is sigma-determined and (b) supply the
x-axis for the training sweep that tests whether the latent objective turns on at high noise.

CPU only. Writes eval/results_sigma_sweep_nuisance.json (one record per noise_scale).
Reuses eval.nuisance_fraction.measure logic but parameterizes noise_scale.
"""
import json
import numpy as np

from prior.scm import SCMPrior, PriorConfig, FuncFamily


def measure_at(noise_scale, n_tables=4000, seed=0):
    cfg = PriorConfig(n_rows=(128, 384), n_cols=(4, 32),
                      p_categorize=0.0, p_missing_table=0.0,
                      noise_scale=noise_scale)
    prior = SCMPrior(cfg, seed=seed)
    mech_frac, theo_frac, sig_eff = [], [], []
    done = 0
    while done < n_tables:
        b = prior.sample_batch(min(256, n_tables - done))
        done += b.x.shape[0]
        for t in range(b.x.shape[0]):
            x, f = b.x[t], b.f[t]
            nodes = b.observed_idx[t]
            fam = b.func_family[t]
            sig = b.noise_sigma[t]
            for d in range(x.shape[1]):
                node = nodes[d]
                if fam[node] == FuncFamily.ROOT:
                    continue
                xv = x[:, d]
                vx = xv.var()
                if vx < 1e-12:
                    continue
                mech_frac.append(float((xv - f[:, d]).var() / vx))
                s = float(sig[node])
                theo_frac.append(s * s / (1.0 + s * s))
                sig_eff.append(s)
    mech = np.asarray(mech_frac)
    theo = np.asarray(theo_frac)
    seff = np.asarray(sig_eff)
    return {
        "noise_scale": noise_scale,
        "sigma_effective_range": [float(seff.min()), float(seff.max())],
        "sigma_effective_median": float(np.median(seff)),
        "n_mechanism_cells": int(mech.size),
        "nuisance_fraction_mean": float(mech.mean()),
        "nuisance_fraction_median": float(np.median(mech)),
        "nuisance_fraction_p90": float(np.percentile(mech, 90)),
        "structural_signal_fraction_mean": float(1.0 - mech.mean()),
        "theoretical_sigma2_over_1plus_sigma2_mean": float(theo.mean()),
    }


if __name__ == "__main__":
    # noise_scale=1 reproduces the paper's 1.3%. Sweep up to 10x (sigma_max -> 3.0,
    # nuisance -> ~90%) so a high-noise regime with real strippable nuisance is covered.
    scales = [1, 2, 3, 5, 7, 10]
    records = [measure_at(s) for s in scales]
    out = {"sweep": records}
    path = "eval/results_sigma_sweep_nuisance.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{'noise_scale':>11} {'sigma_eff_range':>22} {'nuisance_mean':>13} {'struct_frac':>11}")
    for r in records:
        lo, hi = r["sigma_effective_range"]
        print(f"{r['noise_scale']:>11} {f'[{lo:.3f}, {hi:.3f}]':>22} "
              f"{r['nuisance_fraction_mean']:>12.1%} {r['structural_signal_fraction_mean']:>10.1%}")
    print("wrote", path)
