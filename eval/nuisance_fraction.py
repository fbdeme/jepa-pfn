"""Measure the nuisance fraction of the instrumented SCM prior (issue #14 defense).

The I-JEPA reconciliation argument: latent-target prediction wins in vision because
pixel space is dominated by high-entropy, task-irrelevant nuisance that a data-space
loss must waste capacity on; a latent target strips it. This quantifies the analogous
quantity in OUR prior: how much of a mechanism cell's variance is irreducible noise
(the only thing a latent target could beneficially strip) vs structural signal f(pa).

Prior construction (prior/scm.py): non-root node j has x = f + sigma_j * eps, with f
STANDARDIZED to unit variance before noise, so sigma_j is a noise-to-signal ratio
(log-uniform in [0.01, 0.3]). We measure the split EMPIRICALLY from returned (x, f),
not from sigma, so f's ~unit-variance standardization and the eps distribution are
reflected exactly. Postprocessing (categorization, missingness) is turned OFF: it is
orthogonal to the signal/noise decomposition and would only corrupt x - f.

nuisance_frac(cell) = Var(x - f) / Var(x)   (noise indep. of f => ~= noise/(sig+noise))

Roots (f := x, pure exogenous, no mechanism) are reported separately — they carry no
structural mechanism, so the "strip the nuisance" framing does not apply to them.

CPU only, prior is cheap. Writes eval/results_nuisance_fraction.json.
"""

import json
import numpy as np

from prior.scm import SCMPrior, PriorConfig, FuncFamily


def measure(n_tables=4000, seed=0):
    # base-FM prior regime (configs/base_ds.yaml) with postprocessing disabled so
    # x = f + noise cleanly on every non-root column.
    cfg = PriorConfig(n_rows=(128, 384), n_cols=(4, 32),
                      p_categorize=0.0, p_missing_table=0.0)
    prior = SCMPrior(cfg, seed=seed)

    mech_frac = []      # per non-root observed column: noise / var(x)
    theo_frac = []      # sigma^2 / (1 + sigma^2) cross-check
    n_root = 0
    n_mech = 0
    # batch tables to amortize sample_batch cost
    done = 0
    while done < n_tables:
        b = prior.sample_batch(min(256, n_tables - done))
        done += b.x.shape[0]
        for t in range(b.x.shape[0]):
            x, f = b.x[t], b.f[t]                 # (R, D)
            nodes = b.observed_idx[t]             # (D,) graph node per column
            fam = b.func_family[t]                # (N,)
            sig = b.noise_sigma[t]                # (N,)
            for d in range(x.shape[1]):
                node = nodes[d]
                if fam[node] == FuncFamily.ROOT:
                    n_root += 1
                    continue
                n_mech += 1
                xv = x[:, d]
                resid = xv - f[:, d]              # = sigma * eps
                vx = xv.var()
                if vx < 1e-12:
                    continue
                mech_frac.append(float(resid.var() / vx))
                s = float(sig[node])
                theo_frac.append(s * s / (1.0 + s * s))

    mech_frac = np.asarray(mech_frac)
    theo_frac = np.asarray(theo_frac)
    pct = lambda a, q: float(np.percentile(a, q))
    out = {
        "n_tables": n_tables,
        "n_mechanism_cells": n_mech,
        "n_root_cells": n_root,
        "root_share_of_observed": n_root / (n_root + n_mech),
        "nuisance_fraction_mechanism_cells": {
            "mean": float(mech_frac.mean()),
            "median": float(np.median(mech_frac)),
            "p10": pct(mech_frac, 10), "p90": pct(mech_frac, 90),
            "p99": pct(mech_frac, 99), "max": float(mech_frac.max()),
        },
        "structural_signal_fraction_mean": float(1.0 - mech_frac.mean()),
        "theoretical_sigma2_over_1plus_sigma2_mean": float(theo_frac.mean()),
    }
    return out, mech_frac, theo_frac


def _selfcheck():
    # empirical noise fraction must track the theoretical sigma^2/(1+sigma^2) closely.
    out, mech, theo = measure(n_tables=400, seed=1)
    assert abs(mech.mean() - theo.mean()) < 0.01, (mech.mean(), theo.mean())
    assert out["nuisance_fraction_mechanism_cells"]["mean"] < 0.15  # tiny by construction
    print("selfcheck ok:", out["nuisance_fraction_mechanism_cells"]["mean"])


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit()
    out, _, _ = measure()
    path = "eval/results_nuisance_fraction.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    m = out["nuisance_fraction_mechanism_cells"]
    print(f"mechanism cells: {out['n_mechanism_cells']}  roots: {out['n_root_cells']} "
          f"({out['root_share_of_observed']:.1%} of observed)")
    print(f"nuisance fraction (noise / var): mean {m['mean']:.4f}  median {m['median']:.4f}  "
          f"p90 {m['p90']:.4f}  p99 {m['p99']:.4f}  max {m['max']:.4f}")
    print(f"=> structural signal = {out['structural_signal_fraction_mean']:.1%} of mechanism-cell variance")
    print(f"theoretical sigma^2/(1+sigma^2) mean {out['theoretical_sigma2_over_1plus_sigma2_mean']:.4f}")
    print("wrote", path)
