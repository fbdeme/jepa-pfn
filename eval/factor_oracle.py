"""C4 substrate validation (CPU, no GPU): is the latent-factor prior actually latent-favorable?

A latent objective can only beat its data-space twin when the substrate has a compressible shared
structure buried under per-cell noise the data-space objective is forced to model. We prove that
property analytically BEFORE spending any GPU, with a linear cross-column oracle:

    latent-advantage index = R2_f - R2_x   (averaged over columns and tables), where
      R2_f = how well a column's CLEAN signal f_d is linearly recovered from the other columns' x,
      R2_x = how well its NOISY value x_d is recovered from the other columns' x.

R2_f high  => the shared signal is amortizable across columns (a latent target exists).
R2_x low   => value prediction is noise-limited (data-space wastes capacity on noise).
index > 0  => latent-favorable. On the SCM prior (per-column mechanisms, no shared cause) the
index is ~0 no matter the noise -- which is exactly why our sigma-sweep never revived the latent
arm. We sweep sigma on the factor prior to pick ONE operating point (max index with R2_f still
high) = the C4 substrate; SCM (paper default + matched D) is the ~0 baseline.

Deterministic: fixed seed, fixed n_tables. Writes eval/results_factor_oracle.json.
Run: uv run python -m eval.factor_oracle
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from prior.factor import FactorConfig, FactorPrior
from prior.scm import PriorConfig, SCMPrior

ROOT = Path(__file__).parents[1]
N_TABLES = 80
SEED = 0
SIGMAS = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]   # factor per-cell noise levels to select among
GATE_INDEX = 0.15                 # factor index must clear this to be latent-favorable
GATE_RATIO = 3.0                  # ... and beat the SCM baseline by this factor
R2F_FLOOR = 0.5                   # latent target must be solidly learnable (lat can succeed)
R2X_FLOOR = 0.5                   # value prediction must ALSO be learnable, so the data-space arm
                                  # gets traction -> a fair both-arms-train control. Learned the hard
                                  # way: at sigma=1.0 (R2_x .272) the transformer ds stuck at val_mse
                                  # 1.0 (SNR 1, value prediction noise-limited). We need BOTH arms to
                                  # train, then show latent's representation is the better one.


def _cross_column_r2(bt):
    """Median over tables/columns of (R2 recovering clean f_d, R2 recovering noisy x_d) from other cols.

    Median, not mean: r2_score is unbounded below (a bad ridge extrapolation on a heavy-tailed SCM
    root column can score -400), so the mean is dominated by a handful of outliers while the typical
    column is well-behaved. The median is the robust central tendency of the per-column advantage.
    """
    r2f, r2x = [], []
    for b in range(bt.x.shape[0]):
        X, F = bt.x[b], bt.f[b]                 # (R, D)
        R, D = X.shape
        half = R // 2
        for d in range(D):
            others = [j for j in range(D) if j != d]
            Xtr, Xte = X[:half][:, others], X[half:][:, others]
            for target, lst in ((F, r2f), (X, r2x)):
                m = Ridge(alpha=1.0).fit(Xtr, target[:half, d])
                lst.append(r2_score(target[half:, d], m.predict(Xte)))
    return float(np.median(r2f)), float(np.median(r2x))


def _entry(label, bt):
    r2f, r2x = _cross_column_r2(bt)
    return dict(label=label, r2_f=round(r2f, 4), r2_x=round(r2x, 4), index=round(r2f - r2x, 4))


def main():
    out = {"n_tables": N_TABLES, "seed": SEED,
           "note": "index = median(R2_f) - median(R2_x), cross-column linear ridge (median robust to r2 outliers)"}

    factor = []
    for s in SIGMAS:
        cfg = FactorConfig(n_rows=(256, 256), n_cols=(12, 12), n_factors=4, sigma=s)
        e = _entry(f"factor_K4_D12_sigma{s}", FactorPrior(cfg, seed=SEED).sample_batch(N_TABLES))
        e["sigma"] = s
        factor.append(e)
    out["factor_sweep"] = factor

    # SCM baselines (expected index ~0): paper default, and D matched to the factor substrate.
    # Continuous + no missingness so the contrast isolates latent structure, not NaN/categorical
    # handling (the factor substrate is continuous and complete).
    scm = {"n_rows": (256, 256), "n_hidden": (0, 3), "p_missing_table": 0.0, "p_categorize": 0.0}
    out["scm_default"] = _entry("scm_default", SCMPrior(PriorConfig(**scm), seed=SEED).sample_batch(N_TABLES))
    out["scm_matched_D12"] = _entry(
        "scm_matched_D12",
        SCMPrior(PriorConfig(n_cols=(12, 12), **scm), seed=SEED).sample_batch(N_TABLES))

    # select: largest latent advantage among points where BOTH arms can learn -- the latent target
    # is solidly learnable (r2_f >= floor, lat can succeed) AND value prediction has traction
    # (r2_x >= floor, ds gets a fair shot). A control where ds cannot train is not a fair flip.
    learnable = [e for e in factor if e["r2_f"] >= R2F_FLOOR and e["r2_x"] >= R2X_FLOOR] or factor
    best = max(learnable, key=lambda e: e["index"])
    scm_index = max(out["scm_default"]["index"], out["scm_matched_D12"]["index"], 1e-6)
    passed = best["index"] >= GATE_INDEX and best["index"] >= GATE_RATIO * scm_index
    out["selected"] = dict(sigma=best["sigma"], index=best["index"], r2_f=best["r2_f"], r2_x=best["r2_x"],
                           note=f"max index s.t. r2_f>={R2F_FLOOR} (lat learnable) AND r2_x>={R2X_FLOOR} (ds learnable)")
    out["gate"] = dict(gate_index=GATE_INDEX, gate_ratio=GATE_RATIO, scm_index_max=round(scm_index, 4),
                       ratio_over_scm=round(best["index"] / scm_index, 2), passed=bool(passed))

    (ROOT / "eval/results_factor_oracle.json").write_text(json.dumps(out, indent=2))
    print("factor sweep (index = R2_f - R2_x):")
    for e in factor:
        print(f"  sigma {e['sigma']:<4} R2_f {e['r2_f']:.3f}  R2_x {e['r2_x']:.3f}  index {e['index']:+.3f}")
    print(f"SCM default        R2_f {out['scm_default']['r2_f']:.3f}  R2_x {out['scm_default']['r2_x']:.3f}  index {out['scm_default']['index']:+.3f}")
    print(f"SCM matched D=12   R2_f {out['scm_matched_D12']['r2_f']:.3f}  R2_x {out['scm_matched_D12']['r2_x']:.3f}  index {out['scm_matched_D12']['index']:+.3f}")
    print(f"-> selected sigma {best['sigma']}, index {best['index']:+.3f} "
          f"({out['gate']['ratio_over_scm']}x SCM); GATE {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
