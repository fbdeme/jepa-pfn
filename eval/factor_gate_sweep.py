"""The fairness gate across the WHOLE factor sweep, plus the generator's own ceiling.

Round 12 (three reviewers, convergent; reproduced): the paper attached its
value-learnability caveat to the compressibility reading only, while the gate estimator --
run across the sweep instead of at one rank -- disqualifies the substrate for the
data-space arm at EVERY rank, including the one where it converges. And the named
deciding rerun at a single lower sigma still fails the gate at high rank. Both facts
were computed ad hoc in review; this commits them as artifacts the paper can anchor.

Two parts, one JSON:

  gate      _cross_column_r2 (the project's own estimator, eval/factor_oracle.py) on the
            probe protocol's table geometry, for every trained rank x sigma in
            {0.5, 0.6, 1.0}. r2_x below the R2X_FLOOR the selection code requires means
            the level is unfair to the value arm.
  ceiling   the generator's closed-form reducible fraction: the factor model is jointly
            Gaussian per table, so Var(x_d | x_-d) follows from Sigma = LL^T + s^2 I by
            Gaussian conditioning (loadings standardized exactly as sample_batch does).
            Median over columns and loading draws. If this is far from zero while the
            trained value arm captures nothing, the arm's stall is not noise-limited --
            it is unseparated from an optimization failure, and the paper must say so.

Run: uv run python -m eval.factor_gate_sweep            (CPU, ~minutes)
Writes eval/results_factor_gate_sweep.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.factor_oracle import (R2F_FLOOR, R2X_FLOOR, SEED,          # noqa: E402
                                _cross_column_r2)
from prior.factor import FactorConfig, FactorPrior                    # noqa: E402

OUT = ROOT / "eval/results_factor_gate_sweep.json"
KS = (1, 2, 4, 6, 8, 12)
SIGMAS = (0.5, 0.6, 1.0)
N_TABLES = 80          # factor_oracle's own table count
N_DRAWS = 400          # loading draws for the closed-form ceiling
D = 12


def closed_form_reducible(K, sigma, rng):
    """Median over columns/draws of 1 - Var(x_d|x_-d)/Var(x_d), by Gaussian conditioning.
    Loadings standardized to unit signal variance per column, as sample_batch does."""
    fr = []
    for _ in range(N_DRAWS):
        W = rng.standard_normal((D, K))
        W = W / np.linalg.norm(W, axis=1, keepdims=True)     # unit-variance clean signal
        Sig = W @ W.T + (sigma ** 2) * np.eye(D)
        for d in range(D):
            o = [j for j in range(D) if j != d]
            cond = Sig[d, d] - Sig[d, o] @ np.linalg.solve(Sig[np.ix_(o, o)], Sig[o, d])
            fr.append(1.0 - cond / Sig[d, d])
    return float(np.median(fr))


def main():
    out = {"protocol": {"gate": "eval/factor_oracle.py _cross_column_r2, floors "
                                f"r2_f>={R2F_FLOOR} (latent learnable) and "
                                f"r2_x>={R2X_FLOOR} (value arm fair), on the probe "
                                "geometry (178 rows, 12 cols)",
                        "ceiling": "closed-form Gaussian conditioning on Sigma=WW^T+s^2I, "
                                   f"median over {N_DRAWS} loading draws x {D} columns"},
           "gate": {}, "ceiling": {}}
    rng = np.random.default_rng(SEED)
    for K in KS:
        out["ceiling"][str(K)] = round(closed_form_reducible(K, 1.0, rng), 4)
    for s in SIGMAS:
        for K in KS:
            bt = FactorPrior(FactorConfig(n_rows=(178, 178), n_cols=(D, D),
                                          n_factors=K, sigma=s),
                             seed=SEED).sample_batch(N_TABLES)
            r2f, r2x = _cross_column_r2(bt)
            out["gate"][f"K{K}_s{s}"] = {
                "r2_f": round(r2f, 4), "r2_x": round(r2x, 4),
                "fair_to_value_arm": bool(r2x >= R2X_FLOOR),
                "latent_learnable": bool(r2f >= R2F_FLOOR)}
            print(f"  K={K:>2} sigma={s}  r2_f={r2f:.3f} r2_x={r2x:.3f}  "
                  f"fair={'Y' if r2x >= R2X_FLOOR else 'n'}", flush=True)
    print("ceiling (reducible fraction, sigma=1.0):",
          {k: v for k, v in out['ceiling'].items()})
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
