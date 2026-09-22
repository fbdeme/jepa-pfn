"""CPU oracle boundary map: latent-advantage index over (K, sigma) for the factor substrate.

Sweeps intrinsic rank K (1 = maximal compression, D = no shared structure ~ SCM) and noise sigma,
computing the same median(R2_f) - median(R2_x) index as factor_oracle. Draws the PREDICTED boundary
of where a latent objective should be credited -- free, so it tells us which (few) points are worth
GPU training. The oracle is a linear-algebra proxy: it sees latent-favorability but NOT the trained
model's dynamics (collapse, ds learnability), so the GPU points also VALIDATE it.

Run: uv run python -m eval.factor_boundary   (CPU, seconds)
"""
import json
from pathlib import Path

from eval.factor_oracle import N_TABLES, SEED, _cross_column_r2
from prior.factor import FactorConfig, FactorPrior

ROOT = Path(__file__).parents[1]
D = 12
KS = [1, 2, 3, 4, 6, 8, 10, 12]
SIGMAS = [0.5, 1.0, 1.5, 2.0]


def main():
    grid = {}
    print(f"latent-advantage index = median(R2_f) - median(R2_x)   (D={D}, n_tables={N_TABLES})")
    print("       " + "".join(f"  sig{s:<4}" for s in SIGMAS))
    for K in KS:
        cells = []
        for s in SIGMAS:
            cfg = FactorConfig(n_rows=(256, 256), n_cols=(D, D), n_factors=K, sigma=s)
            r2f, r2x = _cross_column_r2(FactorPrior(cfg, seed=SEED).sample_batch(N_TABLES))
            idx = round(r2f - r2x, 3)
            grid[f"K{K}_s{s}"] = dict(K=K, sigma=s, r2_f=round(r2f, 3), r2_x=round(r2x, 3), index=idx)
            cells.append(idx)
        print(f"K={K:<3}  " + "".join(f" {v:+6.3f}" for v in cells))
    (ROOT / "eval/results_factor_boundary.json").write_text(json.dumps(grid, indent=1))
    print("\nindex high (>~0.2) => latent-favorable; near 0 => no advantage (SCM-like).")
    print("wrote eval/results_factor_boundary.json")


if __name__ == "__main__":
    main()
