"""Latent-factor prior: a deliberately latent-favorable substrate for the C4 positive control.

Unlike the SCM prior (each column's mechanism is its own function of a few parents, so there is
NO shared low-dimensional cause across columns and a latent objective has nothing to amortize),
here every observed column is a noisy linear view of K shared latent factors:

    z ~ N(0, I_K)  per row      (R, K)   -- the shared latent state
    W ~ N(0, I)    per table    (D, K)   -- the loadings (the "mechanism" to infer in-context)
    f_d = standardize(z @ W_d)           -- clean signal per column (unit variance, as in SCM)
    x_d = f_d + sigma * eps              -- observed value; sigma LARGE => value prediction is
                                            noise-limited while the shared signal f is clean.

This is the regime JEPA-style latent prediction is designed to win: predicting a masked cell's
VALUE is capped at the noise floor, but predicting its REPRESENTATION (which can converge to the
predictable shared z) discards the unpredictable per-cell noise. The instrument's mechanism probe
(rep -> f, eval/probe_base.c4_mse_f) then favors the latent arm -- IF the objective realizes it.

sample_batch() returns an SCMBatch so training (train/ only reads x) and the probe (reads f +
func_family) reuse the SCM pipeline unchanged. Ground truth we own as gods: f (clean signal),
the bipartite z->column adjacency, per-column sigma. Training may only see x.

Nodes: 0..K-1 = latent factors (ROOT, hidden), K..K+D-1 = observed columns (non-root). Column
order carries no topology (the factor model is column-symmetric), so no shuffle is needed.
"""

from dataclasses import dataclass

import numpy as np

from prior.scm import FuncFamily, SCMBatch


@dataclass(frozen=True)
class FactorConfig:
    n_rows: tuple = (128, 256)     # rows per table (context + query), sampled per batch
    n_cols: tuple = (12, 12)       # observed columns D; D >> K so z is identifiable from context
    n_factors: int = 4             # K shared latent factors (the compressible abstraction)
    sigma: float = 1.0             # per-cell noise std added to the unit-variance signal (SNR=1/sigma^2).
                                   # large sigma => value prediction is noise-limited (latent-favorable).


class FactorPrior:
    """Infinite stream of instrumented latent-factor tables. Prior = this code."""

    def __init__(self, config: FactorConfig = FactorConfig(), seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int) -> SCMBatch:
        c, rng = self.config, self.rng
        R = int(rng.integers(c.n_rows[0], c.n_rows[1] + 1))
        D = int(rng.integers(c.n_cols[0], c.n_cols[1] + 1))
        K = c.n_factors
        N = K + D

        x = np.empty((batch_size, R, D), np.float32)
        f = np.empty((batch_size, R, D), np.float32)
        adjacency = np.zeros((batch_size, N, N), bool)
        observed_idx = np.empty((batch_size, D), np.int16)
        func_family = np.zeros((batch_size, N), np.int8)
        noise_sigma = np.zeros((batch_size, N), np.float32)
        categorical = np.zeros((batch_size, D), np.int8)

        cols = np.arange(K, N, dtype=np.int16)          # observed-column node indices
        for b in range(batch_size):
            z = rng.standard_normal((R, K))             # (R, K) shared latent state
            W = rng.standard_normal((D, K))             # (D, K) per-table loadings
            raw = z @ W.T                               # (R, D) clean signal, pre-standardization
            std = raw.std(axis=0, keepdims=True)
            fb = (raw - raw.mean(axis=0, keepdims=True)) / (std + 1e-8)   # unit-variance per column
            xb = fb + c.sigma * rng.standard_normal((R, D))
            x[b], f[b] = xb.astype(np.float32), fb.astype(np.float32)
            adjacency[b, :K, K:] = True                 # every factor -> every column (bipartite)
            func_family[b, K:] = int(FuncFamily.MLP)    # columns are non-root => probe keeps them
            noise_sigma[b, K:] = c.sigma
            observed_idx[b] = cols
        return SCMBatch(x, f, adjacency, observed_idx, func_family, noise_sigma, categorical)


def _demo():
    """Self-check: shapes, x = f + sigma*noise, and f is genuinely rank-K (shared low-dim cause)."""
    cfg = FactorConfig(n_rows=(512, 512), n_cols=(12, 12), n_factors=4, sigma=1.5)
    bt = FactorPrior(cfg, seed=0).sample_batch(3)
    assert bt.x.shape == (3, 512, 12) and bt.f.shape == (3, 512, 12)
    resid = (bt.x - bt.f).std()
    assert abs(resid - 1.5) < 0.1, f"noise std {resid} != sigma 1.5"          # x = f + sigma*eps
    sv = np.linalg.svd(bt.f[0] - bt.f[0].mean(0), compute_uv=False)
    rank = int((sv > 1e-3 * sv[0]).sum())
    assert rank == 4, f"signal rank {rank} != K=4"                            # f is low-rank (shared z)
    assert (bt.func_family[:, 4:] == int(FuncFamily.MLP)).all()               # columns non-root
    print(f"OK: x=f+{resid:.2f}eps, signal rank {rank}=K, shapes {bt.x.shape}")


if __name__ == "__main__":
    _demo()
