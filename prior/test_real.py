"""Self-check for the real-data-matched prior. Run: uv run python -m prior.test_real"""
import numpy as np

from prior.real import RealConfig, RealPrior
from prior.scm import FuncFamily, SCMBatch, SCMPrior


def test_scm_unchanged():
    a = SCMPrior(seed=42).sample_batch(4)
    assert a.hyper is None and len(SCMBatch.__dataclass_fields__) == 8


def test_shapes_and_latents():
    p = RealPrior(RealConfig(p_missing_table=0.0), seed=7)
    for _ in range(20):
        b = p.sample_batch(2)
        B, R, D = b.x.shape
        K = b.hyper[0]["K"]
        assert b.adjacency.shape == (B, D + K, D + K) and 16 <= R <= 1024 and 3 <= D <= 65   # 64 features + target
        assert K == round(b.hyper[0]["rho"] * D)
        assert b.hyper[0]["D"] * b.hyper[0]["R"] <= 32768 or b.hyper[0]["R"] == 10   # cell budget
        assert not b.adjacency[:, :, :K].any(), "latents must be roots"
        assert (b.func_family[:, :K] == int(FuncFamily.ROOT)).all() and (b.noise_sigma[:, :K] == 0).all()
        assert set(b.observed_idx[0].tolist()) == set(range(K, K + D))
        # x - f is the per-node noise on continuous non-root columns (categorised columns are digitised)
        cont = (b.func_family[0][b.observed_idx[0]] != int(FuncFamily.ROOT)) & (b.categorical[0] == 0)
        if cont.any() and R >= 128:
            res = b.x[0][:, cont] - b.f[0][:, cont]
            assert np.allclose(res.std(0), b.noise_sigma[0][b.observed_idx[0]][cont], rtol=0.5, atol=0.05)
        if K:
            assert b.hyper[0]["latent_parents_mean"] > 0
        tc = b.hyper[0]["target_col"]                  # the target is continuous and exactly f + sigma * eps
        if tc is not None and R >= 128:
            assert b.categorical[0][tc] == 0
            res = b.x[0][:, tc] - b.f[0][:, tc]
            res = res[~np.isnan(res)]
            assert abs(res.std() - b.hyper[0]["target_sigma"]) <= 0.35 * b.hyper[0]["target_sigma"] + 0.02
        assert 0 < b.hyper[0]["sigma_table"] <= 3.0 * np.exp(0.3) + 1e-6


def test_force_cols():
    b = RealPrior(RealConfig(n_rows=(1024, 1024), force_cols=64), seed=2).sample_batch(1)
    D, R = b.hyper[0]["D"], b.hyper[0]["R"]
    assert D in (64, 65) and R == 32768 // D


def test_reproducible():
    a = RealPrior(seed=3).sample_batch(3); b = RealPrior(seed=3).sample_batch(3)
    assert np.array_equal(a.x, b.x, equal_nan=True) and a.hyper == b.hyper


if __name__ == "__main__":
    for fn in (test_scm_unchanged, test_shapes_and_latents, test_force_cols, test_reproducible):
        fn(); print("ok", fn.__name__)


def test_target_classes():
    """Classification prior: the designated target becomes a K-class label column, K from the pool, ids 0..K-1."""
    cfg = RealConfig(n_rows=(64, 64), force_cols=6, target_classes=(3, 5))
    pr = RealPrior(cfg, seed=3)
    seen = set()
    for _ in range(12):
        b = pr.sample_batch(2)
        for i, h in enumerate(b.hyper):
            if h["target_col"] is None:
                continue
            K = h["n_classes"]; assert K in (3, 5); seen.add(K)
            col = b.x[i][:, h["target_col"]]
            vals = col[~np.isnan(col)]
            assert np.all(vals == np.round(vals)) and vals.min() >= 0 and vals.max() <= K - 1, (K, vals.min(), vals.max())
            assert b.categorical[i][h["target_col"]] == K
            # C4: the cut that made the ids is in hyper, and re-applying it to the stored column reproduces... nothing
            # (x is already ids), but on the noiseless f it must give the SAME ids on the rows where noise did not
            # cross a cut - i.e. at least the majority of rows
            e, perm = h["class_edges"], h["class_perm"]
            assert len(e) == K - 1 and e == sorted(e) and sorted(perm) == list(range(K))
            ids_f = np.asarray(perm)[np.digitize(b.f[i][:, h["target_col"]], e)]
            assert (ids_f[~np.isnan(col)] == vals).mean() > 0.5
    assert seen == {3, 5}
    b0 = pr.sample_batch(1)
    assert all(h["class_edges"] is None for h in b0.hyper if not h["n_classes"])
    # nominal ids: the class means of the noiseless target are NOT increasing in the id for every table (identity map would be)
    pr = RealPrior(cfg, seed=3); increasing = []
    for _ in range(12):
        b = pr.sample_batch(2)
        for i, h in enumerate(b.hyper):
            col, f = b.x[i][:, h["target_col"]], b.f[i][:, h["target_col"]]
            means = [f[col == k].mean() for k in range(h["n_classes"]) if (col == k).any()]
            increasing.append(bool(np.all(np.diff(means) > 0)))
    assert not all(increasing), "class ids are still ordinal"
    b0 = RealPrior(RealConfig(n_rows=(64, 64), force_cols=6), seed=3).sample_batch(1)
    assert b0.hyper[0]["n_classes"] == 0
