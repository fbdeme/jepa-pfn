"""Self-check for the instrumented SCM prior. Run: uv run python -m prior.test_scm"""

import numpy as np

from prior.scm import FuncFamily, PriorConfig, SCMBatch, SCMPrior

CLEAN = dict(p_categorize=0.0, p_missing_table=0.0)  # postprocess off


def test_seed_reproducibility():
    a = SCMPrior(seed=42).sample_batch(8)
    b = SCMPrior(seed=42).sample_batch(8)
    for name in SCMBatch.__dataclass_fields__:
        if name == "hyper":                      # prior/real.py only; None for the SCM prior
            assert a.hyper is None and b.hyper is None
            continue
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    c = SCMPrior(seed=43).sample_batch(8)
    assert not np.array_equal(a.x, c.x, equal_nan=True)


def test_sigma_zero_observation_equals_mechanism():
    prior = SCMPrior(PriorConfig(noise_scale=0.0, **CLEAN), seed=1)
    for _ in range(5):
        batch = prior.sample_batch(8)
        assert np.allclose(batch.x, batch.f, atol=1e-6)
        assert not np.allclose(batch.x, 0)  # roots keep exogenous variation


def test_noise_matches_labels():
    # x - f must equal per-node noise: std ~= sigma, exactly 0 for roots
    batch = SCMPrior(PriorConfig(**CLEAN), seed=2).sample_batch(16)
    for b in range(16):
        for d in range(batch.x.shape[2]):
            node = batch.observed_idx[b, d]
            resid = batch.x[b, :, d] - batch.f[b, :, d]
            sigma = batch.noise_sigma[b, node]
            if batch.func_family[b, node] == FuncFamily.ROOT:
                assert sigma == 0 and np.allclose(resid, 0)
            else:
                assert abs(resid.std() - sigma) < max(0.5 * sigma, 1e-3)


def test_dag_and_labels_consistent():
    batch = SCMPrior(seed=3).sample_batch(16)
    B, N, _ = batch.adjacency.shape
    assert not np.tril(batch.adjacency).any()  # topo order => acyclic
    in_deg = batch.adjacency.sum(1)
    assert ((in_deg == 0) == (batch.func_family == FuncFamily.ROOT)).all()
    assert (in_deg <= PriorConfig().max_parents).all()
    assert ((batch.noise_sigma == 0) == (batch.func_family == FuncFamily.ROOT)).all()
    for b in range(B):  # observed_idx: valid node ids, no duplicates
        assert len(set(batch.observed_idx[b].tolist())) == batch.x.shape[2]
        assert batch.observed_idx[b].max() < N


def test_postprocess():
    prior = SCMPrior(PriorConfig(p_categorize=1.0, p_missing_table=1.0,
                                 missing_rate=(0.05, 0.15)), seed=4)
    batch = prior.sample_batch(8)
    assert (batch.categorical >= 2).all()
    assert np.isnan(batch.x).any()
    assert not np.isnan(batch.f).any()  # truth stays complete & continuous
    vals = batch.x[0][~np.isnan(batch.x[0][:, 0]), 0]
    assert set(np.unique(vals)) <= set(range(batch.categorical[0, 0]))


def test_batch_shape_variety():
    prior = SCMPrior(seed=5)
    shapes = {prior.sample_batch(2).x.shape for _ in range(20)}
    assert len(shapes) > 3  # R and D vary across batches


def test_temporal_dbn():
    # time-unrolled DBN: N = V*(T+1), Markov-1 layered, weight-shared transition
    V, T = 3, 6
    prior = SCMPrior(PriorConfig(temporal=(V, T), n_rows=(128, 128), **CLEAN), seed=7)
    batch = prior.sample_batch(8)
    N = V * (T + 1)
    B, _, D = batch.x.shape
    assert D == N
    slc = lambda n: n // V          # idx(v,t) = t*V + v  ->  slice = n // V
    var = lambda n: n % V
    for b in range(B):
        adj = batch.adjacency[b]
        src, dst = np.nonzero(adj)
        assert np.all(slc(dst) == slc(src) + 1)              # every edge slice t-1 -> t
        esets = [set(zip(var(src[slc(dst) == t]).tolist(),   # transition structure...
                         var(dst[slc(dst) == t]).tolist())) for t in range(1, T + 1)]
        assert all(es == esets[0] for es in esets)           # ...time-invariant (weight-share)
        assert all((v, v) in esets[0] for v in range(V))     # self-edge always present
        for v in range(V):                                   # length-T chain exists (find_path)
            ch = [t * V + v for t in range(T + 1)]
            assert all(adj[ch[i], ch[i + 1]] for i in range(T))
    roots = np.array([slc(n) == 0 for n in range(N)])        # t=0 = roots, t>=1 = mechanism
    assert (batch.func_family[:, roots] == FuncFamily.ROOT).all()
    assert (batch.func_family[:, ~roots] != FuncFamily.ROOT).all()
    assert ((batch.noise_sigma == 0) == (batch.func_family == FuncFamily.ROOT)).all()
    assert np.isfinite(batch.f).all()
    for b in range(B):                                       # instrument: x - f std ~= sigma
        for d in range(D):
            node = batch.observed_idx[b, d]
            resid = batch.x[b, :, d] - batch.f[b, :, d]
            sigma = batch.noise_sigma[b, node]
            if batch.func_family[b, node] == FuncFamily.ROOT:
                assert sigma == 0 and np.allclose(resid, 0)
            else:
                assert abs(resid.std() - sigma) < max(0.5 * sigma, 1e-3)


def test_temporal_squash_bounds_transition():
    # tanh-squash: every transition f bounded (no heavy tails to compound over T);
    # roots (t=0) untouched (f := x). observed_idx maps column -> node -> slice.
    V, T = 3, 6
    b = SCMPrior(PriorConfig(temporal=(V, T), temporal_squash=True,
                             n_rows=(128, 128), **CLEAN), seed=8).sample_batch(8)
    assert np.isfinite(b.f).all()
    is_trans = (b.observed_idx // V) >= 1                 # (B,N) column is a t>=1 node
    for bi in range(8):
        assert np.abs(b.f[bi][:, is_trans[bi]]).max() <= 4.0 + 1e-6   # clipped to +-4


def test_temporal_hidden_partial_obs():
    # PO axis (S15): H whole variables hidden at every t -> D=(V-H)*(T+1) observed columns,
    # hidden variable's columns absent, but it stays in the dynamics (adj) as a parent of an
    # observed variable (the belief-state niche). Full node space preserved for adj/fam/sig.
    V, T, H = 3, 6, 1
    prior = SCMPrior(PriorConfig(temporal=(V, T), temporal_hidden=(H, H),
                                 n_rows=(128, 128), **CLEAN), seed=11)
    batch = prior.sample_batch(8)
    N = V * (T + 1)
    B, _, D = batch.x.shape
    assert D == (V - H) * (T + 1)                          # H variables (all t) dropped
    assert batch.adjacency.shape[1:] == (N, N)             # adj still over ALL nodes
    for b in range(B):
        obs_vars = set((batch.observed_idx[b] % V).tolist())
        assert len(obs_vars) == V - H                      # exactly H variables hidden
        hidden_var = (set(range(V)) - obs_vars).pop()
        # hidden var appears in NO observed column, at ANY slice
        assert all((c % V) != hidden_var for c in batch.observed_idx[b])
        # ...yet it is a transition-parent of an OBSERVED variable (niche): adj[hv, V+w]
        adj = batch.adjacency[b]
        child = [w for w in range(V) if w != hidden_var and adj[hidden_var, V + w]]
        assert child and any(w in obs_vars for w in child)  # confounds an observed var
    # (0,0) must reproduce full observability (D==N), i.e. the T0/T1 path is unchanged
    full = SCMPrior(PriorConfig(temporal=(V, T), n_rows=(64, 64), **CLEAN), seed=11).sample_batch(4)
    assert full.x.shape[2] == N


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all prior self-checks passed")
