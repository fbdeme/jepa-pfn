"""Instrumented generic SCM prior (SSOT section 4, Phase 1).

Samples random SCMs and executes them into tables. Unlike TabPFN's prior,
sample_batch() also returns the ground truth we own as simulation gods:
noise-free mechanism values, the true DAG, and per-node labels. Truth is for
EVALUATION ONLY (C4 noise alignment, C5 mechanism probing) - never feed it
into a training loss.

Conventions (see docs/history.md for rationale):
- f_ij = mechanism value of node j given its ACTUAL (noisy) parent values, so
  x_ij = f_ij + sigma_j * eps exactly per node. Root nodes have no mechanism:
  f := x (neutral in C4 contrasts, filter with func_family == ROOT if needed).
- Each non-root node's mechanism output is standardized across rows before
  noise is added, so sigma is a noise-to-signal ratio comparable across nodes
  (prerequisite for the Phase 4 sigma-sweep figure).
- Post-processing (categorization, missingness) touches x only; f stays the
  continuous pre-noise value.
- Node index = topological order; adjacency is strictly upper-triangular.
  Observed column order is shuffled so column position leaks no topology.
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class FuncFamily(IntEnum):
    ROOT = 0
    MLP = 1
    TREE = 2
    DISC = 3
    LINEAR = 4   # prior/real.py only: a linear view of its parents (real tables are largely redundant
                 # near-linear views of shared factors); never sampled by SCMPrior itself


@dataclass(frozen=True)
class PriorConfig:
    n_rows: tuple = (64, 150)        # rows per table, sampled per batch
    n_cols: tuple = (3, 8)           # observed columns D, sampled per batch
    n_hidden: tuple = (0, 3)         # hidden (confounder) nodes, per batch
    edge_prob: tuple = (0.15, 0.5)   # per-graph edge probability (sparsity bias)
    p_connect: float = 0.7           # orphaned node gets 1 random parent with
                                     # this prob (small graphs drown in roots otherwise)
    max_parents: int = 3             # keeps tree/disc mechanisms sensible
    p_family: tuple = (0.4, 0.3, 0.3)  # MLP / TREE / DISC for non-root nodes
    mlp_depth: tuple = (1, 1)        # hidden layers per MLP mechanism. (1,1)=paper prior
                                     # (single layer, ~linear); deeper => genuine nonlinearity.
    mlp_gain: float = 1.0            # weight-scale; >1 pushes activations off the ~linear
                                     # regime near 0 into saturation (the "real SCM" knob).
    sigma: tuple = (0.01, 0.3)       # per-node noise, log-uniform (non-root)
    noise_scale: float = 1.0         # global multiplier on sigma (Phase 4 knob)
    p_categorize: float = 0.2        # per observed column
    n_categories: tuple = (2, 5)
    p_missing_table: float = 0.3     # prob. a table has MCAR missingness at all
    missing_rate: tuple = (0.0, 0.1)
    temporal: tuple = None           # None = structural (default). (V, T) = time-unrolled
                                     # DBN: V state vars over T+1 slices, weight-shared
                                     # transition. n_cols/n_hidden ignored (temporal_hidden below).
    temporal_squash: bool = False    # tanh-bound each transition output so heavy tails
                                     # can't compound over T (T0 blow-up fix); T1 uses True
    temporal_hidden: tuple = (0, 0)  # #23 PO axis: hide H whole variables (all t) from the
                                     # observed table but keep them in the dynamics => a
                                     # persistent hidden state (POMDP). (0,0)=full obs (T0/T1).


@dataclass
class SCMBatch:
    """One batch of tables sharing (R, D, N); each table has its own SCM.

    Training may only see x. Everything else is evaluation truth.
    """
    x: np.ndarray             # (B,R,D) float32 observed values, NaN = missing
    f: np.ndarray             # (B,R,D) float32 mechanism values (pre-noise)
    adjacency: np.ndarray     # (B,N,N) bool, [b,i,j] True means node i -> node j
    observed_idx: np.ndarray  # (B,D) int16, graph node backing each column
    func_family: np.ndarray   # (B,N) int8, FuncFamily per node
    noise_sigma: np.ndarray   # (B,N) float32 effective sigma (0 for roots)
    categorical: np.ndarray   # (B,D) int8, 0 = continuous, else category count
    hyper: list = None        # per-table generator hyperparameters (prior/real.py only); truth, eval-only


def _rand_mlp(rng, n_parents, depth=1, gain=1.0):
    if depth == 1 and gain == 1.0:  # paper prior: single ~linear layer (keep byte-for-byte)
        width = int(rng.integers(4, 17))
        w1 = rng.normal(0, 1 / np.sqrt(n_parents), (n_parents, width))
        b1 = rng.normal(0, 0.5, width)
        w2 = rng.normal(0, 1 / np.sqrt(width), (width, 1))
        act = rng.choice([np.tanh, lambda z: np.maximum(z, 0), np.sin])
        return lambda P: (act(P @ w1 + b1) @ w2)[:, 0]
    # real prior: deeper stack + higher gain => activations saturate => genuine nonlinearity
    dims = [n_parents] + [int(rng.integers(8, 33)) for _ in range(depth)] + [1]
    Ws = [rng.normal(0, gain / np.sqrt(dims[i]), (dims[i], dims[i + 1]))
          for i in range(len(dims) - 1)]
    bs = [rng.normal(0, 0.5, dims[i + 1]) for i in range(len(dims) - 1)]
    acts = [rng.choice([np.tanh, lambda z: np.maximum(z, 0), np.sin]) for _ in range(depth)]

    def f(P):
        h = P
        for i in range(depth):          # hidden layers get an activation, output stays linear
            h = acts[i](h @ Ws[i] + bs[i])
        return (h @ Ws[-1] + bs[-1])[:, 0]
    return f


def _rand_tree(rng, n_parents, depth=3):
    # top level always splits: a bare-leaf tree is a constant "mechanism",
    # which would label a node TREE while carrying no functional dependence
    if depth < 3 and (depth == 0 or rng.random() < 0.3):
        leaf = rng.normal()
        return lambda P: np.full(len(P), leaf)
    dim = int(rng.integers(n_parents))
    thr = rng.normal()  # parents are ~standardized, N(0,1) thresholds cover them
    left = _rand_tree(rng, n_parents, depth - 1)
    right = _rand_tree(rng, n_parents, depth - 1)

    def f(P):
        mask = P[:, dim] <= thr
        out = np.empty(len(P))
        out[mask] = left(P[mask]) if mask.any() else 0.0
        out[~mask] = right(P[~mask]) if (~mask).any() else 0.0
        return out
    return f


def _rand_disc(rng, n_parents):
    w = rng.normal(size=n_parents)
    k = int(rng.integers(3, 9))
    edges = np.sort(rng.normal(size=k - 1))  # on standardized projection
    values = rng.normal(size=k)

    def f(P):
        s = P @ w
        std = s.std()
        s = (s - s.mean()) / (std + 1e-8)
        return values[np.digitize(s, edges)]
    return f


_NOISE = {  # unit-variance samplers
    "normal": lambda rng, n: rng.normal(size=n),
    "laplace": lambda rng, n: rng.laplace(scale=1 / np.sqrt(2), size=n),
    "uniform": lambda rng, n: rng.uniform(-np.sqrt(3), np.sqrt(3), size=n),
}
_FAMILY_FN = {FuncFamily.MLP: _rand_mlp, FuncFamily.TREE: _rand_tree,
              FuncFamily.DISC: _rand_disc}


def _make_mechanism(rng, family, n_parents, c):
    """Build one node mechanism. MLP honors the prior's depth/gain (the real-SCM knobs);
    default (1,1)/1.0 falls through to the byte-for-byte paper mechanisms."""
    if family == FuncFamily.MLP and (tuple(c.mlp_depth) != (1, 1) or c.mlp_gain != 1.0):
        depth = int(rng.integers(c.mlp_depth[0], c.mlp_depth[1] + 1))
        return _rand_mlp(rng, n_parents, depth, c.mlp_gain)
    return _FAMILY_FN[family](rng, n_parents)


class SCMPrior:
    """Infinite stream of instrumented synthetic tables. Prior = this code."""

    def __init__(self, config: PriorConfig = PriorConfig(), seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int) -> SCMBatch:
        if self.config.temporal is not None:
            return self._sample_temporal_batch(batch_size)
        c, rng = self.config, self.rng
        R = int(rng.integers(c.n_rows[0], c.n_rows[1] + 1))
        D = int(rng.integers(c.n_cols[0], c.n_cols[1] + 1))
        H = int(rng.integers(c.n_hidden[0], c.n_hidden[1] + 1))
        N = D + H

        x = np.empty((batch_size, R, D), np.float32)
        f = np.empty((batch_size, R, D), np.float32)
        adjacency = np.zeros((batch_size, N, N), bool)
        observed_idx = np.empty((batch_size, D), np.int16)
        func_family = np.zeros((batch_size, N), np.int8)
        noise_sigma = np.zeros((batch_size, N), np.float32)
        categorical = np.zeros((batch_size, D), np.int8)

        # ponytail: python loop over tables/nodes (~1ms/table); vectorize across
        # the batch only if prior throughput ever bottlenecks GPU feeding
        for b in range(batch_size):
            xb, fb, adj, fam, sig = self._sample_table(R, N)
            hidden = rng.choice(N, size=H, replace=False) if H else np.array([], int)
            obs = np.setdiff1d(np.arange(N), hidden)
            rng.shuffle(obs)  # column order must not leak topology
            x[b], f[b] = xb[:, obs], fb[:, obs]
            adjacency[b], func_family[b], noise_sigma[b] = adj, fam, sig
            observed_idx[b] = obs
            categorical[b] = self._postprocess(x[b], rng)

        return SCMBatch(x, f, adjacency, observed_idx, func_family,
                        noise_sigma, categorical)

    def _sample_table(self, R, N):
        c, rng = self.config, self.rng
        p_edge = rng.uniform(*c.edge_prob)
        adj = np.zeros((N, N), bool)
        for j in range(1, N):
            parents = np.flatnonzero(rng.random(j) < p_edge)
            if len(parents) == 0 and rng.random() < c.p_connect:
                parents = rng.integers(j, size=1)
            if len(parents) > c.max_parents:
                parents = rng.choice(parents, c.max_parents, replace=False)
            adj[parents, j] = True

        xn = np.empty((R, N))
        fn = np.empty((R, N))
        fam = np.zeros(N, np.int8)
        sig = np.zeros(N, np.float32)
        for j in range(N):
            parents = np.flatnonzero(adj[:, j])
            if len(parents) == 0:
                dist = rng.choice(list(_NOISE))
                xn[:, j] = fn[:, j] = _NOISE[dist](rng, R)  # root: f := x
                continue
            family = FuncFamily(rng.choice([1, 2, 3], p=c.p_family))
            fam[j] = family
            for _ in range(3):  # resample degenerate (constant) mechanisms
                raw = _make_mechanism(rng, family, len(parents), c)(xn[:, parents])
                if raw.std() > 1e-6:
                    break
            std = raw.std()
            fj = (raw - raw.mean()) / (std + 1e-8) if std > 1e-8 else raw - raw.mean()
            s = np.exp(rng.uniform(np.log(c.sigma[0]), np.log(c.sigma[1])))
            s *= c.noise_scale
            sig[j] = s
            fn[:, j] = fj
            xn[:, j] = fj + s * _NOISE[rng.choice(list(_NOISE))](rng, R)
        return xn, fn, adj, fam, sig

    def _sample_temporal_batch(self, batch_size: int) -> SCMBatch:
        """Time-unrolled DBN batch. N = V*(T+1) fully-observed nodes; each table is
        one dynamical system. Same SCMBatch instrument as the structural path, so
        make_batch / rollout_sim reuse unchanged (fam/sig/adj in node order,
        observed_idx maps column -> node)."""
        c, rng = self.config, self.rng
        V, T = c.temporal
        N = V * (T + 1)
        H = min(int(rng.integers(c.temporal_hidden[0], c.temporal_hidden[1] + 1)), V - 1)
        D = (V - H) * (T + 1)           # observed columns (H whole variables hidden)
        R = int(rng.integers(c.n_rows[0], c.n_rows[1] + 1))
        x = np.empty((batch_size, R, D), np.float32)
        f = np.empty((batch_size, R, D), np.float32)
        adjacency = np.zeros((batch_size, N, N), bool)   # full node space (hidden kept)
        observed_idx = np.empty((batch_size, D), np.int16)
        func_family = np.zeros((batch_size, N), np.int8)
        noise_sigma = np.zeros((batch_size, N), np.float32)
        categorical = np.zeros((batch_size, D), np.int8)  # ponytail: continuous state;
        # add per-VARIABLE (not per-column) categorization if realism ever needed
        for b in range(batch_size):
            for _ in range(50):         # resample until the hidden state actually confounds an
                xb, fb, adj, fam, sig = self._sample_temporal_table(R)   # observed var (S15.2
                hv = self._pick_hidden_vars(adj, V, H, rng)              # pilot: niche guaranteed)
                if H == 0 or self._hidden_ok(adj, V, hv):
                    break
            obs = np.array([col for col in range(N) if (col % V) not in hv], np.int16)
            rng.shuffle(obs)            # column position must not leak time/topology
            x[b], f[b] = xb[:, obs], fb[:, obs]
            adjacency[b], func_family[b], noise_sigma[b] = adj, fam, sig
            observed_idx[b] = obs
            if c.p_missing_table and rng.random() < c.p_missing_table:
                rate = rng.uniform(*c.missing_rate)
                x[b, rng.random((R, D)) < rate] = np.nan
        return SCMBatch(x, f, adjacency, observed_idx, func_family,
                        noise_sigma, categorical)

    @staticmethod
    def _pick_hidden_vars(adj, V, H, rng):
        """Choose H hidden variables, preferring ones with a non-self child so the hidden
        state actually confounds observations (the belief-state niche, S15). Edges are
        variable-invariant (weight-shared), read off the t=0->1 slice: adj[v, V+w] = v is a
        transition-parent of w. Validity (child is OBSERVED) is enforced by resample in the
        caller via _hidden_ok."""
        if H == 0:
            return set()
        cands = [v for v in range(V) if any(adj[v, V + w] for w in range(V) if w != v)]
        pool = cands if len(cands) >= H else list(range(V))
        return set(rng.choice(pool, H, replace=False).tolist())

    @staticmethod
    def _hidden_ok(adj, V, hv):
        """Every hidden var confounds an OBSERVED var: has a non-self child outside hv."""
        return all(any(w != v and w not in hv and adj[v, V + w] for w in range(V)) for v in hv)

    def _sample_temporal_table(self, R):
        """One dynamical system. Node index idx(v,t) = t*V + v. t=0 = initial state
        (roots, f := x). t>=1 = transition: node (v,t)'s parents live in slice t-1
        (Markov-1) and its mechanism g_v is sampled ONCE and applied at every t
        (weight-shared = what makes it a dynamical system, not a deep feedforward SCM).
        Per-(v,t) standardization keeps each slice ~unit variance so mechanism inputs
        stay in the family samplers' regime and sigma is a comparable noise ratio."""
        c, rng = self.config, self.rng
        V, T = c.temporal
        N = V * (T + 1)
        idx = lambda v, t: t * V + v

        xn = np.empty((R, N))
        fn = np.empty((R, N))
        fam = np.zeros(N, np.int8)
        sig = np.zeros(N, np.float32)
        adj = np.zeros((N, N), bool)

        for v in range(V):              # t=0: initial state = roots (f := x)
            col = idx(v, 0)
            xn[:, col] = fn[:, col] = _NOISE[rng.choice(list(_NOISE))](rng, R)

        p_edge = rng.uniform(*c.edge_prob)
        g = {}                          # per-variable time-invariant transition
        for v in range(V):
            pa = set(np.flatnonzero(rng.random(V) < p_edge).tolist()) | {v}  # self-edge always in
            if len(pa) > c.max_parents:
                pa = set(rng.choice(sorted(pa), c.max_parents, replace=False).tolist())
            pa = sorted(pa)
            family = FuncFamily(rng.choice([1, 2, 3], p=c.p_family))
            probe = xn[:, [idx(p, 0) for p in pa]]
            for _ in range(3):          # resample degenerate (constant) mechanism
                fn_v = _make_mechanism(rng, family, len(pa), c)
                if fn_v(probe).std() > 1e-6:
                    break
            s = np.exp(rng.uniform(np.log(c.sigma[0]), np.log(c.sigma[1]))) * c.noise_scale
            g[v] = (pa, family, fn_v, s)

        for t in range(1, T + 1):       # apply the SAME g_v at every step
            for v in range(V):
                pa, family, fn_v, s = g[v]
                col = idx(v, t)
                pa_cols = [idx(p, t - 1) for p in pa]
                adj[pa_cols, col] = True
                fam[col], sig[col] = family, s
                raw = fn_v(xn[:, pa_cols])
                std = raw.std()
                fj = (raw - raw.mean()) / (std + 1e-8) if std > 1e-8 else raw - raw.mean()
                if c.temporal_squash:      # clip standardized output to +-4: only the
                    fj = np.clip(fj, -4.0, 4.0)  # pathological tail is cut (bulk & sigma
                                           # kept), so f can't compound over T (T0 blow-up)
                fn[:, col] = fj
                xn[:, col] = fj + s * _NOISE[rng.choice(list(_NOISE))](rng, R)
        return xn, fn, adj, fam, sig

    def _postprocess(self, xb, rng):
        """Categorize some columns (in place, x only) and apply MCAR NaNs."""
        c = self.config
        R, D = xb.shape
        cat = np.zeros(D, np.int8)
        for d in range(D):
            if rng.random() < c.p_categorize:
                k = int(rng.integers(c.n_categories[0], c.n_categories[1] + 1))
                # random (non-equal) quantile edges -> naturally imbalanced bins
                edges = np.quantile(xb[:, d], np.sort(rng.uniform(size=k - 1)))
                xb[:, d] = np.digitize(xb[:, d], edges)
                cat[d] = k
        if rng.random() < c.p_missing_table:
            rate = rng.uniform(*c.missing_rate)
            xb[rng.random((R, D)) < rate] = np.nan
        return cat
