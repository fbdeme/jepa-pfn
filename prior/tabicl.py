"""TabICL prior adapter (`prior_source: tabicl`, docs/prior_v2_plan.md, 2026-09-10).

Wraps the vendored `third_party/tabicl_prior` generator (soda-inria/tabicl, BSD-3, commit in VENDORED.md) as an
SCMBatch producer so the two trainers, make_batch and the evaluation stack see it exactly like our own priors.

    prior_type  mix_scm    TabICL v1 engine: MLP-SCM 70 % + tree(XGBoost)-SCM 30 %, per-node Gaussian noise
                graph_scm  TabICLv2 engine: Cauchy DAG, 8 function families (mlp,tree,disc,lin,quad,gp,em,prod),
                           7 categorical converters, Kumaraswamy warping, no node noise (paper Appx E)
    task        cls        every table gets a K-class NOMINAL label column (K ~ U[2, max_classes], codes permuted upstream)
                reg        continuous target column
                mixed      cls with probability p_cls, else reg (per batch)

What this prior is NOT: instrumented. It returns no mechanism truth (f = x, empty adjacency, family -1), so the
instrument track (truth probes, C4 boundary) keeps the SCM / real priors; this one serves the transfer track.

Generation cost: ~0.3 s per batch of 4 tables (1024 rows, <= 64 features) on one CPU core, about one GPU step of the
base model, so with prefetch > 0 the vendor runs in a child PROCESS (spawn) that fills a queue ahead of the trainer.
Determinism: the vendor draws from numpy's / torch's GLOBAL RNGs, seeded once from `seed` in whichever process runs
it, so the table stream is a fixed function of (config, seed, batch size) in both modes and identical between them
(prior/test_tabicl.py). Our own draws (rows, task coin, missingness, column order) use `self.rng` like every prior.
Resume (`resume.pt`) restarts the vendor stream from its seed, so a resumed run is not bitwise-equal to an
uninterrupted one on this prior (documented limitation; the SCM prior is bitwise, train/test_resume.py).
"""
import multiprocessing as mp
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from prior.scm import SCMBatch

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class TabICLConfig:
    prior_type: str = "graph_scm"       # mix_scm (v1) | graph_scm (v2)
    task: str = "cls"                   # cls | reg | mixed
    p_cls: float = 0.7                  # mixed only
    min_features: int = 4               # feature columns, the label column comes on top (D = features + 1)
    max_features: int = 64              # census p85 = 64 (docs/real_prior_plan.md); eval crops to K_FEAT
    max_classes: int = 10               # TabICL / TabPFN v2 native limit
    n_rows: tuple = (128, 1024)         # context-size range, log-uniform per batch
    max_cells: int = 32768              # cell budget (rows dropped, not features): the 48 GB peak = 32 cols x 1024 rows
    p_missing_table: float = 0.3        # MCAR on features only (the vendor generates no NaN; real suites have them)
    missing_rate: tuple = (0.0, 0.1)
    filter_unpredictable: bool = False  # TabICLv2 ExtraTrees dataset filter (paper: ~35 % cls / 25 % reg rejected); off = vendor default
    filter_graphs: bool = False         # TabICLv2 graph filter (x nodes must share ancestors with y). The v2 recipe
                                        # (vendor scripts/train_v2_*_stage*.sh) turns BOTH filters on: that is "the prior as-is"
    remove_trivial: bool = False
    max_cat_detect: int = 16            # a feature column with <= this many integer-valued uniques is marked categorical
    prefetch: int = 8                   # > 0: vendor generation in child process(es) with this many batches queued ahead
    workers: int = 1                    # producer processes (seed + 1000 k each), consumed round-robin; 1 = same stream as prefetch 0.
                                        # ladder size generates ~9 tables/s per core vs ~12 needed at 3 step/s -> use 2 on the box


def _seed_globals(seed):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)


def _make_datasets(c, seed):
    from third_party.tabicl_prior import PriorDataset
    from third_party.tabicl_prior.graph_lib._config import PriorConfig as GraphConfig
    _seed_globals(seed)
    gcfg = GraphConfig(filter_unpredictable_datasets=c.filter_unpredictable, filter_unpredictable_graphs=c.filter_graphs,
                       remove_trivial_datasets=c.remove_trivial)
    mk = lambda reg: PriorDataset(regression=reg, batch_size=4, batch_size_per_gp=4, min_features=c.min_features,
                                  max_features=c.max_features, max_classes=c.max_classes, min_seq_len=None,
                                  max_seq_len=c.n_rows[1], prior_type=c.prior_type, config=gcfg,
                                  n_jobs=1, num_threads_per_generate=1, device="cpu")
    return {"cls": mk(False) if c.task != "reg" else None, "reg": mk(True) if c.task != "cls" else None}


def _raw_batch(ds, batch_size):
    while True:
        try:
            X, y, d, seq, _ = ds.get_batch(batch_size)               # X (B, R_v, max_features) zero-padded beyond d; y (B, R_v)
            break
        except ValueError as e:   # v1 mix_scm can hand the v2 ExtraTrees filter a NaN target ("Input y contains NaN",
            print(f"tabicl prior: vendor get_batch failed ({e}); drawing the next batch", file=sys.stderr, flush=True)
            # abl-prior-tabicl1 2026-09-10, worker died -> trainer hung on its queue): skip that draw, keep the stream deterministic
    return X.numpy(), y.numpy(), d.numpy().ravel(), seq.numpy().ravel()


def _producer(cfg_dict, seed, batch_size, tasks, q):
    """Child process: same seeding, same vendor calls, same order of task draws as the in-process path."""
    torch.set_num_threads(1)
    ds = _make_datasets(TabICLConfig(**cfg_dict), seed)
    for task in iter(tasks.get, None):
        q.put(_raw_batch(ds[task], batch_size))


class TabICLPrior:
    def __init__(self, config: TabICLConfig = TabICLConfig(), seed: int = 0):
        self.config, self.seed = config, seed
        self.rng = np.random.default_rng(seed)                       # rows / missingness / column order (per batch, both modes)
        self.rng_task = np.random.default_rng(seed + 1)              # the cls/reg coin: its own stream, so prefetching coins ahead
        self._ds = None if config.prefetch > 0 else _make_datasets(config, seed)   # leaves the per-batch stream untouched
        self._proc = []

    def _start(self, batch_size):
        ctx = mp.get_context("spawn")                                # fork after CUDA init is unsafe; spawn re-imports this module
        W = max(1, self.config.workers)
        self._batch_size, self._turn = batch_size, 0
        self._tasks, self._qs, self._fifos, self._proc = [], [], [], []
        for k in range(W):
            tq, q = ctx.Queue(), ctx.Queue(maxsize=self.config.prefetch)
            p = ctx.Process(target=_producer, args=(asdict(self.config), self.seed + 1000 * k, batch_size, tq, q), daemon=True)
            p.start()
            self._tasks.append(tq); self._qs.append(q); self._fifos.append([]); self._proc.append(p)
        for _ in range(self.config.prefetch * W):                    # announce the queue depth ahead; children produce in this order
            self._announce()

    def _announce(self):
        k = self._turn % len(self._proc); self._turn += 1
        t = self._task(); self._tasks[k].put(t); self._fifos[k].append(t)

    def _task(self):
        c = self.config
        if c.task == "mixed":
            return "cls" if self.rng_task.random() < c.p_cls else "reg"
        return c.task

    def _next_raw(self, batch_size):
        if self.config.prefetch <= 0:
            task = self._task()
            return task, _raw_batch(self._ds[task], batch_size)
        if not self._proc:
            self._start(batch_size)
        assert batch_size == self._batch_size, "prefetch mode generates one fixed batch size"   # ponytail: val uses the train size
        k = self._take % len(self._proc) if hasattr(self, "_take") else 0
        self._take = k + 1
        self._announce()
        return self._fifos[k].pop(0), self._qs[k].get()              # round-robin over workers; per worker FIFO = announce order

    def sample_batch(self, batch_size: int) -> SCMBatch:
        c, rng = self.config, self.rng
        task, (X, y, d, seq) = self._next_raw(batch_size)
        while d.min() < 1 or np.isnan(y).any():                     # min_features 1 + the vendor's near-constant pruning can leave
            task, (X, y, d, seq) = self._next_raw(batch_size)       # a table with no feature, and mix_scm can leave a NaN target
                                                                    # (filters off would pass it through): take the next batch instead
        n_classes_batch = task == "cls"
        # the vendor samples its sequence length itself (<= n_rows[1]); we draw ours and keep min(ours, theirs) rows
        R_want = int(np.exp(rng.uniform(np.log(c.n_rows[0]), np.log(c.n_rows[1]))))
        n_feat = int(d.min())                                       # one shared D per batch (SCMBatch contract): crop to the narrowest
        D = n_feat + 1
        R = int(min(R_want, seq.min(), max(c.max_cells // D, 10)))
        x = np.empty((batch_size, R, D), np.float32)
        categorical = np.zeros((batch_size, D), np.int8)
        hyper = []
        for b in range(batch_size):
            rows = rng.permutation(int(seq[b]))[:R]                 # rows are iid: a random subset keeps the table's distribution
            feats = X[b][rows][:, :n_feat]
            lab = y[b][rows].astype(np.float32)
            n_classes = int(lab.max()) + 1 if n_classes_batch else 0
            order = rng.permutation(n_feat)                         # column order carries nothing
            feats = feats[:, order]
            if rng.random() < c.p_missing_table:                    # MCAR on features, never on the label
                rate = rng.uniform(*c.missing_rate)
                feats[rng.random(feats.shape) < rate] = np.nan
            target_col = int(rng.integers(D))                       # label column position is random too
            x[b] = np.insert(feats, target_col, lab, axis=1)
            categorical[b] = self._detect_categorical(x[b], target_col, n_classes)
            hyper.append({"target_col": target_col, "n_classes": n_classes, "task": task, "D": D, "R": R,
                          "prior_type": c.prior_type, "n_feat_generated": int(d[b]),
                          "cat_frac": round(float((np.delete(categorical[b], target_col) > 0).mean()), 4)})
        f = x.copy()                                                # no mechanism truth: f == x
        adjacency = np.zeros((batch_size, D, D), bool)
        observed_idx = np.tile(np.arange(D, dtype=np.int16), (batch_size, 1))
        func_family = np.full((batch_size, D), -1, np.int8)
        noise_sigma = np.zeros((batch_size, D), np.float32)
        return SCMBatch(x, f, adjacency, observed_idx, func_family, noise_sigma, categorical, hyper)

    def _detect_categorical(self, xb, target_col, n_classes):
        c = self.config
        cat = np.zeros(xb.shape[1], np.int8)
        for j in range(xb.shape[1]):
            if j == target_col:
                cat[j] = n_classes                                  # 0 for a regression target
                continue
            v = xb[:, j][~np.isnan(xb[:, j])]
            u = np.unique(v)
            if 2 <= len(u) <= c.max_cat_detect and np.all(np.abs(u - np.round(u)) < 1e-6):
                cat[j] = len(u)
        return cat

    def close(self):
        for tq, p in zip(self._tasks if self._proc else [], self._proc):
            tq.put(None); p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._proc = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
