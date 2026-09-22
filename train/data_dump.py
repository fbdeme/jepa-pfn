"""Loop-vs-prior diagnostic: serve the nanoTabPFN dump (real TabPFN-v2 prior samples) through
OUR pipeline. DumpPrior mimics SCMPrior.sample_batch's `.x` contract so train.data.make_batch
and the training loop run UNCHANGED -- only the data source (the prior) is swapped. Each table
is (X features + the label as the last column); the label is just another cell for any_cell.

Truth fields (noise-free f, DAG, family) do NOT exist for this prior -- TabPFN's entangled
computational-graph SCM has no clean per-node mechanism to read off -- so return_truth is
unsupported and the mechanism probe cannot run here. This is a stack-vs-prior diagnostic only.
"""
import h5py
import numpy as np


class _Batch:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


class DumpPrior:
    """Batches of tables from an HDF5 prior dump; `.x` = (B, R, feat+1) with the label as the
    last column. Restricted to tables of a fixed feature width (`feat`) so every batch is
    rectangular with no padding (this dump is 99.8% 5-feature). split='train' draws from the
    first 90% of the kept tables, 'val' from the last 10% (disjoint, fixed regardless of seed)."""

    def __init__(self, h5_path, seed=0, split="train", val_frac=0.1, feat=5):
        self.rng = np.random.default_rng(seed)
        self.feat = feat
        with h5py.File(h5_path, "r") as f:
            self.X = np.asarray(f["X"], np.float32)           # (N, R, Fmax)
            self.y = np.asarray(f["y"], np.float32)           # (N, R) class labels
            nf = np.asarray(f["num_features"]).astype(int)
        keep = np.flatnonzero(nf >= feat)                     # uniform width; no ragged batches
        keep = keep[np.random.default_rng(12345).permutation(len(keep))]  # split indep. of seed
        n_val = int(len(keep) * val_frac)
        self.idx = keep[n_val:] if split == "train" else keep[:n_val]

    def sample_batch(self, batch_size):
        pick = self.idx[self.rng.integers(len(self.idx), size=batch_size)]
        F = self.feat
        x = np.concatenate([self.X[pick, :, :F], self.y[pick, :, None]], -1)  # label = last col
        return _Batch(x.astype(np.float32))
