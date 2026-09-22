"""Pre-generate a TabICL-prior dump into the HDF5 format DumpPrior already reads, so our loop
trains on TabICL's OPEN SCM prior (soda-inria/tabicl) with NO change to train.data / DumpPrior.

Why pre-dump: TabICL generates on CPU serially (n_jobs=1 forced -- HpSampler closures don't
pickle), so on-the-fly generation is a training bottleneck. We draw a fixed pool of tables once
and DumpPrior resamples from it across steps (same pattern as the nanoTabPFN dump).

Fixed width F (min_features=max_features=F) => rectangular tables => DumpPrior(feat=F) serves
x = [F features, target] = (B, R, F+1) with the target as the last column, consumed by the
`y_last` policy (TabICL features are not mutually predictable, so predict the target column).

SETUP (vendor/tabicl is the gitignored throwaway clone; deps pulled per-run by uv):
    git clone --depth 1 https://github.com/soda-inria/tabicl vendor/tabicl
    echo '# neutered for prior-only use' > vendor/tabicl/src/tabicl/__init__.py
RUN:
    uv run --with psutil --with einops --with xgboost --with networkx --with h5py \
        python scripts/gen_tabicl_dump.py --out data/tabicl_dump.h5 \
        --n_tables 8000 --rows 1024 --features 10 --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "vendor/tabicl/src"))
from tabicl.prior import PriorDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tables", type=int, default=8000)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--prior_type", default="mlp_scm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # min_seq_len < max_seq_len is required (randint(low, high)); [rows, rows+1) => exactly rows.
    ds = PriorDataset(regression=True, batch_size=8, min_features=a.features,
                      max_features=a.features, max_seq_len=a.rows + 1, min_seq_len=a.rows,
                      log_seq_len=False, prior_type=a.prior_type, device="cpu", n_jobs=1)
    # PriorDataset seeds off its own RNG; vary the first draw by seed so dumps differ per seed.
    for _ in range(a.seed):
        ds.get_batch(8)

    X = np.empty((a.n_tables, a.rows, a.features), np.float32)
    Y = np.empty((a.n_tables, a.rows), np.float32)
    n, t0 = 0, time.time()
    while n < a.n_tables:
        xb, yb, d, sl, _ = ds.get_batch(8)
        xb, yb = xb.cpu().numpy(), yb.cpu().numpy()
        d, sl = d.cpu().numpy().ravel(), sl.cpu().numpy().ravel()
        for i in range(xb.shape[0]):
            if n >= a.n_tables:
                break
            if sl[i] < a.rows or d[i] < a.features:      # skip any short/narrow draw
                continue
            X[n] = xb[i, :a.rows, :a.features]
            Y[n] = yb[i, :a.rows]
            n += 1
        if n % 400 < 8:
            print(f"  {n}/{a.n_tables}  ({(time.time()-t0):.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=1)
        f.create_dataset("y", data=Y, compression="gzip", compression_opts=1)
        f.create_dataset("num_features", data=np.full(a.n_tables, a.features, np.int32))
        f.attrs.update(prior_type=a.prior_type, rows=a.rows, features=a.features, seed=a.seed)
    print(f"wrote {a.out}: {a.n_tables} tables x {a.rows} rows x {a.features} feat "
          f"in {(time.time()-t0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
