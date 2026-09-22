"""Filter a prior dump to its LEARNABLE tables. TabICL's prior is bimodal BY DESIGN -- most
tables are near-noise (only ~17% have GBM-extractable signal at 1024 rows), which is fine for
TabICL's own batch-512 / millions-of-tables regime but drowns the signal for a small model at
batch ~12: the loss is dominated by tables where predicting the context mean is already optimal,
so val_mse sits flat at ~1.0 and the model never learns the 17%.

This keeps only tables whose target is GBM-learnable (holdout R^2 > threshold), scoring the SAME
task the model trains on (features -> target = the y_last column). Densifying the signal makes
both the learning and the val_mse gate legible. Output is the same HDF5 format DumpPrior reads.

RUN:  uv run --with h5py --with xgboost python scripts/filter_dump_learnable.py \
          --in data/tabicl_dump_r1024.h5 --out data/tabicl_dump_r1024_learn.h5 --min_r2 0.1
"""
import argparse
import time

import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor


def _r2(ytr, yte, pred):
    v = ((yte - ytr.mean()) ** 2).mean()
    return -1.0 if v < 1e-8 else 1 - ((yte - pred) ** 2).mean() / v


def _score(Xi, yi, min_r2):
    # some TabICL tables carry NaN targets/features (training masks them; GBM can't fit them):
    # score on the finite rows only, and skip a table with too few to judge.
    fin = np.isfinite(yi) & np.isfinite(Xi).all(1)
    yi, Xi = yi[fin], Xi[fin]
    if len(yi) < 100 or yi.std() < 1e-6:
        return False
    k = int(len(yi) * 0.7)
    g = HistGradientBoostingRegressor(max_iter=150).fit(Xi[:k], yi[:k]).predict(Xi[k:])
    return _r2(yi[:k], yi[k:], g) > min_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_r2", type=float, default=0.1)
    a = ap.parse_args()

    with h5py.File(a.inp, "r") as f:
        X = np.asarray(f["X"], np.float32)          # (N,R,F)
        Y = np.asarray(f["y"], np.float32)          # (N,R)
        nf = np.asarray(f["num_features"])
        attrs = dict(f.attrs)
    t0 = time.time()
    flags = Parallel(n_jobs=-1, verbose=5)(
        delayed(_score)(X[i], Y[i], a.min_r2) for i in range(len(X)))
    keep = np.flatnonzero(np.array(flags))
    print(f"  scored {len(X)} in {time.time()-t0:.0f}s (parallel)", flush=True)
    with h5py.File(a.out, "w") as f:
        f.create_dataset("X", data=X[keep], compression="gzip", compression_opts=1)
        f.create_dataset("y", data=Y[keep], compression="gzip", compression_opts=1)
        f.create_dataset("num_features", data=nf[keep])
        f.attrs.update(attrs)
        f.attrs["filtered_min_r2"] = a.min_r2
        f.attrs["kept"] = len(keep)
    print(f"wrote {a.out}: kept {len(keep)}/{len(X)} learnable tables "
          f"(GBM holdout R^2 > {a.min_r2})", flush=True)


if __name__ == "__main__":
    main()
