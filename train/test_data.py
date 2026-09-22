"""C5: the row_block_partial masking policy - per query row 50-80% of the cells hidden, never all, context untouched."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from prior.scm import PriorConfig, SCMPrior     # noqa: E402
from train.data import make_batch               # noqa: E402


def test_row_block_partial_fraction_and_never_whole_row():
    rng = np.random.default_rng(0)
    for D in (3, 4, 8):
        prior = SCMPrior(PriorConfig(n_rows=(40, 40), n_cols=(D, D), p_missing_table=0.0), seed=D)
        for _ in range(5):
            bt = make_batch(prior, 4, "row_block_partial", rng, split=20)
            tm = bt["target_mask"].numpy()
            assert not tm[:, :20].any()                                   # context rows are never targets
            per_row = tm[:, 20:].sum(-1)                                  # (B, R-split) hidden cells per query row
            assert (per_row >= 1).all() and (per_row <= D - 1).all(), per_row
            k = per_row[:, 0][:, None]
            assert (per_row == k).all()                                   # one block size per table
            assert (round(0.5 * D) - 1 <= k).all() and (k <= round(0.8 * D)).all(), (D, k.ravel())
            assert (bt["input_mask"].numpy() == tm).all()                 # no missingness: input mask = target mask


def test_row_block_partial_is_seeded():
    prior = SCMPrior(PriorConfig(n_rows=(30, 30), n_cols=(5, 5)), seed=1)
    a = make_batch(prior, 2, "row_block_partial", np.random.default_rng(7), split=15)["target_mask"]
    prior = SCMPrior(PriorConfig(n_rows=(30, 30), n_cols=(5, 5)), seed=1)
    b = make_batch(prior, 2, "row_block_partial", np.random.default_rng(7), split=15)["target_mask"]
    assert (a == b).all()
