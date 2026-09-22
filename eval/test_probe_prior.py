"""C3 / C4: the cfg-based probe prior and the truth-cell selector."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import N_QUERY, probe_prior, truth_cells      # noqa: E402
from eval.truth_eval import CLEAN                                     # noqa: E402
from prior.real import RealConfig, RealPrior                          # noqa: E402
from prior.scm import FuncFamily, PriorConfig, SCMPrior               # noqa: E402
from train.data import make_batch                                     # noqa: E402


def test_probe_prior_paper_is_the_pre_c3_construction():
    ctx, seed = 10, 3
    p = probe_prior({}, ctx, seed, noise_scale=2.0, **CLEAN)
    q = SCMPrior(PriorConfig(n_rows=(ctx + N_QUERY, ctx + N_QUERY), noise_scale=2.0, **CLEAN), seed=seed)
    a, b = p.sample_batch(2), q.sample_batch(2)
    assert isinstance(p, SCMPrior) and a.x.shape == b.x.shape and np.array_equal(a.x, b.x, equal_nan=True)
    assert a.x.shape[1] == ctx + N_QUERY


def test_probe_prior_real_pins_rows_and_drops_paper_knobs():
    ctx, seed = 10, 3
    p = probe_prior({"prior_source": "real", "prior": {}}, ctx, seed, noise_scale=2.0, **CLEAN)
    assert isinstance(p, RealPrior)
    c = next(v for v in vars(p).values() if isinstance(v, RealConfig))
    assert c.n_rows == (ctx + N_QUERY, ctx + N_QUERY)
    b = p.sample_batch(2)
    n = ctx + N_QUERY
    assert b.x.shape[1] in (n - 1, n)          # log-uniform draw truncates: exp(log n) can land 1 below
    assert hasattr(b, "adjacency") and b.hyper and "K" in b.hyper[0]


def test_probe_prior_run_cfg_overrides_survive():
    """A run cfg's own prior keys (a scaled experiment) reach the constructor; only n_rows is pinned."""
    p = probe_prior({"prior": {"n_cols": (7, 7)}}, 10, 1, noise_scale=1.0, **CLEAN)
    assert p.sample_batch(1).x.shape[2] == 7


def test_truth_cells_excludes_root_and_categorical():
    bt = {"cell_family": torch.tensor([[int(FuncFamily.ROOT), 1, 2, 1]], dtype=torch.int8),
          "categorical": torch.tensor([[0, 0, 3, 0]], dtype=torch.int8)}
    assert truth_cells(bt).tolist() == [[[False, True, False, True]]]
    # on a real-prior batch with a class column, no scored cell is categorical
    pr = RealPrior(RealConfig(n_rows=(40, 40), force_cols=6, target_classes=(3,)), seed=0)
    bt = make_batch(pr, 2, "any_cell", np.random.default_rng(0), split=20, return_truth=True)
    sel = bt["target_mask"] & truth_cells(bt)
    assert sel.any() and not (bt["categorical"].unsqueeze(1).expand_as(sel)[sel] > 0).any()
