"""TabICL prior adapter (prior/tabicl.py): SCMBatch contract, label column, determinism, make_batch policies, and a
CPU smoke of both trainers on it (prior_source: tabicl)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from prior.tabicl import TabICLConfig, TabICLPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
SMALL = dict(min_features=3, max_features=6, n_rows=(32, 64), max_cells=10_000)


def _prior(seed=0, **kw):
    return TabICLPrior(TabICLConfig(**{**SMALL, **kw}), seed=seed)


@pytest.mark.parametrize("prior_type", ["graph_scm", "mix_scm"])
def test_contract_cls(prior_type):
    b = _prior(prior_type=prior_type, task="cls").sample_batch(3)
    B, R, D = b.x.shape
    assert B == 3 and 32 <= R <= 64 and 4 <= D <= 7, b.x.shape
    assert b.f.shape == b.x.shape and not b.adjacency.any() and (b.func_family == -1).all()
    assert (b.observed_idx == np.arange(D)).all()
    for i, h in enumerate(b.hyper):
        t, k = h["target_col"], h["n_classes"]
        assert 0 <= t < D and 2 <= k <= 10 and b.categorical[i, t] == k
        lab = b.x[i, :, t]
        assert not np.isnan(lab).any() and np.array_equal(lab, np.round(lab)) and lab.min() >= 0 and lab.max() <= k - 1
        feats = np.delete(b.x[i], t, axis=1)
        assert np.isnan(feats).mean() <= 0.12                       # MCAR rate <= 0.1 (+ sampling slack)


def test_as_is_recipe_filters_on():
    """The TabICLv2 recipe: both ExtraTrees filters on, features from 1 (Tabular-JEPA v3 config)."""
    b = _prior(task="cls", filter_unpredictable=True, filter_graphs=True, min_features=1, max_features=4, prefetch=0).sample_batch(2)
    assert 2 <= b.x.shape[2] <= 5 and all(2 <= h["n_classes"] <= 10 for h in b.hyper)


def test_reg_and_mixed():
    b = _prior(task="reg").sample_batch(2)
    assert all(h["n_classes"] == 0 for h in b.hyper) and all(b.categorical[i, h["target_col"]] == 0 for i, h in enumerate(b.hyper))
    b = _prior(task="mixed", p_cls=1.0).sample_batch(2)
    assert all(h["task"] == "cls" for h in b.hyper)


def test_determinism():
    a, b, c = _prior(0).sample_batch(2), _prior(0).sample_batch(2), _prior(1).sample_batch(2)
    assert np.array_equal(a.x, b.x, equal_nan=True) and a.hyper == b.hyper
    assert a.x.shape != c.x.shape or not np.array_equal(a.x, c.x, equal_nan=True)


def test_prefetch_process_equals_in_process():
    """The child-process generator yields the same table stream as in-process generation (same seed, same batch size)."""
    p0, p1 = _prior(0, task="mixed", prefetch=0), _prior(0, task="mixed", prefetch=3)
    for _ in range(4):
        a, b = p0.sample_batch(2), p1.sample_batch(2)
        assert a.x.shape == b.x.shape and np.array_equal(a.x, b.x, equal_nan=True) and a.hyper == b.hyper
    p1.close()


def test_make_batch_policies():
    p, rng = _prior(task="cls"), np.random.default_rng(0)
    bt = make_batch(p, 2, "y_target", rng)
    assert bt["y_ids"].shape == bt["z"].shape and (bt["n_classes"] >= 2).all() and bt["target_mask"].any()
    bt = make_batch(p, 2, "mixed", rng)
    assert bt["target_mask"].any() and not bt["target_mask"][:, :bt["split"]].any()


BASE = yaml.safe_load((ROOT / "configs/v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0.yaml").read_text())
TINY = {"emb": 32, "heads": 2, "mlp": 64, "layers": 1, "predictor": "twoway-1", "batch_size": 2, "val_batches": 1, "val_every": 2,
        "log_every": 1, "curriculum_steps": 1, "steps": 3, "prior_source": "tabicl",
        "prior": {"prior_type": "graph_scm", "task": "mixed", "min_features": 3, "max_features": 5, "n_rows": [16, 16]}}
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize("module,extra", [("train.train_jepa", {}), ("train.train", {"policy": "any_cell"})])
def test_trainers_smoke(tmp_path, module, extra):
    name = f"_t_tabicl_{module.split('.')[-1]}"
    cfg = {**BASE, **TINY, **extra, "run_name": name}
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, "-m", module, str(p)], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (ROOT / "runs" / name / "metrics.jsonl").exists()
    import shutil; shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
