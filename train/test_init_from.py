"""C14 / E6: init_encoder loads a JEPA checkpoint's EMA target encoder into a CellPFN, optionally freezing all but the head."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from model.jepa import JEPA                 # noqa: E402
from model.pfn import CellPFN               # noqa: E402
from train.train import init_encoder        # noqa: E402

ARCH = dict(emb=16, heads=2, mlp=32, layers=1, n_bins=8)


def _jepa_ckpt(tmp_path):
    torch.manual_seed(0)
    j = JEPA(**ARCH, n_cls=2, r_scheme="rope")
    with torch.no_grad():                                   # make target != online so the test can tell them apart
        for q in j.target.parameters():
            q.add_(1.0)
    p = tmp_path / "ckpt.pt"
    torch.save(dict(model=j.state_dict(), cfg={}), p)
    return j, p


def test_init_encoder_loads_target_weights_and_freezes(tmp_path):
    j, p = _jepa_ckpt(tmp_path)
    m = CellPFN(**ARCH, n_cls=2, r_scheme="rope")
    n_t, n_tr = init_encoder(m, str(p), freeze=True)
    tgt = dict(j.target.named_parameters())
    for n, q in m.named_parameters():
        assert torch.equal(q, tgt[n]), n                   # the EMA target copy, not the online encoder
        assert q.requires_grad == n.startswith("head"), n
    assert n_tr == sum(q.numel() for n, q in m.named_parameters() if n.startswith("head"))
    m2 = CellPFN(**ARCH, n_cls=2, r_scheme="rope")
    init_encoder(m2, str(p), freeze=False)
    assert all(q.requires_grad for q in m2.parameters())


def test_init_encoder_refuses_architecture_mismatch(tmp_path):
    _, p = _jepa_ckpt(tmp_path)
    m = CellPFN(**{**ARCH, "layers": 2}, n_cls=2, r_scheme="rope")
    try:
        init_encoder(m, str(p))
    except RuntimeError:
        return
    raise AssertionError("a layer-count mismatch must not load silently")


def test_two_step_smoke_from_init(tmp_path):
    """train.train runs with init_from + freeze_encoder on a tiny model / tiny prior (CPU)."""
    _, p = _jepa_ckpt(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("run_name: _test_init_from\npolicy: any_cell\nr_scheme: rope\nn_cls: 2\nemb: 16\nheads: 2\nmlp: 32\nlayers: 1\nn_bins: 8\n"
                   "batch_size: 2\nsteps: 2\nlog_every: 1\nval_every: 2\nval_batches: 1\nlr: 0.001\nprior: {n_rows: [20, 24], n_cols: [3, 4]}\n"
                   f"init_from: {p}\nfreeze_encoder: true\n")
    out = subprocess.run([sys.executable, "-m", "train.train", str(cfg)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert "init_from=" in out.stdout and "freeze_encoder=True" in out.stdout and "val_mse" in out.stdout
    import shutil
    shutil.rmtree(ROOT / "runs/_test_init_from", ignore_errors=True)
