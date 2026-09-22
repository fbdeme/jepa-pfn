"""C25 / E8: train_jepa saves a step-tagged snapshot at every `save_at` step (a validation step) and refuses non-val steps."""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
BASE = yaml.safe_load((ROOT / "configs/v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s0.yaml").read_text())


def _cfg(tmp_path, **kw):
    c = {**BASE, "emb": 32, "heads": 2, "mlp": 64, "layers": 1, "steps": 4, "val_every": 2, "val_batches": 1, "batch_size": 2, "log_every": 1,
         "prior": {"n_rows": [12, 12]}, **kw}
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(c)); return p


def test_save_at_writes_snapshots(tmp_path):
    p = _cfg(tmp_path, run_name="_t_save_at", save_at=[2, 4])
    r = subprocess.run([sys.executable, "-m", "train.train_jepa", str(p)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    d = ROOT / "runs/_t_save_at"
    assert (d / "ckpt_2.pt").exists() and (d / "ckpt_4.pt").exists() and (d / "ckpt.pt").exists()
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def test_save_at_rejects_non_val_step(tmp_path):
    p = _cfg(tmp_path, run_name="_t_save_at_bad", save_at=[3])
    r = subprocess.run([sys.executable, "-m", "train.train_jepa", str(p)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "save_at" in r.stderr
