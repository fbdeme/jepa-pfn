"""C27 / E10: train.train writes step-tagged snapshots at save_at and can re-initialise the head after init_from."""
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parents[1]
BASE = yaml.safe_load((ROOT / "configs/v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s0.yaml").read_text())


def _cfg(tmp_path, **kw):
    c = {**BASE, "emb": 32, "heads": 2, "mlp": 64, "layers": 1, "steps": 4, "val_every": 2, "val_batches": 1, "batch_size": 2, "log_every": 1,
         "prior": {"n_rows": [12, 12]}, **kw}
    p = tmp_path / f"{kw.get('run_name', 'cfg')}.yaml"; p.write_text(yaml.safe_dump(c)); return p


def _run(p):
    return subprocess.run([sys.executable, "-m", "train.train", str(p)], cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_save_at_then_reinit_head(tmp_path):
    r = _run(_cfg(tmp_path, run_name="_t_ds_save_at", save_at=[2, 4], policy="mixed"))
    assert r.returncode == 0, r.stderr[-800:]
    d = ROOT / "runs/_t_ds_save_at"
    assert (d / "ckpt_2.pt").exists() and (d / "ckpt_4.pt").exists()
    head_before = torch.load(d / "ckpt_2.pt", map_location="cpu", weights_only=False)["model"]["head.0.weight"]
    r = _run(_cfg(tmp_path, run_name="_t_ds_reinit", init_from=f"runs/_t_ds_save_at/ckpt_2.pt", reinit_head=True, steps=2, val_every=2))
    assert r.returncode == 0 and "reinit_head:" in r.stdout, r.stderr[-800:]
    r2 = _run(_cfg(tmp_path, run_name="_t_ds_keep", init_from=f"runs/_t_ds_save_at/ckpt_2.pt", reinit_head=False, steps=2, val_every=2))
    assert r2.returncode == 0 and "reinit_head:" not in r2.stdout
    for name in ("_t_ds_save_at", "_t_ds_reinit", "_t_ds_keep"):
        dd = ROOT / "runs" / name
        for f in dd.iterdir():
            f.unlink()
        dd.rmdir()
