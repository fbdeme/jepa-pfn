"""Resume determinism (long runs, 2026-09-08): stop at a val step, restart from resume.pt with a longer horizon, and the
run equals the uninterrupted one — model tensors bitwise, metrics.jsonl record for record (wall-clock aside).
Also pins the curriculum: curriculum_steps overrides curriculum_frac, so the y_only phase is the same in both runs."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parents[1]
BASE = yaml.safe_load((ROOT / "configs/v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0.yaml").read_text())
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def _run(tmp_path, name, **kw):
    c = {**BASE, "emb": 32, "heads": 2, "mlp": 64, "layers": 1, "predictor": "twoway-1", "batch_size": 2, "val_batches": 1,
         "val_every": 2, "log_every": 1, "curriculum_steps": 2, "curriculum_frac": 0.9, "run_name": name,
         "prior": {"n_rows": [12, 12]}, **kw}
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(c))
    r = subprocess.run([sys.executable, "-m", "train.train_jepa", str(p)], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1200:]
    return r.stdout


def _records(name):
    recs = [json.loads(l) for l in (ROOT / "runs" / name / "metrics.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if k != "sec"} for r in recs]


def test_resume_equals_uninterrupted_run(tmp_path):
    a, b = "_t_resume_a", "_t_resume_b"
    for n in (a, b):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
    out_a = _run(tmp_path, a, steps=6)
    _run(tmp_path, b, steps=4)                                            # stops after the val step at 4 -> resume.pt
    out_b = _run(tmp_path, b, steps=6, resume=f"runs/{b}/resume.pt")     # horizon raised on resume
    assert "resumed" in out_b and "at step 4" in out_b, out_b[-300:]
    assert "policy': 'y_only'" in out_a and out_a.count("'y_only'") == 2   # curriculum_steps=2 pinned (frac 0.9 would give 5)
    ma, mb = torch.load(ROOT / "runs" / a / "ckpt.pt")["model"], torch.load(ROOT / "runs" / b / "ckpt.pt")["model"]
    assert ma.keys() == mb.keys()
    diff = [k for k in ma if not torch.equal(ma[k], mb[k])]
    assert not diff, f"{len(diff)} tensors differ after resume, e.g. {diff[:5]}"
    ra, rb = _records(a), _records(b)
    assert [r["step"] for r in ra] == [1, 2, 2, 3, 4, 4, 5, 6, 6]           # train logs + val records, no duplicates after resume
    assert ra == rb, "metrics diverge after resume"
    assert (ROOT / "runs" / b / "resume.pt").exists()
    for n in (a, b):
        shutil.rmtree(ROOT / "runs" / n)
