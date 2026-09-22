from train.plateau import Plateau


def test_plateau_stops_after_patience_and_resets_on_improvement():
    p = Plateau(delta=0.01, patience=3)
    assert [p.step(v) for v in (1.0, 0.95, 0.949, 0.948, 0.947)] == [False, False, False, False, True]   # .95 best, 3 flat vals
    p = Plateau(delta=0.01, patience=3)
    assert [p.step(v) for v in (1.0, 0.999, 0.998, 0.9, 0.899, 0.898, 0.897)] == [False, False, False, False, False, False, True]


def test_plateau_off_when_delta_zero():
    p = Plateau(delta=0.0, patience=1)
    assert not any(p.step(v) for v in (1.0, 1.0, 1.0, 1.0))


# end-to-end: both trainers stop early on a plateau and log a `stop` record (stop_delta so large nothing ever "improves")
import json, subprocess, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _run(tmp_path, base, module, **kw):
    c = {**yaml.safe_load((ROOT / base).read_text()), "emb": 32, "heads": 2, "mlp": 64, "layers": 1, "steps": 40, "val_every": 2, "val_batches": 1,
         "batch_size": 2, "log_every": 1, "prior": {"n_rows": [12, 12]}, "stop_delta": 10.0, "stop_patience": 3, **kw}
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(c))
    r = subprocess.run([sys.executable, "-m", module, str(p)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    recs = [json.loads(l) for l in (ROOT / "runs" / c["run_name"] / "metrics.jsonl").read_text().splitlines() if l.strip()]
    stop = [x for x in recs if x.get("stop") == "plateau"]
    return stop, max(x["step"] for x in recs)


def test_train_stops_on_plateau(tmp_path):
    stop, last = _run(tmp_path, "configs/v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s0.yaml", "train.train", run_name="_t_plateau_ds")
    assert stop and stop[0]["step"] == 8 and last == 8   # first val 2 = best, vals 4/6/8 flat -> stop at 8 of 40


def test_train_jepa_stops_on_plateau(tmp_path):
    stop, last = _run(tmp_path, "configs/v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s0.yaml", "train.train_jepa", run_name="_t_plateau_jepa", lambda_ppd=0.1)
    assert stop and stop[0]["step"] == 8 and last == 8
