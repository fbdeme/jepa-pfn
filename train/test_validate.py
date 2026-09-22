"""train.train_jepa.validate contract (C1 + C12): keys, the table-id instrument, the loss identity."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.jepa import JEPA                          # noqa: E402
from train.train_jepa import DEFAULTS, validate      # noqa: E402


def _cfg(**kw):
    return {**DEFAULTS, "seed": 0, "emb": 16, "heads": 2, "mlp": 32, "layers": 1, "n_bins": 8, "batch_size": 3,
            "val_batches": 1, "r_scheme": "rope", "prior": dict(n_rows=(20, 24), n_cols=(3, 4), p_missing_table=0.0), **kw}


def _model(c):
    return JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"], c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
                c["predictor"], c["mode"], c["lambda_sig"], c["sig_proj"], c["r_invariant"], c["n_cls"], c["cls_target"],
                c["r_scheme"], c["shuffle_target"], n_classes=c["n_classes"]).eval()


def test_validate_dual_logs_ppd_total_and_within_table():
    torch.manual_seed(0)
    cfg = _cfg(lambda_ppd=0.1)
    out = validate(_model(cfg), cfg, "cpu")
    assert set(out) == {"val_jepa", "val_total", "val_tgt", "val_pred", "val_ppd", "val_mse"}   # val_mse: train.train's definition (2026-09-08)
    for k in ("val_tgt", "val_pred"):
        assert 0 < out[k]["within_table_var_frac"] <= 1, out[k]
    # one batch + rope (no resampled column code): val_total = val_jepa + lambda * bar-CE up to 4-decimal rounding
    assert abs(out["val_total"] - (out["val_jepa"] + 0.1 * out["val_ppd"])) < 2e-4, out


def test_validate_no_head_has_no_ppd():
    torch.manual_seed(0)
    cfg = _cfg(lambda_ppd=0.0)
    out = validate(_model(cfg), cfg, "cpu")
    assert "val_ppd" not in out and out["val_total"] == out["val_jepa"]


def test_validate_cls_target_uses_cls_rows():
    """cls_target: the scored rows are CLS summaries; one table id per scored entry still lines up."""
    torch.manual_seed(0)
    cfg = _cfg(lambda_ppd=0.0, n_cls=2, cls_target=True)
    out = validate(_model(cfg), cfg, "cpu")
    assert out["val_tgt"]["within_table_var_frac"] is not None
