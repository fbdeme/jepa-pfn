"""E6 doc block (paper_v4/build_docs.py::e6_block) renders from the agg_e6 artifact shape only - no hand-typed numbers."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.test_v4_readout import NO_SRC, SPEC, _ds_run          # noqa: E402
from eval.v4_readout import aggregate                           # noqa: E402

spec = importlib.util.spec_from_file_location("build_docs", ROOT / "paper_v4/build_docs.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)


def _e6_artifact(tmp_path):
    runs, val = {}, {}
    lat = {s: f"v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s{s}" for s in range(3)}
    for s in range(3):
        for mech, extra, steps, init, frz, mse in (("latfrozen", "5k", 5000, lat[s], True, 0.95), ("latinit", "5k", 5000, lat[s], False, 0.70),
                                                    ("none", "5k", 5000, None, False, 0.80), ("latinit", "20k", 20000, lat[s], False, 0.60)):
            name = f"v4_ds_{mech}_lam0_lr5e-4_any_cell_bar32_{extra}_s{s}"
            _ds_run(tmp_path, name, steps, steps, mse)
            runs[name] = dict(arm="ds", mech=mech, lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=steps, reused=False,
                              **({"init_from": init, "freeze_encoder": frz} if init else {}))
            val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": mse}
        name = f"real_ds_lr5e-4_s{s}"
        _ds_run(tmp_path, name, 20000, 20000, 0.65)
        runs[name] = dict(arm="ds", mech="none", lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=20000, reused=True)
        val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": 0.65}
    src = dict(NO_SRC, value={"runs": val})
    return aggregate("E6", {"experiments": {"E6": {"runs": runs, "artifact": "x"}}}, SPEC, src, tmp_path)


def test_e6_block_from_aggregate(tmp_path):
    e6 = _e6_artifact(tmp_path)
    n, rows = bd.e6_block(e6)
    assert set(n) == set(bd.E6_KEYS)
    assert n["e6_n_runs"] == 15 and n["e6_n_complete"] == 15 and n["e6_margin"] == SPEC["experiments"][0]["equivalence_margin"] if SPEC["experiments"][0]["id"] == "E6" else True
    assert n["e6_ft5_gap"] == -0.1 and n["e6_frozen_gap"] == 0.15 and n["e6_ft20_gap"] == -0.05
    assert n["e6_helps"].startswith("사전학습이 도움") and n["e6_repr"].startswith("표현만으로는 불충분")
    lines = rows.splitlines()
    assert lines[0].startswith("| 팔 |") and len(lines) == 2 + 3 * (3 + 1)      # header x2, 3 arms x (3 seed rows + mean row)
    assert "| 0.95 / 0.8 |" in rows and "| 0.6 / 0.65 |" in rows                  # absolute values come from the artifact runs
    assert "| n 3 |" in rows and "**-0.1**" in rows and "| 예 | 아니오 |" in rows


def test_e6_block_handles_incomplete_pairs(tmp_path):
    e6 = _e6_artifact(tmp_path)
    e6["pairs"]["finetune_20k"] = {"ref": "scratch_20k", "pairs": [], "n": 0, "mean_gap_value": None, "mean_gap_val_mse": None, "helps": None, "equivalent": None}
    e6["verdict"] = {"pretraining_helps": None, "representation_sufficient": None}
    n, rows = bd.e6_block(e6)
    assert n["e6_helps"] == "미판정" and n["e6_repr"] == "미판정" and n["e6_ft20_gap"] is None
    assert "| — | — |" in rows
