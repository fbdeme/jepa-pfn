"""C2: the single collapse criterion and the onset / recovery trajectory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.collapse import THR_DIM_STD, THR_WITHIN_TABLE, is_collapsed, trajectory

A, D = {"dim_std": 0.9}, {"dim_std": 0.0}


def test_is_collapsed_dim_std_or_within_table():
    assert not is_collapsed(A) and is_collapsed(D)
    assert is_collapsed({"dim_std": THR_DIM_STD / 2}) and not is_collapsed({"dim_std": THR_DIM_STD})
    assert is_collapsed({"dim_std": 0.9, "within_table_var_frac": THR_WITHIN_TABLE / 2})
    assert not is_collapsed({"dim_std": 0.9, "within_table_var_frac": THR_WITHIN_TABLE})
    assert not is_collapsed({"dim_std": 0.9, "within_table_var_frac": None})      # not logged / undefined: dim_std alone


def test_trajectory_onset_recovery_last_state():
    v = [dict(step=s, val_tgt=t) for s, t in [(250, A), (500, D), (750, D), (1000, A), (1250, A)]]
    assert trajectory(v) == dict(collapsed=False, onset=500, recovered=1000, last_step=1250)
    assert trajectory(v[:3]) == dict(collapsed=True, onset=500, recovered=None, last_step=750)
    assert trajectory(v + [dict(step=1500, val_tgt=D)]) == dict(collapsed=True, onset=500, recovered=1000, last_step=1500)
    assert trajectory(v[:1]) == dict(collapsed=False, onset=None, recovered=None, last_step=250)
    assert trajectory([]) == dict(collapsed=False, onset=None, recovered=None, last_step=None)
