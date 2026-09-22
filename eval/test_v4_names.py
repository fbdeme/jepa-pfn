"""C6 / C13: the v4 config name rule round-trips through the readout regex; the generator's counts match the spec."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.real_readout import PAT                                     # noqa: E402
from scripts.gen_v4_configs import cell, name                         # noqa: E402


def test_pat_matches_real_and_v4_names_only():
    assert PAT.match("real_ds_lr5e-4_s0").groups() == ("ds", "5e-4", "0")
    assert PAT.match("real_dual_lr1.7e-4_s12").groups() == ("dual", "1.7e-4", "12")
    assert PAT.match(name("dual", "ema", 0.1, "5e-4", "mixed", "cls_row", 2)).groups() == ("dual", "5e-4", "2")
    assert PAT.match(name("dual", "sigreg", 0.0, "1.5e-3", "row_block_partial", "cell", 0)).groups() == ("dual", "1.5e-3", "0")
    assert PAT.match(name("ds", "none", 0.0, "1.7e-4", "any_cell", "bar32", 1, extra="60k")).groups() == ("ds", "1.7e-4", "1")
    for bad in ("cls_ds_lr5e-4_s0", "real_ds_smoke", "base_dual_s0", "v4_lat_ema_lam0_lr5e-4_mixed_cls_row_s0"):
        assert PAT.match(bad) is None, bad


def test_cell_writes_the_trainer_keys():
    c, ax = cell("dual", "sigreg", 0.1, "5e-4", "mixed", "cls_row", 1, lambda_sig=0.25)
    assert c["mode"] == "sigreg" and c["lambda_sig"] == 0.25 and c["lambda_ppd"] == 0.1 and c["cls_target"] is True
    assert c["policy_main"] == "mixed" and "policy" not in c and c["prior_source"] == "real" and c["lr"] == 5e-4
    c, ax = cell("ds", "none", 0.0, "1.7e-4", "any_cell", "bar32", 0)
    assert c["policy"] == "any_cell" and "policy_main" not in c and "mode" not in c
    c, _ = cell("dual", "ema", 0.1, "5e-4", "mixed", "cell", 0)
    assert c["cls_target"] is False and c["n_cls"] == 4                # cell target keeps the CLS columns for parity


def test_generator_counts_match_spec(tmp_path):
    """Regenerating is idempotent, and every experiment's generated + reused = runs in experiments.yaml."""
    subprocess.run([sys.executable, "scripts/gen_v4_configs.py"], cwd=ROOT, check=True, capture_output=True)
    m = json.loads((ROOT / "configs/v4_manifest.json").read_text())
    spec = {e["id"]: e for e in yaml.safe_load((ROOT / "paper_v4/experiments.yaml").read_text())["experiments"]}
    for eid, ex in m["experiments"].items():
        assert ex["n_runs"] == spec[eid]["runs"] and ex["n_generated"] + ex["n_reused"] == spec[eid]["runs"]
        assert ex["n_reused"] == len(spec[eid]["reuse_existing"]) + ex["n_reused_from_other"]
        for run, ax in ex["runs"].items():
            assert (ROOT / "configs" / f"{run}.yaml").exists()
            assert PAT.match(run).group(1) == ax["arm"] and PAT.match(run).group(2) == ax["lr"] and int(PAT.match(run).group(3)) == ax["seed"]
            # reused = a POST run named in reuse_existing, or a v4 cell an earlier experiment already trained (reused_from)
            assert ax["reused"] == ((not run.startswith("v4_")) or "reused_from" in ax), (run, ax)
    assert {"E1", "E1b", "E3"} <= set(m["experiments"])
    if "E6" in m["experiments"]:            # reused scratch runs never sit on an init_from cell; every init_from cell is generated
        for run, ax in m["experiments"]["E6"]["runs"].items():
            assert not (ax["reused"] and ax.get("init_from")), (run, ax)
            if ax.get("init_from"):
                assert not ax["reused"] and (ROOT / "runs" / ax["init_from"] / "ckpt.pt").exists(), run
        assert sum(bool(ax.get("init_from")) for ax in m["experiments"]["E6"]["runs"].values()) == 9
    if "E7" in m["experiments"]:            # C23: 3 seeds x (lat-init, scratch) at the same long budget, nothing reused
        e7 = m["experiments"]["E7"]["runs"]
        assert len(e7) == 12 and all(not ax.get("reused") or ax.get("reused_from") for ax in e7.values())   # 60k extension (2026-09-07); scratch 60k s0 = E5's cell
        assert sum(bool(ax.get("init_from")) for ax in e7.values()) == 6 and {ax["steps"] for ax in e7.values()} == {40000, 60000}
        assert all(("_40k_s" in run or "_60k_s" in run) and run.startswith("v4_ds_") for run in e7)
        assert not any(ax.get("freeze_encoder") for ax in e7.values())
    if "E8" in m["experiments"]:            # C25: 3 pretraining runs with snapshots + 6 x 3 fine-tunes from the snapshot files
        e8 = m["experiments"]["E8"]["runs"]
        pt = {r: ax for r, ax in e8.items() if ax["arm"] == "dual"}; ft = {r: ax for r, ax in e8.items() if ax["arm"] == "ds"}
        assert len(pt) == 3 and len(ft) == 18 and all(ax["steps"] == 40000 and ax["save_at"] == [1250, 5000, 10000, 20000, 30000, 40000] for ax in pt.values())
        assert all(ax["steps"] == 5000 and ax["init_from"].startswith("runs/v4_dual_ema_lam0_lr5e-4_mixed_cls_row_40k_s") and ax["init_from"].endswith(".pt") for ax in ft.values())
        assert all(f"/ckpt_{ax['mech'][3:]}.pt" in ax["init_from"] and ax["init_from"].endswith(f"_s{ax['seed']}/ckpt_{ax['mech'][3:]}.pt") for ax in ft.values())
        assert {ax["mech"] for ax in ft.values()} == {f"lat{k}" for k in (1250, 5000, 10000, 20000, 30000, 40000)}
    if "E10" in m["experiments"]:           # C27: 3 ds pretraining runs (mixed masking, snapshots) + 2 x 3 head-reinit fine-tunes
        e10 = m["experiments"]["E10"]["runs"]
        pt = {r: ax for r, ax in e10.items() if ax["mech"] == "dspre"}; ft = {r: ax for r, ax in e10.items() if ax["mech"].startswith("dsinit")}
        assert len(pt) == 3 and len(ft) == 6 and all(ax["policy"] == "mixed" and ax["steps"] == 40000 and ax["save_at"] == [20000, 40000] for ax in pt.values())
        assert all(ax["policy"] == "any_cell" and ax["steps"] == 5000 and ax.get("reinit_head") and ax["init_from"].endswith(f"_s{ax['seed']}/ckpt_{ax['mech'][6:]}.pt") for ax in ft.values())
        assert all((ROOT / "configs" / f"{r}.yaml").read_text().count("reinit_head: true") == 1 for r in ft)

