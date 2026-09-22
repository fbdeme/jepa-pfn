"""Issue #18: does the objective-family GAP depend on the world's noise?

Axis A of the #14 defence says a latent objective pays off when the target is
dominated by nuisance the encoder should discard. Our prior's nuisance
fraction is a median 0.33% - almost nothing to discard - which is why we
predict the data-space objective wins here. That is an observation, not a
manipulation. This is the manipulation: retrain the pair at elevated prior
noise and watch the gap.

Arms come in PAIRS (p4n_ds_sN, p4n_lat_sN) trained at noise_scale N. Only the
gap within a pair is the measurement; absolute numbers move with noise for
trivial reasons. scale 1 is the reproduction control - it should land on the
existing p2_anycell / p3b_mixed_tw2 arms.

Train-noise x eval-noise is a full grid because evaluation is CPU
forward-only and therefore free: an arm trained at scale 8 is scored at every
scale, which separates "the gap depends on the world" from "the gap depends
on the test conditions".

Run: uv run python -m eval.noise_grid   (CPU, forward-only)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import load_jepa
from eval.truth_eval import c4_collect, c5_scores, ridge_mse
from model.pfn import CellPFN

ROOT = Path(__file__).parents[1]
TRAIN_SCALES = [1, 2, 4, 8, 16]
EVAL_SCALES = [1.0, 4.0, 8.0]
# scale-1 arms that already exist; the p4n_*_s1 pair reproduces them
LEGACY = {"ds": "p2_anycell", "lat": "p3b_mixed_tw2"}


def load_any(name):
    """Data-space arms are a bare CellPFN; latent arms are a JEPA wrapper."""
    ck = torch.load(ROOT / "runs" / name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    if any(k.startswith("online.") for k in ck["model"]):
        return load_jepa(name).encoder().eval()
    c = ck["cfg"]
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                n_cls=c.get("n_cls", 0), r_scheme=c.get("r_scheme", "resample"))
    m.load_state_dict(ck["model"])
    return m.eval()


def present(name):
    return (ROOT / "runs" / name / "ckpt.pt").exists()


def score(enc, eval_scale):
    """C4 mechanism recovery + the C5 structural probes, at one eval noise."""
    H, F_, X = c4_collect(enc, eval_scale)
    c5 = c5_scores(enc, noise_scale=eval_scale)
    return dict(mse_f=round(ridge_mse(H, F_), 4), mse_x=round(ridge_mse(H, X), 4),
                edge_auc=c5["edge_auc"], fam_acc=c5["fam_acc"],
                deg_mse=c5["deg_mse"])


def run():
    out = {}
    for tr in TRAIN_SCALES:
        pair = {}
        for kind in ("ds", "lat"):
            name = f"p4n_{kind}_s{tr}"
            if not present(name):
                if tr == 1 and present(LEGACY[kind]):
                    name = LEGACY[kind]          # fall back to the existing arm
                else:
                    print(f"skip {name}: not trained yet", flush=True)
                    continue
            enc = load_any(name)
            pair[kind] = {"arm": name,
                          **{f"@{ev}": score(enc, ev) for ev in EVAL_SCALES}}
            print(f"train_scale={tr:2} {kind:3} ({name})", flush=True)
            for ev in EVAL_SCALES:
                print(f"    eval@{ev}: {pair[kind][f'@{ev}']}", flush=True)
        if len(pair) == 2:
            # the measurement: data-space minus latent, positive = latent better
            # on mse (lower is better) and negative = latent better on auc/acc
            pair["gap"] = {
                f"@{ev}": dict(
                    mse_f=round(pair["ds"][f"@{ev}"]["mse_f"]
                                - pair["lat"][f"@{ev}"]["mse_f"], 4),
                    edge_auc=round(pair["ds"][f"@{ev}"]["edge_auc"]
                                   - pair["lat"][f"@{ev}"]["edge_auc"], 4),
                    fam_acc=round(pair["ds"][f"@{ev}"]["fam_acc"]
                                  - pair["lat"][f"@{ev}"]["fam_acc"], 4))
                for ev in EVAL_SCALES}
            print(f"  GAP (ds - lat, >0 = data-space better on auc/acc, "
                  f"<0 on mse_f): {pair['gap']}", flush=True)
        out[f"train_s{tr}"] = pair
    (ROOT / "eval/results_noise_grid.json").write_text(json.dumps(out, indent=1))
    print("\nwrote eval/results_noise_grid.json")
    return out


if __name__ == "__main__":
    run()
