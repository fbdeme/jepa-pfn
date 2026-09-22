"""Pre-collapse steelman probe for the collapsed H2 (EMA no-head) seeds.

So the H2 reading does not rest on the single surviving seed: for each collapsed
base_ema_nohead_s* run, pick the pre-collapse peak checkpoint (max val_tgt dim_std
over the saved step ckpts, same rule as the 35M steelman: geometry, never probe
score) and probe it with the mechanism-table recipe. Results merge into
eval/results_probe_h2.json under "<run>_steelman".

    uv run python -m eval.h2_steelman_probe
"""
import json
import time
from pathlib import Path

import torch

from eval import probe_base as pb
from eval.latent_probe import load_jepa
from model.jepa import JEPA

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_probe_h2.json"


def build_from_ckpt(run, step):
    ck = torch.load(ROOT / "runs" / run / f"ckpt_{step}.pt", map_location="cpu",
                    weights_only=False)
    c = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]   # step ckpts may store weights only
    m = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
             c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
             c.get("predictor", "mlp"), c.get("mode", "ema"),
             c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
             c.get("r_invariant", False), c.get("n_cls", 0),
             c.get("cls_target", False), c.get("r_scheme", "resample"))
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m.eval()


def main():
    res = json.loads(OUT.read_text())
    for d in sorted((ROOT / "runs").glob("base_ema_nohead_s*")):
        run = d.name
        key = f"{run}_steelman"
        if key in res or "_steelman" in run:
            continue
        vals = [json.loads(l) for l in open(d / "metrics.jsonl") if "val_jepa" in l]
        onset = next((v["step"] for v in vals if v["val_tgt"]["dim_std"] < 0.01), None)
        if onset is None:
            print(f"{run}: never collapsed, steelman n/a", flush=True)
            continue
        pre = [v for v in vals if v["step"] < onset and v["val_tgt"]["dim_std"] >= 0.01]
        if not pre:
            print(f"{run}: no healthy pre-collapse val checkpoint", flush=True)
            continue
        pick = max(pre, key=lambda v: v["val_tgt"]["dim_std"])
        step = pick["step"]
        if not (d / f"ckpt_{step}.pt").exists():
            print(f"{run}: ckpt_{step}.pt missing", flush=True)
            continue
        t = time.time()
        enc = build_from_ckpt(run, step).encoder().eval()
        b1 = pb.collect(enc, 1.0)
        edge_auc, fam_acc, fam_maj = pb.c5_scores(b1)
        mse_f = {"1.0": pb.c4_mse_f(b1)}
        res[key] = dict(edge_auc=edge_auc, fam_acc=fam_acc, fam_majority=fam_maj,
                        mse_f=mse_f, steelman_step=step,
                        steelman_dim_std=pick["val_tgt"]["dim_std"], onset=onset)
        print(f"{key} (step {step}, dim_std {pick['val_tgt']['dim_std']}) "
              f"edge={edge_auc} fam={fam_acc} mse_f={mse_f['1.0']}  "
              f"[{time.time()-t:.0f}s]", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT.name} ({len(res)} entries)")


if __name__ == "__main__":
    main()
