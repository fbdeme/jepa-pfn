"""Probe the completed c4_factor_lat reruns (review C9: the truncated-lat fix).

The original c4_factor_lat_s{0..3} runs stopped at 175-200/20000 steps; the paper's
lat companion numbers were sensitivity-grade. The reruns (runs/c4_factor_lat_s*_full)
completed 20k with full metrics + step ckpts. For each seed, probe on the factor
substrate (probe_base PROBE_PRIOR=factor recipe, mse_f only):
  - "final": the 20k checkpoint (the deliverable for surviving seeds),
  - "steelman": the pre-collapse peak (max val_tgt dim_std before onset, geometry
    rule) for seeds that collapsed.

    uv run python -m eval.c4_lat_full_probe

Writes eval/results_c4_lat_full.json.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PROBE_PRIOR", "factor")   # must precede probe_base import

import torch

from eval import probe_base as pb
from model.jepa import JEPA

ROOT = Path(__file__).parents[1]


def build(run, ckpt_file):
    ck = torch.load(ROOT / "runs" / run / ckpt_file, map_location="cpu",
                    weights_only=False)
    c = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]
    m = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
             c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
             c.get("predictor", "mlp"), c.get("mode", "ema"),
             c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
             c.get("r_invariant", False), c.get("n_cls", 0),
             c.get("cls_target", False), c.get("r_scheme", "resample"))
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m.eval()


def mse_of(run, ckpt_file):
    enc = build(run, ckpt_file).encoder().eval()
    return pb.c4_mse_f(pb.collect(enc, 1.0))


def main():
    res = {}
    for d in sorted(ROOT.glob("runs/c4_factor_lat_s*_full")):
        run = d.name
        t = time.time()
        vals = [json.loads(l) for l in open(d / "metrics.jsonl") if "val_jepa" in l]
        onset = next((v["step"] for v in vals if v["val_tgt"]["dim_std"] < 0.01), None)
        f = vals[-1]["val_tgt"]
        entry = dict(final_step=vals[-1]["step"], onset=onset,
                     final_dim_std=f["dim_std"], final_cos=f["cos"],
                     mse_f_final=mse_of(run, "ckpt.pt"))
        if onset is not None:
            pre = [v for v in vals if v["step"] < onset and v["val_tgt"]["dim_std"] >= 0.01]
            if pre:
                pick = max(pre, key=lambda v: v["val_tgt"]["dim_std"])
                s = pick["step"]
                if (d / f"ckpt_{s}.pt").exists():
                    entry.update(steelman_step=s,
                                 steelman_dim_std=pick["val_tgt"]["dim_std"],
                                 mse_f_steelman=mse_of(run, f"ckpt_{s}.pt"))
        res[run] = entry
        print(run, entry, f"[{time.time()-t:.0f}s]", flush=True)
    (ROOT / "eval/results_c4_lat_full.json").write_text(json.dumps(res, indent=1))
    print("wrote eval/results_c4_lat_full.json")


if __name__ == "__main__":
    main()
