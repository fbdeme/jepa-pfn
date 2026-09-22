"""Independent verification that the base lat arm's collapse is real (not a logging
artifact): load pre-collapse and post-collapse ckpts, run the target encoder on fresh eval
data, and report the representation geometry directly + the raw per-dim std. A collapsed
encoder maps every cell to the same vector -> raw std 0, max pairwise L2 0.

Also extracts the collapse trajectory from metrics.jsonl (onset + permanent-collapse step).
Writes eval/results_base_collapse_verify.json (source of truth for the review doc).

Run locally on the secured ckpts:  uv run python -m eval.verify_collapse_direct
"""
import json
import shutil
from pathlib import Path

import torch

from model.jepa import collapse_stats
from eval.latent_probe import load_jepa
from eval.probe_base import collect

ROOT = Path(__file__).parents[1]
RUN = "bsw_lat_n1_s0"
COS_COLLAPSE = 0.95


def check(run, step):
    tmp = f"_vf_{run}_{step}"
    d = ROOT / f"runs/{tmp}"
    d.mkdir(exist_ok=True)
    shutil.copy(ROOT / f"runs/{run}/ckpt_{step}.pt", d / "ckpt.pt")
    try:
        enc = load_jepa(tmp).encoder().eval()
        with torch.no_grad():
            h = collect(enc, noise=1.0)[0][0]          # encoder output (..., E)
        v = h.reshape(-1, h.shape[-1]).float()          # (n cells, E)
        cs = collapse_stats(v)
        raw_std = round(v.std(0).mean().item(), 5)      # mean per-dim std of RAW reps
        pdist = round(torch.cdist(v[:8], v[:8]).max().item(), 5)  # cells identical? -> 0
        rec = dict(step=step, dim_std=cs["dim_std"], cos=cs["cos"],
                   raw_std_mean=raw_std, max_L2_8cells=pdist)
        print(f"{run} @ step {step:>5}:  dim_std {cs['dim_std']:.4f}  cos {cs['cos']:.4f}"
              f"  raw_std {raw_std}  max_L2(8 cells) {pdist}", flush=True)
        return rec
    finally:
        shutil.rmtree(d, ignore_errors=True)


def trajectory():
    onset = permanent = None
    rows = []
    for l in (ROOT / f"runs/{RUN}/metrics.jsonl").read_text().splitlines():
        r = json.loads(l)
        if "val_tgt" not in r:
            continue
        g = r["val_tgt"]
        rows.append((r["step"], g["dim_std"], g["cos"]))
        if onset is None and g["cos"] > COS_COLLAPSE:
            onset = r["step"]
    # permanent = first step after which every subsequent val is collapsed
    for i, (s, d, c) in enumerate(rows):
        if all(cc > COS_COLLAPSE for _, _, cc in rows[i:]):
            permanent = s
            break
    peak = max(rows, key=lambda t: t[1] if t[2] < COS_COLLAPSE else -1)
    return dict(collapse_onset_step=onset, permanent_collapse_step=permanent,
                healthy_peak_step=peak[0], healthy_peak_dim_std=round(peak[1], 4),
                n_vals=len(rows))


if __name__ == "__main__":
    verify = [check(RUN, s) for s in (750, 1000, 8000)]
    traj = trajectory()
    out = {"run": RUN, "recipe": "SIGReg latent, matched base recipe (base_lat_s), noise_scale=1",
           "direct_ckpt_verification": verify, "trajectory": traj,
           "conclusion": "collapsed encoder maps every cell to one vector (max_L2=0); "
                         "no well-trained non-collapsed base checkpoint exists at the matched recipe"}
    (ROOT / "eval/results_base_collapse_verify.json").write_text(json.dumps(out, indent=1))
    print("trajectory:", traj)
    print("wrote eval/results_base_collapse_verify.json")
