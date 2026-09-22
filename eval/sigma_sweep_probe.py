"""Mechanism-probe eval for the sigma-sweep (C-1b): does the ds-lat gap close as
strippable nuisance grows?

For each trained arm (ds / lat_s at noise_scale in {1,3,10}, 2 seeds), freeze the
encoder and probe row-averaged column embeddings for edge presence (AUC), function
family (acc), and noise-free f recovery (MSE) -- EVALUATED ON THE PRIOR AT ITS OWN
TRAINING noise_scale, because nuisance-stripping (the mechanism the paper credits for
the vision win) can only help where the eval data actually carries nuisance. A
random-init encoder at each noise level is the null anchor (Devil's Advocate C-4:
random projection is linear, so this also shows whether "below random" survives noise).

Reuses eval.probe_base.{load_cellpfn, collect, c4_mse_f, c5_scores}. CPU forward-only.
Reads runs/<arm>/ckpt.pt for the 12 sw_* arms. Writes eval/results_sigma_sweep_probe.json.

Run:  uv run python -m eval.sigma_sweep_probe
"""
import json
import os
import time
from pathlib import Path

import torch

from eval.probe_base import load_cellpfn, collect, c4_mse_f, c5_scores
from eval.latent_probe import load_jepa
from model.pfn import CellPFN

ROOT = Path(__file__).parents[1]
NOISE = {1: "n1", 3: "n3", 10: "n10"}
SEEDS = [0, 1]
# nuisance fraction at each noise_scale (from eval/results_sigma_sweep_nuisance.json)
NUIS = {1: 0.013, 3: 0.087, 10: 0.337}
# PREFIX=sw -> pilot arms (0.36M); PREFIX=bsw -> base arms (6.6M). Output tagged to match.
PREFIX = os.environ.get("PREFIX", "sw")


def load_arm(kind, tag, seed):
    run = f"{PREFIX}_{kind}_{tag}_s{seed}"
    if kind == "ds":
        return load_cellpfn(run), run
    return load_jepa(run).encoder().eval(), run


def rand_encoder(ref_run):
    """arch-only random-init anchor, seeded like probe_base.encoders()."""
    c = torch.load(ROOT / "runs" / ref_run / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]
    torch.manual_seed(999)
    return CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                   n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample").eval()


def probe_at(enc, ns):
    """edge AUC, family acc, and f-MSE, all at eval noise_scale = ns (matched)."""
    b = collect(enc, noise=float(ns))
    edge_auc, fam_acc, fam_maj = c5_scores(b)
    mse_f = c4_mse_f(b)
    return dict(edge_auc=edge_auc, fam_acc=fam_acc, fam_majority=fam_maj, mse_f=mse_f)


def main():
    records = []
    for ns, tag in NOISE.items():
        ref = f"{PREFIX}_ds_{tag}_s0"   # any arm's cfg gives the (shared) arch
        for kind in ("ds", "lat"):
            for seed in SEEDS:
                enc, run = load_arm(kind, tag, seed)
                t = time.time()
                r = probe_at(enc, ns)
                rec = dict(kind=kind, noise_scale=ns, seed=seed, run=run,
                           nuisance=NUIS[ns], **r)
                records.append(rec)
                print(f"{run:16} nuis={NUIS[ns]:.3f} edge={r['edge_auc']} "
                      f"fam={r['fam_acc']}(maj{r['fam_majority']}) mse_f={r['mse_f']} "
                      f"[{time.time()-t:.0f}s]", flush=True)
        # random-init anchor at this eval noise
        rr = probe_at(rand_encoder(ref), ns)
        records.append(dict(kind="random_init", noise_scale=ns, seed=-1,
                            run="random_init", nuisance=NUIS[ns], **rr))
        print(f"{'random_init':16} nuis={NUIS[ns]:.3f} edge={rr['edge_auc']} "
              f"mse_f={rr['mse_f']}", flush=True)

    # ds-lat gap per (noise_scale, seed): positive edge-gap => ds encodes MORE structure
    gaps = []
    by = {(r["kind"], r["noise_scale"], r["seed"]): r for r in records}
    for ns in NOISE:
        for seed in SEEDS:
            d, l = by.get(("ds", ns, seed)), by.get(("lat", ns, seed))
            if d and l:
                gaps.append(dict(noise_scale=ns, seed=seed, nuisance=NUIS[ns],
                                 edge_gap=round(d["edge_auc"] - l["edge_auc"], 4),
                                 mse_f_gap=round(l["mse_f"] - d["mse_f"], 4)))  # +ve => lat worse
    out = {"records": records, "ds_minus_lat_gaps": gaps, "nuisance_by_noise_scale": NUIS,
           "prefix": PREFIX}
    tag = "" if PREFIX == "sw" else f"_{PREFIX}"
    path = ROOT / f"eval/results_sigma_sweep_probe{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print("\n=== ds-lat gap vs nuisance (does it close at high noise?) ===")
    for g in gaps:
        print(f"  ns={g['noise_scale']:>2} nuis={g['nuisance']:.3f} seed={g['seed']} "
              f"edge_gap(ds-lat)={g['edge_gap']:+.4f}  mse_f_gap(lat-ds)={g['mse_f_gap']:+.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
