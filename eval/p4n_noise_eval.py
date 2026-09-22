"""Preview: mechanism-probe the EXISTING p4n noise sweep (Issue #18) to see whether the
ds-lat gap closes as noise grows -- before spending GPU on the headline-recipe sweep.

p4n arms (already trained, ckpts local): p4n_{ds,lat}_s{1,2,4,8,16} = data-space vs
latent at noise_scale {1,2,4,8,16}, seed 0. CAVEAT: p4n_lat is the EARLY pilot recipe
(EMA, single-cell target, narrow default regime, no rope/CLS) -- i.e. exactly the
"degenerate single-cell latent" the reviewer flagged. So this is a lower bound on the
latent objective; the headline sweep (sw_*, SIGReg+CLS) tests the strong version.

Each arm probed at its OWN training noise_scale. CPU. Writes eval/results_p4n_noise_eval.json.
Run: uv run python -m eval.p4n_noise_eval
"""
import json
from pathlib import Path

from eval.probe_base import load_cellpfn, collect, c4_mse_f, c5_scores
from eval.latent_probe import load_jepa
from eval.sigma_sweep_nuisance import measure_at

ROOT = Path(__file__).parents[1]
NS = [1, 2, 4, 8, 16]


def probe(enc, ns):
    b = collect(enc, noise=float(ns))
    edge, fam, maj = c5_scores(b)
    return dict(edge_auc=edge, fam_acc=fam, fam_majority=maj, mse_f=c4_mse_f(b))


def main():
    nuis = {ns: round(measure_at(ns, n_tables=1500)["nuisance_fraction_mean"], 4) for ns in NS}
    recs, gaps = [], []
    for ns in NS:
        d = probe(load_cellpfn(f"p4n_ds_s{ns}"), ns)
        l = probe(load_jepa(f"p4n_lat_s{ns}").encoder().eval(), ns)
        for kind, r in [("ds", d), ("lat", l)]:
            recs.append(dict(kind=kind, noise_scale=ns, nuisance=nuis[ns], **r))
        g = dict(noise_scale=ns, nuisance=nuis[ns],
                 edge_gap_ds_minus_lat=round(d["edge_auc"] - l["edge_auc"], 4),
                 mse_f_gap_lat_minus_ds=round(l["mse_f"] - d["mse_f"], 4))
        gaps.append(g)
        print(f"ns={ns:>2} nuis={nuis[ns]:.3f} | ds edge={d['edge_auc']:.3f} mse_f={d['mse_f']:.3f}"
              f" | lat edge={l['edge_auc']:.3f} mse_f={l['mse_f']:.3f}"
              f" | GAP edge(ds-lat)={g['edge_gap_ds_minus_lat']:+.3f}"
              f" mse_f(lat-ds)={g['mse_f_gap_lat_minus_ds']:+.3f}", flush=True)
    out = {"records": recs, "gaps": gaps, "nuisance": nuis,
           "caveat": "p4n_lat = EMA single-cell latent, narrow regime, no rope/CLS (weak latent)"}
    (ROOT / "eval/results_p4n_noise_eval.json").write_text(json.dumps(out, indent=1))
    print("\nDoes the gap close at high noise? (edge_gap -> 0 or negative = latent catches up)")
    print("wrote eval/results_p4n_noise_eval.json")


if __name__ == "__main__":
    main()
