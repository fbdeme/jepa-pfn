"""C5 (matched nonlinear rerun): does the ds-lat mechanism gap change when the prior's
mechanism nonlinearity is raised from the base (+.023 headroom) to L4 (+.128, richest learnable)?
Derives the comparison from the two source-backed probe JSONs (no hand-typed gaps) and writes
eval/results_matched_L4_compare.json.

  base = base_ds/base_dual/base_lat_s, trained+probed at the +.023 prior (results_probe_base.json).
  L4   = matched_L4_*, trained+probed at the +.128 prior (results_probe_matched_L4.json), lat at
         its pre-collapse steelman (results_matched_L4_steelman_pick.json).

Run:  uv run python -m eval.matched_L4_compare
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load(name):
    return json.loads((ROOT / "eval" / name).read_text())


def summarize(d, ds, dual, lat):
    D, U, L, F = d[ds], d[dual], d[lat], d["random_init"]
    mf = lambda a: a["mse_f"]["1.0"]
    return dict(
        ds_edge=D["edge_auc"], lat_edge=L["edge_auc"], floor_edge=F["edge_auc"],
        edge_gap=round(D["edge_auc"] - L["edge_auc"], 4),
        ds_msef=mf(D), dual_msef=mf(U), lat_msef=mf(L), floor_msef=mf(F),
        msef_gap=round(mf(D) - mf(L), 4),
        lat_over_ds_msef=round(mf(L) / mf(D), 2),
        dual_beats_ds_msef=bool(mf(U) < mf(D)),
    )


def main():
    base = summarize(load("results_probe_base.json"), "base_ds", "base_dual", "base_lat_s")
    l4 = summarize(load("results_probe_matched_L4.json"),
                   "matched_L4_ds", "matched_L4_dual", "matched_L4_lat")
    steel = load("results_matched_L4_steelman_pick.json")["matched_L4_lat"]
    out = dict(
        note="C5 matched nonlinear rerun, seed 0. edge_gap/msef_gap = ds - lat (msef lower=better).",
        headroom_base=0.023, headroom_L4=0.128,
        lat_steelman_step=steel["step"], lat_steelman_dim_std=steel["dim_std"],
        base=base, L4=l4,
        verdict=dict(
            latent_catches_up_at_L4=bool(l4["lat_over_ds_msef"] <= base["lat_over_ds_msef"]),
            lat_relatively_worse_at_L4=bool(l4["lat_over_ds_msef"] > base["lat_over_ds_msef"]),
            dual_beats_ds_both=bool(base["dual_beats_ds_msef"] and l4["dual_beats_ds_msef"]),
        ),
    )
    (ROOT / "eval" / "results_matched_L4_compare.json").write_text(json.dumps(out, indent=2))
    print(f"base f-MSE  ds {base['ds_msef']} lat {base['lat_msef']} (lat/ds {base['lat_over_ds_msef']}x)")
    print(f"L4   f-MSE  ds {l4['ds_msef']} lat {l4['lat_msef']} (lat/ds {l4['lat_over_ds_msef']}x)")
    print(f"latent catches up at L4? {out['verdict']['latent_catches_up_at_L4']}  "
          f"(relatively worse: {out['verdict']['lat_relatively_worse_at_L4']})")
    print(f"dual beats ds both scales? {out['verdict']['dual_beats_ds_both']}")
    print("wrote eval/results_matched_L4_compare.json")


if __name__ == "__main__":
    main()
