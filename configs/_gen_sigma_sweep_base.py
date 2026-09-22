"""Generate the BASE-scale sigma-sweep configs (Path B: closes the scale x noise gap).

12 arms = {ds, lat_s} x noise_scale{1,3,10} x seed{0,1} at the base 6.6M scale, mirroring
configs/base_ds.yaml + base_lat_s.yaml EXACTLY (emb256/h8/mlp1024/L6, wide regime
n_rows[128,384] n_cols[4,32], ds batch8/lr1e-3, lat batch4/lr1e-3) -- only noise_scale
and seed vary. Recipe is unchanged from the trained base arms, so no re-tuning / no
collapse risk from batch changes. lat sigreg ~40GB -> one arm per 48GB GPU (run in waves).

Run:  python configs/_gen_sigma_sweep_base.py   (writes configs/bsw_*.yaml)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE = {1: "n1", 3: "n3", 10: "n10"}
SEEDS = [0, 1]
STEPS = 20000       # ds: full training (no collapse)
STEPS_LAT = 8000    # lat collapses ~step 5500 at base scale; 8000 captures the pre-collapse
                    # steelman (save_steps keeps step-tagged ckpts) and confirms collapse.
DIMS = "emb: 256\nheads: 8\nmlp: 1024\nlayers: 6\nn_bins: 32\n"


def prior_block(ns):
    return f"prior:\n  n_rows: [128, 384]\n  n_cols: [4, 32]\n  noise_scale: {ns}\n"


def ds_cfg(ns, seed, name):
    return (f"# base sigma-sweep ds arm, noise_scale={ns} seed={seed}. Mirrors base_ds.\n"
            f"run_name: {name}\npolicy: any_cell\nr_scheme: rope\nn_cls: 4\n"
            f"batch_size: 8\nlr: 0.001\nseed: {seed}\nsteps: {STEPS}\n{DIMS}{prior_block(ns)}")


def lat_cfg(ns, seed, name):
    return (f"# base sigma-sweep lat_s arm, noise_scale={ns} seed={seed}. Mirrors base_lat_s.\n"
            f"run_name: {name}\npolicy_main: mixed\npredictor: twoway-2\nr_scheme: rope\n"
            f"n_cls: 4\ncls_target: true\nlambda_ppd: 0.0\nmode: sigreg\nlambda_sig: 0.25\n"
            f"batch_size: 4\nlr: 0.001\nseed: {seed}\nsteps: {STEPS_LAT}\nsave_steps: true\n"
            f"{DIMS}{prior_block(ns)}")


def main():
    written = []
    for ns, tag in NOISE.items():
        for seed in SEEDS:
            for kind, fn in [("ds", ds_cfg), ("lat", lat_cfg)]:
                name = f"bsw_{kind}_{tag}_s{seed}"
                with open(os.path.join(HERE, f"{name}.yaml"), "w") as f:
                    f.write(fn(ns, seed, name))
                written.append(name)
    print(f"wrote {len(written)} base configs:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
