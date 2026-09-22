"""Generate the sigma-sweep pilot configs (C-1b).

12 arms = {ds, lat_s} x noise_scale{1,3,10} x seed{0,1}, pilot scale (train_jepa/train
DEFAULTS: emb96/h4/mlp192/L3, ~0.36M). Same data regime as base (n_rows[128,384],
n_cols[4,32]); only the model is small and only noise_scale varies. ds mirrors base_ds
(routes to train.train via `policy:`), lat_s mirrors base_lat_s (train.train_jepa, SIGReg
CLS-latent). Tests whether the ds-lat mechanism gap closes as strippable nuisance grows.

Run:  python configs/_gen_sigma_sweep.py   (writes configs/sw_*.yaml)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE = {1: "n1", 3: "n3", 10: "n10"}
SEEDS = [0, 1]
STEPS = 20000   # matches the pilot grid; trim via smoke timing if CPU-bound run is long


def prior_block(ns):
    # DEFAULT pilot regime (n_rows 64-150, n_cols 3-8), matching the p6/p4n pilot grid;
    # only noise_scale varies. The base wide-D regime (n_cols 4-32) OOMs at batch 32 and
    # is reserved for the base-scale confirmation (Path B).
    return f"prior: {{noise_scale: {ns}}}\n"


def ds_cfg(ns, seed, name):
    # data-space arm: `policy:` routes remote.sh to train.train. NO batch/lr override --
    # inherit the pilot DEFAULTS (batch 32, lr 4e-3) exactly as the p4n/p6rc pilot grid,
    # for reproducibility. Big-VRAM GPU runs all 12 concurrently (never shrink batch).
    return (f"# sigma-sweep ds arm, noise_scale={ns} seed={seed}. Pilot of base_ds.\n"
            f"run_name: {name}\npolicy: any_cell\nr_scheme: rope\nn_cls: 4\n"
            f"seed: {seed}\nsteps: {STEPS}\n{prior_block(ns)}")


def lat_cfg(ns, seed, name):
    # pure-latent SIGReg CLS-target arm: mirrors base_lat_s minus base-scale dims. NO
    # batch/lr override -> pilot DEFAULTS (batch 32, lr 2e-3), same as the pilot grid.
    return (f"# sigma-sweep lat_s arm, noise_scale={ns} seed={seed}. Pilot of base_lat_s.\n"
            f"run_name: {name}\npolicy_main: mixed\npredictor: twoway-2\nr_scheme: rope\n"
            f"n_cls: 4\ncls_target: true\nlambda_ppd: 0.0\nmode: sigreg\nlambda_sig: 0.25\n"
            f"seed: {seed}\nsteps: {STEPS}\n{prior_block(ns)}")


def main():
    written = []
    for ns, tag in NOISE.items():
        for seed in SEEDS:
            for kind, fn in [("ds", ds_cfg), ("lat", lat_cfg)]:
                name = f"sw_{kind}_{tag}_s{seed}"
                with open(os.path.join(HERE, f"{name}.yaml"), "w") as f:
                    f.write(fn(ns, seed, name))
                written.append(name)
    print(f"wrote {len(written)} configs:")
    for n in written:
        print(" ", n)


if __name__ == "__main__":
    main()
