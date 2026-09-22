"""Extract the training budget of every run the paper cites into a committed JSON.

Round-3 A3: the paper called every latent arm an "equal-budget twin" of its data-space
arm. It is not true everywhere, and the truth is only in `runs/<name>/config.yaml` -- the
RESOLVED config (defaults applied, CLI overrides baked in), which `configs/*.yaml` does
not capture (the sigma-sweep pilots got batch/lr from the command line). `runs/` is
gitignored, so that truth vanishes on a fresh clone or when a box is destroyed.

So: read the resolved configs once, derive parameter counts from the model code, read the
last logged step from metrics.jsonl, and write eval/results_run_configs.json. Everything
downstream (paper/build_tables.py) reads that JSON. Same discipline as
eval/extract_collapse_curve.py.

Mismatches are DERIVED, not asserted: within each group we compare every arm against the
group's data-space arm and record which knobs differ.

Run:  uv run python eval/extract_run_configs.py
"""
import re
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.jepa import JEPA          # noqa: E402
from model.pfn import CellPFN        # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "eval" / "results_run_configs.json"

# group -> (label, reference arm glob, [(arm label, glob)]). Globs expand over runs/.
GROUPS = [
    ("base", "base scale (headline, 5 seeds)", [
        ("ds", "base_ds_s*"),
        ("dual", "base_dual_s*"),
        ("lat (SIGReg)", "base_lat_s_s*"),
        ("EMA, no head", "base_ema_nohead_s*"),
        ("dual+LeWM", "base_dual_lewm"),
    ]),
    ("big", "$35$M scale-up", [
        ("ds", "big_ds"),
        ("dual", "big_dual"),
        ("lat (SIGReg)", "big_lat_s"),
        ("lat steelman", "big_lat_s_h"),
        # round-5 repair 9: this arm supplies an intro-level number and a row of the
        # addressing table, so it belongs in the budget audit. It is a checkpoint lifted
        # from big_dual_lewm, with no run dir of its own -- see collect()'s "ckpt" source.
        ("dual+LeWM steel.", "big_dual_lewm_h"),
    ]),
    ("sigma", "nuisance sweep (pilot)", [
        ("ds", "sw_ds_n*"),
        ("lat (SIGReg)", "sw_lat_n*"),
    ]),
    ("collapse", "collapse curve / steelman (base)", [
        ("ds", "bsw_ds_n1_s0"),
        ("lat (SIGReg)", "bsw_lat_n1_s0"),
    ]),
    ("addr", "addressing and CLS grid (pilot)", [
        ("ds", "p2_anycell"), ("ds (RoPE)", "p6adr_ds_*"), ("ds (RoPE+CLS)", "p6rc_ds*"),
        ("latent (cell target)", "p3b_mixed_tw2"),
        ("latent (RoPE)", "p6adr_lat_*"),
        ("latent (CLS)", "p6cls_lat_*"),
        ("latent (RoPE+CLS)", "p6rc_lat*"),
    ]),
    ("substrate", "shared-factor substrate", [
        ("ds", "c4_factor_ds_s[0-9]"),
        ("dual", "c4_factor_dual_s[0-9]"),
        # the paper reads the COMPLETED reruns (eval/c4_lat_full_probe.py), not the
        # truncated originals that stopped at ~175/20000 steps
        ("lat (SIGReg)", "c4_factor_lat_s[0-9]_full"),
        ("arbitrary aux.", "c4_factor_arbaux_s[0-9]"),
    ]),
    # Round-8 repair 8: the family carrying Table 8, Figure 6, the permutation p and the oracle
    # rho was absent from this audit entirely -- the substrate globs above match K=4 only, so the
    # other 40 runs of the sweep were never budget-checked and never appeared in the hparams
    # table, while the Setup claimed a committed config for every run.
    ("boundary", "compressibility sweep over intrinsic rank K", [
        (f"ds (K={k})", f"c4_factor_ds_K{k}_s[0-9]") for k in (1, 2, 6, 8, 12)
    ] + [
        (f"dual (K={k})", f"c4_factor_dual_K{k}_s[0-9]") for k in (1, 2, 6, 8, 12)
    ]),
    ("nonlinear", "nonlinear matched rerun (L4)", [
        ("ds", "matched_L4_ds"),
        ("dual", "matched_L4_dual"),
        ("lat (SIGReg)", "matched_L4_lat"),
    ]),
]

# Round 4: four of six reviewers found that the masking policy and curriculum differ across
# arms while this audit reported them "matched" -- because it only looked at three knobs.
# The policy is not a scalar: a ds run stores `policy`, a latent run stores `policy_main`
# plus `curriculum_frac`, so normalise both into one comparable string before diffing.
KNOBS = ("batch_size", "lr", "steps", "masking")



def masking_of(c):
    """One comparable string for what an arm is actually asked to reconstruct.

    train/train.py feeds `policy` every step. train/train_jepa.py feeds y_only for the first
    `curriculum_frac` of steps and `policy_main` after, and "mixed" itself redraws
    iid / column-block / rectangular-block per batch (train/data.py). Two arms can therefore
    agree on batch, lr and steps and still be trained on different masks.
    """
    main = c.get("policy_main") or c.get("policy") or "y_only"
    frac = c.get("curriculum_frac", 0.0) if "policy_main" in c else 0.0
    return f"{main}" if not frac else f"y_only({frac:g})->{main}"


def is_latent(c):
    return any(k in c for k in ("mode", "cls_target", "lambda_ppd", "predictor"))


def params(c):
    """Encoder params (what we probe) and total trainable params (what we train)."""
    kw = dict(emb=c.get("emb", 96), heads=c.get("heads", 4), mlp=c.get("mlp", 192),
              layers=c.get("layers", 3), n_bins=c.get("n_bins", 32))
    n_cls, rs = c.get("n_cls", 0), c.get("r_scheme", "resample")
    if not is_latent(c):
        m = CellPFN(**kw, n_cls=n_cls, r_scheme=rs)
        n = sum(p.numel() for p in m.parameters())
        return n, n
    m = JEPA(**kw, n_reg_tokens=c.get("n_reg_tokens", 0), ema=c.get("ema", 0.996),
             lambda_ppd=c.get("lambda_ppd", 0.0), predictor=c.get("predictor", "mlp"),
             mode=c.get("mode", "ema"), lambda_sig=c.get("lambda_sig", 0.05),
             sig_proj=c.get("sig_proj", 128), r_invariant=c.get("r_invariant", False),
             n_cls=n_cls, cls_target=c.get("cls_target", False), r_scheme=rs)
    return (sum(p.numel() for p in m.online.parameters()),
            sum(p.numel() for p in m.parameters() if p.requires_grad))


def last_step(name):
    f = RUNS / name / "metrics.jsonl"
    if not f.exists():
        return None
    tail = None
    for line in f.read_text().splitlines():
        if line.strip():
            tail = line
    try:
        return json.loads(tail).get("step")
    except Exception:
        return None


def collect(glob):
    """One arm = >=1 run sharing a recipe. Records the knobs and flags disagreement.

    Prefer runs/<name>/config.yaml (RESOLVED: defaults + CLI overrides). Where the box was
    destroyed and the run dir is gone, fall back to the committed configs/<name>.yaml and
    say so, since that file can omit a default the run actually used.
    """
    names = sorted(p.name for p in RUNS.glob(glob) if (p / "config.yaml").exists())
    if names:
        cfgs = [yaml.safe_load((RUNS / n / "config.yaml").read_text()) for n in names]
        source = "run"
    else:
        fallback = sorted(ROOT.glob(f"configs/{glob}.yaml"))
        if fallback:
            names = [p.stem for p in fallback]
            cfgs = [yaml.safe_load(p.read_text()) for p in fallback]
            source = "config"
        else:
            # Last resort: the resolved config torch.save'd inside the checkpoint. This is
            # what a steelman checkpoint lifted out of another run carries -- it has no run
            # dir and no config file of its own, but the cfg it was trained under travels
            # with the weights. Recorded as source "ckpt" so the appendix can say so.
            ck = sorted(RUNS.glob(f"{glob}/ckpt.pt"))
            if not ck:
                return None
            names = [p.parent.name for p in ck]
            cfgs = [torch.load(p, map_location="cpu", weights_only=False)["cfg"] for p in ck]
            source = "ckpt"
    rec = {"runs": names, "n_runs": len(names), "source": source}
    for c in cfgs:
        c["masking"] = masking_of(c)
    for k in KNOBS:
        vals = sorted({c.get(k) for c in cfgs}, key=str)
        rec[k] = vals[0] if len(vals) == 1 else vals
    enc, tot = params(cfgs[0])
    rec["params_encoder"], rec["params_trainable"] = enc, tot
    steps_done = [s for s in (last_step(n) for n in names) if s is not None]
    rec["last_step_min"] = min(steps_done) if steps_done else None
    rec["last_step_max"] = max(steps_done) if steps_done else None
    for k in ("r_scheme", "cls_target", "mode", "lambda_ppd", "lambda_sig", "n_cls"):
        if k in cfgs[0]:
            rec[k] = cfgs[0][k]
    return rec


def main():
    out = {}
    for gid, label, arms in GROUPS:
        g = {"label": label, "arms": {}}
        for arm_label, glob in arms:
            r = collect(glob)
            if r:
                g["arms"][arm_label] = r
        ref = g["arms"].get("ds")                # DERIVE the mismatch, do not assert it
        for arm_label, r in g["arms"].items():
            twin = ref
            if twin is None:
                # Round-8 repair 8: the sweep group has one ds arm per rank rather than a single
                # "ds", so a group-level reference finds nothing. Pair each arm with the ds arm at
                # its own K instead of dropping the whole family out of the audit.
                m = re.search(r"\((K=\d+)\)", arm_label)
                twin = g["arms"].get(f"ds ({m.group(1)})") if m else None
            r["differs_from_ds"] = (sorted(k for k in KNOBS if r.get(k) != twin.get(k))
                                    if twin else None)
        out[gid] = g
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT)
    for gid, g in out.items():
        for a, r in g["arms"].items():
            d = ",".join(r.get("differs_from_ds", [])) or "-"
            print(f"  {gid:10s} {a:22s} b={r['batch_size']} lr={r['lr']} "
                  f"steps={r['steps']} enc={r['params_encoder']/1e6:.2f}M differs:{d}")


if __name__ == "__main__":
    main()
