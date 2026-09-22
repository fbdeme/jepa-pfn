"""Phase 3 JEPA training loop.

Run:  uv run python -m train.train_jepa configs/p3_jepa.yaml [key=value ...]

Masking curriculum (SSOT section 3): y_only for the first curriculum_frac of
steps (or a fixed curriculum_steps, for open-horizon runs), then any_cell. Collapse
stats (target & prediction) are logged with every metrics record - always-on per
SSOT section 7.

Long runs (2026-09-08): the optimizer is schedule-free, so every val-step checkpoint
is an "anytime" model and the horizon need not be fixed. resume.pt (model + optimizer
+ every RNG stream + step) is written at every val step; resume=PATH restarts from it
bit-for-bit (train/test_resume.py), steps may be raised on resume. At save_at steps a
step-tagged copy is kept and, with hf_repo set, uploaded in the background
(train/hf_upload.py; token from HF_TOKEN env only).
"""

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
from train.plateau import Plateau
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.jepa import JEPA, collapse_stats
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

DEFAULTS = dict(seed=0, steps=20000, batch_size=32, lr=2e-3, curriculum_frac=0.2,
                emb=96, heads=4, mlp=192, layers=3, n_bins=32, n_reg_tokens=0,
                ema=0.996, lambda_ppd=0.0, predictor="mlp", policy_main="any_cell",
                mode="ema", lambda_sig=0.05, sig_proj=128, r_invariant=False,
                n_cls=0, cls_target=False, r_scheme="resample", shuffle_target=False,
                log_every=25, val_every=250, val_batches=16, run_name="run", prior={},
                dump_path="", feat=5, val_policy="any_cell", n_classes=0, ctx_only=False, fresh_target_r=False, diff_target=False,
                head_on_pred=False, lambda_jepa=1.0, dae_sigma=0.5, sigreg_detach=True, sigreg_additive=False, ckpt_pass2=False,
                curriculum_steps=0, resume="", hf_repo="", stop_delta=0.0, stop_patience=80)   # curriculum_steps > 0 overrides curriculum_frac; stop_delta > 0 = open horizon


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_prior(cfg, seed, split):
    """Paper prior (SCMPrior), or a DumpPrior over a pre-generated dump when dump_path is set.
    Backward-compatible: with dump_path unset the SCMPrior path is unchanged."""
    if cfg.get("dump_path"):
        from train.data_dump import DumpPrior
        return DumpPrior(cfg["dump_path"], seed=seed, split=split, feat=cfg["feat"])
    if cfg.get("prior_source") == "real":              # real-data-matched prior (docs/real_prior_plan.md)
        from prior.real import RealConfig, RealPrior
        return RealPrior(RealConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("prior", {}).items()}), seed=seed)
    if cfg.get("prior_source") == "factor":            # C4 latent-favorable substrate
        from prior.factor import FactorConfig, FactorPrior
        return FactorPrior(FactorConfig(**cfg["prior"]), seed=seed)
    if cfg.get("prior_source") == "tabicl":            # TabICL v1/v2 engine, transfer track (docs/prior_v2_plan.md)
        from prior.tabicl import TabICLConfig, TabICLPrior
        return TabICLPrior(TabICLConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("prior", {}).items()}), seed=seed)
    return SCMPrior(PriorConfig(**cfg["prior"]), seed=seed)


def _rng_state(rng, prior):
    """Every stream the training step draws from: batch masks (rng), tables (prior.rng), column codes / sigreg (torch)."""
    return dict(np=rng.bit_generator.state, prior=prior.rng.bit_generator.state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(st, rng, prior):
    rng.bit_generator.state = st["np"]
    prior.rng.bit_generator.state = st["prior"]
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


@torch.no_grad()
def validate(model, cfg, device):
    """Deterministic holdout: JEPA loss, total loss (same formula as the train step), collapse stats with
    table ids (within_table_var_frac, C1), and for a value-head dual the head's own bar CE (val_ppd, C12)."""
    val_seed = 10_000 + cfg["seed"]
    prior = _make_prior(cfg, val_seed, "val")
    rng = np.random.default_rng(val_seed)
    tot, vtot, n = 0.0, 0.0, 0
    preds, tgts, tids = [], [], []
    cnll, cerr, cn = 0.0, 0, 0
    ppd, pn, mse = 0.0, 0, 0.0
    enc = model.online
    for bi in range(cfg["val_batches"]):
        bt = make_batch(prior, cfg["batch_size"], "y_target" if cfg["n_classes"] else cfg["val_policy"], rng, device)
        loss, lj, pred, tgt = model(bt["z"], bt["input_mask"], bt["target_mask"], bt["split"],
                                    y_ids=bt.get("y_ids"), n_classes=bt.get("n_classes"))
        tot += lj.item() * len(pred)
        vtot += loss.item() * len(pred)
        n += len(pred)
        preds.append(pred)
        tgts.append(tgt)
        tm = model._cls_mask(bt["target_mask"], enc.n_cls) if model.cls_target else bt["target_mask"]   # same mask forward scored
        tids.append((bi * cfg["batch_size"] + torch.arange(tm.shape[0], device=device))[:, None, None].expand(tm.shape)[tm])
        if cfg["n_classes"]:   # the class head's own held-out NLL / error (comparable with train.train's val_nll / val_err)
            lg, y = enc.cls_logits(enc.encode(bt["z"], bt["input_mask"], bt["split"]), bt["y_ids"], bt["target_mask"], bt["n_classes"])
            cnll += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item(); cerr += (lg.argmax(-1) != y).sum().item(); cn += len(y)
        elif model.lambda_ppd > 0:   # value head's bar CE at the target cells (the ds arm's val metric, on the dual's online encoder)
            m = bt["target_mask"]
            lg = model.value_logits(bt["z"], bt["input_mask"], bt["split"])[m]   # head on the encoder or on the predictor
            ppd += torch.nn.functional.cross_entropy(lg, enc.to_bins(bt["z"][m]), reduction="sum").item(); pn += len(lg)
            z_sup = bt["z"][m].clamp(enc.bin_centers[0], enc.bin_centers[-1])      # the fixed-R value-MSE definition (real_readout)
            mse += ((enc.point_pred(lg) - z_sup) ** 2).sum().item()
    tids = torch.cat(tids)
    out = dict(val_jepa=round(tot / max(n, 1), 4), val_total=round(vtot / max(n, 1), 4),
               val_tgt=collapse_stats(torch.cat(tgts), tids=tids), val_pred=collapse_stats(torch.cat(preds), tids=tids))
    if cfg["n_classes"]:
        out["val_tgt"]["cls_nll"] = round(cnll / max(cn, 1), 4); out["val_tgt"]["cls_err"] = round(cerr / max(cn, 1), 4)
    if pn:
        out["val_ppd"] = round(ppd / pn, 4)
        out["val_mse"] = round(mse / pn, 4)
    return out


def main():
    cfg_path, *overrides = sys.argv[1:]
    cfg = {**DEFAULTS, **yaml.safe_load(open(cfg_path))}
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bad = [s for s in (cfg.get("save_at") or []) if s % cfg["val_every"] or s > cfg["steps"]]
    assert not bad, f"save_at steps must be validation steps <= steps: {bad}"
    run_dir = Path(__file__).parents[1] / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))

    prior = _make_prior(cfg, cfg["seed"], "train")
    rng = np.random.default_rng(cfg["seed"])
    model = JEPA(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"],
                 cfg["n_reg_tokens"], cfg["ema"], cfg["lambda_ppd"],
                 cfg["predictor"], cfg["mode"], cfg["lambda_sig"],
                 cfg["sig_proj"], cfg["r_invariant"],
                 cfg["n_cls"], cfg["cls_target"], cfg["r_scheme"],
                 cfg["shuffle_target"], n_classes=cfg["n_classes"], ctx_only=cfg["ctx_only"],
                 fresh_target_r=cfg["fresh_target_r"], diff_target=cfg["diff_target"],
                 head_on_pred=cfg["head_on_pred"], lambda_jepa=cfg["lambda_jepa"], dae_sigma=cfg["dae_sigma"],
                 sigreg_detach=cfg["sigreg_detach"], sigreg_additive=cfg["sigreg_additive"], ckpt_pass2=cfg["ckpt_pass2"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = schedulefree.AdamWScheduleFree(trainable, lr=cfg["lr"], weight_decay=0.0)
    print(f"device={device} trainable={sum(p.numel() for p in trainable):,}", flush=True)

    start = 0
    if cfg["resume"]:   # state saved in opt.eval() mode at a val step; opt.train() below folds x -> y from the restored z
        st = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=True)
        opt.load_state_dict(st["opt"])
        _set_rng_state(st["rng"], rng, prior)
        start = st["step"]
        mp = run_dir / "metrics.jsonl"
        if mp.exists():   # drop records past the resume point so the log equals an uninterrupted run's
            keep = [l for l in mp.read_text().splitlines() if l and json.loads(l)["step"] <= start]
            mp.write_text("".join(l + "\n" for l in keep))
        print(f"resumed {cfg['resume']} at step {start}", flush=True)
    metrics = open(run_dir / "metrics.jsonl", "a", buffering=1)

    switch = cfg["curriculum_steps"] or int(cfg["steps"] * cfg["curriculum_frac"])
    model.train()
    opt.train()
    t0 = time.time()
    stop = Plateau(cfg["stop_delta"], cfg["stop_patience"])   # open-horizon runs (train/plateau.py); state resets on resume
    for step in range(start + 1, cfg["steps"] + 1):
        policy = "y_target" if cfg["n_classes"] else ("y_only" if step <= switch else cfg["policy_main"])
        bt = make_batch(prior, cfg["batch_size"], policy, rng, device)
        loss, lj, pred, tgt = model(bt["z"], bt["input_mask"], bt["target_mask"],
                                    bt["split"], y_ids=bt.get("y_ids"), n_classes=bt.get("n_classes"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad()
        model.ema_update()

        if step % cfg["log_every"] == 0:
            # sigreg term is recoverable as (loss - (1 - lambda_sig) * pred_loss) / lambda_sig (convex form) or
            # (loss - pred_loss) / lambda_sig (sigreg_additive); watching it fall while pred_loss plateaus is the
            # LeJEPA signature (../EEG-WM-JEPA/docs/supplementary/loss_analysis.md). dae: loss - lambda_jepa * pred_loss
            # (- lambda_ppd * value CE) = the y-encoder's reconstruction CE.
            rec = dict(step=step, loss=round(loss.item(), 4),
                       pred_loss=round(lj.item(), 4), policy=policy,
                       sec=round(time.time() - t0, 1),
                       tgt=collapse_stats(tgt), pred=collapse_stats(pred))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if step % cfg["val_every"] == 0:
            model.eval()
            opt.eval()
            rec = dict(step=step, **validate(model, cfg, device))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=_rng_state(rng, prior), step=step),
                       run_dir / "resume.pt")   # crash safety: at most val_every steps lost
            if cfg.get("save_steps"):   # keep step-tagged ckpts for steelman selection
                torch.save(dict(model=model.state_dict(), cfg=cfg),
                           run_dir / f"ckpt_{step}.pt")
            if step in set(cfg.get("save_at") or []):   # E8 (C25): pretraining-budget snapshots, step must be a val step
                torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / f"ckpt_{step}.pt")
                shutil.copyfile(run_dir / "resume.pt", run_dir / f"resume_{step}.pt")
                if cfg["hf_repo"]:   # background upload; training never waits on the network
                    subprocess.Popen([sys.executable, "-m", "train.hf_upload", str(run_dir), str(step), cfg["hf_repo"]],
                                     cwd=run_dir.parents[1], start_new_session=True, stdin=subprocess.DEVNULL,
                                     stdout=open(run_dir / "hf_upload.log", "a"), stderr=subprocess.STDOUT)
            model.train()
            opt.train()
            v = rec.get("val_mse", rec.get("val_err"))   # None for pure-latent runs (no value metric): rule inactive
            if v is not None and stop.step(v):
                rec = dict(step=step, stop="plateau", best=round(stop.best, 4), patience_vals=cfg["stop_patience"], delta=cfg["stop_delta"])
                metrics.write(json.dumps(rec) + "\n"); print(rec, flush=True)
                break

    opt.eval()
    torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
    print(f"saved {run_dir}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
