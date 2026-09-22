"""Phase 2 baseline training loop.

Run:  uv run python -m train.train configs/p2_yonly.yaml [key=value ...]
e.g.  uv run python -m train.train configs/p2_anycell.yaml steps=50 run_name=smoke

Writes runs/<run_name>/{config.yaml, metrics.jsonl, ckpt.pt}. metrics.jsonl is
line-buffered and flushed per record (remote-monitoring lesson from Phase 0).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
from train.plateau import Plateau
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.pfn import CellPFN
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

DEFAULTS = dict(seed=0, steps=20000, batch_size=32, lr=4e-3, policy="y_only", stop_delta=0.0, stop_patience=80,
                emb=96, heads=4, mlp=192, layers=3, n_bins=32, n_cls=0,
                r_scheme="resample",
                log_every=25, val_every=250, val_batches=8, run_name="run",
                prior={}, dump_path="", feat=5, n_classes=0,
                init_from="", freeze_encoder=False,   # C14 / E6: two-stage = latent pretrain -> data-space head
                reinit_head=False, save_at=[])        # C27 / E10: encoder-only init from a ds ckpt; step-tagged snapshots


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_prior(cfg, seed, split):
    """Paper prior (SCMPrior); a DumpPrior over a pre-generated dump when dump_path is set; or the
    revived TabPFN-v1 SCM/MLP prior (on-the-fly) when prior_source=tabpfn_v1. Backward-compatible:
    with neither set the SCMPrior path is unchanged."""
    if cfg.get("prior_source") == "tabpfn_v1":
        from train.data_tabpfn_v1 import TabPFNv1Prior
        return TabPFNv1Prior(seed=seed, rows=cfg.get("v1_rows", 512), feat=cfg["feat"])
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


@torch.no_grad()
def validate(model, cfg, device):
    """Deterministic holdout: fresh prior + mask rng from a fixed seed, so the
    same SCMs/masks are scored every call. Returns NLL and point-pred MSE."""
    val_seed = 10_000 + cfg["seed"]  # disjoint from the training stream
    prior = _make_prior(cfg, val_seed, "val")
    rng = np.random.default_rng(val_seed)
    nll, se, n = 0.0, 0.0, 0
    for _ in range(cfg["val_batches"]):
        bt = make_batch(prior, cfg["batch_size"], cfg["policy"], rng, device)
        if cfg["n_classes"]:   # class head: NLL and error rate at the label cells (se = #errors)
            lg, y = model.cls_logits(model.encode(bt["z"], bt["input_mask"], bt["split"]),
                                     bt["y_ids"], bt["target_mask"], bt["n_classes"])
            nll += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item()
            se += (lg.argmax(-1) != y).sum().item()
            n += int(len(y))
            continue
        logits = model(bt["z"], bt["input_mask"], bt["split"])
        m = bt["target_mask"]
        nll += torch.nn.functional.cross_entropy(
            logits[m], model.to_bins(bt["z"][m]), reduction="sum").item()
        z_sup = bt["z"][m].clamp(model.bin_centers[0], model.bin_centers[-1])
        se += ((model.point_pred(logits)[m] - z_sup) ** 2).sum().item()  # on head support
        n += int(m.sum())
    return nll / n, se / n


def init_encoder(model, src, freeze=False):
    """C14 (E6): load a JEPA checkpoint's EMA TARGET encoder (the representation every probe reads) into this
    CellPFN; the head it carries was never trained, so it starts random either way. src = a run name (runs/<src>/ckpt.pt)
    or a .pt path. freeze=True leaves only the head(s) trainable. Returns (n_loaded_tensors, n_trainable_params)."""
    path = Path(src) if str(src).endswith(".pt") else Path(__file__).parents[1] / "runs" / src / "ckpt.pt"
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    for prefix in ("target.", "online.", ""):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            break
    model.load_state_dict(sub, strict=True)          # same architecture or fail loudly (no silent partial init)
    if freeze:
        for n, q in model.named_parameters():
            q.requires_grad_(n.startswith("head") or n.startswith("cls_head"))
    return len(sub), sum(q.numel() for q in model.parameters() if q.requires_grad)


def main():
    cfg_path, *overrides = sys.argv[1:]
    cfg = {**DEFAULTS, **yaml.safe_load(open(cfg_path))}
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(__file__).parents[1] / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    metrics = open(run_dir / "metrics.jsonl", "a", buffering=1)

    prior = _make_prior(cfg, cfg["seed"], "train")
    rng = np.random.default_rng(cfg["seed"])
    model = CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"],
                    cfg["n_bins"], n_cls=cfg["n_cls"],
                    r_scheme=cfg["r_scheme"], n_classes=cfg["n_classes"]).to(device)
    if cfg["init_from"]:
        n_t, n_tr = init_encoder(model, cfg["init_from"], freeze=cfg["freeze_encoder"])
        if cfg.get("reinit_head"):                    # E10: a ds checkpoint carries a trained head; compare encoders only
            n_re = 0
            for name in ("head", "cls_head"):
                for m in getattr(model, name, torch.nn.Identity()).modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters(); n_re += 1
            print(f"reinit_head: {n_re} layers re-initialised", flush=True)
        print(f"init_from={cfg['init_from']} tensors={n_t} freeze_encoder={cfg['freeze_encoder']} trainable={n_tr:,}", flush=True)
    bad = [s for s in (cfg.get("save_at") or []) if s % cfg["val_every"] or s > cfg["steps"]]
    assert not bad, f"save_at steps must be validation steps <= steps: {bad}"
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = schedulefree.AdamWScheduleFree(trainable, lr=cfg["lr"], weight_decay=0.0)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} trainable={sum(p.numel() for p in trainable):,}",
          flush=True)

    model.train()
    opt.train()
    t0 = time.time()
    stop = Plateau(cfg["stop_delta"], cfg["stop_patience"])   # open-horizon runs: stop_delta > 0
    for step in range(1, cfg["steps"] + 1):
        bt = make_batch(prior, cfg["batch_size"], cfg["policy"], rng, device)
        if cfg["n_classes"]:   # classification-only head: class CE at the label cell (policy y_target)
            loss = model.loss_cls(model.encode(bt["z"], bt["input_mask"], bt["split"]),
                                  bt["y_ids"], bt["target_mask"], bt["n_classes"])
        else:
            logits = model(bt["z"], bt["input_mask"], bt["split"])
            loss = model.loss(logits, bt["z"], bt["target_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad()

        if step % cfg["log_every"] == 0:
            rec = dict(step=step, loss=round(loss.item(), 4),
                       sec=round(time.time() - t0, 1))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if step % cfg["val_every"] == 0:
            model.eval()
            opt.eval()
            vnll, vmse = validate(model, cfg, device)
            rec = dict(step=step, val_nll=round(vnll, 4), **({"val_err": round(vmse, 4)} if cfg["n_classes"] else {"val_mse": round(vmse, 4)}))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
            if step in set(cfg.get("save_at") or []):   # E10 (C27): pretraining-budget snapshots (val steps only)
                torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / f"ckpt_{step}.pt")
            model.train()
            opt.train()
            if stop.step(vmse):
                rec = dict(step=step, stop="plateau", best=round(stop.best, 4), patience_vals=cfg["stop_patience"], delta=cfg["stop_delta"])
                metrics.write(json.dumps(rec) + "\n"); print(rec, flush=True)
                break

    opt.eval()  # schedulefree: fold averaged weights before saving
    torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
    print(f"saved {run_dir}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
