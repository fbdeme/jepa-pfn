"""Loop-vs-prior diagnostic training: OUR CellPFN + OUR loop, but data from the TabPFN-prior
dump instead of SCMPrior. Identical to train.train except the prior source, so dump_ds vs
base_ds is a PURE prior swap (same architecture, loop, head, eval). If dump_ds becomes strong
on real benchmarks, our stack is fine and the base_ds weakness is the prior, not the pipeline.

Run:  uv run --with h5py python -m train.train_dump configs/dump_ds_s1.yaml
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.pfn import CellPFN
from train.data import make_batch
from train.data_dump import DumpPrior

DEFAULTS = dict(seed=0, steps=20000, batch_size=8, lr=1e-3, policy="any_cell",
                emb=256, heads=8, mlp=1024, layers=6, n_bins=32, n_cls=4,
                r_scheme="rope", log_every=25, val_every=250, val_batches=8,
                run_name="dump_ds", dump_path="data/tabpfn_prior_dump.h5", feat=5)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def validate(model, val_prior, cfg, device):
    """Deterministic holdout: reseed the val prior + mask rng each call so the same dump
    tables/masks are scored every time (mirrors train.train.validate)."""
    val_prior.rng = np.random.default_rng(20_000 + cfg["seed"])
    rng = np.random.default_rng(20_000 + cfg["seed"])
    nll, se, n = 0.0, 0.0, 0
    for _ in range(cfg["val_batches"]):
        bt = make_batch(val_prior, cfg["batch_size"], cfg["policy"], rng, device)
        logits = model(bt["z"], bt["input_mask"], bt["split"])
        m = bt["target_mask"]
        nll += torch.nn.functional.cross_entropy(
            logits[m], model.to_bins(bt["z"][m]), reduction="sum").item()
        z_sup = bt["z"][m].clamp(model.bin_centers[0], model.bin_centers[-1])
        se += ((model.point_pred(logits)[m] - z_sup) ** 2).sum().item()
        n += int(m.sum())
    return nll / n, se / n


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

    train_prior = DumpPrior(cfg["dump_path"], seed=cfg["seed"], split="train", feat=cfg["feat"])
    val_prior = DumpPrior(cfg["dump_path"], seed=10_000 + cfg["seed"], split="val", feat=cfg["feat"])
    rng = np.random.default_rng(cfg["seed"])
    model = CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"],
                    cfg["n_bins"], n_cls=cfg["n_cls"], r_scheme=cfg["r_scheme"]).to(device)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"train_tables={len(train_prior.idx)}", flush=True)

    model.train()
    opt.train()
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        bt = make_batch(train_prior, cfg["batch_size"], cfg["policy"], rng, device)
        logits = model(bt["z"], bt["input_mask"], bt["split"])
        loss = model.loss(logits, bt["z"], bt["target_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % cfg["log_every"] == 0:
            rec = dict(step=step, loss=round(loss.item(), 4), sec=round(time.time() - t0, 1))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if step % cfg["val_every"] == 0:
            model.eval()
            opt.eval()
            vnll, vmse = validate(model, val_prior, cfg, device)
            rec = dict(step=step, val_nll=round(vnll, 4), val_mse=round(vmse, 4))
            metrics.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
            model.train()
            opt.train()

    opt.eval()
    torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")
    print(f"saved {run_dir}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
