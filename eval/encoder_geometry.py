"""Encoder-output geometry (erank, dim_std, dim_std_min, within-table variance fraction) for any checkpoint - a bare CellPFN
(data-space arm) or a JEPA ckpt (its EMA target encoder) - on masked cells of the run's own prior (E10 secondary read; the
same statistic train_jepa logs as val_tgt, but computed by one tool for both arms so the numbers are comparable).

Run: uv run python -m eval.encoder_geometry RUN_OR_CKPT [...]   -> eval/results_encoder_geometry.json (merged)
     e.g. runs/v4_ds_dspre_lam0_lr5e-4_mixed_bar32_40k_s0/ckpt_20000.pt  or  a run name (runs/<name>/ckpt.pt)"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from model.jepa import collapse_stats                          # noqa: E402
from model.pfn import CellPFN                                  # noqa: E402
from train.data import make_batch                              # noqa: E402
from train.train import _make_prior                            # noqa: E402

OUT = ROOT / "eval/results_encoder_geometry.json"
BATCHES, BATCH, ROWS, POLICY = 8, 8, 256, "any_cell"


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["cfg"]; sd = ck["model"]
    for prefix in ("target.", "online.", ""):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            break
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"], n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample",
                n_classes=c.get("n_classes") or 0)
    m.load_state_dict(sub, strict=True); m.eval()
    return m, c, prefix or "cellpfn"


@torch.no_grad()
def geometry(path):
    m, cfg, kind = load(path)
    prior = _make_prior({**cfg, "prior": {**cfg.get("prior", {}), "n_rows": [ROWS, ROWS]}}, 10_000 + cfg["seed"], None)
    rng = np.random.default_rng(10_000 + cfg["seed"])
    feats, tids = [], []
    for bi in range(BATCHES):
        bt = make_batch(prior, BATCH, POLICY, rng)
        h = m.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        k = getattr(m, "n_cls", 0) or 0
        h = h[:, :, k:] if k else h
        tm = bt["target_mask"]
        feats.append(h[tm]); tids.append((bi * BATCH + torch.arange(tm.shape[0]))[:, None, None].expand(tm.shape)[tm])
    v = torch.cat(feats); t = torch.cat(tids)
    st = collapse_stats(v, max_n=4096, tids=t)
    return dict(kind=kind, seed=cfg["seed"], steps=cfg["steps"], n_cells=int(v.shape[0]), **st)


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arg in sys.argv[1:]:
        path = Path(arg) if arg.endswith(".pt") else ROOT / "runs" / arg / "ckpt.pt"
        key = str(path.relative_to(ROOT)) if path.is_absolute() and str(path).startswith(str(ROOT)) else str(path)
        res[key] = geometry(path)
        print(key, {k: res[key][k] for k in ("erank", "dim_std", "dim_std_min", "within_table_var_frac") if k in res[key]}, flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), len(res), "entries")


if __name__ == "__main__":
    main()
