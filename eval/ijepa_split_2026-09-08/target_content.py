"""What does the EMA latent target at a hidden cell encode? Variance share explained (held-out R2, linear) by
(a) the column index (one-hot), (b) the cell's observed value z, (c) both, (d) the TRUE f at the cell.
Also the same for the predictor's output (ctx_only arms) or the online encoder's in-filled embedding.
Usage: uv run python target_content.py RUN [RUN ...]   -- real prior of the run itself, any_cell, ctx 128."""
import sys, json, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from eval.latent_probe import load_jepa, load_cfg, probe_prior, truth_cells
from train.data import make_batch
import torch.nn.functional as F
SEED, CTX, NB, B = 30_000, 128, 8, 8

def r2(X, Y):
    n = len(Y) // 2
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = X[:n]; W = np.linalg.solve(A.T @ A + 1e-3 * len(A) * np.eye(A.shape[1]), A.T @ Y[:n])   # ridge: the 256-wide reverse fits blow up under plain lstsq
    res = ((Y[n:] - X[n:] @ W) ** 2).sum(); tot = ((Y[n:] - Y[:n].mean(0)) ** 2).sum()
    return round(float(1 - res / tot), 3)

@torch.no_grad()
def main(run):
    m = load_jepa(run); c = load_cfg(run)
    prior = probe_prior(c, CTX, SEED, n_query=50); rng = np.random.default_rng(SEED)
    tg, pr, col, val, f_, Dmax = [], [], [], [], [], 64
    for _ in range(NB):
        bt = make_batch(prior, B, "any_cell", rng, split=CTX, return_truth=True)
        tm = bt["target_mask"]; sel = tm & truth_cells(bt).expand_as(tm)
        if not sel.any(): continue
        z, im, split = bt["z"], bt["input_mask"], bt["split"]
        r_ = m.online.sample_r(z.shape[0], z.shape[2], z.device)
        t = m.target_latent(z, im, split, r_)[sel]; t = F.layer_norm(t, t.shape[-1:])   # the arm's own target (diff-aware)
        p = m.predict_latent(z, im, tm, split)[sel[tm]]
        b, r, d = sel.nonzero(as_tuple=True)
        tg.append(t); pr.append(p); col.append(d); val.append(z[b, r, d]); f_.append(bt["z_f"][b, r, d])
    T, P = torch.cat(tg).numpy(), torch.cat(pr).numpy()
    ci = torch.cat(col).numpy(); oh = np.eye(int(ci.max()) + 1)[ci]; v = torch.cat(val).numpy()[:, None].clip(-3, 3); f = torch.cat(f_).numpy()[:, None].clip(-3, 3)
    print(f"{run}: n={len(T)}  target erank-ish dims={T.shape[1]}")
    for nm, Y in (("EMA target", T), ("prediction", P)):
        print(f"  {nm:11} R2 by column one-hot {r2(oh, Y):.3f} | value z {r2(v, Y):.3f} | column+z {r2(np.concatenate([oh, v], 1), Y):.3f} | true f {r2(f, Y):.3f} | column+f {r2(np.concatenate([oh, f], 1), Y):.3f}")
    print(f"  reverse: value z from target {r2(T, v):.3f}, from prediction {r2(P, v):.3f} | true f from target {r2(T, f):.3f}, from prediction {r2(P, f):.3f}")
    rec = {"n": int(len(T)), "step": int(json.loads(Path(ROOT / "runs" / run / "metrics.jsonl").read_text().splitlines()[-1])["step"]),
           "target": {"col": r2(oh, T), "z": r2(v, T), "f": r2(f, T)}, "pred": {"col": r2(oh, P), "z": r2(v, P), "f": r2(f, P)},
           "reverse": {"z_from_target": r2(T, v), "z_from_pred": r2(P, v), "f_from_target": r2(T, f), "f_from_pred": r2(P, f)}}
    out = Path(os.environ.get("TC_OUT", str(Path(__file__).parent / "target_content.json")))
    d = json.loads(out.read_text()) if out.exists() else {"protocol": {"prior": "run cfg (probe_prior)", "policy": "any_cell", "ctx": CTX, "n_batches": NB, "batch": B, "seed": SEED, "fit": "lstsq on half, R2 on the other half"}, "runs": {}}
    d["runs"][run] = rec; out.write_text(json.dumps(d, indent=1))

for run in sys.argv[1:]:
    main(run)
