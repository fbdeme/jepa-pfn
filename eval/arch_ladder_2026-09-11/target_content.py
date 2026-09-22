"""Target -> value decodability of the ladder-2 arms (docs/prior_v2_plan.md 5.2, the dae judge): held-out ridge R2 of a hidden
cell's value z from the arm's OWN latent target at that cell, and from the predictor's output there. Same protocol as
eval/ijepa_split_2026-09-08/target_content.py (the EMA record .93-.95: ctx 128, any_cell, 8 batches x 8 tables of the run's
prior, ridge on one half, R2 on the other) minus the SCM truth columns (the TabICL prior carries none).
Target per mode, exactly as the loss sees it: ema = target_latent (diff-aware) + LayerNorm; dae = y-encoder on the noised
full table + LayerNorm (also the noise-free encoding); sigreg = projector(online encoder on the full table), no LayerNorm.
Usage: uv run python eval/arch_ladder_2026-09-11/target_content.py RUN [RUN ...]   -> target_content.json in this directory."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from eval.latent_probe import load_jepa, load_cfg, probe_prior
from model.jepa import collapse_stats
from train.data import make_batch
SEED, CTX, NB, B = 30_000, 128, 8, 8


def r2(X, Y):
    n = len(Y) // 2
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = X[:n]; W = np.linalg.solve(A.T @ A + 1e-3 * len(A) * np.eye(A.shape[1]), A.T @ Y[:n])   # ridge, as the 2026-09-08 record
    res = ((Y[n:] - X[n:] @ W) ** 2).sum(); tot = ((Y[n:] - Y[:n].mean(0)) ** 2).sum()
    return round(float(1 - res / tot), 3)


@torch.no_grad()
def targets(m, z, im, tm, split):
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device); no = torch.zeros_like(im)
    ln = lambda t: F.layer_norm(t, t.shape[-1:])
    if m.mode == "ema":
        return {"target": ln(m.target_latent(z, im, split, r)[tm])}
    if m.mode == "dae":
        return {"target": ln(m.yenc.encode(z + m.dae_sigma * torch.randn_like(z), no, split, r=r)[tm]),
                "target_noise_free": ln(m.yenc.encode(z, no, split, r=r)[tm])}
    return {"target": m.projector(m.online.encode(z, no, split, r=r))[tm]}


@torch.no_grad()
def main(run):
    m = load_jepa(run); c = load_cfg(run)
    prior = probe_prior(c, CTX, SEED, n_query=50); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    T, P, V, C = {}, [], [], []
    for _ in range(NB):
        bt = make_batch(prior, B, "any_cell", rng, split=CTX)
        z, im, tm, split = bt["z"], bt["input_mask"], bt["target_mask"], bt["split"]
        if not tm.any():
            continue
        for k, v in targets(m, z, im, tm, split).items():
            T.setdefault(k, []).append(v)
        P.append(m.predict_latent(z, im, tm, split))
        b, rr, d = tm.nonzero(as_tuple=True); V.append(z[b, rr, d]); C.append(d)
    v = torch.cat(V).numpy()[:, None].clip(-3, 3); ci = torch.cat(C).numpy(); oh = np.eye(int(ci.max()) + 1)[ci]
    P = torch.cat(P)
    rec = {"n": int(len(v)), "mode": m.mode, "step": int(json.loads(Path(ROOT / "runs" / run / "metrics.jsonl").read_text().splitlines()[-1])["step"]),
           "pred": {"col": r2(oh, P.numpy()), "z": r2(v, P.numpy()), "z_from_pred": r2(P.numpy(), v), "erank": collapse_stats(P)["erank"]}}
    for k, t in T.items():
        t = torch.cat(t)
        rec[k] = {"col": r2(oh, t.numpy()), "z": r2(v, t.numpy()), "z_from_target": r2(t.numpy(), v), "erank": collapse_stats(t)["erank"]}
        print(f"{run} {k:17s} n={rec['n']} R2 target<-col {rec[k]['col']:.3f} target<-z {rec[k]['z']:.3f} | z<-target {rec[k]['z_from_target']:.3f} | erank {rec[k]['erank']}")
    print(f"{run} {'prediction':17s} R2 pred<-col {rec['pred']['col']:.3f} pred<-z {rec['pred']['z']:.3f} | z<-pred {rec['pred']['z_from_pred']:.3f} | erank {rec['pred']['erank']}")
    out = Path(__file__).parent / "target_content.json"
    d = json.loads(out.read_text()) if out.exists() else {"protocol": {"prior": "run cfg (probe_prior, rows ctx+50)", "policy": "any_cell", "ctx": CTX, "n_batches": NB, "batch": B, "seed": SEED,
                                                                        "fit": "ridge on half, R2 on the other half; EMA record for comparison = eval/ijepa_split_2026-09-08 (.93-.95 value-decodable)"}, "runs": {}}
    d["runs"][run] = rec; out.write_text(json.dumps(d, indent=1) + "\n")


if __name__ == "__main__":   # the TabICL prior's spawn workers re-import this module
    for run in sys.argv[1:]:
        main(run)
