"""Same-metric convergence evidence for the factor substrate, from frozen checkpoints.

Round 12 (perspective + devil's advocate, convergent): the substrate's convergence claim
compared the two arms on DISJOINT logged metrics -- the data-space runs log `val_mse`, the
dual runs log `pred_loss`/`val_jepa` and no value MSE at all. "The latent term restores
convergence" therefore compared one arm's held-out value MSE against another arm's
admission by the masked probe. This computes the missing half: every factor-substrate arm
that carries a value head is scored with EXACTLY train.validate()'s protocol -- the same
deterministic val stream (seed 10000+cfg.seed), the same prior maker, the same point_pred
on head support -- so the two arms finally share a yardstick.

Each arm is scored under BOTH masking policies (its own and the counterpart's), because
the policies differ across arms and a single-policy number would smuggle the masking
confound back in. The data-space arms are re-scored through the same code path as a
self-check: their recomputed val_mse must reproduce the last value in their own
metrics.jsonl.

Run: uv run python -m eval.factor_value_mse           (CPU, frozen ckpts)
Writes eval/results_factor_value_mse.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import load_cellpfn                      # noqa: E402
from model.jepa import JEPA                                         # noqa: E402
from train.data import make_batch                                   # noqa: E402
from train.train import _make_prior                                 # noqa: E402

OUT = ROOT / "eval/results_factor_value_mse.json"
POLICIES = ("any_cell", "mixed")
KS = {"": 4, "_K1": 1, "_K2": 2, "_K6": 6, "_K8": 8, "_K12": 12}


def _load_dual(run):
    ck = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
             c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
             c.get("predictor", "mlp"), c.get("mode", "ema"),
             c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
             c.get("r_invariant", False), c.get("n_cls", 0),
             c.get("cls_target", False), c.get("r_scheme", "resample"), n_classes=c.get("n_classes", 0))
    m.load_state_dict(ck["model"])
    return m.eval(), c


@torch.no_grad()
def val_mse(kind, run, policy):
    """train.validate()'s protocol verbatim, with the model's value path made explicit."""
    if kind == "dual":
        jepa, cfg = _load_dual(run)
        enc, head_owner = jepa.online, jepa.online
        k = cfg.get("n_cls", 0) or 0
    else:
        enc = load_cellpfn(run)
        head_owner, k = enc, getattr(enc, "n_cls", 0) or 0
        cfg = torch.load(ROOT / "runs" / run / "ckpt.pt", map_location="cpu",
                         weights_only=False)["cfg"]
    val_seed = 10_000 + cfg["seed"]
    prior = _make_prior(cfg, val_seed, "val")
    rng = np.random.default_rng(val_seed)
    se, n = 0.0, 0
    for _ in range(cfg["val_batches"]):
        bt = make_batch(prior, cfg["batch_size"], policy, rng)
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        logits = head_owner.head(h[:, :, k:] if k else h)
        m = bt["target_mask"]
        z_sup = bt["z"][m].clamp(head_owner.bin_centers[0], head_owner.bin_centers[-1])
        se += ((head_owner.point_pred(logits)[m] - z_sup) ** 2).sum().item()
        n += int(m.sum())
    return round(se / n, 4)


def main():
    out = {"protocol": {"source": "train.train.validate(): same val stream (10000+seed), "
                                  "same prior maker, point_pred on head support; both "
                                  "masking policies scored for every arm",
                        "selfcheck": "ds arms recomputed through this code path must "
                                     "reproduce their own logged val_mse"},
           "runs": {}, "by_k": {}}
    for suf, K in KS.items():
        grp = {"ds": {p: [] for p in POLICIES}, "dual": {p: [] for p in POLICIES}}
        for arm in ("ds", "dual"):
            for s in range(4):
                run = f"c4_factor_{arm}{suf}_s{s}"
                if not (ROOT / "runs" / run / "ckpt.pt").exists():
                    continue
                t = time.time()
                rec = {}
                for p in POLICIES:
                    rec[p] = val_mse(arm, run, p)
                    grp[arm][p].append(rec[p])
                if arm == "ds":     # self-check against the run's own log
                    rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
                    logged = [r["val_mse"] for r in rows if "val_mse" in r]
                    rec["logged_last"] = logged[-1] if logged else None
                out["runs"][run] = rec
                print(f"  {run:24} " + "  ".join(f"{p}={rec[p]:.4f}" for p in POLICIES)
                      + (f"  (logged {rec.get('logged_last')})" if arm == "ds" else "")
                      + f"  [{time.time()-t:.0f}s]", flush=True)
        out["by_k"][str(K)] = {a: {p: round(sum(v) / len(v), 4) for p, v in d.items() if v}
                               for a, d in grp.items() if any(d.values())}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
