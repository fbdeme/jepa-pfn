"""Phase 2 evaluation: holdout error vs context size, both baselines x both tasks.

Every (task, context-size) cell rebuilds the same holdout prior and mask rng,
so both models are scored on identical tables and masks.

Run: uv run python -m eval.context_curve   (CPU is fine - forward passes only)
Writes eval/results_p2.json and notebooks/figures/p2_context_curves.png.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.pfn import CellPFN
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
CTX_SIZES = [10, 25, 50, 100]
N_QUERY = 50
N_BATCHES = 20
BATCH = 16
EVAL_SEED = 20_000


def load(run_name):
    ck = torch.load(ROOT / "runs" / run_name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["cfg"]
    m = CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"])
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def score(model, task, ctx):
    prior = SCMPrior(PriorConfig(n_rows=(ctx + N_QUERY, ctx + N_QUERY)),
                     seed=EVAL_SEED + ctx)
    rng = np.random.default_rng(EVAL_SEED + ctx)  # identical masks for all models
    nll, se, n = 0.0, 0.0, 0
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, task, rng, split=ctx)
        logits = model(bt["z"], bt["input_mask"], bt["split"])
        m = bt["target_mask"]
        nll += torch.nn.functional.cross_entropy(
            logits[m], model.to_bins(bt["z"][m]), reduction="sum").item()
        z_sup = bt["z"][m].clamp(model.bin_centers[0], model.bin_centers[-1])
        se += ((model.point_pred(logits)[m] - z_sup) ** 2).sum().item()
        n += int(m.sum())
    return nll / n, se / n


def main():
    models = {name: load(name) for name in ("p2_yonly", "p2_anycell")}
    results = {}
    for task in ("y_only", "any_cell"):
        for ctx in CTX_SIZES:
            for name, model in models.items():
                nll, mse = score(model, task, ctx)
                results[f"{name}|{task}|{ctx}"] = dict(nll=round(nll, 4),
                                                       mse=round(mse, 4))
                print(f"{name:11} task={task:8} ctx={ctx:3}  "
                      f"nll={nll:.4f}  mse={mse:.4f}", flush=True)
    (ROOT / "eval" / "results_p2.json").write_text(json.dumps(results, indent=1))

    # dataviz palette slots 1-2; NLL panels per task, series = trained policy
    colors = {"p2_yonly": "#2a78d6", "p2_anycell": "#008300"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), dpi=150)
    for ax, task in zip(axes, ("y_only", "any_cell")):
        for name in models:
            ys = [results[f"{name}|{task}|{c}"]["nll"] for c in CTX_SIZES]
            ax.plot(CTX_SIZES, ys, "-o", ms=4, color=colors[name],
                    label=f"trained: {name[3:]}")
        ax.set_title(f"eval task: {task}", fontsize=9)
        ax.set_xlabel("context rows")
        ax.set_xscale("log")
        ax.set_xticks(CTX_SIZES, CTX_SIZES)
        ax.grid(color="#eceae2", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("holdout NLL (32-bin)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 2 baselines: ICL adaptation with context size", y=1.04)
    fig.tight_layout()
    fig.savefig(ROOT / "notebooks/figures/p2_context_curves.png", bbox_inches="tight")
    print("wrote eval/results_p2.json + notebooks/figures/p2_context_curves.png")


if __name__ == "__main__":
    main()
