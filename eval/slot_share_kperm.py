"""How much the address-share metric depends on its own protocol constant, measured.

The paper says the absolute slot share "is not protocol-free: it moves with k", and until now
that sentence was backed by three numbers hand-typed into a prose field of
`results_slot_share_all.json` and regex-scraped back out by the table builder. Round 9's artifact
audit found them: the only source-backed-numbers violation in the paper, and one of the three
disagreed with the same file's own measurement of the same arm.

So measure it. Same arms, same batches, same paired re-seeding as eval/slot_share_all.py; the
only thing that varies is `slot_share`'s `k_perm`. What the paper's claim needs is whether the
>0.5 classification is stable across k while the third decimal is not, and both are derived here
rather than asserted.

Run: uv run python -m eval.slot_share_kperm      (CPU)
Writes eval/results_slot_share_kperm.json
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.address_leak import slot_share                       # noqa: E402
from eval.slot_share_all import (ARMS, BATCH, EVAL_SEED, N_BATCHES,   # noqa: E402
                                 load_any)
from prior.scm import PriorConfig, SCMPrior                            # noqa: E402
from train.data import make_batch                                      # noqa: E402
import numpy as np                                                    # noqa: E402

OUT = ROOT / "eval/results_slot_share_kperm.json"
KS = (4, 8, 16)
DOMINATED = 0.5          # the classification threshold the paper reads


@torch.no_grad()
def measure(enc, batches, k):
    """Paired across arms and across k by re-seeding, exactly as slot_share_all.measure does.

    This does NOT reproduce slot_share_all's own numbers to the last digit, and the reason is
    worth recording: that loop interleaves an `enc.encode` + `collapse_stats` call between
    successive `slot_share` calls, which advances the global RNG, so its permutation sequence
    differs from a loop that only permutes. On the arm the paper quotes, k=8 here reads .8489
    against .8481 there -- a difference of .0008, about one percent of that arm's across-batch
    SD. The claim this script supports is about the SPREAD across k, which is two orders of
    magnitude larger, so the discrepancy does not touch it.
    """
    torch.manual_seed(EVAL_SEED)
    return round(st.mean(slot_share(enc, b, k_perm=k) for b in batches), 4)


def main():
    torch.manual_seed(EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    prior = SCMPrior(PriorConfig())
    batches = [make_batch(prior, BATCH, "any_cell", rng) for _ in range(N_BATCHES)]
    out = {"protocol": {"source": "eval/slot_share_all.py (arms, batches, paired re-seeding)",
                        "varies": "slot_share k_perm only", "k_values": list(KS),
                        "threshold": DOMINATED},
           "by_arm": {}}
    t0 = time.time()
    for run, scale, label in ARMS:
        if not (ROOT / "runs" / run / "ckpt.pt").exists():
            print(f"  {run:22} MISSING", flush=True)
            continue
        try:
            enc = load_any(run).eval()
        except FileNotFoundError:
            print(f"  {run:22} MISSING", flush=True)
            continue
        vals = {str(k): measure(enc, batches, k) for k in KS}
        v = list(vals.values())
        out["by_arm"][run] = {"scale": scale, "label": label, "by_k": vals,
                              "range": round(max(v) - min(v), 4),
                              "dominated_by_k": {k: bool(x > DOMINATED) for k, x in vals.items()},
                              "classification_stable": len({x > DOMINATED for x in v}) == 1}
        print(f"  {run:22} " + "  ".join(f"k={k}:{vals[str(k)]:.4f}" for k in KS)
              + f"   range {out['by_arm'][run]['range']:.4f}"
              + ("" if out["by_arm"][run]["classification_stable"] else "   CLASS FLIPS")
              + f"   [{time.time() - t0:.0f}s]", flush=True)
    arms = out["by_arm"]
    dom = [a for a, r in arms.items() if any(r["dominated_by_k"].values())]
    out["summary"] = {
        "n_arms": len(arms),
        "n_classification_stable": sum(1 for r in arms.values() if r["classification_stable"]),
        "all_classifications_stable": all(r["classification_stable"] for r in arms.values()),
        "max_range": max(r["range"] for r in arms.values()) if arms else None,
        "max_range_arm": max(arms, key=lambda a: arms[a]["range"]) if arms else None,
        "n_dominated": len(dom),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
