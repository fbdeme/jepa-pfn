"""Balanced per-GPU queues for one v4 experiment, from configs/v4_manifest.json (deterministic: longest-processing-time first).

Weights = expected wall time in units of one 20k dual run (E1 measured on A6000: dual 20k ~ 2.0 h, ds 20k ~ 1.15 h):
dual 1.0, ds 0.6, scaled by steps / 20000, frozen-encoder ds x 0.7 (no encoder backward). Reused runs are skipped.

Run: python3 scripts/gen_v4_queues.py E6 --gpus 2 [--box a] [--print-cmds]
      -> one line per GPU: "qN: run run ..." and, with --print-cmds, the remote.sh bg launch lines (+ push line for init_from ckpts)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "configs/v4_manifest.json"


def weight(ax):
    w = (1.0 if ax["arm"] == "dual" else 0.6) * ax["steps"] / 20000
    if ax.get("freeze_encoder"):
        w *= 0.7
    return w


def complete(run, ax):
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return False
    last = [json.loads(l) for l in open(p)][-1:]
    return bool(last) and last[0].get("step", 0) >= ax["steps"]


def _snap(ax):
    s = ax.get("init_from", "")
    return int(s.split("ckpt_")[1].split(".")[0]) if "ckpt_" in s else 0


def chains_dependent(m):
    """E8 / E10: a fine-tune whose init_from points inside another run of the same experiment must run after it on the same
    GPU, so (pretraining run + its fine-tunes, in budget order) is one chain."""
    runs = {r: ax for r, ax in m["runs"].items() if not ax["reused"]}
    parents = sorted(r for r in runs if any(a.get("init_from", "").startswith(f"runs/{r}/") for a in runs.values()))
    out = []
    for pt in parents:
        fts = sorted((r for r, a in runs.items() if a.get("init_from", "").startswith(f"runs/{pt}/")), key=lambda r: _snap(runs[r]))
        out.append(([pt] + fts, weight(runs[pt]) + sum(weight(runs[r]) for r in fts)))
    return out


def queues(eid, n_gpus):
    m = json.loads(MANIFEST.read_text())["experiments"][eid]
    flt = sys.argv[sys.argv.index("--filter") + 1] if "--filter" in sys.argv else ""   # e.g. --filter _60k_ (E7 extension only)
    if eid in ("E8", "E10"):                                    # dependency chains (pretrain -> its fine-tunes), LPT over chains
        units = chains_dependent(m)
    else:
        # a run is queued unless it is a POST run named in reuse_existing or already complete locally (runs/<r>/metrics.jsonl at
        # its configured steps) - so an E7 60k cell shared with E5 (never trained) is queued, the trained 40k cells are not
        units = [([r], weight(ax)) for r, ax in m["runs"].items() if flt in r and not (ax["reused"] and not ax.get("reused_from")) and not complete(r, ax)]
    units.sort(key=lambda x: (-x[1], x[0][0]))                  # LPT: heaviest first, name-stable ties
    qs, load = [[] for _ in range(n_gpus)], [0.0] * n_gpus
    for rs, w in units:
        i = min(range(n_gpus), key=lambda k: (load[k], k))
        qs[i].extend(rs); load[i] += w
    deps = sorted({ax["init_from"] for ax in m["runs"].values() if ax.get("init_from")})
    return qs, load, deps


def main():
    eid = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--gpus") + 1]) if "--gpus" in sys.argv else 1
    box = sys.argv[sys.argv.index("--box") + 1] if "--box" in sys.argv else "BOX"
    qs, load, deps = queues(eid, n)
    for i, q in enumerate(qs):
        print(f"q{i} (load {load[i]:.2f} x 20k-dual): {' '.join(q)}")
    if deps:
        print("init_from checkpoints to push first:", " ".join(deps))
    if "--print-cmds" in sys.argv:
        if deps:
            print(f"BOX={box} scripts/remote.sh push {' '.join(deps)}")
        for i, q in enumerate(qs):
            print(f"BOX={box} scripts/remote.sh bg q{i} env CUDA_VISIBLE_DEVICES={i} bash scripts/box/queue_gpu.sh {' '.join(q)}")


if __name__ == "__main__":
    main()
