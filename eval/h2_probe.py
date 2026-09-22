"""Probe the H2 (EMA-target, no-head) arms with the mechanism-table recipe (§sec:hypotheses H2).

Same probes as results_probe_seeds.json (probe_base: c5_scores edge/family + c4_mse_f over
the sigma sweep, SCM prior, paired eval tables), applied to each completed base_ema_nohead_s*
run found locally. The encoder under test is JEPA.encoder() (EMA arm -> target copy),
matching every other latent-arm probe in the paper.

    uv run python -m eval.h2_probe

Writes/updates eval/results_probe_h2.json (one entry per seed; rerun as seeds land).
"""
import json
import time
from pathlib import Path

from eval import probe_base as pb
from eval.latent_probe import load_jepa

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_probe_h2.json"


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = sorted(p.name for p in (ROOT / "runs").glob("base_ema_nohead_s*")
                  if (p / "ckpt.pt").exists() and (p / "metrics.jsonl").exists())
    for name in runs:
        if name in res:
            continue
        t = time.time()
        enc = load_jepa(name).encoder().eval()
        b1 = pb.collect(enc, 1.0)
        edge_auc, fam_acc, fam_maj = pb.c5_scores(b1)
        mse_f = {"1.0": pb.c4_mse_f(b1)}
        for s in pb.SIGMAS[1:]:
            mse_f[str(s)] = pb.c4_mse_f(pb.collect(enc, s))
        res[name] = dict(edge_auc=edge_auc, fam_acc=fam_acc, fam_majority=fam_maj,
                         mse_f=mse_f)
        print(f"{name} edge_auc={edge_auc} fam_acc={fam_acc}(maj {fam_maj}) "
              f"mse_f={mse_f}  [{time.time()-t:.0f}s]", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT.name} ({len(res)} arms)")


if __name__ == "__main__":
    main()
