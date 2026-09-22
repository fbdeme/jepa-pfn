"""Training-log read-out for the ctx_only smoke: val_jepa + representation stats at fixed val steps, per run.
Usage: python ctx_readout.py RUN [RUN ...]   (reads runs/<RUN>/metrics.jsonl from the repo root)"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).parents[2]
STEPS = (250, 1000, 2500, 5000, 10000, 15000, 20000)
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in (ROOT / "runs" / run / "metrics.jsonl").read_text().splitlines()]
    val = {r["step"]: r for r in rows if "val_jepa" in r}
    last = max(val) if val else None
    print(f"\n{run}  (last val step {last})")
    print(f"{'step':>6} {'val_jepa':>9} {'tgt_erank':>9} {'tgt_std_min':>11} {'tgt_wtvf':>8} {'pred_erank':>10} {'pred_cos':>8} {'pred_wtvf':>9}")
    for s in STEPS + ((last,) if last and last not in STEPS else ()):
        r = val.get(s)
        if not r: continue
        t, p = r["val_tgt"], r["val_pred"]
        print(f"{s:>6} {r['val_jepa']:>9.4f} {t['erank']:>9.1f} {t['dim_std_min']:>11.4f} {str(t.get('within_table_var_frac')):>8} {p['erank']:>10.1f} {p['cos']:>8.3f} {str(p.get('within_table_var_frac')):>9}")
