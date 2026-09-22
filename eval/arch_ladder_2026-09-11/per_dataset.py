"""Ladder-2 result analysis (2026-09-13): per-dataset E9 deltas of the surviving arms against the ds baseline, the
training step lag behind ds, and the config confounds between the ds baseline and headenc. Writes per_dataset.json.
Run from the repo root: python eval/arch_ladder_2026-09-11/per_dataset.py"""
import collections, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS, HE, DA = "PFN_tabicl2_ds_s0", "TabularJEPA_v3_tabicl2_headenc_s0", "TabularJEPA_v3_tabicl2_dae_headenc_s0"
NCLS = {r["key"]: r["n_classes"] for r in json.load(open(ROOT / "eval/results_class_count_census.json"))["datasets"]}


def suite(run):
    s = json.load(open(ROOT / f"eval/results_suite_{run}.json"))
    return {k: v[run] for k, v in s.items() if "|" in k and run in v}


def metrics(run):
    return [json.loads(l) for l in open(ROOT / f"runs/{run}/metrics.jsonl") if l.strip()]


def deltas(a, b, task, m):
    return sorted(((round(a[k][m] - b[k][m], 4)), k) for k in a if k in b and k.endswith("|" + task))


def summary(d):
    v = [x for x, _ in d]
    return dict(n=len(v), mean=round(st.mean(v), 4), median=round(st.median(v), 4),
                q10=round(st.quantiles(v, n=10)[0], 4), q90=round(st.quantiles(v, n=10)[-1], 4),
                gain_gt_02=sum(x > .02 for x in v), loss_gt_02=sum(x < -.02 for x in v), loss_gt_05=sum(x < -.05 for x in v))


def strata(d):
    out = {}
    for name, keep in (("ncls=2", lambda c: c == 2), ("ncls=3-5", lambda c: c is not None and 3 <= c <= 5),
                       ("ncls=6-10", lambda c: c is not None and 6 <= c <= 10)):
        v = [x for x, k in d if keep(NCLS.get(k))]
        out[name] = dict(n=len(v), mean=round(st.mean(v), 4), win=sum(x > 0 for x in v), loss=sum(x < 0 for x in v))
    g = collections.defaultdict(list)
    for x, k in d:
        g[k.split("|")[0]].append(x)
    out["by_benchmark"] = {k: dict(n=len(v), mean=round(st.mean(v), 4), win=sum(x > 0 for x in v), loss=sum(x < 0 for x in v))
                           for k, v in g.items()}
    return out


ds, he, da = suite(DS), suite(HE), suite(DA)
dv = [(x["step"], x["val_mse"]) for x in metrics(DS) if "val_mse" in x]


def lag(run):
    rows = []
    for x in metrics(run):
        if "val_mse" in x and x["step"] % 2500 == 0:
            ds_at = dict(dv)[x["step"]]
            reach = next((s for s, m in dv if m <= x["val_mse"]), None)
            rows.append(dict(step=x["step"], val_mse=round(x["val_mse"], 4), ds_val_mse=round(ds_at, 4),
                             gap=round(x["val_mse"] - ds_at, 4), ds_reaches_at=reach))
    return rows


he_ds = deltas(he, ds, "clf", "acc"); da_ds = deltas(da, ds, "clf", "acc")
lose_he = {k for x, k in he_ds if x < 0}; lose_da = {k for x, k in da_ds if x < 0}
out = dict(
    note="E9 per-dataset deltas (run minus ds baseline; clf = acc, reg = r2), training step lag behind ds, config confounds",
    headenc_vs_ds=dict(clf_acc=summary(he_ds), clf_auc=summary(deltas(he, ds, "clf", "auc")), reg_r2=summary(deltas(he, ds, "reg", "r2")),
                       strata=strata(he_ds),
                       worst_clf=[(k.split("|")[1], x, ds[k]["acc"], NCLS.get(k)) for x, k in he_ds[:12]],
                       best_clf=[(k.split("|")[1], x, ds[k]["acc"], NCLS.get(k)) for x, k in he_ds[-6:]],
                       worst_reg=[(k.split("|")[1], x, ds[k]["r2"]) for x, k in deltas(he, ds, "reg", "r2")[:8]]),
    dae_vs_ds=dict(clf_acc=summary(da_ds), reg_r2=summary(deltas(da, ds, "reg", "r2")), strata=strata(da_ds)),
    headenc_vs_dae=dict(clf_acc=summary(deltas(he, da, "clf", "acc"))),
    lose_set_overlap=dict(headenc=len(lose_he), dae=len(lose_da), both=len(lose_he & lose_da)),
    step_lag=dict(headenc=lag(HE), dae=lag(DA)),
    config_confounds=dict(
        note="what differs between PFN_tabicl2_ds_s0 (train.train) and headenc (train.train_jepa) besides the latent term",
        ds=dict(module="train.train", policy="any_cell (iid cells, no curriculum)", ctx_only=False),
        headenc=dict(module="train.train_jepa", policy="y_only for the first 20% (4k steps), then mixed = iid / col_block / block",
                     ctx_only=True, note="hidden cells are not keys in feature attention; the ds encoder attends to them")),
)
json.dump(out, open(Path(__file__).with_name("per_dataset.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("lose_set_overlap",)}), "| headenc clf", out["headenc_vs_ds"]["clf_acc"], "| strata", out["headenc_vs_ds"]["strata"])
