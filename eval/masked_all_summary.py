"""Aggregate eval/results_probe_masked_all.json into the numbers the four tables quote.

Nothing downstream reads the per-arm file and averages it by hand; the build scripts read
this. Two things it derives rather than assumes:

  * aliveness, which must clear BOTH null anchors. A collapsed encoder leaves ridge with no
    usable variation, so it predicts the training half's mean and scores exactly
    `eval.probe_masked_all.const_mse` -- computed with no encoder at all. Under `mixed` the
    random-init floor is WORSE than that constant map, so a floor-relative rule would certify
    collapsed arms as alive; but on the factor substrate the ordering reverses and the floor
    sits BELOW DEGENERATE x the constant map, so a constant-map-relative rule admits an
    untrained network. Neither anchor alone is safe. The cutoff is therefore the stricter of
    the two, min(DEGENERATE x const_map, floor), which cannot admit either anchor by
    construction. Making the rule two-sided changes no reported count (checked against the
    one-sided rule on every family); it closes a leak rather than moving a number.
  * the policy comparison. `any_cell` is exactly the policy every data-space arm trains
    under and `mixed` is the latent arms', so neither is a neutral test on its own. Every
    aggregate is emitted under both, and `agrees` records whether the within-family ordering
    survives the swap -- the only form in which the comparison is admissible.

Run: uv run python -m eval.masked_all_summary
Writes eval/results_probe_masked_all_summary.json
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import const_mse          # noqa: E402

SRC = ROOT / "eval/results_probe_masked_all.json"
OUT = ROOT / "eval/results_probe_masked_all_summary.json"
DEGENERATE = 0.98        # fraction of the constant-map score at/above which an arm is one
POLICIES = ("any_cell", "mixed")
EPS = 1e-9


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy)


def t_tail_p(r, n):
    """Exact two-sided Student-t tail for a Pearson r, closed form at df=4. No scipy."""
    assert n - 2 == 4, "closed form is exact for df=4 only; generalize if the sweep changes"
    t2 = r ** 2 * (n - 2) / (1 - r ** 2)
    y = t2 / (t2 + (n - 2))
    return 1.0 - (1.5 * math.sqrt(y) - 0.5 * y * math.sqrt(y))


def agg(xs):
    return {"mean": round(st.mean(xs), 4),
            "sd": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0, "n": len(xs)}


def cell(rec, pol, field="mse_f"):
    return rec[pol]["cell"][field]


def cutoff(const, floor):
    """The stricter of the two null anchors, per policy. See the module docstring."""
    return {p: round(min(DEGENERATE * const[p], floor[p]["mean"]), 4) for p in POLICIES}


def anchors(key, floor=None):
    """{policy: constant-map score} plus the admission cutoff derived from both anchors.

    `floor` is omitted only for families that decide no admission (the addressing grid
    reports gaps, never an alive set); there the constant-map cutoff is reported for
    reference and nothing downstream reads it as a rule.
    """
    c = {p: const_mse(key, p) for p in POLICIES}
    if floor is None:
        return c, {p: round(DEGENERATE * c[p], 4) for p in POLICIES}
    return c, cutoff(c, floor)


def group_stats(arms, recs, pol, cut):
    vals = [cell(recs[a], pol) for a in arms]
    alive = [a for a in arms if cell(recs[a], pol) < cut]
    return {"all_seeds": agg(vals),
            "alive": agg([cell(recs[a], pol) for a in alive]) if alive else None,
            "mse_x_alive": agg([cell(recs[a], pol, "mse_x") for a in alive]) if alive else None,
            "n_alive": f"{len(alive)}/{len(arms)}", "alive_runs": alive}


def by_meta(recs, key):
    out = {}
    for a, r in recs.items():
        out.setdefault(r["meta"][key], []).append(a)
    return out


def main():
    src = json.loads(SRC.read_text())
    out = {"rule": {"degenerate_fraction": DEGENERATE,
                    "anchors": ["eval.probe_masked_all.const_mse -- the score a constant map "
                                "gets on this family and policy, computed without an encoder",
                                "the seeded random-init floor measured in the same family"],
                    "alive": "masked cell f-MSE < min(DEGENERATE x const_map, floor)",
                    "excludes_null_anchor": "by construction: the cutoff is <= either anchor, "
                                            "so neither anchor can satisfy the strict inequality"},
           "policies": list(POLICIES)}

    # ---- recipe grid (tab_recipe_grid / Table tab:deconfound) ----------------------
    if "recipe" in src:
        recs, floor = src["recipe"]["arms"], src["recipe"]["floor"]
        const, cut = anchors("recipe", floor)
        rec = {"floor": floor, "const_map": const, "cutoff": cut, "per_group": {}}
        for g, arms in by_meta(recs, "group").items():
            arms = sorted(arms)
            rec["per_group"][g] = {p: group_stats(arms, recs, p, cut[p]) for p in POLICIES}
            rec["per_group"][g]["runs"] = arms
        ds = rec["per_group"]["ds"]
        for g, r in rec["per_group"].items():
            if g == "ds":
                continue
            sign = []
            for p in POLICIES:
                a, b = r[p]["alive"], ds[p]["alive"]
                sign.append(None if not (a and b) else a["mean"] > b["mean"] + EPS)
            r["ds_ahead_by_policy"] = dict(zip(POLICIES, sign))
            r["agrees"] = sign[0] is not None and sign[0] == sign[1]
        out["recipe"] = rec

    # ---- sigma sweep (tab_sigma) --------------------------------------------------
    if "sigma" in src:
        recs = src["sigma"]["arms"]
        rec = {"by_noise": {}}
        for ns, blk in src["sigma"]["by_noise"].items():
            const, cut = anchors(f"sigma:{ns}", blk["floor"])
            arms = [a for a, r in recs.items() if str(r["meta"]["noise_scale"]) == ns]
            kinds = {k: sorted(a for a in arms if recs[a]["meta"]["kind"] == k)
                     for k in ("ds", "lat")}
            e = {"floor": blk["floor"], "const_map": const, "cutoff": cut, "runs": kinds}
            for p in POLICIES:
                d = agg([cell(recs[a], p) for a in kinds["ds"]])
                l = agg([cell(recs[a], p) for a in kinds["lat"]])
                gaps = [cell(recs[b], p) - cell(recs[a], p)          # seed-paired, lat - ds
                        for a, b in zip(kinds["ds"], kinds["lat"])]
                e[p] = {"ds": d, "lat": l, "gap_mean": round(st.mean(gaps), 4),
                        "gap_half": round((max(gaps) - min(gaps)) / 2, 4),
                        "lat_degenerate": l["mean"] >= cut[p],
                        "ds_degenerate": d["mean"] >= cut[p],
                        "lat_at_floor": l["mean"] >= blk["floor"][p]["mean"],
                        "floor": blk["floor"][p]["mean"]}
            e["agrees"] = (e["any_cell"]["gap_mean"] > 0) == (e["mixed"]["gap_mean"] > 0)
            rec["by_noise"][ns] = e
        out["sigma"] = rec

    # ---- addressing grid (tab_addr_body panel a) -----------------------------------
    if "addr" in src:
        recs = src["addr"]["arms"]
        const, cut = anchors("addr")
        rec = {"const_map": const, "cutoff": cut, "by_scheme": {}}
        for scheme, blk in src["addr"]["by_scheme"].items():
            ds, lat = blk["ds"], blk["lat"]
            e = {"ds_run": ds, "lat_run": lat, "floor": blk["floor"]}
            for p in POLICIES:
                e[p] = {"ds": cell(recs[ds], p), "lat": cell(recs[lat], p),
                        "gap": round(cell(recs[lat], p) - cell(recs[ds], p), 4)}
                e[p]["latent_ahead"] = e[p]["gap"] < 0
            e["agrees"] = e["any_cell"]["latent_ahead"] == e["mixed"]["latent_ahead"]
            rec["by_scheme"][scheme] = e
        out["addr"] = rec

    # ---- factor substrate (tab_substrate) ------------------------------------------
    if "substrate" in src:
        recs, floor = src["substrate"]["arms"], src["substrate"]["floor"]
        const, cut = anchors("substrate", floor)
        rec = {"floor": floor, "const_map": const, "cutoff": cut, "per_group": {}}
        for g, arms in by_meta(recs, "group").items():
            arms = sorted(arms)
            e = {p: group_stats(arms, recs, p, cut[p]) for p in POLICIES}
            e["runs"] = arms
            # "below the random-init floor" is this table's whole claim; and whether
            # the arm is a constant map is what round 5 asked of ds and arbaux.
            e["below_floor_by_policy"] = {
                p: e[p]["all_seeds"]["mean"] < floor[p]["mean"] for p in POLICIES}
            e["is_constant_map_by_policy"] = {
                p: abs(e[p]["all_seeds"]["mean"] - const[p]) < 5e-4 for p in POLICIES}
            e["agrees"] = e["below_floor_by_policy"]["any_cell"] == \
                e["below_floor_by_policy"]["mixed"]
            rec["per_group"][g] = e
        out["substrate"] = rec

    # ---- compressibility sweep (the positive result) --------------------------------
    # What this family measures is a SUCCESS RATE, not a margin. On this substrate the
    # data-space arm is bimodal: a seed either solves the task or never leaves the constant
    # map, and the constant map is computed per K because Var[f] moves with intrinsic rank.
    # Two derived quantities carry the claim: how often each arm escapes the constant map,
    # and -- conditional on escaping -- whether the two arms land in the same place.
    if "boundary" in src:
        recs = src["boundary"]["arms"]
        rec = {"by_k": {}}
        for ks, blk in src["boundary"]["by_k"].items():
            const = blk["const_map_mse_f"]
            cut = cutoff(const, blk["floor"])
            arms = [a for a, r in recs.items() if str(r["meta"]["K"]) == ks]
            grp = {g: sorted(a for a in arms if recs[a]["meta"]["group"] == g)
                   for g in ("ds", "dual")}
            e = {"const_map": const, "cutoff": cut, "floor": blk["floor"], "runs": grp}
            for p in POLICIES:
                ent = {}
                for g, ga in grp.items():
                    vals = [cell(recs[a], p) for a in ga]
                    alive = [a for a in ga if cell(recs[a], p) < cut[p]]
                    ent[g] = {
                        "per_seed": {a: cell(recs[a], p) for a in ga},
                        "n_solved": len(alive), "n": len(ga),
                        "solve_rate": f"{len(alive)}/{len(ga)}",
                        "solved_mean": round(st.mean(cell(recs[a], p) for a in alive), 4)
                        if alive else None,
                        "all_mean": agg(vals)["mean"],
                        # a failure that IS the constant map is a training failure, not a score
                        "n_exactly_constant": sum(
                            1 for a in ga if abs(cell(recs[a], p) - const[p]) < 5e-4),
                    }
                sd_, du_ = ent["ds"], ent["dual"]
                # conditional on both solving, do they land in the same place?
                ent["solved_gap"] = (round(du_["solved_mean"] - sd_["solved_mean"], 4)
                                     if sd_["solved_mean"] and du_["solved_mean"] else None)
                ent["all_gap"] = round(du_["all_mean"] - sd_["all_mean"], 4)
                # Both null anchors move with K (Var[f] does), so the RAW score rising with K
                # is confounded with the anchors rising. Referencing the score to each anchor
                # removes that reading. Neither anchor is itself monotone in K, so these are
                # not monotone by construction.
                ent["dual_minus_floor"] = round(du_["solved_mean"] - e["floor"][p]["mean"], 4) \
                    if du_["solved_mean"] else None
                ent["dual_over_const"] = round(du_["solved_mean"] / const[p], 4) \
                    if du_["solved_mean"] else None
                e[p] = ent
            rec["by_k"][ks] = e
        # the curve itself: solve rate against K, which is the claim in one row each
        rec["solve_rate_curve"] = {
            p: {k: {g: rec["by_k"][k][p][g]["solve_rate"] for g in ("ds", "dual")}
                for k in sorted(rec["by_k"], key=int)} for p in POLICIES}
        # Does the ordering depend on WHICH normalization we report? Derived, not asserted:
        # the raw score, the margin below each anchor, and the ratio to the constant map are
        # each checked for a perfect monotone ordering under each policy.
        kso = sorted(rec["by_k"], key=int)
        stats = {"raw": lambda b, p: b[p]["dual"]["solved_mean"],
                 "minus_floor": lambda b, p: b[p]["dual_minus_floor"],
                 "over_const": lambda b, p: b[p]["dual_over_const"],
                 "anchors_alone": lambda b, p: b["const_map"][p]}
        rec["monotone_by_statistic"] = {
            s: {p: all(a < b for a, b in zip(v, v[1:])) or all(a > b for a, b in zip(v, v[1:]))
                for p in POLICIES
                for v in [[f(rec["by_k"][k], p) for k in kso]]} for s, f in stats.items()}
        # Round-8 repair 3. Monotonicity in K is NOT evidence about the objective on this
        # substrate: K is the target's intrinsic difficulty, so the untrained floor, the linear
        # optimum and the trained arm are all monotone in it. The statistic has to be one that
        # COULD come out flat. This one does: the fraction of the untrained-to-linear-optimal
        # headroom that training closes. The floor sits at 0 and the linear reference at 1 by
        # construction, so neither anchor can produce the trend, and an equally good learner at
        # every rank would give a flat curve.
        lin = json.loads((ROOT / "eval/results_linear_reference.json").read_text())["by_k"]
        for ks_, e in rec["by_k"].items():
            for p in POLICIES:
                opt = lin[ks_][p]["per_table"]
                fl = e["floor"][p]["mean"]
                du = e[p]["dual"]["solved_mean"]
                e[p]["linear_upper_ref"] = opt
                e[p]["linear_competitor"] = lin[ks_][p]["across_table"]
                e[p]["headroom_closed"] = (round((fl - du) / (fl - opt), 4)
                                           if du is not None and fl > opt else None)
        rec["headroom_note"] = ("(floor - dual) / (floor - per_table_linear). 0 = no better than "
                                "an untrained encoder, 1 = matches a linear model that reads the "
                                "ground truth. Both anchors are fixed at 0 and 1 by construction.")
        for s_, f_ in (("headroom_closed", lambda b, p: b[p]["headroom_closed"]),
                       ("linear_upper_ref", lambda b, p: b[p]["linear_upper_ref"]),
                       ("floor_over_const", lambda b, p: b["floor"][p]["mean"] / b["const_map"][p])):
            rec["monotone_by_statistic"][s_] = {
                p: all(a < b for a, b in zip(v, v[1:])) or all(a > b for a, b in zip(v, v[1:]))
                for p in POLICIES
                for v in [[f_(rec["by_k"][k], p) for k in kso]]}
        # Round-10 methodology + devil's advocate: the exact permutation p was 2/n! over six
        # LEVEL MEANS -- the minimum that test can return, a function of the level count alone,
        # and blind to the 24 seed-level runs underneath. And substituting the constant map or
        # the across-table ridge for the arm in the headroom formula also yields a monotone
        # curve, so "neither anchor can generate the trend" was false. Both are replaced by a
        # seed-level rank test plus the anchors' own curves, so a reader can see the substrate
        # trend and the arm's separately.
        import random as _rnd

        def _rank(v):
            srt = sorted(v)
            return [srt.index(x) + (srt.count(x) - 1) / 2 + 1 for x in v]

        def _spear(a, b):
            ra, rb = _rank(a), _rank(b)
            ma, mb = st.mean(ra), st.mean(rb)
            num = sum((p_ - ma) * (q - mb) for p_, q in zip(ra, rb))
            den = math.sqrt(sum((p_ - ma) ** 2 for p_ in ra) * sum((q - mb) ** 2 for q in rb))
            return num / den

        seedpts = {}
        for p_ in POLICIES:
            pts = []
            for ks_ in kso:
                e = rec["by_k"][ks_]
                opt = e[p_]["linear_upper_ref"]
                fl = e["floor"][p_]["mean"]
                for _run, v in e[p_]["dual"]["per_seed"].items():
                    if v < e["cutoff"][p_]:
                        pts.append((int(ks_), (fl - v) / (fl - opt)))
            xs_, ys_ = [q[0] for q in pts], [q[1] for q in pts]
            rho = _spear(xs_, ys_)
            _rnd.seed(0)
            n_perm = 20000
            hits = sum(1 for _ in range(n_perm)
                       if abs(_spear(_rnd.sample(xs_, len(xs_)), ys_)) >= abs(rho))
            seedpts[p_] = {"n_obs": len(pts), "spearman": round(rho, 4),
                           "n_perm": n_perm, "p": round((hits + 1) / (n_perm + 1), 5)}
        rec["seed_level_test"] = {
            "statistic": "Spearman(K, per-seed headroom_closed) over admitted seeds",
            "null": "K labels permuted; two-sided",
            "why": "the level-mean permutation test returns 2/n!, its own minimum, and never "
                   "sees the seed-level runs",
            **seedpts}
        # what the anchors themselves do under the SAME formula -- the honest control
        rec["anchor_curves"] = {
            p_: {nm: [round((rec["by_k"][k]["floor"][p_]["mean"] - f(rec["by_k"][k], p_))
                            / (rec["by_k"][k]["floor"][p_]["mean"]
                               - rec["by_k"][k][p_]["linear_upper_ref"]), 4) for k in kso]
                 for nm, f in (("dual", lambda e, q: e[q]["dual"]["solved_mean"]),
                               ("const_map", lambda e, q: e["const_map"][q]),
                               ("across_table", lambda e, q: e[q]["linear_competitor"]))}
            for p_ in POLICIES}
        # Round 11 (devil's advocate, reproduced): the round-10 "magnitude" defence fails
        # too. A sham arm defined as a fixed multiple c of the per-table ceiling carries no
        # K-specific effect, yet its headroom curve is perfectly monotone for every c > 1
        # (head = 1 - (c-1)*ref/(floor-ref), and ref/(floor-ref) rises with K), and at
        # moderate c its span exceeds the trained arm's. The one normalization that does not
        # divide by the shrinking floor-to-ceiling gap -- the arm over the ceiling itself --
        # is not monotone. Conclusion carried in the JSON so the paper cannot cite the curve
        # without it: on this substrate K moves every yardstick, so no normalization
        # separates degradation of the arm from difficulty of the task.
        yc = {}
        for p_ in POLICIES:
            fl = [rec["by_k"][k]["floor"][p_]["mean"] for k in kso]
            ref = [rec["by_k"][k][p_]["linear_upper_ref"] for k in kso]
            arm = [rec["by_k"][k][p_]["headroom_closed"] for k in kso]
            karr = [int(k) for k in kso]
            sham = {}
            for c in (1.1, 1.2, 1.3, 1.4, 1.5):
                sh = [(f_ - c * r_) / (f_ - r_) for f_, r_ in zip(fl, ref)]
                sham[str(c)] = {"spearman": round(_spear(karr, sh), 4),
                                "span": round(max(sh) - min(sh), 4)}
            ratio = [rec["by_k"][k][p_]["dual"]["solved_mean"] / r_
                     for k, r_ in zip(kso, ref)]
            yc[p_] = {"arm_span": round(max(arm) - min(arm), 4),
                      "sham_multiple_of_ceiling": sham,
                      "arm_over_ceiling_spearman": round(_spear(karr, ratio), 4)}
        rec["yardstick_check"] = {
            "reading": "a quantity with no K-specific content is as monotone as the arm and "
                       "can out-span it; the un-normalized ratio is not monotone; no "
                       "statistic on this substrate separates degradation from difficulty",
            **yc}
        rec["seed_level_test"]["retired"] = (
            "round 11: p=(hits+1)/(n_perm+1) with 0 hits is the test's own floor, and the "
            "null it rejects is one the anchor_curves already show to be false")
        # The training-free oracle has to be scored against the estimand we actually report.
        # It was previously correlated with the dual-ds gap read under the SUPERSEDED probe,
        # which is both the wrong probe and a quantity that does not exist above the lowest
        # rank. Recomputed here on the reported margin, over the full sweep with no subset.
        # Caveat carried in the JSON so the paper cannot quote rho without it: both quantities
        # are monotone in the same designed variable K, so rho sharpens the monotone result
        # rather than testing it independently.
        _bsum = json.loads((ROOT / "eval/results_boundary_summary.json").read_text())
        orc = {str(p["K"]): p["oracle"] for p in _bsum["points"]}
        xs = [orc[k] for k in kso]
        rec["oracle_corr"] = {"n": len(kso), "subset": "none -- full pre-registered sweep",
                              "estimand": "headroom_closed (masked read-out, round-8 repair 3)",
                              "caveat": "both series are monotone in K by design; rho is a "
                                        "precision statement about the same ordering, not an "
                                        "independent test of it",
                              "superseded": {"rho": _bsum["corr_all"],
                                             "estimand": "dual-ds gap, unmasked probe"}}
        for p in POLICIES:
            ys = [rec["by_k"][k][p]["headroom_closed"] for k in kso]
            rec["oracle_corr"][p] = {"rho": round(pearson(xs, ys), 4),
                                     "p": round(t_tail_p(pearson(xs, ys), len(kso)), 4)}
        out["boundary"] = rec

    # ---- 35M scale rung (tab_app_probe_big) -----------------------------------------
    # Round-6 finding 1: the scale route was still read under the superseded probe. Same
    # protocol as the base rung, so the two are directly comparable.
    for fam, key in (("big", "recipe"), ("deconf", "recipe"), ("cls", "addr")):
        if fam not in src:
            continue
        recs, floor = src[fam]["arms"], src[fam]["floor"]
        const, cut = anchors(key, floor)
        rec = {"floor": floor, "const_map": const, "cutoff": cut, "per_arm": {}}
        for a_, r in sorted(recs.items()):
            e = {"meta": r["meta"]}
            for p_ in POLICIES:
                e[p_] = {"mse_f": cell(r, p_), "mse_x": cell(r, p_, "mse_x"),
                         "alive": cell(r, p_) < cut[p_],
                         "is_constant_map": abs(cell(r, p_) - const[p_]) < 5e-4}
            rec["per_arm"][a_] = e
        if fam == "cls":
            # the target-location grid: gap = latent - ds within each k, the quantity Section 5.1
            # calls its dominant evidence. Recomputed here on the masked read-out.
            byk = {}
            for k in sorted({r["meta"]["k"] for r in recs.values()}):
                at = {r["meta"]["group"]: a_ for a_, r in recs.items() if r["meta"]["k"] == k}
                if "ds" not in at:
                    continue
                e = {"runs": at}
                for p_ in POLICIES:
                    e[p_] = {g: cell(recs[at[g]], p_) for g in at}
                    e[p_]["gap"] = (round(e[p_]["lat"] - e[p_]["ds"], 4)
                                    if "lat" in e[p_] else None)
                byk[str(k)] = e
            rec["by_k"] = byk
            ks = sorted(byk, key=int)
            if len(ks) > 1:
                rec["gap_narrowing"] = {
                    p_: {"cell_target": byk[ks[0]][p_]["gap"],
                         "row_summary": byk[ks[-1]][p_]["gap"]} for p_ in POLICIES}
        out[fam] = rec

    out["_missing"] = {"nonlinear": "matched_L4_* checkpoints are not on this machine; "
                                    "tab_nonlinear cannot be re-measured without retraining"}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), "\n")

    for fam in ("recipe", "substrate"):
        if fam not in out:
            continue
        r0 = out[fam]
        print(f"=== {fam}: masked cell f-MSE ===")
        print("  " + "  ".join(f"{p}: floor {r0['floor'][p]['mean']:.4f} / const map "
                               f"{r0['const_map'][p]:.4f}" for p in POLICIES))
        for g, r in r0["per_group"].items():
            cells = []
            for p in POLICIES:
                a = r[p]["alive"]
                cells.append(f"{p} all {r[p]['all_seeds']['mean']:.4f} "
                             + (f"alive {a['mean']:.4f} ({r[p]['n_alive']})" if a
                                else f"alive -- ({r[p]['n_alive']})"))
            flag = ""
            if fam == "substrate":
                flag = ("  CONSTANT MAP" if r["is_constant_map_by_policy"]["any_cell"]
                        else "  below floor" if r["below_floor_by_policy"]["any_cell"] else "")
            print(f"  {g:11} " + " | ".join(cells) + flag)
        print()
    if "sigma" in out:
        print("=== sigma sweep: masked lat-ds gap (+ve = data-space ahead) ===")
        for ns, e in out["sigma"]["by_noise"].items():
            print(f"  noise x{ns:<3} " + " | ".join(
                f"{p} {e[p]['gap_mean']:+.4f}+/-{e[p]['gap_half']:.4f}"
                f" (ds {e[p]['ds']['mean']:.3f} lat {e[p]['lat']['mean']:.3f}"
                f"{', lat DEGENERATE' if e[p]['lat_degenerate'] else ''}"
                f"{', lat at floor' if e[p]['lat_at_floor'] else ''})" for p in POLICIES)
                + ("  [agree]" if e["agrees"] else "  [DISAGREE]"))
        print()
    if "addr" in out:
        print("=== addressing: masked lat-ds gap (-ve = latent ahead) ===")
        for s, e in out["addr"]["by_scheme"].items():
            print(f"  {s:12} " + " | ".join(
                f"{p} {e[p]['gap']:+.4f} (ds {e[p]['ds']:.3f} lat {e[p]['lat']:.3f})"
                for p in POLICIES) + ("  [agree]" if e["agrees"] else "  [DISAGREE]"))
    if "boundary" in out:
        print("\n=== compressibility sweep: who escapes the constant map, and where they land ===")
        for p in POLICIES:
            print(f"  --- policy {p} ---")
            for k in sorted(out["boundary"]["by_k"], key=int):
                b_ = out["boundary"]["by_k"][k]
                e = b_[p]
                sm, dm = e["ds"]["solved_mean"], e["dual"]["solved_mean"]
                print(f"   K={k:>2}  ds {e['ds']['solve_rate']} "
                      f"({'  --  ' if sm is None else f'{sm:.3f}'})"
                      f"  dual {e['dual']['solve_rate']} "
                      f"({'  --  ' if dm is None else f'{dm:.3f}'})"
                      f"   const={b_['const_map'][p]:.3f}"
                      f" floor={b_['floor'][p]['mean']:.3f}"
                      f"   ds-at-const {e['ds']['n_exactly_constant']}/{e['ds']['n']}"
                      + (f"   solved-gap {e['solved_gap']:+.3f}"
                         if e["solved_gap"] is not None else ""))


if __name__ == "__main__":
    main()
