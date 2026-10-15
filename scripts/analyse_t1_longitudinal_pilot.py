"""Pre-registered pilot of FastSurfer's longitudinal stream (documentation/preregistration_t1_longitudinal_pilot.md).

    select    <out>/selection.csv: 100 participants with >= 2 usable, real-dated T1 visits >= 1 year apart (seeded,
              stratified by cohort: 40 HC, 30 PD, 30 prodromal) and the 29 same-day repeat pairs; reads dates, QC flags
              and cohort only, no volume
    run       FastSurfer's longitudinal stream per selected template, into <out>/long and <out>/long_sameday
    analyse   <out>/t1_long_pilot_result.json: the zero-change and plausibility gates and the descriptives

Gates: (1) zero change: >= 95 % of same-day pair x structure differences from the long stream lie within the
cross-sectional MDC95 of that structure; (2) plausibility: the HC median annualised change of hippocampus and putamen
(left + right) lies in its published window.

    PYTHONPATH=. python scripts/analyse_t1_longitudinal_pilot.py select --out <dir>
    PYTHONPATH=. python scripts/analyse_t1_longitudinal_pilot.py run --out <dir>
    PYTHONPATH=. python scripts/analyse_t1_longitudinal_pilot.py analyse --out <dir>
    PYTHONPATH=. python scripts/analyse_t1_longitudinal_pilot.py analyse --synthetic --out <dir>     # dry run: all fake
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED, B, MIN_SPAN, GATE_ZERO, MIN_HC = 20261015, 2000, 1.0, 0.95, 20
N = {"HC": 40, "PD": 30, "Prodromal": 30}
STRUCT = ["Putamen", "Caudate", "Thalamus", "Pallidum", "Hippocampus", "Amygdala", "Lateral_Ventricle"]
KEYS = [f"vol_{s}_{r}" for r in STRUCT for s in ("Left", "Right")] + ["vol_Brain_Stem"]   # the reliability analysis's 15
WINDOWS = {"Hippocampus": (-1.70, -0.27), "Putamen": (-1.20, -0.10)}       # %/yr, HC median (sources in the pre-registration)
REPORTED = {"Thalamus": -0.69, "Lateral_Ventricle": 4.40}                    # Fjell 2009, report only
D = Path("Imaging/derived")
COHORT = {"Healthy Control": "HC", "Parkinson's Disease": "PD", "Prodromal": "Prodromal"}


def _visits(derived, ppmi):
    from pie.imaging.labels import covariates
    from pie.imaging.manifest import t1_visits

    v = t1_visits(derived)
    v = v[v["t1_qc_pass"] & v["SCAN_DATE"].notna()].merge(covariates(ppmi)[["PATNO", "COHORT"]], on="PATNO", how="left")
    return v.assign(cohort=v["COHORT"].map(COHORT))


def select(derived, ppmi, out):
    from ch4_cat12_validation import same_day_pairs

    v = _visits(derived, ppmi)
    p = v.groupby("PATNO").agg(n=("IMAGEID", "size"), cohort=("cohort", "first"),
                               span=("SCAN_DATE", lambda s: (s.max() - s.min()).days / 365.25))
    elig = p[(p["n"] >= 2) & (p["span"] >= MIN_SPAN) & p["cohort"].notna()].sort_index()
    avail = elig["cohort"].value_counts()
    if avail.get("HC", 0) < MIN_HC:
        raise SystemExit(f"{int(avail.get('HC', 0))} eligible HC, fewer than {MIN_HC}: the plausibility gate could not be judged. "
                         "Download and process the controls' follow-up T1s first")
    k = {"HC": min(N["HC"], int(avail.get("HC", 0)))}
    short = N["HC"] - k["HC"]                                    # an HC shortfall goes to PD and prodromal, PD first
    k.update(PD=N["PD"] + (short + 1) // 2, Prodromal=N["Prodromal"] + short // 2)
    pick = pd.concat([elig[elig["cohort"].eq(c)].sample(n=min(m, int(avail.get(c, 0))), random_state=SEED) for c, m in k.items()])
    sel = v[v["PATNO"].isin(pick.index)].sort_values(["PATNO", "SCAN_DATE"])[["PATNO", "IMAGEID", "SCAN_DATE", "cohort"]]
    sel = sel.assign(role="pilot", tid=sel["PATNO"].astype(str))
    pairs = same_day_pairs().rename(columns={"day": "SCAN_DATE"})
    pairs = pairs.assign(role="sameday", cohort=pairs["PATNO"].map(v.groupby("PATNO")["cohort"].first()),
                         tid="sameday_" + pairs["PATNO"].astype(str) + "_" + pairs["SCAN_DATE"].astype(str))
    pd.concat([sel, pairs], ignore_index=True).to_csv(out / "selection.csv", index=False)
    print(f"eligible: {avail.to_dict()}; selected: {pick['cohort'].value_counts().to_dict()} "
          f"({len(sel)} visits); same-day pairs: {len(pairs) // 2}")


def run(out, derived, device, threads):
    from pie.imaging.fastsurfer import longitudinal

    sel = pd.read_csv(out / "selection.csv", dtype={"IMAGEID": str})
    failed = []
    for tid, g in sel.groupby("tid", sort=False):
        sd = out / ("long_sameday" if g["role"].iloc[0] == "sameday" else "long")
        t1s = [Path(derived) / "nifti" / str(p) / f"{i}_T1w.nii.gz" for p, i in zip(g["PATNO"], g["IMAGEID"])]
        try:
            longitudinal(tid, t1s, g["IMAGEID"].tolist(), sd, device=device, threads=threads)
        except Exception as e:
            failed.append({"tid": tid, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    pd.DataFrame(failed, columns=["tid", "error"]).to_csv(out / "long_failures.csv", index=False)
    print(f"templates: {sel['tid'].nunique()}, failed: {len(failed)}")


def load(out, derived, ppmi):
    """One row per (template, scan, stream): long-stream volumes from the stats files, cross-sectional ones from the
    IDP tables (the pre-11-Oct table for the same-day pairs, where both scans of a day were rows)."""
    from pie.imaging.fastsurfer import STATS_FILE, parse_stats
    from pie.imaging.features import _clean

    sel = pd.read_csv(out / "selection.csv", dtype={"IMAGEID": str})
    rows = []
    for r in sel.itertuples(index=False):
        f = out / ("long_sameday" if r.role == "sameday" else "long") / r.IMAGEID / STATS_FILE
        vol = {f"vol_{_clean(k)}": x for k, x in parse_stats(f).items()} if f.exists() else {}
        rows.append({"tid": r.tid, "IMAGEID": r.IMAGEID, "stream": "long", **{k: vol.get(k, np.nan) for k in KEYS}})
    cross = pd.concat([pd.read_csv(Path(derived) / t, dtype={"IMAGEID": str}, low_memory=False)[["IMAGEID", *KEYS]]
                       for t in ("fastsurfer_idps.csv", "fastsurfer_idps.pre_20261011.csv")]).drop_duplicates("IMAGEID")
    s = pd.concat([pd.DataFrame(rows), sel[["tid", "IMAGEID"]].merge(cross, on="IMAGEID", how="left").assign(stream="cross")])
    v = _visits(derived, ppmi).astype({"IMAGEID": str})
    s = s.merge(sel[["tid", "IMAGEID", "role", "cohort"]], on=["tid", "IMAGEID"]).merge(
        v[["IMAGEID", "years_from_bl", "scanner_changed_since_bl"]].rename(columns={"years_from_bl": "years", "scanner_changed_since_bl": "scanner_changed"}),
        on="IMAGEID", how="left")
    return s, {"templates": int(sel["tid"].nunique())}


def synthetic(seed=1):
    """Fully synthetic scans (no real value, label, date or identifier): 29 same-day pairs and 100 participants with
    2-4 visits; known annual change per cohort; the long stream's measurement noise 0.6 x the cross-sectional."""
    rng = np.random.default_rng(seed)
    mean = dict(zip(KEYS, [4800, 4800, 3500, 3500, 7500, 7500, 2000, 2000, 4000, 4000, 1600, 1600, 15000, 15000, 21000]))
    noise = {k: (0.025 if "Putamen" in k or "Amygdala" in k else 0.012) * m for k, m in mean.items()}    # cross SD per scan
    rate = {"HC": {"Hippocampus": -0.9, "Putamen": -0.5}, "PD": {"Hippocampus": -1.2, "Putamen": -0.8}}
    rate["Prodromal"] = rate["HC"]
    rows = []

    def scans(tid, role, cohort, years, changed, truth):
        for j, y in enumerate(years):
            for stream, f in (("cross", 1.0), ("long", 0.6)):
                vol = {k: truth[k] * (1 + next((r for s, r in rate[cohort].items() if s in k), -0.3 if "Ventricle" not in k else 4.0) / 100 * y)
                       + rng.normal(0, f * noise[k]) for k in KEYS}
                rows.append({"tid": tid, "IMAGEID": f"{tid}_{j}", "stream": stream, "role": role, "cohort": cohort,
                             "years": y, "scanner_changed": bool(changed and j > 0), **vol})

    for i in range(29):
        scans(f"sameday_{i}", "sameday", "Prodromal", [0.0, 0.0], False, {k: m * rng.normal(1, 0.1) for k, m in mean.items()})
    for i, c in enumerate(["HC"] * 40 + ["PD"] * 30 + ["Prodromal"] * 30):
        years = np.r_[0, np.cumsum(rng.uniform(0.9, 2.2, rng.integers(1, 4)))]
        scans(f"p{i}", "pilot", c, years, rng.random() < 0.15, {k: m * rng.normal(1, 0.1) for k, m in mean.items()})
    return pd.DataFrame(rows), {"synthetic": True}


def _wide(s, k):
    """n x 2 array of a structure over complete same-day pairs."""
    x = s.sort_values(["tid", "IMAGEID"]).groupby("tid")[k].agg(list)
    return np.array([v for v in x if len(v) == 2 and np.isfinite(v).all()], float)


def _median_ci(x, rng):
    x = np.asarray(x, float)
    boot = [np.median(rng.choice(x, len(x))) for _ in range(B)]
    return {"n": int(len(x)), "median": float(np.median(x)), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}


def annual(s, stream):
    """Annualised % change, first to last visit, of each structure (left + right), per pilot participant."""
    x = s[s["role"].eq("pilot") & s["stream"].eq(stream)].sort_values(["tid", "years"])
    g = x.groupby("tid")
    first, last = g.first(), g.last()
    yrs = last["years"] - first["years"]
    out = pd.DataFrame({"cohort": first["cohort"], "years": yrs, "scanner_changed": g["scanner_changed"].max().fillna(False).astype(bool)})
    for st in STRUCT:
        a, b = (t[f"vol_Left_{st}"] + t[f"vol_Right_{st}"] for t in (first, last))
        out[st] = 100 * (b - a) / a / yrs
    return out


def analyse(s, flow):
    from pie.stats.agreement import icc21

    rng = np.random.default_rng(SEED)
    long_ = s[s["stream"].eq("long")]
    failed = long_.loc[long_[KEYS].isna().any(axis=1), "tid"].unique()        # any scan of the template without volumes
    res = {"flow": {**flow, "excluded_long_failure": s[s["tid"].isin(failed)].drop_duplicates("tid")["role"].value_counts().to_dict()}}
    s = s[~s["tid"].isin(failed)]
    res["flow"]["analysed_templates"] = s.drop_duplicates("tid")["role"].value_counts().to_dict()

    # Gate 1: zero change on the same-day pairs
    sd_ = s[s["role"].eq("sameday")]
    per, within = {}, []
    for k in KEYS:
        c, lg = _wide(sd_[sd_["stream"].eq("cross")], k), _wide(sd_[sd_["stream"].eq("long")], k)
        mdc_c, mdc_l = 1.96 * np.std(c[:, 0] - c[:, 1], ddof=1), 1.96 * np.std(lg[:, 0] - lg[:, 1], ddof=1)
        w = np.abs(lg[:, 0] - lg[:, 1]) <= mdc_c
        within += list(w)
        per[k] = {"n_pairs_cross": len(c), "n_pairs_long": len(lg), "icc21_cross": icc21(c), "icc21_long": icc21(lg),
                  "mdc95_pct_cross": float(100 * mdc_c / c.mean()), "mdc95_pct_long": float(100 * mdc_l / lg.mean()),
                  "fraction_within_cross_mdc95": float(w.mean())}
    res["zero_change"] = {"fraction_within": float(np.mean(within)), "n_values": len(within), "per_structure": per,
                          "structures_long_mdc95_lower": int(sum(v["mdc95_pct_long"] < v["mdc95_pct_cross"] for v in per.values()))}

    # Gate 2: plausibility of change in controls
    a_long, a_cross = annual(s, "long"), annual(s, "cross")
    hc = a_long[a_long["cohort"].eq("HC")]
    plaus = {st: {**_median_ci(hc[st], rng), "window": WINDOWS.get(st), "published": REPORTED.get(st)} for st in STRUCT}
    for st, (lo, hi) in WINDOWS.items():
        plaus[st]["in_window"] = bool(lo <= plaus[st]["median"] <= hi)
    res["plausibility_hc"] = plaus
    res["hc_same_scanner"] = {st: _median_ci(hc.loc[~hc["scanner_changed"], st], rng) for st in WINDOWS} if (~hc["scanner_changed"]).sum() >= 5 else {}

    # Reported: noise of change, long vs cross; cohort change
    res["sd_annual_change_long_vs_cross"] = {st: {"long": float(a_long[st].std()), "cross": float(a_cross[st].std())} for st in STRUCT}
    res["by_cohort"] = {c: {st: _median_ci(g[st], rng) for st in WINDOWS} for c, g in a_long.groupby("cohort")}
    diff = {}
    for st in WINDOWS:
        h, p = hc[st].to_numpy(), a_long.loc[a_long["cohort"].eq("PD"), st].to_numpy()
        boot = [np.median(rng.choice(p, len(p))) - np.median(rng.choice(h, len(h))) for _ in range(B)]
        diff[st] = {"pd_minus_hc_median": float(np.median(p) - np.median(h)), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}
    res["pd_minus_hc"] = diff
    res["gates"] = {"zero_change": res["zero_change"]["fraction_within"] >= GATE_ZERO,
                    "plausibility": all(plaus[st]["in_window"] for st in WINDOWS)}
    res["pass"] = bool(all(res["gates"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "run", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--derived", default=str(D))
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true", help="analyse: dry run on fully synthetic data (reads nothing real)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        return select(a.derived, a.ppmi, out)
    if a.step == "run":
        return run(out, a.derived, a.device, a.threads)
    s, flow = synthetic() if a.synthetic else load(out, a.derived, a.ppmi)
    res = analyse(s, flow)
    (out / ("synthetic_result.json" if a.synthetic else "t1_long_pilot_result.json")).write_text(json.dumps(res, indent=1, default=str))
    z, p = res["zero_change"], res["plausibility_hc"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}zero change {z['fraction_within']:.3f} within cross MDC95 "
          f"({z['n_values']} values); HC median %/yr hippocampus {p['Hippocampus']['median']:.2f}, putamen "
          f"{p['Putamen']['median']:.2f}; gates {res['gates']} -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
