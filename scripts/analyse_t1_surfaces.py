"""Pre-registered analysis (documentation/preregistration_t1_surfaces.md). Part A: PIE's DK cortical thickness (FastSurfer
segmentation, its surface stream on FreeSurfer 7.4.1 tools; ``run --surfaces``) against PPMI's FreeSurfer 7 table
(``FS7_APARC_CTH``), across same-day repeat scans, and with age in controls. Part B (descriptive): PIE's T1 robust-z
outlier flag (``manifest.t1_outliers``) against PPMI's MRIQC image-quality metrics.

    select    <out>/sids.csv in processing order: every baseline healthy control with an FS7 row, both scans of each
              same-day pair, then a seeded random 250 of the other baseline scans with an FS7 row. Reads no thickness
    run       fastsurfer.surfaces on the selected scans, in that order
    analyse   <out>/t1_surfaces_result.json: the three Part A gates and the Part B description

    PYTHONPATH=. python scripts/analyse_t1_surfaces.py select --out <dir>
    PYTHONPATH=. python scripts/analyse_t1_surfaces.py run --out <dir> --workers 4 --threads 2
    PYTHONPATH=. python scripts/analyse_t1_surfaces.py analyse --out <dir>
    PYTHONPATH=. python scripts/analyse_t1_surfaces.py analyse --synthetic --out <dir>      # dry run: every value is fake
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch4_cat12_validation import _r_ci, _resid, baseline, same_day_pairs  # noqa: E402  (the T1 sample rules already in use)
from nm_staged import auc_with_ci  # noqa: E402  (stratified AUROC with a within-stratum bootstrap)

from pie.stats.agreement import agreement, icc21  # noqa: E402

SEED, B = 20261015, 2000
GATE_CCC, GATE_ICC, GATE_AGE_R = 0.75, 0.85, -0.30
N_OTHER = 250          # seeded non-HC baseline scans for the FS7 agreement gate; the rest run later, outside this test
IQM = {"cjv": 1, "cnr": -1, "efc": 1, "fber": -1, "snr_total": -1}     # +1: higher is worse quality (Esteban 2017)
D = Path("Imaging/derived")
HC = "Healthy Control"


def select(ppmi, out):
    from pie.imaging.features import fs7_tables
    from pie.imaging.labels import covariates

    fs7 = fs7_tables(ppmi)
    fs7 = fs7.loc[fs7["fs7_cth_lh_MeanThickness"].notna(), ["PATNO", "EVENT_ID"]]
    b = baseline().merge(fs7, on=["PATNO", "EVENT_ID"]).merge(covariates(ppmi)[["PATNO", "COHORT"]], on="PATNO", how="left")
    hc, rest = b[b["COHORT"].eq(HC)], b[~b["COHORT"].eq(HC)].sort_values("IMAGEID").sample(frac=1, random_state=SEED).head(N_OTHER)
    pairs = same_day_pairs()
    rows = pd.concat([hc[["PATNO", "IMAGEID"]].assign(role="fs7"), pairs.assign(role="repeat"),
                      rest[["PATNO", "IMAGEID"]].assign(role="fs7")], ignore_index=True)
    rows.to_csv(out / "sids.csv", index=False)
    print(f"selected: {len(b)} baseline scans with FS7 thickness ({len(hc)} HC), {len(pairs) // 2} same-day pairs; "
          f"{rows['IMAGEID'].nunique()} scans to run")


def _run_one(job):
    from pie.imaging.fastsurfer import surfaces

    sid, threads = job
    try:
        surfaces(D / "fastsurfer", str(sid), threads=threads)
        return sid, ""
    except Exception as e:
        return sid, f"{type(e).__name__}: {str(e)[:200]}"


def run(out, workers, threads):
    sids = pd.read_csv(out / "sids.csv")["IMAGEID"].drop_duplicates().tolist()      # file order = processing order
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (sid, err) in enumerate(ex.map(_run_one, [(s, threads) for s in sids]), start=1):
            if err:
                errors.append({"IMAGEID": sid, "error": err})
                pd.DataFrame(errors).to_csv(out / "surfaces_errors.csv", index=False)
            if err or k % 20 == 0:
                print(re.sub(r"\d{4,}", "####", f"{k}/{len(sids)} done, {len(errors)} failed {err}"), flush=True)


def load(ppmi, out):
    """Three frames without identifiers: baseline scans with PIE and FS7 thickness, same-day repeat scans, MRIQC sessions."""
    from pie.imaging.features import fs7_tables, thickness_columns
    from pie.imaging.labels import _age_at, covariates
    from pie.imaging.manifest import _vendor, t1_outliers

    s = pd.read_csv(out / "sids.csv")
    cov, fs7 = covariates(ppmi), fs7_tables(ppmi)
    idps = pd.read_csv(D / "fastsurfer_idps.csv", low_memory=False)
    idps = idps.join(t1_outliers(idps))       # batch medians over every session, as the manifest computes them
    idps["batch"] = _vendor(idps["Manufacturer"]) + "_" + idps["MagneticFieldStrength"].astype(str)
    th = pd.DataFrame([{"IMAGEID": i, **thickness_columns(D / "fastsurfer" / str(i) / "stats")} for i in s["IMAGEID"].unique()])
    keep = ["PATNO", "IMAGEID", "EVENT_ID", "SCAN_DATE", "batch", "t1_outlier", "t1_robust_z_max"]
    main = s[s["role"].eq("fs7")].merge(idps[keep], on=["PATNO", "IMAGEID"]).merge(fs7, on=["PATNO", "EVENT_ID"], how="left")
    main = main.merge(cov[["PATNO", "COHORT", "SEX"]], on="PATNO", how="left").merge(th, on="IMAGEID", how="left")
    main["age"] = _age_at(cov, main["PATNO"], main["SCAN_DATE"])
    rep = s[s["role"].eq("repeat")].merge(th, on="IMAGEID", how="left")
    rep["pair"] = rep.groupby(["PATNO", "day"]).ngroup()
    mq = [c for c in fs7 if c.startswith("mriqc_")]
    qc = idps.sort_values(["PATNO", "SCAN_DATE"]).merge(fs7[["PATNO", "EVENT_ID"] + mq].dropna(subset=["mriqc_cjv"]),
                                                       on=["PATNO", "EVENT_ID"]).drop_duplicates("PATNO")
    ids = ["PATNO", "IMAGEID", "EVENT_ID", "SCAN_DATE", "day"]
    return (main.drop(columns=[c for c in ids if c in main]), rep.drop(columns=[c for c in ids if c in rep]),
            qc[["batch", "t1_outlier", "t1_robust_z_max"] + mq].reset_index(drop=True))


def synthetic(n=600, n_hc=165, age_slope=-0.005, n_qc=1300):
    """Fully synthetic scans (no real value, label or identifier): for dry runs and the test."""
    rng = np.random.default_rng(7)
    regions = [f"{h}_r{k}" for h in ("lh", "rh") for k in range(34)]
    age, batch = rng.normal(62, 10, n), rng.choice(["SIEMENS_3.0", "GE_3.0", "PHILIPS_1.5"], n, p=[0.6, 0.25, 0.15])
    subj = rng.normal(0, 0.10, n) + age_slope * (age - 62)
    cols = {}
    for r in regions:
        true = rng.uniform(2.0, 3.2) + subj + rng.normal(0, 0.08, n)
        cols["cth_" + r], cols["fs7_cth_" + r] = true + rng.normal(0, 0.03, n), true + 0.05 + rng.normal(0, 0.03, n)
        area = rng.uniform(500, 4000) * np.exp(rng.normal(0, 0.12, n))
        cols["area_" + r], cols["fs7_sa_" + r] = area * (1 + rng.normal(0, 0.03, n)), area * (1 + rng.normal(0, 0.03, n))
    main = pd.DataFrame({"COHORT": np.where(np.arange(n) < n_hc, HC, "Parkinson's Disease"), "age": age,
                         "SEX": rng.integers(0, 2, n).astype(float), "batch": batch, "t1_outlier": rng.random(n) < 0.01, **cols})
    for h in ("lh", "rh"):
        main[f"cth_{h}_mean"] = main[[f"cth_{h}_r{k}" for k in range(34)]].mean(axis=1)
        main[f"fs7_cth_{h}_MeanThickness"] = main[[f"fs7_cth_{h}_r{k}" for k in range(34)]].mean(axis=1)
    t = {r: rng.uniform(2.0, 3.2) + rng.normal(0, 0.12, 29) for r in regions + ["lh_mean", "rh_mean"]}
    rep = pd.DataFrame({"role": "repeat", "pair": np.repeat(np.arange(29), 2),
                        **{"cth_" + r: np.repeat(v, 2) + rng.normal(0, 0.03, 58) for r, v in t.items()}})
    bad = rng.random(n_qc) < 0.02
    z = np.abs(rng.normal(0, 1.3, n_qc)) + 4 * bad
    qc = pd.DataFrame({"batch": rng.choice(["SIEMENS_3.0", "GE_3.0", "PHILIPS_1.5"], n_qc), "t1_outlier": z > 4, "t1_robust_z_max": z,
                       **{"mriqc_" + k: 10 + s * (rng.normal(size=n_qc) + 1.5 * bad) for k, s in IQM.items()}})
    return main, rep, qc


def _ccc_cols(X, Y):
    """Lin's CCC per column (population moments, as pie.stats.agreement.ccc)."""
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    return 2 * ((X - mx) * (Y - my)).mean(axis=0) / (X.var(axis=0) + Y.var(axis=0) + (mx - my) ** 2)


def _median_ccc(m, pie_cols, ref_cols, n_boot):
    X, Y = m[pie_cols].to_numpy(float), m[ref_cols].to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = [np.median(_ccc_cols(X[i], Y[i])) for i in (rng.integers(0, len(X), len(X)) for _ in range(n_boot))]
    return {"n": int(len(X)), "median_ccc": float(np.median(_ccc_cols(X, Y))), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}


def _repeatability(rep, regions, n_boot):
    w = {r: rep.pivot_table(index="pair", columns=rep.groupby("pair").cumcount(), values="cth_" + r) for r in regions}
    pairs = sorted(set.intersection(*(set(v.dropna().index) for v in w.values())))
    arr = np.stack([w[r].loc[pairs].to_numpy(float) for r in regions])            # region x pair x scan
    icc = np.array([icc21(a) for a in arr])
    sem = arr.std(axis=(1, 2), ddof=1) * np.sqrt(np.clip(1 - icc, 0, None))
    rng = np.random.default_rng(SEED)
    boot = [np.median([icc21(a[i]) for a in arr]) for i in (rng.integers(0, len(pairs), len(pairs)) for _ in range(n_boot))]
    return {"n_pairs": len(pairs), "median_icc21": float(np.median(icc)), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "min_icc21": float(icc.min()), "regions_below_0.75": int((icc < 0.75).sum()),
            "median_mdc95_pct": float(np.median(100 * 1.96 * np.sqrt(2) * sem / arr.mean(axis=(1, 2))))}, dict(zip(regions, icc.round(4).tolist()))


def part_b(qc, n_boot):
    """PIE's flag against each MRIQC metric, oriented so that higher is worse; pairs compared within vendor x field."""
    f = qc["t1_outlier"].astype(bool).to_numpy()
    out = {"n": int(len(qc)), "flagged": int(f.sum()), "by_batch": qc.groupby("batch")["t1_outlier"].agg(["size", "sum"]).astype(int).to_dict("index")}
    worst = np.zeros(len(qc), bool)
    for k, sign in IQM.items():
        v = sign * pd.to_numeric(qc["mriqc_" + k], errors="coerce")
        pct = v.groupby(qc["batch"]).rank(pct=True)
        worst |= (pct > 0.95).to_numpy()
        ok = v.notna().to_numpy()
        out[k] = {"auroc_flag": auc_with_ci(v[f & ok], v[~f & ok], qc["batch"][f & ok], qc["batch"][~f & ok], n_boot),
                  "spearman_robust_z": float(stats.spearmanr(v, qc["t1_robust_z_max"], nan_policy="omit")[0]),
                  "flag_rate_by_within_batch_quintile": qc["t1_outlier"].astype(float).groupby(np.ceil(5 * pct).clip(1, 5)).mean().to_dict()}
    out["worst_5pct_any_iqm"] = {"n": int(worst.sum()), "flagged_fraction": float(f[worst].mean()) if worst.any() else None,
                                 "flagged_fraction_rest": float(f[~worst].mean())}
    return out


def analyse(main, rep, qc, n_boot=B):
    regions = [c[4:] for c in main if c.startswith("cth_") and not c.endswith("_mean") and "fs7_" + c in main]
    done = main[[f"cth_{h}_mean" for h in ("lh", "rh")]].notna().all(axis=1)
    m = main[done & main[["fs7_cth_" + r for r in regions]].notna().all(axis=1)].reset_index(drop=True)
    res = {"flow": {"selected_scans": int(len(main)), "surfaces_missing": int((~done).sum()), "fs7_missing": int((done.sum() - len(m))),
                    "analysed": int(len(m)), "analysed_by_cohort": m["COHORT"].value_counts().to_dict(), "n_regions": len(regions)}}
    cth = lambda d, n=n_boot: _median_ccc(d, ["cth_" + r for r in regions], ["fs7_cth_" + r for r in regions], n)
    a = {"all": cth(m), "regions": {r: agreement(m["cth_" + r], m["fs7_cth_" + r]) for r in regions},
         "mean_thickness": {h: agreement(m[f"cth_{h}_mean"], m[f"fs7_cth_{h}_MeanThickness"]) for h in ("lh", "rh")},
         "by_vendor": {k: cth(g, 200) for k, g in m.groupby(m["batch"].str.split("_").str[0]) if len(g) >= 20},
         "by_field_strength": {k: cth(g, 200) for k, g in m.groupby(m["batch"].str.split("_").str[1]) if len(g) >= 20},
         "without_t1_outliers": cth(m[~m["t1_outlier"].astype(bool)], 200)}
    sa = [r for r in regions if "area_" + r in m and "fs7_sa_" + r in m]
    a["area_no_gate"] = _median_ccc(m.dropna(subset=["area_" + r for r in sa]), ["area_" + r for r in sa], ["fs7_sa_" + r for r in sa], 200) if sa else None
    rep_res, rep_regions = _repeatability(rep, regions, n_boot)
    hc = m[m["COHORT"].eq(HC)].dropna(subset=["age", "SEX"]).reset_index(drop=True)
    X = pd.concat([hc[["SEX"]], pd.get_dummies(hc["batch"], drop_first=True, dtype=float)], axis=1)
    age = {}
    for k, y in {"pie": hc[["cth_lh_mean", "cth_rh_mean"]].mean(axis=1), "fs7": hc[["fs7_cth_lh_MeanThickness", "fs7_cth_rh_MeanThickness"]].mean(axis=1)}.items():
        r = float(stats.pearsonr(_resid(y.to_numpy(float), X, np.ones(len(hc), bool)), hc["age"])[0])
        r0 = float(stats.pearsonr(y, hc["age"])[0])
        age[k] = {"r_sex_batch_adjusted": r, "ci95": _r_ci(r, len(hc)), "r_unadjusted": r0, "ci95_unadjusted": _r_ci(r0, len(hc)), "n_hc": len(hc)}
    res.update(agreement_fs7=a, same_day=rep_res, same_day_regions=rep_regions, age_hc=age, part_b_mriqc=part_b(qc, n_boot))
    res["gates"] = {"agreement_fs7": bool(a["all"]["median_ccc"] >= GATE_CCC), "same_day": bool(rep_res["median_icc21"] >= GATE_ICC),
                    "age": bool(age["pie"]["r_sex_batch_adjusted"] <= GATE_AGE_R and age["pie"]["ci95"][1] < 0)}
    res["validated"] = all(res["gates"].values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "run", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        return select(a.ppmi, out)
    if a.step == "run":
        return run(out, a.workers, a.threads)
    res = analyse(*(synthetic() if a.synthetic else load(a.ppmi, out)), a.n_boot)
    (out / ("synthetic_t1_surfaces.json" if a.synthetic else "t1_surfaces_result.json")).write_text(json.dumps(res, indent=1, default=float))
    g, r, h = res["agreement_fs7"]["all"], res["same_day"], res["age_hc"]["pie"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}median regional CCC vs FS7 {g['median_ccc']:.3f} {np.round(g['ci95'], 3).tolist()} "
          f"(n {g['n']}); same-day median ICC {r['median_icc21']:.3f} ({r['n_pairs']} pairs); HC age r {h['r_sex_batch_adjusted']:.3f} "
          f"{np.round(h['ci95'], 3).tolist()} (n {h['n_hc']}); gates {res['gates']} -> {'VALIDATED' if res['validated'] else 'NOT VALIDATED'}")
    return res


if __name__ == "__main__":
    main()
