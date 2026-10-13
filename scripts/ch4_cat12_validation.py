"""Ch4 (nucleus basalis) grey-matter volume with CAT12: validation on PPMI (documentation/preregistration_ch4_cat12.md).

    select    <out>/sids.csv: every baseline healthy control, a seeded random equal number of baseline PD, and both scans
              of each same-day repeat T1 pair; reads no Ch4 value
    run       pie.imaging.cat12.ch4_volume on every selected scan (stats/cat12_ch4.json in its FastSurfer folder)
    analyse   <out>/ch4_cat12_result.json: the three gates and the descriptives

    PYTHONPATH=. python scripts/ch4_cat12_validation.py select --out <dir>
    PYTHONPATH=. python scripts/ch4_cat12_validation.py run --out <dir> --workers 6
    PYTHONPATH=. python scripts/ch4_cat12_validation.py analyse --out <dir>
    PYTHONPATH=. python scripts/ch4_cat12_validation.py analyse --synthetic --out <dir>     # dry run on fake values
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED, B = 20261013, 2000
GATE_AGE_R, GATE_RHO, GATE_ICC, IQR_MIN = -0.30, 0.40, 0.75, 60.0
D = Path("Imaging/derived")
MEASURES = {"ch4": "Ch4 GM, p >= 0.5 (primary)", "ch4_p": "Ch4 GM, probability-weighted", "nmm_bf": "neuromorphometrics basal forebrain"}


def baseline():
    """Earliest QC-passing T1 per participant."""
    from pie.imaging.manifest import QC

    idps = pd.read_csv(D / "fastsurfer_idps.csv", low_memory=False)
    idps = idps[QC["t1"](idps).to_numpy()]
    return idps.sort_values(["PATNO", "SCAN_DATE"]).drop_duplicates("PATNO")


def same_day_pairs():
    """The 29 same-day MPRAGE + repeat pairs of the T1 reliability analysis (pre-fix table, where they were rows)."""
    from pie.imaging.manifest import QC

    idps = pd.read_csv(D / "fastsurfer_idps.pre_20261011.csv", low_memory=False)
    idps = idps[QC["t1"](idps).to_numpy()]
    idps["day"] = pd.to_datetime(idps["SCAN_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    dated = idps[idps["day"].notna() & ~idps["day"].str.startswith("9999")]
    g = dated.groupby(["PATNO", "day"]).filter(lambda x: len(x) >= 2).sort_values(["PATNO", "day", "IMAGEID"])
    return g.groupby(["PATNO", "day"]).head(2)[["PATNO", "day", "IMAGEID"]]


def select(ppmi, out):
    from pie.imaging.labels import covariates

    b = baseline().merge(covariates(ppmi)[["PATNO", "COHORT"]], on="PATNO", how="left")
    hc = b[b["COHORT"].eq("Healthy Control")]
    pdm = b[b["COHORT"].eq("Parkinson's Disease")]
    pdm = pdm.sample(n=min(len(hc), len(pdm)), random_state=SEED)
    pairs = same_day_pairs()
    rows = pd.concat([hc[["PATNO", "IMAGEID"]].assign(role="hc"), pdm[["PATNO", "IMAGEID"]].assign(role="pd"),
                      pairs.assign(role="repeat")], ignore_index=True)
    rows.to_csv(out / "sids.csv", index=False)
    print(f"selected: {len(hc)} HC, {len(pdm)} PD, {len(pairs) // 2} same-day pairs; {rows['IMAGEID'].nunique()} scans")


def _run_one(job):
    from pie.imaging.cat12 import ch4_volume

    sid, nii = job
    try:
        ch4_volume(D / "fastsurfer", sid, nii, threads=1)
        return sid, ""
    except Exception as e:
        return sid, f"{type(e).__name__}: {str(e)[:200]}"


def run(out, workers):
    s = pd.read_csv(out / "sids.csv").drop_duplicates("IMAGEID")
    jobs = [(str(i), str(D / "nifti" / str(p) / f"{i}_T1w.nii.gz")) for p, i in zip(s["PATNO"], s["IMAGEID"])]
    print(f"{len(jobs)} scans", flush=True)
    fails = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(_run_one, j) for j in jobs]), start=1):
            sid, err = f.result()
            fails += bool(err)
            if err:
                print(f"{k}/{len(jobs)} {sid} ERROR {err}", flush=True)
            elif k % 20 == 0:
                print(f"{k}/{len(jobs)} done, {fails} failed", flush=True)
    print(f"finished: {len(jobs) - fails}/{len(jobs)} ok", flush=True)


def _values(sids):
    rows = []
    for sid in sids:
        f = D / "fastsurfer" / str(sid) / "stats" / "cat12_ch4.json"
        if f.exists():
            j = json.loads(f.read_text())
            rows.append({"IMAGEID": sid, "ch4": j["ch4_lh_gm_mm3"] + j["ch4_rh_gm_mm3"], "ch4_p": j["ch4_lh_pgm_mm3"] + j["ch4_rh_pgm_mm3"],
                         "ch4_lh": j["ch4_lh_gm_mm3"], "ch4_rh": j["ch4_rh_gm_mm3"], "tiv": j["cat12_tiv_mm3"], "iqr": j["cat12_iqr_pct"],
                         "nmm_bf": j.get("nmm_lh_basal_forebrain_mm3", np.nan) + j.get("nmm_rh_basal_forebrain_mm3", np.nan)})
    return pd.DataFrame(rows, columns=["IMAGEID", "ch4", "ch4_p", "ch4_lh", "ch4_rh", "tiv", "iqr", "nmm_bf"])


def _fast(sid):
    """The failed FAST measure from its own file (stats/ch4.json; the IDP table's vol_ch4_* is CAT12's once rebuilt)."""
    f = D / "fastsurfer" / str(sid) / "stats" / "ch4.json"
    j = json.loads(f.read_text()) if f.exists() else {}
    return j.get("ch4_lh_gm_mm3", np.nan) + j.get("ch4_rh_gm_mm3", np.nan)


def load(ppmi, out):
    """Analysis frames: one row per selected participant (HC, PD) and one per repeat scan, with the CAT12 values."""
    from pie.imaging.labels import _age_at, covariates

    s = pd.read_csv(out / "sids.csv")
    v = _values(s["IMAGEID"].unique())
    cov = covariates(ppmi)
    b = baseline()
    b["bf"] = b["vol_sclimbic_Left_Basal_Forebrain"] + b["vol_sclimbic_Right_Basal_Forebrain"]
    b["fast"] = [_fast(i) for i in b["IMAGEID"]]
    main = s[s["role"].isin(["hc", "pd"])].merge(b[["PATNO", "IMAGEID", "SCAN_DATE", "eTIV", "bf", "fast"]], on=["PATNO", "IMAGEID"], how="left")
    main = main.merge(cov[["PATNO", "SEX"]], on="PATNO", how="left")
    main["age"] = _age_at(cov, main["PATNO"], main["SCAN_DATE"])
    main = main.merge(v, on="IMAGEID", how="left").drop(columns=["PATNO", "IMAGEID", "SCAN_DATE"])
    rep = s[s["role"].eq("repeat")].merge(v, on="IMAGEID", how="left")
    rep["pair"] = rep.groupby(["PATNO", "day"]).ngroup()
    return main, rep.drop(columns=["PATNO", "IMAGEID", "day"])


def synthetic():
    """Fake values only: no real value, label or identifier."""
    rng = np.random.default_rng(5)
    n = 446
    role = np.where(np.arange(n) < 223, "hc", "pd")
    age = rng.normal(63, 9, n)
    tiv = rng.normal(1.5e6, 1.4e5, n)
    true = 300 * tiv / 1.5e6 - 3.0 * (age - 63) + rng.normal(0, 25, n)
    main = pd.DataFrame({"role": role, "age": age, "SEX": rng.integers(0, 2, n).astype(float), "tiv": tiv, "eTIV": tiv * 1.02,
                         "iqr": rng.normal(82, 6, n), "ch4": true, "ch4_p": true * 1.6, "ch4_lh": true * 0.7, "ch4_rh": true * 0.3,
                         "nmm_bf": true * 1.3 + rng.normal(0, 30, n), "bf": true * 1.2 + rng.normal(0, 40, n), "fast": rng.normal(60, 20, n)})
    t = rng.normal(300, 40, 29)
    rep = pd.DataFrame({"role": "repeat", "pair": np.repeat(np.arange(29), 2), "iqr": 85.0,
                        **{m: np.repeat(t, 2) * (1.6 if m == "ch4_p" else 1) + rng.normal(0, 12, 58) for m in MEASURES}})
    return main, rep


def _resid(y, X, fit):
    X = np.column_stack([np.ones(len(X))] + [np.asarray(X[c], float) for c in X])
    b = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
    return y - X @ b


def _r_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def _icc(rep, m, n_boot=B):
    from pie.stats.agreement import icc21

    w = rep.pivot_table(index="pair", columns=rep.groupby("pair").cumcount(), values=m).dropna().to_numpy(float)
    if len(w) < 5:
        return {"n_pairs": int(len(w)), "icc21": None}
    rng = np.random.default_rng(SEED)
    boot = [icc21(w[rng.integers(0, len(w), len(w))]) for _ in range(n_boot)]
    return {"n_pairs": int(len(w)), "icc21": float(icc21(w)), "ci95": list(np.percentile(boot, [2.5, 97.5]))}


def analyse(main, rep, n_boot=B, iqr_min=IQR_MIN):
    res = {"flow": {"selected_hc": int(main["role"].eq("hc").sum()), "selected_pd": int(main["role"].eq("pd").sum()),
                    "cat12_missing": int(main["ch4"].isna().sum()), "iqr_below_min": int((main["iqr"] < iqr_min).sum()),
                    "repeat_scans": int(len(rep)), "repeat_cat12_missing": int(rep["ch4"].isna().sum())}}
    m = main[(main["iqr"] >= iqr_min) & main[["ch4", "tiv", "age", "SEX", "bf", "eTIV"]].notna().all(axis=1)].reset_index(drop=True)
    rep = rep[rep["iqr"] >= iqr_min]
    hc, pdm = m["role"].eq("hc").to_numpy(), m["role"].eq("pd").to_numpy()
    res["flow"].update({"analysed_hc": int(hc.sum()), "analysed_pd": int(pdm.sum())})
    bf = _resid(m["bf"].to_numpy(float), m[["eTIV"]], np.ones(len(m), bool))
    out = {}
    for k in MEASURES:
        ok = m[k].notna().to_numpy()
        y = m[k].to_numpy(float)
        h = hc & ok
        ya = _resid(np.where(ok, y, 0), m[["tiv", "SEX"]], h)
        r = float(stats.pearsonr(ya[h], m["age"].to_numpy()[h])[0])
        rho = float(stats.spearmanr(_resid(np.where(ok, y, 0), m[["tiv"]], ok)[ok], bf[ok])[0])
        z = _resid(np.where(ok, y, 0), m[["tiv", "SEX", "age"]], h)          # HC fit: no PD label enters it
        zp = z[pdm & ok]
        d = (z[h].mean() - zp.mean()) / np.sqrt((z[h].var(ddof=1) + zp.var(ddof=1)) / 2)
        out[k] = {"label": MEASURES[k], "hc_age_r": r, "hc_age_r_ci95": _r_ci(r, int(h.sum())), "n_hc": int(h.sum()),
                  "rho_sclimbic_bf": rho, "n_rho": int(ok.sum()), "same_day": _icc(rep, k, n_boot),
                  "hc_minus_pd_d": float(d), "median_mm3": float(np.nanmedian(y)), "median_pct_tiv": float(np.nanmedian(y / m["tiv"]) * 100)}
    g = out["ch4"]
    res["measures"] = out
    res["gates"] = {"age": bool(g["hc_age_r"] <= GATE_AGE_R and g["hc_age_r_ci95"][1] < 0), "convergent": bool(g["rho_sclimbic_bf"] >= GATE_RHO),
                    "reliability": bool(g["same_day"]["icc21"] is not None and g["same_day"]["icc21"] >= GATE_ICC)}
    res["validated"] = all(res["gates"].values())
    res["descriptive"] = {"lh_rh_ratio_median": float(np.nanmedian(m["ch4_lh"] / m["ch4_rh"])),
                          "iqr_median": float(np.nanmedian(main["iqr"])), "iqr_below_70_pct": float((main["iqr"] < 70).mean() * 100),
                          "rho_with_fast_ch4": float(stats.spearmanr(m["ch4"], m["fast"], nan_policy="omit")[0]) if m["fast"].notna().sum() > 10 else None}
    if iqr_min < 70:                 # sensitivity: the stricter CAT12 quality cut (grade C- or better)
        res["descriptive"]["sensitivity_iqr_70"] = analyse(main, rep, 200, iqr_min=70)["measures"]["ch4"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "run", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        return select(a.ppmi, out)
    if a.step == "run":
        return run(out, a.workers)
    main_, rep = synthetic() if a.synthetic else load(a.ppmi, out)
    res = analyse(main_, rep, a.n_boot)
    (out / ("synthetic_ch4_cat12.json" if a.synthetic else "ch4_cat12_result.json")).write_text(json.dumps(res, indent=1, default=float))
    g = res["measures"]["ch4"]
    tag = "synthetic dry run: " if a.synthetic else ""
    print(f"{tag}HC age r {g['hc_age_r']:.3f} {np.round(g['hc_age_r_ci95'], 3).tolist()} (n {g['n_hc']}); rho ScLimbic BF {g['rho_sclimbic_bf']:.3f}; "
          f"same-day ICC {g['same_day']['icc21']}; gates {res['gates']} -> {'VALIDATED' if res['validated'] else 'NOT VALIDATED'}")
    return res


if __name__ == "__main__":
    main()
