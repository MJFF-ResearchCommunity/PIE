"""Ch4 v2 validation (documentation/preregistration_ch4_v2.md, DRAFT): configuration ``julich303-1mm`` (Julich-Brain v3.0.3
Ch4 map, CAT12 output at 1 mm), probability-weighted grey matter as the primary measure, gates on data the 13 Oct test
did not use.

    select    <out>/sids.csv: IXI healthy adults aged >= 50 with a T1 (age and symmetry gates), a seeded sample of PPMI
              baseline PD not selected on 13 Oct (convergence, reported), and the 29 PPMI same-day pairs (reliability)
    run       pie.imaging.cat12.ch4_volume(config="julich303-1mm") on every selected scan
    analyse   <out>/ch4_v2_result.json

    PYTHONPATH=. python scripts/ch4_v2_validation.py select --ixi-dir <IXI> --used <13 Oct sids.csv> --out <dir>
    PYTHONPATH=. python scripts/ch4_v2_validation.py run --ixi-dir <IXI> --out <dir> --workers 6
    PYTHONPATH=. python scripts/ch4_v2_validation.py analyse --ixi-dir <IXI> --out <dir>
    PYTHONPATH=. python scripts/ch4_v2_validation.py analyse --synthetic --out <dir>     # dry run: every value and label fake
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch4_cat12_validation import D, _icc, _r_ci, _resid, baseline, same_day_pairs  # noqa: E402  (frozen 13 Oct helpers)

CONFIG, SEED, B = "julich303-1mm", 20261016, 2000
AGE_MIN, IQR_MIN, N_PD = 50.0, 70.0, 223
GATE_AGE_R, GATE_ICC, SYMMETRY = -0.30, 0.75, (0.9, 1.2)
GATE_RHO = None            # DRAFT decision at the freeze: None = convergence reported without a gate, or a number (e.g. 0.30)
MEASURES = {"ch4_p": "Ch4 GM, probability-weighted (primary)", "ch4": "Ch4 GM, p >= 0.5", "nmm_bf": "neuromorphometrics basal forebrain"}
IXI_T1 = re.compile(r"^(IXI\d{3})-(Guys|HH|IOP)-\d{4}-T1\.nii\.gz$")


def ixi_scans(ixi_dir):
    """IXI T1 files -> IXI_ID, site (Guys 1.5 T Philips, HH 3 T Philips, IOP 1.5 T GE) and path."""
    rows = [(m[1], m[2], str(p)) for p in sorted(Path(ixi_dir).glob("IXI*-T1.nii.gz")) if (m := IXI_T1.match(p.name))]
    return pd.DataFrame(rows, columns=["IXI_ID", "site", "nii"])


def ixi_demographics(ixi_dir):
    d = pd.read_excel(Path(ixi_dir) / "IXI.xls")
    d = d.assign(IXI_ID=d["IXI_ID"].map(lambda i: f"IXI{int(i):03d}"), age=pd.to_numeric(d["AGE"], errors="coerce"),
                 SEX=d["SEX_ID (1=m, 2=f)"].map({1: 1.0, 2: 0.0}))
    return d.dropna(subset=["age"]).drop_duplicates("IXI_ID")[["IXI_ID", "age", "SEX"]]


def fresh_pd(used, ppmi, n=N_PD):
    """Seeded sample of PPMI baseline PD whose scans and participants the 13 Oct test did not select."""
    from pie.imaging.labels import covariates

    b = baseline().merge(covariates(ppmi)[["PATNO", "COHORT"]], on="PATNO", how="left")
    pdm = b[b["COHORT"].eq("Parkinson's Disease") & ~b["PATNO"].isin(used["PATNO"])]
    return pdm.sample(n=min(n, len(pdm)), random_state=SEED)[["PATNO", "IMAGEID"]]


def select(ixi_dir, used_csv, ppmi, out):
    ixi = ixi_scans(ixi_dir).merge(ixi_demographics(ixi_dir), on="IXI_ID")
    ixi = ixi[ixi["age"] >= AGE_MIN]
    pdm, pairs = fresh_pd(pd.read_csv(used_csv), ppmi), same_day_pairs()
    rows = pd.concat([ixi[["IXI_ID"]].rename(columns={"IXI_ID": "IMAGEID"}).assign(PATNO=-1, role="ixi"),
                      pdm.assign(role="pd"), pairs.assign(role="repeat")], ignore_index=True)
    rows.to_csv(out / "sids.csv", index=False)
    print(f"selected: {len(ixi)} IXI aged >= {AGE_MIN:g}, {len(pdm)} fresh PPMI PD, {len(pairs) // 2} same-day pairs")


def _job(row, ixi_dir):
    if row["role"] == "ixi":
        nii = ixi_scans(ixi_dir).set_index("IXI_ID").loc[row["IMAGEID"], "nii"]
        return str(row["IMAGEID"]), str(D.parent / "derived_ixi" / "cat12"), str(nii)
    return str(row["IMAGEID"]), str(D / "fastsurfer"), str(D / "nifti" / str(row["PATNO"]) / f"{row['IMAGEID']}_T1w.nii.gz")


def _run_one(job):
    from pie.imaging.cat12 import ch4_volume

    sid, subjects_dir, nii = job
    try:
        Path(subjects_dir).mkdir(parents=True, exist_ok=True)
        ch4_volume(subjects_dir, sid, nii, threads=1, config=CONFIG)
        return sid, ""
    except Exception as e:
        return sid, f"{type(e).__name__}: {str(e)[:200]}"


def run(ixi_dir, out, workers):
    s = pd.read_csv(out / "sids.csv").drop_duplicates("IMAGEID")
    jobs = [_job(r, ixi_dir) for _, r in s.iterrows()]
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


def _values(subjects_dir, sids):
    from pie.imaging.cat12 import result_name

    rows = []
    for sid in sids:
        f = Path(subjects_dir) / str(sid) / "stats" / result_name(CONFIG)
        if f.exists():
            j = json.loads(f.read_text())
            rows.append({"IMAGEID": str(sid), "ch4": j["ch4_lh_gm_mm3"] + j["ch4_rh_gm_mm3"], "ch4_p": j["ch4_lh_pgm_mm3"] + j["ch4_rh_pgm_mm3"],
                         "ch4_p_lh": j["ch4_lh_pgm_mm3"], "ch4_p_rh": j["ch4_rh_pgm_mm3"], "tiv": j["cat12_tiv_mm3"],
                         "iqr": j["cat12_iqr_pct"], "nmm_bf": j.get("nmm_lh_basal_forebrain_mm3", np.nan) + j.get("nmm_rh_basal_forebrain_mm3", np.nan)})
    return pd.DataFrame(rows, columns=["IMAGEID", "ch4", "ch4_p", "ch4_p_lh", "ch4_p_rh", "tiv", "iqr", "nmm_bf"])


def load(ixi_dir, out):
    s = pd.read_csv(out / "sids.csv", dtype={"IMAGEID": str})
    ixi = s[s["role"].eq("ixi")].merge(ixi_scans(ixi_dir)[["IXI_ID", "site"]].rename(columns={"IXI_ID": "IMAGEID"}), on="IMAGEID")
    ixi = ixi.merge(ixi_demographics(ixi_dir).rename(columns={"IXI_ID": "IMAGEID"}), on="IMAGEID")
    ixi = ixi.merge(_values(D.parent / "derived_ixi" / "cat12", ixi["IMAGEID"]), on="IMAGEID", how="left")
    b = baseline().assign(IMAGEID=lambda x: x["IMAGEID"].astype(str))
    b["bf"] = b["vol_sclimbic_Left_Basal_Forebrain"] + b["vol_sclimbic_Right_Basal_Forebrain"]
    pdm = s[s["role"].eq("pd")].merge(b[["IMAGEID", "eTIV", "bf"]], on="IMAGEID", how="left")
    pdm = pdm.merge(_values(D / "fastsurfer", pdm["IMAGEID"]), on="IMAGEID", how="left")
    rep = s[s["role"].eq("repeat")].merge(_values(D / "fastsurfer", s.loc[s["role"].eq("repeat"), "IMAGEID"]), on="IMAGEID", how="left")
    rep["pair"] = rep.groupby(["PATNO", "day"]).ngroup()
    return ixi.drop(columns=["PATNO", "IMAGEID"]), pdm.drop(columns=["PATNO", "IMAGEID"]), rep.drop(columns=["PATNO", "IMAGEID", "day"])


def analyse(ixi, pdm, rep, n_boot=B, iqr_min=IQR_MIN):
    res = {"config": CONFIG, "flow": {"ixi_selected": int(len(ixi)), "ixi_missing": int(ixi["ch4_p"].isna().sum()),
                                      "pd_selected": int(len(pdm)), "pd_missing": int(pdm["ch4_p"].isna().sum()),
                                      "repeat_scans": int(len(rep)), "repeat_missing": int(rep["ch4_p"].isna().sum())}}
    c = ixi[(ixi["iqr"] >= iqr_min) & ixi[["ch4_p", "tiv", "age", "SEX"]].notna().all(axis=1)].reset_index(drop=True)
    p = pdm[(pdm["iqr"] >= iqr_min) & pdm[["ch4_p", "tiv", "bf", "eTIV"]].notna().all(axis=1)].reset_index(drop=True)
    rep = rep[rep["iqr"] >= iqr_min]
    res["flow"].update({"ixi_analysed": int(len(c)), "ixi_below_iqr": int((ixi["iqr"] < iqr_min).sum()),
                        "pd_analysed": int(len(p)), "pd_below_iqr": int((pdm["iqr"] < iqr_min).sum())})
    every = np.ones(len(c), bool)
    out = {}
    for k in MEASURES:
        y = _resid(c[k].to_numpy(float), c[["tiv", "SEX"]], every)
        r = float(stats.pearsonr(y, c["age"])[0])
        rho = float(stats.spearmanr(_resid(p[k].to_numpy(float), p[["tiv"]], np.ones(len(p), bool)),
                                    _resid(p["bf"].to_numpy(float), p[["eTIV"]], np.ones(len(p), bool)))[0]) if len(p) > 10 else None
        by_site = {s: float(stats.pearsonr(y[c["site"].eq(s).to_numpy()], c.loc[c["site"].eq(s), "age"])[0])
                   for s in sorted(c["site"].unique()) if c["site"].eq(s).sum() >= 20}
        out[k] = {"label": MEASURES[k], "age_r": r, "age_r_ci95": _r_ci(r, len(c)), "n_age": int(len(c)), "age_r_by_site": by_site,
                  "rho_sclimbic_bf_pd": rho, "n_rho": int(len(p)), "same_day": _icc(rep, k, n_boot),
                  "median_mm3": float(np.nanmedian(c[k])), "median_pct_tiv": float(np.nanmedian(c[k] / c["tiv"]) * 100)}
    g = out["ch4_p"]
    lr = float(np.nanmedian(c["ch4_p_lh"] / c["ch4_p_rh"]))
    res["measures"] = out
    res["gates"] = {"symmetry": bool(SYMMETRY[0] <= lr <= SYMMETRY[1]),
                    "age": bool(g["age_r"] <= GATE_AGE_R and g["age_r_ci95"][1] < 0),
                    "reliability": bool(g["same_day"]["icc21"] is not None and g["same_day"]["icc21"] >= GATE_ICC)}
    if GATE_RHO is not None:
        res["gates"]["convergent"] = bool(g["rho_sclimbic_bf_pd"] is not None and g["rho_sclimbic_bf_pd"] >= GATE_RHO)
    res["validated"] = all(res["gates"].values())
    res["descriptive"] = {"lh_rh_ratio_median": lr, "iqr_median_ixi": float(np.nanmedian(ixi["iqr"])),
                          "iqr_median_pd": float(np.nanmedian(pdm["iqr"])), "age_range_ixi": [float(c["age"].min()), float(c["age"].max())] if len(c) else None}
    if iqr_min > 60:                 # sensitivity: the 13 Oct quality cut
        res["descriptive"]["sensitivity_iqr_60"] = analyse(ixi, pdm, rep, 200, iqr_min=60)["measures"]["ch4_p"]
    return res


def synthetic():
    """Fake values only: no real value, label or identifier."""
    rng = np.random.default_rng(16)
    n = 280
    age, tiv = rng.uniform(50, 86, n), rng.normal(1.5e6, 1.4e5, n)
    true = 450 * tiv / 1.5e6 - 3.0 * (age - 65) + rng.normal(0, 30, n)
    ixi = pd.DataFrame({"role": "ixi", "site": rng.choice(["A", "B", "C"], n), "age": age, "SEX": rng.integers(0, 2, n).astype(float),
                        "tiv": tiv, "iqr": rng.normal(82, 6, n), "ch4_p": true, "ch4": true * 0.6, "nmm_bf": true * 1.4 + rng.normal(0, 30, n),
                        "ch4_p_lh": true * 0.52, "ch4_p_rh": true * 0.48})
    m = 223
    t = rng.normal(450, 50, m)
    pdm = pd.DataFrame({"role": "pd", "tiv": rng.normal(1.5e6, 1.4e5, m), "eTIV": rng.normal(1.5e6, 1.4e5, m), "iqr": rng.normal(82, 6, m),
                        "ch4_p": t, "ch4": t * 0.6, "nmm_bf": t * 1.4, "bf": t * 1.3 + rng.normal(0, 60, m), "ch4_p_lh": t * 0.52, "ch4_p_rh": t * 0.48})
    s = rng.normal(450, 50, 29)
    rep = pd.DataFrame({"role": "repeat", "pair": np.repeat(np.arange(29), 2), "iqr": 85.0,
                        **{k: np.repeat(s, 2) * f + rng.normal(0, 15, 58) for k, f in (("ch4_p", 1), ("ch4", 0.6), ("nmm_bf", 1.4))}})
    return ixi, pdm, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "run", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ixi-dir", help="folder with the IXI T1 NIfTIs and IXI.xls")
    ap.add_argument("--used", help="the 13 Oct test's sids.csv (its participants are excluded from the fresh PD sample)")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        return select(a.ixi_dir, a.used, a.ppmi, out)
    if a.step == "run":
        return run(a.ixi_dir, out, a.workers)
    ixi, pdm, rep = synthetic() if a.synthetic else load(a.ixi_dir, out)
    res = analyse(ixi, pdm, rep, a.n_boot)
    (out / ("synthetic_ch4_v2.json" if a.synthetic else "ch4_v2_result.json")).write_text(json.dumps(res, indent=1, default=float))
    g = res["measures"]["ch4_p"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}age r {g['age_r']:.3f} {np.round(g['age_r_ci95'], 3).tolist()} (n {g['n_age']}); "
          f"left/right {res['descriptive']['lh_rh_ratio_median']:.2f}; ICC {g['same_day']['icc21']}; rho ScLimbic (PD) {g['rho_sclimbic_bf_pd']}; "
          f"gates {res['gates']} -> {'VALIDATED' if res['validated'] else 'NOT VALIDATED'}")
    return res


if __name__ == "__main__":
    main()
