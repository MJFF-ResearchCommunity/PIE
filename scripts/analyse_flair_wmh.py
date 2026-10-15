"""Pre-registered analysis (documentation/preregistration_flair_wmh.md): TrUE-Net WMH volume with PIE's FastSurfer
preparation (``flair --truenet-prep fastsurfer``), as a vascular covariate.

G1, technical: against manual labels on the MICCAI WMH Segmentation Challenge 2017 test set (110 subjects, 5 scanners).
   Reads a per-subject table (one row per test subject, metrics from ``mask_metrics``). Pass: median Dice >= 0.70 and
   volume ICC(2,1) >= 0.85.
G2, biological, on PPMI (one baseline QC-passing scan per participant): partial Spearman of log WMH with age (pass:
   rho >= 0.35 and 95 % CI lower bound > 0.18) and with a vascular-risk count (pass: CI lower bound > 0), adjusted for
   sex, eTIV, FLAIR batch (vendor x 2D/3D) and cohort (+ age for the risk count).

    PYTHONPATH=. python scripts/analyse_flair_wmh.py g1 --metrics <wmh_challenge_test_metrics.csv> --out <dir>
    PYTHONPATH=. python scripts/analyse_flair_wmh.py g2 --features <flair_truenet_fs>/flair_features.csv \
        --idps <derived>/fastsurfer_idps.csv --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_flair_wmh.py --synthetic --out <dir>      # dry run: every value and label is fake
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

from pie.stats.agreement import agreement, icc21
from pie.stats.small_sample import bootstrap_partial_correlation

SEED, B = 20261015, 2000
DICE_MIN, ICC_MIN = 0.70, 0.85                 # G1
AGE_RHO_MIN, AGE_CI_FLOOR = 0.35, 0.18         # G2a; 0.18 = the superseded threshold method's age rho
SCANNERS = ["Utrecht_PhilipsAchieva3T", "Singapore_SiemensTrioTim3T", "Amsterdam_GE3T", "Amsterdam_GE1T5",
            "Amsterdam_PhilipsIngenuityPETMR3T"]
UNSEEN = SCANNERS[3:]                          # not in the challenge training data, so not in TrUE-Net's mwsc training
Y = "wmh_log_mm3"                              # log1p(WMH mm^3), PIE's column

# G2 vascular-risk items: documented on or before the scan date in the Medical Conditions Log (MHTERM, diagnosis year
# <= scan year or missing) or the Concomitant Medication Log (coded indication CMINDC, or the indication text where the
# older EDC left the code blank; start month <= scan month or missing)
TERMS = {"hypertension": r"hypertens|\bhtn\b|high blood pressure|elevated blood pressure",
         "diabetes": r"diabet|\bt[12]dm\b|\bniddm\b|\biddm\b|\bdm2?\b",
         "hyperlipidaemia": r"hyperlipid|hypercholest|dyslipid|cholesterol|hypertriglycerid"}
NOT_TERMS = r"pulmonary|ocular|portal|intracranial|white coat|gestational|pregnan|family|pre-?hypertens|pre-?diabet|insipidus|borderline"
CMINDC = {"hypertension": 14, "diabetes": 11, "hyperlipidaemia": 13}   # PPMI code list, CONMED.CMINDC
ITEMS = list(TERMS)


def mask_metrics(ref, pred, vox_mm3=1.0):
    """The challenge's per-subject metrics (Kuijf et al. 2019; hjkuijf/wmhchallenge evaluation.py @ cd4f691): ``ref``
    label 1 = WMH, label 2 = other pathology (removed from the prediction, scored as background); ``pred`` binary on the
    same grid. Lesions are 3D 26-connected components. H95 is not computed."""
    ref = np.asarray(ref)
    wmh, pred = ref == 1, np.asarray(pred).astype(bool) & (ref != 2)
    s = wmh.sum() + pred.sum()
    full = np.ones((3, 3, 3))
    lr, nr = ndimage.label(wmh, full)
    lp, npred = ndimage.label(pred, full)
    recall = np.count_nonzero(np.unique(lr[pred])) / nr if nr else 1.0
    precision = np.count_nonzero(np.unique(lp[wmh])) / npred if npred else 1.0
    return {"dice": float(2 * (wmh & pred).sum() / s) if s else np.nan, "lesion_recall": float(recall),
            "lesion_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "avd_pct": float(abs(int(pred.sum()) - int(wmh.sum())) / wmh.sum() * 100) if wmh.any() else np.nan,
            "vol_ref_mm3": float(wmh.sum() * vox_mm3), "vol_pred_mm3": float(pred.sum() * vox_mm3)}


def _ci(x):
    return [float(v) for v in np.nanquantile(x, [0.025, 0.975])]


def g1(m, n_boot=B):
    """G1 on the per-subject test table. A subject that failed to process (non-empty ``error``) counts as Dice 0 and is
    left out of the volume agreement (and counted)."""
    t = m.copy()
    failed = t["error"].fillna("").astype(str).ne("")
    t.loc[failed, ["dice", "lesion_recall", "lesion_f1"]] = 0.0
    t.loc[failed, ["vol_pred_mm3", "avd_pct"]] = np.nan
    rng = np.random.default_rng(SEED)
    dice, vol = t["dice"].to_numpy(float), t[["vol_pred_mm3", "vol_ref_mm3"]].to_numpy(float) / 1e3     # mL
    boot = [rng.integers(0, len(t), len(t)) for _ in range(n_boot)]

    def icc(v):
        v = v[np.isfinite(v).all(axis=1)]
        return icc21(v) if len(v) >= 3 else np.nan

    def summary(g):
        v = g[["vol_pred_mm3", "vol_ref_mm3"]].to_numpy(float) / 1e3
        return {"n": int(len(g)), "median_dice": float(g["dice"].median()), "icc21": icc(v),
                **{f"mean_{k}": float(g[k].mean()) for k in ("dice", "lesion_recall", "lesion_f1", "avd_pct")}}

    res = {"n": int(len(t)), "n_failed": int(failed.sum()), **summary(t),
           "median_dice_ci95": _ci([np.median(dice[i]) for i in boot]), "icc21_ci95": _ci([icc(vol[i]) for i in boot]),
           "volume_ml": agreement(vol[:, 0], vol[:, 1]), "log_volume": agreement(np.log1p(vol[:, 0]), np.log1p(vol[:, 1])),
           "by_scanner": {s: summary(g) for s, g in t.groupby("scanner")},
           "unseen_scanners": summary(t[t["scanner"].isin(UNSEEN)])}
    res["pass"] = bool(res["median_dice"] >= DICE_MIN and res["icc21"] >= ICC_MIN)
    return res


def vascular(ppmi, scans):
    """Per participant (index of ``scans``: patno -> scan date): the three risk items (1/0; NaN when the participant has
    no Medical Conditions Log and no Concomitant Medication Log record at all), their count ``vrs``, supine systolic BP
    at the vital-signs visit nearest the scan (within 365 days) and ever-smoking (PPMI Online, >= 100 cigarettes)."""
    from pie.imaging.labels import _latest, _month

    s = pd.to_datetime(scans)
    mh = _latest(ppmi, "Medical_History", "Medical_Conditions_Log")
    cm = _latest(ppmi, "Medical_History", "Concomitant_Medication_Log")
    mh, cm = mh[mh["PATNO"].isin(s.index)], cm[cm["PATNO"].isin(s.index)]
    mh_on_time = ~(pd.to_numeric(mh["MHDIAGYR"], errors="coerce") > mh["PATNO"].map(s).dt.year)
    cm_on_time = ~(_month(cm["STARTDT"]) > cm["PATNO"].map(s))
    mt, ct = mh["MHTERM"].fillna("").str.lower(), cm["CMINDC_TEXT"].fillna("").str.lower()
    code = pd.to_numeric(cm["CMINDC"], errors="coerce")
    out = pd.DataFrame(index=s.index)
    for k, pat in TERMS.items():
        hit = set(mh.loc[mh_on_time & mt.str.contains(pat) & ~mt.str.contains(NOT_TERMS), "PATNO"]) | set(
            cm.loc[cm_on_time & (code.eq(CMINDC[k]) | (ct.str.contains(pat) & ~ct.str.contains(NOT_TERMS))), "PATNO"])
        out[k] = out.index.isin(hit).astype(float)
    out.loc[~out.index.isin(set(mh["PATNO"]) | set(cm["PATNO"])), ITEMS] = np.nan
    out["vrs"] = out[ITEMS].sum(axis=1, min_count=len(ITEMS))
    vs = _latest(ppmi, "Medical_History", "Vital_Signs")
    vs = vs[vs["PATNO"].isin(s.index)].assign(gap=lambda v: (_month(v["INFODT"]) - v["PATNO"].map(s)).abs(),
                                               sbp=lambda v: pd.to_numeric(v["SYSSUP"], errors="coerce"))
    vs = vs[vs["sbp"].between(70, 250) & (vs["gap"] <= pd.Timedelta(days=365))].sort_values("gap").drop_duplicates("PATNO")
    out["sbp"] = vs.set_index("PATNO")["sbp"]
    sm = _latest(ppmi, "PPMI_Online", "Smoking_history__Online_")
    sm = sm[sm["PATNO"].isin(s.index)].assign(a=lambda v: pd.to_numeric(v["FIVE_PACKS_IN_LIFE_OL"], errors="coerce"))
    ever = sm.groupby("PATNO")["a"].agg(lambda a: 1.0 if (a == 1).any() else (0.0 if (a == 0).any() else np.nan))
    out["smoking_ever"] = ever                    # codes 2 and 3 are not in the code list: missing
    return out


def load(features, idps, ppmi):
    """PIE's FLAIR rows (one per participant: the cohort run's session nearest the earliest finished T1), every
    exclusion counted by reason; eTIV and T1 QC from the FastSurfer IDP table; age, sex, cohort; vascular risk."""
    from pie.imaging.labels import _age_at, covariates
    from pie.imaging.manifest import QC, _flag, _vendor

    d = pd.read_csv(features, low_memory=False, dtype={"fs_image_id": str})
    d = d.assign(date=pd.to_datetime(d["acquisition_date"], errors="coerce")).sort_values("date").drop_duplicates("patno")
    flow = {"participants": int(len(d))}
    err = d["error"].fillna("").astype(str)
    flow["excluded_processing_error"] = err[err.ne("")].str.split(":").str[0].value_counts().to_dict()
    d = d[err.eq("")]
    same = d["wmh_method"].eq("truenet") & d["truenet_prep"].eq("fastsurfer") & d["truenet_model"].eq("mwsc")
    flow["excluded_not_truenet_fastsurfer_mwsc"] = int((~same).sum())
    d = d[same].copy()
    d["batch"] = _vendor(d["manufacturer"]) + "_" + np.where(_flag(d, "flair_3d"), "3D", "2D")
    qc = QC["flair"](d)
    flow["qc_pass_rate_by_batch"] = qc.groupby(d["batch"]).mean().round(3).to_dict()
    flow["excluded_flair_qc"] = {"total": int((~qc).sum()), "registration_mi": int((~(d["reg_flair_t1_mi"] < -0.2)).sum()),
                                 "wm_volume": int((~(d["wm_mm3"] > 2e5)).sum()), "flair_wm_mad": int((~(d["flair_wm_mad"] > 0)).sum()),
                                 "si_coverage_lt_130mm": int((pd.to_numeric(d["flair_si_mm"], errors="coerce") < 130).sum())}
    d = d[qc]
    t1 = pd.read_csv(idps, low_memory=False, dtype={"IMAGEID": str})
    t1 = t1.assign(t1_ok=QC["t1"](t1))[["IMAGEID", "eTIV", "t1_ok", "vol_WM_hypointensities"]]
    d = d.merge(t1, left_on="fs_image_id", right_on="IMAGEID", how="left")
    bad = ~(d["t1_ok"].eq(True) & (pd.to_numeric(d["eTIV"], errors="coerce") > 0))
    flow["excluded_t1_qc_or_no_etiv"] = int(bad.sum())
    d = d[~bad]
    cov = covariates(Path(ppmi))
    d = d.merge(cov[["PATNO", "SEX", "COHORT"]], left_on="patno", right_on="PATNO", how="left")
    d["age"] = _age_at(cov, d["patno"], d["date"])
    d["age"] = d["age"].where(d["age"].between(18, 100))
    d["sex"], d["cohort"] = pd.to_numeric(d["SEX"], errors="coerce"), d["COHORT"]
    miss = d[["age", "sex", "cohort"]].isna().any(axis=1)
    flow["excluded_no_age_sex_or_cohort"] = int(miss.sum())
    d = d[~miss].copy()
    d = d.join(vascular(ppmi, d.set_index("patno")["date"]), on="patno")
    flow["vrs_missing"] = int(d["vrs"].isna().sum())
    d["fs_hypo_log"] = np.log1p(d["vol_WM_hypointensities"])
    d["pv_log"], d["deep_log"] = np.log1p(d["wmh_pv_mm3"]), np.log1p(d["wmh_deep_mm3"])
    return d.reset_index(drop=True), flow


def _pc(t, x, extra=(), y=Y, n_boot=B):
    """Partial Spearman of ``x`` with ``y`` given sex, eTIV, FLAIR batch and cohort (+ ``extra``); percentile bootstrap
    95 % CI; one-sided P for a positive association."""
    t = t.dropna(subset=[x, y, *extra])
    dm = pd.get_dummies(t[["batch", "cohort"]], drop_first=True, dtype=float)
    t = pd.concat([t, dm], axis=1)
    cov = [c for c in ["sex", "eTIV", *extra, *dm.columns] if t[c].nunique() > 1]
    r = bootstrap_partial_correlation(t, x, y, cov, method="spearman", n_boot=n_boot, seed=SEED)
    r["p_one_sided"] = r["p"] / 2 if r["r"] > 0 else 1 - r["p"] / 2
    return r


def analyse_g2(d, flow, n_boot=B):
    res = {"flow": {**flow, "analysed": int(len(d)), "by_cohort": d["cohort"].value_counts().to_dict(),
                    "by_batch": d["batch"].value_counts().to_dict(),
                    "vrs_distribution": {str(k): int(v) for k, v in d["vrs"].value_counts(dropna=False).sort_index().items()}}}
    age, vrs = _pc(d, "age", n_boot=n_boot), _pc(d, "vrs", extra=("age",), n_boot=n_boot)
    res["g2a_age"] = {**age, "pass": bool(age["r"] >= AGE_RHO_MIN and age["ci_low"] > AGE_CI_FLOOR)}
    res["g2b_vascular"] = {**vrs, "pass": bool(vrs["ci_low"] > 0)}
    res["pass"] = res["g2a_age"]["pass"] and res["g2b_vascular"]["pass"]
    acq = np.where(d["batch"].str.endswith("3D"), "3D", "2D")
    enough = lambda g: len(g) >= 30                                   # noqa: E731
    res["secondary"] = {
        "age_unadjusted_spearman": float(spearmanr(d["age"], d[Y])[0]),
        "age_by_cohort": {c: _pc(g, "age", n_boot=n_boot) for c, g in d.groupby("cohort") if enough(g)},
        "age_by_acquisition": {k: _pc(g, "age", n_boot=n_boot) for k, g in d.groupby(acq) if enough(g)},
        "age_fastsurfer_wm_hypointensities": _pc(d, "age", y="fs_hypo_log", n_boot=n_boot),
        "age_periventricular": _pc(d, "age", y="pv_log", n_boot=n_boot),
        "age_deep": _pc(d, "age", y="deep_log", n_boot=n_boot),
        "vascular_items": {k: _pc(d, k, extra=("age",), n_boot=n_boot) for k in ITEMS},
        "vrs_3d_only": _pc(d[acq == "3D"], "vrs", extra=("age",), n_boot=n_boot) if enough(d[acq == "3D"]) else None,
        "systolic_bp": _pc(d, "sbp", extra=("age",), n_boot=n_boot),
        "smoking_ever": _pc(d, "smoking_ever", extra=("age",), n_boot=n_boot) if d["smoking_ever"].notna().sum() >= 30 else None}
    return res


def synthetic_g1(dice_mean=0.77, vol_sd=0.25, n_fail=1):
    """Fully synthetic per-subject G1 table (no real value): 30/30/30/10/10 subjects as the test set."""
    rng = np.random.default_rng(1)
    scanner = np.repeat(SCANNERS, [30, 30, 30, 10, 10])
    ref = np.exp(rng.normal(np.log(8000), 1.0, len(scanner)))
    pred = 0.9 * ref * np.exp(rng.normal(0, vol_sd, len(scanner)))
    dice = np.clip(rng.normal(dice_mean, 0.08, len(scanner)), 0, 1)
    m = pd.DataFrame({"subject": [f"synthetic/{i}" for i in range(len(scanner))], "scanner": scanner, "dice": dice,
                      "lesion_recall": np.clip(dice - 0.03, 0, 1), "lesion_f1": np.clip(dice - 0.07, 0, 1),
                      "avd_pct": 100 * np.abs(pred - ref) / ref, "vol_ref_mm3": ref, "vol_pred_mm3": pred, "error": ""})
    m.loc[: n_fail - 1, "error"] = "RuntimeError: synthetic failure"
    return m


def synthetic_g2(n=1100, age_slope=0.06, vrs_slope=0.15):
    """Fully synthetic PPMI-like participants (no real value, label or identifier)."""
    rng = np.random.default_rng(2)
    age, sex = rng.normal(65, 8.7, n), rng.integers(1, 3, n).astype(float)
    items = {"hypertension": rng.random(n) < 0.4, "diabetes": rng.random(n) < 0.1, "hyperlipidaemia": rng.random(n) < 0.4}
    vrs = sum(v.astype(float) for v in items.values())
    y = 7 + age_slope * (age - 65) + vrs_slope * vrs + rng.normal(0, 1, n)
    d = pd.DataFrame({"patno": np.arange(n), "age": age, "sex": sex, "eTIV": rng.normal(1.5e6, 1.5e5, n),
                      "batch": rng.choice(["SIEMENS_3D", "SIEMENS_2D", "GE_2D", "PHILIPS_2D"], n), Y: y,
                      "cohort": rng.choice(["Parkinson's Disease", "Prodromal", "Healthy Control"], n, p=[0.4, 0.45, 0.15]),
                      **{k: v.astype(float) for k, v in items.items()}, "vrs": vrs,
                      "sbp": 130 + 8 * items["hypertension"] + rng.normal(0, 15, n),
                      "smoking_ever": np.where(rng.random(n) < 0.35, (rng.random(n) < 0.4).astype(float), np.nan),
                      "fs_hypo_log": y + rng.normal(0, 1, n), "pv_log": y - 0.3, "deep_log": y - 2 + rng.normal(0, 1, n)})
    return d, {"synthetic": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", choices=["g1", "g2"])
    ap.add_argument("--metrics", help="G1 per-subject table (columns in the pre-registration)")
    ap.add_argument("--features", help="PIE flair_features.csv of the TrUE-Net FastSurfer-prep cohort run")
    ap.add_argument("--idps", help="fastsurfer_idps.csv (eTIV, T1 QC, vol_WM_hypointensities)")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run of G1 and G2 on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    if a.synthetic:
        res = {"g1": g1(synthetic_g1(), a.n_boot), "g2": analyse_g2(*synthetic_g2(), a.n_boot)}
    elif a.mode == "g1":
        res = {"g1": g1(pd.read_csv(a.metrics), a.n_boot)}
    elif a.mode == "g2":
        res = {"g2": analyse_g2(*load(a.features, a.idps, a.ppmi), a.n_boot)}
    else:
        ap.error("give g1, g2 or --synthetic")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "synthetic_result.json" if a.synthetic else f"wmh_{a.mode}_result.json"
    (out / name).write_text(json.dumps(res, indent=1, default=float))
    tag = "synthetic dry run: " if a.synthetic else ""
    if "g1" in res:
        r = res["g1"]
        ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"                     # noqa: E731
        print(f"{tag}G1 median Dice {r['median_dice']:.3f} {ci(r['median_dice_ci95'])}, volume ICC {r['icc21']:.3f} "
              f"{ci(r['icc21_ci95'])} (n {r['n']}, failed {r['n_failed']}) -> {'PASS' if r['pass'] else 'FAIL'}")
    if "g2" in res:
        r = res["g2"]
        for k in ("g2a_age", "g2b_vascular"):
            print(f"{tag}{k} partial rho {r[k]['r']:.3f} [{r[k]['ci_low']:.3f}, {r[k]['ci_high']:.3f}] (n {r[k]['n']}) "
                  f"-> {'PASS' if r[k]['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
