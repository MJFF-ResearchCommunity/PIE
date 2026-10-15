"""Pre-registered first harmonisation stage (documentation/preregistration_harmonisation_ppmi.md): spline-age ComBat
(ComBat-GAM style, ``harmonize.combat``) across PPMI's own T1 acquisition batches (vendor x protocol phase x field
strength) on healthy controls' baseline FastSurfer volumes.

Gates: (1) batch classifiers (multinomial logistic and random forest) on in-fold harmonised, covariate-residualised
volumes reach macro one-vs-rest AUROC <= 0.60, each the mean of 10 repeats of 5-fold CV; (2) the controls' age
slope of hippocampus, thalamus and lateral ventricles stays within 10 % of the unharmonised within-batch slope;
(3) protecting diagnosis does not inflate PD - HC differences: every |d_protected - d_within-batch| <= 0.10 SD.

    PYTHONPATH=. python scripts/analyse_harmonisation_ppmi.py --derived Imaging/derived --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_harmonisation_ppmi.py --synthetic --out <dir>     # dry run: every value is fake
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pie.imaging.harmonize import combat, combat_apply  # noqa: E402

SEED, FOLDS, REPEATS, MIN_HC = 20261015, 5, 10, 10
GATE_AUROC, GATE_AGE_REL, GATE_D = 0.60, 0.10, 0.10
COVARS = ["age", "sex", "eTIV"]                       # protected; age as a cubic B-spline (4 df)
ASEG = ["Lateral_Ventricle", "Inf_Lat_Vent", "Cerebellum_White_Matter", "Cerebellum_Cortex", "Thalamus", "Caudate", "Putamen",
        "Pallidum", "Hippocampus", "Amygdala", "Accumbens_area", "VentralDC", "Cerebral_White_Matter"]
DKT = ["caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
       "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal",
       "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
       "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
       "superiorparietal", "superiortemporal", "supramarginal", "transversetemporal", "insula"]
FEATURES = ([f"vol_{s}_{a}" for a in ASEG for s in ("Left", "Right")] + ["vol_3rd_Ventricle", "vol_4th_Ventricle", "vol_Brain_Stem"]
            + [f"vol_ctx_{h}_{r}" for h in ("lh", "rh") for r in DKT])                       # 91 volumes
AGE_SET = ["Hippocampus", "Thalamus", "Lateral_Ventricle"]                                    # left + right


def load(derived, ppmi):
    """Earliest QC-passing, real-dated T1 per participant; cohort, age, sex, eTIV and the acquisition batch."""
    from pie.imaging.labels import _age_at, covariates
    from pie.imaging.manifest import _vendor, t1_visits

    v = t1_visits(derived)
    flow = {"t1_participants": int(v["PATNO"].nunique())}
    v = v[v["t1_qc_pass"] & v["SCAN_DATE"].notna()].sort_values(["PATNO", "SCAN_DATE"]).drop_duplicates("PATNO")
    cov = covariates(ppmi)
    v = v.merge(cov[["PATNO", "COHORT", "SEX"]], on="PATNO", how="left")
    v["dx"] = v["COHORT"].map({"Healthy Control": "HC", "Parkinson's Disease": "PD"})
    flow["baseline_usable"] = v["dx"].value_counts().to_dict()
    v["age"] = _age_at(cov, v["PATNO"], v["SCAN_DATE"])
    v["sex"] = pd.to_numeric(v["SEX"], errors="coerce")
    fs = pd.to_numeric(v["MagneticFieldStrength"], errors="coerce")
    field = pd.Series(np.where(fs > 100, fs / 10000, fs), index=v.index).round(1)
    v["batch"] = (_vendor(v["Manufacturer"].fillna("")) + "|PPMI" + v["protocol_phase"].map({1: "1", 2: "2"}).astype(str)
                  + "|" + field.astype(str) + "T")
    v.loc[v["protocol_phase"].isna() | field.isna(), "batch"] = np.nan
    v = v[v["dx"].notna()]
    bad = v[["batch", "age", "sex", "eTIV", *FEATURES]].isna().any(axis=1)
    flow["excluded_missing_batch_covariate_or_volume"] = v[bad]["dx"].value_counts().to_dict()
    return v[~bad].reset_index(drop=True), flow


def synthetic(n_hc=230, n_pd=900, seed=1):
    """Fully synthetic HC and PD (no real value, label or identifier): 9 batches with additive and multiplicative
    effects, a curved age effect, sex, eTIV, a small PD effect and a PD share that differs by batch."""
    rng = np.random.default_rng(seed)
    batches = [f"{v}|PPMI{p}|{f}T" for v, p, f in [("GE", 1, 1.5), ("GE", 1, 3.0), ("GE", 2, 3.0), ("PHILIPS", 1, 1.5), ("PHILIPS", 1, 3.0),
                                                  ("PHILIPS", 2, 3.0), ("SIEMENS", 1, 1.5), ("SIEMENS", 1, 3.0), ("SIEMENS", 2, 3.0)]]
    p_hc = np.array([27, 14, 17, 11, 12, 11, 13, 86, 24], float)
    p_pd = np.array([60, 50, 73, 30, 28, 77, 60, 250, 280], float)
    b = np.r_[rng.choice(9, n_hc, p=p_hc / p_hc.sum()), rng.choice(9, n_pd, p=p_pd / p_pd.sum())]
    dx = np.r_[["HC"] * n_hc, ["PD"] * n_pd]
    age = np.clip(rng.normal(np.where(np.arange(9)[b] % 3 == 2, 64, 61), 9), 30, 85)
    sex = rng.integers(0, 2, len(b)).astype(float)
    etiv = rng.normal(1.5e6, 1.4e5, len(b)) + 1e5 * sex
    shift, scale = rng.normal(0, 0.6, (9, len(FEATURES))), rng.uniform(0.7, 1.4, (9, len(FEATURES)))
    age_eff = -0.04 * (age - 60) - 0.0015 * np.clip(age - 60, 0, None) ** 2               # SD units, curved
    vent = np.array(["Ventricle" in f or "Inf_Lat" in f for f in FEATURES])
    z = (np.where(vent, -1, 1) * age_eff[:, None] + 0.3 * sex[:, None] + (etiv[:, None] - 1.5e6) / 2e5
         - 0.15 * (dx == "PD")[:, None] + rng.normal(size=(len(b), len(FEATURES))) * scale[b] + shift[b])
    d = pd.DataFrame(1000 * (5 + z), columns=FEATURES)
    d = d.assign(PATNO=np.arange(len(b)), dx=dx, batch=np.array(batches)[b], age=age, sex=sex, eTIV=etiv)
    return d, {"synthetic": True}


def _resid(train, test, cols):
    """Volumes minus their age (+ age^2), sex and eTIV fit on the training rows, so the classifier sees what
    harmonisation should have equalised, not covariate imbalance between batches (which ComBat keeps on purpose)."""
    X = lambda t: np.column_stack([np.ones(len(t)), t["age"], (t["age"] - 60) ** 2, t["sex"], t["eTIV"] / 1e6])
    beta = np.linalg.lstsq(X(train), train[cols].to_numpy(float), rcond=None)[0]
    return train[cols].to_numpy(float) - X(train) @ beta, test[cols].to_numpy(float) - X(test) @ beta


def batch_auroc(hc, mode, seed, clf="logistic"):
    """Out-of-fold macro one-vs-rest AUROC of predicting batch. ``mode``: harmonised (ComBat fitted in the training
    fold, applied to the held-out fold), raw, or covariates (age, sex, eTIV only: the demographic floor)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = hc["batch"].to_numpy()
    classes = np.unique(y)
    proba = np.zeros((len(hc), len(classes)))
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(hc, y):
        a, b = hc.iloc[tr], hc.iloc[te]
        if mode == "harmonised":
            a, model = combat(a, FEATURES, "batch", COVARS)
            b = combat_apply(model, b)
        if mode == "covariates":
            Xa, Xb = a[COVARS].to_numpy(float), b[COVARS].to_numpy(float)
        else:
            Xa, Xb = _resid(a, b, FEATURES)
        m = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)) if clf == "logistic"
             else RandomForestClassifier(500, random_state=seed, n_jobs=1))
        proba[te] = m.fit(Xa, a["batch"]).predict_proba(Xb)
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))


def _ols(y, X):
    """Coefficients and residual SD."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, np.sqrt((r ** 2).sum(0) / (len(y) - X.shape[1]))


def _design(t, batch_fe=False, group=False):
    cols = [np.ones(len(t)), (t["age"] - 60) / 10, t["sex"], t["eTIV"] / 1e6]
    if group:
        cols.insert(1, (t["dx"] == "PD").astype(float))
    if batch_fe:
        cols += list(pd.get_dummies(t["batch"], drop_first=True, dtype=float).to_numpy().T)
    return np.column_stack(cols)


def analyse(d, flow, repeats=REPEATS):
    hc_all = d[d["dx"].eq("HC")]
    n = hc_all["batch"].value_counts()
    keep = n[n >= MIN_HC].index
    hc, pd_ = hc_all[hc_all["batch"].isin(keep)].reset_index(drop=True), d[d["dx"].eq("PD") & d["batch"].isin(keep)]
    res = {"flow": {**flow, "hc_per_batch": n.to_dict(), "batches_used": sorted(keep),
                    "hc_excluded_small_batch": int((~hc_all["batch"].isin(keep)).sum()), "hc_analysed": len(hc),
                    "pd_analysed": len(pd_), "pd_excluded_no_hc_batch": int((d["dx"].eq("PD") & ~d["batch"].isin(keep)).sum())}}

    # Gate 1: batch is no longer predictable
    runs = {f"{m}_{c}": [batch_auroc(hc, m, SEED + r, c) for r in range(repeats)]
            for m, c in (("harmonised", "logistic"), ("harmonised", "forest"), ("raw", "logistic"), ("raw", "forest"), ("covariates", "logistic"))}
    res["batch_auroc"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in runs.items()}

    # Gate 2: the controls' age effect survives (ComBat on every analysed control, nothing else protected)
    hc_h, _ = combat(hc, FEATURES, "batch", COVARS)
    age = {}
    for s in AGE_SET:
        cols = [f"vol_Left_{s}", f"vol_Right_{s}"]
        (bu, _), (bh, _), (bn, _) = (_ols(t[cols].sum(axis=1).to_numpy(float), _design(hc, fe)) for t, fe in ((hc, True), (hc_h, False), (hc, False)))
        age[s] = {"beta_per_decade_within_batch_raw": float(bu[1]), "beta_per_decade_harmonised": float(bh[1]),
                  "beta_per_decade_pooled_raw": float(bn[1]), "relative_difference": float(abs(bh[1] - bu[1]) / abs(bu[1]))}
    res["age_effect"] = age

    # Gate 3: protecting diagnosis does not inflate PD - HC differences
    both = pd.concat([hc, pd_], ignore_index=True).assign(pd=lambda t: t["dx"].eq("PD").astype(int))
    prot, model = combat(both, FEATURES, "batch", COVARS, label="pd")
    unprot, _ = combat(both, FEATURES, "batch", COVARS)
    dd = {}
    for name, t, fe in (("within_batch_raw", both, True), ("protected", prot, False), ("unprotected", unprot, False)):
        beta, sd = _ols(t[FEATURES].to_numpy(float), _design(t, fe, group=True))
        dd[name] = beta[1] / sd
    gap = np.abs(dd["protected"] - dd["within_batch_raw"])
    res["pd_hc_d"] = {"max_abs_protected_minus_within_batch": float(gap.max()), "feature_at_max": FEATURES[int(gap.argmax())],
                      "median_abs_d": {k: float(np.median(np.abs(v))) for k, v in dd.items()},
                      "per_feature": {f: {k: float(v[i]) for k, v in dd.items()} for i, f in enumerate(FEATURES)}}
    c = model["confounding"]
    res["confounding_report"] = {"cramers_v": float(c["cramers_v"]), "flagged": c["flagged"],
                                 "table": c["table"].to_dict("index"), "sites": c["sites"].round(3).to_dict("index")}

    res["gates"] = {"batch_auroc": max(res["batch_auroc"][f"harmonised_{c}"]["mean"] for c in ("logistic", "forest")) <= GATE_AUROC,
                    "age_effect": all(v["relative_difference"] <= GATE_AGE_REL for v in age.values()),
                    "no_inflation": res["pd_hc_d"]["max_abs_protected_minus_within_batch"] <= GATE_D}
    res["pass"] = bool(all(res["gates"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--derived", default="Imaging/derived")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    d, flow = synthetic() if a.synthetic else load(a.derived, a.ppmi)
    res = analyse(d, flow, a.repeats)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_result.json" if a.synthetic else "harmonisation_result.json")).write_text(json.dumps(res, indent=1, default=str))
    b, g = res["batch_auroc"], res["gates"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}batch AUROC logistic/forest raw {b['raw_logistic']['mean']:.3f}/"
          f"{b['raw_forest']['mean']:.3f} -> harmonised {b['harmonised_logistic']['mean']:.3f}/{b['harmonised_forest']['mean']:.3f} "
          f"(covariates only {b['covariates_logistic']['mean']:.3f}); age rel. diff "
          f"{max(v['relative_difference'] for v in res['age_effect'].values()):.3f}; max |d gap| "
          f"{res['pd_hc_d']['max_abs_protected_minus_within_batch']:.3f}; gates {g} -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
