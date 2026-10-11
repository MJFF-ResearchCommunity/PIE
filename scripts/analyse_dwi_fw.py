"""Pre-registered analysis (documentation/preregistration_dwi_fw.md): multi-shell posterior-SN free water, PPMI 2.0 PD vs HC.

One scalar, ``dwisf_fw_sn_posterior_mean`` (``dwi --eddy --study``), HC-referenced within vendor x correction tier, reported
as a standardised difference with its CI against the published PPMI effect (Burciu et al. 2017, d 0.4-0.5); no pass/fail gate.

    PYTHONPATH=. python scripts/analyse_dwi_fw.py --tables <dwi_features.csv>[,...] --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_dwi_fw.py --synthetic --out <dir>      # dry run: every value and label is fake
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEED, B = 20261011, 2000
ANCHOR = (0.40, 0.50)              # Burciu 2017 PPMI-1 posterior-SN free water, PD vs HC
TIER = {"topup_eddy": "topup_eddy", "eddy_no_reverse_PE": "eddy_only", "topup_rigid": "rigid", "rigid": "rigid"}
VALUE, SECONDARY = "dwisf_fw_sn_posterior_mean", ("dwisf_fw_sn_anterior_mean", "dwisf_fat_sn_posterior_mean")


def load(tables, ppmi):
    """PIE rows with the endpoint under its own QC, PPMI 2.0 (acquired 2020 or later), earliest per participant; cohort, age, sex."""
    from pie.imaging.labels import covariates

    d = pd.concat([pd.read_csv(t, low_memory=False) for t in tables], ignore_index=True)
    d = d.assign(date=pd.to_datetime(d["acquisition_date"], errors="coerce")).sort_values("date").drop_duplicates("patno")
    flow = {"rows": len(d)}
    d = d[d["date"].dt.year >= 2020]
    d["_ok"] = d["error"].fillna("").astype(str).eq("") & d["dwisf_qc_pass"].astype(str).eq("True")
    cov = covariates(Path(ppmi))[["PATNO", "BIRTHDT", "SEX", "COHORT"]].rename(columns={"PATNO": "patno"})
    d = d.merge(cov, on="patno", how="left")
    d["group"] = d["COHORT"].map({"Parkinson's Disease": "PD", "Healthy Control": "HC"})
    flow["ppmi2_candidates"] = d["group"].value_counts().to_dict()
    flow["qc_or_error_excluded"] = d[~d["_ok"]]["group"].value_counts().to_dict()
    d = d[d["_ok"] & d["group"].notna()]
    d = d.assign(age=(d["date"] - pd.to_datetime(d["BIRTHDT"])).dt.days / 365.25, sex=pd.to_numeric(d["SEX"], errors="coerce"),
                 vendor=d["manufacturer"].astype(str), tier=d["dwisf_tier"].map(TIER), serial=d.get("DeviceSerialNumber", ""))
    d.loc[~d["age"].between(18, 110), "age"] = np.nan
    flow["no_age_or_sex"] = d[d["age"].isna() | d["sex"].isna()]["group"].value_counts().to_dict()
    return d[d["age"].notna() & d["sex"].notna()].reset_index(drop=True), flow


def synthetic(n_pd=220, n_hc=80, d_true=0.45):
    """Fully synthetic participants (no real value, label or identifier): for dry runs and the test."""
    rng = np.random.default_rng(1)
    n = n_pd + n_hc
    g = np.r_[["PD"] * n_pd, ["HC"] * n_hc]
    vendor = rng.choice(["Siemens", "GE", "Philips"], n, p=[0.5, 0.3, 0.2])
    tier = rng.choice(["topup_eddy", "eddy_only"], n, p=[0.7, 0.3])
    age, sex = rng.normal(64, 9, n), rng.integers(0, 2, n).astype(float)
    shift = {"Siemens": 0.0, "GE": 0.02, "Philips": -0.015}
    fw = 0.18 + 0.002 * (age - 64) + np.array([shift[v] for v in vendor]) + 0.03 * (rng.normal(size=n) + d_true * (g == "PD"))
    d = pd.DataFrame({"patno": np.arange(n), "group": g, "vendor": vendor, "tier": tier, "age": age, "sex": sex, VALUE: fw,
                      "serial": vendor, SECONDARY[0]: fw - 0.02 + 0.01 * rng.normal(size=n), SECONDARY[1]: 0.6 + 0.02 * rng.normal(size=n),
                      "date": pd.Timestamp("2022-01-01")})
    return d, {"synthetic": True}


def batches(hc):
    """vendor|tier; a vendor any of whose tiers has < 5 HC is one batch; a vendor with < 5 HC has none (as the 28 Sep gate)."""
    n = hc.groupby(["vendor", "tier"]).size()
    split = {v: bool((n.loc[v] >= 5).all()) for v in n.index.get_level_values(0).unique() if hc["vendor"].eq(v).sum() >= 5}
    return lambda v, t: None if v not in split else (f"{v}|{t}" if n.get((v, t), 0) >= 5 else None) if split[v] else f"{v}|any"


def harmonise(hc, df, col):
    """Fit col ~ age + sex + batch in HC only; z = residual / the batch's HC residual SD."""
    name = batches(hc)
    hb = hc.assign(batch=[name(v, t) for v, t in zip(hc["vendor"], hc["tier"])]).dropna(subset=["batch", col])
    levels = sorted(hb["batch"].unique())
    X = lambda f: np.column_stack([f["age"], f["sex"]] + [f["batch"].eq(b).astype(float) for b in levels])
    beta = np.linalg.lstsq(X(hb), hb[col].to_numpy(), rcond=None)[0]
    sd = (hb[col] - X(hb) @ beta).groupby(hb["batch"]).std(ddof=1)
    out = df.assign(batch=[name(v, t) for v, t in zip(df["vendor"], df["tier"])]).dropna(subset=[col])
    out = out[out["batch"].isin(levels)].copy()
    out["z"] = (out[col] - X(out) @ beta) / out["batch"].map(sd)
    return out


def effect(h):
    pd_, hc = h.loc[h["group"] == "PD", "z"], h.loc[h["group"] == "HC", "z"]
    pooled = np.sqrt(((len(pd_) - 1) * pd_.var() + (len(hc) - 1) * hc.var()) / (len(pd_) + len(hc) - 2))
    return float((pd_.mean() - hc.mean()) / pooled), float(roc_auc_score(h["group"].eq("PD"), h["z"]))


def bootstrap(d, col, n_boot=B):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n_boot):            # resample within group and refit the HC reference every draw
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(2**31))) for _, g in d.groupby("group")], ignore_index=True)
        h = harmonise(s[s["group"] == "HC"], s, col)
        if h["group"].nunique() == 2 and np.isfinite(h["z"]).all():
            out.append(effect(h))
    return np.array(out)


def verdict(lo, hi):
    """The pre-registered reading of the 95 % CI of d (no single gate)."""
    if lo > 0 and hi >= ANCHOR[0] and lo <= ANCHOR[1]:
        return "consistent with the published PPMI effect"
    if lo > 0 and hi < ANCHOR[0]:
        return "present but smaller than published"
    if lo > ANCHOR[1]:
        return "larger than published"
    if hi < ANCHOR[0]:
        return "not detected; excludes the published effect"
    return "inconclusive"


def analyse(d, flow, n_boot=B, ppmi=None):
    hc = d[d["group"] == "HC"]
    h = harmonise(hc, d, VALUE)
    flow["no_hc_batch"] = d[~d.index.isin(h.index)]["group"].value_counts().to_dict()
    flow["analysed"] = h["group"].value_counts().to_dict()
    dd, a = effect(h)
    boots = bootstrap(d, VALUE, n_boot)
    (dlo, dhi), (alo, ahi) = np.percentile(boots[:, 0], [2.5, 97.5]), np.percentile(boots[:, 1], [2.5, 97.5])
    res = {"flow": flow, "d": dd, "d_ci95": [dlo, dhi], "auroc": a, "auroc_ci95": [alo, ahi], "n_boot": len(boots),
           "anchor_d": ANCHOR, "verdict": verdict(dlo, dhi),
           "glass_delta": float(h.loc[h["group"] == "PD", "z"].mean() - h.loc[h["group"] == "HC", "z"].mean()),
           "unadjusted_auroc": float(roc_auc_score(d["group"].eq("PD"), d[VALUE])),
           "auroc_by_vendor": {v: float(roc_auc_score(g["group"].eq("PD"), g["z"])) for v, g in h.groupby("vendor") if g["group"].nunique() == 2},
           "hc_age_rho": float(spearmanr(hc["age"], hc[VALUE])[0]),       # positive control: free water rises with age
           "batches": h.groupby(["batch", "group"]).size().unstack(fill_value=0).to_dict("index")}
    for col in SECONDARY:              # specificity (anterior SN) and the tissue compartment
        hs = harmonise(hc, d, col)
        res[f"d_{col}"] = effect(hs)[0] if hs["group"].nunique() == 2 else np.nan
    within = d.groupby("serial").filter(lambda g: (g["group"] == "HC").sum() >= 5 and (g["group"] == "PD").sum() >= 5)
    if len(within):                    # sensitivity: single-scanner batches
        hw = harmonise(within[within["group"] == "HC"], within.assign(vendor=within["serial"].astype(str)), VALUE)
        res["sensitivity_within_scanner_d"], res["sensitivity_within_scanner_n"] = effect(hw)[0], len(hw)
    if ppmi is not None:               # descriptive: PPMI core-lab lowest putamen SBR in PD, expected negative
        from pie.imaging.labels import dat_labels
        pdz = h[h["group"] == "PD"]
        sess = pd.DataFrame({"patno": pdz["patno"], "image_id": pdz["patno"].astype(str), "session_date": pdz["date"]})
        j = pdz.merge(dat_labels(ppmi, sess)[["PATNO", "sbr_putamen_min", "sbr_source"]].rename(columns={"PATNO": "patno"}), on="patno")
        res["dat_rho_pd"] = {s: float(spearmanr(g["z"], g["sbr_putamen_min"])[0]) for s, g in j.groupby("sbr_source") if len(g) >= 10}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", help="comma-separated dwi_features.csv paths (PIE, dwi --eddy --study)")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    d, flow = synthetic() if a.synthetic else load(a.tables.split(","), a.ppmi)
    res = analyse(d, flow, a.n_boot, None if a.synthetic else a.ppmi)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_result.json" if a.synthetic else "fw_result.json")).write_text(json.dumps(res, indent=1, default=float))
    print(f"d {res['d']:.3f} [{res['d_ci95'][0]:.3f}, {res['d_ci95'][1]:.3f}]; AUROC {res['auroc']:.3f}; {res['verdict']}")
    return res


if __name__ == "__main__":
    main()
