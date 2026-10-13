"""Pre-registered analysis (documentation/preregistration_dwi_fw.md): multi-shell posterior-SN free water in PPMI 2.0
prodromal participants, DaT-deficient (NSD-ISS) vs DaT-normal, as NM Stage 2b.

One scalar, ``dwisf_fw_sn_posterior_mean`` (``dwi --eddy --study``). Primary: AUROC P(deficient > normal), pairs compared
within vendor x correction tier (``nm_staged.auc_with_ci``); pass if the 95 % CI lower bound > 0.50.

    PYTHONPATH=. python scripts/analyse_dwi_fw.py --tables <dwi_features.csv>[,...] --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_dwi_fw.py --synthetic --out <dir>      # dry run: every value and label is fake
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nm_staged import B, _adjust, auc_with_ci  # noqa: E402  (the Stage 2b statistic, unchanged)

TIER = {"topup_eddy": "topup_eddy", "eddy_no_reverse_PE": "eddy_only", "topup_rigid": "rigid", "rigid": "rigid"}
VALUE, ANTERIOR, FAT = "dwisf_fw_sn_posterior_mean", "dwisf_fw_sn_anterior_mean", "dwisf_fat_sn_posterior_mean"


def load(tables, ppmi):
    """PIE rows with the endpoint under its own QC, PPMI 2.0 (2020 on), earliest per participant; cohort, age, sex, DaT."""
    from pie.imaging.labels import _age_at, covariates, dat_labels

    d = pd.concat([pd.read_csv(t, low_memory=False) for t in tables], ignore_index=True)
    d = d.assign(date=pd.to_datetime(d["acquisition_date"], errors="coerce")).sort_values("date").drop_duplicates("patno")
    flow = {"rows": len(d)}
    d = d[d["date"].dt.year >= 2020]
    d["_ok"] = d["error"].fillna("").astype(str).eq("") & d["dwisf_qc_pass"].astype(str).eq("True")
    cov = covariates(Path(ppmi))
    d = d.merge(cov[["PATNO", "SEX", "COHORT"]], left_on="patno", right_on="PATNO", how="left")
    d["group"] = d["COHORT"].map({"Prodromal": "prodromal", "Parkinson's Disease": "PD"})
    flow["ppmi2_candidates"] = d["group"].value_counts().to_dict()
    flow["qc_or_error_excluded"] = d[~d["_ok"]]["group"].value_counts().to_dict()
    d = d[d["_ok"] & d["group"].notna()].copy()
    sess = d[["patno", "fs_image_id", "acquisition_date"]].rename(columns={"fs_image_id": "image_id", "acquisition_date": "session_date"})
    dat = dat_labels(Path(ppmi), sess)
    d = d.merge(dat[["PATNO", "dat_d_nsdiss", "dat_deficit_sbr", "sbr_pct_expected", "sbr_putamen_min", "sbr_source"]]
                .drop_duplicates("PATNO"), on="PATNO", how="left")
    d["age"] = _age_at(cov, d["patno"], d["acquisition_date"])
    d["sex"] = pd.to_numeric(d["SEX"], errors="coerce")
    d["stratum"] = d["manufacturer"].astype(str) + "|" + d["dwisf_tier"].map(TIER).astype(str)
    flow["no_dat_label"] = d[d["dat_d_nsdiss"].isna()]["group"].value_counts().to_dict()
    return d.reset_index(drop=True), flow


def synthetic(n_pro=240, n_pd=80, auc_true=0.60):
    """Fully synthetic participants (no real value, label or identifier): for dry runs and the test."""
    rng = np.random.default_rng(1)
    n = n_pro + n_pd
    group = np.r_[["prodromal"] * n_pro, ["PD"] * n_pd]
    defi = np.where(group == "PD", np.nan, (rng.random(n) < 0.47).astype(float))
    stratum = rng.choice(["Siemens|topup_eddy", "Siemens|eddy_only", "GE|eddy_only", "Philips|eddy_only"], n, p=[0.7, 0.1, 0.1, 0.1])
    age, sex = rng.normal(66, 7, n), rng.integers(0, 2, n).astype(float)
    shift = np.sqrt(2) * norm.ppf(auc_true)                  # binormal AUROC -> mean shift
    z = rng.normal(size=n) + shift * ((defi == 1) | (group == "PD"))
    fw = 0.18 + 0.002 * (age - 66) + 0.03 * z
    sbr = 1.4 - 0.4 * ((defi == 1) | (group == "PD")) + 0.2 * rng.normal(size=n)
    d = pd.DataFrame({"patno": np.arange(n), "group": group, "dat_d_nsdiss": defi, "dat_deficit_sbr": defi, "stratum": stratum,
                      "age": age, "sex": sex, VALUE: fw, ANTERIOR: fw - 0.02 + 0.03 * rng.normal(size=n),
                      FAT: 0.6 + 0.02 * rng.normal(size=n), "sbr_putamen_min": sbr, "sbr_pct_expected": 100 * sbr / 1.4,
                      "sbr_source": "synthetic"})
    return d, {"synthetic": True}


def _auc(pos, neg, col, n_boot):
    return auc_with_ci(pos[col], neg[col], pos["stratum"], neg["stratum"], n_boot)


def analyse(d, flow, n_boot=B):
    pro = d[d["group"].eq("prodromal")].dropna(subset=[VALUE, "dat_d_nsdiss"])
    normal, defi = pro[pro["dat_d_nsdiss"].eq(0)], pro[pro["dat_d_nsdiss"].eq(1)]
    res = {"flow": {**flow, "analysed": {"dat_normal": len(normal), "dat_deficient": len(defi)},
                    "strata": pro.groupby(["stratum", "dat_d_nsdiss"]).size().unstack(fill_value=0).to_dict("index")}}
    res["primary"] = _auc(defi, normal, VALUE, n_boot)
    res["pass"] = bool(res["primary"]["ci95"][0] > 0.50)
    adj = _adjust(pro.assign(y=pro[VALUE]), lambda t: np.ones(len(t), bool))       # fitted on all prodromal: no DaT label enters
    x2 = pro.dropna(subset=["dat_deficit_sbr"])
    s = pro.dropna(subset=["sbr_pct_expected"])
    pd_ = d[d["group"].eq("PD")].dropna(subset=[VALUE])
    res["secondary"] = {
        "age_sex_adjusted": auc_with_ci(adj.loc[adj["dat_d_nsdiss"].eq(1), "y"], adj.loc[adj["dat_d_nsdiss"].eq(0), "y"],
                                        adj.loc[adj["dat_d_nsdiss"].eq(1), "stratum"], adj.loc[adj["dat_d_nsdiss"].eq(0), "stratum"], n_boot),
        "dat_deficit_sbr_0.65": _auc(x2[x2["dat_deficit_sbr"].eq(1)], x2[x2["dat_deficit_sbr"].eq(0)], VALUE, n_boot),
        "spearman_vs_sbr_pct_expected": {"rho": float(spearmanr(s[VALUE], s["sbr_pct_expected"])[0]), "n": int(len(s))},
        "age_rho_dat_normal": {"rho": float(spearmanr(normal["age"], normal[VALUE], nan_policy="omit")[0]), "n": int(len(normal))},
        "anterior_sn_fw": _auc(defi.dropna(subset=[ANTERIOR]), normal.dropna(subset=[ANTERIOR]), ANTERIOR, n_boot),
        "posterior_sn_fat": _auc(defi.dropna(subset=[FAT]), normal.dropna(subset=[FAT]), FAT, n_boot),
        "pd_vs_dat_normal_prodromal": _auc(pd_, normal, VALUE, n_boot),
        "pd_rho_putamen_min_sbr": {str(k): {"rho": float(spearmanr(g[VALUE], g["sbr_putamen_min"], nan_policy="omit")[0]), "n": int(len(g))}
                                   for k, g in pd_.groupby("sbr_source") if len(g) >= 10}}
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
    res = analyse(d, flow, a.n_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_result.json" if a.synthetic else "fw_result.json")).write_text(json.dumps(res, indent=1, default=float))
    p = res["primary"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}AUROC {p['auroc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] "
          f"(deficient {p['n_pos']}, normal {p['n_neg']}) -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
