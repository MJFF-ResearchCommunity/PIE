"""Pre-registered analysis (documentation/preregistration_dwi_fw_saa.md): posterior-SN free water and concurrent CSF SAA
status in the paper's prodromal clinical cohort. Runs only after the frozen DaT test (preregistration_dwi_fw.md) passed.

Same scalar, sessions and QC as that test (``analyse_dwi_fw.load``). Primary (biology): age- and sex-adjusted stratified
AUROC P(SAA+ > SAA-), associated if the 95 % CI lower bound > 0.50. Increment (prediction): the paper's nested engine,
unchanged, clinical model + free water vs the clinical model, under the paper's decision rule.

    PYTHONPATH=. python scripts/analyse_dwi_fw_saa.py --tables <dwi_features.csv>[,...] --ppmi PPMI \\
        --study <study1_virtual_biomarkers> --dat-result <fw_result.json> --out <dir>
    PYTHONPATH=. python scripts/analyse_dwi_fw_saa.py --synthetic --study <study1_virtual_biomarkers> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_dwi_fw import ANTERIOR, TIER, VALUE, load  # noqa: E402  (the frozen DaT test's loader, unchanged)
from nm_staged import B, _adjust, auc_with_ci  # noqa: E402

WINDOW = 18                                    # months between the DWI session and the CSF month, as the DaT match
FEATURE = "dwi_fw_sn_posterior"                # dwi_ prefix: the engine residualises it without ICV
SCORES = ["upsit", "rbd_symptoms12", "scopa_nonsexual21", "updrs3"]
CLINICAL = ["age_at_scan", "sex_male", "LRRK2_carrier", "GBA_carrier", "SNCA_carrier", "APOE_e4"] + SCORES
BATCHES = ["manufacturer", "model", "tier"]
ROLES = ["development", "calibration", "test"]


def clinical(study):
    """The paper's prodromal clinical cohort: frozen triage features and baseline CSF SAA labels, all three roles."""
    s = Path(study) / "saa_triage_20260909"
    labels = pd.concat([pd.read_csv(s / f"{r}_labels.csv") for r in ROLES], ignore_index=True)
    return pd.read_parquet(s / "features.parquet").merge(labels, on="PATNO", validate="one_to_one")


def join(d, cl, window=WINDOW):
    pro = d[d["group"].eq("prodromal")]
    m = pro.drop(columns="PATNO").merge(cl, left_on="patno", right_on="PATNO", validate="one_to_one")
    a0, a1 = pd.to_datetime(m["anchor_start"]), pd.to_datetime(m["anchor_end"])
    m["csf_gap_months"] = np.maximum((a0 - m["date"]).dt.days, (m["date"] - a1).dt.days).clip(lower=0) / 30.44
    keep = m["csf_gap_months"] <= window
    return m[keep].reset_index(drop=True), {"prodromal_qc_pass": len(pro), "in_clinical_cohort": len(m), f"within_{window}_months": int(keep.sum())}


def synthetic(n=300, saa_shift=0.3):
    """Fully synthetic participants (no real value, label or identifier). SAA+ are older, as in PPMI, so an unadjusted
    difference appears from age alone; ``saa_shift`` (SD) is the free-water difference left after age."""
    rng = np.random.default_rng(2)
    saa = (rng.random(n) < 0.68).astype(int)
    dat = np.where(rng.random(n) < 0.25 + 0.3 * saa, 1.0, 0.0)
    age = rng.normal(64, 6, n) + 4 * saa
    sex = rng.integers(0, 2, n).astype(float)
    fw = 0.18 + 0.004 * (age - 64) + 0.02 * (rng.normal(size=n) + saa_shift * saa)
    m = pd.DataFrame({"patno": np.arange(n), "PATNO": np.arange(n), "group": "prodromal", "saa_prodromal": saa, "dat_d_nsdiss": dat,
                      "age": age, "sex": sex, VALUE: fw, ANTERIOR: fw - 0.02 + 0.02 * rng.normal(size=n),
                      "stratum": rng.choice(["Siemens|topup_eddy", "GE|eddy_only", "Philips|eddy_only"], n, p=[0.7, 0.15, 0.15]),
                      "csf_gap_months": np.where(rng.random(n) < 0.8, 0.0, rng.uniform(0, 18, n)),
                      "role": rng.choice(ROLES, n, p=[0.2, 0.2, 0.6]),
                      "age_at_scan": age, "sex_male": sex, "upsit": rng.normal(25 - 6 * saa, 6), "updrs3": rng.normal(4, 3, n)})
    for c in ["LRRK2_carrier", "GBA_carrier", "SNCA_carrier", "APOE_e4"]:
        m[c] = (rng.random(n) < 0.1).astype(float)
    m["rbd_symptoms12"], m["scopa_nonsexual21"] = rng.normal(5, 3, n), rng.normal(8, 5, n)
    m["manufacturer"] = m["stratum"].str.split("|").str[0]
    m["model"], m["dwisf_tier"] = "synthetic", m["stratum"].str.split("|").str[1].map({"topup_eddy": "topup_eddy", "eddy_only": "eddy_no_reverse_PE"})
    return m, m.copy(), {"synthetic": True}


def _auc(x, col, n_boot):
    pos, neg = x[x["saa_prodromal"].eq(1)], x[x["saa_prodromal"].eq(0)]
    return auc_with_ci(pos[col], neg[col], pos["stratum"], neg["stratum"], n_boot)


def increment(frame, workers=8):
    """The paper's nested engine through its own fold driver (mri_clinical_comparator_20260918/run.py), unchanged:
    clinical comparator vs comparator + free water, same seeds and folds; unadjusted and scanner-adjusted."""
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.model_selection import StratifiedKFold
    import run as mcc                                            # on sys.path via --study
    y = frame["saa_prodromal"].to_numpy(int)
    out = {}
    for adjusted in (False, True):
        jobs = [(frame, {"fw": [FEATURE]}, BATCHES, None, adjusted, "clinical", r, k, tr, te, seed + k)
                for r, seed in enumerate(mcc.engine.SEEDS)
                for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, y))]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            oof = pd.concat([r[0] for r in pool.map(mcc.fold, jobs)], ignore_index=True)
        assert len(oof) == 3 * len(frame) and not oof.duplicated(["PATNO", "repeat"]).any()
        m = mcc.engine.paired_metrics(oof).set_index("model").to_dict("index")
        out["scanner_adjusted" if adjusted else "primary"] = {k: m[k] for k in ("selected", "fw")}
    return out


def demonstrated(inc):
    """The paper's rule: 98.75 % lower bound > 0, positive in all three repeats, no reversal under scanner adjustment."""
    p, s = inc["primary"]["selected"], inc["scanner_adjusted"]["selected"]
    return bool(p["delta_ci98_75_low"] > 0 and p["positive_in_all_repeats"] and s["delta_auc"] > 0)


def analyse(m, pro, flow, n_boot=B, increment=increment):
    adj = _adjust(pro.assign(y=pro[VALUE]), lambda t: np.ones(len(t), bool)).set_index("patno")["y"]     # no SAA label enters
    adj_a = _adjust(pro.assign(y=pro[ANTERIOR]), lambda t: np.ones(len(t), bool)).set_index("patno")["y"]
    x = m.assign(fw_adj=m["patno"].map(adj), fw_anterior_adj=m["patno"].map(adj_a)).dropna(subset=[VALUE, "fw_adj", "saa_prodromal"])
    pos, neg = x[x["saa_prodromal"].eq(1)], x[x["saa_prodromal"].eq(0)]
    res = {"flow": {**flow, "analysed": {"saa_positive": len(pos), "saa_negative": len(neg)},
                    "analysed_by_role": x.groupby(["role", "saa_prodromal"]).size().unstack(fill_value=0).to_dict("index"),
                    "strata": x.groupby(["stratum", "saa_prodromal"]).size().unstack(fill_value=0).to_dict("index")}}
    res["primary"] = _auc(x, "fw_adj", n_boot)
    res["associated"] = bool(res["primary"]["ci95"][0] > 0.50)
    z = (x[VALUE] - x[VALUE].mean()) / x[VALUE].std()
    fit = sm.Logit(x["saa_prodromal"].to_numpy(float), sm.add_constant(np.column_stack([z, x["age"], x["sex"]]))).fit(disp=0)
    lo, hi = fit.conf_int()[1]
    dn = x[x["dat_d_nsdiss"].eq(0)]
    res["secondary"] = {
        "unadjusted": _auc(x, VALUE, n_boot),
        "mwu_one_sided_p": float(mannwhitneyu(pos[VALUE], neg[VALUE], alternative="greater").pvalue),
        "odds_ratio_per_sd": {"or": float(np.exp(fit.params[1])), "ci95": [float(np.exp(lo)), float(np.exp(hi))], "p": float(fit.pvalues[1])},
        "dat_normal_only": _auc(dn, "fw_adj", n_boot) if dn["saa_prodromal"].nunique() == 2 else None,
        "saa_by_dat": {f"saa{int(s)}_dat{int(dd)}": {"n": len(g), "median_fw_adj": float(g["fw_adj"].median())}
                       for (s, dd), g in x.dropna(subset=["dat_d_nsdiss"]).groupby(["saa_prodromal", "dat_d_nsdiss"])},
        "within_6_months": _auc(x[x["csf_gap_months"] <= 6], "fw_adj", n_boot),
        "anterior_sn_fw": _auc(x.dropna(subset=["fw_anterior_adj"]), "fw_anterior_adj", n_boot),
        # Amendment 30 Oct (before the run): test-role participants enter a structural/diffusion comparison for the first time
        "by_role": {"test": _auc(x[x["role"].eq("test")], "fw_adj", n_boot),
                    "development_calibration": _auc(x[x["role"].isin(["development", "calibration"])], "fw_adj", n_boot)}}
    f = x.assign(**{FEATURE: x[VALUE], "tier": x["dwisf_tier"].map(TIER)}).sort_values("PATNO").reset_index(drop=True)
    counts = f["saa_prodromal"].value_counts()
    if len(f) < 80 or counts.min() < 20:                         # the paper's minimum (run.py dataset)
        res["increment"] = {"ran": False, "reason": "below the paper's minimum of 80 participants and 20 per class"}
    else:
        inc = increment(f[["PATNO", "saa_prodromal", *CLINICAL, *BATCHES, FEATURE]])
        res["increment"] = {**inc, "demonstrated": demonstrated(inc),
                            "upper_95_below_0.05": bool(inc["primary"]["selected"]["delta_ci95_high"] < 0.05)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", help="comma-separated dwi_features.csv paths, as for analyse_dwi_fw.py")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--study", required=True, help="the paper's repository (clinical cohort and engine; read only)")
    ap.add_argument("--dat-result", help="fw_result.json of the frozen DaT test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "synthetic_result.json" if a.synthetic else "fw_saa_result.json"
    if not a.synthetic and not json.loads(Path(a.dat_result).read_text())["pass"]:
        res = {"ran": False, "reason": "DWI_FW_VALIDATED not reached: the frozen DaT test did not pass"}
        (out / name).write_text(json.dumps(res, indent=1))
        print(res["reason"])
        return res
    sys.path[:0] = [str(Path(a.study) / "mri_clinical_comparator_20260918"), str(a.study)]
    if a.synthetic:
        m, pro, flow = synthetic()
    else:
        d, flow = load(a.tables.split(","), a.ppmi)
        m, jflow = join(d, clinical(a.study))
        pro, flow = d[d["group"].eq("prodromal")], {**flow, **jflow}
    res = analyse(m, pro, flow, a.n_boot, lambda f: increment(f, a.workers))
    (out / name).write_text(json.dumps(res, indent=1, default=float))
    p, i = res["primary"], res["increment"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}adjusted AUROC {p['auroc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] "
          f"(SAA+ {p['n_pos']}, SAA- {p['n_neg']}) -> {'associated' if res['associated'] else 'not associated'}; increment "
          + (f"{i['primary']['selected']['delta_auc']:+.3f} -> {'demonstrated' if i['demonstrated'] else 'not demonstrated'}" if i.get("ran", True) else i["reason"]))
    return res


if __name__ == "__main__":
    main()
