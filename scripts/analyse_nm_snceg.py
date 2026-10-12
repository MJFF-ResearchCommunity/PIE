"""Pre-registered analysis (documentation/preregistration_nm_snceg.md): snceg neuromelanin contrast on PPMI Siemens
2D GRE-MT, HC vs de novo PD.

Primary: AUROC (HC > de novo PD) of ``nms_sn_mean_cr`` with a DeLong 95 % CI; pass if the CI contains 0.743
(Lillebostad 2025) and its lower bound is > 0.50. De novo PD = PD whose NM session is PPMI's BL visit (as R1).

    PYTHONPATH=. python scripts/analyse_nm_snceg.py --nm-dir <work dir with nm_features.csv, nm_native_features.csv> --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_nm_snceg.py --synthetic --out <dir>      # dry run: every value and label is fake
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED, B = 20261011, 2000
ANCHOR, ANCHOR2 = 0.743, 0.72
PRIMARY = "nms_sn_mean_cr"
SECONDARY = ("nms_sn_volume_mm3", "nms_sn_mean_cr_allslices")


def delong(pos, neg):
    """AUROC P(pos > neg) and its DeLong (1988) standard error."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    a = psi.mean()
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    se = np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))
    return float(a), float(se)


def rule(lo, hi):
    return bool(lo <= ANCHOR <= hi and lo > 0.50)


def load(nm_dir, ppmi):
    """Siemens QC-passing slabs with snceg measures, cohort, age, sex and the NM visit; exclusion counts per step."""
    from pie.imaging.labels import _age_at, _latest, covariates
    from pie.imaging.manifest import QC, _vendor

    nm_dir = Path(nm_dir)
    nat_csv = nm_dir / "nm_native_features.csv"
    flow = {"nm_native_features_sha256": hashlib.sha256(nat_csv.read_bytes()).hexdigest()}   # before any label is joined
    nm = pd.read_csv(nm_dir / "nm_features.csv", low_memory=False)
    nm = nm[nm["error"].fillna("") == ""]
    nat = pd.read_csv(nat_csv, low_memory=False)
    nat = nat[nat["error"].fillna("") == ""].drop(columns="error")
    d = nm.merge(nat, on="patno", suffixes=("", "_nat"))
    flow["slabs_with_measures"] = len(d)
    steps = [("qc_nm", QC["nm"](d)), ("siemens", _vendor(d["manufacturer"]).eq("SIEMENS"))]
    dims = lambda c: d[c].astype(str).str.split("x", expand=True).apply(pd.to_numeric, errors="coerce")
    steps.append(("slab_ge_24mm", (dims("shape") * dims("voxel_mm")).min(axis=1) >= 24 - 1e-6))
    steps.append(("snceg_sn_nonempty", pd.to_numeric(d.get("nms_sn_volume_mm3"), errors="coerce") > 0))
    keep = pd.Series(True, index=d.index)       # masks are on the unfiltered rows: apply them cumulatively
    for name, ok in steps:
        flow[f"excluded_{name}"] = int((keep & ~ok.fillna(False)).sum())
        keep &= ok.fillna(False)
    d = d[keep.to_numpy()]
    cov = covariates(ppmi)
    d = d.merge(cov[["PATNO", "COHORT", "SEX"]], left_on="patno", right_on="PATNO")
    meta = _latest(ppmi, "Imaging", "MRI_Acquisition_Metadata")
    meta = meta[meta["MRI_SEQ_NM_MT"].astype(str).str.strip().eq("Yes")].assign(
        date=lambda m: pd.to_datetime(m["MRI_SCAN_DATE"], format="%m/%Y", errors="coerce"))
    scan = pd.to_datetime(d["acquisition_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    ev = []
    for p, s in zip(d["patno"], scan):
        m = meta[(meta["PATNO"] == p) & meta["date"].notna()]
        ev.append(m.loc[(m["date"] - s).abs().idxmin(), "EVENT_ID"] if len(m) and pd.notna(s) else None)
    d["group"] = np.where(d["COHORT"].eq("Healthy Control"), "HC",
                          np.where(d["COHORT"].eq("Parkinson's Disease"), np.where(pd.Series(ev, index=d.index).eq("BL"), "PD_denovo", "PD_later"), None))
    d["age"] = _age_at(cov, d["patno"], d["acquisition_date"])
    d["sex"] = pd.to_numeric(d["SEX"], errors="coerce")
    d = d[d["group"].notna()]
    flow["groups"] = d["group"].value_counts().to_dict()
    return d, flow


def synthetic(n_hc=70, n_pd=350, auc=0.74):
    """Fully synthetic slabs (no real value, label or identifier) for the dry run and the test."""
    rng = np.random.default_rng(3)
    dshift = np.sqrt(2) * norm.ppf(auc)
    g = np.r_[["HC"] * n_hc, ["PD_denovo"] * n_pd, ["PD_later"] * 40]
    cr = 0.25 + 0.06 * (rng.normal(size=len(g)) - dshift * (g != "HC"))
    return pd.DataFrame({"patno": np.arange(len(g)), "group": g, PRIMARY: cr, "nms_sn_volume_mm3": 300 + 50 * rng.normal(size=len(g)),
                         "nms_sn_mean_cr_allslices": cr + 0.02 * rng.normal(size=len(g)), "age": rng.normal(63, 9, len(g)),
                         "sex": rng.integers(0, 2, len(g)).astype(float)}), {"synthetic": True}


def compare(d, col, pd_groups=("PD_denovo",), adjust=False, n_boot=B):
    x = d[d["group"].isin(("HC", *pd_groups))].dropna(subset=[col] + (["age", "sex"] if adjust else []))
    y = x[col].to_numpy(float)
    if adjust:          # residuals of an age + sex fit on HC only: no PD label enters the adjustment
        hc = x["group"].eq("HC").to_numpy()
        X = np.column_stack([np.ones(len(x)), x["age"], x["sex"]])
        beta = np.linalg.lstsq(X[hc], y[hc], rcond=None)[0]
        y = y - X @ beta
    hc, pdv = y[x["group"].eq("HC").to_numpy()], y[~x["group"].eq("HC").to_numpy()]
    a, se = delong(hc, pdv)
    rng = np.random.default_rng(SEED)
    boot = [delong(rng.choice(hc, len(hc)), rng.choice(pdv, len(pdv)))[0] for _ in range(n_boot)]
    lo, hi = a - 1.96 * se, a + 1.96 * se
    return {"n_hc": int(len(hc)), "n_pd": int(len(pdv)), "auroc": a, "delong_ci95": [lo, hi],
            "bootstrap_ci95": list(np.percentile(boot, [2.5, 97.5])), "ci_contains_0743": bool(lo <= ANCHOR <= hi),
            "ci_contains_072": bool(lo <= ANCHOR2 <= hi)}


def analyse(d, flow, n_boot=B):
    res = {"flow": flow, "primary": compare(d, PRIMARY, n_boot=n_boot)}
    p = res["primary"]
    res["pass"] = rule(*p["delong_ci95"])
    res["secondary"] = {**{c: compare(d, c, n_boot=n_boot) for c in SECONDARY if c in d},
                        f"{PRIMARY}_age_sex_adjusted": compare(d, PRIMARY, adjust=True, n_boot=n_boot),
                        f"{PRIMARY}_all_pd": compare(d, PRIMARY, pd_groups=("PD_denovo", "PD_later"), n_boot=n_boot)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nm-dir")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    d, flow = synthetic() if a.synthetic else load(a.nm_dir, a.ppmi)
    res = analyse(d, flow, a.n_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_result.json" if a.synthetic else "nm_snceg_result.json")).write_text(json.dumps(res, indent=1, default=float))
    if a.synthetic:     # a dry run shows shapes and the decision only
        print(f"synthetic dry run: groups {d['group'].value_counts().to_dict()}; pass={res['pass']}")
    else:
        p = res["primary"]
        print(f"AUROC {p['auroc']:.3f} [{p['delong_ci95'][0]:.3f}, {p['delong_ci95'][1]:.3f}] -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
