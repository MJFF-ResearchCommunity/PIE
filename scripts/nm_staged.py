"""Stages 2b and 3 of the staged NM validation (documentation/preregistration_nm_staged_validation.md).

    prodromal-dat   Stage 2b: within PIE's held prodromal NM sessions, is the primary NM measure lower where PPMI's DaT
                    SBR is already deficient (NSD-ISS D+, <= 75 % of expected)? Vendor-stratified AUROC
                    (DaT-normal > DaT-deficit); pass if its 95 % CI lower bound is > 0.50.
    fresh-clinical  Stage 3: HC vs de novo PD among participants no PIE download held before 12 Oct 2026.
                    Vendor-stratified AUROC (HC > PD); proposed pass: CI lower bound > 0.50.

The primary measure is the one Stage 1 chose (``--stage1`` = nm_split_half_result.json).

    PYTHONPATH=. python scripts/nm_staged.py prodromal-dat --nm-dir <work> --stage1 <json> --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/nm_staged.py fresh-clinical --nm-dir <work> --stage1 <json> --fresh <list> [<list>] --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/nm_staged.py <analysis> --synthetic --out <dir>       # dry run: every value and label fake
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

SEED, B = 20261012, 2000


def stratified_auc(pos, neg, strata_pos, strata_neg):
    """P(pos > neg) over pairs in the same stratum only (ties count 1/2), pooled by number of pairs."""
    pos, neg, sp, sn = map(np.asarray, (pos, neg, strata_pos, strata_neg))
    num = den = 0.0
    for s in np.unique(np.r_[sp, sn]):
        a, b = pos[sp == s], neg[sn == s]
        if len(a) and len(b):
            num += ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).sum()
            den += len(a) * len(b)
    return float(num / den) if den else np.nan


def auc_with_ci(pos, neg, strata_pos, strata_neg, n_boot=B):
    """Stratified AUROC and a bootstrap 95 % CI resampling within each stratum x group."""
    pos, neg, sp, sn = map(np.asarray, (pos, neg, strata_pos, strata_neg))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        ip = np.concatenate([rng.choice(np.flatnonzero(sp == s), (sp == s).sum()) for s in np.unique(sp)])
        ineg = np.concatenate([rng.choice(np.flatnonzero(sn == s), (sn == s).sum()) for s in np.unique(sn)])
        boot.append(stratified_auc(pos[ip], neg[ineg], sp[ip], sn[ineg]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    strata = sorted(set(sp) & set(sn))
    return {"auroc": stratified_auc(pos, neg, sp, sn), "ci95": [float(lo), float(hi)], "n_pos": int(len(pos)),
            "n_neg": int(len(neg)), "strata_used": {str(s): [int((sp == s).sum()), int((sn == s).sum())] for s in strata},
            "pos_outside_shared_strata": int((~np.isin(sp, strata)).sum()), "neg_outside_shared_strata": int((~np.isin(sn, strata)).sum())}


def primary_of(stage1):
    """(column, table suffix) of the Stage 1 primary, e.g. nmf_sn_mean_cr_mode_n4 -> (nmf_sn_mean_cr_mode, _n4)."""
    p = json.loads(Path(stage1).read_text())["primary"]
    return (p[:-3], "_n4") if p.endswith("_n4") else (p, "")


def load(nm_dir, column, suffix):
    """QC-passing full-coverage slabs with the primary measure; exclusion counts; the table's sha256 before any label."""
    from pie.imaging.manifest import QC, _vendor

    nm_dir = Path(nm_dir)
    nat_csv = nm_dir / f"nm_native_features{suffix}.csv"
    flow = {"features_sha256": hashlib.sha256(nat_csv.read_bytes()).hexdigest()}
    nm = pd.read_csv(nm_dir / "nm_features.csv", low_memory=False)
    nm = nm[nm["error"].fillna("") == ""]
    nat = pd.read_csv(nat_csv, low_memory=False)
    nat = nat[nat["error"].fillna("") == ""][["patno", column]]
    d = nm.merge(nat, on="patno")
    flow["slabs_with_measures"] = len(d)
    dims = lambda c: d[c].astype(str).str.split("x", expand=True).apply(pd.to_numeric, errors="coerce")
    steps = [("qc_nm", QC["nm"](d)), ("slab_ge_24mm", (dims("shape") * dims("voxel_mm")).min(axis=1) >= 24 - 1e-6),
             ("measure_missing", pd.to_numeric(d[column], errors="coerce").notna())]
    keep = pd.Series(True, index=d.index)
    for name, ok in steps:
        flow[f"excluded_{name}"] = int((keep & ~ok.fillna(False)).sum())
        keep &= ok.fillna(False)
    d = d[keep.to_numpy()].copy()
    d["vendor"] = _vendor(d["manufacturer"]).to_numpy()
    d["y"] = pd.to_numeric(d[column])
    return d, flow


def _nm_event(d, ppmi):
    """PPMI EVENT_ID of each NM session: the NM-flagged MRI visit nearest the scan month (as the 11 Oct analysis)."""
    from pie.imaging.labels import _latest

    meta = _latest(ppmi, "Imaging", "MRI_Acquisition_Metadata")
    meta = meta[meta["MRI_SEQ_NM_MT"].astype(str).str.strip().eq("Yes")].assign(
        date=lambda m: pd.to_datetime(m["MRI_SCAN_DATE"], format="%m/%Y", errors="coerce"))
    scan = pd.to_datetime(d["acquisition_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    ev = []
    for p, s in zip(d["patno"], scan):
        m = meta[(meta["PATNO"] == p) & meta["date"].notna()]
        ev.append(m.loc[(m["date"] - s).abs().idxmin(), "EVENT_ID"] if len(m) and pd.notna(s) else None)
    return pd.Series(ev, index=d.index)


def label_prodromal_dat(d, ppmi):
    from pie.imaging.labels import _age_at, covariates, dat_labels

    cov = covariates(ppmi)
    d = d.merge(cov[["PATNO", "COHORT", "SEX"]], left_on="patno", right_on="PATNO")
    d = d[d["COHORT"].eq("Prodromal")].copy()
    sess = d[["patno", "fs_image_id", "acquisition_date"]].rename(columns={"fs_image_id": "image_id", "acquisition_date": "session_date"})
    dat = dat_labels(ppmi, sess)
    d = d.merge(dat[["PATNO", "dat_d_nsdiss", "dat_deficit_sbr", "sbr_pct_expected", "sbr_putamen_min"]].drop_duplicates("PATNO"),
                on="PATNO", how="left")
    d["age"] = _age_at(cov, d["patno"], d["acquisition_date"])
    d["sex"] = pd.to_numeric(d["SEX"], errors="coerce")
    return d


def label_fresh(d, ppmi, fresh):
    from pie.imaging.labels import _age_at, covariates, dat_labels

    d = d[d["patno"].isin(fresh)].copy()
    cov = covariates(ppmi)
    d = d.merge(cov[["PATNO", "COHORT", "SEX"]], left_on="patno", right_on="PATNO")
    ev = _nm_event(d, ppmi)
    d["group"] = np.where(d["COHORT"].eq("Healthy Control"), "HC",
                          np.where(d["COHORT"].eq("Parkinson's Disease") & ev.isin(["BL", "SC"]).to_numpy(), "PD_denovo", None))
    d = d[d["group"].notna()].copy()
    sess = d[["patno", "fs_image_id", "acquisition_date"]].rename(columns={"fs_image_id": "image_id", "acquisition_date": "session_date"})
    dat = dat_labels(ppmi, sess)
    d = d.merge(dat[["PATNO", "sbr_putamen_min"]].drop_duplicates("PATNO"), on="PATNO", how="left")
    d["age"] = _age_at(cov, d["patno"], d["acquisition_date"])
    d["sex"] = pd.to_numeric(d["SEX"], errors="coerce")
    return d


def _adjust(d, fit_rows):
    """Residuals of an age + sex fit on ``fit_rows`` (never chosen by the outcome being tested)."""
    x = d.dropna(subset=["y", "age", "sex"])
    X = np.column_stack([np.ones(len(x)), x["age"], x["sex"]])
    beta = np.linalg.lstsq(X[fit_rows(x)], x["y"].to_numpy()[fit_rows(x)], rcond=None)[0]
    return x.assign(y=x["y"].to_numpy() - X @ beta)


def prodromal_dat(d, flow, n_boot=B):
    res = {"flow": {**flow, "prodromal": int(len(d)), "dat_matched": int(d["dat_d_nsdiss"].notna().sum())}}
    x = d.dropna(subset=["dat_d_nsdiss"])
    norm_, defi = x[x["dat_d_nsdiss"].eq(0)], x[x["dat_d_nsdiss"].eq(1)]
    res["primary"] = auc_with_ci(norm_["y"], defi["y"], norm_["vendor"], defi["vendor"], n_boot)
    res["pass"] = bool(res["primary"]["ci95"][0] > 0.50)
    x2 = x.dropna(subset=["dat_deficit_sbr"])
    a, b = x2[x2["dat_deficit_sbr"].eq(0)], x2[x2["dat_deficit_sbr"].eq(1)]
    adj = _adjust(x, lambda t: np.ones(len(t), bool))               # fitted on all prodromal: no DaT label enters it
    an, ad = adj[adj["dat_d_nsdiss"].eq(0)], adj[adj["dat_d_nsdiss"].eq(1)]
    s = x.dropna(subset=["sbr_pct_expected"])
    res["secondary"] = {"dat_deficit_sbr_0.65": auc_with_ci(a["y"], b["y"], a["vendor"], b["vendor"], n_boot),
                        "primary_age_sex_adjusted": auc_with_ci(an["y"], ad["y"], an["vendor"], ad["vendor"], n_boot),
                        "spearman_vs_sbr_pct_expected": {"rho": float(spearmanr(s["y"], s["sbr_pct_expected"])[0]), "n": int(len(s))}}
    return res


def fresh_clinical(d, flow, n_boot=B):
    res = {"flow": {**flow, "groups": d["group"].value_counts().to_dict(), "vendors": d.groupby(["vendor", "group"]).size().astype(int).to_dict()}}
    res["flow"]["vendors"] = {f"{k[0]}|{k[1]}": v for k, v in res["flow"]["vendors"].items()}
    hc, pdv = d[d["group"].eq("HC")], d[d["group"].eq("PD_denovo")]
    res["primary"] = auc_with_ci(hc["y"], pdv["y"], hc["vendor"], pdv["vendor"], n_boot)
    res["pass"] = bool(res["primary"]["ci95"][0] > 0.50)
    sh, sp = hc[hc["vendor"].eq("SIEMENS")], pdv[pdv["vendor"].eq("SIEMENS")]
    adj = _adjust(d, lambda t: t["group"].eq("HC").to_numpy())     # HC-only fit: no PD label enters it
    ah, ap = adj[adj["group"].eq("HC")], adj[adj["group"].eq("PD_denovo")]
    s = d.dropna(subset=["sbr_putamen_min"])
    res["secondary"] = {"siemens_only": auc_with_ci(sh["y"], sp["y"], sh["vendor"], sp["vendor"], n_boot) if len(sh) and len(sp) else None,
                        "age_sex_adjusted": auc_with_ci(ah["y"], ap["y"], ah["vendor"], ap["vendor"], n_boot),
                        "spearman_vs_lowest_putamen_sbr": {"rho": float(spearmanr(s["y"], s["sbr_putamen_min"])[0]), "n": int(len(s))}}
    return res


def synthetic(analysis, auc=0.65):
    """Fully synthetic rows (no real value, label or identifier)."""
    rng = np.random.default_rng(4)
    shift = np.sqrt(2) * norm.ppf(auc)
    n = 400
    vendor = rng.choice(["SIEMENS", "GE", "PHILIPS"], n, p=[0.6, 0.3, 0.1])
    offset = pd.Series(vendor).map({"SIEMENS": 0.0, "GE": -0.03, "PHILIPS": 0.02}).to_numpy()
    d = pd.DataFrame({"patno": np.arange(n), "vendor": vendor, "age": rng.normal(64, 8, n), "sex": rng.integers(0, 2, n).astype(float)})
    if analysis == "prodromal-dat":
        d["dat_d_nsdiss"] = (rng.random(n) < 0.15).astype(float)
        d["dat_deficit_sbr"] = d["dat_d_nsdiss"] * (rng.random(n) < 0.7)
        d["sbr_pct_expected"] = np.where(d["dat_d_nsdiss"] == 1, 0.6, 0.95) + rng.normal(0, 0.1, n)
        d["y"] = 0.2 + offset + 0.05 * (rng.normal(size=n) - shift * d["dat_d_nsdiss"])
    else:
        d["group"] = np.where(rng.random(n) < 0.12, "HC", "PD_denovo")
        d["sbr_putamen_min"] = np.where(d["group"] == "HC", 2.2, 1.0) + rng.normal(0, 0.3, n)
        d["y"] = 0.2 + offset + 0.05 * (rng.normal(size=n) - shift * (d["group"] == "PD_denovo"))
    return d, {"synthetic": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("analysis", choices=["prodromal-dat", "fresh-clinical"])
    ap.add_argument("--nm-dir")
    ap.add_argument("--stage1")
    ap.add_argument("--fresh", nargs="+", help="fresh-clinical: comma- or whitespace-separated PATNO lists written before the download")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    if a.synthetic:
        d, flow = synthetic(a.analysis)
    else:
        column, suffix = primary_of(a.stage1)
        d, flow = load(a.nm_dir, column, suffix)
        flow["primary_measure"] = column + suffix
        if a.analysis == "prodromal-dat":
            d = label_prodromal_dat(d, a.ppmi)
        else:
            fresh = {int(x) for f in a.fresh for x in Path(f).read_text().replace(",", " ").split()}
            d = label_fresh(d, a.ppmi, fresh)
    res = (prodromal_dat if a.analysis == "prodromal-dat" else fresh_clinical)(d, flow, a.n_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = ("synthetic_" if a.synthetic else "") + a.analysis.replace("-", "_")
    (out / f"{tag}_result.json").write_text(json.dumps(res, indent=1, default=float))
    p = res["primary"]
    if a.synthetic:
        print(f"synthetic dry run ({a.analysis}): n {p['n_pos']} / {p['n_neg']}; pass={res['pass']}")
    else:
        print(f"{a.analysis}: AUROC {p['auroc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] (n {p['n_pos']} / {p['n_neg']}) -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
