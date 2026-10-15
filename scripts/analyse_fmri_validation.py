"""Pre-registered rs-fMRI validation (documentation/preregistration_fmri.md), per protocol: end-to-end throughput,
motion (QC-FC on 4S456 against Ciric et al. 2017), split-half reliability (PPMI 2.0) and, reported only, posterior
putamen-SomMot coupling against the same side's post-commissural putamen DaT SBR.

    select    <out>/fmri_selection.csv: up to 100 scans per protocol, one per participant (earliest), stratified by the
              session's series naming (a proxy for site protocol), seed 20261015; reads series names and census TRs only
    analyse   <out>/fmri_validation.json: every statistic and the gates per protocol

    PYTHONPATH=. python scripts/analyse_fmri_validation.py select --zips <PPMI 2.0 zips> [--census <LONI census.csv>] --out <dir>
    PYTHONPATH=. python scripts/analyse_fmri_validation.py analyse --scans <scans.csv> --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/analyse_fmri_validation.py --synthetic --out <dir>     # dry run: every value and label is fake
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from pie.imaging.fmri_connectivity import network_means, qc_fc

SEED, N_PER_PROTOCOL, N_PERM = 20261015, 100, 1000
PPMI2_FROM = "2020-07-01"     # PPMI MRI metadata: protocol 001 rs-fMRI up to 2020 Q1, protocol 002 from 2020 Q4
RS = r"rs.?fmri|resting|bold"
# The median and distance criteria are judged against a permutation null (mean FD shuffled across scans): at N 100 the
# median |QC-FC| of a motion-free pipeline moves by several hundredths with the dependence between edges, so a fixed
# threshold would fail clean data by chance. A protocol fails only when it is worse than Ciric's value beyond the null's
# 95th percentile.
GATES = {"throughput": 0.80,            # feature rows / (selected - acquisition-driven exclusions)
         "qcfc_pct_fdr": 7.0,           # Ciric 2017 Results: 36P + any censoring left < 7 % of edges FDR-significant
         "qcfc_median_excess": None,    # Ciric 2017 Fig. 3A, 36P+scrub, minus the null median at N 393 (0.034): TO COPY
         "qcfc_distance_abs": None,     # Ciric 2017 Fig. 4A, 36P+scrub, |r| of QC-FC with distance: TO COPY
         "icc_network": 0.40}           # proposed: median ICC(3,1) of the 4S456 network-pair features, PPMI 2.0 strata
ICC_FAMILY_MIN = 0.40                   # a feature family below this median split-half ICC is not exported (per protocol)
COVERAGE_MAX_NAN = 0.10                 # proposed: > 10 % of 4S456 parcels blanked by XCP-D = a coverage failure
ACQUISITION = {"no_verified_reverse_pe_pair", "too_few_volumes", "header_tr_outside_stratum", "no_t1"}
FD, RMS, RETAINED = "fmri_xcpd_qc_mean_fd", "fmri_xcpd_qc_mean_relative_rms", "fmri_xcpd_qc_retained_seconds"
COUPLING = {s: f"fmri_xcpd_striatal_putamen_posterior_{s}__SomMot" for s in "lr"}
SBR = {s: f"sbr_post_putamen_{s}" for s in "lr"}


# ---------------------------------------------------------------- selection (series names and census TRs only)

def _stratum(era, tr, appa):
    if era == "P1":
        return next((f"P1_TR{t}" for t in (2.4, 2.5) if abs(tr - t) < 0.05), None)
    if np.isfinite(tr):
        return "P2_TR1" if tr <= 1.5 else "P2_TR2.5" if abs(tr - 2.5) < 0.05 else None
    return "P2_TR1" if appa else "P2_TR2.5"          # held zips have no TR: AP/PA naming is the multiband variant


def _sample(g, n):
    """Proportional allocation over naming signatures (largest remainder), then a seeded random draw within each."""
    if len(g) <= n:
        return g
    w = g["signature"].value_counts().sort_index()
    q = w / w.sum() * n
    k = np.floor(q).astype(int)
    k[(q - k).sort_values(ascending=False, kind="stable").index[:n - k.sum()]] += 1
    return pd.concat([g[g["signature"].eq(s)].sample(int(m), random_state=SEED) for s, m in k.items() if m])


def select(series, n=N_PER_PROTOCOL):
    """``series``: patno, date, desc, image_id[, tr in s]. One row per selected scan: the earliest rs-fMRI session of
    each participant within each stratum, all of that session's rs-fMRI series."""
    s = series[series["desc"].str.contains(RS, case=False)].copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["tr"] = pd.to_numeric(s.get("tr", np.nan), errors="coerce")
    s = s.dropna(subset=["date"]).assign(norm=s["desc"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True),
                                         appa=s["desc"].str.contains(r"(?:^|_)(?:AP|PA)(?:_|$)"))
    sess = s.groupby(["patno", "date"]).agg(signature=("norm", lambda x: "|".join(sorted(set(x)))), appa=("appa", "any"),
                                            tr=("tr", "median"), image_ids=("image_id", lambda x: ";".join(sorted(map(str, x))))).reset_index()
    sess["era"] = np.where(sess["date"] < PPMI2_FROM, "P1", "P2")
    sess["stratum"] = [_stratum(e, t, x) for e, t, x in zip(sess["era"], sess["tr"], sess["appa"])]
    sess = sess.dropna(subset=["stratum"]).sort_values(["date", "patno"]).drop_duplicates(["stratum", "patno"])
    picked = pd.concat([_sample(g, n) for _, g in sess.groupby("stratum")])
    counts = {k: {"eligible": int((sess["stratum"] == k).sum()), "selected": int((picked["stratum"] == k).sum()),
                  "signatures": int(sess.loc[sess["stratum"] == k, "signature"].nunique())} for k in sorted(sess["stratum"].unique())}
    return picked, counts


def _census(path):
    """LONI search export: Subject ID, Study Date, Description, Image ID, Imaging Protocol (TR in ms)."""
    c = pd.read_csv(path, dtype=str)
    tr = c["Imaging Protocol"].fillna("").str.extract(r"(?:^|;)TR=([0-9.]+)")[0].astype(float) / 1000
    return pd.DataFrame({"patno": pd.to_numeric(c["Subject ID"], errors="coerce"), "date": c["Study Date"],
                         "desc": c["Description"].fillna(""), "image_id": c["Image ID"], "tr": tr}).dropna(subset=["patno"])


# ---------------------------------------------------------------- statistics

def _edges(ts):
    """Fisher-z upper triangle (np.triu_indices order) of the parcel correlation matrix, as ``xcpd.xcpd_edges``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(np.asarray(ts, float).T)
    return np.arctanh(np.clip(r[np.triu_indices(len(r), 1)], -1 + 1e-7, 1 - 1e-7))


def _icc31(a, b):
    """ICC(3,1), two-way mixed, consistency, single measure (Shrout & Fleiss 1979), column-wise for n x F halves."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), (a + b) / 2
    gm = m.mean(axis=0)
    msr = 2 * ((m - gm) ** 2).sum(axis=0) / (n - 1)
    sse = ((a - gm) ** 2 + (b - gm) ** 2).sum(axis=0) - 2 * ((m - gm) ** 2).sum(axis=0) - n * ((a.mean(0) - gm) ** 2 + (b.mean(0) - gm) ** 2)
    mse = sse / (n - 1)
    return (msr - mse) / (msr + mse)


def _feature_icc(f1, f2):
    out = {}
    for c in f1.columns:
        ok = f1[c].notna() & f2[c].notna()
        out[c] = float(_icc31(f1.loc[ok, [c]], f2.loc[ok, [c]])[0]) if ok.sum() >= 10 else np.nan
    return pd.Series(out, dtype=float)


def _identification(e1, e2):
    """Finn et al. 2015: share of scans whose half-1 edges correlate best with their own half-2 edges, both directions."""
    r = np.corrcoef(e1, e2)[: len(e1), len(e1):]
    return float((np.mean(r.argmax(axis=1) == np.arange(len(r))) + np.mean(r.argmax(axis=0) == np.arange(len(r)))) / 2)


def _flow(g):
    reasons = g.loc[g["exclusion"].ne(""), "exclusion"].value_counts().to_dict()
    acq = sum(v for k, v in reasons.items() if k in ACQUISITION)
    rows = int(g["exclusion"].eq("").sum())
    eligible = len(g) - acq
    return {"selected": len(g), "feature_rows": rows, "acquisition_excluded": int(acq), "by_reason": reasons,
            "throughput": rows / eligible if eligible else None, "throughput_of_selected": rows / len(g) if len(g) else None,
            "retained_seconds_median": float(g.loc[g["exclusion"].eq(""), RETAINED].median())}


def _qcfc(d, edges, centroids, n_perm=N_PERM):
    d = d.dropna(subset=["age", "sex", FD])
    e, cov, dist = np.vstack([edges[k] for k in d["key"]]), d[["age", "sex"]], pdist(centroids)

    def median_and_distance(fd):                        # Pearson with distance, as Ciric (qc_fc's own is Spearman)
        q = qc_fc(e, fd, covariates=cov)["qcfc"]
        ok = np.isfinite(q)
        return np.median(np.abs(q[ok])), np.corrcoef(q[ok], dist[ok])[0, 1]

    rep = qc_fc(e, d[FD], covariates=cov, centroids=centroids)
    rep.pop("qcfc")
    rep["distance_dependence_r"] = float(median_and_distance(d[FD].to_numpy(float))[1])
    rng = np.random.default_rng(SEED)
    null = np.array([median_and_distance(rng.permutation(d[FD].to_numpy(float))) for _ in range(n_perm)])
    rep["null_median_abs_qcfc"], rep["null_p95_median_abs_qcfc"] = map(float, np.percentile(null[:, 0], [50, 95]))
    rep["null_p95_abs_distance_r"] = float(np.percentile(np.abs(null[:, 1]), 95))
    rep["median_excess"] = rep["median_abs_qcfc"] - rep["null_median_abs_qcfc"]
    if d[RMS].notna().all():                            # Ciric's own motion measure, reported
        rms = qc_fc(e, d[RMS], covariates=cov)
        rep["relative_rms"] = {k: rms[k] for k in ("percent_significant_fdr", "median_abs_qcfc")}
    return rep


def _reliability(d, ts, at):
    h = {k: [(_edges(x[: len(x) // 2]), _edges(x[len(x) // 2:])) for x in (ts[k]["4S456"], ts[k]["4S156"])] for k in d["key"]}
    e1, e2 = (np.vstack([h[k][0][i] for k in d["key"]]) for i in (0, 1))
    ok = np.isfinite(e1).all(axis=0) & np.isfinite(e2).all(axis=0)
    fam = {}
    for a, groups in ((0, at["groups456"]), (1, at["groups156"])):
        f1, f2 = (pd.DataFrame([network_means(h[k][a][i], groups) for k in d["key"]]) for i in (0, 1))
        fam[f"fc_{'4S456' if a == 0 else '4S156'}"] = _feature_icc(f1, f2)
    halves = sorted(c[:-4] for c in d.columns if c.endswith("__h1") and c[:-4] + "__h2" in d.columns)
    if halves:                                          # striatal seeds: halves computed by the run step on the BOLD
        fam["striatal"] = _feature_icc(d[[c + "__h1" for c in halves]].set_axis(halves, axis=1).reset_index(drop=True),
                                       d[[c + "__h2" for c in halves]].set_axis(halves, axis=1).reset_index(drop=True))
    edge_icc = _icc31(e1[:, ok], e2[:, ok])
    return {"n": len(d), "edge_icc_median": float(np.median(edge_icc)), "edges_used": int(ok.sum()),
            "identification_accuracy": _identification(e1[:, ok], e2[:, ok]),
            "families": {f: {"median_icc": float(v.median()), "n_features": int(v.notna().sum()),
                             "exported": bool(v.median() >= ICC_FAMILY_MIN)} for f, v in fam.items()}}


def _partial_r(d, x, y, covs):
    d = d.dropna(subset=[x, y] + covs)
    if len(d) < 20:
        return {"n": len(d), "r": None}
    X = np.column_stack([np.ones(len(d)), pd.get_dummies(d[covs], drop_first=True, dtype=float).to_numpy(float)])
    rx, ry = (v - X @ np.linalg.lstsq(X, v, rcond=None)[0] for v in (d[x].to_numpy(float), d[y].to_numpy(float)))
    r = float(np.corrcoef(rx, ry)[0, 1])
    se = 1 / np.sqrt(len(d) - X.shape[1] - 2)
    return {"n": len(d), "r": r, "ci95": [float(np.tanh(np.arctanh(r) - 1.96 * se)), float(np.tanh(np.arctanh(r) + 1.96 * se))]}


def _biology(scans):
    d = scans[scans["exclusion"].eq("") & ~scans["protocol"].str.contains(":")]
    d = d.sort_values("session_date").drop_duplicates("patno")          # a participant in two strata: earlier scan
    covs = ["age", "sex", "protocol", FD]
    return {grp: {s: _partial_r(sub, COUPLING[s], SBR[s], covs) for s in "lr"}
            for grp, sub in (("all", d), ("pd", d[d["cohort"].eq("Parkinson's Disease")]))}


def analyse(scans, ts, atlas, gates=GATES, n_perm=N_PERM):
    edges = {k: _edges(ts[k]["4S456"]) for k in scans.loc[scans["exclusion"].eq(""), "key"]}
    res = {"protocols": {}, "biology": _biology(scans), "gates_used": gates, "n_perm": n_perm}
    for p, g in scans.groupby("protocol"):
        base = p.split(":")[0]                          # "P2_TR1:no_notch": a reported-only variant of P2_TR1
        ok = g[g["exclusion"].eq("")]
        if len(ok) < 10:                                # too few feature rows to estimate anything: a failure, kept
            res["protocols"][p] = {"flow": _flow(g), "validated": None if ":" in p else False}
            continue
        r = {"flow": _flow(g), "qcfc": _qcfc(ok, edges, atlas[base]["centroids"], n_perm)}
        r["reliability"] = _reliability(ok, ts, atlas[base])
        q, thr = r["qcfc"], r["flow"]["throughput"]
        chance_median = q["null_p95_median_abs_qcfc"] - q["null_median_abs_qcfc"]
        gate = {"throughput": thr is not None and thr >= gates["throughput"],
                "qcfc_fdr": q["percent_significant_fdr"] < gates["qcfc_pct_fdr"],
                "qcfc_median": None if gates["qcfc_median_excess"] is None
                else q["median_excess"] <= gates["qcfc_median_excess"] + chance_median,
                "qcfc_distance": None if gates["qcfc_distance_abs"] is None
                else abs(q["distance_dependence_r"]) <= gates["qcfc_distance_abs"] + q["null_p95_abs_distance_r"]}
        if base.startswith("P2"):
            gate["reliability"] = r["reliability"]["families"]["fc_4S456"]["median_icc"] >= gates["icc_network"]
        r["gates"] = gate
        r["validated"] = None if ":" in p else (False if False in gate.values() else None if None in gate.values() else True)
        res["protocols"][p] = r
    return res


# ---------------------------------------------------------------- inputs

def synthetic(n=100, parcels=48, frames=240, motion=0.5, stable=1.0, seed=7):
    """Fully synthetic scans (no real value, label or identifier). ``motion`` scales a shared signal proportional to the
    scan's FD and strongest near a random point (so short edges inflate most); ``stable`` scales each scan's own
    between-network coupling, which is what split halves can recover."""
    rng = np.random.default_rng(seed)
    nets = np.repeat(["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Default"], parcels // 6)
    cen = rng.uniform(-70, 70, (parcels, 3))
    atlas = {p: {"groups456": list(nets), "groups156": list(nets[::2]), "centroids": cen}
             for p in ("P1_TR2.4", "P1_TR2.5", "P2_TR1", "P2_TR2.5")}
    net_idx = pd.factorize(nets)[0]
    reasons = ["no_verified_reverse_pe_pair", "too_few_volumes", "fmriprep_error", "xcpd_min_time", "coverage"]
    rows, ts = [], {}
    for j, p in enumerate(list(atlas) + ["P2_TR1:no_notch"]):
        for i in range(n):
            age, sex = rng.normal(66, 8), float(rng.integers(0, 2))
            fd = rng.lognormal(np.log(0.15) + 0.01 * (age - 66), 0.5)
            excl = rng.choice(reasons) if rng.random() < 0.12 else ""
            z = rng.normal(size=(frames, 6)) @ (np.eye(6) + stable * rng.normal(0, 0.5, (6, 6))).T
            x = 0.8 * z[:, net_idx] + rng.normal(size=(frames, parcels))
            near = np.exp(-np.linalg.norm(cen - rng.uniform(-70, 70, 3), axis=1) / 40)
            x += motion * fd / 0.15 * rng.normal(size=(frames, 1)) * near
            key = f"{p}/s{i}"
            ts[key] = {"4S456": x, "4S156": x.reshape(frames, parcels // 2, 2).mean(axis=2)}
            sbr = {s: max(0.1, rng.normal(1.2, 0.4)) for s in "lr"}
            c = {s: 0.15 + 0.08 * (sbr[s] - 1.2) + rng.normal(0, 0.06) for s in "lr"}
            rows.append({"key": key, "scan": f"s{i}", "patno": -(1000 * j + i), "session_date": pd.Timestamp("2000-01-01"),
                         "protocol": p, "exclusion": excl, "age": age, "sex": sex, FD: fd, RMS: fd / 2,
                         RETAINED: frames * 2.0, "cohort": rng.choice(["Parkinson's Disease", "Prodromal"]),
                         **{COUPLING[s]: c[s] for s in "lr"}, **{SBR[s]: sbr[s] for s in "lr"},
                         **{f"{COUPLING[s]}__h{h}": c[s] + rng.normal(0, 0.03) for s in "lr" for h in (1, 2)}})
    return pd.DataFrame(rows), ts, atlas


def _post_putamen_sbr(ppmi, s, max_months=18):
    """Xing post-commissural putamen SBR per side, the DaTscan nearest the fMRI session within 18 months."""
    from pie.imaging.labels import dat_sbr_table

    cols = {"POSCOMMISSURAL_PUTAMEN_L_REF_CWM": SBR["l"], "POSCOMMISSURAL_PUTAMEN_R_REF_CWM": SBR["r"]}
    sbr, _ = dat_sbr_table(Path(ppmi), "xing")
    m = s[["patno", "session_date"]].reset_index().merge(sbr[["PATNO", "DATSCAN_DATE", *cols]], left_on="patno", right_on="PATNO")
    m["gap"] = (m["DATSCAN_DATE"] - m["session_date"]).dt.days.abs() / 30.44
    m = m[m["gap"] <= max_months].sort_values("gap").drop_duplicates("index").set_index("index")
    return m[list(cols)].rename(columns=cols).reindex(s.index)


def _atlas_info(xcpd_dir, names456, names156):
    import nibabel as nib

    from pie.imaging.fmri_connectivity import parcel_centroids
    from pie.imaging.xcpd import _atlas

    root = Path(xcpd_dir)
    found = lambda pat: next(p for r in ("atlases", "sourcedata/atlases") for p in sorted((root / r).glob(f"**/{pat}")))
    index = pd.read_table(found("*atlas-4S456Parcels_*dseg.tsv")).set_index("name")["index"]
    g456, g156 = _atlas(root, "4S456Parcels")[0], _atlas(root, "4S156Parcels")[0]
    return {"groups456": [g456[n] for n in names456], "groups156": [g156[n] for n in names156],
            "centroids": parcel_centroids(nib.load(found("*atlas-4S456Parcels_*dseg.nii.gz")), [index[n] for n in names456])}


def load(scans_csv, ppmi):
    """The run step's table: one row per selected scan (scan = BIDS label, patno, session_date, protocol, exclusion,
    xcpd_dir, the ``xcpd_features`` and ``xcpd_striatal_features`` columns, striatal ``__h1``/``__h2`` halves).
    Parcel series come from XCP-D's censored ``stat-mean_timeseries.tsv``; edges recomputed from them must equal
    XCP-D's own Pearson matrix on every scan."""
    from pie.imaging.labels import _age_at, covariates
    from pie.imaging.xcpd import _one, xcpd_edges

    s = pd.read_csv(scans_csv, low_memory=False)
    s["exclusion"] = s["exclusion"].fillna("").astype(str)
    s["session_date"] = pd.to_datetime(s["session_date"])
    s["key"] = s["protocol"] + "/" + s["scan"].astype(str)
    cov = covariates(Path(ppmi))
    s = s.merge(cov[["PATNO", "SEX", "COHORT"]], left_on="patno", right_on="PATNO", how="left")
    s["age"], s["sex"], s["cohort"] = _age_at(cov, s["patno"], s["session_date"]), pd.to_numeric(s["SEX"], errors="coerce"), s["COHORT"]
    s = s.join(_post_putamen_sbr(ppmi, s))
    ts, atlas = {}, {}
    for row in s[s["exclusion"].eq("")].itertuples():
        func = Path(row.xcpd_dir) / f"sub-{row.scan}"
        series = {a: pd.read_table(_one(func, f"sub-{row.scan}_*task-rest_*atlas-{a}Parcels_*stat-mean_timeseries.tsv"))
                  for a in ("4S456", "4S156")}
        names, e = xcpd_edges(row.xcpd_dir, row.scan)
        if list(series["4S456"].columns) != names or not np.allclose(_edges(series["4S456"]), e, atol=1e-6, equal_nan=True):
            raise ValueError(f"{row.key}: parcel series and XCP-D's matrix disagree")
        base = row.protocol.split(":")[0]
        if base not in atlas:
            atlas[base] = _atlas_info(row.xcpd_dir, names, list(series["4S156"].columns))
        ts[row.key] = {a: v.to_numpy(float) for a, v in series.items()}
    bad = s["exclusion"].eq("") & (s["fmri_xcpd_parcels_nan_4S456Parcels"] > COVERAGE_MAX_NAN * 456)
    s.loc[bad, "exclusion"] = "coverage"
    return s, ts, atlas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", nargs="?", default="analyse", choices=["select", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--zips", nargs="*", default=[], help="held LONI zips (PPMI 2.0 rs-fMRI)")
    ap.add_argument("--census", help="LONI fMRI search export with Imaging Protocol (TR), for scans not yet held")
    ap.add_argument("--scans", help="the run step's per-scan table")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-perm", type=int, default=N_PERM, help="permutations of mean FD for the QC-FC null")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        from pie.imaging.batch import index_series

        parts = ([index_series(a.zips)[["patno", "date", "desc", "image_id"]]] if a.zips else []) + ([_census(a.census)] if a.census else [])
        picked, counts = select(pd.concat(parts, ignore_index=True))
        picked.to_csv(out / "fmri_selection.csv", index=False)
        print(json.dumps(counts, indent=1))             # counts only: the list itself stays in <out>
        return counts
    scans, ts, atlas = synthetic() if a.synthetic else load(a.scans, a.ppmi)
    res = analyse(scans, ts, atlas, n_perm=a.n_perm)
    (out / ("synthetic_fmri_validation.json" if a.synthetic else "fmri_validation.json")).write_text(json.dumps(res, indent=1, default=float))
    for p, r in res["protocols"].items():
        if "qcfc" not in r:
            print(f"{p}: {r['flow']['feature_rows']} feature rows of {r['flow']['selected']} -> {r['validated']}")
            continue
        f, q, rel = r["flow"], r["qcfc"], r["reliability"]
        print(f"{'synthetic dry run: ' if a.synthetic else ''}{p}: throughput {f['feature_rows']}/{f['selected'] - f['acquisition_excluded']}"
              f" ({np.nan if f['throughput'] is None else f['throughput']:.2f}); QC-FC FDR {q['percent_significant_fdr']:.1f} %, median |r| {q['median_abs_qcfc']:.3f}"
              f" (null {q['null_median_abs_qcfc']:.3f}, p95 {q['null_p95_median_abs_qcfc']:.3f}), distance r {q['distance_dependence_r']:.2f}; 4S456 network ICC"
              f" {rel['families']['fc_4S456']['median_icc']:.2f}, edge ICC {rel['edge_icc_median']:.2f}, identification"
              f" {rel['identification_accuracy']:.2f}; gates {r['gates']} -> {r['validated']}")
    return res


if __name__ == "__main__":
    main()
