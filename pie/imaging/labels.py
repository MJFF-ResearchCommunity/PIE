"""
labels.py — outcome labels and basic covariates aligned to an MRI session.

dat_labels : DaTscan closest in time to the MRI. ``dat_visual`` = PPMI visual read positive;
             ``dat_deficit_sbr`` = lowest putamen SBR below ``threshold`` x the age/sex-expected
             value, the expectation being a linear fit on visually-negative healthy controls
             (PPMI's prodromal-cohort convention, 65 %). SBRs come from PPMI's primary table, the
             XingImaging core lab (cerebral white-matter reference, all scans since 2024 re-analysed),
             else the archived Invicro table (occipital reference); ``sbr_source`` says which. The 65 %
             rule was derived on Invicro's scale and is applied unchanged to Xing rows unless
             ``xing_threshold`` is given; ``xing_deficit_threshold`` derives the matching Xing cut-off.
saa_labels : CSF alpha-synuclein SAA status at the MRI visit (SC/BL pairs allowed).
             Unmatched visits require an explicit sensitivity-analysis opt-in; provenance is retained.
covariates : sex, birth month, cohort, enrolment, LRRK2/GBA/SNCA/APOE genotype and (when the Polygenic_Risk_Scores table
             is present) the GP2 and META5-without-LRRK2/GBA polygenic risk scores per PATNO.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _latest(ppmi_dir, sub, stem):
    """Newest release of a PPMI table by the date in its name (``18Mar2025`` sorts after ``08Sep2026`` as text)."""
    from .viewer.catalog import latest_table

    path = latest_table(Path(ppmi_dir, sub), stem)
    if path is None:
        raise FileNotFoundError(f"{stem}_*.csv not found under {sub}")
    return pd.read_csv(path, low_memory=False)


def _month(s):
    return pd.to_datetime(s, format="%m/%Y", errors="coerce")


def _carrier(values):
    """1 = carrier, 0 = explicit negative (0, 0.0, "0", "0.0"), NaN = untested (missing or blank), for any dtype.
    A CSV column with blanks reads as float, so comparing its text with "0" would code every 0.0 as a carrier."""
    s = pd.Series(values).astype(object)
    text = s.astype(str).str.strip()
    missing = s.isna().to_numpy() | text.eq("").to_numpy()
    negative = pd.to_numeric(text, errors="coerce").eq(0).to_numpy()
    return np.where(missing, np.nan, np.where(negative, 0.0, 1.0))


def covariates(ppmi_dir):
    ps = _latest(ppmi_dir, "_Subject_Characteristics", "Participant_Status")
    ps = ps[["PATNO", "COHORT_DEFINITION", "ENROLL_DATE", "ENROLL_AGE"]].rename(columns={"COHORT_DEFINITION": "COHORT"})
    dm = _latest(ppmi_dir, "_Subject_Characteristics", "Demographics")
    dm = dm.sort_values("LAST_UPDATE").drop_duplicates("PATNO", keep="last")[["PATNO", "SEX", "BIRTHDT", "HANDED"]]
    dm["BIRTHDT"] = _month(dm["BIRTHDT"])
    gen = _latest(ppmi_dir, "_Subject_Characteristics", "iu_genetic_consensus")
    gen = gen[["PATNO", "LRRK2", "GBA", "SNCA", "APOE", "PATHVAR_COUNT"]].drop_duplicates("PATNO")
    for g in ["LRRK2", "GBA", "SNCA"]:
        gen[f"{g}_carrier"] = _carrier(gen[g])
    gen["APOE_e4"] = gen["APOE"].astype(str).str.count("E4").where(gen["APOE"].notna())
    out = ps.merge(dm, on="PATNO", how="left").merge(gen, on="PATNO", how="left")
    try:  # PPMI Risk_SNP_PGS table: PD-GWAS polygenic scores; META5 variant excludes the LRRK2 and GBA loci already coded above
        p = _latest(ppmi_dir, "_Subject_Characteristics", "Polygenic_Risk_Scores").drop_duplicates("PATNO")
        p = p[["PATNO", "GP2_PGS", "META5_excl_LRRK2_GBA_PGS"]]
        out = out.merge(p.rename(columns={"GP2_PGS": "PRS_GP2", "META5_excl_LRRK2_GBA_PGS": "PRS_META5_noLRRK2GBA"}), on="PATNO", how="left")
    except FileNotFoundError:
        pass
    return out


def _age_at(cov, patnos, dates):
    birth = cov.set_index("PATNO")["BIRTHDT"].reindex(patnos).to_numpy()
    return (pd.to_datetime(dates).to_numpy() - birth) / np.timedelta64(1, "D") / 365.25


def _ioflupane(t):
    """Rows not recorded as another tracer: PPMI-1 rows leave DATSCAN_LIGAND blank, and all of them are DaTscan."""
    lig = t.get("DATSCAN_LIGAND", pd.Series(index=t.index, dtype=object))
    return lig.isna() | lig.astype(str).str.contains("DaTscan", case=False)


XING_SBR, XING_VISUAL = "Xing_Core_Lab_-_Quant_SBR", "Xing_Core_Lab_-_Visual_Read"
INVICRO_SBR, INVICRO_VISUAL = "DaTScan_SBR_Analysis", "DaTScan_Visual_Interpretation_Results"


def dat_sbr_table(ppmi_dir, source="auto"):
    """PPMI's DaTscan SBRs as ``DATSCAN_{CAUDATE,PUTAMEN}_{R,L}`` plus ``sbr_source``, and its visual reads.

    ``xing``: the XingImaging core lab table, PPMI's primary SBRs since December 2024 (all earlier scans
    re-analysed; MIAKAT, affine to a DaT template in MNI152, CIC atlas, cerebral white-matter reference); its
    sub-regional columns (``*_REF_CWM``) are kept. ``invicro``: the archived Invicro table (occipital reference;
    no scans after 1 December 2024). ``auto`` takes Xing when present. The two scales differ, so never pool them.
    Only 123I-ioflupane scans are returned (the Xing table also holds 99mTc-TRODAT-1 scans).
    """
    if source in ("auto", "xing"):
        try:
            sbr, vis = _latest(ppmi_dir, "Imaging", XING_SBR), _latest(ppmi_dir, "Imaging", XING_VISUAL)
            sbr = sbr.assign(**{f"DATSCAN_{r}_{s}": sbr[f"{r}_{s}_REF_CWM"] for r in ("CAUDATE", "PUTAMEN") for s in "RL"},
                             sbr_source="xing_cwm")
            source = "xing"
        except FileNotFoundError:
            if source == "xing":
                raise
    if source in ("auto", "invicro"):
        sbr, vis = _latest(ppmi_dir, "Imaging", INVICRO_SBR), _latest(ppmi_dir, "Imaging", INVICRO_VISUAL)
        sbr = sbr.assign(sbr_source="invicro_occipital")
    sbr = sbr[_ioflupane(sbr)]
    sbr = sbr[sbr["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()
    sbr["DATSCAN_DATE"] = _month(sbr["DATSCAN_DATE"])
    vis = vis[_ioflupane(vis)]
    vis = vis.assign(DATSCAN_DATE=_month(vis["DATSCAN_DATE"]))
    return sbr, vis


def sbr_indices(caudate_l, caudate_r, putamen_l, putamen_r):
    """DaTQUANT-style indices from four SBR Series: putamen-to-caudate ratio per side (``pc_ratio_{l,r}``) and the
    asymmetry index |L - R| / mean(L, R) x 100 of each nucleus (``ai_{putamen,caudate}``). Non-positive SBRs give NaN."""
    c_l, c_r, p_l, p_r = (pd.to_numeric(v, errors="coerce").where(lambda x: x > 0) for v in (caudate_l, caudate_r, putamen_l, putamen_r))
    ai = lambda left, right: (left - right).abs() / ((left + right) / 2) * 100
    return pd.DataFrame({"pc_ratio_l": p_l / c_l, "pc_ratio_r": p_r / c_r, "ai_putamen": ai(p_l, p_r), "ai_caudate": ai(c_l, c_r)})


def percent_expected(sbr, value="sbr_putamen_min"):
    """``value`` as a fraction of its age + sex expectation and as a z-score against the residual SD, the expectation
    being a linear fit on visually-negative healthy controls (one scan each, the earliest: repeat visits are not new
    people). ``sbr`` needs PATNO, DATSCAN_DATE, COHORT, dat_visual, age_at_datscan, SEX and ``value``."""
    hc = sbr[(sbr["COHORT"] == "Healthy Control") & (sbr["dat_visual"] == 0)].dropna(subset=["age_at_datscan", "SEX", value])
    hc = hc.sort_values("DATSCAN_DATE").drop_duplicates("PATNO")
    X = np.column_stack([np.ones(len(hc)), hc["age_at_datscan"], hc["SEX"]])
    beta, *_ = np.linalg.lstsq(X, hc[value].to_numpy(), rcond=None)
    ok = sbr[["age_at_datscan", "SEX"]].notna().all(axis=1)
    expected = beta[0] + beta[1] * sbr["age_at_datscan"] + beta[2] * sbr["SEX"]
    resid_sd = float(np.std(hc[value].to_numpy() - X @ beta, ddof=X.shape[1]))
    return np.where(ok, sbr[value] / expected, np.nan), np.where(ok, (sbr[value] - expected) / resid_sd, np.nan)


def dat_sbr_scored(ppmi_dir, source):
    """``dat_sbr_table(ppmi_dir, source)`` (``xing`` or ``invicro``) with visual reads, covariates, the summary SBRs and
    indices, and ``sbr_pct_expected`` / ``sbr_putamen_min_z`` against that table's own controls: one row per scan."""
    cov = covariates(ppmi_dir)
    sbr, vis = dat_sbr_table(ppmi_dir, source)
    vis["dat_visual"] = vis["DATSCAN_VISINTRP"].str.lower().map({"positive": 1.0, "negative": 0.0})
    sbr = sbr.merge(vis[["PATNO", "DATSCAN_DATE", "dat_visual"]].drop_duplicates(["PATNO", "DATSCAN_DATE"]),
                    on=["PATNO", "DATSCAN_DATE"], how="left")
    sbr["sbr_putamen_min"] = sbr[["DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L"]].min(axis=1)
    sbr["sbr_caudate_min"] = sbr[["DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L"]].min(axis=1)
    sbr["sbr_putamen_mean"] = sbr[["DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L"]].mean(axis=1)
    idx = sbr_indices(sbr["DATSCAN_CAUDATE_L"], sbr["DATSCAN_CAUDATE_R"], sbr["DATSCAN_PUTAMEN_L"], sbr["DATSCAN_PUTAMEN_R"])
    sbr[[f"sbr_{c}" for c in idx]] = idx.to_numpy()
    sbr["age_at_datscan"] = _age_at(cov, sbr["PATNO"], sbr["DATSCAN_DATE"])
    sbr = sbr.merge(cov[["PATNO", "SEX", "COHORT"]], on="PATNO", how="left")
    sbr["sbr_pct_expected"], sbr["sbr_putamen_min_z"] = percent_expected(sbr)     # DaTQUANT-style z vs controls
    return sbr


def xing_deficit_threshold(ppmi_dir, invicro_threshold=0.65):
    """The percent-of-expected cut-off on Xing's cerebral-WM scale that calls as many deficits as ``invicro_threshold``
    does on Invicro's occipital scale (where the 65 % rule was derived: Jennings et al. 2017), on the scans both core
    labs analysed (same PATNO and scan month): midway between the k-th and (k+1)-th lowest Xing values, k = Invicro's
    deficit count. Percent-of-expected is not scale-free: SBR = target / reference - 1, and the white-matter reference
    is hotter than the occipital cortex, so the same fraction marks a different loss. Returns a dict: threshold,
    n_overlap, n_deficit, invicro_rate, kappa (agreement of the two calls)."""
    on = ["PATNO", "DATSCAN_DATE"]
    x, i = (dat_sbr_scored(ppmi_dir, s).drop_duplicates(on) for s in ("xing", "invicro"))
    m = x[on + ["sbr_pct_expected"]].merge(i[on + ["sbr_pct_expected"]], on=on, suffixes=("_x", "_i")).dropna()
    xs = np.sort(m["sbr_pct_expected_x"].to_numpy())
    k = int((m["sbr_pct_expected_i"] <= invicro_threshold).sum())
    t = xs[0] if k == 0 else xs[-1] if k == len(xs) else (xs[k - 1] + xs[k]) / 2
    return {"threshold": float(t), "n_overlap": len(m), "n_deficit": k, "invicro_rate": k / max(len(m), 1),
            "kappa": kappa(m["sbr_pct_expected_i"] <= invicro_threshold, m["sbr_pct_expected_x"] <= t)}


def kappa(a, b):
    """Cohen's kappa of two boolean calls on the same scans (NaN when chance agreement is 1)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float(((a == b).mean() - pe) / (1 - pe)) if pe < 1 else np.nan


def dat_labels(ppmi_dir, sessions, threshold=0.65, max_months=18, source="auto", xing_threshold=None):
    """Return PATNO, IMAGEID, DATSCAN_DATE, months_to_datscan, sbr_* columns, dat_visual, dat_deficit_sbr, sbr_source.
    ``source``: see ``dat_sbr_table``; the age/sex expectation is fitted within the chosen table. ``threshold`` (0.65)
    was derived on Invicro's occipital scale; ``xing_threshold`` is the cut-off for Xing rows, None = ``threshold``
    (what PIE has applied to Xing rows since adopting that table, kept so exported labels do not change silently;
    ``xing_deficit_threshold`` derives the Xing equivalent). ``dat_deficit_threshold`` records the cut-off each row
    used. ``auto``: Xing for every session it
    has a scan for, else the Invicro match (the September 2026 Xing release lacks ~716 PPMI-1 participants Invicro
    analysed); each row's ``sbr_source`` says which, and raw SBRs of the two sources must not be pooled
    (percent-of-expected, deficit and visual labels are each on their own table's norm)."""
    if source == "auto":
        parts = []
        for s in ("xing", "invicro"):
            try:
                parts.append(dat_labels(ppmi_dir, sessions, threshold, max_months, source=s, xing_threshold=xing_threshold))
            except FileNotFoundError:
                pass
        if not parts:
            raise FileNotFoundError("no PPMI DaTscan SBR table")
        parts = [x for x in parts if len(x)] or parts[:1]          # a table with no match has no columns
        return pd.concat(parts, ignore_index=True).drop_duplicates(["PATNO", "IMAGEID"], keep="first").reset_index(drop=True) if len(parts[0]) else parts[0]
    sbr = dat_sbr_scored(ppmi_dir, source)
    # deficit rules, each named after its source (all "at or below" the fraction of the expected value):
    #   dat_deficit_sbr  <= threshold (0.65: PPMI prodromal / PARS, Jennings et al. 2017, on Invicro's occipital scale;
    #                    0.65-0.80 is their indeterminate band)
    #   dat_d_nsdiss     <= 0.75: the D+ rule of the Neuronal alpha-Synuclein Disease Integrated Staging System (Dam et al. 2024)
    cut = xing_threshold if (source == "xing" and xing_threshold is not None) else threshold
    pct = sbr["sbr_pct_expected"]
    sbr["dat_deficit_sbr"] = np.where(pct.notna(), (pct <= cut).astype(float), np.nan)
    sbr["dat_deficit_threshold"] = cut
    sbr["dat_d_nsdiss"] = np.where(pct.notna(), (pct <= 0.75).astype(float), np.nan)

    keep = ["DATSCAN_DATE", "EVENT_ID", "DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L", "DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L",
            "sbr_putamen_min", "sbr_caudate_min", "sbr_putamen_mean", "sbr_pc_ratio_l", "sbr_pc_ratio_r", "sbr_ai_putamen",
            "sbr_ai_caudate", "sbr_pct_expected", "sbr_putamen_min_z", "dat_visual", "dat_deficit_sbr", "dat_deficit_threshold",
            "dat_d_nsdiss", "sbr_source"]
    out = []
    for s in sessions[["patno", "image_id", "session_date"]].itertuples(index=False):
        cand = sbr[sbr["PATNO"] == s.patno].dropna(subset=["DATSCAN_DATE"])
        if cand.empty or not 1900 < pd.to_datetime(s.session_date, errors="coerce").year < 2100:   # LONI masks dates as 9999
            continue
        months = ((cand["DATSCAN_DATE"] - pd.Timestamp(s.session_date)).dt.days / 30.44).abs()
        i = months.idxmin()
        if months[i] > max_months:
            continue
        # signed gap too: a DaTscan after the MRI (up to max_months) can carry a later disease state, e.g. a converter's
        row = {"PATNO": s.patno, "IMAGEID": s.image_id, "months_to_datscan": round(float(months[i]), 1),
               "days_datscan_after_mri": int((cand.loc[i, "DATSCAN_DATE"] - pd.Timestamp(s.session_date)).days)}
        row.update({k: cand.loc[i, k] for k in keep})
        out.append(row)
    return pd.DataFrame(out).rename(columns={"EVENT_ID": "DATSCAN_EVENT_ID"})


def ppmi_dti_roi_table(ppmi_dir):
    """PPMI's hand-drawn nigral DTI ROIs (CIND; Schuff et al. 2015, doi:10.1002/mds.26325; three ROIs per side: I rostral,
    II middle, III caudal; two cerebral-peduncle references), one row per PATNO and scan month: a reference for
    validating automated nigral values. Columns: PATNO, DTI_DATE, sn_fa, sn_rostral_fa, sn_caudal_fa, sn_md (mean of
    the three eigenvalues), peduncle_fa, n_rows. A scan month read more than once (33 of 263 in September 2026, SN FA
    differing by up to 0.07) is averaged over its readings, each measure over the same rows; n_rows counts them."""
    t = _latest(ppmi_dir, "Imaging", "DTI_Regions_of_Interest")
    roi, ref = [f"ROI{i}" for i in range(1, 7)], ["REF1", "REF2"]
    t[roi + ref] = t[roi + ref].apply(pd.to_numeric, errors="coerce")
    t["DTI_DATE"] = _month(t["INFODT"])
    w = t.pivot_table(index=["PATNO", "DTI_DATE"], columns="Measure", values=roi + ref, aggfunc="mean")
    fa = w.xs("FA", axis=1, level=1)
    out = pd.DataFrame({"sn_fa": fa[roi].mean(axis=1), "sn_rostral_fa": fa[["ROI1", "ROI4"]].mean(axis=1),
                        "sn_caudal_fa": fa[["ROI3", "ROI6"]].mean(axis=1), "peduncle_fa": fa[ref].mean(axis=1),
                        "sn_md": sum(w.xs(e, axis=1, level=1)[roi].mean(axis=1) for e in ("E1", "E2", "E3")) / 3,
                        "n_rows": t[t["Measure"] == "FA"].groupby(["PATNO", "DTI_DATE"]).size()})
    return out.reset_index()


def saa_labels(ppmi_dir, sessions, allow_unmatched=False):
    """Visit-aligned SAA with match provenance. RUNDATE is an assay date, not a CSF collection date.

    Do not substitute a future positive/negative assay for an inconclusive concurrent visit. Conflicting calls
    at the selected visit have no binary label. ``allow_unmatched`` reproduces the historical fallback for a
    separately labelled sensitivity analysis; it does not establish temporal alignment.
    """
    saa = _latest(ppmi_dir, "Biospecimen", "SAA_Biospecimen_Analysis_Results_*.csv")
    saa = saa.assign(SAA_Type=saa.get("SAA_Type"))
    saa = saa[["PATNO", "CLINICAL_EVENT", "SAA_Status", "SAA_Type"]].copy()
    order = {"BL": 0, "SC": 1}
    out = []
    for s in sessions[["patno", "image_id", "EVENT_ID"]].itertuples(index=False):
        cand = saa[saa["PATNO"] == s.patno]
        if cand.empty:
            continue
        pick = cand[cand["CLINICAL_EVENT"] == s.EVENT_ID]
        match = "same_visit"
        if pick.empty and s.EVENT_ID in ("SC", "BL"):
            pick = cand[cand["CLINICAL_EVENT"].isin(["SC", "BL"])]
            match = "screening_baseline_pair"
        if pick.empty and allow_unmatched:
            cand = cand.assign(_o=cand["CLINICAL_EVENT"].map(order).fillna(9))
            event = cand.sort_values(["_o", "CLINICAL_EVENT"]).iloc[0]["CLINICAL_EVENT"]
            pick = cand[cand["CLINICAL_EVENT"] == event]
            match = "unmatched_visit"
        if pick.empty:
            continue
        statuses = pick["SAA_Status"].dropna().unique()
        valid = len(statuses) == 1 and statuses[0] in ("Positive", "Negative")
        r = pick.iloc[0]
        out.append({"PATNO": s.patno, "IMAGEID": s.image_id, "SAA_EVENT_ID": r["CLINICAL_EVENT"],
                    "SAA_Status": r["SAA_Status"] if len(statuses) <= 1 else "Conflicting",
                    "SAA_Type": r["SAA_Type"], "saa_positive": float(r["SAA_Status"] == "Positive") if valid else np.nan,
                    "saa_match": match, "saa_visit_concurrent": match != "unmatched_visit"})
    return pd.DataFrame(out, columns=["PATNO", "IMAGEID", "SAA_EVENT_ID", "SAA_Status", "SAA_Type", "saa_positive", "saa_match", "saa_visit_concurrent"])
