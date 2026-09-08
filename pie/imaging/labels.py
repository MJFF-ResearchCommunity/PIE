"""
labels.py — outcome labels and basic covariates aligned to an MRI session.

dat_labels : DaTscan closest in time to the MRI. ``dat_visual`` = PPMI visual read positive;
             ``dat_deficit_sbr`` = lowest putamen SBR below ``threshold`` x the age/sex-expected
             value, the expectation being a linear fit on visually-negative healthy controls
             (PPMI's prodromal-cohort convention, 65 %).
saa_labels : CSF alpha-synuclein SAA status (Positive=1 / Negative=0; Inconclusive dropped),
             sample at the MRI visit if present, else baseline, else the earliest visit.
covariates : sex, birth month, cohort, enrolment, LRRK2/GBA/SNCA/APOE genotype per PATNO.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _latest(ppmi_dir, sub, pattern):
    files = sorted(Path(ppmi_dir, sub).glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} not found under {sub}")
    return pd.read_csv(files[-1], low_memory=False)


def _month(s):
    return pd.to_datetime(s, format="%m/%Y", errors="coerce")


def covariates(ppmi_dir):
    ps = _latest(ppmi_dir, "_Subject_Characteristics", "Participant_Status_*.csv")
    ps = ps[["PATNO", "COHORT_DEFINITION", "ENROLL_DATE", "ENROLL_AGE"]].rename(columns={"COHORT_DEFINITION": "COHORT"})
    dm = _latest(ppmi_dir, "_Subject_Characteristics", "Demographics_*.csv")
    dm = dm.sort_values("LAST_UPDATE").drop_duplicates("PATNO", keep="last")[["PATNO", "SEX", "BIRTHDT", "HANDED"]]
    dm["BIRTHDT"] = _month(dm["BIRTHDT"])
    gen = _latest(ppmi_dir, "_Subject_Characteristics", "iu_genetic_consensus_*.csv")
    gen = gen[["PATNO", "LRRK2", "GBA", "SNCA", "APOE", "PATHVAR_COUNT"]].drop_duplicates("PATNO")
    for g in ["LRRK2", "GBA", "SNCA"]:
        gen[f"{g}_carrier"] = np.where(gen[g].isna(), np.nan, (gen[g].astype(str) != "0").astype(float))
    gen["APOE_e4"] = gen["APOE"].astype(str).str.count("E4").where(gen["APOE"].notna())
    return ps.merge(dm, on="PATNO", how="left").merge(gen, on="PATNO", how="left")


def _age_at(cov, patnos, dates):
    birth = cov.set_index("PATNO")["BIRTHDT"].reindex(patnos).to_numpy()
    return (pd.to_datetime(dates).to_numpy() - birth) / np.timedelta64(365, "D")


def dat_labels(ppmi_dir, sessions, threshold=0.65, max_months=18):
    """Return PATNO, IMAGEID, DATSCAN_DATE, months_to_datscan, sbr_* columns, dat_visual, dat_deficit_sbr."""
    cov = covariates(ppmi_dir)
    sbr = _latest(ppmi_dir, "Imaging", "DaTScan_SBR_Analysis_*.csv")
    sbr = sbr[sbr["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()
    sbr["DATSCAN_DATE"] = _month(sbr["DATSCAN_DATE"])
    vis = _latest(ppmi_dir, "Imaging", "DaTScan_Visual_Interpretation_Results_*.csv")
    vis["DATSCAN_DATE"] = _month(vis["DATSCAN_DATE"])
    vis["dat_visual"] = vis["DATSCAN_VISINTRP"].str.lower().map({"positive": 1.0, "negative": 0.0})
    sbr = sbr.merge(vis[["PATNO", "DATSCAN_DATE", "dat_visual"]].drop_duplicates(["PATNO", "DATSCAN_DATE"]),
                    on=["PATNO", "DATSCAN_DATE"], how="left")
    sbr["sbr_putamen_min"] = sbr[["DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L"]].min(axis=1)
    sbr["sbr_caudate_min"] = sbr[["DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L"]].min(axis=1)
    sbr["sbr_putamen_mean"] = sbr[["DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L"]].mean(axis=1)
    sbr["age_at_datscan"] = _age_at(cov, sbr["PATNO"], sbr["DATSCAN_DATE"])
    sbr = sbr.merge(cov[["PATNO", "SEX", "COHORT"]], on="PATNO", how="left")

    # expected lowest-putamen SBR from visually-negative healthy controls: linear in age + sex
    hc = sbr[(sbr["COHORT"] == "Healthy Control") & (sbr["dat_visual"] == 0)].dropna(subset=["age_at_datscan", "SEX", "sbr_putamen_min"])
    X = np.column_stack([np.ones(len(hc)), hc["age_at_datscan"], hc["SEX"]])
    beta, *_ = np.linalg.lstsq(X, hc["sbr_putamen_min"].to_numpy(), rcond=None)
    ok = sbr[["age_at_datscan", "SEX"]].notna().all(axis=1)
    expected = beta[0] + beta[1] * sbr["age_at_datscan"] + beta[2] * sbr["SEX"]
    sbr["sbr_pct_expected"] = np.where(ok, sbr["sbr_putamen_min"] / expected, np.nan)
    sbr["dat_deficit_sbr"] = np.where(sbr["sbr_pct_expected"].notna(), (sbr["sbr_pct_expected"] < threshold).astype(float), np.nan)

    keep = ["DATSCAN_DATE", "EVENT_ID", "DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L", "DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L",
            "sbr_putamen_min", "sbr_caudate_min", "sbr_putamen_mean", "sbr_pct_expected", "dat_visual", "dat_deficit_sbr"]
    out = []
    for s in sessions[["patno", "image_id", "session_date"]].itertuples(index=False):
        cand = sbr[sbr["PATNO"] == s.patno].dropna(subset=["DATSCAN_DATE"])
        if cand.empty:
            continue
        months = ((cand["DATSCAN_DATE"] - pd.Timestamp(s.session_date)).dt.days / 30.44).abs()
        i = months.idxmin()
        if months[i] > max_months:
            continue
        row = {"PATNO": s.patno, "IMAGEID": s.image_id, "months_to_datscan": round(float(months[i]), 1)}
        row.update({k: cand.loc[i, k] for k in keep})
        out.append(row)
    return pd.DataFrame(out).rename(columns={"EVENT_ID": "DATSCAN_EVENT_ID"})


def saa_labels(ppmi_dir, sessions):
    """Return PATNO, IMAGEID, SAA_EVENT_ID, SAA_Status, saa_positive (1/0)."""
    saa = _latest(ppmi_dir, "Biospecimen", "SAA_Biospecimen_Analysis_Results_*.csv")
    saa = saa[saa["SAA_Status"].isin(["Positive", "Negative"])]
    saa = saa[["PATNO", "CLINICAL_EVENT", "SAA_Status", "RUNDATE"]].drop_duplicates(["PATNO", "CLINICAL_EVENT"])
    order = {"BL": 0, "SC": 1}
    out = []
    for s in sessions[["patno", "image_id", "EVENT_ID"]].itertuples(index=False):
        cand = saa[saa["PATNO"] == s.patno]
        if cand.empty:
            continue
        pick = cand[cand["CLINICAL_EVENT"] == s.EVENT_ID]
        if pick.empty:
            cand = cand.assign(_o=cand["CLINICAL_EVENT"].map(order).fillna(9))
            pick = cand.sort_values(["_o", "CLINICAL_EVENT"]).head(1)
        r = pick.iloc[0]
        out.append({"PATNO": s.patno, "IMAGEID": s.image_id, "SAA_EVENT_ID": r["CLINICAL_EVENT"],
                    "SAA_Status": r["SAA_Status"], "saa_positive": float(r["SAA_Status"] == "Positive")})
    return pd.DataFrame(out)
