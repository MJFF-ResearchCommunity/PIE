"""
link.py — attach a PPMI EVENT_ID to each MRI session.

PPMI's `Magnetic_Resonance_Imaging__MRI_` table records the visit (EVENT_ID) and the
month/year (INFODT) of every MRI. The DICOM folder gives the exact acquisition date, so a
session is linked to the visit whose INFODT is the same month, else the nearest month
within ``max_months``. Unlinked sessions get EVENT_ID "UNK" (still usable, joined by date).
"""

from pathlib import Path

import pandas as pd


def _load_mri_table(ppmi_dir):
    from .viewer.catalog import latest_table

    path = latest_table(Path(ppmi_dir, "Imaging"), "Magnetic_Resonance_Imaging__MRI_")
    if path is None:
        raise FileNotFoundError("Magnetic_Resonance_Imaging__MRI_ table not found under PPMI/Imaging")
    df = pd.read_csv(path, usecols=["PATNO", "EVENT_ID", "INFODT", "MRICMPLT"])
    df = df[df["MRICMPLT"] == 1].dropna(subset=["INFODT"])
    df["visit_month"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce").dt.to_period("M")
    return df.dropna(subset=["visit_month"])


def link_sessions_to_events(sessions, ppmi_dir, max_months=3):
    """Add EVENT_ID (and months_off) to a sessions frame with columns patno, session_date. Other cohorts' sessions get
    UNK (``link_adni`` labels ADNI's); without PPMI sessions the PPMI table is not needed."""
    from .cohorts import cohort_of

    out = sessions.copy()
    if not any(cohort_of(p) == "PPMI" for p in out["patno"]):
        return out.assign(EVENT_ID="UNK", months_off=pd.NA)
    mri = _load_mri_table(ppmi_dir)
    out["scan_month"] = pd.to_datetime(out["session_date"]).dt.to_period("M")
    event, off = [], []
    for patno, month in zip(out["patno"], out["scan_month"]):
        cand = mri[mri["PATNO"] == patno]
        if cand.empty or pd.isna(month):
            event.append("UNK"); off.append(pd.NA); continue
        diff = (cand["visit_month"] - month).apply(lambda d: abs(d.n))
        i = diff.idxmin()
        if diff[i] <= max_months:
            event.append(cand.loc[i, "EVENT_ID"]); off.append(int(diff[i]))
        else:
            event.append("UNK"); off.append(int(diff[i]))
    out["EVENT_ID"], out["months_off"] = event, off
    return out.drop(columns=["scan_month"])


def link_adni(sessions, adnimerge_csv, max_days=90):
    """ADNI sessions (``cohorts`` ADNI keys) get ADNIMERGE's visit code (VISCODE: bl, m06, ...) as EVENT_ID, from the
    exam date nearest the scan within ``max_days``; ``months_off`` is that gap in months. Other rows are unchanged."""
    from .cohorts import OFFSET, cohort_of

    adni = pd.read_csv(adnimerge_csv, usecols=["RID", "VISCODE", "EXAMDATE"], low_memory=False)
    adni["EXAMDATE"] = pd.to_datetime(adni["EXAMDATE"], errors="coerce")
    out = sessions.copy()
    for i, (patno, date) in enumerate(zip(out["patno"], pd.to_datetime(out["session_date"], errors="coerce"))):
        if cohort_of(patno) != "ADNI" or pd.isna(date):
            continue
        cand = adni[adni["RID"] == int(patno) - OFFSET["ADNI"]].dropna(subset=["EXAMDATE"])
        if cand.empty:
            continue
        gap = (cand["EXAMDATE"] - date).abs().dt.days
        j = gap.idxmin()
        if gap[j] <= max_days:
            out.iloc[i, out.columns.get_loc("EVENT_ID")] = cand.loc[j, "VISCODE"]
            out.iloc[i, out.columns.get_loc("months_off")] = round(gap[j] / 30.44)
    return out
