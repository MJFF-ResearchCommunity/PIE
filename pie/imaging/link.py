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
    files = sorted(Path(ppmi_dir, "Imaging").glob("Magnetic_Resonance_Imaging__MRI__*.csv"))
    if not files:
        raise FileNotFoundError("Magnetic_Resonance_Imaging__MRI_ table not found under PPMI/Imaging")
    df = pd.read_csv(files[-1], usecols=["PATNO", "EVENT_ID", "INFODT", "MRICMPLT"])
    df = df[df["MRICMPLT"] == 1].dropna(subset=["INFODT"])
    df["visit_month"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce").dt.to_period("M")
    return df.dropna(subset=["visit_month"])


def link_sessions_to_events(sessions, ppmi_dir, max_months=3):
    """Add EVENT_ID (and months_off) to a sessions frame with columns patno, session_date."""
    mri = _load_mri_table(ppmi_dir)
    out = sessions.copy()
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
