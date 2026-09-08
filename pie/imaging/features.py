"""
features.py — assemble the wide imaging-derived-phenotype (IDP) table from FastSurfer stats.

One row per processed session: identifiers (PATNO, EVENT_ID, IMAGEID, scan date), scanner
metadata from the dcm2niix sidecar, every FastSurfer regional volume (mm^3, prefix ``vol_``),
global measures (MaskVol, BrainSegVol, ...), and a few derived indices that matter for PD:
left/right asymmetry of subcortical structures and bilateral totals.
"""

import re
from pathlib import Path

import pandas as pd

from pie.imaging.fastsurfer import STATS_FILE, parse_stats

BILATERAL = ["Putamen", "Caudate", "Pallidum", "Thalamus", "Hippocampus", "Amygdala", "Accumbens-area",
             "Lateral-Ventricle", "Cerebellum-Cortex", "Cerebellum-White-Matter", "VentralDC"]
META_COLS = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "SoftwareVersions",
             "InstitutionName", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness"]


def _clean(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def build_idp_table(sessions, subjects_dir):
    """``sessions`` needs patno, image_id, session_date, EVENT_ID, protocol_phase and sidecar metadata."""
    rows = []
    for s in sessions.itertuples(index=False):
        stats = Path(subjects_dir) / s.image_id / STATS_FILE
        if not stats.exists():
            continue
        d = parse_stats(stats)
        row = {"PATNO": s.patno, "EVENT_ID": s.EVENT_ID, "IMAGEID": s.image_id, "SCAN_DATE": s.session_date,
               "protocol_phase": s.protocol_phase}
        row.update({c: getattr(s, c, None) for c in META_COLS})
        measures = {k: v for k, v in d.items() if k.endswith("Vol")}
        row.update(measures)
        for k, v in d.items():
            if k not in measures:
                row[f"vol_{_clean(k)}"] = v
        for st in BILATERAL:
            l, r = d.get(f"Left-{st}"), d.get(f"Right-{st}")
            if l is not None and r is not None:
                row[f"sum_{_clean(st)}"] = l + r
                row[f"asym_{_clean(st)}"] = (l - r) / (l + r) if (l + r) else None
        vent = [d.get(k, 0.0) for k in ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "3rd-Ventricle",
                                        "4th-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"]]
        row["sum_Ventricles"] = sum(vent)
        rows.append(row)
    return pd.DataFrame(rows)
