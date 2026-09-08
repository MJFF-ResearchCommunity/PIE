"""
index.py — enumerate DICOM series inside LONI/IDA zip downloads without extracting them,
and choose one T1-weighted series per imaging session.

LONI zips are laid out as  PPMI/<PATNO>/<SeriesDescription>/<yyyy-mm-dd_HH_MM_SS.0>/<ImageID>/*.dcm
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

# Series that are not a usable 3D T1w volume (localisers, calibration, sequence-specific extras).
_NOT_T1 = re.compile(r"loc|scout|survey|cal_head|calib|coronal|midline|phase|field|mt_|_mt|t2|flair|dti|dwi|fmri|rs_|rest|gre|swi|asl|pd_", re.I)
_REPEAT = re.compile(r"repeat|rpt|_2$|artifact|_ia$", re.I)
# Prefer explicitly 3D / MPRAGE-family descriptions; penalise 2D/axial/coronal/transverse ones (some sites label
# a 2-frame axial T1 as the only "T1", which FastSurfer cannot use).
_IS_3D = re.compile(r"3D|MPRAGE|MP-RAGE|MP_RAGE|SPGR|BRAVO|TFE|T1W_3D|ADNI", re.I)
_IS_2D = re.compile(r"^(?:AX|AXIAL|TRA|COR|Transverse)|_2D|2D_", re.I)


def index_zips(zip_paths, cache_csv=None):
    """Return one row per DICOM series found in the given zips.

    Columns: zip, patno, series_desc, session, image_id, n_files, bytes, member_prefix.
    ``cache_csv`` (optional) stores/loads the result so the 20 GB zips are only read once.
    """
    if cache_csv and Path(cache_csv).exists():
        return pd.read_csv(cache_csv, dtype={"patno": int, "image_id": str})
    rows = {}
    for zp in map(str, zip_paths):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                parts = info.filename.split("/")
                if len(parts) != 6 or info.is_dir():
                    continue
                _, patno, desc, session, image_id, _fname = parts
                key = (zp, patno, desc, session, image_id)
                r = rows.setdefault(key, {"zip": zp, "patno": int(patno), "series_desc": desc, "session": session,
                                          "image_id": image_id, "n_files": 0, "bytes": 0,
                                          "member_prefix": "/".join(parts[:5]) + "/"})
                r["n_files"] += 1
                r["bytes"] += info.file_size
    df = pd.DataFrame(list(rows.values()))
    df["session_date"] = pd.to_datetime(df["session"].str[:10], errors="coerce")
    if cache_csv:
        df.to_csv(cache_csv, index=False)
    return df


def select_t1_series(index):
    """Pick one T1w series per (patno, session date).

    Rules: drop non-T1 descriptions; rank by (3D/MPRAGE-family description, not 2D/axial,
    not a repeat), then the series with the most data (bytes, which also handles single-file
    multi-frame DICOM). Sessions dated 9999 (LONI date-masked) are kept but flagged with ``date_masked``.
    """
    df = index.copy()
    df = df[~df["series_desc"].str.contains(_NOT_T1)]
    desc = df["series_desc"]
    df["score"] = (2 * desc.str.contains(_IS_3D).astype(int) - 2 * desc.str.contains(_IS_2D).astype(int)
                   - desc.str.contains(_REPEAT).astype(int))
    df["date_masked"] = df["session"].str.startswith("9999")
    df = df.sort_values(["patno", "session", "score", "bytes"], ascending=[True, True, False, False])
    chosen = df.drop_duplicates(["patno", "session"], keep="first").reset_index(drop=True)
    return chosen.drop(columns=["score"])


def probe_headers(sessions, fields=("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName",
                                    "SoftwareVersions", "SliceThickness", "PixelSpacing", "Rows", "Columns")):
    """Read one DICOM header per session straight from the zip (no extraction).

    Cheap way to get scanner vendor / field strength for every session before conversion,
    e.g. for ComBat batches. Returns a DataFrame keyed by image_id.
    """
    import pydicom

    out = []
    for zp, group in sessions.groupby("zip"):
        with zipfile.ZipFile(zp) as z:
            first = {}  # series prefix -> first file (one pass over the listing)
            for n in z.namelist():
                if not n.endswith("/"):
                    first.setdefault(n.rsplit("/", 1)[0] + "/", n)
            for s in group.itertuples(index=False):
                member = first.get(s.member_prefix)
                row = {"image_id": s.image_id}
                if member:
                    with z.open(member) as fh:
                        d = pydicom.dcmread(fh, stop_before_pixels=True, force=True)
                    for f in fields:
                        v = d.get(f)
                        row[f] = ";".join(map(str, v)) if isinstance(v, (list, pydicom.multival.MultiValue)) else v
                out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":  # self-check on a synthetic listing
    demo = pd.DataFrame([
        dict(zip="z", patno=1, series_desc="MPRAGE", session="2011-01-01_10_00_00.0", image_id="I1", n_files=170, bytes=100, member_prefix="p/"),
        dict(zip="z", patno=1, series_desc="MPRAGE_Repeat", session="2011-01-01_10_00_00.0", image_id="I2", n_files=176, bytes=120, member_prefix="p/"),
        dict(zip="z", patno=1, series_desc="Coronal", session="2011-01-01_10_00_00.0", image_id="I3", n_files=1, bytes=1, member_prefix="p/"),
        dict(zip="z", patno=2, series_desc="3D_T1-weighted", session="2021-03-23_09_05_05.0", image_id="I4", n_files=1, bytes=25e6, member_prefix="p/"),
        dict(zip="z", patno=2, series_desc="Transverse", session="2021-03-23_09_05_05.0", image_id="I5", n_files=1, bytes=60e6, member_prefix="p/"),
    ])
    demo["session_date"] = pd.to_datetime(demo["session"].str[:10])
    sel = select_t1_series(demo)
    assert sel["image_id"].tolist() == ["I1", "I4"], sel
    print("index self-check OK")


def read_ida_metadata(paths):
    """Parse LONI/IDA 'Advanced Download' metadata (a zip or directory of idaxs XML files).

    Only the full ``idaxs`` records carry information (visit, research group, sex, age, acquisition
    date and the protocol terms: manufacturer, model, field strength, slice thickness, plane, type);
    the tiny ``<metadata>`` stubs are skipped. Returns one row per image_id.
    """
    import xml.etree.ElementTree as ET

    rows = []
    for p in map(Path, [paths] if isinstance(paths, (str, Path)) else paths):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                docs = [z.read(n) for n in z.namelist() if n.endswith(".xml")]
        else:
            docs = [f.read_bytes() for f in p.rglob("*.xml")]
        for doc in docs:
            root = ET.fromstring(doc)
            if root.tag != "idaxs":
                continue
            get = lambda tag: (root.find(f".//{tag}").text if root.find(f".//{tag}") is not None else None)
            prot = {e.get("term"): e.text for e in root.iter("protocol")}
            rows.append({"image_id": "I" + get("imageUID"), "patno": int(get("subjectIdentifier")), "ida_group": get("researchGroup"),
                         "ida_visit": get("visitIdentifier"), "ida_age": get("subjectAge"), "ida_date": get("dateAcquired"),
                         "ida_desc": get("description"), "ida_manufacturer": prot.get("Manufacturer"), "ida_model": prot.get("Mfg Model"),
                         "ida_field": prot.get("Field Strength"), "ida_slice": prot.get("Slice Thickness"), "ida_plane": prot.get("Acquisition Plane"),
                         "ida_acq_type": prot.get("Acquisition Type"), "ida_weighting": prot.get("Weighting")})
    return pd.DataFrame(rows).drop_duplicates("image_id")


IDA_VISIT_TO_EVENT = {"Baseline": "BL", "Screening": "SC"}


def read_loni_collection_csv(paths):
    """Parse the collection CSV that every LONI/IDA download produces (one row per series: Image Data ID,
    Subject, Group, Sex, Age, Visit, Description, Acq Date, ...). Returns one row per image_id with
    ``loni_*`` columns; covers every downloaded series, unlike the IDA metadata zip."""
    frames = []
    for p in map(Path, [paths] if isinstance(paths, (str, Path)) else paths):
        df = pd.read_csv(p)
        frames.append(pd.DataFrame({"image_id": "I" + df["Image Data ID"].astype(str).str.lstrip("I"),
                                    "loni_visit": df.get("Visit"), "loni_group": df.get("Group"), "loni_sex": df.get("Sex"),
                                    "loni_age": pd.to_numeric(df.get("Age"), errors="coerce"), "loni_desc": df.get("Description"),
                                    "loni_acq_date": pd.to_datetime(df.get("Acq Date"), errors="coerce")}))
    return pd.concat(frames, ignore_index=True).drop_duplicates("image_id")


LONI_VISIT_TO_EVENT = {"BL": "BL", "SC": "SC", "Baseline": "BL", "Screening": "SC"}
