"""
index.py — enumerate DICOM series inside LONI/IDA zip downloads without extracting them,
and choose one T1-weighted series per imaging session.

LONI zips are laid out as  PPMI/<PATNO>/<SeriesDescription>/<yyyy-mm-dd_HH_MM_SS.0>/<ImageID>/*.dcm
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .cohorts import subject_key

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
    ``cache_csv`` (optional) stores/loads the result so the 20 GB zips are only read once; zips missing from the cache
    (a later download added to the same work dir) are indexed and appended.
    """
    zip_paths = list(map(str, zip_paths))
    cached = None
    if cache_csv and Path(cache_csv).exists():
        cached = pd.read_csv(cache_csv, dtype={"patno": int, "image_id": str})
        todo = [z for z in zip_paths if z not in set(cached["zip"].astype(str))]
        if not todo:
            return cached[cached["zip"].astype(str).isin(zip_paths)].reset_index(drop=True)
        zip_paths = todo
    rows = {}
    for zp in zip_paths:
        if Path(zp).is_dir():          # a BIDS dataset: its T1w images, already NIfTI
            rows.update({(zp, r["member_prefix"]): r for r in index_bids(zp)})
            continue
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                parts = info.filename.split("/")
                if len(parts) != 6 or info.is_dir():
                    continue
                project, subject, desc, session, image_id, _fname = parts
                patno = subject_key(project, subject)              # PPMI PATNO, ADNI RID range, else a hashed key
                if patno is None:                                  # phantom scans are not participants
                    continue
                key = (zp, subject, desc, session, image_id)
                r = rows.setdefault(key, {"zip": zp, "patno": patno, "series_desc": desc, "session": session,
                                          "image_id": image_id, "n_files": 0, "bytes": 0,
                                          "member_prefix": "/".join(parts[:5]) + "/", "cohort": project.upper(), "participant_id": subject})
                r["n_files"] += 1
                r["bytes"] += info.file_size
    df = pd.DataFrame(list(rows.values()))
    df["session_date"] = pd.to_datetime(df["session"].str[:10], errors="coerce")
    if cached is not None:
        df = pd.concat([cached, df], ignore_index=True)
    if cache_csv:
        df.to_csv(cache_csv, index=False)
    return df


def _visit_day(df):
    """LONI stamps every series with its own time, so a visit is a day; masked dates (9999-...) say nothing about the day."""
    return df["session"].where(df["session"].str.startswith("9999"), df["session"].str[:10])


def t1_candidates(index):
    """T1 candidate series in visits that have more than one: the ones ``select_t1_series`` ranks with headers."""
    df = index[~index["series_desc"].str.contains(_NOT_T1)]
    return df[df.groupby([df["patno"], _visit_day(df)])["image_id"].transform("size") > 1]


T1_HEADER_FIELDS = ("ImageType", "MRAcquisitionType", "InversionTime", "PixelSpacing")


def header_quality(headers):
    """image_id -> header score: +2 3D acquisition (-2 2D), +1 ORIGINAL (-1 derived, reformatted, MPR, projection),
    +1 an inversion time (MPRAGE / IR-SPGR), -1 in-plane spacing under 0.75 mm (a 512-matrix interpolated
    reconstruction, which the byte count would otherwise favour)."""
    h = headers.set_index("image_id")
    text = lambda c: h.get(c, pd.Series("", index=h.index)).fillna("").astype(str).str.upper()
    acq, kind = text("MRAcquisitionType"), text("ImageType")
    derived = kind.str.contains("DERIVED|REFORMAT|MPR|MIP|PROJECTION")
    ti = pd.to_numeric(h.get("InversionTime", pd.Series(np.nan, index=h.index)), errors="coerce")
    spacing = text("PixelSpacing").str.split(";").str[0].pipe(pd.to_numeric, errors="coerce")
    return (2 * acq.eq("3D") - 2 * acq.eq("2D") + (kind.str.contains("ORIGINAL") & ~derived).astype(int) - derived.astype(int)
            + (ti > 0).astype(int) - (spacing < 0.75).astype(int))


def bids_sessions(root, suffixes=("T1w",)):
    """``(sub, ses, path, sidecar, date_time)`` rows for the NIfTI images of ``suffixes`` in a BIDS dataset.
    The acquisition time comes from the sidecar's AcquisitionDateTime, else the session's scans.tsv ``acq_time``;
    without either, the session is undated."""
    import json

    root = Path(root)
    out = []
    for nii in sorted(root.glob("sub-*/**/*.nii.gz")):
        if "derivatives" in nii.relative_to(root).parts:
            continue
        name = nii.name[:-7]
        if name.rsplit("_", 1)[-1] not in suffixes:
            continue
        ent = dict(p.split("-", 1) for p in name.split("_")[:-1] if "-" in p)
        js = nii.with_name(name + ".json")
        meta = json.loads(js.read_text()) if js.exists() else {}
        when = meta.get("AcquisitionDateTime")
        scans = next(iter(nii.parents[1].glob("*_scans.tsv")), None) if nii.parent.name in ("anat", "dwi", "func", "fmap") else None
        if when is None and scans is not None:
            t = pd.read_csv(scans, sep="\t")
            hit = t[t["filename"].astype(str).str.endswith(str(nii.relative_to(nii.parents[1])))]
            when = hit["acq_time"].iloc[0] if len(hit) and "acq_time" in t else None
        out.append({"sub": ent.get("sub"), "ses": ent.get("ses", ""), "path": nii, "sidecar": js if js.exists() else None,
                    "date_time": pd.to_datetime(when, errors="coerce")})
    return out


def index_bids(root):
    """A BIDS dataset's T1w images as ``index_zips`` rows: ``zip`` = the dataset root, ``member_prefix`` = the image's
    path inside it, ``patno`` = ``cohorts.subject_key("BIDS:<dataset folder>", sub)``, ``session`` from the
    acquisition time (undated sessions get LONI's masked form, 9999-01-01, kept apart by session label) and a stable
    ``image_id`` (B + hash of the path), so the rest of the pipeline is unchanged."""
    import hashlib

    cohort = f"BIDS:{Path(root).resolve().name}"
    rows = []
    for r in bids_sessions(root):
        rel = str(r["path"].relative_to(root))
        dt = r["date_time"]
        session = dt.strftime("%Y-%m-%d_%H_%M_%S.0") if pd.notna(dt) else f"9999-01-01_ses-{r['ses'] or 'none'}"
        rows.append({"zip": str(root), "patno": subject_key(cohort, r["sub"]), "series_desc": r["path"].name[:-7].rsplit("_", 1)[-1],
                     "session": session, "image_id": "B" + hashlib.sha1(rel.encode()).hexdigest()[:10], "n_files": 1,
                     "bytes": r["path"].stat().st_size, "member_prefix": rel, "cohort": cohort, "participant_id": r["sub"]})
    return rows


def select_t1_series(index, headers=None):
    """Pick one T1w series per (patno, visit day).

    Rules: drop non-T1 descriptions; rank by (3D/MPRAGE-family description, not 2D/axial, not a repeat) plus, when
    ``headers`` (``probe_headers`` of ``t1_candidates`` with ``T1_HEADER_FIELDS``) are given, ``header_quality``; then
    the series with the most data (bytes, which also handles single-file multi-frame DICOM). Sessions dated 9999
    (LONI date-masked) are kept but flagged with ``date_masked``.
    """
    df = index.copy()
    df = df[~df["series_desc"].str.contains(_NOT_T1)]
    desc = df["series_desc"]
    df["score"] = (2 * desc.str.contains(_IS_3D).astype(int) - 2 * desc.str.contains(_IS_2D).astype(int)
                   - desc.str.contains(_REPEAT).astype(int))
    if headers is not None and len(headers):
        df["score"] += df["image_id"].map(header_quality(headers)).fillna(0).astype(int)
    df["date_masked"] = df["session"].str.startswith("9999")
    df = df.assign(_day=_visit_day(df)).sort_values(["patno", "_day", "score", "bytes", "session"], ascending=[True, True, False, False, True])
    chosen = df.drop_duplicates(["patno", "_day"], keep="first").reset_index(drop=True)
    return chosen.drop(columns=["score", "_day"])


def protocol_phase(session_dates):
    """1 = PPMI clinical protocol (001, scans before 2020), 2 = PPMI 2.0 (002, 2020 on); NaN for masked dates."""
    d = pd.to_datetime(session_dates, errors="coerce")
    return pd.Series(np.where(d.dt.year >= 2020, 2.0, 1.0), index=d.index).where(d.notna() & (d.dt.year < 2100))


def probe_headers(sessions, fields=("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName",
                                    "SoftwareVersions", "SliceThickness", "PixelSpacing", "Rows", "Columns")):
    """Read one DICOM header per session straight from the zip (no extraction).

    Cheap way to get scanner vendor / field strength for every session before conversion,
    e.g. for ComBat batches. Returns a DataFrame keyed by image_id.
    """
    import pydicom

    out = []
    for zp, group in sessions.groupby("zip"):
        if Path(zp).is_dir():          # BIDS input: the header fields live in the sidecars, not read here
            continue
        with zipfile.ZipFile(zp) as z:
            first = {}  # series prefix -> first file (one pass over the listing)
            for n in z.namelist():
                if not n.endswith("/"):
                    first.setdefault(n.rsplit("/", 1)[0] + "/", n)
            for s in group.itertuples(index=False):
                member = first.get(s.member_prefix)
                row = {"image_id": s.image_id}
                if member:
                    try:
                        with z.open(member) as fh:
                            d = pydicom.dcmread(fh, stop_before_pixels=True, force=True)
                    except Exception:            # an unreadable header leaves the fields empty; the series stays rankable by name
                        d = {}
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
        dict(zip="z", patno=2, series_desc="3D_T1-weighted", session="2000-01-01_09_00_00.0", image_id="I4", n_files=1, bytes=25e6, member_prefix="p/"),
        dict(zip="z", patno=2, series_desc="Transverse", session="2000-01-01_09_00_00.0", image_id="I5", n_files=1, bytes=60e6, member_prefix="p/"),
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
            if root.tag.rsplit("}", 1)[-1] != "idaxs":          # newer downloads namespace the root: {http://ida.loni.usc.edu}idaxs
                continue
            get = lambda tag: (root.find(f".//{tag}").text if root.find(f".//{tag}") is not None else None)
            if not (get("subjectIdentifier") or "").isdigit():   # LONI phantom scans are not participants
                continue
            prot = {e.get("term"): e.text for e in root.iter("protocol")}
            rows.append({"image_id": "I" + get("imageUID"), "patno": int(get("subjectIdentifier")), "ida_group": get("researchGroup"),
                         "ida_visit": get("visitIdentifier"), "ida_age": get("subjectAge"), "ida_date": get("dateAcquired"),
                         "ida_desc": get("description"), "ida_manufacturer": prot.get("Manufacturer"), "ida_model": prot.get("Mfg Model"),
                         "ida_field": prot.get("Field Strength"), "ida_slice": prot.get("Slice Thickness"), "ida_plane": prot.get("Acquisition Plane"),
                         "ida_acq_type": prot.get("Acquisition Type"), "ida_weighting": prot.get("Weighting")})
    return pd.DataFrame(rows).drop_duplicates("image_id")




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


# PPMI's visit schedule: months after baseline -> EVENT_ID (quarterly to 12 months, 6-monthly to 60, then yearly)
MONTH_TO_EVENT = {3: "V01", 6: "V02", 9: "V03", 12: "V04", 18: "V05", 24: "V06", 30: "V07", 36: "V08", 42: "V09", 48: "V10",
                  54: "V11", 60: "V12", **{72 + 12 * i: f"V{13 + i:02d}" for i in range(10)}}
_EVENT = re.compile(r"(SC|BL|V\d{2}|ST|PW|U\d{2}|R\d{2})")


def visit_to_event(label):
    """A LONI collection-CSV visit (already an EVENT_ID: BL, V04, U01, ...) or an IDA visit name ("Baseline",
    "Screening", "Month 24", "Unscheduled Visit 01", "Symptomatic Therapy", "Premature Withdrawal") as PPMI's
    EVENT_ID; None when it cannot be read, so the date-based link stands."""
    t = str(label).strip()
    if _EVENT.fullmatch(t.upper()):
        return t.upper()
    low = t.lower()
    named = {"baseline": "BL", "screening": "SC", "symptomatic therapy": "ST", "premature withdrawal": "PW"}
    if low in named:
        return named[low]
    m = re.fullmatch(r"month (\d+)", low) or re.fullmatch(r"unscheduled visit (\d+)", low)
    if m and low.startswith("month"):
        return MONTH_TO_EVENT.get(int(m.group(1)))
    return f"U{int(m.group(1)):02d}" if m else None
