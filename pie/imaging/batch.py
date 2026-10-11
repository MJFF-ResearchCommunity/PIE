"""
Shared plumbing for the per-modality batch pipelines (dwi, nm, flair): LONI zip series index, dcm2niix
conversion, FastSurfer lookup, session choice, and a resumable parallel runner that appends one CSV row per
subject, records failures, retries them on request and writes a PID file so a run can be controlled without
pattern-matching process lists.
"""

import csv
import os
import re
import shutil
from contextlib import nullcontext
import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .archives import ZipArchiveCache
from .cohorts import subject_key
from .convert import DCM2NIIX


def index_series(zips):
    """One row per DICOM series in LONI zips: zip, prefix (PPMI/<patno>/<desc>/<date>/<image_id>/), patno, desc,
    date, image_id, n_files."""
    rows = []
    for zp in zips:
        if Path(zp).is_dir():          # a BIDS dataset
            rows += bids_series(zp)
            continue
        with zipfile.ZipFile(zp) as z:
            counts = {}
            for n in z.namelist():
                if n.endswith(".dcm"):
                    key = "/".join(n.split("/")[:5])
                    counts[key] = counts.get(key, 0) + 1
        for key, c in counts.items():
            project, subject, desc, date, image_id = key.split("/")
            patno = subject_key(project, subject)   # PPMI PATNO, ADNI RID range, else a hashed key (cohorts.py)
            if patno is None:                # LONI phantom scans ("00000JAN00") are not participants
                continue
            rows.append({"zip": str(zp), "prefix": key + "/", "patno": patno, "desc": desc, "date": date[:10], "image_id": image_id, "n_files": c,
                         "cohort": project.upper(), "participant_id": subject})
    return pd.DataFrame(rows)


BIDS_SUFFIXES = ("dwi", "FLAIR", "T1w", "T2w", "PDT2", "MTw", "NM", "epi", "bold")


def bids_series(root):
    """A BIDS dataset's images as ``index_series`` rows: ``prefix`` = the image's path inside the dataset, ``date`` from
    its acquisition time (9999-01-01 when undated), and a ``desc`` the modality selectors read: "BIDS_dwi", "BIDS_FLAIR"
    (+ "_3D" at <= 1.5 mm slices), "NM - BIDS ..." for neuromelanin (``acq-NM``/``neuromelanin`` in the name, or an
    ``_NM``/``_MTw`` image), else "BIDS_<suffix>"."""
    import hashlib

    import nibabel as nib

    from .index import bids_sessions

    cohort, rows = f"BIDS:{Path(root).resolve().name}", []
    for r in bids_sessions(root, suffixes=BIDS_SUFFIXES):
        rel, stem = str(r["path"].relative_to(root)), r["path"].name[:-7]
        suffix = stem.rsplit("_", 1)[-1]
        if suffix in ("NM", "MTw") or re.search(r"acq-nm|neuromelanin", stem, re.I):
            desc = f"NM - BIDS {stem}"
        elif suffix == "FLAIR":
            desc = "BIDS_FLAIR" + ("_3D" if max(nib.load(r["path"]).header.get_zooms()[:3]) <= 1.5 else "")
        else:
            desc = f"BIDS_{suffix}"
        dt = r["date_time"]
        rows.append({"zip": str(root), "prefix": rel, "patno": subject_key(cohort, r["sub"]), "desc": desc,
                     "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "9999-01-01", "image_id": "B" + hashlib.sha1(rel.encode()).hexdigest()[:10],
                     "n_files": 1, "cohort": cohort, "participant_id": r["sub"]})
    return rows


def load_index(index_file, zips, flag_fn):
    """Cached series index with a boolean selection column added by ``flag_fn(idx) -> Series``."""
    index_file = Path(index_file)
    idx = pd.read_csv(index_file, dtype={"image_id": str}) if index_file.exists() else index_series(zips)
    idx["selected"] = flag_fn(idx)
    idx.to_csv(index_file, index=False)
    return idx


def convert_series(zip_path, prefix, out_dir, archive_cache=None):
    """Extract one series from the zip and run dcm2niix into out_dir. Returns the NIfTI paths written (sorted).
    ``archive_cache`` (``archives.ZipArchiveCache``) keeps the zip's index between the series of one subject: a LONI
    zip with millions of members takes ~13 s and ~3 GB to index, and PPMI stores each NM repeat as its own series."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(zip_path).is_dir():        # BIDS (bids_series): copy the image with its sidecar and gradients
        src = Path(zip_path) / prefix
        stem = src.name[:-7]
        for ext in (".nii.gz", ".json", ".bval", ".bvec"):
            if (src.parent / (stem + ext)).exists():
                shutil.copyfile(src.parent / (stem + ext), out_dir / (stem + ext))
        return [str(out_dir / src.name)]
    tag = prefix.rstrip("/").split("/")[-1]
    log = out_dir / f"{tag}_dcm2niix.log"
    with (tempfile.TemporaryDirectory(dir=out_dir) as tmp, (ZipArchiveCache() if archive_cache is None else nullcontext(archive_cache)) as cache,
          cache.open(zip_path) as z):
        src, dst = Path(tmp) / "dcm", Path(tmp) / "nii"
        src.mkdir()
        dst.mkdir()
        for n in z.namelist():
            if n.startswith(prefix) and n.endswith(".dcm"):
                with open(src / Path(n).name, "wb") as f:
                    f.write(z.read(n))
        # a fresh output dir: a retry must not pick up (or suffix itself next to) an earlier attempt's files
        r = subprocess.run([DCM2NIIX, "-z", "y", "-b", "y", "-f", f"{tag}_%s", "-o", str(dst), str(src)], capture_output=True, text=True)
        made = sorted(dst.iterdir())
        if r.returncode != 0 or not any(f.name.endswith(".nii.gz") for f in made):   # keep the reason (e.g. stripped DWI tags)
            log.write_text(f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")
        for f in made:
            os.replace(f, out_dir / f.name)
    return sorted(str(out_dir / f.name) for f in made if f.name.endswith(".nii.gz"))


def fastsurfer_by_patno(sessions_csv, fastsurfer_dir, require="stats/aseg+DKT.stats"):
    """patno -> FastSurfer subject directory (earliest session with finished output)."""
    sess = pd.read_csv(sessions_csv, dtype={"image_id": str})
    root = Path(fastsurfer_dir)
    done = {p.parts[-3] for p in root.glob(f"*/{require}")}
    # A dict comprehension over ascending dates overwrites the first session with the last.
    sess = sess[sess["image_id"].isin(done)].sort_values(["session_date", "image_id"], kind="stable")
    sess = sess.drop_duplicates("patno", keep="first")
    return {int(r.patno): str(root / r.image_id) for r in sess.itertuples()}


def t1_session_dates(sessions_csv, fs):
    """patno -> session date (YYYY-MM-DD) of the FastSurfer T1 ``fastsurfer_by_patno`` chose."""
    sess = pd.read_csv(sessions_csv, dtype={"image_id": str})
    by_id = dict(zip(sess["image_id"], sess["session_date"].astype(str).str[:10]))
    return {p: by_id.get(Path(d).name) for p, d in fs.items()}


def session_rows(group, t1_date=None):
    """Rows of one acquisition date per subject: the date nearest the T1 session whose anatomy the modality is
    measured on (a later visit's scan on the baseline aseg would carry atrophy into the masks), then the most files.
    Without ``t1_date``, the date with the most files."""
    files = group.groupby("date")["n_files"].sum()
    gap = pd.Series(0.0, index=files.index)
    if t1_date:
        gap = (pd.to_datetime(files.index, errors="coerce") - pd.Timestamp(t1_date)).to_series(index=files.index).abs().dt.days
        gap = gap.fillna(float("inf"))     # LONI-masked dates: never preferred over a dated session
    date = pd.DataFrame({"gap": gap, "files": -files}).sort_values(["gap", "files"]).index[0]
    return group[group["date"] == date].to_dict("records")


def add_common_args(ap):
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--sessions", required=True, help="PIE sessions.csv (patno -> FastSurfer image_id)")
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--patnos", help="text file of PATNOs to process")
    ap.add_argument("--retry-errors", action="store_true", help="re-process subjects whose previous row recorded an error")
    ap.add_argument("--keep-nifti", action="store_true", help="keep per-subject NIfTI outputs")
    ap.add_argument("--pid-file", help="write the runner's PID here")
    ap.add_argument("--per-visit", action="store_true", help="longitudinal: one session per finished T1 session (each measured on "
                    "its own visit's T1), not one per participant; use a separate --work-dir")
    ap.add_argument("--visit-window-days", type=int, default=180, help="--per-visit: largest gap between a session and its T1")
    return ap


def done_subjects(out_csv, retry_errors=False, key="patno"):
    """Values of ``key`` already in ``out_csv`` (``image_id`` for runners with one row per scan, e.g. pet; a list of
    columns gives tuples, e.g. ["patno", "fs_image_id"] for per-visit runs)."""
    out_csv = Path(out_csv)
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return set()
    prev = pd.read_csv(out_csv, dtype={"fs_image_id": str})
    if retry_errors and prev["error"].fillna("").ne("").any():
        prev = prev[prev["error"].fillna("") == ""]
        prev.to_csv(out_csv, index=False)
    if isinstance(key, list):
        return set(prev.reindex(columns=key).itertuples(index=False, name=None))
    return set(prev[key])


def visit_jobs(idx, sessions_csv, fastsurfer_dir, window_days=180, require="stats/aseg+DKT.stats"):
    """Per-visit jobs (longitudinal runs): (patno, rows of one modality session, FastSurfer dir of the T1 session it is
    measured on). Each dated modality session goes to the participant's finished T1 session nearest in date, within
    ``window_days``; a T1 session keeps only its nearest modality session. LONI-masked dates never match."""
    sess = pd.read_csv(sessions_csv, dtype={"image_id": str})
    root = Path(fastsurfer_dir)
    finished = {p.parts[-3] for p in root.glob(f"*/{require}")}
    t1 = sess[sess["image_id"].isin(finished)].assign(_d=lambda x: pd.to_datetime(x["session_date"], errors="coerce"))
    t1 = t1[t1["_d"].dt.year < 2100]
    pairs = []
    for patno, g in idx.groupby("patno"):
        tp = t1[t1["patno"] == patno]
        for date, rows in g.groupby("date") if len(tp) else ():
            d = pd.to_datetime(date, errors="coerce")
            if pd.isna(d) or d.year >= 2100:
                continue
            gap = (tp["_d"] - d).abs().dt.days
            if gap.min() <= window_days:
                pairs.append((int(gap.min()), int(patno), tp.loc[gap.idxmin(), "image_id"], rows.to_dict("records")))
    jobs, used = [], set()
    for _, patno, image_id, rows in sorted(pairs, key=lambda x: (x[0], x[1])):
        if (patno, image_id) not in used:
            used.add((patno, image_id))
            jobs.append((patno, rows, str(root / image_id)))
    return sorted(jobs, key=lambda j: (j[0], j[2]))


def modality_jobs(idx, a, out_csv):
    """(patno, series rows, FastSurfer dir) still to run for a modality CLI: per participant, the session nearest its
    earliest finished T1 (``session_rows``); with ``--per-visit``, one per finished T1 session (``visit_jobs``), resumed
    by (patno, fs_image_id)."""
    if getattr(a, "per_visit", False):
        done = done_subjects(out_csv, a.retry_errors, key=["patno", "fs_image_id"])
        return [j for j in visit_jobs(idx, a.sessions, a.fastsurfer_dir, a.visit_window_days) if (j[0], Path(j[2]).name) not in done]
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    t1d = t1_session_dates(a.sessions, fs)
    done = done_subjects(out_csv, a.retry_errors)
    return [(int(p), session_rows(g, t1d.get(int(p))), fs[int(p)]) for p, g in idx.groupby("patno") if p not in done and int(p) in fs]


def filter_jobs(jobs, patnos_file=None, limit=None):
    if patnos_file:
        keep = {int(x) for x in Path(patnos_file).read_text().split()}
        jobs = [j for j in jobs if j[0] in keep]
    return jobs[:limit] if limit else jobs


def run_batch(jobs, job_fn, out_csv, workers=4, log_every=10, pid_file=None):
    """Run ``job_fn(job) -> row dict`` (must never raise; put failures in row['error']) over ``jobs`` in parallel,
    appending rows to ``out_csv`` as they complete (resumable: rows are flushed one by one)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))
    print(f"{len(jobs)} subjects to process", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex, open(out_csv, "a+", newline="") as fh:
        writer, pending = None, []          # error rows seen before the first full row wait, so the header holds every column

        def start(fieldnames):
            nonlocal writer
            existing = pd.read_csv(out_csv, dtype=str).fillna("") if out_csv.stat().st_size else None
            if existing is not None and set(fieldnames) - set(existing.columns):     # a header written from error rows only
                fieldnames = sorted(set(fieldnames) | set(existing.columns))
                fh.seek(0)
                fh.truncate()
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing.to_dict("records"))
            elif existing is not None:
                writer = csv.DictWriter(fh, fieldnames=existing.columns.tolist(), extrasaction="ignore")
            else:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()

        def emit(row):
            nonlocal writer
            new = set(row) - set(writer.fieldnames)
            if new:                          # a later row brings columns the header lacks (e.g. optional stages): rewrite with the union
                fh.flush()
                existing = pd.read_csv(out_csv, dtype=str).fillna("") if out_csv.stat().st_size else pd.DataFrame()
                fields = list(writer.fieldnames) + sorted(new)
                fh.seek(0)
                fh.truncate()
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing.to_dict("records"))
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            fh.flush()

        for i, fut in enumerate(as_completed([ex.submit(job_fn, j) for j in jobs]), start=1):
            row = fut.result()
            row.setdefault("error", "")
            if writer is None and row["error"]:
                pending.append(row)
            else:
                if writer is None:
                    start(sorted(set(row) | set().union(*(set(r) for r in pending))))
                    for r in pending:
                        emit(r)
                    pending = []
                emit(row)
            if i % log_every == 0 or row.get("error"):
                print(f"{i}/{len(jobs)} {row['patno']} {'ERROR ' + str(row['error']) if row.get('error') else 'ok'}", flush=True)
        if pending:                          # every job failed
            start(sorted(set().union(*(set(r) for r in pending))))
            for r in pending:
                emit(r)
    print(f"done -> {out_csv}", flush=True)
