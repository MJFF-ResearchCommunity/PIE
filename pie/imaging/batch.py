"""
Shared plumbing for the per-modality batch pipelines (dwi, nm, flair): LONI zip series index, dcm2niix
conversion, FastSurfer lookup, session choice, and a resumable parallel runner that appends one CSV row per
subject, records failures, retries them on request and writes a PID file so a run can be controlled without
pattern-matching process lists.
"""

import csv
import os
import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .convert import DCM2NIIX


def index_series(zips):
    """One row per DICOM series in LONI zips: zip, prefix (PPMI/<patno>/<desc>/<date>/<image_id>/), patno, desc,
    date, image_id, n_files."""
    rows = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            counts = {}
            for n in z.namelist():
                if n.endswith(".dcm"):
                    key = "/".join(n.split("/")[:5])
                    counts[key] = counts.get(key, 0) + 1
        for key, c in counts.items():
            _, patno, desc, date, image_id = key.split("/")
            rows.append({"zip": str(zp), "prefix": key + "/", "patno": int(patno), "desc": desc, "date": date[:10], "image_id": image_id, "n_files": c})
    return pd.DataFrame(rows)


def load_index(index_file, zips, flag_fn):
    """Cached series index with a boolean selection column added by ``flag_fn(idx) -> Series``."""
    index_file = Path(index_file)
    idx = pd.read_csv(index_file, dtype={"image_id": str}) if index_file.exists() else index_series(zips)
    idx["selected"] = flag_fn(idx)
    idx.to_csv(index_file, index=False)
    return idx


def convert_series(zip_path, prefix, out_dir):
    """Extract one series from the zip and run dcm2niix into out_dir. Returns the NIfTI paths written (sorted)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = prefix.rstrip("/").split("/")[-1]
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp, zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.startswith(prefix) and n.endswith(".dcm"):
                with open(Path(tmp) / Path(n).name, "wb") as f:
                    f.write(z.read(n))
        subprocess.run([DCM2NIIX, "-z", "y", "-b", "y", "-f", f"{tag}_%s", "-o", str(out_dir), tmp], capture_output=True, text=True)
    return sorted(str(p) for p in out_dir.glob(f"{tag}_*.nii.gz"))


def fastsurfer_by_patno(sessions_csv, fastsurfer_dir, require="stats/aseg+DKT.stats"):
    """patno -> FastSurfer subject directory (earliest session with finished output)."""
    sess = pd.read_csv(sessions_csv, dtype={"image_id": str})
    root = Path(fastsurfer_dir)
    done = {p.parts[-3] for p in root.glob(f"*/{require}")}
    return {int(r.patno): str(root / r.image_id) for r in sess.sort_values("session_date").itertuples() if r.image_id in done}


def session_rows(group):
    """Rows of the acquisition date with the most files (one session per subject)."""
    date = group.groupby("date")["n_files"].sum().idxmax()
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
    return ap


def done_subjects(out_csv, retry_errors=False):
    out_csv = Path(out_csv)
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return set()
    prev = pd.read_csv(out_csv)
    if retry_errors and prev["error"].fillna("").ne("").any():
        prev = prev[prev["error"].fillna("") == ""]
        prev.to_csv(out_csv, index=False)
    return set(prev["patno"])


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
    with ProcessPoolExecutor(max_workers=workers) as ex, open(out_csv, "a", newline="") as fh:
        writer = None
        for i, fut in enumerate(as_completed([ex.submit(job_fn, j) for j in jobs]), start=1):
            row = fut.result()
            row.setdefault("error", "")
            if writer is None:
                if out_csv.stat().st_size == 0:
                    writer = csv.DictWriter(fh, fieldnames=sorted(row))
                    writer.writeheader()
                else:
                    writer = csv.DictWriter(fh, fieldnames=pd.read_csv(out_csv, nrows=0).columns.tolist(), extrasaction="ignore")
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            fh.flush()
            if i % log_every == 0 or row.get("error"):
                print(f"{i}/{len(jobs)} {row['patno']} {'ERROR ' + str(row['error']) if row.get('error') else 'ok'}", flush=True)
    print(f"done -> {out_csv}", flush=True)
