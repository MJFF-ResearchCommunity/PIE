#!/usr/bin/env python
"""
run.py — resumable CLI: LONI zips -> NIfTI -> FastSurfer -> IDP table.

    venv_imaging/bin/python -m pie.imaging.run \
        --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
        --ppmi-dir PPMI --work-dir Imaging/derived --workers 4 --threads 4 \
        [--priority patnos.txt] [--limit N] [--features-only] [--device cpu] [--etiv]

Work dir layout: index.csv (all series), sessions.csv (chosen T1 per session + EVENT_ID +
scanner metadata), nifti/<PATNO>/<IMAGEID>_T1w.nii.gz, fastsurfer/<IMAGEID>/..., failures.csv,
fastsurfer_idps.csv (rebuilt on every call from whatever has finished).
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pie.imaging import cat12
from pie.imaging.convert import convert_series
from pie.imaging.fastsurfer import (STATS_FILE, complete_segmentation, finish_stats, hippocampal_subfields, longitudinal,
                                    run_fastsurfer, sclimbic, segment_batch, surfaces, synthseg, talairach_etiv)
from pie.imaging.features import build_idp_table
from pie.imaging.index import (T1_HEADER_FIELDS, visit_to_event, index_zips, probe_headers, protocol_phase,
                               read_ida_metadata, read_loni_collection_csv, select_t1_series, t1_candidates)
from pie.imaging.link import link_adni, link_sessions_to_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("PIE.imaging")


def prepare_sessions(zips, ppmi_dir, work, ida_metadata=(), loni_csv=(), adnimerge=None):
    """Index zips, choose one T1 per session, link to EVENT_ID. Cached in sessions.csv.
    IDA metadata zips (LONI 'Advanced Download'), when given, override the date-based visit link
    and add scanner/protocol columns."""
    sessions_csv = work / "sessions.csv"
    if sessions_csv.exists():
        cached = pd.read_csv(sessions_csv, dtype={"image_id": str}, parse_dates=["session_date"])
        if not all(Path(z).exists() for z in zips):   # e.g. a backfill stage with the download drive unmounted
            log.warning("some zips are not on disk: using sessions.csv as it is")
            return cached
    index = index_zips(zips, cache_csv=work / "index.csv")
    sessions = select_t1_series(index, t1_headers(index, work / "t1_headers.csv"))
    if sessions_csv.exists():
        if set(cached["image_id"]) == set(sessions["image_id"]):
            return cached
        log.info("sessions.csv does not match the zips' T1 selection (new download or selection fix): rebuilding")
    sessions["protocol_phase"] = protocol_phase(sessions["session_date"])
    sessions = link_sessions_to_events(sessions, ppmi_dir)
    if adnimerge:      # ADNI sessions: ADNIMERGE's VISCODE by exam date
        sessions = link_adni(sessions, adnimerge)
    if loni_csv:  # the collection CSV: visit label for every series, group, age at scan
        loni = read_loni_collection_csv(loni_csv)
        sessions = sessions.merge(loni, on="image_id", how="left")
        ev = sessions["loni_visit"].map(visit_to_event, na_action="ignore")
        sessions["EVENT_ID"] = ev.where(ev.notna(), sessions["EVENT_ID"])
        log.info("LONI collection CSV: %d of %d sessions covered", int(sessions["loni_visit"].notna().sum()), len(sessions))
    if ida_metadata:
        ida = read_ida_metadata(ida_metadata).drop(columns=["patno"])
        sessions = sessions.merge(ida, on="image_id", how="left")
        ida_event = sessions["ida_visit"].map(visit_to_event, na_action="ignore")
        sessions["EVENT_ID"] = ida_event.where(ida_event.notna(), sessions["EVENT_ID"])
        log.info("IDA metadata: %d of %d sessions covered", int(sessions["ida_visit"].notna().sum()), len(sessions))
    sessions.to_csv(sessions_csv, index=False)
    log.info("sessions: %d (from %d series, %d subjects)", len(sessions), len(index), sessions["patno"].nunique())
    return sessions


def t1_headers(index, cache_csv):
    """DICOM headers (``T1_HEADER_FIELDS``) of the T1 candidates in visits with more than one, one file per series read
    from the zips; cached and extended incrementally."""
    cand = t1_candidates(index)
    cached = pd.read_csv(cache_csv, dtype={"image_id": str}) if Path(cache_csv).exists() else pd.DataFrame({"image_id": []})
    todo = cand[~cand["image_id"].astype(str).isin(set(cached["image_id"]))]
    if len(todo):
        cached = pd.concat([cached, probe_headers(todo, fields=T1_HEADER_FIELDS)], ignore_index=True)
        cached.to_csv(cache_csv, index=False)
    return cached


def convert_one(row, work):
    meta = convert_series(row["zip"], row["member_prefix"], row["patno"], row["image_id"], work / "nifti")
    import nibabel as nib  # reject volumes FastSurfer cannot segment before they abort a whole GPU batch
    img = nib.load(meta["nifti"])
    shape, zooms = img.shape, img.header.get_zooms()[:3]
    if len(shape) != 3 or min(shape) < 40 or max(zooms) > 2.5:
        raise ValueError(f"not a usable 3D T1 volume: shape {shape}, voxel {tuple(round(float(z), 2) for z in zooms)} mm")
    return row["image_id"], meta


SEG_PARTS = ("orig.mgz", "aparc.DKTatlas+aseg.deep.mgz", "aseg.auto_noCCseg.mgz", "mask.mgz")


def sweep_incomplete(work):
    """Delete segmentations that a killed batch left half-written so the next GPU batch redoes them."""
    n = 0
    for seg in (work / "fastsurfer").glob("*/mri/aparc.DKTatlas+aseg.deep.mgz"):
        if not (seg.parent / "orig.mgz").exists():
            seg.unlink()
            n += 1
    return n


def stats_one(row, work, threads):
    """N4 + stats (CPU only). Returns the image_id, or None if the segmentation was incomplete (it is deleted
    and the caller re-queues the scan for the next GPU batch; never wait for the GPU inside the CPU pool)."""
    mri = work / "fastsurfer" / row["image_id"] / "mri"
    if not (mri / "orig.mgz").exists() or not complete_segmentation(work / "fastsurfer", row["image_id"]):
        (mri / "aparc.DKTatlas+aseg.deep.mgz").unlink(missing_ok=True)
        return None
    finish_stats(work / "fastsurfer", row["image_id"], threads=threads)
    return row["image_id"]


def retry_one(row, work, threads, device="cuda"):
    """Single-scan fallback (own FastSurfer process) for scans a batch skipped."""
    nifti = work / "nifti" / str(row["patno"]) / f'{row["image_id"]}_T1w.nii.gz'
    run_fastsurfer(nifti, work / "fastsurfer", row["image_id"], threads=threads, device=device)
    return row["image_id"]


def process_all(todo, work, workers, threads, chunk, meta_csv, log_ok, log_fail, device="cuda"):
    """Pipeline: convert (CPU pool) -> FastSurferVINN on a chunk of scans in one GPU process
    (models loaded once) -> N4 + stats (CPU pool). Conversion of the next chunk and stats of the
    previous chunk overlap with the GPU stage, which is the bottleneck. Scans whose segmentation
    turns out incomplete are re-queued for a second pass."""
    log.info("swept %d incomplete segmentations", sweep_incomplete(work))
    rows = todo.to_dict("records")
    requeue = _run_chunks(rows, work, workers, threads, chunk, meta_csv, log_ok, log_fail, device)
    if requeue:
        log.info("second pass for %d re-queued scans", len(requeue))
        left = _run_chunks(requeue, work, workers, threads, chunk, meta_csv, log_ok, log_fail, device)
        for r in left:
            log_fail(r, "segmentation still incomplete after re-run")


def _run_chunks(rows, work, workers, threads, chunk, meta_csv, log_ok, log_fail, device="cuda"):
    requeue = []
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        convert_futs = {r["image_id"]: ex.submit(convert_one, r, work) for r in chunks[0]} if chunks else {}
        stats_futs = {}
        for i, rows_i in enumerate(chunks):
            if i + 1 < len(chunks):  # start converting the next chunk now
                convert_futs.update({r["image_id"]: ex.submit(convert_one, r, work) for r in chunks[i + 1]})
            metas, niftis, by_id = {}, [], {r["image_id"]: r for r in rows_i}
            for r in rows_i:
                try:
                    image_id, meta = convert_futs.pop(r["image_id"]).result()
                    metas[image_id], _ = meta, niftis.append(meta["nifti"])
                except Exception as e:
                    log_fail(r, e)
            segmented = set(segment_batch(niftis, work / "fastsurfer", threads=threads, device=device))
            done_before = {Path(n).name.removesuffix("_T1w.nii.gz") for n in niftis} - segmented
            for image_id in done_before:  # not segmented by the batch: already done earlier, or failed -> retry alone
                if not (work / "fastsurfer" / image_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
                    try:
                        retry_one(by_id[image_id], work, threads, device)
                    except Exception as e:
                        log_fail(by_id[image_id], e)
                        continue
                segmented.add(image_id)
            for image_id in segmented:
                stats_futs[ex.submit(stats_one, by_id[image_id], work, threads)] = (by_id[image_id], metas[image_id])
            # harvest finished stats without blocking the GPU stage
            for f in [f for f in stats_futs if f.done()]:
                r, meta = stats_futs.pop(f)
                _record(f, r, meta, meta_csv, log_ok, log_fail, requeue)
        for f in as_completed(list(stats_futs)):
            r, meta = stats_futs.pop(f)
            _record(f, r, meta, meta_csv, log_ok, log_fail, requeue)
    return requeue


def _record(fut, row, meta, meta_csv, log_ok, log_fail, requeue):
    try:
        image_id = fut.result()
    except Exception as e:
        return log_fail(row, e)
    if image_id is None:
        log.warning("incomplete segmentation for %s; re-queued", row["image_id"])
        return requeue.append(row)
    append_meta(meta_csv, {"image_id": image_id, **meta})
    log_ok(image_id)


def append_meta(meta_csv, row):
    """Append one row to scan_metadata.csv; a row with fields the table lacks (a newer PIE records more) widens the
    table, keeping every earlier row, instead of appending a row the header does not describe."""
    meta_csv = Path(meta_csv)
    header = next(csv.reader(open(meta_csv)), []) if meta_csv.exists() and meta_csv.stat().st_size else []
    if header and set(row) - set(header):
        old = pd.read_csv(meta_csv, dtype=str)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(meta_csv, index=False)
        return
    with open(meta_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header or list(row), extrasaction="ignore")
        if not header:
            w.writeheader()
        w.writerow(row)


def long_timepoints(sessions, subjects_dir):
    """PATNO -> finished FastSurfer image IDs, oldest first, for subjects with at least two real-dated sessions (LONI's
    9999 masked dates cannot be ordered, so they are left out)."""
    s = sessions.assign(_d=pd.to_datetime(sessions["session_date"], errors="coerce"))
    s = s[s["_d"].dt.year.lt(2100) & s["image_id"].map(lambda i: (Path(subjects_dir) / str(i) / STATS_FILE).exists())]
    out = {int(p): g.sort_values("_d")["image_id"].astype(str).tolist() for p, g in s.groupby("patno")}
    return {p: ids for p, ids in out.items() if len(ids) >= 2}


def _freesurfer_ready(tcsh=False):
    """Stop the whole run once, with one message, when FreeSurfer, its license or tcsh (talairach_avi) is missing;
    otherwise every subject would fail on its own and the table would be rewritten with an empty column."""
    from pie.imaging import freesurfer
    try:
        freesurfer.fs_env()
    except RuntimeError as e:
        raise SystemExit(str(e))
    if tcsh and not Path("/bin/tcsh").exists():
        raise SystemExit("FreeSurfer's talairach_avi needs /bin/tcsh: sudo apt install tcsh")


def _etiv_job(args):
    """FreeSurfer eTIV for one finished subject; a failure is reported and left missing (module level: pickled by name)."""
    fs_dir, sid = args
    try:
        return sid, talairach_etiv(fs_dir, sid), ""
    except Exception as e:
        return sid, float("nan"), f"{type(e).__name__}: {str(e)[:200]}"


STAGES = {   # optional per-subject stages on finished FastSurfer subjects: (subjects dir, sid, threads, raw T1 NIfTI)
    "surfaces": lambda fs, sid, t, nii: surfaces(fs, sid, threads=t),
    "sclimbic": lambda fs, sid, t, nii: sclimbic(fs, sid, threads=t),
    "subfields": lambda fs, sid, t, nii: hippocampal_subfields(fs, sid, threads=t),
    "synthseg": lambda fs, sid, t, nii: synthseg(nii, Path(fs) / sid / "synthseg", threads=t),
    "ch4": lambda fs, sid, t, nii: cat12.ch4_volume(fs, sid, nii, threads=t),
}


def _stage_job(args):
    """One optional stage for one finished subject; a failure is reported, not raised (module level: pickled by name)."""
    stage, fs_dir, sid, threads, nii = args
    try:
        STAGES[stage](fs_dir, sid, threads, nii)
        return sid, ""
    except Exception as e:
        return sid, f"{type(e).__name__}: {str(e)[:200]}"


def _with_metadata(sessions, work):
    meta_csv = work / "scan_metadata.csv"
    if not meta_csv.exists():
        return sessions
    meta = pd.read_csv(meta_csv, dtype={"image_id": str}).drop_duplicates("image_id", keep="last")
    return sessions.merge(meta, on="image_id", how="left")


def write_long_idps(sessions, work, timepoints):
    """``fastsurfer_long_idps.csv``: the IDP table of the longitudinal stream's time points (``stream`` = long,
    ``long_template`` = the participant), same columns as ``fastsurfer_idps.csv``; read with ``manifest.t1_visits``."""
    ids = {i for tps in timepoints.values() for i in tps}
    idps = build_idp_table(_with_metadata(sessions[sessions["image_id"].astype(str).isin(ids)], work), work / "fastsurfer_long")
    if len(idps):
        idps = idps.assign(stream="long", long_template=idps["PATNO"].astype(str))
    idps.to_csv(work / "fastsurfer_long_idps.csv", index=False)
    return idps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True, help="LONI zips (PPMI, ADNI or any LONI project) and/or BIDS dataset folders")
    ap.add_argument("--ppmi-dir", default="PPMI")
    ap.add_argument("--work-dir", default="Imaging/derived")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per FastSurfer worker")
    ap.add_argument("--chunk", type=int, default=20, help="scans per GPU inference process (models loaded once per chunk)")
    ap.add_argument("--device", default="cuda", help="FastSurfer inference device, passed to its --device (e.g. cuda, cuda:1, cpu; cpu is far slower)")
    ap.add_argument("--ida-metadata", nargs="*", default=(), help="LONI 'Advanced Download' metadata zip(s) (idaxs XML)")
    ap.add_argument("--loni-csv", nargs="*", default=(), help="LONI collection CSV(s) downloaded with the images (visit, group, age)")
    ap.add_argument("--adnimerge", help="ADNIMERGE.csv: visit codes for ADNI sessions (zips whose top folder is ADNI)")
    ap.add_argument("--priority", help="text file of PATNOs to process first")
    ap.add_argument("--limit", type=int, help="process at most N sessions this call")
    ap.add_argument("--features-only", action="store_true", help="only rebuild fastsurfer_idps.csv")
    ap.add_argument("--long", action="store_true", help="FastSurfer longitudinal stream for every subject with >= 2 finished sessions "
                    "into <work-dir>/fastsurfer_long (needs PIE_FREESURFER_HOME = FreeSurfer 7.4.1 and PIE_FS_LICENSE; hours per subject)")
    ap.add_argument("--surfaces", action="store_true", help="FastSurfer surfaces (cortical thickness, DK aparc) for every finished "
                    "subject, then rebuild the table with cth_*/area_* columns; needs PIE_FREESURFER_HOME and PIE_FS_LICENSE; ~1 h per scan")
    ap.add_argument("--sclimbic", action="store_true", help="FreeSurfer ScLimbic (basal forebrain / nucleus basalis region, hypothalamus, "
                    "fornix, septal nuclei) for every finished subject, then rebuild the table; ~1 min and ~3.3 GB per scan")
    ap.add_argument("--subfields", action="store_true", help="hippocampal subfields and amygdala nuclei (segment_subregions) for every "
                    "subject that has run --surfaces, then rebuild the table")
    ap.add_argument("--ch4", action="store_true", help="nucleus basalis (Julich-Brain Ch4) grey-matter volume with CAT12 (modulated GM "
                    "after Shooting, the published method; pie.imaging.cat12), on the raw T1 of every finished subject; needs PIE_CAT12 "
                    "and PIE_MCR; ~10-20 CPU-min per scan")
    ap.add_argument("--synthseg", action="store_true", help="SynthSeg-robust volumes and QC on the raw T1 of every finished subject "
                    "(legacy 1.5 T / thick-slice robustness, FastSurfer cross-check), then rebuild the table")
    ap.add_argument("--etiv", action="store_true", help="add FreeSurfer eTIV (talairach registration) to every finished subject, then "
                    "rebuild the table; needs PIE_FREESURFER_HOME, PIE_FS_LICENSE and /bin/tcsh")
    a = ap.parse_args(argv)

    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sessions = prepare_sessions(a.zips, a.ppmi_dir, work, a.ida_metadata, a.loni_csv, a.adnimerge)
    meta_csv = work / "scan_metadata.csv"
    done = {p.parent.parent.name for p in (work / "fastsurfer").glob(f"*/{STATS_FILE}")}

    if a.long:
        _freesurfer_ready()
        tps = dict(sorted(long_timepoints(sessions, work / "fastsurfer").items())[:a.limit])
        for patno, ids in tps.items():
            t1s = [work / "nifti" / str(patno) / f"{i}_T1w.nii.gz" for i in ids]
            try:
                longitudinal(str(patno), t1s, ids, work / "fastsurfer_long", device=a.device, threads=a.threads)
                log.info("longitudinal %s: %d time points", patno, len(ids))
            except Exception as e:
                log.error("longitudinal %s failed: %s", patno, str(e)[:300])
        log.info("long IDP table: %s", write_long_idps(sessions, work, tps).shape)
        return

    stages = [st for st in STAGES if getattr(a, st)]     # surfaces before subfields, which need them
    if stages:
        _freesurfer_ready()
        sids = sorted(done)[:a.limit] if a.limit else sorted(done)
        nii = dict(zip(sessions["image_id"].astype(str), (work / "nifti" / str(p) / f"{i}_T1w.nii.gz" for p, i in zip(sessions["patno"], sessions["image_id"]))))
        for st in stages:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                jobs = [(st, work / "fastsurfer", s, a.threads, nii.get(s)) for s in sids]
                for i, (sid, err) in enumerate(ex.map(_stage_job, jobs), start=1):
                    if err or i % 20 == 0:
                        log.info("%s %d/%d %s %s", st, i, len(sids), sid, err)
        a.features_only = True                     # rebuild fastsurfer_idps.csv with the new columns

    if a.etiv:
        _freesurfer_ready(tcsh=True)
        sids = sorted(done)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (sid, v, err) in enumerate(ex.map(_etiv_job, [(work / "fastsurfer", s) for s in sids]), start=1):
                if err or i % 50 == 0:
                    log.info("eTIV %d/%d %s %s", i, len(sids), sid, err or round(v))
        a.features_only = True                     # rebuild fastsurfer_idps.csv with the eTIV column

    if not a.features_only:
        todo = sessions[~sessions["image_id"].isin(done)]
        if a.priority:
            prio = {int(x) for x in Path(a.priority).read_text().split()}
            todo = todo.assign(_p=todo["patno"].isin(prio)).sort_values("_p", ascending=False, kind="stable").drop(columns="_p")
        if a.limit:
            todo = todo.head(a.limit)
        log.info("done: %d, to do now: %d", len(done), len(todo))
        t0, counter = time.time(), [0]
        fail_fh = open(work / "failures.csv", "a")

        def log_ok(image_id):
            counter[0] += 1
            log.info("ok %s (%d/%d, %.1f min elapsed)", image_id, counter[0], len(todo), (time.time() - t0) / 60)

        def log_fail(row, e):
            log.error("FAILED %s: %s", row["image_id"], e)
            fail_fh.write(f'{row["patno"]},{row["image_id"]},"{str(e)[:300]}"\n')
            fail_fh.flush()

        process_all(todo, work, a.workers, a.threads, a.chunk, meta_csv, log_ok, log_fail, device=a.device)
        fail_fh.close()

    idps = build_idp_table(_with_metadata(sessions, work), work / "fastsurfer")
    tmp = work / "fastsurfer_idps.csv.partial"
    try:
        idps.to_csv(tmp, index=False)
        tmp.replace(work / "fastsurfer_idps.csv")   # a failed write (disk full, kill) keeps the previous table
    finally:
        tmp.unlink(missing_ok=True)
    log.info("IDP table: %s -> %s", idps.shape, work / "fastsurfer_idps.csv")


if __name__ == "__main__":
    main()
