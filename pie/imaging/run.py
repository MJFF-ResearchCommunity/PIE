#!/usr/bin/env python
"""
run.py — resumable CLI: LONI zips -> NIfTI -> FastSurfer -> IDP table.

    venv_imaging/bin/python -m pie.imaging.run \
        --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
        --ppmi-dir PPMI --work-dir Imaging/derived --workers 4 --threads 4 \
        [--priority patnos.txt] [--limit N] [--features-only]

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

from pie.imaging.convert import convert_series
from pie.imaging.fastsurfer import STATS_FILE, complete_segmentation, finish_stats, run_fastsurfer, segment_batch
from pie.imaging.features import build_idp_table
from pie.imaging.index import IDA_VISIT_TO_EVENT, LONI_VISIT_TO_EVENT, index_zips, read_ida_metadata, read_loni_collection_csv, select_t1_series
from pie.imaging.link import link_sessions_to_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("PIE.imaging")


def prepare_sessions(zips, ppmi_dir, work, ida_metadata=(), loni_csv=()):
    """Index zips, choose one T1 per session, link to EVENT_ID. Cached in sessions.csv.
    IDA metadata zips (LONI 'Advanced Download'), when given, override the date-based visit link
    and add scanner/protocol columns."""
    sessions_csv = work / "sessions.csv"
    if sessions_csv.exists():
        return pd.read_csv(sessions_csv, dtype={"image_id": str}, parse_dates=["session_date"])
    index = index_zips(zips, cache_csv=work / "index.csv")
    sessions = select_t1_series(index)
    sessions["protocol_phase"] = sessions["zip"].map({z: i + 1 for i, z in enumerate(map(str, zips))})
    sessions = link_sessions_to_events(sessions, ppmi_dir)
    if loni_csv:  # the collection CSV: visit label for every series, group, age at scan
        loni = read_loni_collection_csv(loni_csv)
        sessions = sessions.merge(loni, on="image_id", how="left")
        ev = sessions["loni_visit"].map(LONI_VISIT_TO_EVENT)
        sessions["EVENT_ID"] = ev.where(ev.notna(), sessions["EVENT_ID"])
        log.info("LONI collection CSV: %d of %d sessions covered", int(sessions["loni_visit"].notna().sum()), len(sessions))
    if ida_metadata:
        ida = read_ida_metadata(ida_metadata).drop(columns=["patno"])
        sessions = sessions.merge(ida, on="image_id", how="left")
        ida_event = sessions["ida_visit"].map(IDA_VISIT_TO_EVENT)
        sessions["EVENT_ID"] = ida_event.where(ida_event.notna(), sessions["EVENT_ID"])
        log.info("IDA metadata: %d of %d sessions covered", int(sessions["ida_visit"].notna().sum()), len(sessions))
    sessions.to_csv(sessions_csv, index=False)
    log.info("sessions: %d (from %d series, %d subjects)", len(sessions), len(index), sessions["patno"].nunique())
    return sessions


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


def retry_one(row, work, threads):
    """Single-scan fallback (own FastSurfer process) for scans a batch skipped."""
    nifti = work / "nifti" / str(row["patno"]) / f'{row["image_id"]}_T1w.nii.gz'
    run_fastsurfer(nifti, work / "fastsurfer", row["image_id"], threads=threads)
    return row["image_id"]


def process_all(todo, work, workers, threads, chunk, meta_csv, log_ok, log_fail):
    """Pipeline: convert (CPU pool) -> FastSurferVINN on a chunk of scans in one GPU process
    (models loaded once) -> N4 + stats (CPU pool). Conversion of the next chunk and stats of the
    previous chunk overlap with the GPU stage, which is the bottleneck. Scans whose segmentation
    turns out incomplete are re-queued for a second pass."""
    log.info("swept %d incomplete segmentations", sweep_incomplete(work))
    rows = todo.to_dict("records")
    requeue = _run_chunks(rows, work, workers, threads, chunk, meta_csv, log_ok, log_fail)
    if requeue:
        log.info("second pass for %d re-queued scans", len(requeue))
        left = _run_chunks(requeue, work, workers, threads, chunk, meta_csv, log_ok, log_fail)
        for r in left:
            log_fail(r, "segmentation still incomplete after re-run")


def _run_chunks(rows, work, workers, threads, chunk, meta_csv, log_ok, log_fail):
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
            segmented = set(segment_batch(niftis, work / "fastsurfer", threads=threads))
            done_before = {Path(n).name.removesuffix("_T1w.nii.gz") for n in niftis} - segmented
            for image_id in done_before:  # not segmented by the batch: already done earlier, or failed -> retry alone
                if not (work / "fastsurfer" / image_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
                    try:
                        retry_one(by_id[image_id], work, threads)
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
    write_header = not meta_csv.exists()
    with open(meta_csv, "a") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", *meta])
        if write_header:
            w.writeheader()
        w.writerow({"image_id": image_id, **meta})
    log_ok(image_id)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--ppmi-dir", default="PPMI")
    ap.add_argument("--work-dir", default="Imaging/derived")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per FastSurfer worker")
    ap.add_argument("--chunk", type=int, default=20, help="scans per GPU inference process (models loaded once per chunk)")
    ap.add_argument("--ida-metadata", nargs="*", default=(), help="LONI 'Advanced Download' metadata zip(s) (idaxs XML)")
    ap.add_argument("--loni-csv", nargs="*", default=(), help="LONI collection CSV(s) downloaded with the images (visit, group, age)")
    ap.add_argument("--priority", help="text file of PATNOs to process first")
    ap.add_argument("--limit", type=int, help="process at most N sessions this call")
    ap.add_argument("--features-only", action="store_true", help="only rebuild fastsurfer_idps.csv")
    a = ap.parse_args(argv)

    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sessions = prepare_sessions(a.zips, a.ppmi_dir, work, a.ida_metadata, a.loni_csv)
    meta_csv = work / "scan_metadata.csv"
    done = {p.parent.parent.name for p in (work / "fastsurfer").glob(f"*/{STATS_FILE}")}

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

        process_all(todo, work, a.workers, a.threads, a.chunk, meta_csv, log_ok, log_fail)
        fail_fh.close()

    if meta_csv.exists():
        meta = pd.read_csv(meta_csv, dtype={"image_id": str}).drop_duplicates("image_id", keep="last")
        sessions = sessions.merge(meta, on="image_id", how="left")
    idps = build_idp_table(sessions, work / "fastsurfer")
    idps.to_csv(work / "fastsurfer_idps.csv", index=False)
    log.info("IDP table: %s -> %s", idps.shape, work / "fastsurfer_idps.csv")


if __name__ == "__main__":
    main()
