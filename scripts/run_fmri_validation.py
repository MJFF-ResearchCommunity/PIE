"""Run step of the pre-registered rs-fMRI validation (documentation/preregistration_fmri.md, "Run step").

Each scan of ``fmri_selection.csv`` goes IDA zip -> NIfTI -> BIDS -> fMRIPrep 25.2.5 -> XCP-D 26.2.0 -> one row of
``scans.csv``, the table ``analyse_fmri_validation.py analyse --scans`` reads. P2_TR1 scans also run XCP-D with the
motion filter off (protocol ``P2_TR1:no_notch``). A scan ends as a feature row or one exclusion reason, the first met
in the order the steps run:
- conversion_error;
- too_few_volumes (no rest candidate), no_verified_reverse_pe_pair, header_tr_outside_stratum, no_t1;
- fmriprep_error;
- xcpd_error or xcpd_min_time.

Visual review (``--review``) is applied when scans.csv is written; coverage is applied by the analysis.

    PYTHONPATH=. python scripts/run_fmri_validation.py --selection <dir>/fmri_selection.csv --zips <zips> \
        --sessions Imaging/derived/sessions.csv --work <dir> --fmriprep-sif <sif> --xcpd-sif <sif> --license <txt>

Rows persist per scan under <work>/rows/, so a restart resumes; scans.csv is rewritten after every scan. Once a scan's
features are written, its BOLD derivatives, transforms, BIDS copy, conversions and work directories are deleted;
reports, parcel TSVs and atlases stay (moved to ``--keep`` when given). The first scan runs alone, so the TemplateFlow
cache fills before the others share it.
"""
import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_fmri_validation import _stratum  # noqa: E402  the frozen stratum rule, reused for the header-TR check

from pie.imaging import fmri, fmriprep, xcpd  # noqa: E402
from pie.imaging.fmri_bids import export_subject  # noqa: E402

CODE = {"P1_TR2.4": "P1TR24", "P1_TR2.5": "P1TR25", "P2_TR1": "P2TR1", "P2_TR2.5": "P2TR25"}
PAIRED = {"P2_TR1", "P2_TR2.5"}      # protocols that acquire a reverse-PE reference; PPMI-1 only with --paired-p1
NO_NOTCH = {"P2_TR1"}
T1_DESC = r"mprage|t1.?weighted|spgr|bravo|t1.?tfe|3d.?t1"
ATLASES = ("4S456Parcels", "4S156Parcels")
MIN_TIME_LOG = "high-motion outlier scrubbing"      # XCP-D's message when a run keeps under --min-time
FROZEN = ["scripts/run_fmri_validation.py", "pie/imaging/fmri_bids.py", "pie/imaging/xcpd.py", "pie/imaging/fmri_connectivity.py"]


def label(row):
    return f"{int(row['patno'])}{CODE[row['stratum']]}"


def _meta(o):
    return json.loads(Path(o["sidecar"]).read_text(), strict=False)


def convert_session(row, series, work):
    """Every converter output of the session's selected rs-fMRI series (``series``: its ``index_series`` rows). A series
    split into 3D volumes is reassembled only when its DICOM headers prove the order (``fmri_assembly``)."""
    from pie.imaging.archives import ZipArchiveCache
    from pie.imaging.dicom_audit import audit_archive_headers
    from pie.imaging.fmri_assembly import assemble_temporal_volumes

    patno, out = str(int(row["patno"])), []
    conv = work / "convert"
    conv.mkdir(parents=True, exist_ok=True)
    with ZipArchiveCache() as cache:
        for image_id in str(row["image_ids"]).split(";"):
            s = series.loc[series["image_id"].eq(image_id)].iloc[0]
            rec = fmri.convert_archive_series({"PATNO": patno, "image_id": image_id, "series_prefix": s["prefix"],
                                               "archive": s["zip"], "dicom_entries": int(s["n_files"])}, conv, archive_cache=cache)
            src = conv / patno / image_id
            if len(rec["outputs"]) > 1 and not any(len(o["shape"]) == 4 for o in rec["outputs"]):
                audit = work / "audit" / f"{image_id}.json"
                if not audit.exists():
                    audit.parent.mkdir(exist_ok=True)
                    fmri.write_json(audit, audit_archive_headers(s["zip"], s["prefix"], expected_count=int(s["n_files"]),
                                                                 archive_cache=cache))
                src = work / "assembled" / patno / image_id
                src.parent.mkdir(parents=True, exist_ok=True)
                rec = assemble_temporal_volumes(conv / patno / image_id, audit, src)
            out += [dict(o, image_id=image_id, nifti=str(src / o["nifti"]), sidecar=str(src / o["sidecar"])) for o in rec["outputs"]]
    return out


def choose(outputs, paired):
    """(bold, reference). bold: the rest candidate with the most volumes (ties: earliest AcquisitionTime, then name),
    None when there is none. reference: the first other 4D output (short references first) that forms a verified
    reverse-PE pair with it (``phase_encoding_pair`` and equal echo times, as ``export_subject`` requires), else None."""
    rest = [o for o in outputs if o["run_class"] == "rest_candidate"]
    if not rest:
        return None, None
    order = lambda o: (str(_meta(o).get("AcquisitionTime", "")), o["nifti"])
    bold = min(rest, key=lambda o: (-o["shape"][3], *order(o)))
    if not paired:
        return bold, None
    te = _meta(bold).get("EchoTime")
    others = sorted((o for o in outputs if o is not bold and len(o["shape"]) == 4 and o["shape"][3] >= 3),
                    key=lambda o: (o["run_class"] != "short_reference_candidate", *order(o)))
    for o in others:
        e = _meta(o).get("EchoTime")
        if not fmri.phase_encoding_pair(bold, o) and all(isinstance(x, (int, float)) for x in (te, e)) and abs(te - e) <= 1e-5:
            return bold, o
    return bold, None


def find_t1(row, series, sessions_csv, work):
    """(nifti, sidecar, source) of a 3D T1 from the scan's own date: PIE's T1 (``sessions.csv``) first, else the best T1
    series in the fMRI zips (``index.select_t1_series``); None when there is none."""
    from pie.imaging.convert import convert_series
    from pie.imaging.index import select_t1_series

    patno, day = int(row["patno"]), str(row["date"])[:10]
    s = pd.read_csv(sessions_csv, dtype={"image_id": str})
    s = s[(s["patno"] == patno) & s["session_date"].astype(str).str[:10].eq(day)]
    if len(s):
        r = s.iloc[0]
        m, source = convert_series(r["zip"], r["member_prefix"], patno, r["image_id"], Path(sessions_csv).parent / "nifti"), "sessions"
    else:
        z = series[series["date"].eq(day) & series["desc"].str.contains(T1_DESC, case=False)
                   & ~series["desc"].str.contains("flair", case=False)]
        if z.empty:
            return None
        r = select_t1_series(z.rename(columns={"desc": "series_desc", "date": "session", "n_files": "bytes"})).iloc[0]
        m, source = convert_series(r["zip"], r["prefix"], patno, r["image_id"], work / "t1"), "fmri_zip"
    return (m["nifti"], m["sidecar"], source) if nib.load(m["nifti"]).ndim == 3 else None


def features(xcpd_dir, lab):
    """The feature row: ``xcpd_features`` (both atlases), striatal seeds, and the striatal seeds on each half of the
    retained frames (``__h1``/``__h2``, split as the analysis splits the parcel series). No BOLD: xcpd_min_time."""
    row = xcpd.xcpd_features(xcpd_dir, lab, atlases=ATLASES)
    if not row.get("fmri_xcpd_available"):
        return {"exclusion": "xcpd_min_time"}
    row |= xcpd.xcpd_striatal_features(xcpd_dir, lab)
    ts = xcpd._one(Path(xcpd_dir) / f"sub-{lab}", f"sub-{lab}_*task-rest_*atlas-4S456Parcels_*stat-mean_timeseries.tsv")
    n = len(pd.read_table(ts))
    for tag, frames in (("__h1", slice(0, n // 2)), ("__h2", slice(n // 2, None))):
        row |= {k + tag: v for k, v in xcpd.xcpd_striatal_features(xcpd_dir, lab, frames=frames).items()}
    return row


def _min_time_logged(output_dir):
    return any(MIN_TIME_LOG in p.read_text(errors="replace").lower() for p in Path(output_dir).glob("pie_provenance/*/attempt-*.log"))


def cleanup(work, lab, patno, outputs):
    """Delete what the analysis does not read: BOLD derivatives, transforms, the BIDS copy, conversions and work dirs."""
    for d in ("fmriprep_work", "xcpd_work", "xcpd_no_notch_work", "bids"):
        shutil.rmtree(work / d / lab, ignore_errors=True)
    for d in ("fmriprep", "xcpd", "xcpd_no_notch"):
        for p in [*(work / d / lab).rglob("*_bold.nii.gz"), *(work / d / lab).rglob("*_xfm.h5")]:
            p.unlink()
    for image_id in {o["image_id"] for o in outputs}:
        for d in ("convert", "assembled"):
            shutil.rmtree(work / d / patno / image_id, ignore_errors=True)


def process(row, series, a):
    """One selected scan -> {protocol: row}."""
    work, lab, stratum = Path(a.work), label(row), row["stratum"]
    patno = str(int(row["patno"]))
    base = {"scan": lab, "patno": int(row["patno"]), "session_date": str(row["date"])[:10], "protocol": stratum,
            "exclusion": "", "detail": "", "xcpd_dir": ""}
    stop = lambda reason, detail="": {stratum: dict(base, exclusion=reason, detail=str(detail)[-300:])}
    try:
        outputs = convert_session(row, series, work)
    except Exception as e:
        return stop("conversion_error", e)
    paired = stratum in PAIRED or (a.paired_p1 and stratum.startswith("P1"))
    bold, ref = choose(outputs, paired)
    if bold is None:
        return stop("too_few_volumes")
    m = _meta(bold)
    tr = bold["nifti_tr_seconds"] or bold["tr_seconds"]
    base.update(manufacturer=m.get("Manufacturer", ""), model=m.get("ManufacturersModelName", ""), header_tr=tr,
                bold_image_id=bold["image_id"])
    if paired and ref is None:
        return stop("no_verified_reverse_pe_pair")
    if _stratum(stratum[:2], tr, False) != stratum:
        return stop("header_tr_outside_stratum", tr)
    try:
        t1 = find_t1(row, series, a.sessions, work)
    except Exception as e:
        t1, base["detail"] = None, str(e)[-300:]
    if t1 is None:
        return stop("no_t1", base["detail"])
    bids = work / "bids" / lab
    try:
        rec = export_subject(bids, lab, t1[0], t1[1], bold["nifti"], bold["sidecar"],
                             *((ref["nifti"], ref["sidecar"]) if ref else ()), discard_seconds=10)
    except Exception as e:
        return stop("conversion_error", e)
    if paired and not rec["correction_pair_exported"]:
        return stop("no_verified_reverse_pe_pair", rec["unresolved_reasons"])
    base.update(distortion_correction=bool(rec["correction_pair_exported"]), t1_source=t1[2])
    common = dict(backend="apptainer", executable=a.apptainer, nprocs=a.nprocs, omp_nthreads=a.omp,
                  memory_mb=a.mem_mb, fs_license_file=a.license, participants=[lab])
    fp = fmriprep.FMRIPrepConfig(bids_dir=str(bids), output_dir=str(work / "fmriprep" / lab), work_dir=str(work / "fmriprep_work" / lab),
                                 cache_dir=str(work / "cache"), container_image=a.fmriprep_sif, expected_version="25.2.5", **common)
    try:
        fmriprep.run_fmriprep(fp)
    except Exception as e:
        return stop("fmriprep_error", e)
    rows = {}
    for protocol, motion in [(stratum, None)] + ([(stratum + ":no_notch", False)] if stratum in NO_NOTCH else []):
        tag = "xcpd" if motion is None else "xcpd_no_notch"
        out = work / tag / lab
        cfg = xcpd.XCPDConfig(fmriprep_dir=fp.output_dir, output_dir=str(out), work_dir=str(work / f"{tag}_work" / lab),
                              cache_dir=str(work / "cache_xcpd"), repetition_time=tr, container_image=a.xcpd_sif,
                              motion_filter=motion, **common)
        r = dict(base, protocol=protocol, xcpd_dir=str(out), fmriprep_report=str(Path(fp.output_dir) / f"sub-{lab}.html"),
                 xcpd_report=str(out / f"sub-{lab}.html"))
        try:
            xcpd.run_xcpd(cfg)
            r |= features(out, lab)
        except Exception as e:
            r |= {"exclusion": "xcpd_min_time" if _min_time_logged(out) else "xcpd_error", "detail": str(e)[-300:]}
        rows[protocol] = r
    cleanup(work, lab, patno, outputs)
    if a.keep:
        for tag in ("fmriprep", "xcpd", "xcpd_no_notch"):
            if (work / tag / lab).exists():
                (Path(a.keep) / tag).mkdir(parents=True, exist_ok=True)
                shutil.move(str(work / tag / lab), Path(a.keep) / tag / lab)
        for r in rows.values():
            for k in ("xcpd_dir", "fmriprep_report", "xcpd_report"):
                r[k] = r[k].replace(str(work), str(Path(a.keep)), 1)
    return rows


def write_scans(work, out_csv, review=None):
    """scans.csv from every finished scan; a failed visual review excludes a row that had none."""
    rows = [r for p in sorted((Path(work) / "rows").glob("*.json")) for r in json.loads(p.read_text()).values()]
    s = pd.DataFrame(rows)
    if review is not None and len(s):
        rv = pd.read_csv(review, dtype=str)
        bad = ~rv["pass"].str.strip().str.lower().isin({"true", "1", "yes", "pass"})
        failed = set(zip(rv.loc[bad, "scan"], rv.loc[bad, "protocol"]))
        hit = s["exclusion"].eq("") & pd.Series([k in failed for k in zip(s["scan"].astype(str), s["protocol"])], index=s.index)
        s.loc[hit, "exclusion"] = "visual_review_fail"
    s.to_csv(out_csv, index=False)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selection", required=True)
    ap.add_argument("--zips", nargs="+", required=True, help="every zip holding the selected series (and session T1s)")
    ap.add_argument("--sessions", default="Imaging/derived/sessions.csv")
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", help="move each scan's retained outputs here (e.g. a larger disk)")
    ap.add_argument("--out", help="scans.csv path (default <work>/scans.csv)")
    ap.add_argument("--review", help="visual review CSV: scan, protocol, pass")
    ap.add_argument("--fmriprep-sif", default=str(Path.home() / "containers/sif/fmriprep-25.2.5.sif"))
    ap.add_argument("--xcpd-sif", default=str(Path.home() / "containers/sif/xcp_d-26.2.0.sif"))
    ap.add_argument("--apptainer", default="apptainer")
    ap.add_argument("--license", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--omp", type=int, default=4)
    ap.add_argument("--mem-mb", type=int, default=16000)
    ap.add_argument("--paired-p1", action="store_true", help="PPMI-1 acquires a reverse-PE reference (from the census)")
    ap.add_argument("--retry-errors", action="store_true", help="re-run scans whose row is a pipeline error")
    a = ap.parse_args(argv)
    from pie.imaging.batch import index_series

    work = Path(a.work)
    (work / "rows").mkdir(parents=True, exist_ok=True)
    out_csv = Path(a.out or work / "scans.csv")
    root = Path(__file__).resolve().parents[1]
    fmri.write_json(work / "run_settings.json", {**{k: v for k, v in vars(a).items() if k != "license"},
                    "sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in FROZEN}})
    if a.retry_errors:
        for p in (work / "rows").glob("*.json"):
            if any(r["exclusion"] in {"conversion_error", "fmriprep_error", "xcpd_error"} for r in json.loads(p.read_text()).values()):
                p.unlink()
                for d in ("bids", "fmriprep", "fmriprep_work", "xcpd", "xcpd_work", "xcpd_no_notch", "xcpd_no_notch_work"):
                    shutil.rmtree(work / d / p.stem, ignore_errors=True)
    sel = pd.read_csv(a.selection)
    todo = [r for _, r in sel.iterrows() if not (work / "rows" / f"{label(r)}.json").exists()]
    index = index_series(a.zips) if todo else None
    jobs = [(r, index[index["patno"].eq(int(r["patno"]))]) for r in todo]

    def finish(r, rows):
        fmri.write_json(work / "rows" / f"{label(r)}.json", rows)
        write_scans(work, out_csv, a.review)
        print(label(r) + ": " + ", ".join(f"{p} {x['exclusion'] or 'ok'}" for p, x in rows.items()), flush=True)

    if jobs:
        finish(jobs[0][0], process(*jobs[0], a))        # alone: fills the TemplateFlow cache the others then share
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(process, r, s, a): r for r, s in jobs[1:]}
        for f in as_completed(futures):
            finish(futures[f], f.result())
    s = write_scans(work, out_csv, a.review)
    print(f"done: {len(sel)} selected, {len(s)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
