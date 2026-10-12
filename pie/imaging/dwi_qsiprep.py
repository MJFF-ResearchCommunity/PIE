"""
Opt-in QSIPrep / QSIRecon reference path for PIE's diffusion measures. Not the cohort pipeline.

QSIPrep (Cieslak et al. 2021, Nature Methods 18:775) is the community-standard diffusion preprocessing: b0 scaling
per run before the shells are concatenated, MP-PCA denoising and Gibbs unringing, FSL eddy with outlier replacement,
TOPUP or SyN-SDC. It does none of PPMI's conversion or series selection, and it resamples every scan onto the T1w
(ACPC) grid, one more interpolation for 2 mm nuclei than PIE's native-space measure. So it is a reference comparator:

    export   PIE's selected DWI session (``dwi.assemble``'s main runs as dwi/ runs; the reverse-PE b0s, as PIE gives
             them to topup, as fmap/*_epi with IntendedFor) and PIE's T1 nearest that session -> BIDS.
             PhaseEncodingDirection and TotalReadoutTime are written only when PIE knows both (``bids_sidecar``).
    run      pinned QSIPrep per participant: output resolution = the finest native voxel size, dwidenoise, mrdegibbs
             (rpg for partial-Fourier scans), eddy, TOPUP from the reverse-PE b0s, else SyN-SDC.
    recon    pinned QSIRecon with ``RECON_SPEC``: TORTOISE tensor (b <= 1200) and AMICO NODDI ISOVF.
    compare  PIE's study nigral ROIs (``dwi.study_nigra_labels``) pulled onto QSIPrep's grid; ``dwiq_*``: PIE's tensor
             rule (``dwi.study_features``) and multi-shell free water (``freewater_qc``) on QSIPrep's preprocessed
             DWI, and the QSIRecon maps; ``agreement`` with PIE's ``dwis_*`` (Lin's CCC, Bland-Altman, ICC(2,1)).

    venv_imaging/bin/python -m pie.imaging.dwi_qsiprep export --zips <DTI zips> --sessions <derived>/sessions.csv \\
        --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/dwi --bids-dir <qsi>/bids --patnos pilot.txt
    (run, recon, compare: documentation/imaging_dwi.md)
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import dwi
from .archives import ZipArchiveCache
from .fmri import sha256, write_json

# Docker Hub digests of the current stable releases (verified 11 October 2026; QSIPrep 26.0.0 of 20 April 2026, QSIRecon
# 26.0.1 of 29 September 2026). Pinning by digest makes Docker refuse any other image content under the tag.
QSIPREP_IMAGE = "pennlinc/qsiprep:26.0.0@sha256:b578687c4667ae43f4229212b3ae91c2ff842d49a29d42e64c013aaae9600373"
QSIRECON_IMAGE = "pennlinc/qsirecon:26.0.1@sha256:5f95877eeea494c5fd82be4b8dca657be63d132f0425f71d82b0448d20ff5d69"
# QSIPrep 26.0.0's default eddy settings (qsiprep/data/eddy_params.json), written per run so provenance holds them
EDDY_PARAMS = {"flm": "quadratic", "slm": "linear", "fep": False, "interp": "spline", "nvoxhp": 1000, "fudge_factor": 10,
               "dont_sep_offs_move": False, "dont_peas": False, "niter": 5, "method": "jac", "repol": True, "num_threads": 1,
               "is_shelled": True, "use_cuda": False, "cnr_maps": True, "residuals": False, "output_type": "NIFTI_GZ", "args": ""}
# No QSIRecon 26.0.1 built-in spec fits the bi-tensor free-water model (Pasternak 2009; Hoy 2014) that the PPMI nigral
# literature uses; AMICO NODDI's isotropic fraction (ISOVF) is the closest, from a different model. So this spec takes the
# built-in TORTOISE tensor and AMICO NODDI nodes as they are (hbcd_scalar_maps, amico_noddi), and ``compare`` fits PIE's
# own free-water model to QSIPrep's preprocessed DWI. The tensor's b <= 1200 includes PPMI 2.0's b = 700 shell; PIE's
# dwis_* use the b = 1000 run alone. JSON is valid YAML, so the spec is written with json.
RECON_SPEC = {"name": "pie_dti_noddi", "space": "T1w", "anatomical": [], "nodes": [
    {"name": "tortoise_tensor", "software": "TORTOISE", "action": "estimate", "input": "qsirecon", "qsirecon_suffix": "TORTOISE",
     "parameters": {"big_delta": None, "small_delta": None, "estimate_tensor_separately": True,
                    "estimate_tensor": {"bval_cutoff": 1200, "write_cs": True}}},
    {"name": "fit_noddi", "software": "AMICO", "action": "fit_noddi", "input": "qsirecon", "qsirecon_suffix": "NODDI",
     "parameters": {"dIso": 0.003, "dPar": 0.0017, "isExvivo": False}}]}
RECON_MAPS = {("tensor", "fa"), ("tensor", "md"), ("tensor", "ad"), ("tensor", "rd"), ("noddi", "isovf")}
# fields QSIPrep 26.0.0 derives a readout time or polarity from (interfaces/fmap.py get_trt); removed when PIE does not know them
GUESSABLE = ("PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing", "EstimatedTotalReadoutTime",
             "EstimatedEffectiveEchoSpacing", "WaterFatShift")


# ------------------------------------------------------------------------------------------ export
def bids_sidecar(meta):
    """dcm2niix metadata with PIE's phase encoding (``dwi._pe_from_name``: Philips polarity from PPMI series names). When
    PIE does not know both PhaseEncodingDirection and TotalReadoutTime (``dwi_acquisition.pe_row``), those and the fields
    QSIPrep would derive them from are removed: QSIPrep's eddy then stops on the missing field instead of guessing."""
    meta = dwi._pe_from_name(meta)
    return meta if dwi._pe_recorded(meta) else {k: v for k, v in meta.items() if k not in GUESSABLE}


def session_t1(sessions_csv, patno, date):
    """PIE's T1 (``sessions.csv``, converted as ``pie.imaging.run`` does, under <sessions dir>/nifti) acquired nearest the DWI
    date: (nifti, sidecar, gap in days; 0 = the same session, None = undated)."""
    from .convert import convert_series

    s = pd.read_csv(sessions_csv, dtype={"image_id": str})
    s = s[s["patno"] == int(patno)]
    if s.empty:
        raise ValueError("no T1 session")
    gap = (pd.to_datetime(s["session_date"], errors="coerce") - pd.to_datetime(date, errors="coerce")).abs().dt.days
    r = s.loc[gap.fillna(np.inf).idxmin()]
    meta = convert_series(r["zip"], r["member_prefix"], r["patno"], r["image_id"], Path(sessions_csv).parent / "nifti")
    return meta["nifti"], meta["sidecar"], None if pd.isna(gap.min()) else float(gap.min())


def export_subject(bids_dir, patno, series_rows, sessions_csv, work_dir):
    """One PIE-selected DWI session (``series_rows``: ``batch.session_rows``) as BIDS sub-<PATNO>/ses-<date>. Returns the
    provenance record (also sourcedata/sub-<PATNO>.json). Idempotent: an export is checked against its record and
    returned; a subject directory without a record raises."""
    label, ses = str(int(patno)), re.sub(r"[^0-9A-Za-z]", "", str(series_rows[0]["date"])) or "1"
    root = Path(bids_dir).resolve()
    t1, t1_json, gap = session_t1(sessions_csv, patno, series_rows[0]["date"])
    identity = {"series": sorted(r["prefix"] for r in series_rows), "t1_sha256": sha256(t1)}
    record_path = root / "sourcedata" / f"sub-{label}.json"
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        if saved["identity"] != identity or any(sha256(root / k) != v for k, v in saved["outputs"].items()):
            raise ValueError(f"sub-{label}: the BIDS export differs from its sources or record")
        return saved
    if (root / f"sub-{label}").exists():
        raise FileExistsError(f"sub-{label}: incomplete BIDS export requires review")
    (root / "sourcedata").mkdir(parents=True, exist_ok=True)
    if not (root / "dataset_description.json").exists():
        write_json(root / "dataset_description.json", {"Name": "PIE PPMI diffusion for QSIPrep", "BIDSVersion": "1.10.0",
                                                       "DatasetType": "raw"})
    work = Path(work_dir) / "qsiprep_export" / label
    try:
        runs = []
        with ZipArchiveCache() as zips:      # a session's DWI runs usually share a zip
            for r in series_rows:
                runs += dwi.convert(r["zip"], r["prefix"], work, archive_cache=zips)
        with tempfile.TemporaryDirectory(prefix="bids-export-", dir=root) as tmp:
            record = _stage(Path(tmp) / f"sub-{label}", runs, f"sub-{label}_ses-{ses}", ses, t1, t1_json)
            (Path(tmp) / f"sub-{label}").rename(root / f"sub-{label}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    record.update(identity=identity, t1_gap_days=gap, outputs={str(p.relative_to(root)): sha256(p)
                                                               for p in sorted((root / f"sub-{label}").rglob("*")) if p.is_file()})
    write_json(record_path, record)
    return record


def _stage(sub, runs, base, ses, t1, t1_json):
    ds = dwi.assemble(runs)
    out = sub / f"ses-{ses}"
    for d in ("anat", "dwi"):
        (out / d).mkdir(parents=True)
    shutil.copyfile(t1, out / "anat" / f"{base}_T1w.nii.gz")
    shutil.copyfile(t1_json, out / "anat" / f"{base}_T1w.json")
    by_nii = {r[0]: r for r in runs}
    names, metas = [], []
    for k, nii in enumerate(ds["source_nifti"], 1):       # the raw files of PIE's main group, in its order
        _, bval, bvec, js = by_nii[nii]
        name = f"{base}_run-{k}_dwi"
        for src, ext in ((nii, ".nii.gz"), (bval, ".bval"), (bvec, ".bvec")):
            shutil.copyfile(src, out / "dwi" / (name + ext))
        metas.append(bids_sidecar(json.loads(Path(js).read_text()) if Path(js).exists() else {}))
        write_json(out / "dwi" / f"{name}.json", metas[-1])
        names.append(f"ses-{ses}/dwi/{name}.nii.gz")      # IntendedFor: relative to the subject (BIDS URIs unsupported in 26.0.0)
    if ds["rev_b0"] is not None:     # PIE's reverse-PE choice (verified PE and readout), its b0s on the main grid as topup gets them
        (out / "fmap").mkdir()
        nib.save(nib.Nifti1Image(ds["rev_b0"], ds["affine"]), out / "fmap" / f"{base}_dir-reverse_epi.nii.gz")
        write_json(out / "fmap" / f"{base}_dir-reverse_epi.json", dict(ds["rev_meta"], IntendedFor=names))
    zooms = [float(z) for z in nib.load(ds["source_nifti"][0]).header.get_zooms()[:3]]
    pf = metas[0].get("PartialFourier")
    return {"session": ses, "dwi": names, "shells": ds["shells"], "reverse_pe_fmap": ds["rev_b0"] is not None,
            "pe_known": all("PhaseEncodingDirection" in m for m in metas), "voxel_mm": zooms,
            "output_resolution": round(min(zooms), 2),       # the finest native spacing: QSIPrep never downsamples a PE plane
            # QSIPrep's advice: TORTOISE's rpg for partial-Fourier acquisitions, mrdegibbs (PIE's --denoise) otherwise
            "unringing_method": "rpg" if isinstance(pf, float) and pf < 1 else "mrdegibbs"}


# ------------------------------------------------------------------------------------------ run / recon
def _separate(*paths):
    ps = [Path(p).resolve() for p in paths]
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("input, output and work directories must be separate, non-nested paths")


def _docker(image, mounts, fs_license, gpus=False):
    if not fs_license or not Path(fs_license).is_file():
        raise ValueError(f"FreeSurfer license not found: set PIE_FS_LICENSE or pass --fs-license (got {fs_license!r})")
    cmd = ["docker", "run", "--rm", "--pull", "never", "--user", f"{os.getuid()}:{os.getgid()}"] + (["--gpus", "all"] if gpus else [])
    for host, target in [*mounts, (fs_license, "/license.txt:ro")]:
        cmd += ["-v", f"{Path(host).resolve()}:{target}"]
    return cmd + [image]


def qsiprep_command(bids_dir, out_dir, work_dir, label, record, fs_license, nprocs=8, omp_nthreads=8, mem_mb=24000,
                    cuda=False, image=QSIPREP_IMAGE):
    """Docker argv for one participant (no writes). ``record`` is the export's: output resolution and unringing method.
    Distortion: TOPUP when the export wrote a reverse-PE fmap, else SyN-SDC (QSIPrep uses SyN only without a fieldmap;
    'warn' lets a scan without a usable PE direction go uncorrected). MNI normalisation is skipped: PIE maps its atlas
    through its own cached SyN."""
    _separate(bids_dir, out_dir, work_dir)
    w = Path(work_dir) / f"sub-{label}"
    return _docker(image, [(bids_dir, "/data:ro"), (out_dir, "/out"), (w, "/work")], fs_license, gpus=cuda) + [
        "/data", "/out", "participant", "--participant-label", label, "-w", "/work",
        "--output-resolution", str(record["output_resolution"]), "--denoise-method", "dwidenoise",
        "--unringing-method", record["unringing_method"], "--hmc-model", "eddy", "--eddy-config", "/work/pie_eddy_params.json",
        "--pepolar-method", "TOPUP", "--use-syn-sdc", "warn", "--skip-anat-based-spatial-normalization",
        "--fs-license-file", "/license.txt", "--nprocs", str(nprocs), "--omp-nthreads", str(omp_nthreads),
        "--mem-mb", str(mem_mb), "--notrack", "--stop-on-first-crash"]


def qsirecon_command(qsiprep_dir, out_dir, work_dir, label, fs_license, nprocs=8, omp_nthreads=8, mem_mb=24000,
                     image=QSIRECON_IMAGE):
    """Docker argv for QSIRecon with ``RECON_SPEC`` (written to the participant's work dir by ``run_container``)."""
    _separate(qsiprep_dir, out_dir, work_dir)
    w = Path(work_dir) / f"sub-{label}"
    return _docker(image, [(qsiprep_dir, "/qsiprep:ro"), (out_dir, "/out"), (w, "/work")], fs_license) + [
        "/qsiprep", "/out", "participant", "--participant-label", label, "-w", "/work",
        "--recon-spec", "/work/pie_recon_spec.yaml", "--fs-license-file", "/license.txt", "--nprocs", str(nprocs),
        "--omp-nthreads", str(omp_nthreads), "--mem-mb", str(mem_mb), "--notrack", "--stop-on-first-crash"]


def run_container(cmd, image, out_dir, work_dir, files, expected=()):
    """Run one pinned container call with PIE's provenance (as ``fmriprep.run_fmriprep``): the local image's ID and repo
    digests, its --version and argv are recorded; a finished identical call is reused; a work dir used with other
    settings is refused. ``files`` ({name: dict}) are written into ``work_dir`` as JSON after those checks. PIE never
    pulls: the image must be present. A zero exit is execution completion, not a scientific QC pass."""
    inspect = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}} {{json .RepoDigests}}", image],
                             capture_output=True, text=True)
    if inspect.returncode:
        raise RuntimeError(f"{image} is not present locally and PIE does not pull implicitly: docker pull {image}")
    version = subprocess.check_output(["docker", "run", "--rm", "--pull", "never", image, "--version"], text=True,
                                      stderr=subprocess.STDOUT).strip()
    tag = image.split(":")[1].split("@")[0]
    if not re.search(r"(?<![\d.])" + re.escape(tag) + r"(?![\d.])", version):
        raise RuntimeError(f"Unexpected container version: {version}")
    identity = {"argv": cmd, "image": image, "image_inspect": inspect.stdout.strip(), "version": version, "files": files}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    work, record_dir = Path(work_dir), Path(out_dir) / "pie_provenance" / digest
    work.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    settings = work / "pie_identity.json"
    if settings.exists() and json.loads(settings.read_text()) != identity:
        raise ValueError("Work directory belongs to different settings; use a separate work directory")
    write_json(settings, identity)
    finished = record_dir / "execution_completed.json"
    if finished.exists():
        return json.loads(finished.read_text())
    for name, content in files.items():
        write_json(work / name, content)
    write_json(record_dir / "configuration.json", identity)
    started = datetime.now(timezone.utc)
    attempt = record_dir / ("attempt-" + started.strftime("%Y%m%dT%H%M%S%fZ") + ".log")
    with attempt.open("x") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    record = {"started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
              "returncode": result.returncode, "log": str(attempt), "identity_sha256": digest, "version": version,
              "image_inspect": identity["image_inspect"], "scientific_qc_pass": False}
    write_json(attempt.with_suffix(".json"), record)
    if result.returncode:
        raise RuntimeError(f"container failed ({result.returncode}); inspect {attempt}")
    missing = [str(p) for p in expected if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"container exited successfully but expected outputs are missing: {missing}")
    write_json(finished, record)
    return record


# ------------------------------------------------------------------------------------------ compare
def _find(root, pattern, folder):
    hits = [p for p in sorted(Path(root).rglob(pattern)) if p.parent.name == folder and "space-MNI" not in p.name]
    if len(hits) != 1:
        raise ValueError(f"expected one {folder}/{pattern} under {root}, found {len(hits)}")
    return hits[0]


def run_bounds(original_file):
    """(start, stop) of each source run in QSIPrep's concatenated DWI, from the confounds' ``original_file`` column."""
    f = np.asarray(original_file, dtype=str)
    starts = np.r_[0, np.flatnonzero(f[1:] != f[:-1]) + 1]
    if len(set(f[starts])) != len(starts):
        raise ValueError("source runs are not contiguous in the preprocessed DWI")
    return [(int(a), int(z)) for a, z in zip(starts, np.r_[starts[1:], len(f)])]


def recon_maps(recon_dir, label):
    """{(model, param): path} of the ``RECON_MAPS`` QSIRecon wrote for one participant on the preprocessed-DWI grid:
    <recon_dir>/derivatives/qsirecon-*/sub-<label>/[ses-*/]dwi/*_model-<m>_param-<p>[_desc-*]_dwimap.nii.gz."""
    maps = {}
    for p in sorted(Path(recon_dir).glob(f"derivatives/qsirecon-*/sub-{label}/**/dwi/*_dwimap.nii.gz")):
        m = re.search(r"_model-([A-Za-z0-9]+)_param-([A-Za-z0-9]+)_", p.name)
        if m and "space-MNI" not in p.name and (m[1], m[2]) in RECON_MAPS:
            if (m[1], m[2]) in maps:
                raise ValueError(f"two {m[1]} {m[2]} maps for sub-{label}")
            maps[m[1], m[2]] = p
    return maps


def compare_subject(patno, qsiprep_dir, fastsurfer_dir, recon_dir=None):
    """``dwiq_*`` for one participant: PIE's study nigral labels (CIT168 SNc/SNr through the cached T1 -> MNI SyN) pulled
    in one nearest-neighbour resampling onto QSIPrep's preprocessed-DWI grid, via a rigid QSIPrep T1w -> FastSurfer T1
    registration (so QSIPrep's own b0 -> T1w coregistration is the one compared). On QSIPrep's DWI: PIE's tensor rule and
    QC (``dwi.study_features``: the b = 1000 run, its own b0s; motion = QSIPrep's mean framewise displacement), and
    multi-shell free water (``freewater_qc.fit_multishell_checked``, b <= 2050) under the same region rule; and the
    QSIRecon ``RECON_MAPS`` (``dwiq_<model>_<param>_*``)."""
    from .freewater_qc import fit_multishell_checked

    label = str(int(patno))
    sub = Path(qsiprep_dir) / f"sub-{label}"
    nii = _find(sub, "*_desc-preproc_dwi.nii.gz", "dwi")
    stem = str(nii)[:-len(".nii.gz")]
    img = nib.load(nii)
    data = np.asarray(img.dataobj, dtype=np.float32)       # ponytail: whole series in memory (~1 GB at 2 mm)
    b, v = np.loadtxt(stem + ".bval").ravel(), np.loadtxt(stem + ".bvec").reshape(3, -1)
    conf = pd.read_csv(_find(sub, "*_desc-confounds_timeseries.tsv", "dwi"), sep="\t")
    t1q = nib.load(_find(sub, "*_desc-preproc_T1w.nii.gz", "anat"))
    brainq = np.asarray(nib.load(_find(sub, "*_desc-brain_mask.nii.gz", "anat")).dataobj) > 0
    mri = Path(fastsurfer_dir) / "mri"
    moving = nib.Nifti1Image(np.where(brainq, np.asarray(t1q.dataobj, np.float32), 0).astype(np.float32), t1q.affine)
    tx, metric = dwi.register_b0_to_t1(moving, nib.load(mri / "orig.mgz"), nib.load(mri / "mask.mgz"), sampling_seed=dwi.STUDY_SEED)
    codes = np.asarray(dwi.study_nigra_labels(nib.Nifti1Image(np.zeros(img.shape[:3], np.float32), img.affine), fastsurfer_dir, tx))
    motion = float(np.nanmean(pd.to_numeric(conf["framewise_displacement"], errors="coerce")))
    row = {"patno": int(patno), "dwiq_source": nii.name, "dwiq_n_volumes": int(data.shape[3]), "dwiq_reg_t1_t1_mi": metric,
           "dwiq_voxel_mm": float(np.linalg.norm(img.affine[:3, :3], axis=0).mean()), "dwiq_motion_metric": "qsiprep_mean_fd_mm"}
    study = dwi.study_features(data, b, v, img.affine, run_bounds(conf["original_file"]), codes,
                               dwi._b0_brain(data[..., b <= 50].mean(axis=3)), motion)
    row.update({"dwiq_" + k[len("dwis_"):]: val for k, val in study.items()})
    xyz, rois = dwi.study_rois(codes, img.affine)
    fw = fit_multishell_checked(data[tuple(xyz.T)], b, v)
    row.update(dwi.roi_means(np.where(fw["physical"], fw["f"], np.nan), rois, "dwiq_fw"))
    for (model, param), path in (sorted(recon_maps(recon_dir, label).items()) if recon_dir else []):
        m = nib.load(path)
        if m.shape[:3] != img.shape[:3] or not np.allclose(m.affine, img.affine, atol=1e-3):
            raise ValueError(f"{path.name} is not on the preprocessed-DWI grid")
        row.update(dwi.roi_means(np.asarray(m.dataobj, float)[tuple(xyz.T)], rois, f"dwiq_{model}_{param}"))
    return row


def agreement_stats(x, y):
    """``pie.stats.agreement.agreement`` of y against x (Lin's CCC, ICC(2,1), Bland-Altman bias and limits, Pearson r)."""
    from pie.stats.agreement import agreement as _agreement
    return _agreement(y, x)


def agreement(pie, qsi, pairs=None, qc=True):
    """Agreement per measure between PIE's ``dwis_*`` (``dwi --study``) and the ``dwiq_*`` table, over participants with
    both values (and, with ``qc``, both ``*_qc_pass``). ``pairs`` {dwiq column: PIE column} defaults to every
    ``dwiq_[tensor_]{fa,md,ad,rd}_<roi>`` against ``dwis_<metric>_<roi>``."""
    if "dwis_qc_pass" not in pie:
        raise ValueError("the PIE table lacks the study measure: run pie.imaging.dwi with --study")
    df = pie.merge(qsi, on="patno")
    if qc:
        df = df[df["dwis_qc_pass"].astype(str).eq("True") & df["dwiq_qc_pass"].astype(str).eq("True")]
    pairs = pairs or {c: re.sub(r"^dwiq_(?:tensor_)?", "dwis_", c) for c in qsi.columns
                      if re.fullmatch(r"dwiq_(?:tensor_)?(fa|md|ad|rd)_(sn|snc|snr)(_posterior|_anterior)?_(l|r|mean)", c)}
    rows = []
    for q, p in pairs.items():
        if p in df and q in df:
            x, y = pd.to_numeric(df[p], errors="coerce").to_numpy(), pd.to_numeric(df[q], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            rows.append({"pie": p, "qsiprep": q, **agreement_stats(x[ok], y[ok])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ CLI
def _export_job(args):
    patno, rows, bids_dir, sessions, work = args
    try:
        r = export_subject(bids_dir, patno, rows, sessions, work)
        return {"patno": patno, "error": "", **{k: r[k] for k in ("session", "reverse_pe_fmap", "pe_known", "output_resolution",
                                                                   "unringing_method", "t1_gap_days")}, "n_dwi_runs": len(r["dwi"])}
    except Exception as e:  # keep the batch going
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _compare_job(args):
    patno, qsiprep_dir, fs_dir, recon_dir = args
    try:
        return dict(compare_subject(patno, qsiprep_dir, fs_dir, recon_dir), error="")
    except Exception as e:  # keep the batch going
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main(argv=None):
    import argparse

    from .batch import add_common_args, done_subjects, fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows, t1_session_dates
    from .freesurfer import LICENSE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    ex = add_common_args(sub.add_parser("export", help="PIE-selected DWI sessions -> BIDS (use PIE's dwi work dir: same series index)"))
    ex.add_argument("--bids-dir", required=True)
    stages = {"run": sub.add_parser("run", help="QSIPrep per exported participant"),
              "recon": sub.add_parser("recon", help="QSIRecon (PIE's tensor + NODDI spec) per preprocessed participant")}
    stages["run"].add_argument("--bids-dir", required=True)
    stages["run"].add_argument("--cuda", action="store_true", help="eddy_cuda (docker --gpus all: needs the NVIDIA container toolkit)")
    stages["run"].add_argument("--output-resolution", type=float, help="override the export's finest native voxel size (mm)")
    stages["recon"].add_argument("--qsiprep-dir", required=True)
    for name, p in stages.items():
        p.add_argument("--output-dir", required=True)
        p.add_argument("--work-dir", required=True)
        p.add_argument("--participants", nargs="+", help="PATNOs (default: every exported / preprocessed participant)")
        p.add_argument("--fs-license", default=LICENSE, help="FreeSurfer license (default $PIE_FS_LICENSE, else $FS_LICENSE)")
        p.add_argument("--nprocs", type=int, default=8)
        p.add_argument("--omp-nthreads", type=int, default=8)
        p.add_argument("--mem-mb", type=int, default=24000)
        p.add_argument("--image", default=QSIPREP_IMAGE if name == "run" else QSIRECON_IMAGE)
        p.add_argument("--dry-run", action="store_true", help="print each participant's argv without running")
    cp = sub.add_parser("compare", help="dwiq_* on QSIPrep's output in PIE's study nigral ROIs; agreement with dwis_*")
    cp.add_argument("--qsiprep-dir", required=True)
    cp.add_argument("--recon-dir", help="QSIRecon output dir (adds the dwiq_<model>_<param>_* columns)")
    cp.add_argument("--sessions", required=True)
    cp.add_argument("--fastsurfer-dir", required=True)
    cp.add_argument("--work-dir", required=True, help="-> dwiq_features.csv, dwiq_agreement.csv")
    cp.add_argument("--participants", nargs="+")
    cp.add_argument("--pie-features", help="PIE's dwi_features.csv from a --study run")
    cp.add_argument("--workers", type=int, default=2)
    cp.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args(argv)

    if a.stage == "export":
        work = Path(a.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        idx = load_index(work / "dwi_index.csv", a.zips, dwi._flag_dwi)      # dwi.main's selection, series by series
        idx = idx[idx["selected"]]
        fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
        t1d = t1_session_dates(a.sessions, fs)
        out_csv = work / "qsiprep_export.csv"
        done = done_subjects(out_csv, a.retry_errors)
        jobs = [(int(p), session_rows(g, t1d.get(int(p))), a.bids_dir, a.sessions, str(work))
                for p, g in idx.groupby("patno") if p not in done and int(p) in fs]
        run_batch(filter_jobs(jobs, a.patnos, a.limit), _export_job, out_csv, workers=a.workers, pid_file=a.pid_file)
        return
    if a.stage == "compare":
        fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
        labels = a.participants or sorted(p.name[4:] for p in Path(a.qsiprep_dir).glob("sub-*") if p.is_dir())
        work = Path(a.work_dir)
        out_csv = work / "dwiq_features.csv"
        done = done_subjects(out_csv, a.retry_errors)
        jobs = [(int(x), a.qsiprep_dir, fs[int(x)], a.recon_dir) for x in labels if int(x) not in done and int(x) in fs]
        run_batch(jobs, _compare_job, out_csv, workers=a.workers)
        if a.pie_features:
            stats = agreement(pd.read_csv(a.pie_features), pd.read_csv(out_csv))
            stats.to_csv(work / "dwiq_agreement.csv", index=False)
            print(stats.to_string(index=False), flush=True)
        return
    failed = []
    if a.stage == "run":
        records = {p.name[4:-5]: json.loads(p.read_text()) for p in sorted(Path(a.bids_dir, "sourcedata").glob("sub-*.json"))}
        labels = a.participants or sorted(records)
    else:
        labels = a.participants or sorted(p.name[4:] for p in Path(a.qsiprep_dir).glob("sub-*") if p.is_dir())
    for label in labels:
        try:
            if a.stage == "run":
                rec = dict(records[label], **({"output_resolution": a.output_resolution} if a.output_resolution else {}))
                cmd = qsiprep_command(a.bids_dir, a.output_dir, a.work_dir, label, rec, a.fs_license, a.nprocs, a.omp_nthreads,
                                      a.mem_mb, a.cuda, a.image)
                files, expected = {"pie_eddy_params.json": dict(EDDY_PARAMS, use_cuda=a.cuda)}, [Path(a.output_dir) / f"sub-{label}.html"]
            else:
                cmd = qsirecon_command(a.qsiprep_dir, a.output_dir, a.work_dir, label, a.fs_license, a.nprocs, a.omp_nthreads,
                                       a.mem_mb, a.image)
                files, expected = {"pie_recon_spec.yaml": RECON_SPEC}, []
            if a.dry_run:
                print(json.dumps({"participant": label, "argv": cmd, "files": sorted(files)}), flush=True)
                continue
            print(json.dumps(run_container(cmd, a.image, a.output_dir, Path(a.work_dir) / f"sub-{label}", files, expected)), flush=True)
        except Exception as e:  # one participant's failure leaves the others to run
            failed.append(label)
            print(f"sub-{label}: {type(e).__name__}: {e}", flush=True)
    if failed:
        raise SystemExit(f"{a.stage} failed for {len(failed)} participant(s): {' '.join(failed)}")


if __name__ == "__main__":
    main()
