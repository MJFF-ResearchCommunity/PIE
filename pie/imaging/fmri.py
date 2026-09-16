"""Outcome-independent fMRI conversion and technical-pilot measurements.

Not a complete connectivity pipeline. Preserve every converter output, distinguish
short EPI references by actual dimensions, and never guess phase encoding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

import nibabel as nib
import numpy as np

from .convert import DCM2NIIX


class DICOMConversionError(RuntimeError):
    """Converter execution failed; log remains available for acquisition review."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Capture the implementation that this process actually loaded, even if a
# developer edits the source file while a long conversion is still running.
_CONVERSION_CODE_SHA256 = sha256(__file__)

# Power-style FD: rotations become arc length on a sphere of this radius; the pilot
# counts frames above the threshold as high motion. qc.json records both values.
FD_RADIUS_MM = 50
FD_THRESHOLD_MM = 0.3


def _device(path):
    return os.stat(path).st_dev


def _check_free_space(output_root, scratch_root, dicom_bytes):
    """Raw DICOM copies (1x + 1 GiB) go to scratch, converter staging (3x + 1 GiB)
    to output_root. Roots on one filesystem draw on the same free space, so sum."""
    need = {}
    for path, amount in ((output_root, 3 * dicom_bytes + 1024**3), (scratch_root, dicom_bytes + 1024**3)):
        first, total = need.get(_device(path), (path, 0))
        need[_device(path)] = (first, total + amount)
    for path, amount in need.values():
        if shutil.disk_usage(path).free < amount:
            raise OSError(f"Insufficient free space on {path}: {amount} bytes required")


def write_json(path, value):
    """Publish a complete metadata file; reject NaNs instead of concealing them."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
        temporary = Path(out.name)
    temporary.replace(path)


def command(args, log, *, env=None, timeout=3600, cwd=None):
    with Path(log).open("w") as stream:
        result = subprocess.run(list(map(str, args)), stdout=stream,
                                stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=cwd)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def classify_run(shape, tr, *, min_volumes=100, min_seconds=300):
    if len(shape) != 4 or shape[3] < 2:
        return "not_functional_4d"
    if tr is None or not np.isfinite(tr) or tr <= 0:
        return "unknown_timing"
    if shape[3] <= 20:
        return "short_reference_candidate"
    if shape[3] < min_volumes or shape[3] * tr < min_seconds:
        return "insufficient_duration"
    return "rest_candidate"


def inspect_nifti(nifti, sidecar):
    img = nib.load(nifti)
    # dcm2niix can emit raw control characters inside string fields (e.g. comments);
    # strict=False accepts them instead of failing the whole conversion.
    meta = json.loads(Path(sidecar).read_text(), strict=False)
    tr = meta.get("RepetitionTime")
    if tr is not None:
        tr = float(tr)
        if not np.isfinite(tr) or tr <= 0:
            raise ValueError("Invalid sidecar TR")
    units = img.header.get_xyzt_units()
    header_tr = None
    if img.ndim == 4 and units[1] in ("sec", "msec", "usec"):
        factor = {"sec": 1, "msec": 1e-3, "usec": 1e-6}[units[1]]
        header_tr = float(img.header.get_zooms()[3]) * factor
        if tr is not None and not np.isclose(tr, header_tr, atol=1e-4, rtol=1e-4):
            raise ValueError(f"NIfTI/JSON TR disagreement: {header_tr} versus {tr}")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-9:
        raise ValueError("Invalid spatial affine")
    return {"shape": list(img.shape), "voxel_sizes": list(map(float, img.header.get_zooms()[:3])),
            "axis_codes": list(nib.aff2axcodes(img.affine)), "affine": img.affine.tolist(),
            "spatial_units": units[0], "tr_seconds": tr, "nifti_tr_seconds": header_tr,
            "duration_seconds": img.shape[3] * tr if img.ndim == 4 and tr else None,
            "run_class": classify_run(img.shape, tr),
            "phase_encoding": meta.get("PhaseEncodingDirection"),
            "total_readout_time": meta.get("TotalReadoutTime"),
            "slice_timing_count": len(meta.get("SliceTiming", [])),
            "manufacturer": meta.get("Manufacturer"),
            "image_type": meta.get("ImageType"),
            "series_description": meta.get("SeriesDescription")}


def convert_archive_series(record, output_root, *, dcm2niix=DCM2NIIX, archive_cache=None,
                           scratch_root=None, extra_args=()):
    """CRC-check selected DICOM members, convert, and retain ALL output images.

    Temporary raw copies live under output_root; the archive is never modified.
    Existing completed outputs require matching source metadata and output hashes.
    Incomplete directories are not reused or silently overwritten.
    Optional ``ZipArchiveCache`` reuses a bounded index, not extracted data.
    Converter failures retain a diagnostic log outside temporary scratch.
    ``extra_args`` passes additional converter switches (recorded in the
    saved command), e.g. ``("-m", "y")`` to merge split temporal volumes.
    ``scratch_root`` optionally places temporary raw DICOMs on a different
    configured filesystem. Output staging stays on output_root for atomic
    publication. A missing/full scratch location is an error, never a fallback.
    """
    patno, image_id = str(record["PATNO"]), record["image_id"]
    if not re.fullmatch(r"\d+", patno) or not re.fullmatch(r"I\d+", image_id):
        raise ValueError("Invalid participant/image identifier")
    prefix = record["series_prefix"]
    parts = PurePosixPath(prefix).parts
    if not prefix.endswith("/") or ".." in parts or PurePosixPath(prefix).is_absolute():
        raise ValueError("Unsafe archive prefix")
    if parts[-1] != image_id or patno not in parts:
        raise ValueError("Archive prefix/participant/image mismatch")
    archive = Path(record["archive"]).resolve(strict=True)
    stat = archive.stat()
    identity = {"archive": str(archive), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "series_prefix": prefix, "PATNO": patno, "image_id": image_id}
    root = Path(output_root).resolve(strict=True)
    scratch = root if scratch_root is None else Path(scratch_root).resolve(strict=True)
    if not scratch.is_dir():
        raise ValueError('Configured conversion scratch must be a directory')
    target = root / patno / image_id
    completion = target / "conversion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["source"] != identity:
            raise ValueError("Existing conversion source changed")
        for row in saved["outputs"]:
            for key in ("nifti", "sidecar"):
                if sha256(target / row[key]) != row[key + "_sha256"]:
                    raise ValueError("Existing conversion failed checksum verification")
        return saved
    if target.exists():
        raise FileExistsError(f"Incomplete output must be reviewed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    from .archives import ZipArchiveCache
    from contextlib import nullcontext
    with (ZipArchiveCache() if archive_cache is None else nullcontext(archive_cache)) as cache, cache.open(archive) as z:
        members = [i for i in z.infolist() if i.filename.startswith(prefix)
                   and i.filename.lower().endswith(".dcm") and not i.is_dir()]
        if not members or any(i.file_size == 0 for i in members):
            raise ValueError("Missing/empty DICOM members")
        if len({i.filename for i in members}) != len(members):
            raise ValueError("Duplicate archive member paths")
        if record.get("dicom_entries") and len(members) != int(record["dicom_entries"]):
            raise ValueError("Archive member count differs from inventory")
        _check_free_space(root, scratch, sum(i.file_size for i in members))
        with tempfile.TemporaryDirectory(prefix="fmri-convert-", dir=root) as temporary, \
                tempfile.TemporaryDirectory(prefix="fmri-dicom-", dir=scratch) as raw_temporary:
            temp = Path(temporary)
            raw, out = Path(raw_temporary), temp / "converted"
            out.mkdir()
            source_digest = hashlib.sha256()
            for index, member in enumerate(members):
                # Deliberately ignore member filenames; no traversal or basename collision.
                with z.open(member) as src, (raw / f"{index:08d}.dcm").open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                source_digest.update(json.dumps([member.filename, member.CRC,
                                                  member.file_size]).encode())
            import pydicom
            header = pydicom.dcmread(raw / "00000000.dcm", stop_before_pixels=True,
                                    specific_tags=["StudyInstanceUID", "SeriesInstanceUID"])
            # Ignore per-user defaults and use internal single-process compression.
            cmd = [dcm2niix, "-g", "i", "-z", "i", "-b", "y", "-ba", "y", *map(str, extra_args),
                   "-f", image_id + "_%s", "-o", out, raw]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

            def retain_failure(error):
                failure = root / "conversion_failures" / patno / image_id
                failure.mkdir(parents=True, exist_ok=True)
                log = failure / (temp.name + ".log")
                if (out / "dcm2niix.log").exists():
                    shutil.copyfile(out / "dcm2niix.log", log)
                # Retain converter products for technical diagnosis, never publish
                # them as completed/scientifically usable acquisition outputs.
                diagnostics = failure / temp.name
                out.rename(diagnostics)
                write_json(failure / (temp.name + ".json"), {
                    "source": identity, "command": list(map(str, cmd)),
                    "error": str(error), "log": str(log),
                    "unvalidated_outputs": str(diagnostics),
                    "selected_members_crc_checked": True,
                    "source_member_index_sha256": source_digest.hexdigest(),
                    "converter_sha256": sha256(dcm2niix)})
                raise DICOMConversionError(f"DICOM conversion failed; retained diagnostic: {log}") from error

            try:
                command(cmd, out / "dcm2niix.log", env=env)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                retain_failure(error)
            images = sorted(out.glob("*.nii.gz"))
            if not images:
                retain_failure(ValueError("Converter produced no NIfTI"))
            outputs = []
            for nii in images:
                js = nii.with_suffix("").with_suffix(".json")
                if not js.exists():
                    retain_failure(ValueError(f"Converter output lacks required metadata sidecar: {nii.name}"))
                try:
                    inspection = inspect_nifti(nii, js)
                except (ValueError, nib.filebasedimages.ImageFileError) as error:
                    retain_failure(error)
                outputs.append({"nifti": nii.name, "sidecar": js.name,
                                "nifti_sha256": sha256(nii), "sidecar_sha256": sha256(js),
                                **inspection})
            now = archive.stat()
            if (now.st_size, now.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError("Archive changed during conversion")
            result = {"source": identity, "dicom_count": len(members),
                      "selected_members_crc_checked": True,
                      "source_member_index_sha256": source_digest.hexdigest(),
                      "study_uid": str(header.get("StudyInstanceUID", "")),
                      "series_uid": str(header.get("SeriesInstanceUID", "")),
                      "converter_sha256": sha256(dcm2niix),
                      "code_sha256": _CONVERSION_CODE_SHA256, "command": list(map(str, cmd)),
                      "outputs": outputs}
            write_json(out / "conversion.json", result)
            out.rename(target)
    return result


def alignment_pilot(t1, motion_dir, output, *, fsl_dir):
    """Initial rigid BOLD-to-T1 feasibility check; not BBR or final registration."""
    output, motion_dir = Path(output), Path(motion_dir)
    if output.exists():
        raise FileExistsError(f"Will not overwrite alignment: {output}")
    if nib.load(t1).ndim != 3:
        raise ValueError("Anatomical pilot requires unambiguous 3D T1")
    output.mkdir(parents=True)
    env = dict(os.environ, FSLDIR=str(fsl_dir), FSLOUTPUTTYPE="NIFTI_GZ",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    fsl = Path(fsl_dir) / "bin"
    # BET's shell wrapper expands filenames unquoted internally. Keep its
    # arguments relative and space-free; the working directory stays external.
    shutil.copyfile(t1, output / "t1_input.nii.gz")
    command([fsl / "bet", "t1_input.nii.gz", "t1_brain", "-R", "-f", "0.5", "-m"],
            output / "bet_t1.log", env=env, cwd=output)
    command([fsl / "flirt", "-in", motion_dir / "mean_brain.nii.gz", "-ref",
             output / "t1_brain.nii.gz", "-out", output / "bold_in_t1.nii.gz",
             "-omat", output / "bold_to_t1.mat", "-dof", "6", "-cost", "normmi"],
            output / "flirt.log", env=env)
    command([fsl / "flirt", "-in", motion_dir / "mean_brain_mask.nii.gz", "-ref",
             output / "t1_brain.nii.gz", "-applyxfm", "-init", output / "bold_to_t1.mat",
             "-interp", "nearestneighbour", "-out", output / "bold_mask_in_t1.nii.gz"],
            output / "mask_transform.log", env=env)
    t1img = nib.load(output / "t1_brain.nii.gz")
    t1mask = np.asarray(nib.load(output / "t1_brain_mask.nii.gz").dataobj) > 0
    boldmask = np.asarray(nib.load(output / "bold_mask_in_t1.nii.gz").dataobj) > 0
    if not t1mask.any() or not boldmask.any():
        raise ValueError("Empty brain mask after alignment")
    from .qc import montage
    canonical = nib.as_closest_canonical(t1img)
    canonical_mask = nib.as_closest_canonical(nib.Nifti1Image(boldmask.astype(np.uint8), t1img.affine))
    montage(np.asarray(canonical.dataobj), [(np.asarray(canonical_mask.dataobj) > 0, "cyan")],
            output / "alignment.png", title="Pilot rigid BOLD mask on T1 (not final BBR)")
    result = {"scope": "initial_rigid_alignment_requires_visual_review",
              "t1_sha256": sha256(t1), "code_sha256": sha256(__file__),
              "mask_dice": float(2 * (t1mask & boldmask).sum() / (t1mask.sum() + boldmask.sum())),
              "t1_mask_covered_fraction": float((t1mask & boldmask).sum() / t1mask.sum()),
              "transform": np.loadtxt(output / "bold_to_t1.mat").tolist(),
              "susceptibility_corrected": False, "boundary_based_registration": False,
              "nigral_coverage_verified": False}
    write_json(output / "alignment.json", result)
    return result


def phase_encoding_pair(first, second):
    """Return explicit reasons a reverse-PE candidate cannot yet be used.

    Geometry/readout/TE must be checked after conversion. Descriptions do not
    establish polarity, and matching metadata alone does not establish usable SDC.
    """
    reasons = []
    p, q = first.get("phase_encoding"), second.get("phase_encoding")
    valid = {"i", "i-", "j", "j-", "k", "k-"}
    if p not in valid or q not in valid:
        reasons.append("missing_verified_phase_encoding")
    elif p[0] != q[0] or p == q:
        reasons.append("not_opposite_phase_encoding")
    for row in (first, second):
        t = row.get("total_readout_time")
        if t is None or not np.isfinite(t) or t <= 0:
            reasons.append("missing_verified_readout_time")
    if first["shape"][:3] != second["shape"][:3] or not np.allclose(
            first["affine"], second["affine"], atol=1e-4, rtol=0):
        reasons.append("geometry_requires_explicit_reconciliation")
    return sorted(set(reasons))


def framewise_displacement(parameters, radius_mm=FD_RADIUS_MM):
    """Power-style FD from MCFLIRT rotations (radians), then translations (mm)."""
    params = np.asarray(parameters, dtype=float)
    if params.ndim != 2 or params.shape[1] != 6 or not np.isfinite(params).all():
        raise ValueError("Expected finite N x 6 MCFLIRT parameters")
    if len(params) < 2 or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Need multiple volumes and a positive rotation radius")
    delta = np.abs(np.diff(params, axis=0))
    return np.r_[0.0, radius_mm * delta[:, :3].sum(axis=1) + delta[:, 3:].sum(axis=1)]


def render_pilot_review(motion_dir, alignment_dir, output, *, title=""):
    """Multi-slice anatomy/BOLD and motion review, without assigning a QC pass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion_dir, alignment_dir, output = map(Path, (motion_dir, alignment_dir, output))
    output.mkdir(parents=True, exist_ok=True)

    def load(path):
        img = nib.as_closest_canonical(nib.load(path))
        return np.asarray(img.dataobj), img.header.get_zooms()[:3]

    mean, mean_zoom = load(motion_dir / "mean.nii.gz")
    native_mask, _ = load(motion_dir / "mean_brain_mask.nii.gz")
    t1, zoom = load(alignment_dir / "t1_input.nii.gz")
    t1mask, _ = load(alignment_dir / "t1_brain_mask.nii.gz")
    bold, _ = load(alignment_dir / "bold_in_t1.nii.gz")
    boldmask, _ = load(alignment_dir / "bold_mask_in_t1.nii.gz")

    def views(data, mask, spacing):
        pts = np.argwhere(mask > 0)
        low, high = pts.min(axis=0), pts.max(axis=0)
        centre = np.round((low + high) / 2).astype(int)
        axes = [(2, int(low[2] + f * (high[2] - low[2]))) for f in (.2, .45, .7)]
        axes += [(1, centre[1]), (0, centre[0])]
        result = []
        for axis, index in axes:
            remaining = [i for i in range(3) if i != axis]
            result.append((np.take(data, index, axis=axis).T,
                           spacing[remaining[1]] / spacing[remaining[0]], axis, index))
        return result

    fig, axs = plt.subplots(4, 5, figsize=(17, 12))
    rows = [(mean, native_mask, mean_zoom), (t1, t1mask, zoom),
            (t1, t1mask, zoom), (bold, t1mask, zoom)]
    labels = ["Native mean / brain mask", "T1 / T1 mask", "T1 / BOLD mask", "Aligned BOLD / T1 mask"]
    contours = [native_mask, t1mask, boldmask, t1mask]
    for row, (data, mask, spacing) in enumerate(rows):
        positive = data[data > 0]
        vmax = np.percentile(positive, 99) if positive.size else 1
        for col, (sl, aspect, axis, index) in enumerate(views(data, mask, spacing)):
            ax = axs[row, col]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax, aspect=aspect)
            contour = np.take(contours[row], index, axis=axis).T
            if contour.any() and not contour.all():
                ax.contour(contour, levels=[.5], colors="cyan" if row == 2 else "lime", linewidths=.6)
            ax.set_title(f"{labels[row]} | {'xyz'[axis]}={index}", fontsize=8)
            ax.axis("off")
    fig.suptitle(title + " — uncorrected rigid/motion pilot; no QC pass implied", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "spatial_review.png", dpi=110)
    plt.close(fig)

    qc = json.loads((motion_dir / "qc.json").read_text())
    values = np.loadtxt(motion_dir / "motion_qc.tsv", skiprows=1)
    time = np.arange(len(values)) * qc["tr_seconds"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axs[0].plot(time[1:], values[1:, 0], linewidth=.8)
    axs[0].axhline(qc.get("fd_threshold_mm", FD_THRESHOLD_MM), color="red", linestyle="--", linewidth=.8)
    axs[0].set_ylabel("FD (mm)")
    axs[1].plot(time[1:], values[1:, 1], linewidth=.8)
    axs[1].set_ylabel("Raw DVARS")
    axs[1].set_xlabel("Seconds after initial-volume discard")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "motion_review.png", dpi=110)
    plt.close(fig)
    return {"spatial_review": str(output / "spatial_review.png"),
            "motion_review": str(output / "motion_review.png"), "visual_qc_pass_assigned": False}


def verify_pilot_metrics(motion_dir, alignment_dir):
    """Independently recompute saved diagnostics; no preprocessing rerun."""
    motion_dir, alignment_dir = Path(motion_dir), Path(alignment_dir)
    qc = json.loads((motion_dir / "qc.json").read_text())
    alignment = json.loads((alignment_dir / "alignment.json").read_text())
    # Verify with the values the record was written with; records that predate
    # these keys were written with the current defaults.
    radius = qc.get("fd_radius_mm", FD_RADIUS_MM)
    threshold = qc.get("fd_threshold_mm", FD_THRESHOLD_MM)
    params = np.loadtxt(motion_dir / "motion.par")
    delta = np.abs(np.diff(params, axis=0))
    fd = np.r_[0, radius * delta[:, :3].sum(axis=1) + delta[:, 3:].sum(axis=1)]
    tab = np.loadtxt(motion_dir / "motion_qc.tsv", skiprows=1)
    np.testing.assert_allclose(fd, tab[:, 0], rtol=1e-12)
    array = np.asarray(nib.load(motion_dir / "motion.nii.gz").dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(motion_dir / "mean_brain_mask.nii.gz").dataobj) > 0
    samples = array[mask].astype(np.float64)
    dvars = np.sqrt(np.square(np.diff(samples, axis=1)).mean(axis=0))
    np.testing.assert_allclose(dvars, tab[1:, 1], rtol=1e-12)
    sd = samples.std(axis=1, ddof=1)
    variable = sd > np.finfo(float).eps
    tsnr = np.median(samples.mean(axis=1)[variable] / sd[variable])
    good = fd[1:] <= threshold
    checks = [(fd[1:].mean(), "mean_fd_mm_excluding_first"), ((~good).mean(), "fraction_fd_gt_0p3"),
              (tsnr, "median_native_motion_corrected_tsnr")]
    # Legacy key: records written before the rename also counted frame 0 (FD fixed at 0).
    if "seconds_fd_le_0p3_excluding_first" in qc:
        checks.append((good.sum() * qc["tr_seconds"], "seconds_fd_le_0p3_excluding_first"))
    else:
        checks.append(((fd <= threshold).sum() * qc["tr_seconds"], "seconds_fd_le_0p3"))
    for value, key in checks:
        np.testing.assert_allclose(value, qc[key], rtol=1e-12)
    t1mask = np.asarray(nib.load(alignment_dir / "t1_brain_mask.nii.gz").dataobj) > 0
    boldmask = np.asarray(nib.load(alignment_dir / "bold_mask_in_t1.nii.gz").dataobj) > 0
    np.testing.assert_allclose(2 * (t1mask & boldmask).sum() / (t1mask.sum() + boldmask.sum()),
                               alignment["mask_dice"])
    matrix = np.loadtxt(alignment_dir / "bold_to_t1.mat")
    # FLIRT's float precision and text serialization do not support 1e-8 checks.
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
    longest = streak = 0
    for flag in good:
        streak = streak + 1 if flag else 0
        longest = max(longest, streak)
    return {"fd_dvars_tsnr_alignment_recomputed": True,
            "low_fd_seconds_excluding_first": float(good.sum() * qc["tr_seconds"]),
            "longest_consecutive_low_fd_seconds_excluding_first": longest * qc["tr_seconds"],
            "high_fd_frames_excluding_first": int((~good).sum()),
            "rigid_rotation_check_atol": 1e-6,
            "motion_sha256": sha256(motion_dir / "motion.nii.gz")}


def validate_motion_resume(nifti, output, discard):
    """Verify legacy completed MCFLIRT products before resuming later QC.

    Compare the actual trimmed voxels with the source, and require the complete
    finite motion image, parameters and per-volume transforms. This is an
    explicit technical recovery check, not a new motion estimate.
    """
    output = Path(output)
    source = nib.load(nifti)
    trimmed = nib.load(output / "trimmed.nii.gz")
    moved = nib.load(output / "motion.nii.gz")
    expected_shape = source.shape[:3] + (source.shape[3] - discard,)
    for img in (trimmed, moved):
        if img.shape != expected_shape or not np.allclose(img.affine, source.affine):
            raise ValueError("Resume image geometry differs from source")
        if not np.isfinite(np.asarray(img.dataobj)).all():
            raise ValueError("Nonfinite resume image")
    # Reproduce the saved representation: a copied integer NIfTI header can
    # quantize float32 data on write. Comparing to unsaved floats would falsely
    # reject the original output. Decode both with the same writer/header.
    with tempfile.TemporaryDirectory(prefix="resume-verify-", dir=output) as temporary:
        expected = Path(temporary) / "expected.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(source.dataobj, dtype=np.float32)[..., discard:],
                               source.affine, source.header), expected)
        if not np.array_equal(np.asarray(trimmed.dataobj), np.asarray(nib.load(expected).dataobj)):
            raise ValueError("Resume trimmed data differ from source")
    params = np.loadtxt(output / "motion.par")
    if params.shape != (expected_shape[-1], 6) or not np.isfinite(params).all():
        raise ValueError("Incomplete resume motion parameters")
    matrices = sorted((output / "motion.mat").glob("MAT_*"))
    if len(matrices) != expected_shape[-1]:
        raise ValueError("Incomplete resume motion transforms")
    for matrix in matrices:
        value = np.loadtxt(matrix)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("Invalid resume motion transform")
    return {"trimmed_matches_source": True, "motion_sha256": sha256(output / "motion.nii.gz"),
            "parameters_sha256": sha256(output / "motion.par"), "transform_count": len(matrices)}


def motion_pilot(nifti, sidecar, output, *, fsl_dir, discard_seconds=10, resume=False):
    """Bounded native-space motion/QC pilot, NOT fully preprocessed BOLD.

    No slice timing, susceptibility correction, nuisance regression, filtering,
    atlas connectivity, or label access. Raw/native tSNR is a technical diagnostic.
    """
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"Will not overwrite pilot: {output}")
    info = inspect_nifti(nifti, sidecar)
    if info["run_class"] != "rest_candidate":
        raise ValueError("Not a sufficiently long resting-state candidate")
    tr = info["tr_seconds"]
    discard = math.ceil(discard_seconds / tr)
    img = nib.load(nifti)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite image intensities")
    if data.shape[-1] - discard < 100:
        raise ValueError("Too few volumes after initial-volume removal")
    if resume and (output / "qc.json").exists():
        raise FileExistsError("Completed motion QC must not be overwritten")
    recovery = validate_motion_resume(nifti, output, discard) if resume else None
    output.mkdir(parents=True, exist_ok=resume)
    trimmed = output / "trimmed.nii.gz"
    if not resume:
        nib.save(nib.Nifti1Image(data[..., discard:], img.affine, img.header), trimmed)
    del data
    env = dict(os.environ, FSLDIR=str(fsl_dir), FSLOUTPUTTYPE="NIFTI_GZ",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    prefix = output / "motion"
    if not resume:
        command([Path(fsl_dir) / "bin/mcflirt", "-in", trimmed, "-out", prefix,
                 "-plots", "-mats", "-rmsrel", "-rmsabs"], output / "mcflirt.log", env=env)
    moved = nib.load(output / "motion.nii.gz")
    array = np.asarray(moved.dataobj, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite motion-corrected image")
    mean = array.mean(axis=-1)
    nib.save(nib.Nifti1Image(mean, moved.affine), output / "mean.nii.gz")
    if resume and (output / "bet.log").exists():
        backup = output / "bet.before_resume.log"
        if backup.exists():
            raise FileExistsError("Previous recovery log already exists; review before retrying")
        (output / "bet.log").rename(backup)
    command([Path(fsl_dir) / "bin/bet", "mean.nii.gz", "mean_brain",
             "-f", "0.3", "-m"], output / "bet.log", env=env, cwd=output)
    mask = np.asarray(nib.load(output / "mean_brain_mask.nii.gz").dataobj) > 0
    if mask.sum() < 1000:
        raise ValueError("Implausibly small pilot brain mask")
    params = np.loadtxt(output / "motion.par")
    if params.shape != (array.shape[-1], 6):
        raise ValueError("Motion parameter/image length mismatch")
    fd = framewise_displacement(params, FD_RADIUS_MM)
    low = fd[1:] <= FD_THRESHOLD_MM   # frame 0 has no predecessor; its FD is fixed at 0
    samples = array[mask].astype(np.float64)
    sd = samples.std(axis=1, ddof=1)
    variable = sd > np.finfo(float).eps
    if not variable.any():
        raise ValueError("No temporally varying brain voxels")
    tsnr = samples[variable].mean(axis=1) / sd[variable]
    dvars = np.r_[np.nan, np.sqrt(np.mean(np.diff(samples, axis=1)**2, axis=0))]
    np.savetxt(output / "motion_qc.tsv", np.c_[fd, dvars], delimiter="\t",
               header="framewise_displacement_mm\tdvars_raw", comments="")
    result = {"scope": "technical_motion_pilot_not_analysis_ready",
              "resumed_existing_motion": recovery,
              "source_sha256": sha256(nifti), "code_sha256": sha256(__file__),
              "discarded_initial_volumes": discard, "discard_rule_seconds": discard_seconds,
              "tr_seconds": tr, "remaining_volumes": len(fd), "brain_mask_voxels": int(mask.sum()),
              "mean_fd_mm_excluding_first": float(fd[1:].mean()),
              "median_fd_mm_excluding_first": float(np.median(fd[1:])),
              "max_fd_mm": float(fd.max()), "fd_radius_mm": FD_RADIUS_MM, "fd_threshold_mm": FD_THRESHOLD_MM,
              "fraction_fd_gt_0p3": float((~low).mean()),
              "seconds_fd_le_0p3_excluding_first": float(low.sum() * tr),
              "median_native_motion_corrected_tsnr": float(np.median(tsnr)),
              "median_dvars_raw": float(np.median(dvars[1:])),
              "susceptibility_corrected": False, "slice_timing_corrected": False,
              "nuisance_regressed": False, "connectivity_measured": False,
              "outcomes_read": False}
    write_json(output / "qc.json", result)
    return result
