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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def command(args, log, *, env=None, timeout=3600):
    with Path(log).open("w") as stream:
        result = subprocess.run(list(map(str, args)), stdout=stream,
                                stderr=subprocess.STDOUT, env=env, timeout=timeout)
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
    meta = json.loads(Path(sidecar).read_text())
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


def convert_archive_series(record, output_root, *, dcm2niix=DCM2NIIX):
    """CRC-check selected DICOM members, convert, and retain ALL output images.

    Temporary raw copies live under output_root; the archive is never modified.
    Existing completed outputs require matching source metadata and output hashes.
    Incomplete directories are not reused or silently overwritten.
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
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if i.filename.startswith(prefix)
                   and i.filename.lower().endswith(".dcm") and not i.is_dir()]
        if not members or any(i.file_size == 0 for i in members):
            raise ValueError("Missing/empty DICOM members")
        if len({i.filename for i in members}) != len(members):
            raise ValueError("Duplicate archive member paths")
        if record.get("dicom_entries") and len(members) != int(record["dicom_entries"]):
            raise ValueError("Archive member count differs from inventory")
        required = sum(i.file_size for i in members) * 4 + 2 * 1024**3
        if shutil.disk_usage(root).free < required:
            raise OSError("Insufficient working space for selected series")
        with tempfile.TemporaryDirectory(prefix="fmri-convert-", dir=root) as temporary:
            temp = Path(temporary)
            raw, out = temp / "dicom", temp / "converted"
            raw.mkdir()
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
            cmd = [dcm2niix, "-g", "i", "-z", "i", "-b", "y", "-ba", "y", "-f", image_id + "_%s",
                   "-o", out, raw]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            command(cmd, out / "dcm2niix.log", env=env)
            images = sorted(out.glob("*.nii.gz"))
            if not images:
                raise ValueError("Converter produced no NIfTI")
            outputs = []
            for nii in images:
                js = nii.with_suffix("").with_suffix(".json")
                if not js.exists():
                    raise ValueError("Converter output lacks required metadata sidecar")
                outputs.append({"nifti": nii.name, "sidecar": js.name,
                                "nifti_sha256": sha256(nii), "sidecar_sha256": sha256(js),
                                **inspect_nifti(nii, js)})
            now = archive.stat()
            if (now.st_size, now.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError("Archive changed during conversion")
            result = {"source": identity, "dicom_count": len(members),
                      "selected_members_crc_checked": True,
                      "source_member_index_sha256": source_digest.hexdigest(),
                      "study_uid": str(header.get("StudyInstanceUID", "")),
                      "series_uid": str(header.get("SeriesInstanceUID", "")),
                      "converter_sha256": sha256(dcm2niix),
                      "code_sha256": sha256(__file__), "command": list(map(str, cmd)),
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
    command([fsl / "bet", t1, output / "t1_brain", "-R", "-f", "0.5", "-m"],
            output / "bet_t1.log", env=env)
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


def framewise_displacement(parameters, radius_mm=50):
    """Power-style FD from MCFLIRT rotations (radians), then translations (mm)."""
    params = np.asarray(parameters, dtype=float)
    if params.ndim != 2 or params.shape[1] != 6 or not np.isfinite(params).all():
        raise ValueError("Expected finite N x 6 MCFLIRT parameters")
    if len(params) < 2 or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Need multiple volumes and a positive rotation radius")
    delta = np.abs(np.diff(params, axis=0))
    return np.r_[0.0, radius_mm * delta[:, :3].sum(axis=1) + delta[:, 3:].sum(axis=1)]


def motion_pilot(nifti, sidecar, output, *, fsl_dir, discard_seconds=10):
    """Bounded native-space motion/QC pilot, NOT fully preprocessed BOLD.

    No slice timing, susceptibility correction, nuisance regression, filtering,
    atlas connectivity, or label access. Raw/native tSNR is a technical diagnostic.
    """
    output = Path(output)
    if output.exists():
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
    output.mkdir(parents=True)
    trimmed = output / "trimmed.nii.gz"
    nib.save(nib.Nifti1Image(data[..., discard:], img.affine, img.header), trimmed)
    del data
    env = dict(os.environ, FSLDIR=str(fsl_dir), FSLOUTPUTTYPE="NIFTI_GZ",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    prefix = output / "motion"
    command([Path(fsl_dir) / "bin/mcflirt", "-in", trimmed, "-out", prefix,
             "-plots", "-mats", "-rmsrel", "-rmsabs"], output / "mcflirt.log", env=env)
    moved = nib.load(output / "motion.nii.gz")
    array = np.asarray(moved.dataobj, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite motion-corrected image")
    mean = array.mean(axis=-1)
    nib.save(nib.Nifti1Image(mean, moved.affine), output / "mean.nii.gz")
    command([Path(fsl_dir) / "bin/bet", output / "mean.nii.gz", output / "mean_brain",
             "-f", "0.3", "-m"], output / "bet.log", env=env)
    mask = np.asarray(nib.load(output / "mean_brain_mask.nii.gz").dataobj) > 0
    if mask.sum() < 1000:
        raise ValueError("Implausibly small pilot brain mask")
    params = np.loadtxt(output / "motion.par")
    if params.shape != (array.shape[-1], 6):
        raise ValueError("Motion parameter/image length mismatch")
    fd = framewise_displacement(params)
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
              "source_sha256": sha256(nifti), "code_sha256": sha256(__file__),
              "discarded_initial_volumes": discard, "discard_rule_seconds": discard_seconds,
              "tr_seconds": tr, "remaining_volumes": len(fd), "brain_mask_voxels": int(mask.sum()),
              "mean_fd_mm_excluding_first": float(fd[1:].mean()),
              "median_fd_mm_excluding_first": float(np.median(fd[1:])),
              "max_fd_mm": float(fd.max()), "fraction_fd_gt_0p3": float((fd[1:] > .3).mean()),
              "seconds_fd_le_0p3": float((fd <= .3).sum() * tr),
              "median_native_motion_corrected_tsnr": float(np.median(tsnr)),
              "median_dvars_raw": float(np.median(dvars[1:])),
              "susceptibility_corrected": False, "slice_timing_corrected": False,
              "nuisance_regressed": False, "connectivity_measured": False,
              "outcomes_read": False}
    write_json(output / "qc.json", result)
    return result
