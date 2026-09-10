"""Explicit, unreviewed SPECT/MRI alignment previews; never verified fusion.

Use only a processing row that names its exact FastSurfer reference. An affine
copy places the native SPECT voxel grid in MRI coordinates. No interpolation,
new reconstruction, surface projection, or quantitative calibration is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    from .catalog import Scan
    from .images import ImageStore


def fixed_to_moving_ras(row: dict) -> np.ndarray:
    """Decode PIE ScaleVersor parameters: fixed MRI LPS -> moving SPECT LPS."""
    import SimpleITK as sitk

    params = np.array([float(v) for v in row.get("reg_params", "").split()])
    center = np.array([float(v) for v in row.get("reg_center", "").split()])
    if params.shape != (9,) or center.shape != (3,) or not np.isfinite(np.r_[params, center]).all():
        raise ValueError("Registration needs nine finite parameters and a three-coordinate center")
    if np.linalg.norm(params[:3]) > 1 or np.any(params[6:] <= 0):
        raise ValueError("Invalid registration rotation or scale")
    tx = sitk.ScaleVersor3DTransform()
    tx.SetCenter(center.tolist())
    tx.SetParameters(params.tolist())
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(tx.GetMatrix()).reshape(3, 3)
    matrix[:3, 3] = np.asarray(tx.GetTranslation()) + center - matrix[:3, :3] @ center
    if np.linalg.det(matrix[:3, :3]) <= 1e-6 or np.linalg.cond(matrix[:3, :3]) > 100:
        raise ValueError("Registration is singular, reflected, or ill-conditioned")
    ras_lps = np.diag([-1., -1., 1., 1.])
    return ras_lps @ matrix @ ras_lps


@dataclass
class AnatomyPreview:
    source: Path
    reference: Scan
    row: dict

    def public(self):
        return {"reference_id": self.reference.id, "reference_date": self.reference.date,
                "status": "unreviewed", "processing": "datscan_full",
                "flip_lr": self.row["flip_lr"].lower() == "true"}


def discover_previews(repo: Path, scans: dict[str, Scan]) -> dict[str, AnatomyPreview]:
    table = repo / "Imaging/derived/datscan_full/datscan_sbr.csv"
    if not table.is_file():
        return {}
    with table.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # Ambiguous duplicate rows are not a basis for choosing a transform.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("image_id", "")] = counts.get(row.get("image_id", ""), 0) + 1
    previews = {}
    for row in rows:
        native = scans.get(f"spect-{row.get('image_id')}")
        reference = scans.get(f"mri-{row.get('fs_image_id')}")
        if (not native or not reference or counts[row["image_id"]] != 1 or row.get("error")
                or native.subject != row.get("patno") or reference.subject != native.subject
                or reference.modality != "MRI" or not reference.mask or not reference.atlas
                or row.get("flip_lr", "").lower() not in ("true", "false")):
            continue
        source = (repo / row.get("nifti", "")).resolve()
        if not source.is_file() or not source.is_relative_to(table.parent.resolve()):
            continue
        try:
            fixed_to_moving_ras(row)
        except (ValueError, RuntimeError):
            continue
        previews[native.id] = AnatomyPreview(source, reference, row)
    return previews


def aligned_image(source, row):
    from .images import validate_image

    validate_image(source)
    if len(source.shape) != 3:
        raise ValueError("Anatomy preview requires a reconstructed 3D SPECT volume")
    flip = str(row.get("flip_lr", "")).lower()
    if flip not in ("true", "false"):
        raise ValueError("Left/right processing flag must be explicit")
    data = np.asarray(source.dataobj)
    # This is the processing pipeline's documented pre-registration array flip,
    # not an orientation chosen to make the overlay look more plausible.
    if flip == "true":
        data = data[::-1]
    affine = np.linalg.inv(fixed_to_moving_ras(row)) @ source.affine
    result = nib.Nifti1Image(np.ascontiguousarray(data), affine, dtype=data.dtype)
    result.set_sform(affine, code=1)
    result.set_qform(affine, code=0)  # Sform retains any non-orthogonal scale exactly.
    result.header.set_xyzt_units("mm")
    return result


def prepare_preview(store: ImageStore, native: Scan, preview: AnatomyPreview):
    from .images import same_grid, validate_image, volume_info

    reference = preview.reference
    if reference.subject != native.subject or reference.modality != "MRI":
        raise ValueError("Anatomical reference must be an MRI of the same participant")
    anatomy = store.prepare(reference)
    sources = [preview.source, reference.path, reference.mask, reference.atlas, reference.atlas_lut]
    signature = json.dumps([preview.row, native.public(), reference.public()], sort_keys=True) + "|" + "|".join(
        f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in sources if p)
    key = "anatomy-v1-" + hashlib.sha256(signature.encode()).hexdigest()[:24]
    with store._lock:
        if key in store._prepared:
            return store._prepared[key]
        folder = store.cache / key
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "spect_in_mri_preview.nii.gz"
        aligned = aligned_image(nib.load(preview.source), preview.row)
        if not output.is_file():
            temporary = folder / "spect_in_mri_preview.tmp.nii.gz"
            nib.save(aligned, temporary)
            temporary.replace(output)
        # Estimate contrast within the MRI brain, not the large noisy SPECT FOV.
        # Samples are used for display windowing only; the image stays unmasked.
        mask = nib.load(reference.mask)
        validate_image(mask, label=True)
        if not same_grid(mask, nib.load(reference.path)):
            raise ValueError("Reference brain mask does not match its MRI")
        voxels = np.argwhere(np.asarray(mask.dataobj) > 0)
        voxels = voxels[::max(1, len(voxels) // 200_000)]
        mapping = np.linalg.inv(aligned.affine) @ mask.affine
        samples = map_coordinates(np.asarray(aligned.dataobj, dtype=np.float32),
                                  nib.affines.apply_affine(mapping, voxels).T, order=1, mode="constant")
        values = samples[np.isfinite(samples) & (samples > 0)]
        if len(values) < 0.5 * len(samples) or not len(values):
            raise ValueError("Insufficient SPECT/MRI coverage for an alignment preview")
        upper = float(np.percentile(values, 99.5))
        if upper <= 0:
            raise ValueError("No positive SPECT signal in the MRI reference")
        base = next(v for v in anatomy["volumes"] if v["role"] == "primary")
        info = volume_info(output)
        context = {**preview.public(), "reference_geometry": anatomy["geometry"],
                   "source_geometry": volume_info(preview.source),
                   "fixed_to_moving_ras": fixed_to_moving_ras(preview.row).tolist(),
                   "note": "Automatic pipeline alignment — not reviewed. Gray anatomy and region labels come from the MRI; colors are SPECT signal, not a cortical projection or binding-ratio map."}
        scan = {**native.public(), "space": reference.space, "registration": "unreviewed",
                "reference_id": reference.id, "has_anatomy": True, "has_atlas": True,
                "atlas_name": reference.atlas_name, "qc": "SPECT/MRI alignment not reviewed",
                "provenance": "PIE datscan_full reconstruction and its explicitly named MRI transform. No new registration, intensity calibration, or reconstructed resolution is claimed. Native SPECT remains available separately."}
        result = {"fingerprint": key, "scan": scan, "context": context, "geometry": info, "meshes": [],
                  "regions": anatomy["regions"], "extra": [], "volumes": [
                      {**base, "role": "anatomy", "name": "Patient MRI anatomy"},
                      {"url": store.asset(output), "name": "SPECT in MRI space — unreviewed",
                       "role": "primary", "colormap": "inferno", "cal_min": 0, "cal_max": upper, "opacity": 0.7},
                      *[v for v in anatomy["volumes"] if v["role"] == "atlas"]]}
        from .images import signal_histogram
        result["histogram"] = signal_histogram(values, "SPECT samples inside MRI mask · alignment unreviewed")
        store._prepared[key] = result
        return result
