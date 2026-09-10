"""Lossless geometry-preserving viewer preparation and region measurements."""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .catalog import Scan, token


def label_table(path: Path | None = None):
    if path is None:
        return {}
    with path.open() as f:
        return {int(r["ID"]): {"id": int(r["ID"]), "name": r["LabelName"],
                               "color": [int(r[c]) for c in ("R", "G", "B")]}
                for r in csv.DictReader(f, delimiter="\t") if int(r["ID"]) != 0}


def validate_image(image, label=False):
    if len(image.shape) not in (3, 4) or min(image.shape[:3]) < 2:
        raise ValueError("Expected a reconstructed 3D volume or a 4D image series")
    if not np.all(np.isfinite(image.affine)) or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError("Image has no usable voxel-to-world affine")
    if label and len(image.shape) != 3:
        raise ValueError("Atlas must be a 3D integer label volume")
    if hasattr(image.header, "get_xyzt_units") and image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError("Convert the image's declared spatial units to millimetres before viewing; geometry is never silently relabelled")


def same_grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def write_volume(source: Path, destination: Path, mask_path: Path | None = None, label=False):
    """Keep native affine and voxel values. Masking removes extracranial signal only.

    NIfTI is rewritten only for MGZ conversion or masking; no normalization,
    template warping, orientation guessing, or modality-dependent rescaling.
    """
    image = nib.load(source)
    validate_image(image, label=label)
    data = np.asarray(image.dataobj)
    if mask_path:
        mask = nib.load(mask_path)
        validate_image(mask, label=True)
        if not same_grid(image, mask):
            raise ValueError("Brain mask geometry differs from the source image")
        keep = np.asarray(mask.dataobj) > 0
        data = np.where(keep[..., None] if data.ndim == 4 else keep, data, 0)
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    if label:
        if np.any(data != np.rint(data)) or data.min() < 0 or data.max() > 65535:
            raise ValueError("Atlas values must be nonnegative integer label IDs <= 65535")
        data = data.astype(np.uint16)
    out = nib.Nifti1Image(data, image.affine, dtype=data.dtype)
    out.set_qform(image.affine, code=1)
    out.set_sform(image.affine, code=1)
    out.header.set_xyzt_units("mm", "sec")
    if label:
        out.header.set_intent("label")
    if len(image.shape) == 4:
        # Copy temporal zoom and declared time units when supplied.
        zooms = image.header.get_zooms()
        out.header.set_zooms((*out.header.get_zooms()[:3], float(zooms[3])))
        time_unit = image.header.get_xyzt_units()[1] if hasattr(image.header, "get_xyzt_units") else "unknown"
        out.header.set_xyzt_units("mm", time_unit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(out, temp)
    temp.replace(destination)
    return out


def volume_info(path: Path):
    image = nib.load(path)
    validate_image(image)
    # Window estimation samples the first frame; never read a huge 4D time series in full.
    data = np.asarray(image.dataobj[..., 0] if len(image.shape) == 4 else image.dataobj)
    values = data[np.isfinite(data) & (data != 0)]
    lo, hi = np.percentile(values[::max(1, values.size // 250_000)], [2, 98]) if values.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    return {"shape": list(image.shape), "spacing": [round(float(x), 4) for x in image.header.get_zooms()[:3]],
            "orientation": "".join(nib.aff2axcodes(image.affine)), "affine": image.affine.tolist(),
            "frames": image.shape[3] if len(image.shape) == 4 else 1,
            "frame_step": float(image.header.get_zooms()[3]) if len(image.shape) == 4 else None,
            "time_unit": image.header.get_xyzt_units()[1] if hasattr(image.header, "get_xyzt_units") else "unknown",
            "spatial_unit": image.header.get_xyzt_units()[0] if hasattr(image.header, "get_xyzt_units") else "mm",
            "cal_min": float(lo), "cal_max": float(hi)}


def signal_histogram(values, scope):
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values) & (values != 0)]
    values = values[::max(1, int(np.ceil(len(values) / 250_000)))]
    if not len(values):
        return None
    lo, hi = min(0., float(values.min())), float(values.max())
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(values, bins=64, range=(lo, hi))
    return {"counts": counts.tolist(), "edges": edges.tolist(), "sample_count": len(values), "scope": scope}


def regions_from_atlas(path: Path, lut_path: Path | None = None):
    atlas = nib.load(path)
    labels = np.asarray(atlas.dataobj).astype(np.int32)
    ids, counts = np.unique(labels, return_counts=True)
    ids, counts = ids[ids > 0], counts[ids > 0]
    centers = ndimage.center_of_mass(np.ones(labels.shape, np.uint8), labels, ids)
    mm3 = abs(np.linalg.det(atlas.affine[:3, :3]))
    table = label_table(lut_path)
    boxes = ndimage.find_objects(labels)
    regions = []
    for i, n, c in zip(ids, counts, centers):
        # A curved region's centroid can lie outside its own label. Navigate to
        # the actual labelled voxel closest to the centroid, not a different ROI.
        box = boxes[int(i) - 1]
        coords = np.argwhere(labels[box] == i) + [s.start for s in box]
        focus = coords[np.argmin(np.sum((coords - c) ** 2, axis=1))]
        regions.append({**table.get(int(i), {"id": int(i), "name": f"Label {i}", "color": [180, 180, 180]}),
                        "voxels": int(n), "volume_mm3": round(float(n * mm3), 2),
                        "center_mm": [round(float(x), 2) for x in nib.affines.apply_affine(atlas.affine, c)],
                        "focus_mm": [float(x) for x in nib.affines.apply_affine(atlas.affine, focus)]})
    return regions


class ImageStore:
    def __init__(self, cache: Path):
        self.cache = cache
        self.assets: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._prepared: dict[str, dict] = {}

    def asset(self, path: Path):
        # A stable opaque ID avoids an arbitrary-file download endpoint.
        key = token(str(path.resolve()))
        self.assets[key] = path.resolve()
        return f"/api/assets/{key}/{path.name}"

    def prepare(self, scan: Scan):
        sources = [p for p in (scan.path, scan.mask, scan.atlas, scan.anatomy, scan.atlas_lut) if p]
        sources += [e["path"] for e in scan.extra]
        signature = "viewer-v5|" + json.dumps(scan.public(), sort_keys=True) + "|" + "|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in sources)
        cache_key = hashlib.sha256(signature.encode()).hexdigest()[:24]
        with self._lock:
            if cache_key in self._prepared:
                return self._prepared[cache_key]
            folder = self.cache / cache_key
            folder.mkdir(parents=True, exist_ok=True)
            if scan.kind == "tracts":
                result = {"scan": scan.public(), "volumes": [], "meshes": [{"url": self.asset(scan.path), "name": scan.description}], "regions": [], "geometry": None}
                self._prepared[cache_key] = result
                return result
            volumes = []
            primary = scan.path
            if scan.path.name.endswith(".mgz") or scan.mask:
                primary = folder / "image.nii.gz"
                if not primary.is_file():
                    write_volume(scan.path, primary, scan.mask)
            info = volume_info(primary)
            source_header = nib.load(scan.path).header
            if hasattr(source_header, "get_xyzt_units"):
                info["spatial_unit"] = source_header.get_xyzt_units()[0]
            cmap = {"MRI": "gray", "CT": "gray", "DTI": "viridis", "PET": "inferno", "SPECT": "inferno", "fMRI": "gray"}[scan.modality]
            cal_min, cal_max = (0, 1) if scan.modality == "DTI" else (info["cal_min"], info["cal_max"])
            if scan.modality == "CT":
                cal_min, cal_max = 0, 80
            if scan.anatomy:
                anatomy = scan.anatomy
                if anatomy.name.endswith(".mgz") or (scan.modality == "DTI" and scan.atlas):
                    anatomy = folder / "anatomy.nii.gz"
                    if not anatomy.is_file():
                        write_volume(scan.anatomy, anatomy, scan.atlas if scan.modality == "DTI" else None)
                anatomy_img, primary_img = nib.load(anatomy), nib.load(primary)
                if scan.registration != "verified" and not same_grid(anatomy_img, primary_img):
                    raise ValueError("Anatomy and data have different grids; supply verified registration before fusion")
                ai = volume_info(anatomy)
                volumes.append({"url": self.asset(anatomy), "name": "Mean b0" if scan.modality == "DTI" else "Reference anatomy", "role": "anatomy", "colormap": "gray", "cal_min": ai["cal_min"], "cal_max": ai["cal_max"], "opacity": 1})
            volumes.append({"url": self.asset(primary), "name": scan.description, "role": "primary", "colormap": cmap, "cal_min": cal_min, "cal_max": cal_max, "opacity": 0.7 if scan.anatomy else 1})
            regions = []
            for extra in scan.extra:
                metric = nib.load(extra["path"])
                validate_image(metric)
                if not same_grid(nib.load(primary), metric):
                    raise ValueError("Diffusion metric does not share the source image grid")
            if scan.atlas:
                atlas = folder / "atlas.nii.gz"
                if not atlas.is_file():
                    write_volume(scan.atlas, atlas, label=True)
                if not same_grid(nib.load(primary), nib.load(atlas)):
                    raise ValueError("Atlas and displayed image must share a validated voxel grid")
                region_path = folder / "regions.json"
                if not region_path.is_file():
                    region_path.write_text(json.dumps(regions_from_atlas(atlas, scan.atlas_lut)))
                regions = json.loads(region_path.read_text())
                volumes.append({"url": self.asset(atlas), "name": scan.atlas_name, "role": "atlas", "colormap": "random", "opacity": 0, "cal_min": 0, "cal_max": max((r['id'] for r in regions), default=1)})
            result = {"fingerprint": cache_key, "scan": scan.public(), "volumes": volumes, "meshes": [], "regions": regions, "geometry": info,
                      "extra": [{"name": e["name"], "units": e["units"], "url": self.asset(e["path"]), "key": e["key"]} for e in scan.extra]}
            if scan.modality == "SPECT":
                result["histogram"] = signal_histogram(np.asarray(nib.load(primary).dataobj), "Nonzero native SPECT voxels · sampled, includes background")
            if scan.modality == "fMRI" and scan.kind == "timeseries":
                from .fmri import add_fmri_summary
                add_fmri_summary(self, scan, result, folder)
            self._prepared[cache_key] = result
            return result
