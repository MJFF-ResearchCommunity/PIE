"""
datscan.py — DaTscan (123I-ioflupane) SPECT: raw projections -> reconstructed volume -> striatal binding
ratios (SBR) using the subject's own FastSurfer segmentation as the ROI atlas.

PPMI distributes the *raw* tomographic projections (NM DICOM, ImageType TOMO/EMISSION; 60-480 frames =
detectors x energy windows x angles) and only releases SBR values for some cohorts. This module follows
PPMI's own recipe (reconstruction -> attenuation/filtering -> anatomical alignment -> caudate/putamen vs
occipital reference; SBR = target/reference - 1) with open components:

    read_projections : photopeak window, all detectors, angle per frame (DICOM NM vectors)
    reconstruct      : filtered back-projection per transaxial slice (scikit-image), Gaussian post-filter
    register_to_t1   : rigid registration of the SPECT volume to the subject's conformed T1 (SimpleITK, MI)
    quantify         : mean counts in FastSurfer caudate/putamen (L/R) and occipital cortex -> SBRs

Not implemented (ponytail): Chang attenuation correction; SBRs are validated/calibrated against PPMI's
published values on the cohorts that have them, which absorbs the resulting scale offset.
"""

import io
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage

I123_KEV = 159.0
# FreeSurfer / DKT label ids
CAUDATE_L, CAUDATE_R, PUTAMEN_L, PUTAMEN_R = 11, 50, 12, 51
OCCIPITAL = [1005, 1011, 1013, 1021, 2005, 2011, 2013, 2021]  # cuneus, lateral occipital, lingual, pericalcarine (lh, rh)


# ------------------------------------------------------------------------------------------ reading
def _vec(ds, name, n):
    v = getattr(ds, name, None)
    return np.asarray(list(v), dtype=int) if v is not None else np.ones(n, dtype=int)


def _photopeak_window(ds):
    """Index (1-based) of the energy window containing 159 keV, or None if unknown."""
    seqs = getattr(ds, "EnergyWindowInformationSequence", None) or []
    for i, e in enumerate(seqs, start=1):
        for r in e.get("EnergyWindowRangeSequence") or []:
            lo, hi = float(r.get("EnergyWindowLowerLimit", 0)), float(r.get("EnergyWindowUpperLimit", 0))
            if hi > 1000:  # some vendors store eV*100-ish values (14310..17490)
                lo, hi = lo / 100, hi / 100
            if lo <= I123_KEV <= hi:
                return i
    return None


def read_projections(dcm):
    """Load an NM TOMO DICOM (path, bytes or dataset). Returns dict with
    proj (n_angles, n_rows(axial), n_cols(transaxial)) summed over detectors, angles_deg (n_angles,),
    spacing_mm, and header metadata. Raises ValueError for series that are not tomographic projections."""
    ds = dcm if isinstance(dcm, pydicom.Dataset) else pydicom.dcmread(io.BytesIO(dcm) if isinstance(dcm, bytes) else str(dcm), force=True)
    itype = [str(x).upper() for x in getattr(ds, "ImageType", [])]
    n = int(getattr(ds, "NumberOfFrames", 1))
    rot = getattr(ds, "RotationInformationSequence", None)
    if "TOMO" not in itype or not rot or n < 30:
        raise ValueError(f"not a tomographic projection series: ImageType={itype}, frames={n}, rotation={'yes' if rot else 'no'}")
    px = getattr(ds, "PixelSpacing", None)
    if px is None:
        raise ValueError("no PixelSpacing")
    spacing = float(px[0])
    data = ds.pixel_array.astype(np.float32)
    if data.ndim == 2:
        data = data[None]
    ew, det, ang = _vec(ds, "EnergyWindowVector", n), _vec(ds, "DetectorVector", n), _vec(ds, "AngularViewVector", n)
    peak = _photopeak_window(ds)
    keep = np.ones(n, dtype=bool) if peak is None else ew == peak
    if keep.sum() == 0:
        keep[:] = True
    r0 = rot[0]
    n_views = int(getattr(r0, "NumberOfFramesInRotation", 0) or 0)
    step = float(getattr(r0, "AngularStep", 0) or 0)
    arc = float(getattr(r0, "ScanArc", 0) or 0)
    direction = -1.0 if str(getattr(r0, "RotationDirection", "CW")).upper().startswith("CC") else 1.0
    dets = np.unique(det[keep])
    if n_views <= 1 or step <= 0:  # broken headers (e.g. GE INFINIA 'BRAIN_SPECT'): infer from the frame count
        n_views = int(keep.sum() // len(dets))
        step = (arc if arc > 0 else 360.0) / n_views
    det_info = getattr(ds, "DetectorInformationSequence", None) or []
    start = {}
    for i, d in enumerate(det_info, start=1):
        s = d.get("StartAngle", None)
        start[i] = float(s[0] if hasattr(s, "__len__") and not isinstance(s, str) else s) if s is not None else np.nan
    rot_start = float(getattr(r0, "StartAngle", 0) or 0)
    frames, angles = [], []
    for j, d in enumerate(dets):
        sel = np.flatnonzero(keep & (det == d))
        views = ang[sel]
        order = np.argsort(views)
        sel = sel[order]
        s0 = start.get(int(d), np.nan)
        if not np.isfinite(s0) or s0 < 0 and len(dets) == 1:
            # no per-detector start (GE): heads are spaced evenly around the gantry (H-mode: 180 deg apart),
            # confirmed by head-2 frame k matching head-1 frame k + n/2 on GE Millennium and Siemens series
            s0 = rot_start + 360.0 / len(dets) * j
        # DICOM angles run opposite to skimage's radon angles: the start position maps to -start, the
        # rotation direction maps unchanged. Validated on the striatum position across vendors and start
        # angles (0/180 deg starts are invariant, 90/270 deg starts would otherwise come out 180 deg off).
        a = -s0 + direction * step * (views[order] - 1)
        frames.append(data[sel])
        angles.append(a)
    proj = np.concatenate(frames, axis=0)
    angles = np.mod(np.concatenate(angles), 360.0)
    # merge frames that fall on the same angle (dual-head 360 deg acquisitions) by summing
    key = np.round(angles / (step / 2)).astype(int) % int(round(360.0 / (step / 2)))  # wrap 360 -> 0
    uniq, inv = np.unique(key, return_inverse=True)
    if len(uniq) < len(key):
        merged = np.zeros((len(uniq),) + proj.shape[1:], dtype=np.float32)
        np.add.at(merged, inv, proj)
        proj, angles = merged, np.array([angles[inv == i].mean() for i in range(len(uniq))])
    order = np.argsort(angles)
    collimator = str(det_info[0].get("CollimatorType", "")).upper() if det_info else ""
    zoom = det_info[0].get("ZoomFactor", None) if det_info else None
    # fan-beam data (Marconi/Picker Prism) reconstructed as parallel-beam are magnified transaxially, and the
    # HERMES exports of the same cameras carry neither collimator nor zoom: both get a scale-fitting registration
    scale_fit = collimator.startswith("FAN") or (collimator != "PARA" and zoom is None)
    meta = {"manufacturer": str(getattr(ds, "Manufacturer", "")), "model": str(getattr(ds, "ManufacturerModelName", "")),
            "collimator": collimator, "zoom": float(zoom[0]) if zoom is not None and len(zoom) else np.nan, "scale_fit": bool(scale_fit),
            "n_frames": n, "n_detectors": int(len(dets)), "n_angles": int(len(order)), "angular_step": step, "scan_arc": arc,
            "rotation_direction": "CW" if direction > 0 else "CC", "start_angle": rot_start, "energy_window": peak,
            "rows": int(data.shape[1]), "cols": int(data.shape[2]), "spacing_mm": spacing, "counts_total": float(proj.sum())}
    return {"proj": proj[order], "angles_deg": angles[order], "spacing_mm": spacing, "meta": meta}


def read_volume(dcm):
    """Series stored as an already-reconstructed transaxial stack (no rotation info). Returns (vol xyz, spacing)."""
    ds = dcm if isinstance(dcm, pydicom.Dataset) else pydicom.dcmread(io.BytesIO(dcm) if isinstance(dcm, bytes) else str(dcm), force=True)
    data = ds.pixel_array.astype(np.float32)
    px = getattr(ds, "PixelSpacing", None)
    spacing = float(px[0]) if px is not None else np.nan
    return np.transpose(data, (2, 1, 0)), spacing


# --------------------------------------------------------------------------------- reconstruction
def chang_correction(vol, spacing_mm, mu_per_cm=0.11, n_dirs=36, threshold=0.15):
    """First-order Chang attenuation correction with a uniform attenuation coefficient inside the head.
    The head outline per transaxial slice is the smoothed volume above ``threshold`` x its max (holes
    filled); each voxel is multiplied by 1 / mean_over_directions(exp(-mu * path length to the outline))."""
    from scipy.ndimage import binary_fill_holes, rotate

    mu = mu_per_cm / 10.0 * spacing_mm  # per voxel
    sm = ndimage.gaussian_filter(vol, sigma=8.0 / 2.3548 / spacing_mm)
    corr = np.ones_like(vol)
    angles = np.arange(n_dirs) * (360.0 / n_dirs)
    for z in range(vol.shape[2]):
        sl = sm[:, :, z]
        if sl.max() <= 0:
            continue
        mask = binary_fill_holes(sl > threshold * sl.max())
        if mask.sum() < 50:
            continue
        acc = np.zeros(mask.shape, dtype=np.float32)
        for a in angles:
            m = rotate(mask.astype(np.float32), a, reshape=False, order=0) > 0.5
            # path length from each pixel to the outline along +x in the rotated frame:
            # number of mask pixels from the pixel to the end of its row
            length = np.cumsum(m[:, ::-1], axis=1)[:, ::-1] * m
            att = np.exp(-mu * length)
            acc += rotate(att, -a, reshape=False, order=1)
        acc /= n_dirs
        corr[:, :, z] = np.where(mask & (acc > 1e-3), 1.0 / np.maximum(acc, 1e-3), 1.0)
    return vol * corr


def subtract_point_sources(vol, angles_deg, spacing_mm, filter_name="hann", factor=6.0, max_voxels=200):
    """Remove external point sources (fiducial markers taped to the head on some Philips acquisitions) from
    a filtered back-projection. In single frames a marker is barely brighter than the head, but the
    coherent back-projection makes it 15-45 x the head's brightest voxel and puts a star artefact through
    the whole volume, flattening every SBR. Small blobs above ``factor`` x the 99.5th percentile of the
    6 mm-smoothed volume are re-projected with the same geometry and their reconstruction (marker plus
    streaks) subtracted. Returns (volume, number of marker voxels)."""
    from skimage.transform import iradon, radon

    sm = ndimage.gaussian_filter(vol, sigma=6.0 / 2.3548 / spacing_mm)
    core = sm > factor * np.percentile(sm, 99.5)
    if not core.any():
        return vol, 0
    lab, n = ndimage.label(core)
    sizes = ndimage.sum(core, lab, range(1, n + 1))
    small = np.isin(lab, 1 + np.flatnonzero(sizes <= max_voxels))
    if not small.any():
        return vol, 0
    mask = ndimage.binary_dilation(small, iterations=2)
    out = vol.copy()
    nx = vol.shape[0]
    for z in np.flatnonzero(mask.any(axis=(0, 1))):
        m = np.where(mask[:, :, z], vol[:, :, z], 0.0)
        sino = radon(m, theta=angles_deg, circle=True)
        out[:, :, z] -= iradon(sino, theta=angles_deg, output_size=nx, filter_name=filter_name, circle=False).astype(np.float32)
    out[out < 0] = 0
    return out, int(small.sum())


def reconstruct(proj, angles_deg, spacing_mm, fwhm_mm=6.0, filter_name="hann", attenuation=True, point_sources=True):
    """Filtered back-projection of every transaxial slice. proj: (n_angles, n_axial, n_transaxial).
    Returns a volume indexed (x, y, z) with isotropic ``spacing_mm`` voxels, external point sources removed
    (see ``subtract_point_sources``; the count is stored on ``reconstruct.point_source_voxels``),
    Chang-corrected (optional) and Gaussian-smoothed to ``fwhm_mm``."""
    from skimage.transform import iradon

    n_ang, nz, nx = proj.shape
    vol = np.zeros((nx, nx, nz), dtype=np.float32)
    for z in range(nz):
        sino = proj[:, z, :].T  # (n_transaxial, n_angles)
        if sino.sum() <= 0:
            continue
        vol[:, :, z] = iradon(sino, theta=angles_deg, output_size=nx, filter_name=filter_name, circle=False).astype(np.float32)
    vol[vol < 0] = 0
    reconstruct.point_source_voxels = 0
    if point_sources:
        vol, reconstruct.point_source_voxels = subtract_point_sources(vol, angles_deg, spacing_mm, filter_name)
    if attenuation:
        vol = chang_correction(vol, spacing_mm)
    if fwhm_mm:
        vol = ndimage.gaussian_filter(vol, sigma=fwhm_mm / 2.3548 / spacing_mm)
    return vol


def to_nifti(vol, spacing_mm):
    """Reconstruction (row, col, slice) -> NIfTI in patient axes up to left/right chirality: isotropic
    spacing, centred at the origin, +y anterior, +z superior. With the angle convention of
    ``read_projections`` the anterior direction is -row for every vendor and start angle (measured from the
    striatum position in the mean image per scanner), and projection rows run head -> feet (DICOM row
    direction (0,0,-1); the neck is truncated at the last row), so rows become -y and slices become -z."""
    out = np.transpose(vol[::-1, :, ::-1], (1, 0, 2))
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = -np.array(out.shape) * spacing_mm / 2
    return nib.Nifti1Image(np.ascontiguousarray(out).astype(np.float32), affine)


# --------------------------------------------------------------------------- registration + SBR
def _sitk_from_nib(img):
    import SimpleITK as sitk
    data = np.asanyarray(img.dataobj).astype(np.float32)
    ras = nib.as_closest_canonical(img)
    data = np.asanyarray(ras.dataobj).astype(np.float32)
    aff = ras.affine
    out = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))  # sitk wants (z, y, x)
    out.SetSpacing(tuple(float(x) for x in np.abs(np.diag(aff)[:3])))
    origin = aff[:3, 3]
    out.SetOrigin((-float(origin[0]), -float(origin[1]), float(origin[2])))  # RAS -> LPS
    out.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))  # canonical RAS axes expressed in LPS
    return out


def _clean_spect(img, cap=1.5, head_litres=4.0):
    """Registration copy of the SPECT: keep only the head and winsorise at ``cap`` x the 99.5th percentile
    inside it. The head is the largest connected component of the ``head_litres`` brightest litres of a
    20 mm-smoothed copy: a fixed volume is robust where an Otsu threshold is not (low-count Marconi/Picker
    scans have streak backgrounds that Otsu merges into a 10-20 L 'head'; markers dominate a max-relative
    threshold), and slightly under-sizing the mask only trims the scalp."""
    data = np.asanyarray(img.dataobj).astype(np.float32)
    vox = float(np.abs(np.diag(img.affine)[:3]).mean()) or 1.0
    sm = ndimage.gaussian_filter(data, sigma=20.0 / 2.3548 / vox)
    n_keep = int(min(head_litres * 1e6 / vox**3, sm.size * 0.5))
    thr = np.partition(sm.ravel(), -n_keep)[-n_keep]
    lab, n = ndimage.label(sm >= thr)
    if n == 0:
        return nib.Nifti1Image(data, img.affine)
    sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
    head = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    top = float(np.percentile(data[head], 99.5))
    data = np.where(head, np.minimum(data, cap * top), 0.0)
    return nib.Nifti1Image(data, img.affine)


def synthetic_spect(t1_img, aparc_img, fwhm_mm=10.0, striatum=(CAUDATE_L, CAUDATE_R, PUTAMEN_L, PUTAMEN_R)):
    """A subject-specific DaT-like template in T1 space: striatum 1.0, rest of brain 0.25, non-brain head
    tissue 0.12 (from the T1 intensity), smoothed to SPECT resolution. Registering the real SPECT to this
    image by correlation aligns the hot spots with the labels and the head outline with the blob at once,
    which mutual information against the raw T1 does not do reliably."""
    lab = np.asanyarray(aparc_img.dataobj)
    t1 = np.asanyarray(t1_img.dataobj).astype(np.float32)
    head = t1 > 0.15 * np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else lab > 0
    syn = np.where(head, 0.12, 0.0).astype(np.float32)
    syn[lab > 0] = 0.25
    syn[np.isin(lab, striatum)] = 1.0
    vox = float(np.abs(np.diag(t1_img.affine)[:3]).mean()) or 1.0
    syn = ndimage.gaussian_filter(syn, sigma=fwhm_mm / 2.3548 / vox)
    return nib.Nifti1Image(syn, t1_img.affine)


def register_to_t1(spect_img, t1_img, flip_lr=False, rz=0.0, search=False, aparc_img=None, mask_img=None, scale_fit=False):
    """Rigid registration of the SPECT volume to the subject's T1 space via a synthetic DaT template
    (see ``synthetic_spect``), normalised correlation metric. The volume is already in patient axes up to
    the vendor's transaxial conventions, so the initial pose is the centre-of-mass alignment rotated by
    ``rz`` about the head axis (0 or pi), after an optional left/right mirror (``flip_lr``); ``search=True``
    tries both rz values and keeps the better metric (fallback for unknown acquisition configurations; the
    striatum is faint in patients, so a metric-based choice of pose is less reliable than the per-vendor
    rule). Multi-resolution refinement follows. A second, striatum-masked refinement stage was tried and
    removed: with the faint striata of patients it drifted (side-wise putamen agreement with PPMI fell from
    0.84 to 0.78 pooled, 0.64 to 0.27 on ADAC). Returns (transform, metric, moving image)."""
    import SimpleITK as sitk

    if aparc_img is None:
        raise ValueError("register_to_t1 needs the aparc segmentation to build the synthetic template")
    if flip_lr:
        spect_img = nib.Nifti1Image(np.asanyarray(spect_img.dataobj)[::-1], spect_img.affine)
    moving = _sitk_from_nib(_clean_spect(spect_img))
    fixed = sitk.Shrink(_sitk_from_nib(synthetic_spect(t1_img, aparc_img)), [2, 2, 2])
    # rigid, or rigid + per-axis scale (no shear) when ``scale_fit``: fan-beam acquisitions (Marconi/Picker
    # Prism, collimator FANB) reconstructed as parallel-beam come out magnified transaxially by ~1.5 x, which
    # a rigid fit cannot absorb; the fitted scales are reported for QC. Parallel-hole vendors with a correct
    # pixel size lose a little accuracy with free scales, so they stay rigid.
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.ScaleVersor3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    init = sitk.ScaleVersor3DTransform(init)
    scale_w = [1.0, 1.0, 1.0] if scale_fit else [0.0, 0.0, 0.0]

    def method(sampling):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, seed=0)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg

    fixed_lo = sitk.Shrink(fixed, [2, 2, 2])
    best, best_val = None, np.inf
    for cand_rz in ((0.0, np.pi) if search else (rz,)):
        cand = sitk.ScaleVersor3DTransform(init)
        cand.SetRotation((0.0, 0.0, 1.0), float(cand_rz))
        reg = method(0.5)
        reg.SetInitialTransform(cand, inPlace=False)
        try:
            val = reg.MetricEvaluate(fixed_lo, moving)
        except RuntimeError:
            continue
        if val < best_val:
            best, best_val = cand, val
    coarse = best if best is not None else init
    reg = method(0.3)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetOptimizerWeights([1, 1, 1, 1, 1, 1] + scale_w)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(coarse, inPlace=False)
    try:
        tx = reg.Execute(fixed, moving)
        metric = float(reg.GetMetricValue())
    except RuntimeError:
        tx, metric = coarse, best_val
    return tx, metric, moving


def sbr_from_arrays(S, L, dilate=0, ref_dilate=0, search_vox=2):
    """SBRs from a counts array ``S`` and a co-registered label array ``L`` (same grid, any orientation).
    Striatal masks may be dilated (SPECT resolution ~10 mm); the occipital reference (cortical labels,
    optionally dilated into the brain, striatum excluded); a small translation search (+-``search_vox``
    voxels) maximises the striatal counts, mimicking ROI placement on the hottest striatal region."""
    brain = L > 0
    striatum_ids = {"caudate_l": [CAUDATE_L], "caudate_r": [CAUDATE_R], "putamen_l": [PUTAMEN_L], "putamen_r": [PUTAMEN_R]}
    masks = {k: ndimage.binary_dilation(np.isin(L, v), iterations=dilate) if dilate else np.isin(L, v) for k, v in striatum_ids.items()}
    striatum = np.any(list(masks.values()), axis=0)
    occ = ndimage.binary_dilation(np.isin(L, OCCIPITAL), iterations=ref_dilate) & brain & ~striatum if ref_dilate else np.isin(L, OCCIPITAL)
    best, best_shift = -np.inf, (0, 0, 0)
    r = range(-search_vox, search_vox + 1)
    for dz in r:
        for dy in r:
            for dx in r:
                m = np.roll(striatum, (dz, dy, dx), axis=(0, 1, 2))
                v = S[m].mean() if m.any() else -np.inf
                if v > best:
                    best, best_shift = v, (dz, dy, dx)
    if best_shift != (0, 0, 0):
        masks = {k: np.roll(m, best_shift, axis=(0, 1, 2)) for k, m in masks.items()}
        occ = np.roll(occ, best_shift, axis=(0, 1, 2))
    means = {k: float(S[m].mean()) if m.sum() >= 20 else np.nan for k, m in masks.items()}
    means["occipital"] = float(S[occ].mean()) if occ.sum() >= 50 else np.nan
    ref = means["occipital"]
    out = {f"sbr_{k}": (v / ref - 1.0) if (ref and np.isfinite(ref) and ref > 0) else np.nan for k, v in means.items() if k != "occipital"}
    out.update({f"mean_{k}": v for k, v in means.items()})
    out["n_label_voxels"] = int(striatum.sum())
    out["n_ref_voxels"] = int(occ.sum())
    out["shift_vox"] = float(np.sqrt(sum(d * d for d in best_shift)))
    return out


def quantify(spect_img, t1_img, aparc_img, flip_lr=False, rz=0.0, search=False, dilate=0, ref_dilate=0, search_vox=2, mask_img=None, scale_fit=False):
    """Register SPECT -> T1 space (synthetic-template correlation), resample the FastSurfer labels onto the
    SPECT grid and compute SBRs (see ``sbr_from_arrays``). Returns SBRs, ROI means, the registration
    metric (negative normalised correlation; more negative is better) and QC fields."""
    import SimpleITK as sitk

    tx, metric, moving = register_to_t1(spect_img, t1_img, flip_lr, rz=rz, search=search, aparc_img=aparc_img, mask_img=mask_img, scale_fit=scale_fit)
    labels = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc_img.dataobj).astype(np.float32), aparc_img.affine))
    L = sitk.GetArrayFromImage(sitk.Resample(labels, moving, tx.GetInverse(), sitk.sitkNearestNeighbor, 0.0))
    raw = spect_img if not flip_lr else nib.Nifti1Image(np.asanyarray(spect_img.dataobj)[::-1], spect_img.affine)
    S = sitk.GetArrayFromImage(_sitk_from_nib(raw))  # un-thresholded counts on the same grid
    out = sbr_from_arrays(S, L, dilate=dilate, ref_dilate=ref_dilate, search_vox=search_vox)
    out["reg_metric"] = metric
    out["flip_lr"] = bool(flip_lr)
    out["reg_params"] = " ".join(f"{v:.6g}" for v in tx.GetParameters())  # ScaleVersor3D: versor xyz, translation xyz, scale xyz (fixed -> moving)
    out["reg_scale_x"], out["reg_scale_y"], out["reg_scale_z"] = (float(v) for v in tx.GetParameters()[6:9])
    out["reg_center"] = " ".join(f"{v:.6g}" for v in tx.GetFixedParameters()[:3])
    return out


# ------------------------------------------------------------------------------------- pipeline
def process_series(zip_path, member, out_dir, fastsurfer_subject_dir=None, fwhm_mm=6.0, flip_lr=False, attenuation=False):
    """Reconstruct one series (member path inside zip) to <out_dir>/<image_id>_datscan.nii.gz and, when a
    FastSurfer subject dir is given, quantify SBRs against it. Returns a flat dict for a results table."""
    image_id = member.split("/")[4]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti = out_dir / f"{image_id}_datscan{'_ac' if attenuation else ''}.nii.gz"
    row = {"image_id": image_id, "patno": int(member.split("/")[1]), "series_desc": member.split("/")[2]}
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    ds = pydicom.dcmread(io.BytesIO(raw), force=True)
    try:
        p = read_projections(ds)
        vol = reconstruct(p["proj"], p["angles_deg"], p["spacing_mm"], fwhm_mm=fwhm_mm, attenuation=attenuation)
        row.update({f"hdr_{k}": v for k, v in p["meta"].items()})
        row["point_source_voxels"] = reconstruct.point_source_voxels
        row["kind"] = "projections"
        spacing = p["spacing_mm"]
    except ValueError as e:
        itype = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        if "TOMO" in itype and getattr(ds, "RotationInformationSequence", None):
            raise
        vol, spacing = read_volume(ds)  # stored as a reconstructed stack
        if not np.isfinite(spacing):
            raise ValueError(f"{image_id}: reconstructed stack without pixel spacing") from e
        vol = ndimage.gaussian_filter(vol, sigma=fwhm_mm / 2.3548 / spacing) if fwhm_mm else vol
        row["kind"] = "stored_volume"
        row["hdr_manufacturer"] = str(getattr(ds, "Manufacturer", ""))
        row["hdr_model"] = str(getattr(ds, "ManufacturerModelName", ""))
    img = to_nifti(vol, spacing)
    nib.save(img, nifti)
    row["nifti"] = str(nifti)
    if fastsurfer_subject_dir:
        mri = Path(fastsurfer_subject_dir) / "mri"
        t1, aparc = nib.load(mri / "orig.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
        mask = nib.load(mri / "mask.mgz") if (mri / "mask.mgz").exists() else None
        row.update(quantify(img, t1, aparc, flip_lr=flip_lr, mask_img=mask, scale_fit=row.get("hdr_scale_fit", False)))
    return row


# ------------------------------------------------------------------------------------------ CLI
def _prefer_photopeak_member(idx):
    """Some Philips series are split into one DICOM file per energy window (159 keV photopeak and the
    110-134 keV scatter window, where cobalt fiducial markers are bright). Keep, per image_id, the member
    whose window contains the photopeak; unknown windows fall back to the first member."""
    keep = []
    for image_id, g in idx.groupby("image_id", sort=False):
        if len(g) == 1:
            keep.append(g.index[0])
            continue
        chosen = g.index[0]
        for r in g.itertuples():
            try:
                with zipfile.ZipFile(r.zip) as z:
                    ds = pydicom.dcmread(io.BytesIO(z.read(r.member)), force=True, stop_before_pixels=True)
                if _photopeak_window(ds) is not None:
                    chosen = r.Index
                    break
            except Exception:
                continue
        keep.append(chosen)
    return idx.loc[keep]


def _job(args):
    zip_path, member, out_dir, fs_dir, flip, attenuation = args
    try:
        import SimpleITK as sitk
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)  # workers run in parallel; avoid 4 x 20 threads
    except Exception:
        pass
    try:
        row = process_series(zip_path, member, out_dir, fs_dir, flip_lr=flip, attenuation=attenuation)
        row["error"] = ""
    except Exception as e:  # keep the batch going; record the failure
        row = {"image_id": member.split("/")[4], "patno": int(member.split("/")[1]), "series_desc": member.split("/")[2], "error": str(e)[:200]}
    return row


def main(argv=None):
    """python -m pie.imaging.datscan --index Imaging/derived/spect_index.csv --sessions Imaging/derived/sessions.csv
           --fastsurfer-dir Imaging/derived/fastsurfer --out-dir Imaging/derived/datscan [--workers 4] [--limit N]
           [--patnos file] [--flip-lr]
    Reconstructs every TOMO projection series in the index (one per subject, the largest by frame count) and
    quantifies SBRs against that subject's FastSurfer segmentation; appends rows to <out-dir>/datscan_sbr.csv."""
    import argparse
    import csv
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="CSV with columns zip, member, patno, image_id, kind, frames (see spect_index.csv)")
    ap.add_argument("--zip-dir", default="Imaging", help="directory holding the zips named in the index")
    ap.add_argument("--sessions", required=True, help="PIE sessions.csv (patno -> FastSurfer image_id)")
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--patnos", help="optional text file of PATNOs to process")
    ap.add_argument("--flip-lr", dest="flip_lr", choices=["true", "false"], default="false",
                    help="mirror the reconstruction left/right (validation against PPMI: no vendor needs it)")
    ap.add_argument("--attenuation", dest="attenuation", action="store_true", default=False, help="Chang attenuation correction (experimental; off by default)")
    a = ap.parse_args(argv)

    idx = pd.read_csv(a.index, dtype={"image_id": str})
    idx = idx[idx["kind"] == "TOMO"].sort_values("frames", ascending=False)
    idx = _prefer_photopeak_member(idx).drop_duplicates("patno")
    if a.patnos:
        keep = {int(x) for x in Path(a.patnos).read_text().split()}
        idx = idx[idx["patno"].isin(keep)]
    sess = pd.read_csv(a.sessions, dtype={"image_id": str})
    fs_root = Path(a.fastsurfer_dir)
    fs_done = {p.parent.parent.name for p in fs_root.glob("*/stats/aseg+DKT.stats")}
    fs_by_patno = {int(r.patno): r.image_id for r in sess.sort_values("session_date").itertuples() if r.image_id in fs_done}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "datscan_sbr.csv"
    done = set(pd.read_csv(out_csv, dtype={"image_id": str})["image_id"]) if out_csv.exists() else set()
    jobs = []
    for r in idx.itertuples():
        if r.image_id in done:
            continue
        fs_dir = fs_root / fs_by_patno[int(r.patno)] if int(r.patno) in fs_by_patno else None
        zp = r.zip if Path(r.zip).exists() else str(Path(a.zip_dir) / Path(r.zip).name)
        flip = a.flip_lr == "true"
        jobs.append((zp, r.member, out_dir / "nifti", str(fs_dir) if fs_dir else None, flip, a.attenuation))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} series to process ({len(done)} already done)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(out_csv, "a", newline="") as fh:
        writer = None
        for i, fut in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), start=1):
            row = fut.result()
            if writer is None:
                fields = sorted(set(row) | {"error", "sbr_putamen_l", "sbr_putamen_r", "sbr_caudate_l", "sbr_caudate_r"})
                if out_csv.stat().st_size == 0:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    writer.writeheader()
                else:
                    fields = pd.read_csv(out_csv, nrows=0).columns.tolist()
                    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            fh.flush()
            if i % 20 == 0 or row.get("error"):
                print(f"{i}/{len(jobs)} {row['image_id']} {'ERROR ' + row['error'] if row.get('error') else 'ok'}", flush=True)
    print("done ->", out_csv)


if __name__ == "__main__":
    main()
