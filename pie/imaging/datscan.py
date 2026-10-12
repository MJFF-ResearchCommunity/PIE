"""
datscan.py — DaTscan (123I-ioflupane) SPECT: raw projections -> reconstructed volume -> striatal binding
ratios (SBR) using the subject's own FastSurfer segmentation as the ROI atlas.

PPMI distributes the *raw* tomographic projections (NM DICOM, ImageType TOMO/EMISSION; 60-480 frames =
detectors x energy windows x angles) and only releases SBR values for some cohorts. This module follows
PPMI's own recipe (reconstruction -> attenuation/filtering -> anatomical alignment -> caudate/putamen vs
occipital reference; SBR = target/reference - 1) with open components:

    read_projections : photopeak window, all detectors, angle per frame (DICOM NM vectors)
    reconstruct      : filtered back-projection per transaxial slice (scikit-image), Gaussian post-filter
    read_volume      : already-reconstructed stacks (consortium sites), placed by their DICOM patient geometry
    register_to_t1   : registration of the SPECT volume to the subject's conformed T1 via a brain-only synthetic DaT
                       template (SimpleITK, normalised correlation in the brain; rigid, plus bounded per-axis scale for
                       fan-beam cameras)
    quantify         : mean counts in FastSurfer caudate/putamen (L/R) and occipital cortex -> SBRs

Attenuation correction (first-order Chang, ``chang_correction``) is implemented but off by default everywhere
(``reconstruct``, ``process_series`` and the CLI's ``--attenuation``): it lowered agreement with PPMI's SBRs.
Without it absolute SBRs sit below PPMI's; ``calibrate`` maps them onto a published scale per camera model.
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
WM = [2, 41]  # cerebral white matter (lh, rh): alternative reference region, kept WM_MARGIN_MM clear of the striatum
WM_MARGIN_MM = 15.0


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
    """Series stored as an already-reconstructed stack (NM RECON TOMO, as consortium sites send them). The geometry comes
    from ImageOrientationPatient / ImagePositionPatient (top level, or the first DetectorInformationSequence item where NM
    files keep them), PixelSpacing and SpacingBetweenSlices (signed; frames step along row x column; SliceThickness when
    absent), and the stack is resampled (linear) onto an axis-aligned RAS grid at the finer in-plane spacing, so the
    registration starts from true patient axes. Raises ValueError when any of them is missing: until October 2026 this
    path guessed the orientation with the projection flips, which can mirror left and right. Returns (NIfTI, spacing_mm)."""
    ds = dcm if isinstance(dcm, pydicom.Dataset) else pydicom.dcmread(io.BytesIO(dcm) if isinstance(dcm, bytes) else str(dcm), force=True)
    det = (list(getattr(ds, "DetectorInformationSequence", None) or []) or [pydicom.Dataset()])[0]
    get = lambda k: ds.get(k) if ds.get(k) is not None else det.get(k)
    data = ds.pixel_array.astype(np.float32)
    data = data[None] if data.ndim == 2 else data                       # (frames, rows, columns)
    iop, ipp, px = get("ImageOrientationPatient"), get("ImagePositionPatient"), get("PixelSpacing")
    dz = get("SpacingBetweenSlices") or get("SliceThickness") or (1.0 if len(data) == 1 else None)
    missing = [k for k, v in (("ImageOrientationPatient", iop), ("ImagePositionPatient", ipp), ("PixelSpacing", px),
                              ("SpacingBetweenSlices", dz)) if v is None]
    if missing:
        raise ValueError(f"reconstructed stack without {', '.join(missing)}: patient geometry unknown, not guessing the orientation")
    row, col = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    aff = np.eye(4)
    aff[:3, :3] = np.column_stack([row * float(px[1]), col * float(px[0]), np.cross(row, col) * float(dz)])   # (column, row, frame) index -> LPS
    aff[:3, 3] = np.asarray(ipp, float)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff                          # LPS -> RAS
    vol = np.transpose(data, (2, 1, 0))
    sp = float(min(px))
    corners = aff[:3, :3] @ (np.indices((2, 2, 2)).reshape(3, -1) * (np.array(vol.shape) - 1)[:, None]) + aff[:3, 3:]
    lo = corners.min(axis=1)
    out = np.diag([sp, sp, sp, 1.0])
    out[:3, 3] = lo
    m = np.linalg.inv(aff) @ out                                          # output voxel -> stored voxel
    shape = tuple(int(n) for n in np.round((corners.max(axis=1) - lo) / sp) + 1)
    return nib.Nifti1Image(ndimage.affine_transform(vol, m[:3, :3], m[:3, 3], output_shape=shape, order=1).astype(np.float32), out), sp


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


def reconstruct(proj, angles_deg, spacing_mm, fwhm_mm=6.0, filter_name="hann", attenuation=False, point_sources=True):
    """Filtered back-projection of every transaxial slice. proj: (n_angles, n_axial, n_transaxial).
    Returns a volume indexed (x, y, z) with isotropic ``spacing_mm`` voxels, external point sources removed
    (see ``subtract_point_sources``; the count is stored on ``reconstruct.point_source_voxels``),
    Chang-corrected when ``attenuation`` (off by default, as in the pipeline) and Gaussian-smoothed to ``fwhm_mm``."""
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
# SimpleITK treats seed 0 as "seed from the wall clock" (sitkWallClock): every random-sampling registration in PIE
# uses this fixed seed instead, so a rerun reproduces its transforms (same thread count assumed).
ITK_SEED = 20260909


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
    vox = float(np.mean(nib.affines.voxel_sizes(img.affine)))
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


def synthetic_spect(t1_img, aparc_img, fwhm_mm=10.0, striatum=(CAUDATE_L, CAUDATE_R, PUTAMEN_L, PUTAMEN_R), head=False):
    """A subject-specific, brain-only DaT-like template on the T1 grid: striatum 1.0, rest of brain 0.25, zero outside
    the brain, smoothed to SPECT resolution. Correlation against it aligns the striatal hot spots and the brain's
    non-specific uptake with the labels. Until October 2026 it also held scalp and neck (0.12 from the T1 intensity) and
    was smoothed to ~30 mm instead of ``fwhm_mm`` (the voxel size was read off the diagonal of FastSurfer's LIA affine,
    1/3 mm), so the head outline dominated the fit; with it the ±2-voxel placement search sat on its edge in 81 % of
    scans and fan-beam axial scales shrank to a median 0.72."""
    lab = np.asanyarray(aparc_img.dataobj)
    syn = np.where(lab > 0, 0.25, 0.0).astype(np.float32)
    if head:          # the legacy template (``LEGACY``): non-brain head tissue at 0.12 from the T1 intensity
        t1 = np.asanyarray(t1_img.dataobj).astype(np.float32)
        syn[(t1 > 0.15 * np.percentile(t1[t1 > 0], 99)) & (lab == 0)] = 0.12
    syn[np.isin(lab, striatum)] = 1.0
    vox = float(np.mean(nib.affines.voxel_sizes(t1_img.affine)))
    return nib.Nifti1Image(ndimage.gaussian_filter(syn, sigma=fwhm_mm / 2.3548 / vox), t1_img.affine)


# The pre-October 2026 registration, kept as the fallback for scans the brain-only template fails (``quantify_with_fallback``):
# head-outline template at the ~30 mm it effectively had, correlation over the whole image, +-2-voxel placement. On the
# 236-participant validation set its SBRs agreed with Xing at r 0.75 on the 31 scans where the brain-only fit failed QC
# (r 0.35), and new-or-fallback gave putamen r 0.856 on all 159 matched scans against 0.785 for the old pipeline alone.
LEGACY = {"head": True, "fwhm_mm": 30.0, "mask": False, "search_vox": 2}


# Per-axis scale bounds of ``scale_fit`` registrations, in the T1 frame (x left-right, y anterior-posterior, z axial). A
# converging (fan-beam) collimator reconstructed as parallel-beam magnifies the transaxial plane by F / (F - R) (focal
# length F, radius of rotation R; non-circular orbits make x and y differ) and leaves the axial direction unmagnified,
# because the fan converges in the transaxial plane only. Upper transaxial bound 1.8 = R up to 0.45 F (the 8 September
# 2026 fits: FANB x 1.45-1.75, 5th-95th percentile; 3 of 148 above 1.8); axial 1 +- 0.15. The lower 0.85 also admits
# headers without a recorded collimator that may be parallel-hole. In those fits z sat at a median 0.72 and agreement
# with Xing fell as z left 1 (putamen r L/R, half of the scans nearer 1 vs the other half: FANB 0.96/0.95 vs 0.93/0.90,
# no collimator recorded 0.87/0.81 vs 0.74/0.62).
SCALE_BOUNDS = ((0.85, 1.8), (0.85, 1.8), (0.85, 1.15))


def register_to_t1(spect_img, t1_img, flip_lr=False, rz=0.0, search=False, aparc_img=None, scale_fit=False, legacy=False):
    """Rigid registration of the SPECT volume to the subject's T1 space via a brain-only synthetic DaT template
    (see ``synthetic_spect``), normalised correlation sampled inside the brain and a ~7 mm rim only (template >
    0.01, 4 % of the brain level), so scalp, face and neck uptake cannot steer it. The volume is already in patient
    axes up to the vendor's transaxial conventions, so the initial pose is the centre-of-mass alignment rotated by
    ``rz`` about the head axis (0 or pi), after an optional left/right mirror (``flip_lr``); ``search=True``
    tries both rz values and keeps the better metric (fallback for unknown acquisition configurations; the
    striatum is faint in patients, so a metric-based choice of pose is less reliable than the per-vendor
    rule). Multi-resolution refinement follows. A second, striatum-masked refinement stage was tried and
    removed: with the faint striata of patients it drifted (side-wise putamen agreement with PPMI fell from
    0.84 to 0.78 pooled, 0.64 to 0.27 on ADAC). With ``scale_fit`` (fan-beam) the per-axis scales are bounded by
    ``SCALE_BOUNDS``: a fit that leaves them is clamped and refitted rigidly at the bound, and flagged.
    Returns (transform, metric, moving image, {"t0": centre-of-mass start translation, "scale_at_bound": bool})."""
    import SimpleITK as sitk

    if aparc_img is None:
        raise ValueError("register_to_t1 needs the aparc segmentation to build the synthetic template")
    if flip_lr:
        spect_img = nib.Nifti1Image(np.asanyarray(spect_img.dataobj)[::-1], spect_img.affine)
    moving = _sitk_from_nib(_clean_spect(spect_img))
    template = synthetic_spect(t1_img, aparc_img, LEGACY["fwhm_mm"], head=True) if legacy else synthetic_spect(t1_img, aparc_img)
    fixed = sitk.Shrink(_sitk_from_nib(template), [2, 2, 2])
    mask = None if legacy else fixed > 0.01
    # rigid, or rigid + per-axis scale (no shear) when ``scale_fit``: fan-beam acquisitions (Marconi/Picker
    # Prism, collimator FANB) reconstructed as parallel-beam come out magnified transaxially, which a rigid fit
    # cannot absorb. Parallel-hole vendors with a correct pixel size lose a little accuracy with free scales, so
    # they stay rigid.
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.ScaleVersor3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    init = sitk.ScaleVersor3DTransform(init)
    scale_w = [1.0, 1.0, 1.0] if scale_fit else [0.0, 0.0, 0.0]

    def method(sampling):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        if mask is not None:
            reg.SetMetricFixedMask(mask)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, seed=ITK_SEED)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg

    def refine(start, weights):
        reg = method(0.3)
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetOptimizerWeights([1, 1, 1, 1, 1, 1] + weights)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([3, 2, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(start, inPlace=False)
        out = sitk.ScaleVersor3DTransform(start)
        out.SetParameters(reg.Execute(fixed, moving).GetParameters())
        return out, float(reg.GetMetricValue())

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
    at_bound = False
    try:
        tx, metric = refine(coarse, scale_w)
        lo, hi = np.array(SCALE_BOUNDS).T
        s = np.array(tx.GetScale())
        if scale_fit and ((s < lo) | (s > hi)).any():
            at_bound = True
            tx.SetScale(tuple(float(v) for v in np.clip(s, lo, hi)))
            tx, metric = refine(tx, [0.0, 0.0, 0.0])
    except RuntimeError:
        tx, metric = coarse, best_val
    return tx, metric, moving, {"t0": np.array(init.GetTranslation()), "scale_at_bound": at_bound}


def _ap_split(mask, ap):
    """Anterior / posterior halves of ``mask``, split at the median anterior coordinate of its voxels."""
    if ap is None or mask.sum() < 20:                 # coarse cameras (4.8 mm voxels) hold ~38 putamen voxels
        return {}
    cut = np.median(ap[mask])
    return {"ant": mask & (ap >= cut), "post": mask & (ap < cut)}


# Half-width of the hottest-placement search, in mm. It was +-2 voxels (6-10 mm by camera) until October 2026, and in
# 81 % of the 8 September 2026 scans the best placement sat on that edge (91 % in the lowest putamen-SBR tertile); putamen
# agreement with Invicro was r 0.925 inside the window and 0.77 on its edge, so the search was absorbing real
# misregistration that the window cut short. With perfect registration a noisy synthetic phantom moves it by under
# half a voxel on average. 12 mm (about one system FWHM) cannot carry the whole-striatum mask onto another structure;
# ``shift_edge`` flags placements that still reach the edge and ``manifest.QC["dat"]`` fails them.
SEARCH_MM = 12.0


def sbr_from_arrays(S, L, dilate=0, ref_dilate=0, search_vox=None, ap=None, spacing_mm=None):
    """SBRs from a counts array ``S`` and a co-registered label array ``L`` (same grid, any orientation).
    Striatal masks may be dilated (SPECT resolution ~10 mm); the occipital reference (cortical labels,
    optionally dilated into the brain, striatum excluded); a translation search (+-``search_vox`` voxels per axis;
    None = ``SEARCH_MM`` at ``spacing_mm``) maximises the striatal counts, mimicking ROI placement on the hottest
    striatal region. The chosen shift is reported per array axis in mm (``shift_{x,y,z}_mm`` for arrays indexed
    (z, y, x); right, anterior, superior through ``sbr_with_transform``) and ``shift_edge`` says whether any axis
    reached the window's edge, where the true optimum may lie beyond it.
    With ``ap`` (per-voxel anterior coordinate) every striatal ROI is also split into anterior / posterior
    halves (``_ant`` / ``_post``: the posterior putamen loses dopamine transporter first). A second SBR set
    (``sbrwm_*``) uses cerebral white matter at least WM_MARGIN_MM from the striatum as the reference."""
    sp = spacing_mm or 3.9
    search_vox = int(SEARCH_MM / sp + 0.5) if search_vox is None else search_vox
    brain = L > 0
    striatum_ids = {"caudate_l": [CAUDATE_L], "caudate_r": [CAUDATE_R], "putamen_l": [PUTAMEN_L], "putamen_r": [PUTAMEN_R]}
    masks = {k: ndimage.binary_dilation(np.isin(L, v), iterations=dilate) if dilate else np.isin(L, v) for k, v in striatum_ids.items()}
    striatum = np.any(list(masks.values()), axis=0)
    for k in list(masks):
        masks.update({f"{k}_{s}": m for s, m in _ap_split(masks[k], ap).items()})
    occ = ndimage.binary_dilation(np.isin(L, OCCIPITAL), iterations=ref_dilate) & brain & ~striatum if ref_dilate else np.isin(L, OCCIPITAL)
    far = ndimage.distance_transform_edt(~striatum) > WM_MARGIN_MM / sp if striatum.any() else np.ones_like(striatum)
    wm = np.isin(L, WM) & far
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
        occ, wm = (np.roll(m, best_shift, axis=(0, 1, 2)) for m in (occ, wm))
    means = {k: float(S[m].mean()) if m.sum() >= (10 if k.endswith(("_ant", "_post")) else 20) else np.nan for k, m in masks.items()}
    means["occipital"] = float(S[occ].mean()) if occ.sum() >= 50 else np.nan
    means["wm"] = float(S[wm].mean()) if wm.sum() >= 50 else np.nan
    sbr = lambda v, ref: (v / ref - 1.0) if (ref and np.isfinite(ref) and ref > 0) else np.nan
    out = {f"sbr_{k}": sbr(means[k], means["occipital"]) for k in masks}
    out.update({f"sbrwm_{k}": sbr(means[k], means["wm"]) for k in masks})
    out.update({f"mean_{k}": v for k, v in means.items()})
    out["n_label_voxels"] = int(striatum.sum())
    out["n_ref_voxels"] = int(occ.sum())
    out["n_wm_voxels"] = int(wm.sum())
    out["shift_vox"] = float(np.sqrt(sum(d * d for d in best_shift)))
    out["shift_z_mm"], out["shift_y_mm"], out["shift_x_mm"] = (float(d * sp) for d in best_shift)
    out["shift_edge"] = bool(search_vox > 0 and max(abs(d) for d in best_shift) == search_vox)
    return out


def _anterior_map(ref, target, tx):
    """Anterior (-LPS y) coordinate of ``ref``'s space at every voxel of ``target``, through ``tx``; (z, y, x) array."""
    import SimpleITK as sitk

    pts = sitk.PhysicalPointSource(sitk.sitkVectorFloat32, ref.GetSize(), ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection())
    return -sitk.GetArrayFromImage(sitk.Resample(sitk.VectorIndexSelectionCast(pts, 1), target, tx, sitk.sitkLinear, 0.0))


def sbr_with_transform(spect_img, aparc_img, tx_inv, flip_lr=False, dilate=0, ref_dilate=0, search_vox=None):
    """SBRs of ``spect_img`` given ``tx_inv``, the SPECT -> T1 mapping (inverse of the registration's
    fixed -> moving transform): FastSurfer labels and the T1 anterior coordinate are resampled onto the
    SPECT grid, then ``sbr_from_arrays`` on the un-thresholded counts."""
    import SimpleITK as sitk

    raw = spect_img if not flip_lr else nib.Nifti1Image(np.asanyarray(spect_img.dataobj)[::-1], spect_img.affine)
    moving = _sitk_from_nib(raw)
    labels = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc_img.dataobj).astype(np.float32), aparc_img.affine))
    L = sitk.GetArrayFromImage(sitk.Resample(labels, moving, tx_inv, sitk.sitkNearestNeighbor, 0.0))
    return sbr_from_arrays(sitk.GetArrayFromImage(moving), L, dilate=dilate, ref_dilate=ref_dilate, search_vox=search_vox,
                           ap=_anterior_map(labels, moving, tx_inv), spacing_mm=float(moving.GetSpacing()[0]))


def transform_from_row(row):
    """The stored registration of a results row (``reg_params`` / ``reg_center``) as an AffineTransform,
    T1 (fixed) -> SPECT (moving). ScaleVersor3D itself has no inverse in ITK; the equivalent affine does."""
    import SimpleITK as sitk

    sv = sitk.ScaleVersor3DTransform()
    sv.SetCenter([float(x) for x in str(row["reg_center"]).split()])
    sv.SetParameters([float(x) for x in str(row["reg_params"]).split()])
    tx = sitk.AffineTransform(3)
    tx.SetCenter(sv.GetCenter())
    tx.SetMatrix(sv.GetMatrix())
    tx.SetTranslation(sv.GetTranslation())
    return tx


def quantify(spect_img, t1_img, aparc_img, flip_lr=False, rz=0.0, search=False, dilate=0, ref_dilate=0, search_vox=None, scale_fit=False,
             legacy=False):
    """Register SPECT -> T1 space (synthetic-template correlation), resample the FastSurfer labels onto the
    SPECT grid and compute SBRs (see ``sbr_from_arrays``). Returns SBRs, ROI means, the registration
    metric (negative normalised correlation; more negative is better) and QC fields: the transform's rotation
    (``reg_rot_deg`` in total, ``reg_rot_{x,y,z}_deg`` as xyz Euler angles about the T1's LPS axes), its translation
    beyond the centre-of-mass start (``reg_t{x,y,z}_mm``, LPS), its scales and ``reg_scale_at_bound``."""
    from scipy.spatial.transform import Rotation

    tx, metric, moving, info = register_to_t1(spect_img, t1_img, flip_lr, rz=rz, search=search, aparc_img=aparc_img, scale_fit=scale_fit,
                                              legacy=legacy)
    search_vox = LEGACY["search_vox"] if legacy and search_vox is None else search_vox
    p = tx.GetParameters()       # ScaleVersor3D: versor xyz, translation xyz, scale xyz (fixed -> moving)
    reg = {"reg_params": " ".join(f"{v:.6g}" for v in p), "reg_center": " ".join(f"{v:.6g}" for v in tx.GetFixedParameters()[:3])}
    # through the stored values, so --requantify reproduces these SBRs exactly
    out = sbr_with_transform(spect_img, aparc_img, transform_from_row(reg).GetInverse(), flip_lr=flip_lr, dilate=dilate,
                             ref_dilate=ref_dilate, search_vox=search_vox)
    rot = Rotation.from_quat([*p[:3], np.sqrt(max(0.0, 1.0 - sum(v * v for v in p[:3])))])
    out["reg_metric"] = metric
    out["reg_template"] = "legacy_head" if legacy else "brain"
    out["flip_lr"] = bool(flip_lr)
    out.update(reg)
    out["reg_scale_x"], out["reg_scale_y"], out["reg_scale_z"] = (float(v) for v in p[6:9])
    out["reg_scale_at_bound"] = info["scale_at_bound"]
    out["reg_rot_deg"] = float(np.degrees(rot.magnitude()))
    out["reg_rot_x_deg"], out["reg_rot_y_deg"], out["reg_rot_z_deg"] = (float(v) for v in rot.as_euler("xyz", degrees=True))
    out["reg_tx_mm"], out["reg_ty_mm"], out["reg_tz_mm"] = (float(v) for v in np.array(p[3:6]) - info["t0"])
    return out


def brain_fit_failed(q):
    """The brain-only fit failed by the manifest's rules: correlation under 0.2, rotation over 30 deg, or the hottest
    placement on the edge of its window."""
    return bool(-q["reg_metric"] < 0.2 or q.get("reg_rot_deg", 0) > 30 or q.get("shift_edge", False))


def quantify_with_fallback(spect_img, t1_img, aparc_img, **kw):
    """``quantify`` with the brain-only template; when that fit fails (``brain_fit_failed``), the ``LEGACY`` registration,
    kept when its own correlation reaches the legacy bar (0.4). ``reg_template`` says which one the row carries and
    ``reg_fallback`` that the brain-only fit failed."""
    q = quantify(spect_img, t1_img, aparc_img, **kw)
    if not brain_fit_failed(q):
        return {**q, "reg_fallback": False}
    leg = quantify(spect_img, t1_img, aparc_img, legacy=True, **kw)
    return {**(leg if -leg["reg_metric"] >= 0.4 else q), "reg_fallback": True}


def requantify_row(row, fastsurfer_subject_dir):
    """Recompute the SBR fields of a finished results row from its saved reconstruction (``nifti``) and stored
    registration, without registering again: adds ROI / reference columns introduced after a full run."""
    aparc = nib.load(Path(fastsurfer_subject_dir) / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
    flip = str(row.get("flip_lr", False)).lower() == "true"
    out = sbr_with_transform(nib.load(row["nifti"]), aparc, transform_from_row(row).GetInverse(), flip_lr=flip)
    return {**row, **out}


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
        img = to_nifti(vol, p["spacing_mm"])
    except ValueError:
        itype = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        if "TOMO" in itype and getattr(ds, "RotationInformationSequence", None):
            raise
        img, spacing = read_volume(ds)  # stored as a reconstructed stack: placed by its DICOM geometry, or refused
        if fwhm_mm:
            img = nib.Nifti1Image(ndimage.gaussian_filter(img.get_fdata(dtype=np.float32), sigma=fwhm_mm / 2.3548 / spacing), img.affine)
        row["kind"] = "stored_volume"
        row["hdr_manufacturer"] = str(getattr(ds, "Manufacturer", ""))
        row["hdr_model"] = str(getattr(ds, "ManufacturerModelName", ""))
    nib.save(img, nifti)
    row["nifti"] = str(nifti)
    if fastsurfer_subject_dir:
        mri = Path(fastsurfer_subject_dir) / "mri"
        t1, aparc = nib.load(mri / "orig.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
        row.update(quantify_with_fallback(img, t1, aparc, flip_lr=flip_lr, scale_fit=row.get("hdr_scale_fit", False)))
        row["fs_image_id"] = Path(fastsurfer_subject_dir).name   # the T1 the transform refers to (needed to requantify)
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


def _requant_job(args):
    row, fs_dir = args
    if row.get("error") or not isinstance(row.get("reg_params"), str) or not fs_dir:
        return row
    try:
        import SimpleITK as sitk
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
        return requantify_row(row, fs_dir)
    except Exception as e:
        return {**row, "error": f"requantify: {str(e)[:150]}"}


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
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from .batch import fastsurfer_by_patno, run_batch

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
    ap.add_argument("--requantify", action="store_true", help="recompute the SBR columns of every finished row from its saved reconstruction and "
                    "stored registration (adds ROI/reference columns without re-registering); the previous CSV is kept as *.pre_requant.csv")
    a = ap.parse_args(argv)

    idx = pd.read_csv(a.index, dtype={"image_id": str})
    idx = idx[idx["kind"] == "TOMO"].sort_values("frames", ascending=False)
    idx = _prefer_photopeak_member(idx).drop_duplicates("patno")
    if a.patnos:
        keep = {int(x) for x in Path(a.patnos).read_text().split()}
        idx = idx[idx["patno"].isin(keep)]
    fs_root = Path(a.fastsurfer_dir)
    fs_by_patno = fastsurfer_by_patno(a.sessions, fs_root)     # earliest finished T1, as for the MRI modalities and the manifest
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "datscan_sbr.csv"
    if a.requantify:
        d = pd.read_csv(out_csv, dtype={"image_id": str})
        d["error"] = d["error"].fillna("")
        def _fs_dir(r):   # the T1 recorded with the row; older tables fall back to the subject's earliest FastSurfer session
            if isinstance(r.get("fs_image_id"), str) and (fs_root / r["fs_image_id"] / "mri").exists():
                return str(fs_root / r["fs_image_id"])
            return fs_by_patno.get(int(r["patno"]))
        only = {int(x) for x in Path(a.patnos).read_text().split()} if a.patnos else None
        jobs = [(r, _fs_dir(r) if (only is None or int(r["patno"]) in only) else None) for r in d.to_dict("records")]   # fs_dir None = keep as is
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, row in enumerate(ex.map(_requant_job, jobs[:a.limit] if a.limit else jobs, chunksize=4), start=1):
                rows.append(row)
                if i % 100 == 0:
                    print(f"{i}/{len(jobs)}", flush=True)
        if a.limit:                                   # a sample run must not replace the full table
            out_csv = out_csv.with_name("datscan_sbr_requant_sample.csv")
        else:
            bak = out_csv.with_suffix(".pre_requant.csv")
            if not bak.exists():
                out_csv.rename(bak)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"requantified {sum(1 for r in rows if 'sbr_putamen_l_post' in r)}/{len(rows)} rows ->", out_csv)
        return
    done = set(pd.read_csv(out_csv, dtype={"image_id": str})["image_id"]) if out_csv.exists() and out_csv.stat().st_size else set()
    jobs = []
    for r in idx.itertuples():
        if r.image_id in done:
            continue
        zp = r.zip if Path(r.zip).exists() else str(Path(a.zip_dir) / Path(r.zip).name)
        jobs.append((zp, r.member, out_dir / "nifti", fs_by_patno.get(int(r.patno)), a.flip_lr == "true", a.attenuation))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(done)} series already done", flush=True)
    # the shared runner widens the header to every column seen, so a failed first series cannot truncate the table
    run_batch(jobs, _job, out_csv, workers=a.workers, log_every=20)


if __name__ == "__main__":
    main()


def deming(x, y, delta=1.0):
    """Deming regression of ``y`` on ``x`` (errors in both; ``delta`` = error variance of y / error variance of x, 1 =
    orthogonal regression) -> (slope, intercept). Unlike OLS it does not shrink the slope by x's measurement error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx, syy, sxy = x.var(), y.var(), np.mean((x - x.mean()) * (y - y.mean()))
    slope = (syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    return slope, y.mean() - slope * x.mean()


def calibrate(d, targets, min_ref=15, by="hdr_model", method="deming", delta=1.0):
    """Map PIE SBR columns onto published SBRs per camera model (column ``by``). ``targets`` maps a PIE column (e.g.
    ``sbrwm_putamen_l``) to the published column on the same rows (e.g. Xing's ``PUTAMEN_L_REF_CWM``; keep reference
    regions matched: CWM to ``sbrwm_*``, occipital to ``sbr_*``). The map is fitted on the rows that have both values
    (blank QC failures first), per model with >= ``min_ref`` of them, else on all models pooled, and applied to every
    row (so cohorts without published values get calibrated values too). ``method``: ``deming`` (linear, errors in both
    variables, see ``deming``) or ``equipercentile`` (each value goes to the published value at the same percentile of
    the reference scans: the mapped reference rows reproduce the published distribution exactly; values outside the
    reference range go to its ends). Both keep the published spread; the OLS fit used until October 2026 (never called
    by the pipeline) regressed towards the mean, leaving the mapped putamen SD at 0.82 of the published one. Adds
    ``<column>_cal`` and ``calibration`` (the model or ``POOLED``, from the first target)."""
    import pandas as pd

    out = d.copy()
    for i, (col, target) in enumerate(targets.items()):
        x_all, y_all = pd.to_numeric(out[col], errors="coerce"), pd.to_numeric(out[target], errors="coerce")
        ref = x_all.notna() & y_all.notna()
        out[f"{col}_cal"] = np.nan
        for v, g in out.groupby(by, dropna=False):
            r = ref & (out[by] == v)
            src = v if r.sum() >= min_ref else "POOLED"
            x, y = x_all[r if src == v else ref], y_all[r if src == v else ref]
            if method == "equipercentile":
                cal = np.interp(x_all[g.index], np.sort(x), np.sort(y))
            else:
                slope, intercept = deming(x, y, delta)
                cal = intercept + slope * x_all[g.index]
            out.loc[g.index, f"{col}_cal"] = cal
            if i == 0:
                out.loc[g.index, "calibration"] = src
    return out
