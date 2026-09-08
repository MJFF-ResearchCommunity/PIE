"""
Diffusion MRI (PPMI DTI download) -> per-subject free-water / tensor features in subcortical and nigral ROIs.

PPMI's diffusion data come in two generations: PPMI-1 single-shell (b = 1000, 32-64 directions; Siemens mosaics,
GE and Philips multi-file series, Philips as opposite-phase-encoding LR/RL pairs) and PPMI-2 three-shell Siemens
Prisma (b = 700/1000/2000, 64 directions each, plus reverse-phase b0s). Pipeline per subject:

1. `convert`      dcm2niix on every diffusion series (bval/bvec/json), derived series (ADC, "Reg -") dropped.
2. `assemble`     one DWI dataset: same-geometry, same-phase-encoding runs concatenated (the PPMI-2 shells);
                  opposite-phase runs (Philips LR/RL) are not merged, the run with more directions is used.
3. `preprocess`   brain mask (median Otsu on b0), rigid volume-to-b0 motion correction (SimpleITK MI).
                  No topup/eddy (FSL is not installed): eddy-current and susceptibility distortions remain.
4. `fit`          DTI (weighted least squares, b <= 1000) for FA/MD over the brain; free-water bi-tensor model for
                  FW and tissue FA inside the ROI neighbourhood: DIPY's multi-shell NLS (Hoy et al. 2014) for the
                  PPMI-2 shells, a bounded voxel-wise fit with a tissue-diffusivity prior for single-shell PPMI-1
                  data (`fw_method` records which; single-shell free-water is ill-posed and closer to MD).
5. `register`     mean b0 -> conformed T1 (rigid, mutual information; no susceptibility correction);
                  T1 -> MNI152NLin2009cAsym affine (brain-masked) to bring the CIT168 subcortical atlas
                  (Pauli 2017: SNc, SNr, RN, STN, VTA, ...) into subject space alongside the FastSurfer labels.
6. `features`     mean FA, MD, FW, FAt per ROI (left/right; the substantia nigra also split into anterior and
                  posterior halves, the posterior half being the free-water marker of nigral degeneration).

    venv_imaging/bin/python -m pie.imaging.dwi --zips <DTI zips> --collection <LONI csv> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/dwi --workers 6   # -> dwi_features.csv
"""

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .convert import DCM2NIIX
from .datscan import _sitk_from_nib

log = logging.getLogger("pie.dwi")

DERIVED_SUFFIXES = ("_ADC", "_FA", "_TRACEW", "_ColFA", "_TENSOR", "_EXP")  # dcm2niix suffixes of vendor-derived maps
# FastSurfer (aseg) labels -> ROI names; left/right pairs
FS_ROIS = {"thalamus": (10, 49), "caudate": (11, 50), "putamen": (12, 51), "pallidum": (13, 52),
           "cerebellum_wm": (7, 46), "cerebral_wm": (2, 41)}
FS_SINGLE = {"brainstem": (16,)}
# CIT168 / Pauli 2017 deterministic labels (1-based, nilearn order)
PAULI = ["Pu", "Ca", "NAC", "EXA", "GPe", "GPi", "SNc", "RN", "SNr", "PBP", "VTA", "VeP", "HN", "HTH", "MN", "STH"]
PAULI_ROIS = {"snc": ("SNc",), "snr": ("SNr",), "sn": ("SNc", "SNr"), "red_nucleus": ("RN",), "stn": ("STH",), "vta": ("VTA",),
              "gpe": ("GPe",), "gpi": ("GPi",), "nac": ("NAC",)}
METRICS = ("fa", "md", "fw", "fat")


# ------------------------------------------------------------------------------------------ index / convert
def index_dwi(zips):
    """Series index with a ``derived`` flag (ADC / registered maps), see ``batch.index_series``."""
    from .batch import index_series

    idx = index_series(zips)
    idx["derived"] = ~_flag_dwi(idx)
    return idx


def convert(zip_path, prefix, out_dir):
    """dcm2niix on one series; returns (nii, bval, bvec, json) tuples, vendor-derived maps dropped."""
    from .batch import convert_series

    runs = []
    for nii in convert_series(zip_path, prefix, out_dir):
        base = nii[:-7]
        if base.endswith(DERIVED_SUFFIXES) or not Path(base + ".bval").exists():
            continue
        runs.append((nii, base + ".bval", base + ".bvec", base + ".json"))
    return runs


def assemble(runs):
    """Choose/concatenate runs into one dataset. Returns dict(data, affine, bvals, bvecs, meta, n_runs)."""
    loaded = []
    for nii, bval, bvec, js in runs:
        img = nib.load(nii)
        if img.ndim != 4:
            continue
        b = np.loadtxt(bval).ravel()
        v = np.loadtxt(bvec).reshape(3, -1)
        if len(b) != img.shape[3]:
            continue
        meta = json.load(open(js)) if Path(js).exists() else {}
        loaded.append((img, b, v, meta))
    b0_runs = [x for x in loaded if (x[1] > 50).sum() < 6]     # b0-only series (reverse-phase for topup)
    loaded = [x for x in loaded if (x[1] > 50).sum() >= 6]
    if not loaded:
        raise ValueError("no diffusion-weighted run")
    # group by geometry + phase encoding; take the group with the most directions
    groups = {}
    for img, b, v, meta in loaded:
        key = (img.shape[:3], tuple(np.round(img.header.get_zooms()[:3], 2)), meta.get("PhaseEncodingDirection"))
        groups.setdefault(key, []).append((img, b, v, meta))
    best = max(groups.values(), key=lambda g: sum((x[1] > 50).sum() for x in g))
    data = np.concatenate([np.asanyarray(x[0].dataobj).astype(np.float32) for x in best], axis=3)
    bvals = np.concatenate([x[1] for x in best])
    bvecs = np.concatenate([x[2] for x in best], axis=1)
    meta = best[0][3]
    pe = meta.get("PhaseEncodingDirection")
    rev = [x for x in b0_runs if x[0].shape[:3] == best[0][0].shape[:3] and x[3].get("PhaseEncodingDirection") not in (None, pe)
           and x[3].get("PhaseEncodingDirection", "")[:1] == (pe or "")[:1]]
    rev_b0 = np.concatenate([np.asanyarray(x[0].dataobj).astype(np.float32).reshape(x[0].shape[:3] + (-1,)) for x in rev], axis=3) if rev and pe else None
    return {"data": data, "affine": best[0][0].affine, "bvals": bvals, "bvecs": bvecs, "meta": meta, "n_runs": len(best),
            "shells": sorted(set(int(round(x / 100.0)) * 100 for x in bvals if x > 50)), "rev_b0": rev_b0}


# ------------------------------------------------------------------------------------------ preprocess / fit
FSLDIR = os.environ.get("FSLDIR") or (str(Path.home() / "fsl") if (Path.home() / "fsl" / "bin" / "topup").exists() else None)


def susceptibility_correct(ds, work_dir, threads=2):
    """FSL topup on the mean b0 of the main series and the mean reverse-phase b0, then applytopup (Jacobian
    modulation) on every volume. Needs PhaseEncodingDirection and TotalReadoutTime (dcm2niix json) and a
    reverse-phase b0 series (PPMI-2 Prisma, GE 'Ax DWI B-0 A/P'); otherwise returns the dataset unchanged.
    eddy was measured at 26 min/subject on the RTX 2080 for the three-shell data and is not run."""
    meta, rev = ds["meta"], ds.get("rev_b0")
    pe, trt = meta.get("PhaseEncodingDirection"), meta.get("TotalReadoutTime")
    if not FSLDIR or rev is None or not pe or not trt:
        return dict(ds, topup=False)
    work = Path(work_dir) / "topup"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FSLDIR=FSLDIR, PATH=f"{FSLDIR}/bin:" + os.environ.get("PATH", ""), FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS=str(threads))
    b0_main = ds["data"][..., ds["bvals"] <= 50].mean(axis=3)
    b0_rev = rev.mean(axis=3)
    nib.save(nib.Nifti1Image(np.stack([b0_main, b0_rev], axis=3), ds["affine"]), work / "b0_pair.nii.gz")
    vec = np.array({"i": [1, 0, 0], "j": [0, 1, 0], "k": [0, 0, 1]}[pe[0]]) * (-1 if pe.endswith("-") else 1)
    fmt = lambda v: " ".join(str(int(x)) for x in v)
    (work / "acqparams.txt").write_text(f"{fmt(vec)} {trt}\n{fmt(-vec)} {trt}\n")
    nib.save(nib.Nifti1Image(ds["data"], ds["affine"]), work / "dwi.nii.gz")
    r = subprocess.run(["topup", f"--imain={work / 'b0_pair.nii.gz'}", f"--datain={work / 'acqparams.txt'}", "--config=b02b0.cnf",
                        f"--out={work / 'topup'}", f"--nthr={threads}"], env=env, capture_output=True, text=True)
    if r.returncode:
        log.warning("topup failed: %s", r.stderr[-300:])
        return dict(ds, topup=False)
    r = subprocess.run(["applytopup", f"--imain={work / 'dwi.nii.gz'}", f"--datain={work / 'acqparams.txt'}", "--inindex=1",
                        f"--topup={work / 'topup'}", "--method=jac", f"--out={work / 'dwi_unwarped'}"], env=env, capture_output=True, text=True)
    if r.returncode:
        log.warning("applytopup failed: %s", r.stderr[-300:])
        return dict(ds, topup=False)
    out = nib.load(work / "dwi_unwarped.nii.gz")
    data = np.asanyarray(out.dataobj).astype(np.float32)
    data[data < 0] = 0
    shutil.rmtree(work, ignore_errors=True)
    return dict(ds, data=data, topup=True)


def _sitk_native(arr, affine):
    """SimpleITK image from an (x, y, z) array with a NIfTI affine, keeping the stored voxel order (no reorientation),
    so arrays read back with GetArrayFromImage are simply (z, y, x) of the same grid."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr, (2, 1, 0)).astype(np.float32)))
    M = affine[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    flip = np.diag([-1.0, -1.0, 1.0])  # RAS -> LPS
    img.SetSpacing(tuple(float(x) for x in spacing))
    img.SetOrigin(tuple(float(x) for x in flip @ affine[:3, 3]))
    img.SetDirection(tuple(float(x) for x in (flip @ (M / spacing)).ravel()))
    return img


def _register_volume(fixed, moving, sampling=0.1, iterations=50):
    """Rigid MI registration (SimpleITK, 2-level pyramid) of one DWI volume to the b0 reference. Returns the Euler transform."""
    import SimpleITK as sitk

    tx = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=iterations, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2])
    reg.SetSmoothingSigmasPerLevel([2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    try:
        reg.Execute(fixed, moving)
    except RuntimeError:
        pass
    return tx


def preprocess(ds):
    """Brain mask + rigid motion correction of every volume to the mean b0 (SimpleITK mutual information, ~0.7 s per
    volume against ~5 s for DIPY's affine pipeline). b-vectors are not rotated: PPMI head rotations are ~1 degree,
    below what changes the rotation-invariant scalars used here. Returns the dataset with corrected data, mask,
    b0 and motion summaries (mm translation per volume)."""
    import SimpleITK as sitk
    from dipy.segment.mask import median_otsu

    b = ds["bvals"]
    data = ds["data"]
    b0s = b <= 50
    b0 = data[..., b0s].mean(axis=3)
    _, mask = median_otsu(b0, median_radius=3, numpass=2)
    mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=2))
    ref = _sitk_native(b0, ds["affine"])
    cdata = np.empty_like(data)
    fd, rot = [], []
    for i in range(data.shape[3]):
        mov = _sitk_native(data[..., i], ds["affine"])
        tx = _register_volume(ref, mov)
        cdata[..., i] = np.transpose(sitk.GetArrayFromImage(sitk.Resample(mov, ref, tx, sitk.sitkLinear, 0.0)), (2, 1, 0))
        p = np.array(tx.GetParameters())
        fd.append(float(np.linalg.norm(p[3:6])))
        rot.append(float(np.degrees(np.linalg.norm(p[:3]))))
    return dict(ds, data=cdata, mask=mask, b0=cdata[..., b0s].mean(axis=3),
                motion_mm_mean=float(np.mean(fd)), motion_mm_max=float(np.max(fd)), rotation_deg_max=float(np.max(rot)))


D_WATER = 3.0e-3  # mm^2/s at 37 C
MD_TISSUE = 0.7e-3


def _fw_single_shell(data, bvals, bvecs, mask, prior_weight=0.05):
    """Voxel-wise bi-tensor fit for single-shell data: S = S0 [(1-f) exp(-b g'Dg) + f exp(-b D_WATER)], D = L L'
    (Cholesky, positive definite), f in [0, 0.95]. Single-shell free-water is ill-posed (Pasternak 2009), so the
    tissue mean diffusivity carries a weak prior towards MD_TISSUE (Pasternak's initialisation used as a penalty;
    no spatial regularisation). Returns (f, tissue FA) arrays."""
    from scipy.optimize import least_squares

    from dipy.reconst.dti import fractional_anisotropy

    g = bvecs.T
    B = np.c_[g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]] * bvals[:, None]
    water = np.exp(-bvals * D_WATER)
    f_map = np.zeros(mask.shape, dtype=np.float32)
    fat_map = np.zeros(mask.shape, dtype=np.float32)
    idx = np.argwhere(mask)
    scale = 1e-3

    def resid(p, y):
        logS0, f = p[0], p[1]
        L = np.zeros((3, 3))
        L[np.tril_indices(3)] = p[2:8]
        D = L @ L.T * scale
        d6 = np.array([D[0, 0], D[1, 1], D[2, 2], D[0, 1], D[0, 2], D[1, 2]])
        tissue = np.exp(-B @ d6)
        pred = np.exp(logS0) * ((1 - f) * tissue + f * water)
        md = np.trace(D) / 3
        return np.r_[(pred - y) / np.exp(logS0), prior_weight * np.sqrt(len(y)) * (md - MD_TISSUE) / MD_TISSUE]

    for (i, j, k) in idx:
        y = data[i, j, k].astype(float)
        s0 = max(y[bvals <= 50].mean(), 1e-3)
        # WLS tensor for the initial D and f (linear interpolation of MD between tissue and water)
        w = np.maximum(y, 1e-3)
        coef, *_ = np.linalg.lstsq(np.c_[-B, np.ones(len(y))] * w[:, None], np.log(w) * w, rcond=None)
        d6 = np.clip(coef[:6], -5e-3, 5e-3)
        Dm = np.array([[d6[0], d6[3], d6[4]], [d6[3], d6[1], d6[5]], [d6[4], d6[5], d6[2]]])
        ev, evec = np.linalg.eigh(Dm)
        ev = np.clip(ev, 1e-5, 4e-3)
        md = ev.mean()
        f0 = float(np.clip((md - MD_TISSUE) / (D_WATER - MD_TISSUE), 0.02, 0.9))
        Dt = evec @ np.diag(np.clip(ev * (1 - f0) + 0.0, 1e-5, 3e-3)) @ evec.T / scale  # tissue tensor guess (scaled)
        try:
            L0 = np.linalg.cholesky(Dt + 1e-6 * np.eye(3))
        except np.linalg.LinAlgError:
            L0 = np.linalg.cholesky(np.eye(3) * MD_TISSUE / scale)
        p0 = np.r_[np.log(s0), f0, L0[np.tril_indices(3)]]
        lo = np.r_[-np.inf, 0.0, [-np.inf] * 6]
        hi = np.r_[np.inf, 0.95, [np.inf] * 6]
        try:
            sol = least_squares(resid, p0, args=(y,), bounds=(lo, hi), max_nfev=60, xtol=1e-4, ftol=1e-4)
            p = sol.x
        except Exception:
            p = p0
        L = np.zeros((3, 3))
        L[np.tril_indices(3)] = p[2:8]
        evt = np.linalg.eigvalsh(L @ L.T * scale)
        f_map[i, j, k] = p[1]
        fat_map[i, j, k] = fractional_anisotropy(np.clip(evt, 1e-9, None)[None])[0]
    return f_map, fat_map


def fit_models(ds, fw_mask=None):
    """FA/MD (WLS tensor, b <= 1000, brain mask) and free-water FW / tissue FA inside ``fw_mask`` (or the brain):
    DIPY's multi-shell NLS (Hoy et al. 2014) when >= 2 non-zero shells, else the single-shell fit above."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    b, v, data, mask = ds["bvals"], ds["bvecs"], ds["data"], ds["mask"]
    sel = b <= 1050
    gt = gradient_table(b[sel], bvecs=v[:, sel], b0_threshold=50)
    tf = TensorModel(gt, fit_method="WLS").fit(data[..., sel], mask=mask)
    out = {"fa": np.nan_to_num(tf.fa).astype(np.float32), "md": np.nan_to_num(tf.md).astype(np.float32)}
    fmask = mask if fw_mask is None else (mask & fw_mask)
    shells = sorted(set(int(round(x / 100.0)) * 100 for x in b if x > 50))
    if len(shells) >= 2:
        from dipy.reconst.fwdti import FreeWaterTensorModel

        sel2 = b <= 2050
        gt2 = gradient_table(b[sel2], bvecs=v[:, sel2], b0_threshold=50)
        fw = FreeWaterTensorModel(gt2).fit(data[..., sel2], mask=fmask)
        f, fat = np.nan_to_num(fw.f), np.nan_to_num(fw.fa)
        method = "multishell_nls"
    else:
        f, fat = _fw_single_shell(data[..., sel], b[sel], v[:, sel], fmask)
        method = "singleshell_prior"
    out["fw"] = np.where(fmask, f, np.nan).astype(np.float32)   # NaN where the model was not fitted
    out["fat"] = np.where(fmask, fat, np.nan).astype(np.float32)
    out["fw_method"] = method
    return out


# ------------------------------------------------------------------------------------------ registration
def _brain(img, mask_img, mm=2.0):
    """Brain-masked copy of ``img`` as a SimpleITK image resampled to ``mm`` isotropic (registration at DWI resolution)."""
    import SimpleITK as sitk

    d = np.asanyarray(img.dataobj).astype(np.float32)
    m = np.asanyarray(mask_img.dataobj) > 0
    out = _sitk_from_nib(nib.Nifti1Image(np.where(m, d, 0), img.affine))
    return sitk.Shrink(out, [max(1, int(round(mm / sp))) for sp in out.GetSpacing()])


def register_b0_to_t1(b0_img, t1_img, t1_mask_img):
    """Rigid (mutual information) registration of the mean b0 to the brain-masked conformed T1 at 2 mm.
    Returns (transform fixed(T1)->moving(b0), metric). EPI susceptibility distortion is not corrected: a
    T1-guided B-spline restricted to the phase-encoding axis was tried and cost 9 min per subject without
    improving the fit; FSL topup on the PPMI-2 reverse-phase b0s is the proper upgrade."""
    import SimpleITK as sitk

    fixed = _brain(t1_img, t1_mask_img)
    # native geometry: DWI acquisitions are oblique (AC-PC angled) and `_sitk_from_nib` keeps only the diagonal
    moving = _sitk_native(np.asanyarray(b0_img.dataobj).astype(np.float32), b0_img.affine)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.Euler3DTransform(init), inPlace=False)
    tx = reg.Execute(fixed, moving)
    return tx, float(reg.GetMetricValue())


def register_t1_to_mni(t1_img, t1_mask_img, cache_path=None):
    """Affine T1 (brain) -> MNI152NLin2009cAsym (nilearn template, brain-masked). Returns (transform fixed(MNI)->moving(T1),
    metric). With ``cache_path`` (e.g. <fastsurfer subject>/mri/transforms/t1_to_mni152_affine.tfm) the transform is
    read back if present and written after fitting, so every modality of a subject uses the same atlas mapping."""
    import SimpleITK as sitk
    from nilearn import datasets

    if cache_path is not None and Path(cache_path).exists():
        return sitk.ReadTransform(str(cache_path)), float("nan")

    mni = datasets.load_mni152_template(resolution=2)
    mask = datasets.load_mni152_brain_mask(resolution=2)
    fixed = _brain(mni, mask)
    moving = _brain(t1_img, t1_mask_img)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.AffineTransform(init), inPlace=False)
    tx = reg.Execute(fixed, moving)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(tx, str(cache_path))
    return tx, float(reg.GetMetricValue())


def mni_cache_path(fastsurfer_dir):
    return Path(fastsurfer_dir) / "mri" / "transforms" / "t1_to_mni152_affine.tfm"


def labels_to_dwi(label_img, target, chain):
    """Nearest-neighbour resampling of a label image onto the DWI grid (``target``: SimpleITK image in the native DWI
    voxel order, see ``_sitk_native``). ``chain`` lists the registration results (fixed -> moving transforms) from
    the label space towards the DWI space, e.g. [T1->DWI] for FastSurfer labels or [MNI->T1, T1->DWI] for the atlas.
    Resampling needs the DWI -> label-space mapping, i.e. the inverses applied in reverse order (ITK composites
    apply the last-added transform first). Returns a (z, y, x) array of the DWI grid."""
    import SimpleITK as sitk

    lab = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(label_img.dataobj).astype(np.float32), label_img.affine))
    comp = sitk.CompositeTransform(3)
    for t in chain:
        comp.AddTransform(t.GetInverse())
    return sitk.GetArrayFromImage(sitk.Resample(lab, target, comp, sitk.sitkNearestNeighbor, 0.0)).astype(np.int32)


def pauli_atlas():
    from nilearn import datasets

    a = datasets.fetch_atlas_pauli_2017(atlas_type="deterministic")
    return nib.load(a["maps"]) if isinstance(a["maps"], str) else a["maps"]


# ------------------------------------------------------------------------------------------ features
def _roi_masks(fs_lab, pauli_lab, y_index):
    """ROI name -> boolean mask on the DWI grid (arrays in sitk (z, y, x) order). ``y_index`` gives each voxel's
    anterior-posterior coordinate for the anterior/posterior split of the substantia nigra."""
    rois = {}
    for name, (l, r) in FS_ROIS.items():
        rois[f"{name}_l"], rois[f"{name}_r"] = fs_lab == l, fs_lab == r
    for name, ids in FS_SINGLE.items():
        rois[name] = np.isin(fs_lab, ids)
    code = {n: i + 1 for i, n in enumerate(PAULI)}
    # left/right of the (bilateral) atlas labels from FastSurfer's explicit left/right putamen: midline = midpoint of
    # the two centroids along the x index, left = the side of label 12 (orientation-independent)
    xl = np.nonzero(fs_lab == 12)[2].mean() if (fs_lab == 12).any() else pauli_lab.shape[2] * 0.75
    xr = np.nonzero(fs_lab == 51)[2].mean() if (fs_lab == 51).any() else pauli_lab.shape[2] * 0.25
    mid = (xl + xr) / 2
    xi = np.arange(pauli_lab.shape[2])[None, None, :]
    left = (xi > mid) if xl > xr else (xi < mid)
    for name, parts in PAULI_ROIS.items():
        m = np.isin(pauli_lab, [code[p] for p in parts])
        rois[f"{name}_l"], rois[f"{name}_r"] = m & left, m & ~left
    for s in ("l", "r"):
        m = rois.get(f"sn_{s}")
        if m is not None and m.any():
            y = y_index[m]
            cut = np.median(y)
            ant, post = np.zeros_like(m), np.zeros_like(m)
            ant[m] = y > cut     # LPS: larger y = more posterior
            post[m] = y <= cut
            rois[f"sn_posterior_{s}"], rois[f"sn_anterior_{s}"] = ant, post
    return rois


def features(maps, rois, min_voxels=3):
    out = {}
    for name, m in rois.items():
        n = int(m.sum())
        out[f"n_{name}"] = n
        for k in METRICS:
            vals = maps[k][m]
            vals = vals[np.isfinite(vals)]
            out[f"{name}_{k}"] = float(vals.mean()) if len(vals) >= min_voxels else np.nan
    # bilateral means for the headline measures
    for base in ("sn_posterior", "sn", "snc", "snr", "putamen", "caudate", "sn_posterior_t", "sn_t", "snc_t", "snr_t"):
        for k in METRICS:
            l, r = out.get(f"{base}_l_{k}", np.nan), out.get(f"{base}_r_{k}", np.nan)
            out[f"{base}_mean_{k}"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    return out


# ------------------------------------------------------------------------------------------ per-subject driver
def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, fsl=False):
    """All steps for one subject. Returns a flat dict (features + QC)."""
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    work = Path(work_dir) / str(patno)
    work.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in series_rows:
        runs += convert(r["zip"], r["prefix"], work / "nii")
    ds = assemble(runs)
    row = {"patno": patno, "n_series": len(series_rows), "n_runs_used": ds["n_runs"], "n_volumes": int(ds["data"].shape[3]),
           "shells": " ".join(map(str, ds["shells"])), "voxel_mm": float(np.round(np.abs(np.diag(ds["affine"])[:3]).mean(), 2)),
           "manufacturer": str(ds["meta"].get("Manufacturer", "")), "model": str(ds["meta"].get("ManufacturerModelName", "")),
           "pe_direction": str(ds["meta"].get("PhaseEncodingDirection", "")), "readout_s": ds["meta"].get("TotalReadoutTime", np.nan),
           "series_desc": ";".join(sorted(set(r["desc"] for r in series_rows)))}
    if fsl:
        ds = susceptibility_correct(ds, work)
    row["topup"] = bool(ds.get("topup", False))
    row["n_rev_b0"] = int(ds["rev_b0"].shape[3]) if ds.get("rev_b0") is not None else 0
    ds = preprocess(ds)
    row.update({"motion_mm_mean": ds["motion_mm_mean"], "motion_mm_max": ds["motion_mm_max"], "rotation_deg_max": ds["rotation_deg_max"]})
    b0_img = nib.Nifti1Image(ds["b0"], ds["affine"])
    mri = Path(fastsurfer_dir) / "mri"
    t1, t1_mask, aseg = nib.load(mri / "orig.mgz"), nib.load(mri / "mask.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    tx_t1_dwi, m_rigid = register_b0_to_t1(b0_img, t1, t1_mask)
    tx_mni_t1, m_mni = register_t1_to_mni(t1, t1_mask, cache_path=mni_cache_path(fastsurfer_dir))
    row.update({"reg_b0_t1_mi": m_rigid, "reg_t1_mni_mi": m_mni})
    tgt = _sitk_native(ds["b0"], ds["affine"])       # native DWI grid: label arrays and maps share (z, y, x) order
    fs_lab = labels_to_dwi(aseg, tgt, [tx_t1_dwi])
    pauli_lab = labels_to_dwi(pauli_atlas(), tgt, [tx_mni_t1, tx_t1_dwi])
    # anterior-posterior (LPS y) coordinate of every DWI voxel for the SN split
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in sitk.GetArrayFromImage(tgt).shape], indexing="ij")
    origin, spacing, direction = np.array(tgt.GetOrigin()), np.array(tgt.GetSpacing()), np.array(tgt.GetDirection()).reshape(3, 3)
    phys_y = origin[1] + direction[1, 0] * xx * spacing[0] + direction[1, 1] * yy * spacing[1] + direction[1, 2] * zz * spacing[2]
    rois = _roi_masks(fs_lab, pauli_lab, phys_y)
    small = [m for k, m in rois.items() if not k.startswith(("cerebral_wm", "cerebellum_wm", "brainstem"))]
    fw_mask_xyz = np.transpose(ndimage.binary_dilation(np.any(small, axis=0), iterations=2), (2, 1, 0))
    # maps are computed in (x, y, z); ROI masks are in sitk (z, y, x): transpose the maps once
    maps = fit_models(ds, fw_mask=fw_mask_xyz)
    row["fw_method"] = maps.pop("fw_method")
    maps = {k: np.transpose(v, (2, 1, 0)) for k, v in maps.items()}
    # tissue-restricted nigral variants (suffix _t): the affine-mapped atlas SN at 2 mm takes in cerebral-peduncle
    # fibres (FA ~0.45) and interpeduncular CSF; keep voxels with FA < 0.5 and free water < 0.7
    tissue = (maps["fa"] < 0.5) & ~(np.nan_to_num(maps["fw"], nan=1.0) >= 0.7)
    for name in [k for k in rois if k.startswith(("sn", "snc", "snr", "stn", "vta", "red_nucleus"))]:
        base, side = name.rsplit("_", 1)          # "sn_posterior_l" -> "sn_posterior_t_l"
        rois[f"{base}_t_{side}"] = rois[name] & tissue
    row.update(features(maps, rois))
    row["fw_brain_median"] = float(np.median(maps["fw"][np.transpose(ds["mask"], (2, 1, 0)) & (maps["fw"] > 0)])) if (maps["fw"] > 0).any() else np.nan
    row["fa_wm_median"] = float(np.median(maps["fa"][rois["cerebral_wm_l"] | rois["cerebral_wm_r"]])) if (rois["cerebral_wm_l"] | rois["cerebral_wm_r"]).any() else np.nan
    if keep_nifti:
        for k, v in maps.items():
            nib.save(nib.Nifti1Image(np.transpose(v, (2, 1, 0)), ds["affine"]), work / f"{k}.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(pauli_lab, (2, 1, 0)).astype(np.int16), ds["affine"]), work / "pauli_dwi.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(fs_lab, (2, 1, 0)).astype(np.int16), ds["affine"]), work / "aseg_dwi.nii.gz")
        nib.save(b0_img, work / "b0.nii.gz")
    else:
        shutil.rmtree(work / "nii", ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep, fsl = args
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep, fsl=fsl)
        out["error"] = ""
    except Exception as e:  # keep the batch going
        out = {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def _flag_dwi(idx):
    return ~(idx["desc"].str.contains("ADC", case=False) | idx["desc"].str.startswith(("Reg_", "dReg", "eReg")))


def main(argv=None):
    import argparse

    from .batch import add_common_args, done_subjects, fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows

    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--fsl", action="store_true", help="FSL topup/applytopup susceptibility correction where a reverse-phase b0 exists")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    idx = load_index(work / "dwi_index.csv", a.zips, _flag_dwi)
    idx = idx[idx["selected"]]
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    out_csv = work / "dwi_features.csv"
    done = done_subjects(out_csv, a.retry_errors)
    jobs = [(int(patno), session_rows(g), fs[int(patno)], str(work), a.keep_nifti, a.fsl)
            for patno, g in idx.groupby("patno") if patno not in done and int(patno) in fs]
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, pid_file=a.pid_file)


if __name__ == "__main__":
    main()
