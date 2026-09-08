"""
FLAIR -> white-matter hyperintensity (WMH) burden, as a vascular covariate.

No licensed lesion segmenter is available (SAMSEG/LST need a FreeSurfer or MATLAB licence, BIANCA needs
hand-labelled training data), so this is the classic intensity-threshold approach: N4 bias correction, rigid
registration to the conformed T1, lesions = white-matter voxels (FastSurfer cerebral WM + WM-hypointensity labels,
eroded away from cortex) brighter than the median of normal-appearing WM + K_MAD robust standard deviations,
small components removed, split into periventricular (<= 10 mm from the lateral ventricles) and deep. FastSurfer's
T1-based WM-hypointensity volume (already in the IDP table) is the sanity reference. Both 3D (1 mm) and 2D (5 mm)
FLAIR series occur in PPMI; 3D is preferred when both exist and the type is recorded for harmonisation.

    venv_imaging/bin/python -m pie.imaging.flair --zips <full-MRI zips> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/flair --workers 4 [--keep-nifti]
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .dwi import _brain, _sitk_from_nib, _sitk_native
from .batch import convert_series as convert

FLAIR_PATTERN = r"FLAIR|dark.?fluid|tirm"
EXCLUDE = r"T1 FLAIR|Sag T1 FLAIR|Ax T1 FLAIR|REPEAT|rpt|\*\*"
K_MAD = 3.0
PV_MM = 10.0
MIN_LESION_MM3 = 5.0
WM_LABELS = (2, 41, 77)          # cerebral WM left/right, WM hypointensities
VENTRICLES = (4, 43, 5, 44)      # lateral + inferior lateral ventricles


def flag_flair(idx):
    d = idx["desc"].str.replace("_", " ")
    return d.str.contains(FLAIR_PATTERN, case=False, regex=True) & ~d.str.contains(EXCLUDE, case=False, regex=True)


def flag_flair_3d(idx):
    d = idx["desc"].str.replace("_", " ")
    return flag_flair(idx) & (d.str.contains("3D", case=False) | (idx["n_files"] >= 100))


def index_flair(zips):
    from .batch import index_series

    idx = index_series(zips)
    idx["flair"] = flag_flair(idx)
    idx["flair_3d"] = flag_flair_3d(idx)
    return idx


def n4(img_sitk, shrink=2):
    import SimpleITK as sitk

    mask = sitk.OtsuThreshold(img_sitk, 0, 1, 200)
    small = sitk.Shrink(img_sitk, [shrink] * 3)
    small_mask = sitk.Shrink(mask, [shrink] * 3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50, 50, 30])
    corr.Execute(small, small_mask)
    field = corr.GetLogBiasFieldAsImage(img_sitk)
    return sitk.Cast(img_sitk / sitk.Exp(field), sitk.sitkFloat32)   # keep float32: the registration needs matching pixel types


def register_flair_to_t1(flair_sitk, t1_img, t1_mask_img):
    """Rigid MI registration (T1 brain fixed at 2 mm, FLAIR moving). Returns (transform fixed(T1)->moving(FLAIR), metric)."""
    import SimpleITK as sitk

    fixed = _brain(t1_img, t1_mask_img, mm=2.0)
    init = sitk.CenteredTransformInitializer(fixed, flair_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
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
    tx = reg.Execute(fixed, flair_sitk)
    return tx, float(reg.GetMetricValue())


def wmh(flair_t1, aseg, vox_mm=1.0):
    """Lesion mask and features from the FLAIR resampled onto the T1 grid (array in (z, y, x)) and the aseg labels
    (same order)."""
    wm = np.isin(aseg, WM_LABELS)
    wm = ndimage.binary_erosion(wm, iterations=1)                     # keep clear of cortical partial volume
    normal = wm & (aseg != 77) & (flair_t1 > 0)
    med = float(np.median(flair_t1[normal]))
    mad = float(1.4826 * np.median(np.abs(flair_t1[normal] - med)))
    thr = med + K_MAD * mad
    les = wm & (flair_t1 > thr)
    lab, n = ndimage.label(les)
    if n:
        sizes = ndimage.sum(les, lab, range(1, n + 1)) * vox_mm**3
        les = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_LESION_MM3))
    vent = np.isin(aseg, VENTRICLES)
    dist = ndimage.distance_transform_edt(~vent, sampling=vox_mm) if vent.any() else np.full(les.shape, np.inf)
    pv = les & (dist <= PV_MM)
    n_les = int(ndimage.label(les)[1]) if les.any() else 0
    v = float(vox_mm**3)
    out = {"wmh_mm3": float(les.sum() * v), "wmh_pv_mm3": float(pv.sum() * v), "wmh_deep_mm3": float((les & ~pv).sum() * v),
           "wmh_n_lesions": n_les, "wmh_frac_wm": float(les.sum() / max(wm.sum(), 1)), "wm_mm3": float(wm.sum() * v),
           "flair_wm_median": med, "flair_wm_mad": mad, "wmh_threshold": thr}
    out["wmh_log_mm3"] = float(np.log1p(out["wmh_mm3"]))
    return les, out


def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False):
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    work = Path(work_dir) / str(patno)
    work.mkdir(parents=True, exist_ok=True)
    rows = sorted(series_rows, key=lambda r: (not r["flair_3d"], -r["n_files"]))   # 3D first, then the largest
    niis = []
    for r in rows:
        niis = [n for n in convert(r["zip"], r["prefix"], work / "nii") if nib.load(n).ndim == 3]
        if niis:
            chosen = r
            break
    if not niis:
        raise ValueError("no usable FLAIR volume")
    img = nib.load(niis[0])
    meta = json.load(open(niis[0][:-7] + ".json")) if Path(niis[0][:-7] + ".json").exists() else {}
    row = {"patno": patno, "n_series": len(series_rows), "series_desc": chosen["desc"], "flair_3d": bool(chosen["flair_3d"]),
           "shape": "x".join(map(str, img.shape)), "voxel_mm": "x".join(str(round(float(z), 2)) for z in img.header.get_zooms()[:3]),
           "slice_mm": float(max(img.header.get_zooms()[:3])), "manufacturer": str(meta.get("Manufacturer", "")),
           "model": str(meta.get("ManufacturerModelName", "")), "tr_s": meta.get("RepetitionTime", np.nan), "te_s": meta.get("EchoTime", np.nan), "ti_s": meta.get("InversionTime", np.nan)}
    mri = Path(fastsurfer_dir) / "mri"
    t1, t1_mask, aseg_img = nib.load(mri / "orig.mgz"), nib.load(mri / "mask.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    fl = n4(_sitk_native(np.asanyarray(img.dataobj).astype(np.float32), img.affine))
    tx, metric = register_flair_to_t1(fl, t1, t1_mask)
    row["reg_flair_t1_mi"] = metric
    t1_grid = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(t1.dataobj).astype(np.float32), t1.affine))
    fl_t1 = sitk.GetArrayFromImage(sitk.Resample(fl, t1_grid, tx, sitk.sitkLinear, 0.0))                 # (z, y, x) on the canonical T1 grid
    aseg = sitk.GetArrayFromImage(sitk.Resample(_sitk_from_nib(nib.Nifti1Image(np.asanyarray(aseg_img.dataobj).astype(np.float32), aseg_img.affine)),
                                                t1_grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)).astype(np.int32)
    les, feats = wmh(fl_t1, aseg, vox_mm=float(t1_grid.GetSpacing()[0]))
    row.update(feats)
    if keep_nifti:
        can = nib.as_closest_canonical(t1)
        nib.save(nib.Nifti1Image(np.transpose(fl_t1, (2, 1, 0)), can.affine), work / "flair_t1.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(les, (2, 1, 0)).astype(np.int16), can.affine), work / "wmh_t1.nii.gz")
    shutil.rmtree(work / "nii", ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep = args
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep)
        out["error"] = ""
    except Exception as e:
        out = {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv=None):
    import argparse

    from .batch import add_common_args, done_subjects, fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows

    a = add_common_args(argparse.ArgumentParser()).parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    idx = load_index(work / "flair_index.csv", a.zips, flag_flair)
    idx["flair_3d"] = flag_flair_3d(idx)
    idx = idx[idx["selected"]]
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    out_csv = work / "flair_features.csv"
    done = done_subjects(out_csv, a.retry_errors)
    jobs = [(int(patno), session_rows(g), fs[int(patno)], str(work), a.keep_nifti)
            for patno, g in idx.groupby("patno") if patno not in done and int(patno) in fs]
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, log_every=20, pid_file=a.pid_file)


if __name__ == "__main__":
    main()
