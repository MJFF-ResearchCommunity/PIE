"""
FLAIR -> white-matter hyperintensity (WMH) burden, as a vascular covariate.

Default ``--method truenet``: TrUE-Net (Sundaresan et al. 2021, MedIA, doi:10.1016/j.media.2021.102184), FSL's
triplanar U-Net trained on vascular WMH, run as distributed. ``prepare_truenet_data`` takes the FLAIR and the
FastSurfer conformed T1 (reorientation, BET, FAST bias correction, fsl_anat, rigid T1->FLAIR FLIRT, the BIANCA
white-matter mask); ``truenet evaluate -m mwsc`` (the MICCAI WMH Challenge FLAIR+T1 model) gives a probability map,
which is masked by that WM mask and thresholded at 0.5 (the paper's post-processing). Volumes are measured on the
FLAIR grid. Weights: ``third_party/weights/truenet`` (FSL licence, non-commercial use only).

``--method threshold``: the earlier intensity threshold. Lesions are white-matter voxels (FastSurfer cerebral WM +
WM-hypointensity labels, eroded away from cortex) brighter than the median of normal-appearing WM + K_MAD robust
standard deviations, after N4 and small components removed. Its volume tracked age at only rho 0.18 in PPMI.

Both methods use the same rigid FLAIR->T1 registration (QC, and the periventricular split: <= 10 mm from the
FastSurfer lateral ventricles, else deep). FastSurfer's T1-based WM-hypointensity volume (already in the IDP table)
is the sanity reference. Both 3D (1 mm) and 2D (5 mm) FLAIR series occur in PPMI; 3D is preferred when both exist,
and the type (``flair_3d``) and S-I coverage (``flair_si_mm``) are recorded.

    venv_imaging/bin/python -m pie.imaging.flair --zips <full-MRI zips> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/flair_truenet --workers 4 \
        [--method threshold] [--keep-nifti]
"""

import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .datscan import ITK_SEED
from .dwi import FSLDIR, _brain, _sitk_from_nib, _sitk_native
from .batch import convert_series as convert

FLAIR_PATTERN = r"FLAIR|dark.?fluid|tirm"
EXCLUDE = r"T1 FLAIR|Sag T1 FLAIR|Ax T1 FLAIR|REPEAT|rpt|\*\*"
K_MAD = 3.0
PV_MM = 10.0
MIN_LESION_MM3 = 5.0
WM_LABELS = (2, 41, 77)          # cerebral WM left/right, WM hypointensities
VENTRICLES = (4, 43, 5, 44)      # lateral + inferior lateral ventricles
METHODS = ("truenet", "threshold")
# MICCAI WMH Challenge FLAIR+T1 model: three sites and scanners, 3 mm 2D and 3D FLAIR, closer to PPMI's mix than
# the single-protocol UK Biobank model (ukbb, also installed)
TRUENET_MODEL = "mwsc"
WMH_PROB = 0.5                   # Sundaresan 2021 section 2.1.4: WM-masked probability thresholded at 0.5
TRUENET_MODELS = Path(os.environ.get("TRUENET_PRETRAINED_MODEL_PATH")
                      or Path(os.environ.get("PIE_WEIGHTS_DIR") or Path(__file__).resolve().parents[2] / "third_party" / "weights") / "truenet")
BIN = Path(sys.executable).parent   # prepare_truenet_data and truenet are console scripts of this environment


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
    """Rigid MI registration of the FLAIR (moving) to the conformed T1 head (fixed, 2 mm). Two starts, the scanner header
    (same-session scans share scanner coordinates) and the centre-of-mass alignment; the start whose final pose has the
    better head metric wins. The head, not the brain, is the fixed image: a partial-coverage 2D FLAIR (5 mm slices over
    the upper head) registered to a brain-only T1 from a centre-of-mass start settled centimetres off, which was most of
    the 13 % of PPMI FLAIRs that failed QC until September 2026. Returns (transform fixed(T1)->moving(FLAIR), metric), the
    metric being the brain-masked MI at the final pose, on the scale the QC threshold was set on."""
    import SimpleITK as sitk

    head = sitk.Shrink(_sitk_from_nib(nib.Nifti1Image(np.asanyarray(t1_img.dataobj).astype(np.float32), t1_img.affine)),
                       [max(1, int(round(2.0 / sp))) for sp in np.abs(np.diag(nib.as_closest_canonical(t1_img).affine)[:3])])
    brain = _brain(t1_img, t1_mask_img, mm=2.0)

    def method(sampling=0.2):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, seed=ITK_SEED)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg

    header = sitk.Euler3DTransform()
    header.SetCenter(head.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in head.GetSize()]))
    moments = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(head, flair_sitk, sitk.Euler3DTransform(),
                                                                     sitk.CenteredTransformInitializerFilter.MOMENTS))
    best = None
    for init in (header, moments):
        reg = method()
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([3, 2, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init, inPlace=False)
        try:
            tx = reg.Execute(head, flair_sitk)
        except RuntimeError:
            continue
        score = method(1.0)
        score.SetInitialTransform(tx)
        value = score.MetricEvaluate(head, flair_sitk)
        if best is None or value < best[1]:
            best = (tx, value)
    if best is None:
        raise RuntimeError("FLAIR-to-T1 registration failed from both starts")
    qc = method(1.0)
    qc.SetInitialTransform(best[0])
    return best[0], float(qc.MetricEvaluate(brain, flair_sitk))


def si_coverage_mm(img):
    """Extent of the volume along its voxel axis closest to superior-inferior: slices x spacing for an axial 2D slab."""
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    ax = int(np.argmax(np.abs(img.affine[2, :3]) / zooms))
    return float(img.shape[ax] * zooms[ax])


def normal_wm(flair_t1, aseg):
    """Eroded FastSurfer WM (kept clear of cortical partial volume) and the FLAIR median and robust SD in its
    normal-appearing part (arrays on the T1 grid)."""
    wm = ndimage.binary_erosion(np.isin(aseg, WM_LABELS), iterations=1)
    v = flair_t1[wm & (aseg != 77) & (flair_t1 > 0)]
    med = float(np.median(v))
    return wm, med, float(1.4826 * np.median(np.abs(v - med)))


def ventricle_distance(aseg, vox_mm=1.0):
    vent = np.isin(aseg, VENTRICLES)
    return ndimage.distance_transform_edt(~vent, sampling=vox_mm) if vent.any() else np.full(aseg.shape, np.inf)


def burden(les, dist, vox_mm3):
    """Total, periventricular (``dist`` <= PV_MM) and deep volume (mm^3) and lesion count of a lesion mask."""
    pv = les & (dist <= PV_MM)
    out = {"wmh_mm3": float(les.sum() * vox_mm3), "wmh_pv_mm3": float(pv.sum() * vox_mm3),
           "wmh_deep_mm3": float((les & ~pv).sum() * vox_mm3), "wmh_n_lesions": int(ndimage.label(les)[1]) if les.any() else 0}
    out["wmh_log_mm3"] = float(np.log1p(out["wmh_mm3"]))
    return out


def wmh(flair_t1, aseg, vox_mm=1.0):
    """Threshold method: lesion mask and features from the FLAIR resampled onto the T1 grid (array in (z, y, x)) and
    the aseg labels (same order)."""
    wm, med, mad = normal_wm(flair_t1, aseg)
    thr = med + K_MAD * mad
    les = wm & (flair_t1 > thr)
    lab, n = ndimage.label(les)
    if n:
        sizes = ndimage.sum(les, lab, range(1, n + 1)) * vox_mm**3
        les = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_LESION_MM3))
    out = burden(les, ventricle_distance(aseg, vox_mm), float(vox_mm**3))
    out.update(wmh_frac_wm=float(les.sum() / max(wm.sum(), 1)), wm_mm3=float(wm.sum() * vox_mm**3),
               flair_wm_median=med, flair_wm_mad=mad, wmh_threshold=thr)
    return les, out


def truenet_env():
    """Environment for the TrUE-Net subprocesses; raises naming what is missing (FSLDIR is often not exported)."""
    if not FSLDIR or not Path(FSLDIR, "bin", "fsl_anat").exists():
        raise RuntimeError(f"TrUE-Net preprocessing needs FSL (fsl_anat): set FSLDIR (got {FSLDIR!r})")
    if not (BIN / "prepare_truenet_data").exists():
        raise RuntimeError(f"truenet is not installed next to {sys.executable}: run with venv_imaging/bin/python")
    if not (TRUENET_MODELS / TRUENET_MODEL).is_dir():
        raise RuntimeError(f"TrUE-Net model {TRUENET_MODEL!r} not under {TRUENET_MODELS}: see third_party/weights/truenet/README.md "
                           "or set TRUENET_PRETRAINED_MODEL_PATH")
    return dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ", TRUENET_PRETRAINED_MODEL_PATH=str(TRUENET_MODELS),
                PATH=os.pathsep.join([f"{FSLDIR}/bin", str(BIN), os.environ.get("PATH", "")]), OMP_NUM_THREADS="2")


def _prepare_from_fastsurfer(out, fastsurfer_dir, env, log):
    """TrUE-Net inputs without ``fsl_anat`` (its nonlinear registration takes ~20 min and serves only the post-processing
    WM mask): the FLAIR exactly as ``prepare_truenet_data`` prepares it (reorient, BET, FAST bias correction); the T1 =
    FastSurfer's N4-corrected ``orig_nu`` in its brain mask, rigidly registered to the FLAIR brain (``flirt -dof 6``, as
    the script does); the WM mask = FastSurfer's cerebral white matter + WM hypointensities (aseg 2, 41, 77) carried
    by the same matrix. Writes sub_FLAIR, sub_T1 and sub_WMmask."""
    from .freesurfer import run

    fsl = Path(FSLDIR) / "bin"
    mri = Path(fastsurfer_dir) / "mri"
    t1n, mask = nib.load(mri / "orig_nu.mgz"), np.asanyarray(nib.load(mri / "mask.mgz").dataobj) > 0
    aseg = np.asanyarray(nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz").dataobj)
    nib.save(nib.Nifti1Image((np.asanyarray(t1n.dataobj) * mask).astype(np.float32), t1n.affine), out / "t1_brain_raw.nii.gz")
    nib.save(nib.Nifti1Image(np.isin(aseg, [2, 41, 77]).astype(np.uint8), t1n.affine), out / "wm_raw.nii.gz")
    for cmd in ([fsl / "fslreorient2std", out / "flair", out / "FLAIR"], [fsl / "bet", out / "FLAIR", out / "FLAIR_brain"],
                [fsl / "fast", "-B", "--nopve", out / "FLAIR_brain"], [fsl / "imcp", out / "FLAIR_brain_restore", out / "sub_FLAIR"],
                [fsl / "fslreorient2std", out / "t1_brain_raw", out / "T1_brain"], [fsl / "fslreorient2std", out / "wm_raw", out / "WM"],
                [fsl / "flirt", "-dof", "6", "-in", out / "T1_brain", "-ref", out / "FLAIR_brain", "-out", out / "sub_T1", "-omat", out / "t1_2flair.mat"],
                [fsl / "flirt", "-in", out / "WM", "-ref", out / "FLAIR_brain", "-applyxfm", "-init", out / "t1_2flair.mat", "-interp",
                 "nearestneighbour", "-out", out / "sub_WMmask"]):
        run(cmd, log, out, env=env, timeout=3600)


def run_truenet(flair_nii, t1_img, out_dir, prep="fsl_anat", fastsurfer_dir=None):
    """TrUE-Net as distributed on one FLAIR (NIfTI path) + T1 (nibabel image), in a fresh ``out_dir``:

        prepare_truenet_data --FLAIR=<out>/flair --T1=<out>/t1 --outname=<out>/sub --nodistmaps
        truenet evaluate -i <out> -m mwsc -o <out>/pred

    (the script appends .nii.gz; distance maps only matter for training). Returns the probability map masked by the
    BIANCA WM mask, as a nibabel image on the prepared (reoriented, brain-extracted) FLAIR grid.
    ``CUDA_VISIBLE_DEVICES=`` forces the CPU. ``prep="fastsurfer"`` replaces the script by ``_prepare_from_fastsurfer``
    (~2 min instead of ~25; agreement with the distributed preparation in the docs)."""
    from .freesurfer import run

    out = Path(out_dir).resolve()
    if " " in str(out):
        raise ValueError(f"prepare_truenet_data breaks on paths with spaces: {out}")
    env = truenet_env()
    shutil.rmtree(out, ignore_errors=True)           # a failed attempt's outputs must not pass for this one's
    (out / "pred").mkdir(parents=True)
    shutil.copy(flair_nii, out / "flair.nii.gz")
    nib.save(nib.Nifti1Image(np.asanyarray(t1_img.dataobj), t1_img.affine), out / "t1.nii.gz")
    log = out / "truenet.log"
    if prep == "fastsurfer":
        _prepare_from_fastsurfer(out, fastsurfer_dir, env, log)
    else:
        run([BIN / "prepare_truenet_data", f"--FLAIR={out / 'flair'}", f"--T1={out / 't1'}", f"--outname={out / 'sub'}", "--nodistmaps"],
            log, out, env=env, timeout=6 * 3600)
    missing = [f for f in ("sub_FLAIR", "sub_T1", "sub_WMmask") if not (out / f"{f}.nii.gz").exists()]
    if missing:                      # the script exits 0 whatever fails
        raise RuntimeError(f"prepare_truenet_data wrote no {', '.join(missing)}; see {log}")
    run([BIN / "truenet", "evaluate", "-i", out, "-m", TRUENET_MODEL, "-o", out / "pred"], log, out, env=env)
    prob = nib.load(out / "pred" / "Predicted_probmap_truenet_sub.nii.gz")
    wm = np.asanyarray(nib.load(out / "sub_WMmask.nii.gz").dataobj) > 0
    return nib.Nifti1Image(np.where(wm, np.asanyarray(prob.dataobj), 0).astype(np.float32), prob.affine)


def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, method="truenet", prep="fsl_anat"):
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
           "slice_mm": float(max(img.header.get_zooms()[:3])), "manufacturer": str(meta.get("Manufacturer", "")), **{k: str(meta.get(k, "")) for k in ("DeviceSerialNumber", "StationName", "SoftwareVersions")},
           "model": str(meta.get("ManufacturersModelName", meta.get("ManufacturerModelName", ""))), "tr_s": meta.get("RepetitionTime", np.nan), "te_s": meta.get("EchoTime", np.nan), "ti_s": meta.get("InversionTime", np.nan),
           "acquisition_date": chosen["date"], "fs_image_id": Path(fastsurfer_dir).name,
           "flair_si_mm": si_coverage_mm(img), "wmh_method": method}
    mri = Path(fastsurfer_dir) / "mri"
    t1, t1_mask, aseg_img = nib.load(mri / "orig.mgz"), nib.load(mri / "mask.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    fl = n4(_sitk_native(np.asanyarray(img.dataobj).astype(np.float32), img.affine))
    tx, metric = register_flair_to_t1(fl, t1, t1_mask)
    row["reg_flair_t1_mi"] = metric
    t1_grid = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(t1.dataobj).astype(np.float32), t1.affine))
    fl_t1 = sitk.GetArrayFromImage(sitk.Resample(fl, t1_grid, tx, sitk.sitkLinear, 0.0))                 # (z, y, x) on the canonical T1 grid
    aseg = sitk.GetArrayFromImage(sitk.Resample(_sitk_from_nib(nib.Nifti1Image(np.asanyarray(aseg_img.dataobj).astype(np.float32), aseg_img.affine)),
                                                t1_grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)).astype(np.int32)
    vox = float(t1_grid.GetSpacing()[0])
    if method == "threshold":
        les, feats = wmh(fl_t1, aseg, vox_mm=vox)
    else:
        from importlib.metadata import version

        prob = run_truenet(niis[0], t1, work / "truenet", prep=prep, fastsurfer_dir=fastsurfer_dir)
        p = np.asanyarray(prob.dataobj)
        fl_grid = _sitk_native(p, prob.affine)
        dist_t1 = sitk.GetImageFromArray(ventricle_distance(aseg, vox).astype(np.float32))
        dist_t1.CopyInformation(t1_grid)
        # T1-grid ventricle distance carried onto the FLAIR grid, so lesion volumes stay TrUE-Net's own voxels
        dist = np.transpose(sitk.GetArrayFromImage(sitk.Resample(dist_t1, fl_grid, tx.GetInverse(), sitk.sitkLinear, 1e6)), (2, 1, 0))
        feats = burden(p >= WMH_PROB, dist, float(np.prod(prob.header.get_zooms()[:3])))
        wm, med, mad = normal_wm(fl_t1, aseg)
        fsl = Path(FSLDIR, "etc", "fslversion")
        feats.update(wm_mm3=float(wm.sum() * vox**3), flair_wm_median=med, flair_wm_mad=mad, wmh_prob_threshold=WMH_PROB,
                     truenet_model=TRUENET_MODEL, truenet_version=version("truenet"), truenet_prep=prep,
                     fsl_version=fsl.read_text().strip() if fsl.exists() else "")
        feats["wmh_frac_wm"] = feats["wmh_mm3"] / max(feats["wm_mm3"], 1.0)
        les = sitk.GetArrayFromImage(sitk.Resample(_sitk_native((p >= WMH_PROB).astype(np.float32), prob.affine), t1_grid, tx,
                                                   sitk.sitkNearestNeighbor, 0.0)) > 0          # for the QC gallery only
        if not keep_nifti:
            shutil.rmtree(work / "truenet", ignore_errors=True)
    row.update(feats)
    if keep_nifti:
        can = nib.as_closest_canonical(t1)
        nib.save(nib.Nifti1Image(np.transpose(fl_t1, (2, 1, 0)), can.affine), work / "flair_t1.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(les, (2, 1, 0)).astype(np.int16), can.affine), work / "wmh_t1.nii.gz")
    shutil.rmtree(work / "nii", ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep, method, prep = args
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep, method=method, prep=prep)
        out["error"] = ""
    except Exception as e:
        out = {"patno": patno, "wmh_method": method, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv=None):
    import argparse

    from .batch import add_common_args, filter_jobs, load_index, modality_jobs, run_batch

    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--truenet-prep", choices=["fsl_anat", "fastsurfer"], default="fsl_anat", help="TrUE-Net input preparation: "
                    "the distributed prepare_truenet_data (fsl_anat, ~25 min) or FastSurfer's N4 T1, mask and WM labels (~2 min)")
    ap.add_argument("--method", choices=METHODS, default="truenet",
                    help="truenet (needs FSL and the TrUE-Net weights) or the earlier intensity threshold")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_csv = work / "flair_features.csv"
    if out_csv.exists() and out_csv.stat().st_size:     # resuming must not mix methods (rows from before wmh_method: threshold)
        prev = pd.read_csv(out_csv)
        methods = set(prev["wmh_method"].fillna("threshold")) if "wmh_method" in prev else {"threshold"}
        if methods != {a.method}:
            raise SystemExit(f"{out_csv} holds {', '.join(sorted(methods))} rows: give --method {a.method} its own --work-dir")
        # nor TrUE-Net preparations: on 10 PPMI FLAIRs the FastSurfer one gave 11 % smaller volumes (rows before truenet_prep: fsl_anat)
        preps = set(prev["truenet_prep"].dropna()) if "truenet_prep" in prev else {"fsl_anat"}
        if a.method == "truenet" and preps - {a.truenet_prep}:
            raise SystemExit(f"{out_csv} holds TrUE-Net rows prepared with {', '.join(sorted(preps))}: give --truenet-prep {a.truenet_prep} its own --work-dir")
    if a.method == "truenet":
        truenet_env()                                     # fail now, not once per subject
    idx = load_index(work / "flair_index.csv", a.zips, flag_flair)
    idx["flair_3d"] = flag_flair_3d(idx)
    idx = idx[idx["selected"]]
    jobs = [(p, rows, fsd, str(work), a.keep_nifti, a.method, a.truenet_prep) for p, rows, fsd in modality_jobs(idx, a, out_csv)]
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, log_every=20, pid_file=a.pid_file)


if __name__ == "__main__":
    main()
