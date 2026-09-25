"""
Diffusion MRI (PPMI DTI download) -> per-subject free-water / tensor features in subcortical and nigral ROIs.

PPMI's diffusion data come in two generations: PPMI-1 single-shell (b = 1000, 32-64 directions; Siemens mosaics,
GE and Philips multi-file series, Philips as opposite-phase-encoding LR/RL pairs) and PPMI-2 three-shell Siemens
Prisma (b = 700/1000/2000, 64 directions each, plus reverse-phase b0s). Pipeline per subject:

1. `convert`      dcm2niix on every diffusion series (bval/bvec/json), derived series (ADC, "Reg -") dropped.
2. `assemble`     one DWI dataset: same-geometry, same-phase-encoding runs concatenated (the PPMI-2 shells);
                  opposite-phase runs (Philips LR/RL) are not merged, the run with more directions is used.
3. `preprocess`   optional denoising and topup, brain mask, rigid volume-to-b0 correction with gradient rotation.
                  This is not eddy-current or slice-outlier correction; eddy remains a separate upgrade.
4. `fit`          DTI (weighted least squares, b <= 1000) for FA/MD over the brain; free-water bi-tensor model for
                  FW and tissue FA inside the ROI neighbourhood: DIPY's multi-shell NLS (Hoy et al. 2014) for the
                  PPMI-2 shells, a bounded voxel-wise fit with a tissue-diffusivity prior for single-shell PPMI-1
                  data (`fw_method` records which; single-shell free water is ill-posed, failed its PPMI validity checks and is
                  left out of `manifest.assemble_features` by default).
5. `register`     mean b0 -> conformed T1 (rigid, mutual information; susceptibility distortion is corrected only by
                  the optional topup step, `--fsl`); T1 -> MNI152NLin2009cAsym affine (brain-masked) to bring the
                  CIT168 subcortical atlas (Pauli et al. 2018, the authors' MNI2009c projection bundled in
                  `pie.imaging.atlases`: SNc, SNr, RN, STN, VTA, ...) into subject space alongside the FastSurfer labels.
6. `features`     mean FA, MD, FW, FAt per ROI (left/right; the substantia nigra also split into anterior and
                  posterior halves, the posterior half being the free-water marker of nigral degeneration).

    venv_imaging/bin/python -m pie.imaging.dwi --zips <DTI zips> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/dwi --workers 6   # -> dwi_features.csv
"""

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
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
# CIT168 v1.0 deterministic labels (1-based), in the order of pie/imaging/data/atlases/CIT168_*.json
PAULI = ["Pu", "Ca", "NAC", "EXA", "GPe", "GPi", "SNc", "RN", "SNr", "PBP", "VTA", "VeP", "HN", "HTH", "MN", "STH"]
PAULI_ROIS = {"snc": ("SNc",), "snr": ("SNr",), "sn": ("SNc", "SNr"), "red_nucleus": ("RN",), "stn": ("STH",), "vta": ("VTA",),
              "gpe": ("GPe",), "gpi": ("GPi",), "nac": ("NAC",)}
# fa md ad rd: WLS tensor (every scan). fw fat: free water and tissue FA. mdt (free-water-corrected MD) and mk (DKI
# mean kurtosis): multi-shell scans only, so single-shell rows leave those columns empty
METRICS = ("fa", "md", "fw", "fat", "ad", "rd", "mdt", "mk")


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


def acquisition_metadata_key(meta):
    """Conservative run compatibility, including estimated timing when actual timing is absent.

    An estimate can detect an acquisition mismatch; it cannot supply missing PE
    polarity or qualify a scan for topup/eddy. Receiver bandwidth is retained too.
    """
    actual = meta.get('TotalReadoutTime')
    value = actual if actual is not None else meta.get('EstimatedTotalReadoutTime')
    # dcm2niix's Philips estimates differ by sub-microsecond rounding even within
    # one protocol. Preserve millisecond mismatches without splitting roundoff.
    readout = ('recorded' if actual is not None else 'estimated', round(float(value), 6) if value is not None else None)
    return (meta.get('PhaseEncodingDirection'), meta.get('PhaseEncodingAxis'), readout,
            meta.get('EchoTime'), meta.get('RepetitionTime'), meta.get('PixelBandwidth'))


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
    # Include the full affine, not just voxel sizes: shifted/oblique grids cannot be concatenated voxelwise.
    groups = {}
    for img, b, v, meta in loaded:
        key = (img.shape[:3], tuple(np.round(img.affine, 4).ravel()), acquisition_metadata_key(meta))
        groups.setdefault(key, []).append((img, b, v, meta))
    best = max(groups.values(), key=lambda g: sum((x[1] > 50).sum() for x in g))
    data = np.concatenate([np.asanyarray(x[0].dataobj).astype(np.float32) for x in best], axis=3)
    bvals = np.concatenate([x[1] for x in best])
    bvecs = np.concatenate([x[2] for x in best], axis=1)
    meta = best[0][3]
    pe = meta.get("PhaseEncodingDirection")
    # Full opposite-PE runs (e.g. Philips LR/RL) also contain usable reverse b0s.
    rev = [x for x in b0_runs + loaded if x[0].shape[:3] == best[0][0].shape[:3]
           and np.allclose(x[0].affine, best[0][0].affine, atol=1e-4, rtol=0)
           and (x[1] <= 50).any() and x[3].get("PhaseEncodingDirection") not in (None, pe)
           and x[3].get("PhaseEncodingDirection", "")[:1] == (pe or "")[:1]]
    # Keep one reverse acquisition; it can have a different readout time from the main acquisition.
    reverse = max(rev, key=lambda x: int((x[1] <= 50).sum())) if rev and pe else None
    rev_b0 = np.asanyarray(reverse[0].dataobj)[..., reverse[1] <= 50].astype(np.float32) if reverse else None
    if not (bvals <= 50).any():
        raise ValueError("diffusion dataset has no b0 reference")
    if not np.isfinite(bvecs).all() or np.any(np.linalg.norm(bvecs[:, bvals > 50], axis=0) < 1e-6):
        raise ValueError("missing or invalid diffusion gradients")
    return {"data": data, "affine": best[0][0].affine, "bvals": bvals, "bvecs": bvecs, "meta": meta, "n_runs": len(best),
            "shells": sorted(set(int(round(x / 100.0)) * 100 for x in bvals if x > 50)), "rev_b0": rev_b0,
            "rev_meta": reverse[3] if reverse else {}, "source_nifti": [x[0].get_filename() for x in best]}


# ------------------------------------------------------------------------------------------ preprocess / fit
def denoise_dwi(ds, work_dir=None):
    """Marchenko-Pastur PCA denoising then Gibbs-ringing removal on the raw volumes, before any interpolation.
    MRtrix3's dwidenoise / mrdegibbs are used when on PATH (brain-masked, 5^3 patch, 4 threads: a few minutes per
    subject); otherwise DIPY's mppca and gibbs_removal (the same methods; ~4 + 9 min per three-shell subject). Lowers the noise floor that biases
    the free-water fit (most of all the single-shell one). Returns the dataset with denoised data."""
    if shutil.which("dwidenoise") and shutil.which("mrdegibbs") and work_dir:
        from dipy.segment.mask import median_otsu

        work = Path(work_dir) / "denoise"
        work.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(ds["data"], ds["affine"]), work / "raw.nii.gz")
        # a dilated brain mask halves the work, and the patch is capped at 5^3: MRtrix's default (the smallest patch with
        # more voxels than volumes) reaches 7^3 for the 200-volume three-shell data and then costs ~2 CPU-hours per subject
        _, mask = median_otsu(ds["data"][..., ds["bvals"] <= 50].mean(axis=3), median_radius=3, numpass=2)
        nib.save(nib.Nifti1Image(ndimage.binary_dilation(mask, iterations=4).astype(np.uint8), ds["affine"]), work / "mask.nii.gz")
        for cmd in (["dwidenoise", "raw.nii.gz", "dn.nii.gz", "-mask", "mask.nii.gz", "-extent", "5,5,5"], ["mrdegibbs", "dn.nii.gz", "dg.nii.gz"]):
            r = subprocess.run(cmd + ["-force", "-quiet", "-nthreads", "4"], cwd=work, capture_output=True, text=True)
            if r.returncode:
                log.warning("%s failed: %s", cmd[0], r.stderr[-300:])
                break
        else:
            data = np.asanyarray(nib.load(work / "dg.nii.gz").dataobj).astype(np.float32)
            shutil.rmtree(work, ignore_errors=True)
            return dict(ds, data=np.clip(data, 0, None), denoised=True)
    from dipy.denoise.gibbs import gibbs_removal
    from dipy.denoise.localpca import mppca
    from dipy.segment.mask import median_otsu

    _, mask = median_otsu(ds["data"][..., ds["bvals"] <= 50].mean(axis=3), median_radius=3, numpass=2)
    mask = ndimage.binary_dilation(mask, iterations=3)
    data = mppca(ds["data"], mask=mask, patch_radius=2)
    data[~mask] = ds["data"][~mask]                     # mppca zeroes voxels outside the mask; keep the raw background
    data = gibbs_removal(data, slice_axis=2, num_processes=1)
    return dict(ds, data=np.clip(data, 0, None).astype(np.float32), denoised=True)


def _crop_even(ds):
    """Drop the last slice/row of every odd-sized axis: topup's b02b0.cnf subsamples by 2 and needs even dimensions."""
    odd = [s % 2 for s in ds["data"].shape[:3]]
    if not any(odd):
        return ds
    sl = tuple(slice(0, s - o) for s, o in zip(ds["data"].shape[:3], odd))
    return dict(ds, data=ds["data"][sl], rev_b0=ds["rev_b0"][sl])


FSLDIR = os.environ.get("FSLDIR") or (str(Path.home() / "fsl") if (Path.home() / "fsl" / "bin" / "topup").exists() else None)


def susceptibility_correct(ds, work_dir, threads=2):
    """FSL topup on the mean b0 of the main series and the mean reverse-phase b0, then applytopup (Jacobian
    modulation) on every volume. Needs PhaseEncodingDirection and TotalReadoutTime (dcm2niix json) and a
    reverse-phase b0 series (PPMI-2 Prisma, GE 'Ax DWI B-0 A/P'); otherwise returns the dataset unchanged.
    eddy was measured at 26 min/subject on the RTX 2080 for the three-shell data and is not run."""
    meta, rev = ds["meta"], ds.get("rev_b0")
    pe, trt = meta.get("PhaseEncodingDirection"), meta.get("TotalReadoutTime")
    rev_trt = ds.get("rev_meta", {}).get("TotalReadoutTime")
    if not FSLDIR or rev is None or not pe or not trt or not rev_trt:
        return dict(ds, topup=False)
    ds = _crop_even(ds)
    rev = ds["rev_b0"]
    work = Path(work_dir) / "topup"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FSLDIR=FSLDIR, PATH=f"{FSLDIR}/bin:" + os.environ.get("PATH", ""), FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS=str(threads))
    b0_main = ds["data"][..., ds["bvals"] <= 50].mean(axis=3)
    b0_rev = rev.mean(axis=3)
    nib.save(nib.Nifti1Image(np.stack([b0_main, b0_rev], axis=3), ds["affine"]), work / "b0_pair.nii.gz")
    vec = np.array({"i": [1, 0, 0], "j": [0, 1, 0], "k": [0, 0, 1]}[pe[0]]) * (-1 if pe.endswith("-") else 1)
    fmt = lambda v: " ".join(str(int(x)) for x in v)
    (work / "acqparams.txt").write_text(f"{fmt(vec)} {trt}\n{fmt(-vec)} {rev_trt}\n")
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


def _register_volume(fixed, moving, sampling=0.1, iterations=50, sampling_seed=0):
    """Rigid MI registration (SimpleITK, 2-level pyramid) of one DWI volume to the b0 reference. Returns the Euler transform."""
    import SimpleITK as sitk

    tx = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed=int(sampling_seed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=iterations, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2])
    reg.SetSmoothingSigmasPerLevel([2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed, moving)  # failures must reach the subject error/QC record, not masquerade as zero motion
    return tx


def rotate_bvec(bvec, affine, fixed_to_moving_lps):
    """Reorient one dcm2niix/FSL gradient into the motion-corrected reference grid.

    ITK resampling maps fixed -> moving, so use its inverse rotation. FSL bvecs use voxel axes with an
    x reflection for positive-determinant NIfTI affines; convert through physical LPS and back explicitly.
    A single common reflection does not change FA/MD, but volume-specific missing rotations bias the fit.
    """
    g = np.asarray(bvec, dtype=float)
    if np.linalg.norm(g) < 1e-8:
        return np.zeros(3)
    axes = np.asarray(affine, dtype=float)[:3, :3]
    axes = axes / np.linalg.norm(axes, axis=0)
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
        raise ValueError("sheared DWI affine: resample and reorient gradients explicitly")
    basis = np.diag([-1., -1., 1.]) @ axes
    if np.linalg.det(axes) > 0:
        basis[:, 0] *= -1
    rotation = np.asarray(fixed_to_moving_lps, dtype=float).reshape(3, 3)
    out = basis.T @ rotation.T @ basis @ g
    return out / np.linalg.norm(out)


def preprocess(ds, sampling_seed=0):
    """Brain mask + rigid motion correction of every volume to the mean b0 (SimpleITK mutual information, ~0.7 s per
    volume against ~5 s for DIPY's affine pipeline). Reorients FSL b-vectors for each volume's rotation.
    Returns the dataset with corrected data, gradients, mask,
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
    bvecs = np.asarray(ds["bvecs"], dtype=float).copy()
    fd, rot = [], []
    for i in range(data.shape[3]):
        mov = _sitk_native(data[..., i], ds["affine"])
        tx = _register_volume(ref, mov, sampling_seed=sampling_seed)
        bvecs[:, i] = rotate_bvec(bvecs[:, i], ds["affine"], tx.GetMatrix()) if not b0s[i] else 0.0
        cdata[..., i] = np.transpose(sitk.GetArrayFromImage(sitk.Resample(mov, ref, tx, sitk.sitkLinear, 0.0)), (2, 1, 0))
        p = np.array(tx.GetParameters())
        fd.append(float(np.linalg.norm(p[3:6])))
        rot.append(float(np.degrees(np.linalg.norm(p[:3]))))
    return dict(ds, data=cdata, bvecs=bvecs, bvecs_rotated=True, mask=mask, b0=cdata[..., b0s].mean(axis=3),
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
    f_map = np.full(mask.shape, np.nan, dtype=np.float32)
    fat_map = np.full(mask.shape, np.nan, dtype=np.float32)
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
            if not sol.success or not np.isfinite(sol.x).all():
                continue
            p = sol.x
        except Exception:
            continue  # a failed optimisation is missing data, never a successful measurement at its prior
        L = np.zeros((3, 3))
        L[np.tril_indices(3)] = p[2:8]
        evt = np.linalg.eigvalsh(L @ L.T * scale)
        f_map[i, j, k] = p[1]
        fat_map[i, j, k] = fractional_anisotropy(np.clip(evt, 1e-9, None)[None])[0]
    return f_map, fat_map


def fit_models(ds, fw_mask=None):
    """FA/MD/AD/RD (WLS tensor, b <= 1000, brain mask) and free-water FW / tissue FA inside ``fw_mask`` (or the brain):
    DIPY's multi-shell NLS (Hoy et al. 2014) when >= 2 non-zero shells, else the single-shell fit above. Multi-shell
    scans also get the free-water-corrected MD (``mdt``) and the DKI mean kurtosis (``mk``, WLS on b <= 2000, clipped to
    [0, 3]; noise-sensitive, so run with ``--denoise``) inside ``fw_mask``."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    b, v, data, mask = ds["bvals"], ds["bvecs"], ds["data"], ds["mask"]
    sel = b <= 1050
    gt = gradient_table(b[sel], bvecs=v[:, sel], b0_threshold=50)
    tf = TensorModel(gt, fit_method="WLS").fit(data[..., sel], mask=mask)
    out = {k: np.nan_to_num(getattr(tf, k)).astype(np.float32) for k in ("fa", "md", "ad", "rd")}
    fmask = mask if fw_mask is None else (mask & fw_mask)
    shells = sorted(set(int(round(x / 100.0)) * 100 for x in b if x > 50))
    if len(shells) >= 2:
        from dipy.reconst.dki import DiffusionKurtosisModel
        from dipy.reconst.fwdti import FreeWaterTensorModel

        sel2 = b <= 2050
        gt2 = gradient_table(b[sel2], bvecs=v[:, sel2], b0_threshold=50)
        fw = FreeWaterTensorModel(gt2).fit(data[..., sel2], mask=fmask)
        f, fat = np.nan_to_num(fw.f), np.nan_to_num(fw.fa)
        out["mdt"] = np.where(fmask, fw.md, np.nan).astype(np.float32)
        dk = DiffusionKurtosisModel(gt2, fit_method="WLS").fit(data[..., sel2], mask=fmask)
        out["mk"] = np.where(fmask, dk.mk(min_kurtosis=0, max_kurtosis=3), np.nan).astype(np.float32)
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


def register_b0_to_t1(b0_img, t1_img, t1_mask_img, sampling_seed=0):
    """Rigid (mutual information) registration of the mean b0 to the brain-masked conformed T1 at 2 mm.
    Returns (transform fixed(T1)->moving(b0), metric). EPI susceptibility distortion is corrected only upstream, by
    ``susceptibility_correct`` (FSL topup, ``--fsl``) where a reverse-phase b0 exists; a T1-guided B-spline
    restricted to the phase-encoding axis was tried here and cost 9 min per subject without improving the fit."""
    import SimpleITK as sitk

    fixed = _brain(t1_img, t1_mask_img)
    # native geometry: DWI acquisitions are oblique (AC-PC angled) and `_sitk_from_nib` keeps only the diagonal
    moving = _sitk_native(np.asanyarray(b0_img.dataobj).astype(np.float32), b0_img.affine)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=int(sampling_seed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.Euler3DTransform(init), inPlace=False)
    tx = reg.Execute(fixed, moving)
    return tx, float(reg.GetMetricValue())


def register_t1_to_mni(t1_img, t1_mask_img, cache_path=None, sampling_seed=0):
    """Affine T1 -> verified MNI152NLin2009cAsym; returns its fixed-to-moving pull.

    A versioned cache and provenance prevent reuse of legacy Nilearn-2009a maps.
    A supplied existing cache without matching provenance is rejected and preserved.
    """
    import SimpleITK as sitk
    from .atlases import mni2009c_template

    inputs = {'t1_sha256': _registration_image_hash(t1_img),
              'mask_sha256': _registration_image_hash(t1_mask_img), 'sampling_seed': int(sampling_seed)}

    if cache_path is not None and Path(cache_path).exists():
        return load_mni_cache(cache_path, expected_inputs=inputs), float("nan")

    fixed = _sitk_from_nib(mni2009c_template())
    moving = _brain(t1_img, t1_mask_img)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=int(sampling_seed))
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
        _write_mni_cache_provenance(cache_path, inputs)
    return tx, float(reg.GetMetricValue())


def mni_cache_path(fastsurfer_dir):
    return Path(fastsurfer_dir) / "mri" / "transforms" / "t1_to_MNI152NLin2009cAsym_affine_v2.tfm"


def _registration_image_hash(image):
    import hashlib
    digest = hashlib.sha256(np.asarray(image.affine, dtype='<f8').tobytes())
    data = np.asarray(image.dataobj, dtype='<f4')
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes(order='C'))
    return digest.hexdigest()


def _write_mni_cache_provenance(path, inputs):
    import hashlib
    from .atlases import mni2009c_template_metadata
    path = Path(path)
    meta = mni2009c_template_metadata()
    record = {'reference_space': meta['space'], 'reference_sha256': meta['sha256'],
              'transform_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'registration_version': 2, 'inputs': inputs}
    sidecar = path.with_suffix('.json')
    temporary = sidecar.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(sidecar)


def load_mni_cache(path, expected_inputs=None):
    """Reject cached transforms whose actual reference identity is unverified."""
    import hashlib
    import SimpleITK as sitk
    from .atlases import mni2009c_template_metadata
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if not sidecar.exists():
        raise ValueError('Unverified registration cache: use a fresh versioned MNI2009c cache')
    record = json.loads(sidecar.read_text())
    meta = mni2009c_template_metadata()
    if (record.get('reference_space') != meta['space'] or record.get('reference_sha256') != meta['sha256']
            or record.get('registration_version') != 2
            or record.get('transform_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()
            or (expected_inputs is not None and record.get('inputs') != expected_inputs)):
        raise ValueError('Registration cache provenance mismatch; preserve it and use a fresh cache')
    return sitk.ReadTransform(str(path))


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
    """CIT168 v1 labels transformed by the authors to the registration's MNI2009c space.

    Do not use fetch_atlas_pauli_2017 here: that deterministic file is native CIT168.
    """
    from .atlases import cit168_mni2009c

    return cit168_mni2009c()


# ------------------------------------------------------------------------------------------ features
def _left_mask(fs_lab):
    """Left-hemisphere half-space on the DWI grid from FastSurfer's explicit left/right putamen: midline = midpoint of
    the two centroids along the x index, left = the side of label 12 (orientation-independent)."""
    xl = np.nonzero(fs_lab == 12)[2].mean() if (fs_lab == 12).any() else fs_lab.shape[2] * 0.75
    xr = np.nonzero(fs_lab == 51)[2].mean() if (fs_lab == 51).any() else fs_lab.shape[2] * 0.25
    mid = (xl + xr) / 2
    xi = np.arange(fs_lab.shape[2])[None, None, :]
    return np.broadcast_to((xi > mid) if xl > xr else (xi < mid), fs_lab.shape)


def _roi_masks(fs_lab, pauli_lab, y_index):
    """ROI name -> boolean mask on the DWI grid (arrays in sitk (z, y, x) order). ``y_index`` gives each voxel's
    anterior-posterior coordinate for the anterior/posterior split of the substantia nigra."""
    rois = {}
    for name, (l, r) in FS_ROIS.items():
        rois[f"{name}_l"], rois[f"{name}_r"] = fs_lab == l, fs_lab == r
    for name, ids in FS_SINGLE.items():
        rois[name] = np.isin(fs_lab, ids)
    code = {n: i + 1 for i, n in enumerate(PAULI)}
    left = _left_mask(fs_lab)
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
    metrics = [k for k in METRICS if k in maps]      # mdt / mk exist for multi-shell scans only
    for name, m in rois.items():
        n = int(m.sum())
        out[f"n_{name}"] = n
        for k in metrics:
            vals = maps[k][m]
            vals = vals[np.isfinite(vals)]
            out[f"{name}_{k}"] = float(vals.mean()) if len(vals) >= min_voxels else np.nan
    # bilateral means for the headline measures
    for base in ("sn_posterior", "sn", "snc", "snr", "putamen", "caudate", "sn_posterior_t", "sn_t", "snc_t", "snr_t"):
        for k in metrics:
            l, r = out.get(f"{base}_l_{k}", np.nan), out.get(f"{base}_r_{k}", np.nan)
            out[f"{base}_mean_{k}"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    return out


# ------------------------------------------------------------------------------------------ per-subject driver
def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, fsl=False, denoise=False, fba=False, keep_preproc=False):
    """All steps for one subject. Returns a flat dict (features + QC). ``fba`` adds the MRtrix3 nigrostriatal fixel measures
    (pie.imaging.fba); ``keep_preproc`` keeps the preprocessed DWI + gradients under <work>/<patno>/fba/."""
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    work = Path(work_dir) / str(patno)
    work.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in series_rows:
        runs += convert(r["zip"], r["prefix"], work / "nii")
    ds = assemble(runs)
    if denoise:
        ds = denoise_dwi(ds, work)
    row = {"patno": patno, "denoised": bool(ds.get("denoised", False)), "n_series": len(series_rows), "n_runs_used": ds["n_runs"], "n_volumes": int(ds["data"].shape[3]),
           "shells": " ".join(map(str, ds["shells"])), "voxel_mm": float(np.round(np.linalg.norm(ds["affine"][:3, :3], axis=0).mean(), 2)),
           "manufacturer": str(ds["meta"].get("Manufacturer", "")), "model": str(ds["meta"].get("ManufacturerModelName", "")),
           "pe_direction": str(ds["meta"].get("PhaseEncodingDirection", "")), "readout_s": ds["meta"].get("TotalReadoutTime", np.nan),
           "series_desc": ";".join(sorted(set(r["desc"] for r in series_rows)))}
    if fsl:
        ds = susceptibility_correct(ds, work)
    row["topup"] = bool(ds.get("topup", False))
    row["n_rev_b0"] = int(ds["rev_b0"].shape[3]) if ds.get("rev_b0") is not None else 0
    ds = preprocess(ds)
    row.update({"motion_mm_mean": ds["motion_mm_mean"], "motion_mm_max": ds["motion_mm_max"], "rotation_deg_max": ds["rotation_deg_max"],
                "bvecs_rotated": ds["bvecs_rotated"], "processing_version": "2026-09-09-acquisition-metadata-v3",
                "fs_image_id": Path(fastsurfer_dir).name, "acquisition_date": series_rows[0]["date"],
                "source_image_ids": ";".join(sorted({Path(p).name.split("_")[0] for p in ds["source_nifti"]}))})
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
    fitted_mask = fw_mask_xyz & ds["mask"]
    row["fw_fit_valid_fraction"] = float(np.isfinite(maps["fw"][fitted_mask]).mean()) if fitted_mask.any() else np.nan
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
    if fba or keep_preproc:
        from .fba import nigrostriatal, write_preproc
        (work / "fba").mkdir(exist_ok=True)
        write_preproc(ds, work / "fba")
    if fba:
        left = _left_mask(fs_lab)
        aux = {"hemi_l": left, "hemi_r": ~left, "cerebellum": np.isin(fs_lab, [7, 8, 46, 47])}
        try:
            row.update(nigrostriatal(ds, {**rois, **aux}, maps, work, threads=2))
        except Exception as e:      # the tensor features stand on their own
            row["fba_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if not keep_preproc:
            for f in ("preproc.nii.gz", "preproc.bval", "preproc.bvec"):
                (work / "fba" / f).unlink(missing_ok=True)
    if keep_nifti:
        for k, v in maps.items():
            nib.save(nib.Nifti1Image(np.transpose(v, (2, 1, 0)), ds["affine"]), work / f"{k}.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(pauli_lab, (2, 1, 0)).astype(np.int16), ds["affine"]), work / "pauli_dwi.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(fs_lab, (2, 1, 0)).astype(np.int16), ds["affine"]), work / "aseg_dwi.nii.gz")
        nib.save(b0_img, work / "b0.nii.gz")
    shutil.rmtree(work / "nii", ignore_errors=True)     # raw conversions (~130 MB/subject) are reproducible from the zips
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep, fsl, denoise, fba, keep_preproc = args
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep, fsl=fsl, denoise=denoise, fba=fba, keep_preproc=keep_preproc)
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
    ap.add_argument("--denoise", action="store_true", help="MP-PCA denoising + Gibbs-ringing removal (DIPY) before motion correction")
    ap.add_argument("--fba", action="store_true", help="MRtrix3 nigrostriatal fixel measures (FOD, iFOD2 tractography SN -> striatum, AFD along the tract)")
    ap.add_argument("--keep-preproc", action="store_true", help="keep the preprocessed DWI + gradients under <work>/<patno>/fba/")
    ap.add_argument("--priority", help="text file of PATNOs to process first")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    idx = load_index(work / "dwi_index.csv", a.zips, _flag_dwi)
    idx = idx[idx["selected"]]
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    out_csv = work / "dwi_features.csv"
    done = done_subjects(out_csv, a.retry_errors)
    jobs = [(int(patno), session_rows(g), fs[int(patno)], str(work), a.keep_nifti, a.fsl, a.denoise, a.fba, a.keep_preproc)
            for patno, g in idx.groupby("patno") if patno not in done and int(patno) in fs]
    if a.priority:
        order = {int(p): i for i, p in enumerate(Path(a.priority).read_text().split())}
        jobs.sort(key=lambda j: order.get(j[0], len(order)))
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, pid_file=a.pid_file)


# ====================================================================================================
# JHU ICBM-DTI-81 white-matter tract measures
# Merged from dwi_tracts.py; kept together with the rest of the dwi measures.
#
# White-matter tract measures on the JHU ICBM-DTI-81 atlas (48 labels).
#
# The atlas most studies use for tract-level FA (Mori 2005, Wakana 2007, Hua 2008; e.g. Droby et al. 2025,
# npj Parkinson's Disease). It is fetched at run time from its NeuroVault release (collection 264), not bundled:
# that release states no licence.
#
# Template space is never assumed. The atlas is distributed on a generic "MNI" grid whose exact flavour is
# undeclared, so labels are brought to each subject by registering the atlas's *own* FA template, which shares
# the label grid voxel for voxel, to the subject's FA map (affine then SyN, cross-correlation). No MNI152
# variant is involved at any step, which removes the class of error in which an atlas is read in the wrong
# template space.
#
#     fetch_jhu(cache_dir)              download + verify the label atlas and its FA template (sha256, grid, laterality)
#     map_labels_to_subject(fa, ...)    atlas FA -> subject FA registration; labels pulled with genericLabel interpolation
#     registration_qc(...)              correlation of warped template FA with subject FA, label retention, Jacobians
#     tract_features(maps, labels)      mean of each map (FA, MD, AD, RD, ...) per tract, with voxel counts
#
# Everything outside the registration is plain NumPy and is exercised by tests on synthetic volumes.
# ====================================================================================================

NEUROVAULT = "https://neurovault.org/media/images/264/"
FILES = {"labels": "JHU-ICBM-labels-1mm.nii.gz", "fa": "JHU-ICBM-FA-1mm.nii.gz"}
# sha256 of the NeuroVault files as first verified on 18 September 2026 (grid 182 x 218 x 182, 1 mm, LAS)
SHA256 = {"labels": "3c2bc4a2aab93d388afc8387a2c0af9c20c92beefb6f28030fb4e8c1ffa8989d",
          "fa": "ae88370bbfd37c72dd3ceeb9c625b2e8d0aba4bb952aa5f6a0e392dd40876b27"}

# ICBM-DTI-81 label names, in label order 1..48 (FSL JHU-labels.xml). For 7..48, odd = right, even = left.
_BILATERAL = [
    "corticospinal_tract", "medial_lemniscus", "inferior_cerebellar_peduncle", "superior_cerebellar_peduncle",
    "cerebral_peduncle", "anterior_limb_internal_capsule", "posterior_limb_internal_capsule",
    "retrolenticular_internal_capsule", "anterior_corona_radiata", "superior_corona_radiata",
    "posterior_corona_radiata", "posterior_thalamic_radiation", "sagittal_stratum", "external_capsule",
    "cingulum_cingulate_gyrus", "cingulum_hippocampus", "fornix_cres_stria_terminalis",
    "superior_longitudinal_fasciculus", "superior_fronto_occipital_fasciculus", "uncinate_fasciculus", "tapetum"]
LABELS = {1: "middle_cerebellar_peduncle", 2: "pontine_crossing_tract", 3: "genu_corpus_callosum",
          4: "body_corpus_callosum", 5: "splenium_corpus_callosum", 6: "fornix_column_body"}
for _i, _name in enumerate(_BILATERAL):
    LABELS[7 + 2 * _i] = _name + "_r"
    LABELS[8 + 2 * _i] = _name + "_l"
assert sorted(LABELS) == list(range(1, 49))


def _download(url, path):
    """Fetch with an explicit User-Agent: NeuroVault's CDN answers 403 to urllib's default one."""
    req = urllib.request.Request(url, headers={"User-Agent": "parkinsons-insight-engine (+https://github.com/MJFF-ResearchCommunity/PIE)"})
    with urllib.request.urlopen(req, timeout=120) as response, open(path, "wb") as out:
        out.write(response.read())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_ants(img):
    import ants
    # registration inputs only: real FA and T1 maps carry NaN outside the brain, which ANTs rejects
    data = np.nan_to_num(np.asarray(img.dataobj, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ants.from_nibabel_nifti(nib.Nifti1Image(data, img.affine))


def _from_ants(ants_img, like):
    """Back to nibabel on the grid of ``like``; refuses any output whose geometry differs from ``like``."""
    import ants
    out = ants.to_nibabel_nifti(ants_img)
    if out.shape[:3] != like.shape[:3] or not np.allclose(out.affine, like.affine, atol=1e-3):
        raise ValueError("resampled image is not on the subject grid; orientation handling is wrong")
    return np.asarray(out.dataobj)


def check_laterality(labels_img, min_separation_mm=4.0):
    """Every label named ``_r`` must lie to the right (+x in scanner RAS) of its ``_l`` partner.

    Returns {pair: right_minus_left_x_mm}; raises if any pair is swapped or not separated.
    """
    data = np.asarray(labels_img.dataobj)
    out = {}
    for k, name in LABELS.items():
        if not name.endswith("_r"):
            continue
        cx = []
        for label in (k, k + 1):
            vox = np.argwhere(data == label)
            if not len(vox):
                raise ValueError(f"label {label} ({LABELS[label]}) is empty")
            cx.append(nib.affines.apply_affine(labels_img.affine, vox)[:, 0].mean())
        out[name[:-2]] = float(cx[0] - cx[1])
    bad = {k: v for k, v in out.items() if v < min_separation_mm}
    if bad:
        raise ValueError(f"left/right labels swapped or not separated: {bad}")
    return out


def fetch_jhu(cache_dir, download=True):
    """The JHU label atlas and its FA template, verified; returns (labels_img, fa_img, provenance)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            _download(NEUROVAULT + name, path)
        if _sha(path) != SHA256[key]:
            raise ValueError(f"{name}: checksum differs from the verified NeuroVault release")
        paths[key] = path
    labels, fa = nib.load(paths["labels"]), nib.load(paths["fa"])
    if labels.shape != fa.shape or not np.allclose(labels.affine, fa.affine, atol=1e-6):
        raise ValueError("JHU labels and FA template are not on one grid")
    values = np.unique(np.asarray(labels.dataobj))
    if not np.array_equal(values, np.arange(49)):
        raise ValueError("JHU label identity differs from ICBM-DTI-81 (0..48)")
    lateral = check_laterality(labels)
    provenance = {"atlas": "JHU ICBM-DTI-81 white-matter labels", "source": NEUROVAULT, "sha256": dict(SHA256),
                  "registration_target": "atlas FA template on the same grid (no MNI variant assumed)",
                  "laterality_right_minus_left_mm": lateral}
    return labels, fa, provenance


def map_labels_to_subject(subject_fa, atlas_fa, atlas_labels, brain_mask=None, seed=0, syn=True, max_resolution_mm=None):
    """Register the atlas FA template to the subject FA (affine, then SyN) and pull the labels.

    ``max_resolution_mm`` (e.g. 2.0): when the subject grid is finer than this, registration runs on a copy
    resampled to that isotropic spacing, and the labels are still pulled onto the native grid (transforms live in
    physical space). Scans reconstructed at 1 x 1 x 2 mm then register as fast as native 2 mm scans, and at the
    resolution the diffusion data actually carry. Returns (labels_img on the subject FA grid, {"warped_template_fa",
    "type", "registration_spacing_mm"}); the transform files are deleted, since ANTsPy leaves them in the temp directory.
    Labels use ``genericLabel`` interpolation.
    """
    import ants

    native = _to_ants(subject_fa)
    if brain_mask is not None:
        native = native * _to_ants(brain_mask)
    fixed = native
    spacing = np.asarray(subject_fa.header.get_zooms()[:3], float)
    if max_resolution_mm is not None and spacing.min() < 0.95 * max_resolution_mm:
        fixed = ants.resample_image(native, (max_resolution_mm,) * 3, use_voxels=False, interp_type=0)
    moving = _to_ants(atlas_fa)
    kind = "SyN" if syn else "Affine"
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=kind, random_seed=int(seed),
                            syn_metric="CC", syn_sampling=4)
    warped = ants.apply_transforms(fixed=native, moving=_to_ants(atlas_labels), transformlist=reg["fwdtransforms"],
                                   interpolator="genericLabel")
    out = nib.Nifti1Image(np.rint(_from_ants(warped, subject_fa)).astype(np.int16), subject_fa.affine)
    warped_fa = ants.apply_transforms(fixed=native, moving=moving, transformlist=reg["fwdtransforms"], interpolator="linear")
    from .features import _drop_transforms
    _drop_transforms(reg)
    return out, {"warped_template_fa": _from_ants(warped_fa, subject_fa), "type": kind,
                 "registration_spacing_mm": [float(v) for v in fixed.spacing]}


def registration_qc(subject_fa, warped_template_fa, subject_labels, atlas_labels, brain_mask,
                    min_correlation=0.5, min_retention=0.5):
    """Checks that must pass before tract values are used.

    template_fa_correlation   Pearson r of warped template FA with subject FA inside the brain mask; on PPMI 2 mm data
                              SyN gave 0.69 where affine alone gave 0.54, with correspondingly higher tract FA
    label_retention           per label, subject-space volume / atlas volume (voxel counts scaled by voxel size)
    """
    s = np.asarray(subject_fa.dataobj, float)
    w = np.asarray(warped_template_fa, float)
    # A real brain mask is required: FA is non-zero noise outside the brain, and averaging over the field of view
    # understates alignment (on PPMI data r fell from 0.69 to 0.50 for the same registration).
    if brain_mask is None:
        raise ValueError("registration_qc needs a brain mask (e.g. dipy median_otsu on the b0)")
    inside = (np.asarray(brain_mask.dataobj) > 0) & np.isfinite(s) & np.isfinite(w)
    r = float(np.corrcoef(s[inside], w[inside])[0, 1])
    sub_vox = float(abs(np.linalg.det(subject_fa.affine[:3, :3])))
    atl_vox = float(abs(np.linalg.det(atlas_labels.affine[:3, :3])))
    sub, atl = np.asarray(subject_labels.dataobj), np.asarray(atlas_labels.dataobj)
    retention = {LABELS[k]: float((sub == k).sum() * sub_vox / max((atl == k).sum() * atl_vox, 1e-9)) for k in LABELS}
    failed = [k for k, v in retention.items() if v < min_retention]
    passed = r >= min_correlation and not failed
    return {"template_fa_correlation": r, "label_retention": retention, "labels_below_retention": failed,
            "qc_pass": bool(passed)}


def tract_features(maps, labels_img, min_voxels=10, fa_floor=None, fa_key="fa"):
    """Mean of every map in ``maps`` ({"fa": img, "md": img, ...}, same grid as the labels) per JHU tract.

    ``fa_floor`` (e.g. 0.2, the TBSS convention) restricts each tract to voxels with FA above it, which reduces
    partial volume with grey matter and CSF; default None averages the whole tract, as most atlas studies do.
    Tracts with fewer than ``min_voxels`` usable voxels are returned as NaN, with their count.
    """
    labels = np.asarray(labels_img.dataobj)
    arrays = {}
    for key, img in maps.items():
        if img.shape[:3] != labels.shape or not np.allclose(img.affine, labels_img.affine, atol=1e-4):
            raise ValueError(f"{key} map and labels are not on one grid")
        arrays[key] = np.asarray(img.dataobj, float)
    if fa_floor is not None and fa_key not in arrays:
        raise ValueError("fa_floor needs the FA map")
    out = {}
    for k, name in LABELS.items():
        sel = labels == k
        for arr in arrays.values():
            sel &= np.isfinite(arr)
        if fa_floor is not None:
            sel &= arrays[fa_key] > fa_floor
        n = int(sel.sum())
        out[f"n_{name}"] = n
        for key, arr in arrays.items():
            out[f"{key}_{name}"] = float(arr[sel].mean()) if n >= min_voxels else float("nan")
    return out


def write_provenance(path, provenance, qc):
    Path(path).write_text(json.dumps({"atlas": provenance, "qc": qc}, indent=1, default=float))


if __name__ == "__main__":
    main()
