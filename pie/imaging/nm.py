"""
Neuromelanin-sensitive MRI (PPMI-2 "2D GRE-MT" family) -> nigral contrast ratio and neuromelanin volume.

PPMI-2 acquires a 2D T1-weighted gradient echo with a magnetization-transfer pulse (0.5 x 0.5 x 1.5 mm, 16 slices
through the midbrain, TR ~0.45-0.65 s, TE ~5 ms, FA 40), typically five repeats to be averaged; descriptions vary
by site ("AX T2 GRE MT", "2D GRE-MT", "AXIAL 2D GRE-MT", "2D GRE-MT_ACPC", "NM-GRE", "NM-MT", ...). Per subject:

1. `convert`      dcm2niix per series; repeats with the same geometry are rigidly aligned to the first and averaged.
2. `register`     mean NM slab -> conformed T1 (rigid, mutual information, header-initialised: same session);
                  T1 -> MNI affine (shared with `pie.imaging.dwi`) brings the CIT168 atlas (Pauli 2017) SNc/SNr/RN/VTA/STN
                  onto the NM grid together with the FastSurfer brainstem / ventral DC labels.
3. `features`     substantia nigra (SNc + SNr, left/right, anterior/posterior halves): mean signal, contrast ratio
                  CNR = (SN - ref) / ref against the crus cerebri (the part of a surrounding-midbrain ring, the
                  atlas SN dilated 5 mm minus the nuclei inside brainstem/ventral DC, that lies anterior to the SN on
                  the same side; the whole-ring CNR is kept as `*_cnr_ring`); placement-robust measures on a 1 mm-
                  smoothed image: the contrast of the brightest half-SN-sized volume within the dilated search region
                  (`*_top_cnr`) and the neuromelanin volume = search-region voxels above a fixed 10 % contrast
                  (`nm_vol_*_voxels`, with their mean CNR). QC: repeats, inter-repeat motion, registration metric,
                  slab coverage of the SN. The slab (~24 mm) does not reach the locus coeruleus.

    venv_imaging/bin/python -m pie.imaging.nm --zips <MRI zips> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4 [--keep-nifti]
"""

import json
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
from .dwi import PAULI, _brain, _register_volume, _sitk_from_nib, _sitk_native, labels_to_dwi, mni_cache_path, pauli_atlas, register_t1_to_mni

NM_PATTERN = r"GRE.?MT|MT.?GRE|GRE ?- ?MT|NM-|Neuromelanin|NM_MT|NM MT"
EXCLUDE = r"MTC-NO|B0|Map|TRACEW|ADC|_FA"
CNR_THRESHOLD = 0.10   # neuromelanin volume: smoothed voxels > (1 + CNR_THRESHOLD) x crus cerebri mean
TOP_FRACTION = 0.5     # placement-robust contrast: the brightest (TOP_FRACTION x atlas-SN volume) voxels of the dilated search region
DILATE_MM = 3.0


# ------------------------------------------------------------------------------------------ index / convert
def flag_nm(idx):
    d = idx["desc"].str.replace("_", " ")
    return d.str.contains(NM_PATTERN, case=False, regex=True) & ~d.str.contains(EXCLUDE, case=False, regex=True)


def index_nm(zips):
    from .batch import index_series

    idx = index_series(zips)
    idx["nm"] = flag_nm(idx)
    return idx


from .batch import convert_series as convert  # noqa: E402  (kept as the module-level name used by process_subject)


def average_repeats(niis):
    """Load repeats (same shape/spacing), rigidly align each to the first, average. Returns (img, meta, n_used, motion_mm_max)."""
    import SimpleITK as sitk

    imgs = []
    for p in niis:
        img = nib.load(p)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        if data.ndim == 4:  # some vendors stack repeats in one file
            for i in range(data.shape[3]):
                imgs.append((nib.Nifti1Image(data[..., i], img.affine), p))
        elif data.ndim == 3:
            imgs.append((img, p))
    if not imgs:
        raise ValueError("no neuromelanin volume")
    shapes = {}
    for img, p in imgs:
        shapes.setdefault((img.shape, tuple(np.round(img.header.get_zooms()[:3], 2))), []).append((img, p))
    group = max(shapes.values(), key=len)
    ref_img = group[0][0]
    ref = _sitk_native(np.asanyarray(ref_img.dataobj).astype(np.float32), ref_img.affine)
    acc = sitk.GetArrayFromImage(ref).astype(np.float64)
    motion = [0.0]
    for img, _ in group[1:]:
        mov = _sitk_native(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
        tx = _register_volume(ref, mov, sampling=0.2, iterations=60)
        acc += sitk.GetArrayFromImage(sitk.Resample(mov, ref, tx, sitk.sitkLinear, 0.0))
        motion.append(float(np.linalg.norm(np.array(tx.GetParameters())[3:6])))
    mean = (acc / len(group)).astype(np.float32)
    meta_path = group[0][1][:-7] + ".json"
    meta = json.load(open(meta_path)) if Path(meta_path).exists() else {}
    return nib.Nifti1Image(np.transpose(mean, (2, 1, 0)), ref_img.affine), meta, len(group), float(max(motion))


# ------------------------------------------------------------------------------------------ registration / ROIs
def register_nm_to_t1(nm_img, t1_img, target_center=None):
    """Rigid MI registration between the NM slab (fixed: every metric sample lies inside the 24 mm slab) and the
    *unmasked* conformed T1 (moving). The full head matters: with a brain-masked T1 a thin slab of brain tissue
    matches several heights equally well and the optimizer settled on the striatum for some subjects; the eyes,
    sinuses and skull in the slab pin its height. ``target_center`` (LPS mm) initialises the slab centre there
    (atlas SN centroid in T1 space); otherwise the same-session headers are trusted.
    Returns (transform fixed(T1)->moving(NM), metric), the convention used by ``labels_to_dwi``."""
    import SimpleITK as sitk

    t1 = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(t1_img.dataobj).astype(np.float32), t1_img.affine))
    slab = _sitk_native(np.asanyarray(nm_img.dataobj).astype(np.float32), nm_img.affine)
    tx = sitk.Euler3DTransform()
    center = np.array(slab.TransformContinuousIndexToPhysicalPoint([s / 2 for s in slab.GetSize()]))
    tx.SetCenter(tuple(float(c) for c in center))
    if target_center is not None:
        tx.SetTranslation(tuple(float(v) for v in (np.asarray(target_center) - center)))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    try:
        reg.Execute(slab, t1)          # tx: slab (fixed) point -> T1 (moving) point
        metric = float(reg.GetMetricValue())
    except RuntimeError:
        metric = np.nan
    return tx.GetInverse(), metric     # T1 point -> slab point


def nm_rois(fs_lab, pauli_lab, spacing_mm):
    """Masks on the NM grid: SN (SNc+SNr) left/right, nuclei union, search region (SN dilated DILATE_MM) and the
    reference ring (SN dilated DILATE_MM minus dilated nuclei, inside brainstem / ventral DC / peduncle WM). The search region
    for the threshold measures is the dilated SN restricted to brainstem / ventral DC labels minus the other nuclei."""
    code = {n: i + 1 for i, n in enumerate(PAULI)}
    sn = np.isin(pauli_lab, [code["SNc"], code["SNr"]])
    nuclei = np.isin(pauli_lab, [code["SNc"], code["SNr"], code["RN"], code["VTA"], code["STH"], code["PBP"]])
    it = [max(1, int(round(DILATE_MM / s))) for s in spacing_mm]  # anisotropic dilation via separate passes
    struct = np.zeros((3, 3, 3), bool)
    struct[1, 1, :] = struct[1, :, 1] = struct[:, 1, 1] = True
    search = sn.copy()
    for ax, n in zip((0, 1, 2), it):
        st = np.zeros((3, 3, 3), bool)
        sl = [1, 1, 1]
        sl[ax] = slice(None)
        st[tuple(sl)] = True
        search = ndimage.binary_dilation(search, structure=st, iterations=n)
    near = ndimage.binary_dilation(nuclei, structure=struct, iterations=1)
    midbrain = np.isin(fs_lab, [16, 28, 60, 2, 41, 10, 49])  # brainstem, ventral DC, cerebral WM (peduncle), thalamus edge
    ref = search & ~near & midbrain
    # the neuromelanin search region stays inside brainstem / ventral DC and away from the other nuclei: outside
    # them lie the cisterns, where arteries are bright on gradient-echo images and swamp any threshold measure
    other = ndimage.binary_dilation(nuclei & ~sn, structure=struct, iterations=1)
    search = search & np.isin(fs_lab, [16, 28, 60]) & ~other
    xl = np.nonzero(fs_lab == 12)[2].mean() if (fs_lab == 12).any() else pauli_lab.shape[2] * 0.75
    xr = np.nonzero(fs_lab == 51)[2].mean() if (fs_lab == 51).any() else pauli_lab.shape[2] * 0.25
    mid = (xl + xr) / 2
    xi = np.arange(pauli_lab.shape[2])[None, None, :]
    left = (xi > mid) if xl > xr else (xi < mid)
    return {"sn_l": sn & left, "sn_r": sn & ~left, "search_l": search & left, "search_r": search & ~left, "ref": ref}


def features(nm, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), smooth_fwhm_mm=1.0):
    """Signal, CNR and thresholded neuromelanin volume per side (+ anterior/posterior halves of the SN).
    Reference = crus cerebri: the part of the surrounding-midbrain ring anterior to the SN on the same side (the
    whole ring mixes bright tegmentum with the dark peduncle and halves the contrast). The threshold volume uses a
    lightly smoothed image (``smooth_fwhm_mm`` in-plane) so that ref mean + K_SD * SD reflects tissue, not voxel noise."""
    out = {}
    sigma = [0.0] + [smooth_fwhm_mm / 2.3548 / s for s in spacing_zyx[1:]]
    sm = ndimage.gaussian_filter(nm, sigma=sigma) if smooth_fwhm_mm else nm
    ring_vals = nm[rois["ref"]]
    ring_vals = ring_vals[np.isfinite(ring_vals) & (ring_vals > 0)]
    if len(ring_vals) < 20:
        return {"nm_error": "reference region empty"}
    out.update({"nm_ring_mean": float(ring_vals.mean()), "nm_ring_sd": float(ring_vals.std()), "n_ring": int(len(ring_vals))})
    for side in ("l", "r"):
        sn, search = rois[f"sn_{side}"], rois[f"search_{side}"]
        out[f"n_sn_{side}"] = int(sn.sum())
        if not sn.any():
            continue
        y_sn = np.median(phys_y[sn])
        crus = rois["ref"] & search & (phys_y < y_sn - 1.5)          # anterior (LPS: smaller y) to the SN, same side
        cv = nm[crus]
        cv = cv[cv > 0]
        if len(cv) < 20:
            crus = rois["ref"] & search
            cv = nm[crus]
            cv = cv[cv > 0]
        ref_mean, ref_sd = float(cv.mean()), float(cv.std())
        out[f"nm_ref_{side}_mean"], out[f"nm_ref_{side}_sd"], out[f"n_ref_{side}"] = ref_mean, ref_sd, int(len(cv))
        vals = nm[sn]
        vals = vals[vals > 0]
        out[f"nm_sn_{side}_mean"] = float(vals.mean()) if len(vals) >= 3 else np.nan
        out[f"nm_sn_{side}_cnr"] = float((vals.mean() - ref_mean) / ref_mean) if len(vals) >= 3 else np.nan
        out[f"nm_sn_{side}_cnr_ring"] = float((vals.mean() - out["nm_ring_mean"]) / out["nm_ring_mean"]) if len(vals) >= 3 else np.nan
        # placement-robust measures on the smoothed image, relative to the smoothed crus mean: the brightest quarter of
        # the dilated search region (the neuromelanin band is thin and the atlas SN can sit 1-2 mm off it), and the
        # neuromelanin volume above a fixed CNR_THRESHOLD (SD-based thresholds are dominated by peduncle anatomy)
        cs_mean = float(sm[crus].mean())
        sv = sm[search]
        top = np.sort(sv)[-max(3, int(sn.sum() * TOP_FRACTION)):]   # brightest half-SN-sized volume inside the search region
        out[f"nm_sn_{side}_top_cnr"] = float((top.mean() - cs_mean) / cs_mean)
        hot = search & (sm > cs_mean * (1 + CNR_THRESHOLD))
        out[f"nm_vol_{side}_voxels"] = int(hot.sum())
        out[f"nm_vol_{side}_cnr"] = float((sm[hot].mean() - cs_mean) / cs_mean) if hot.sum() >= 3 else np.nan
        cut = np.median(phys_y[sn])
        for name, m in (("posterior", sn & (phys_y > cut)), ("anterior", sn & (phys_y <= cut))):   # LPS: larger y = posterior
            v = nm[m]
            v = v[v > 0]
            out[f"nm_sn_{name}_{side}_cnr"] = float((v.mean() - ref_mean) / ref_mean) if len(v) >= 3 else np.nan
    for base in ("nm_sn", "nm_sn_posterior", "nm_sn_anterior"):
        l, r = out.get(f"{base}_l_cnr", np.nan), out.get(f"{base}_r_cnr", np.nan)
        out[f"{base}_mean_cnr"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    l, r = out.get("nm_sn_l_top_cnr", np.nan), out.get("nm_sn_r_top_cnr", np.nan)
    out["nm_sn_mean_top_cnr"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    out["nm_sn_min_top_cnr"] = float(np.nanmin([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    out["nm_vol_total_voxels"] = out.get("nm_vol_l_voxels", 0) + out.get("nm_vol_r_voxels", 0)
    l, r = out.get("nm_sn_l_cnr", np.nan), out.get("nm_sn_r_cnr", np.nan)
    out["nm_sn_asym_cnr"] = float(abs(l - r)) if np.isfinite(l) and np.isfinite(r) else np.nan
    return out


# ------------------------------------------------------------------------------------------ driver
def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False):
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    work = Path(work_dir) / str(patno)
    work.mkdir(parents=True, exist_ok=True)
    niis = []
    for r in series_rows:
        niis += convert(r["zip"], r["prefix"], work / "nii")
    nm_img, meta, n_rep, motion = average_repeats(niis)
    row = {"patno": patno, "n_series": len(series_rows), "n_repeats": n_rep, "repeat_motion_mm_max": motion,
           "shape": "x".join(map(str, nm_img.shape)), "voxel_mm": "x".join(str(round(float(z), 2)) for z in nm_img.header.get_zooms()[:3]),
           "manufacturer": str(meta.get("Manufacturer", "")), "model": str(meta.get("ManufacturerModelName", "")),
           "tr_s": meta.get("RepetitionTime", np.nan), "te_s": meta.get("EchoTime", np.nan), "flip_angle": meta.get("FlipAngle", np.nan),
           "mt_flag": str(meta.get("MTState", "")), "series_desc": ";".join(sorted(set(r["desc"] for r in series_rows)))}
    mri = Path(fastsurfer_dir) / "mri"
    t1, t1_mask, aseg = nib.load(mri / "orig.mgz"), nib.load(mri / "mask.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    spacing_guess = [float(z) for z in nm_img.header.get_zooms()[:3]]
    tx_mni_t1, m_mni = register_t1_to_mni(t1, t1_mask, cache_path=mni_cache_path(fastsurfer_dir))
    # atlas SN centroid in T1 (LPS) space: the protocol centres the slab on the midbrain, so start the slab there
    t1_sitk = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(t1.dataobj).astype(np.float32), t1.affine))
    pauli_t1 = labels_to_dwi(pauli_atlas(), t1_sitk, [tx_mni_t1])
    sn_idx = np.argwhere(np.isin(pauli_t1, [7, 9]))
    target = None
    if len(sn_idx):
        zc, yc, xc = sn_idx.mean(axis=0)
        target = t1_sitk.TransformContinuousIndexToPhysicalPoint((float(xc), float(yc), float(zc)))
    nm = np.asanyarray(nm_img.dataobj).astype(np.float32)
    tgt = _sitk_native(nm, nm_img.affine)
    # header initialisation first (right for most sessions); if the atlas SN does not land on the slab, retry from
    # the SN centroid and keep whichever result covers the SN better (ties: better metric)
    best = None
    for init_target, name in ((None, "header"), (target, "sn_centroid")):
        tx_try, m_try = register_nm_to_t1(nm_img, t1, target_center=init_target)
        cov = int(np.isin(labels_to_dwi(pauli_atlas(), tgt, [tx_mni_t1, tx_try]), [7, 9]).sum())
        if best is None or cov > best[2] * 1.2 or (abs(cov - best[2]) <= best[2] * 0.2 and m_try < best[1]):
            best = (tx_try, m_try, cov, name)
        if cov * np.prod(spacing_guess) >= 0.5 * max(len(sn_idx), 1):
            break
    tx_t1_nm, m_rigid, _, init_used = best
    row.update({"reg_nm_t1_mi": m_rigid, "reg_t1_mni_mi": m_mni, "reg_init": init_used})
    fs_lab = labels_to_dwi(aseg, tgt, [tx_t1_nm])
    pauli_lab = labels_to_dwi(pauli_atlas(), tgt, [tx_mni_t1, tx_t1_nm])
    spacing = [float(z) for z in nm_img.header.get_zooms()[:3]]
    rois = nm_rois(fs_lab, pauli_lab, spacing[::-1])
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in pauli_lab.shape], indexing="ij")
    origin, sp, direction = np.array(tgt.GetOrigin()), np.array(tgt.GetSpacing()), np.array(tgt.GetDirection()).reshape(3, 3)
    phys_y = origin[1] + direction[1, 0] * xx * sp[0] + direction[1, 1] * yy * sp[1] + direction[1, 2] * zz * sp[2]
    nm_zyx = np.transpose(nm, (2, 1, 0))
    # slab coverage: fraction of the atlas SN (in T1 space, 1 mm^3 voxels) that falls inside the NM slab
    row["sn_slab_coverage"] = float((rois["sn_l"].sum() + rois["sn_r"].sum()) * np.prod(spacing) / max(len(sn_idx) * 1.0, 1.0))
    row.update(features(nm_zyx, rois, phys_y, spacing_zyx=tuple(spacing[::-1])))
    if keep_nifti:
        nib.save(nm_img, work / "nm_mean.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(pauli_lab, (2, 1, 0)).astype(np.int16), nm_img.affine), work / "pauli_nm.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(fs_lab, (2, 1, 0)).astype(np.int16), nm_img.affine), work / "aseg_nm.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(rois["ref"], (2, 1, 0)).astype(np.int16), nm_img.affine), work / "ref_nm.nii.gz")
    shutil.rmtree(work / "nii", ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep = args
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep)
        out["error"] = out.pop("nm_error", "")
    except Exception as e:
        out = {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv=None):
    import argparse

    from .batch import add_common_args, done_subjects, fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows

    a = add_common_args(argparse.ArgumentParser()).parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    idx = load_index(work / "nm_index.csv", a.zips, flag_nm)
    idx = idx[idx["selected"]]
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    out_csv = work / "nm_features.csv"
    done = done_subjects(out_csv, a.retry_errors)
    jobs = [(int(patno), session_rows(g), fs[int(patno)], str(work), a.keep_nifti)
            for patno, g in idx.groupby("patno") if patno not in done and int(patno) in fs]
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, pid_file=a.pid_file)


if __name__ == "__main__":
    main()
