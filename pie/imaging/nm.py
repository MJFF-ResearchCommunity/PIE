"""
Neuromelanin-sensitive MRI (PPMI-2 "2D GRE-MT" family) -> nigral contrast ratio and neuromelanin volume.

PPMI-2 acquires a 2D T1-weighted gradient echo with a magnetization-transfer pulse (0.5 x 0.5 x 1.5 mm, 16 slices
through the midbrain, TR ~0.45-0.65 s, TE ~5 ms, FA 40), typically five repeats to be averaged; descriptions vary
by site ("AX T2 GRE MT", "2D GRE-MT", "AXIAL 2D GRE-MT", "2D GRE-MT_ACPC", "NM-GRE", "NM-MT", ...). Per subject:

1. `convert`      dcm2niix per series; compatible magnitude acquisitions from one reconstruction class are aligned
                  to the first and averaged. Paired NORM/non-NORM reconstructions are not independent repeats.
2. `register`     mean NM slab -> conformed T1 (rigid, mutual information, fitted from two starts);
                  the subject's T1 -> MNI SyN (`nm_template.syn_cache`, fitted when missing) brings the bundled CIT168
                  atlas (Pauli et al. 2018, MNI2009c projection, `pie.imaging.atlases`) SNc/SNr/RN/VTA/STN onto the
                  NM grid in one label resampling, together with the FastSurfer brainstem / ventral DC labels.
3. `features`     substantia nigra (SNc + SNr, left/right, anterior/posterior halves): mean signal, contrast ratio
                  CNR = (SN - ref) / ref against the surrounding-midbrain ring (atlas SN dilated 3 mm minus nuclei,
                  inside brainstem/ventral DC). The same-side anterior ring, nominally crus cerebri, is retained
                  as `*_cnr_crus`. This is a contrast ratio, not a noise-SD-normalized CNR, and a diagnostic since
                  11 Oct 2026: PIE's default NM measure is the snceg segmentation (`nm_native --snceg`). The atlas SN
                  is read where the SyN puts it; the optional refinement (`features(refine=True)`: a translation
                  <= 2 mm in-plane, +-1 slice, maximising the ROI mean on a 1 mm-smoothed copy) selects on the image
                  it measures and added ~+0.023 contrast, so it is off by default (`nm_sn_shift_mm_*` = 0).
                  Order statistics on single voxels (brightest-fraction contrast, fixed-threshold volumes) were removed
                  after the 2026-09 audit: on PPMI slabs (per-voxel CV ~0.13) they tracked the noise level
                  (Spearman 0.65-0.71) and carried no group signal. QC: repeats, inter-repeat motion, registration
                  metric, per-side slab coverage of the SN, reference-ring homogeneity, and the study's voxel checks
                  (signal and every-repeat support on the measured masks). The slab (~24 mm) does not reach the
                  locus coeruleus.

    venv_imaging/bin/python -m pie.imaging.nm --zips <MRI zips> --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4 [--keep-nifti]
    venv_imaging/bin/python -m pie.imaging.nm ... --work-dir Imaging/derived/nm --refeature   # features again from saved slabs
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

from .archives import ZipArchiveCache
from .convert import DCM2NIIX
from .datscan import ITK_SEED
from .dwi import PAULI, _register_volume, _sitk_from_nib, _sitk_native, labels_to_dwi, pauli_atlas

NM_PATTERN = r"GRE.?MT|MT.?GRE|GRE ?- ?MT|NM\s*-|Neuromelanin|NM_MT|NM MT"
EXCLUDE = r"MTC-NO|B0|Map|TRACEW|ADC|_FA"
DILATE_MM = 3.0
REFINE_MM = 2.0        # max in-plane translation when refining the atlas SN position on the slab
PROCESSING_VERSION = "2026-10-11-coverage-weighted-unrefined-v7"   # complete runs; --refeature appends "-refeatured"


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


def compatible_repeats(niis, minimum=1, limit=None):
    """Select one magnitude reconstruction/protocol; reject ambiguous repeat identity.

    Siemens NORM/non-NORM images can reconstruct the *same* acquisition. Neither
    different pixel values nor different series IDs establish independence. Keep
    the largest compatible group (prefer NORM on a tie), in stable path order.
    Missing acquisition timestamps cannot certify independence; provenance flags it.
    """
    groups, excluded = {}, []
    for path in sorted(map(str, niis)):
        meta_path = Path(path[:-7] + '.json')
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        tags = meta.get('ImageType', [])
        tags = [tags] if isinstance(tags, str) else tags
        tags = {str(t).upper() for t in tags}
        if tags & {'P', 'PHASE', 'R', 'REAL', 'I', 'IMAGINARY'}:
            excluded.append({'path': path, 'reason': 'non-magnitude'}); continue
        img = nib.load(path)
        if img.ndim not in (3, 4):
            excluded.append({'path': path, 'reason': 'not 3D/4D'}); continue
        data = np.asarray(img.dataobj, dtype=np.float32)
        reconstruction = tuple(sorted(tags - {'ORIGINAL', 'PRIMARY', 'M', 'MAGNITUDE', 'OTHER'}))
        key = (img.shape[:3], tuple(np.round(img.header.get_zooms()[:3], 3)),
               meta.get('EchoTime'), meta.get('RepetitionTime'), meta.get('FlipAngle'), meta.get('MTState'), reconstruction)
        group = groups.setdefault(key, [])
        for k in range(data.shape[-1] if img.ndim == 4 else 1):
            stamp = meta.get('AcquisitionDateTime') or meta.get('AcquisitionTime')
            identity = (stamp, meta.get('AcquisitionNumber'), k) if stamp else None
            if identity and any(v['acquisition_identity'] == identity for v in group):
                # Identical times may be anonymized, not true duplicates. Fail visibly
                # instead of silently collapsing a five-repeat session to one image.
                raise ValueError('Ambiguous NM acquisition identity: repeated timestamp/number within one reconstruction; verify original acquisition provenance')
            arr = data[..., k] if img.ndim == 4 else data
            group.append({'image': nib.Nifti1Image(arr, img.affine), 'path': path, 'volume': k,
                          'meta': meta, 'acquisition_identity': identity})
    if not groups:
        raise ValueError('No magnitude NM repeats')
    key = max(groups, key=lambda k: (len(groups[k]), 'NORM' in k[-1]))
    selected = groups[key][:limit]
    if len(selected) < minimum:
        raise ValueError(f'Only {len(selected)} compatible magnitude repeats; need {minimum}')
    for i, v in enumerate(selected):
        if any(np.array_equal(np.asarray(v['image'].dataobj), np.asarray(u['image'].dataobj)) for u in selected[:i]):
            raise ValueError('Exact duplicate NM arrays would inflate repeatability')
    info = {'n_groups': len(groups), 'group_sizes': [len(g) for g in groups.values()],
            'selected_reconstruction': key[-1], 'excluded': excluded,
            'n_missing_acquisition_identity': sum(v['acquisition_identity'] is None for v in selected)}
    return selected, info


def _mppca_centre(stack, frac=0.5):
    """MP-PCA (5-voxel patches) over the central ``frac`` of the two in-plane axes of stacked repeats, the rest left as
    it is. MP-PCA is patch-local: beyond two patch radii of the crop edge (the midbrain is central in PPMI's
    head-centred slabs) values equal whole-slab MP-PCA, about four times faster (whole slab: ~8 CPU-min per subject)."""
    from dipy.denoise.localpca import mppca

    inplane = np.argsort(stack.shape[:3])[1:]
    sl = [slice(None)] * 3
    for ax in inplane:
        n = stack.shape[ax]
        sl[ax] = slice((n - int(n * frac)) // 2, (n - int(n * frac)) // 2 + int(n * frac))
    out = stack.copy()
    out[tuple(sl)] = mppca(stack[tuple(sl)], patch_radius=2, suppress_warning=True)
    return out


def average_repeats(niis, sampling_seed=ITK_SEED, transforms_dir=None, denoise=False):
    """Align/average one compatible reconstruction per acquisition; return image, metadata, count, motion and the
    repeat support: per voxel of the image, the share of repeats whose field of view reaches it (1 = every repeat).
    Each voxel is the mean of the repeats that reach it (coverage-weighted, since 11 Oct 2026).
    ``denoise``: MP-PCA (Veraart et al. 2016, DIPY ``mppca``, 5-voxel patches) over the stacked repeats before
    alignment, as in the published PPMI nigral-volume pipeline (Langley et al. 2025), on the slab's central in-plane
    half (``_mppca_centre``). On a phantom it cut the error of the average about threefold in uniform tissue but only
    by ~20 % around a bright nigra-like block, whose contrast it kept within 10 %."""
    import SimpleITK as sitk

    selected, info = compatible_repeats(niis)
    group = [(r['image'], r['path']) for r in selected]
    # a clean magnitude slab reaches its maximum in a few voxels; a scanner-clipped one (PPMI: 12-bit ceiling 4095) in
    # a large fraction of the brain, which destroys the nigral contrast
    clipped = max(float((a == a.max()).mean()) if a.max() > 0 else 0.0 for a in (np.asanyarray(img.dataobj) for img, _ in group))
    if denoise and len(group) >= 3:
        stack = _mppca_centre(np.stack([np.asanyarray(img.dataobj).astype(np.float32) for img, _ in group], axis=-1))
        group = [(nib.Nifti1Image(stack[..., i], img.affine), path) for i, (img, path) in enumerate(group)]
    ref_img = group[0][0]
    ref = _sitk_native(np.asanyarray(ref_img.dataobj).astype(np.float32), ref_img.affine)
    acc = sitk.GetArrayFromImage(ref).astype(np.float64)
    support = np.ones_like(acc)
    motion = [0.0]
    for index, (img, _) in enumerate(group[1:], 1):
        mov = _sitk_native(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
        tx = _register_volume(ref, mov, sampling=0.2, iterations=60, sampling_seed=sampling_seed)
        acc += sitk.GetArrayFromImage(sitk.Resample(mov, ref, tx, sitk.sitkLinear, 0.0))
        if transforms_dir is not None:
            Path(transforms_dir).mkdir(parents=True, exist_ok=True)
            sitk.WriteTransform(tx, str(Path(transforms_dir) / f'repeat_{index}_to_anchor.tfm'))
        ones = sitk.Image(mov.GetSize(), sitk.sitkFloat32) + 1
        ones.CopyInformation(mov)
        support += sitk.GetArrayFromImage(sitk.Resample(ones, ref, tx, sitk.sitkLinear, 0.0))
        motion.append(float(np.linalg.norm(np.array(tx.GetParameters())[3:6])))
    # coverage-weighted: each voxel is averaged over the repeats that reach it (the anchor reaches all), so a moved
    # repeat's missing edge does not darken it (dividing by the number of repeats did, by 1/n per missing repeat)
    mean = (acc / support).astype(np.float32)
    support = (support / len(group)).transpose(2, 1, 0).astype(np.float32)
    if transforms_dir is not None:
        Path(transforms_dir).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(support, ref_img.affine), Path(transforms_dir) / 'repeat_support_fraction.nii.gz')
    meta = dict(selected[0]['meta'])
    meta['PIERepeatSelection'] = dict(info, volumes=[{k: v for k, v in r.items() if k != 'image'} for r in selected])
    meta['PIEDenoise'] = 'mppca' if denoise and len(group) >= 3 else 'none'
    meta['PIEClippedFraction'] = clipped
    return nib.Nifti1Image(np.transpose(mean, (2, 1, 0)), ref_img.affine), meta, len(group), float(max(motion)), support


# ------------------------------------------------------------------------------------------ registration / ROIs
def register_nm_to_t1(nm_img, t1_img, target_center=None, sampling_seed=ITK_SEED):
    """Rigid MI registration between the NM slab (fixed: every metric sample lies inside the 24 mm slab) and the
    *unmasked* conformed T1 (moving). The full head matters: with a brain-masked T1 a thin slab of brain tissue
    matches several heights equally well and the optimizer settled on the striatum for some subjects; the eyes,
    sinuses and skull in the slab pin its height. ``target_center`` (LPS mm) initialises the slab centre there
    (atlas SN centroid in T1 space); otherwise the same-session headers are trusted.
    Returns (transform fixed(T1)->moving(NM), metric), the convention used by ``labels_to_dwi``.
    ``sampling_seed=0`` is ITK's wall-clock seed (not reproducible); the default is the fixed ``ITK_SEED``."""
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
    reg.SetMetricSamplingPercentage(0.3, seed=int(sampling_seed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(slab, t1)          # tx: slab (fixed) point -> T1 (moving) point
    metric = float(reg.GetMetricValue())
    return tx.GetInverse(), metric     # T1 point -> slab point


def nm_rois(fs_lab, pauli_lab, spacing_mm, left_mask=None):
    """Masks on the NM grid: SN (SNc+SNr) left/right, nuclei union, search region (SN dilated DILATE_MM) and the
    reference ring (SN dilated DILATE_MM minus dilated nuclei, inside brainstem / ventral DC / peduncle WM). The per-side
    search region (which bounds the same-side anterior reference) is the dilated SN restricted to brainstem / ventral DC
    labels minus the other nuclei."""
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
    if left_mask is None:
        if not (fs_lab == 12).any() or not (fs_lab == 51).any():
            raise ValueError("NM slab has no bilateral putamen landmarks; provide the transformed atlas hemisphere mask")
        xl, xr = np.nonzero(fs_lab == 12)[2].mean(), np.nonzero(fs_lab == 51)[2].mean()
        xi = np.arange(pauli_lab.shape[2])[None, None, :]
        left = (xi > (xl + xr) / 2) if xl > xr else (xi < (xl + xr) / 2)
    else:
        left = np.asarray(left_mask, dtype=bool)
        if left.shape != pauli_lab.shape:
            raise ValueError("hemisphere mask must match the NM grid")
    return {"sn_l": sn & left, "sn_r": sn & ~left, "search_l": search & left, "search_r": search & ~left, "ref": ref, "stem": np.isin(fs_lab, [16, 28, 60])}


def _shift_mask(mask, shift):
    """Translate a mask with zero padding; anatomical labels must never wrap across image borders."""
    return ndimage.shift(mask.astype(np.uint8), shift, order=0, mode="constant", cval=0, prefilter=False).astype(bool)


def _refine(sn, sm, stem, spacing_zyx, max_mm=REFINE_MM):
    """Translation (voxels, z/y/x) of the atlas SN that maximises its mean on the smoothed slab, within ``max_mm``
    along each in-plane axis and one slice, keeping >= 95 % of the ROI inside brainstem labels.
    Euclidean displacement can exceed max_mm. Selecting and measuring on the
    same image can bias contrast; assess localization on independent repeats."""
    ny, nx = [int(round(max_mm / s)) for s in spacing_zyx[1:]]
    zz, yy, xx = np.nonzero(sn)
    box = tuple(slice(max(0, a.min() - p), a.max() + p + 1) for a, p in zip((zz, yy, xx), (2, ny + 1, nx + 1)))
    sn_c, sm_c, stem_c = sn[box], sm[box], stem[box]
    best, shift = -np.inf, (0, 0, 0)
    for dz in (-1, 0, 1):
        for dy in range(-ny, ny + 1):
            for dx in range(-nx, nx + 1):
                m = _shift_mask(sn_c, (dz, dy, dx))
                if m.sum() != sn_c.sum() or (m & stem_c).sum() < 0.95 * m.sum():
                    continue
                v = sm_c[m].mean()
                if v > best:
                    best, shift = v, (dz, dy, dx)
    return shift


def features(nm, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), smooth_fwhm_mm=1.0, mask_out=None, support=None, refine=False):
    """Signal and CNR per side (+ anterior/posterior halves of the SN) at the atlas-SN position, or with ``refine`` at
    the position ``_refine`` picks. Refinement is off by default since 11 Oct 2026: it selects the position on the image
    it then measures, which added ~+0.023 to the contrast on PPMI slabs, more than the HC-PD gap; unrefined,
    ``*_cnr`` equals ``*_cnr_atlas`` and ``nm_sn_shift_mm_*`` is 0. This ring contrast is a diagnostic: PIE's default
    NM measure is snceg (``nm_native.snceg_features``).
    Reference = the whole surrounding-midbrain ring (``nm_ring_mean``); the part of the ring anterior to the
    SN on the same side (nominally crus cerebri) is kept as ``*_cnr_crus``. The audit of 2026-09 showed the anterior
    part is contaminated by the neuromelanin band itself (it lies 1-2 mm off the atlas SN and is lost in PD: the
    anterior/ring intensity ratio is lower in PD), which cancelled the group difference. The ring is an alternative
    reference, not an established unbiased biological standard. ``smooth_fwhm_mm`` (in-plane) only serves the position refinement; contrasts are read from the
    unsmoothed slab. On the measured masks (refined SN per side, ring) it reports the study's voxel checks: the share
    with signal (``nm_valid_*``: finite, > 0) and, given ``average_repeats``' support (z, y, x), the share that every
    repeat reaches (``nm_support_*``: support >= 0.99); ``QC["nm"]`` wants >= 0.95 of each."""
    out = {}
    sigma = [0.0] + [smooth_fwhm_mm / 2.3548 / s for s in spacing_zyx[1:]]
    sm = ndimage.gaussian_filter(nm, sigma=sigma) if smooth_fwhm_mm else nm
    # Localize both sides before defining their shared reference. Otherwise a
    # translated SN can enter the original ring and normalize against itself.
    stem = rois.get("stem", np.ones_like(rois["ref"]))
    shifts, refined = {}, {}
    for side in ("l", "r"):
        original = rois[f"sn_{side}"]
        shifts[side] = _refine(original, sm, stem, spacing_zyx) if refine and original.any() else (0, 0, 0)
        refined[side] = _shift_mask(original, shifts[side])
    reference = rois["ref"] & ~(refined['l'] | refined['r'])
    if mask_out is not None:
        mask_out['ref'] = reference
    ring_vals = nm[reference]
    ring_vals = ring_vals[np.isfinite(ring_vals) & (ring_vals > 0)]
    if len(ring_vals) < 20:
        return {"nm_error": "reference region empty"}
    out.update({"nm_ring_mean": float(ring_vals.mean()), "nm_ring_sd": float(ring_vals.std()), "n_ring": int(len(ring_vals))})
    for name, m in (("l", refined["l"]), ("r", refined["r"]), ("ref", reference)):
        out[f"nm_valid_{name}"] = float((np.isfinite(nm[m]) & (nm[m] > 0)).mean()) if m.any() else np.nan
        if support is not None:
            out[f"nm_support_{name}"] = float((support[m] >= 0.99).mean()) if m.any() else np.nan
    for side in ("l", "r"):
        sn0, search = rois[f"sn_{side}"], rois[f"search_{side}"]
        out[f"n_sn_{side}"] = int(sn0.sum())
        if not sn0.any():
            continue
        y_sn = np.median(phys_y[sn0])
        crus = reference & search & (phys_y < y_sn - 1.5)          # anterior (LPS: smaller y) to the SN, same side
        cv = nm[crus]
        cv = cv[cv > 0]
        if len(cv) < 20:
            crus = reference & search
            cv = nm[crus]
            cv = cv[cv > 0]
        ref_mean, ref_sd = float(cv.mean()), float(cv.std())
        out[f"nm_ref_{side}_mean"], out[f"nm_ref_{side}_sd"], out[f"n_ref_{side}"] = ref_mean, ref_sd, int(len(cv))
        shift = shifts[side]
        sn = refined[side]
        if mask_out is not None:
            mask_out[f"sn_{side}"] = sn
        out[f"nm_sn_shift_mm_{side}"] = float(np.sqrt(sum((d * s) ** 2 for d, s in zip(shift, spacing_zyx))))
        cnr = lambda m, ref: float((nm[m][nm[m] > 0].mean() - ref) / ref) if (nm[m] > 0).sum() >= 3 else np.nan
        out[f"nm_sn_{side}_mean"] = float(nm[sn][nm[sn] > 0].mean()) if (nm[sn] > 0).sum() >= 3 else np.nan
        ring_mean = out["nm_ring_mean"]
        out[f"nm_sn_{side}_cnr"] = cnr(sn, ring_mean)
        out[f"nm_sn_{side}_cnr_crus"] = cnr(sn, ref_mean)
        out[f"nm_sn_{side}_cnr_atlas"] = cnr(sn0, ring_mean)
        cut = np.median(phys_y[sn])
        for name, m in (("posterior", sn & (phys_y > cut)), ("anterior", sn & (phys_y <= cut))):   # LPS: larger y = posterior
            out[f"nm_sn_{name}_{side}_cnr"] = cnr(m, ring_mean)
    for base in ("nm_sn", "nm_sn_posterior", "nm_sn_anterior"):
        l, r = out.get(f"{base}_l_cnr", np.nan), out.get(f"{base}_r_cnr", np.nan)
        out[f"{base}_mean_cnr"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    l, r = out.get("nm_sn_l_cnr", np.nan), out.get("nm_sn_r_cnr", np.nan)
    out["nm_sn_min_cnr"] = float(np.nanmin([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    la, ra = out.get("nm_sn_l_cnr_atlas", np.nan), out.get("nm_sn_r_cnr_atlas", np.nan)
    out["nm_sn_mean_cnr_atlas"] = float(np.nanmean([la, ra])) if not (np.isnan(la) and np.isnan(ra)) else np.nan
    out["nm_sn_asym_cnr"] = float(abs(l - r)) if np.isfinite(l) and np.isfinite(r) else np.nan
    return out


def _phys_y(tgt, shape_zyx):
    """LPS y coordinate (posterior positive) of every voxel of a SimpleITK image, as a (z, y, x) array."""
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape_zyx], indexing="ij")
    origin, sp, direction = np.array(tgt.GetOrigin()), np.array(tgt.GetSpacing()), np.array(tgt.GetDirection()).reshape(3, 3)
    return origin[1] + direction[1, 0] * xx * sp[0] + direction[1, 1] * yy * sp[1] + direction[1, 2] * zz * sp[2]


def _mni_codes(sn_only=False):
    """CIT168 labels + 100 x anatomical side (1 left: MNI x < 0, 2 right) on every atlas voxel, so one pull carries the
    nuclei and the hemispheres (the slab's voxel x direction is not an anatomical label); ``sn_only``: just the side
    of the SN (SNc + SNr) voxels, 0 elsewhere."""
    atlas = pauli_atlas()
    lab = np.asanyarray(atlas.dataobj).astype(np.int16)
    ix = np.indices(lab.shape, sparse=True)
    side = np.where(atlas.affine[0, 3] + sum(atlas.affine[0, k] * ix[k] for k in range(3)) < 0, 1, 2)
    codes = np.isin(lab, [7, 9]) * side if sn_only else lab + 100 * side
    return nib.Nifti1Image(np.broadcast_to(codes, lab.shape).astype(np.int16), atlas.affine)


def _atlas_on(grid_img, tx_t1_grid, fastsurfer_dir, sn_only=False, pad=12):
    """``_mni_codes`` on ``grid_img``'s voxels, (x, y, z), through the rigid T1 -> grid transform and the subject's T1
    -> MNI SyN (``nm_template.syn_cache``, fitted when missing), in one label resampling (``nm_native.pull_labels``):
    the study's nonlinear mapping in place of an affine."""
    from .nm_native import pull_labels
    from .nm_template import syn_cache

    warp, affine = syn_cache(fastsurfer_dir)["fwd"]
    return pull_labels(_mni_codes(sn_only), grid_img, tx_t1_grid, warp, affine, pad=pad)[0]


def _atlas_on_slab(nm_img, tx_t1_nm, fastsurfer_dir):
    """CIT168 labels and the anatomical left hemisphere on the slab grid, both (z, y, x), through the SyN."""
    codes = np.transpose(_atlas_on(nm_img, tx_t1_nm, fastsurfer_dir), (2, 1, 0))
    return codes % 100, codes // 100 == 1


def _atlas_provenance():
    """CIT168 provenance with the SyN's registration reference (TemplateFlow's 1 mm MNI152NLin2009cAsym T1w), so a map
    made through the older affine (2 mm template) is never reused by ``refeature_subject``."""
    from .atlases import TEMPLATEFLOW_FILES, cit168_provenance

    return {**cit168_provenance(), "registration_reference_sha256": TEMPLATEFLOW_FILES["tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"]}


def _sn_points_in_t1(t1_img, fastsurfer_dir):
    """LPS mm of the T1 voxels the atlas SN (SNc + SNr) maps onto through the SyN, per anatomical side {1: left, 2: right}."""
    import SimpleITK as sitk

    on_t1 = _atlas_on(t1_img, sitk.Euler3DTransform(), fastsurfer_dir, sn_only=True, pad=0)
    return {s: nib.affines.apply_affine(t1_img.affine, np.argwhere(on_t1 == s)) * [-1, -1, 1] for s in (1, 2)}


def slab_coverage(points, tx_t1_nm, grid):
    """Share of ``points`` (T1 space, LPS mm) that ``tx_t1_nm`` (T1 point -> slab point) maps inside the slab ``grid``
    (voxel centres +- half a voxel): the study's bounded geometric coverage, taken per side of the atlas SN."""
    if not len(points):
        return 0.0
    idx = np.array([grid.TransformPhysicalPointToContinuousIndex(tx_t1_nm.TransformPoint(tuple(map(float, p)))) for p in points])
    return float(np.all((idx >= -0.5) & (idx < np.array(grid.GetSize()) - 0.5), axis=1).mean())


def refeature_subject(work_dir, patno, fastsurfer_dir=None, atlas_sha256=None, registration_reference_sha256=None):
    """Feature columns of a finished subject again, from the saved slab and label maps (``--keep-nifti`` outputs),
    without registering: the way to apply a changed ``features`` to a whole run.

    The saved atlas map is reused only when its atlas AND registration-reference hashes match (``_atlas_provenance``).
    Otherwise (legacy rows mapped the native-space CIT168 file; rows before 27 Sep 2026 went through an affine) the
    atlas and hemisphere maps are regenerated from the saved slab -> T1 transform and the cached SyN, rewritten, and
    ``sn_slab_coverage_l/_r`` plus the atlas provenance are returned too; without those transforms the subject is
    refused rather than measured on a stale atlas.
    A saved repeat support (``repeat_transforms/repeat_support_fraction.nii.gz``) feeds the voxel checks of ``features``."""
    import SimpleITK as sitk

    from .nm_template import syn_cached

    d = Path(work_dir) / str(patno)
    img = nib.load(d / "nm_mean.nii.gz")
    nm = np.asanyarray(img.dataobj).astype(np.float32)
    fs = np.transpose(np.asanyarray(nib.load(d / "aseg_nm.nii.gz").dataobj), (2, 1, 0))
    spacing = [float(z) for z in img.header.get_zooms()[:3]]
    tgt = _sitk_native(nm, img.affine)
    slab = d / "slab_to_t1.tfm"
    mapped = fastsurfer_dir is not None and syn_cached(fastsurfer_dir) and slab.exists()
    provenance, out = _atlas_provenance(), {}
    hemi_path = d / "left_nm.nii.gz"
    if (atlas_sha256 == provenance["atlas_sha256"] and hemi_path.exists()
            and registration_reference_sha256 == provenance["registration_reference_sha256"]):
        # the saved hemisphere map permits refeaturing without new registration or guessing voxel handedness
        pauli, left = (np.transpose(np.asanyarray(nib.load(d / f"{n}.nii.gz").dataobj), (2, 1, 0)) for n in ("pauli_nm", "left_nm"))
        left = left.astype(bool)
    elif not mapped:
        raise ValueError("saved atlas/reference mapping is unverified and the slab transform or SyN cache "
                         "(nm_template syn) is missing; re-run nm for this subject")
    else:
        tx = sitk.ReadTransform(str(slab))
        pauli, left = _atlas_on_slab(img, tx, fastsurfer_dir)
        for name, arr, dtype in (("pauli_nm", pauli, np.int16), ("left_nm", left, np.uint8)):
            nib.save(nib.Nifti1Image(np.transpose(arr, (2, 1, 0)).astype(dtype), img.affine), d / f"{name}.nii.gz")
        sides = _sn_points_in_t1(nib.load(Path(fastsurfer_dir) / "mri" / "orig.mgz"), fastsurfer_dir)
        out.update({f"sn_slab_coverage_{s}": slab_coverage(sides[k], tx, tgt) for k, s in ((1, "l"), (2, "r"))})
        out.update(provenance)
    rois = nm_rois(fs, pauli, spacing[::-1], left_mask=left)
    support_path = d / "repeat_transforms" / "repeat_support_fraction.nii.gz"
    support = np.transpose(np.asanyarray(nib.load(support_path).dataobj), (2, 1, 0)) if support_path.exists() else None
    out.update(features(np.transpose(nm, (2, 1, 0)), rois, _phys_y(tgt, pauli.shape), spacing_zyx=tuple(spacing[::-1]),
                        support=support))
    return out


def _refeature_job(args):
    work_dir, row, *fs_dir = args
    # stale feature columns go; nm_clipped_fraction is measured on the raw repeats, which refeature never reads
    keep = {k: v for k, v in row.items() if not k.startswith(("nm_", "n_sn_", "n_ref_", "n_ring")) or k == "nm_clipped_fraction"}
    if row.get("error") or not (Path(work_dir) / str(row["patno"]) / "nm_mean.nii.gz").exists():
        return keep
    try:
        out = refeature_subject(work_dir, row["patno"], fs_dir[0] if fs_dir else None, atlas_sha256=row.get("atlas_sha256"),
                                registration_reference_sha256=row.get("registration_reference_sha256"))
        keep["processing_version"] = PROCESSING_VERSION + "-refeatured"   # current features, original registration
        return {**keep, **out, "error": out.pop("nm_error", "")} if "nm_error" not in out else {**keep, "error": out["nm_error"]}
    except Exception as e:
        return {**keep, "error": f"refeature: {type(e).__name__}: {str(e)[:150]}"}


def register_slab(nm_img, fastsurfer_dir, sampling_seed=ITK_SEED, min_cov=0.9):
    """Rigid slab -> T1 registration from both initialisations: the same-session headers, and the slab centred on the
    atlas SN centroid (the protocol centres the slab on the midbrain; the SN reaches the T1 through the subject's SyN).
    Of the fits whose slab holds >= ``min_cov`` of each side's atlas SN (``slab_coverage``), the lowest MI metric wins;
    if neither does, the better-covered one (the study's rule). Returns (T1 -> slab transform, metric, init used,
    (left, right) coverage)."""
    t1 = nib.load(Path(fastsurfer_dir) / "mri" / "orig.mgz")
    sides = _sn_points_in_t1(t1, fastsurfer_dir)
    sn = np.vstack([sides[1], sides[2]])
    tgt = _sitk_native(np.asanyarray(nm_img.dataobj).astype(np.float32), nm_img.affine)
    fits = []
    for init_target, name in ((None, "header"), (sn.mean(axis=0) if len(sn) else None, "sn_centroid")):
        if name == "sn_centroid" and init_target is None:
            continue
        try:
            tx, m = register_nm_to_t1(nm_img, t1, target_center=init_target, sampling_seed=sampling_seed)
        except RuntimeError:
            continue
        if np.isfinite(m):
            fits.append((tx, m, name, tuple(slab_coverage(sides[s], tx, tgt) for s in (1, 2))))
    if not fits:
        raise RuntimeError("both NM-to-T1 registration initialisations failed")
    covering = [f for f in fits if min(f[3]) >= min_cov]
    tx, m, name, cov = min(covering, key=lambda f: f[1]) if covering else min(fits, key=lambda f: (-min(f[3]), f[1]))
    return tx, m, name, cov


# ------------------------------------------------------------------------------------------ driver
def process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, sampling_seed=ITK_SEED, keep_raw=False, denoise=False):
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    work = Path(work_dir) / str(patno)
    work.mkdir(parents=True, exist_ok=True)
    niis = []
    with ZipArchiveCache() as zips:  # LONI zip rows are converted; a row with "nifti" (e.g. the Emory Dryad release) is read as is
        for r in series_rows:
            niis += [r["nifti"]] if r.get("nifti") else convert(r["zip"], r["prefix"], work / "nii", archive_cache=zips)
    nm_img, meta, n_rep, motion, support = average_repeats(niis, sampling_seed=sampling_seed, denoise=denoise,
                                                transforms_dir=work / 'repeat_transforms' if keep_nifti else None)
    (work / 'repeat_selection.json').write_text(json.dumps(meta['PIERepeatSelection'], indent=2))
    row = {"patno": patno, "n_series": len(series_rows), "n_repeats": n_rep, "repeat_motion_mm_max": motion, "denoise": meta["PIEDenoise"], "nm_clipped_fraction": meta["PIEClippedFraction"],
           "shape": "x".join(map(str, nm_img.shape)), "voxel_mm": "x".join(str(round(float(z), 2)) for z in nm_img.header.get_zooms()[:3]),
           "manufacturer": str(meta.get("Manufacturer", "")), **{k: str(meta.get(k, "")) for k in ("DeviceSerialNumber", "StationName", "SoftwareVersions")}, "model": str(meta.get("ManufacturersModelName", meta.get("ManufacturerModelName", ""))),
           "tr_s": meta.get("RepetitionTime", np.nan), "te_s": meta.get("EchoTime", np.nan), "flip_angle": meta.get("FlipAngle", np.nan),
           "mt_flag": str(meta.get("MTState", "")), "series_desc": ";".join(sorted(set(r["desc"] for r in series_rows)))}
    tx_t1_nm, m_rigid, init_used, cov = register_slab(nm_img, fastsurfer_dir, sampling_seed=sampling_seed)
    mri = Path(fastsurfer_dir) / "mri"
    aseg = nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    nm = np.asanyarray(nm_img.dataobj).astype(np.float32)
    tgt = _sitk_native(nm, nm_img.affine)
    row.update({"reg_nm_t1_mi": m_rigid, "reg_init": init_used})
    fs_lab = labels_to_dwi(aseg, tgt, [tx_t1_nm])
    pauli_lab, left = _atlas_on_slab(nm_img, tx_t1_nm, fastsurfer_dir)
    spacing = [float(z) for z in nm_img.header.get_zooms()[:3]]
    rois = nm_rois(fs_lab, pauli_lab, spacing[::-1], left_mask=left)
    row.update({"fs_image_id": Path(fastsurfer_dir).name, "acquisition_date": series_rows[0]["date"],
                "processing_version": PROCESSING_VERSION})
    row.update(_atlas_provenance())
    phys_y = _phys_y(tgt, pauli_lab.shape)
    nm_zyx = np.transpose(nm, (2, 1, 0))
    row["sn_slab_coverage_l"], row["sn_slab_coverage_r"] = cov
    refined = {}
    row.update(features(nm_zyx, rois, phys_y, spacing_zyx=tuple(spacing[::-1]), mask_out=refined,
                        support=np.transpose(support, (2, 1, 0))))
    if keep_nifti:
        sitk.WriteTransform(tx_t1_nm, str(work / "slab_to_t1.tfm"))     # T1 point -> slab point; reused by nm_template
        nib.save(nm_img, work / "nm_mean.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(pauli_lab, (2, 1, 0)).astype(np.int16), nm_img.affine), work / "pauli_nm.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(fs_lab, (2, 1, 0)).astype(np.int16), nm_img.affine), work / "aseg_nm.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(refined.get('ref', rois["ref"]), (2, 1, 0)).astype(np.int16), nm_img.affine), work / "ref_nm.nii.gz")
        nib.save(nib.Nifti1Image(np.transpose(left, (2, 1, 0)).astype(np.uint8), nm_img.affine), work / "left_nm.nii.gz")
        if refined:
            labels = sum(i * refined.get(f"sn_{s}", np.zeros_like(pauli_lab, bool)) for i, s in ((1, "l"), (2, "r")))
            nib.save(nib.Nifti1Image(np.transpose(labels, (2, 1, 0)).astype(np.uint8), nm_img.affine), work / "sn_refined_nm.nii.gz")
    if not keep_raw:
        shutil.rmtree(work / "nii", ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep, denoise = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"      # the SyN (nm_template._syn_job's setting): workers x 2 threads
    try:
        out = process_subject(patno, rows, fs_dir, work_dir, keep_nifti=keep, denoise=denoise)
        out["error"] = out.pop("nm_error", "")
    except Exception as e:
        out = {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv=None):
    import argparse

    from .batch import add_common_args, fastsurfer_by_patno, filter_jobs, load_index, modality_jobs, run_batch

    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--denoise", action="store_true", help="MP-PCA over the repeats before averaging (Langley et al. 2025)")
    ap.add_argument("--refeature", action="store_true", help="recompute the feature columns of every finished subject from the saved "
                    "slab and label maps (--keep-nifti outputs); the previous table is kept as *.pre_refeature.csv")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if a.refeature:
        from concurrent.futures import ProcessPoolExecutor

        out_csv = work / "nm_features.csv"
        d = pd.read_csv(out_csv)
        d["error"] = d["error"].fillna("")
        only = {int(x) for x in Path(a.patnos).read_text().split()} if a.patnos else None
        fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
        # Legacy rows did not record their T1 ID. Reuse a cached registration only when that lineage is
        # unambiguous; the new earliest-series rule cannot identify which same-date alternative the old code used.
        legacy_sessions = pd.read_csv(a.sessions, dtype={"image_id": str})
        alternatives = legacy_sessions.groupby("patno")["image_id"].nunique()
        fs = {p: directory for p, directory in fs.items() if alternatives.get(p, 0) == 1}
        jobs = [(str(work), r, str(Path(a.fastsurfer_dir) / str(r["fs_image_id"])) if pd.notna(r.get("fs_image_id")) else fs.get(int(r["patno"])))
                for r in d.to_dict("records") if only is None or int(r["patno"]) in only]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_refeature_job, jobs, chunksize=2))
        bak = out_csv.with_suffix(".pre_refeature.csv")
        if not bak.exists():
            out_csv.rename(bak)
        updated = pd.DataFrame(rows)
        untouched = d[~d["patno"].isin(updated["patno"])] if len(updated) else d
        pd.concat([untouched, updated], ignore_index=True).to_csv(out_csv, index=False)
        print(f"refeatured {sum('nm_sn_shift_mm_l' in r for r in rows)}/{len(rows)} subjects ->", out_csv)
        return
    idx = load_index(work / "nm_index.csv", a.zips, flag_nm)
    idx = idx[idx["selected"]]
    out_csv = work / "nm_features.csv"
    jobs = [(p, rows, fsd, str(work), a.keep_nifti, a.denoise) for p, rows, fsd in modality_jobs(idx, a, out_csv)]
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, pid_file=a.pid_file)


# ====================================================================================================
# Neuromelanin volume and reference-normalised signal
# Merged from nm_volume.py; kept together with the rest of the nm measures.
#
# Neuromelanin volume and reference-normalised signal, for comparison with segmentation-based studies.
#
# ``nm.py`` reports contrast ratios on a fixed atlas mask and deliberately does not report threshold volumes:
# single-voxel order statistics are noisy, and bright arteries in the interpeduncular cistern contaminate any
# threshold applied outside the nigra. Many studies nevertheless report a neuromelanin "SN volume", from manual
# segmentation (e.g. Droby et al. 2025, Ben Bashat et al.) or from thresholding against a reference region. These
# functions compute both kinds of number under explicit, auditable rules so that PIE output can be set beside
# theirs:
#
#     mask_volume(mask)                                    volume of a supplied (e.g. manually drawn) SN mask, mm^3
#     hyperintense_volume(nm, search, reference, k)        voxels inside ``search`` brighter than reference mean + k SD
#     normalised_intensity(nm, roi, reference)             mean(roi) / mean(reference), and the contrast ratio minus one
#
# The threshold volume is only as good as ``search``: pass the atlas SN mask (dilated at most by the
# registration uncertainty), never a box, so that cisternal arteries cannot enter it. ``k`` defaults to 3; papers
# use 2 to 4 and the value changes volumes several-fold, so report it with every result. On 40 PPMI 2D GRE-MT scans
# (18 September 2026) the bilateral atlas-SN volume had medians of 160, 66 and 21 mm^3 at k = 2, 2.5 and 3; per side at
# k = 3 it correlated with the contrast ratio at Spearman 0.56, and ``normalised_intensity`` reproduced ``nm.py``'s
# whole-nigra contrast ratio exactly on the same masks.
# ====================================================================================================

def _voxel_mm3(img):
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def _same_grid(*imgs):
    ref = imgs[0]
    for im in imgs[1:]:
        if im.shape[:3] != ref.shape[:3] or not np.allclose(im.affine, ref.affine, atol=1e-4):
            raise ValueError("images are not on one grid")


def mask_volume(mask_img):
    """Volume in mm^3 of every positive voxel of a mask (e.g. a manual SN segmentation)."""
    return float((np.asarray(mask_img.dataobj) > 0).sum() * _voxel_mm3(mask_img))


def hyperintense_volume(nm_img, search_img, reference_img, k=3.0, min_cluster_voxels=1):
    """Volume of voxels inside ``search`` whose signal exceeds reference mean + k * reference SD.

    Connected clusters smaller than ``min_cluster_voxels`` are dropped (26-connectivity), which removes isolated
    noise voxels without changing contiguous nigral signal. Returns the volume, the threshold and the counts.
    """
    _same_grid(nm_img, search_img, reference_img)
    nm = np.asarray(nm_img.dataobj, float)
    search = (np.asarray(search_img.dataobj) > 0) & np.isfinite(nm)
    ref = (np.asarray(reference_img.dataobj) > 0) & np.isfinite(nm) & ~search
    if ref.sum() < 20:
        raise ValueError("reference region has fewer than 20 voxels outside the search region")
    mu, sd = float(nm[ref].mean()), float(nm[ref].std())
    threshold = mu + k * sd
    hot = search & (nm > threshold)
    if min_cluster_voxels > 1 and hot.any():
        lab, n = ndimage.label(hot, structure=np.ones((3, 3, 3)))
        sizes = ndimage.sum(hot, lab, range(1, n + 1))
        hot = np.isin(lab, np.flatnonzero(sizes >= min_cluster_voxels) + 1)
    return {"volume_mm3": float(hot.sum() * _voxel_mm3(nm_img)), "n_voxels": int(hot.sum()), "threshold": threshold,
            "reference_mean": mu, "reference_sd": sd, "k": float(k), "n_search": int(search.sum()),
            "n_reference": int(ref.sum())}


def normalised_intensity(nm_img, roi_img, reference_img):
    """Mean ROI signal over mean reference signal, and the contrast ratio (that ratio minus one)."""
    _same_grid(nm_img, roi_img, reference_img)
    nm = np.asarray(nm_img.dataobj, float)
    roi = (np.asarray(roi_img.dataobj) > 0) & np.isfinite(nm)
    ref = (np.asarray(reference_img.dataobj) > 0) & np.isfinite(nm) & ~roi
    if not roi.any() or not ref.any():
        raise ValueError("empty ROI or reference")
    ratio = float(nm[roi].mean() / nm[ref].mean())
    return {"normalised_intensity": ratio, "contrast_ratio": ratio - 1.0, "n_roi": int(roi.sum()), "n_reference": int(ref.sum())}


if __name__ == "__main__":
    main()
