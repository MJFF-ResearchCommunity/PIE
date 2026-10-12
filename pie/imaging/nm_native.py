"""
nm_native.py — neuromelanin measured on each subject's own averaged slab (native space). Regions are defined once in
MNI152NLin2009cAsym and pulled onto the slab grid in a single nearest-label resampling (slab -> T1 rigid, T1 -> MNI
through the inverse of the cached ANTs SyN), so the slab itself is never interpolated. Two published measures:

    langley  SNc volume: voxels of the nigral search region brighter than the crus-cerebri mean + 2.8 SD, on
             MP-PCA-denoised repeats (``nm --denoise``). Langley et al. 2025, npj Parkinson's Disease 11:181,
             doi:10.1038/s41531-025-00976-3 (PPMI); Hwang, Langley et al. 2023, PLOS ONE 18:e0282684.
    snceg    the neuromelanin-hyperintense SN segmented by the public snceg Attention U-Net (MIT licence, weights
             pinned), its volume and contrast (CR, CNR) against the crus cerebri on the SN's own slices. Lillebostad
             et al. 2025, Imaging Neuroscience, doi:10.1162/IMAG.a.158. PIE's default NM measure (11 Oct 2026).

Regions come from the published Biondetti et al. 2020 atlas (``atlases.biondetti_mni2009c``): the nigral mask (search
region, dilated 1 mm as Langley dilate theirs; territories) and the crus parts of its background ROI (reference, the
cerebral-peduncle ROI of Langley and the crus of Lillebostad). Langley's own SNc atlas and peduncle ROI are not
public; the substitution is recorded in the documentation.

    venv_imaging/bin/python -m pie.imaging.nm_native --sessions Imaging/derived/sessions.csv \\
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir <nm work dir> --workers 4 [--snceg]

Dual-echo TSE input (Lillebostad et al.'s PPMI data): ``--prepare-tse --zips … [--echo 1]`` first, then
``nm_template syn`` and the command above on the same work dir.
"""
import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LANGLEY_K = 2.8                 # Langley et al.: SNc = voxels > reference mean + 2.8 SD
SEARCH_DILATE_MM = 1.0
RING, CRUS, TEGMENTUM = 5, 21, 23   # region codes beside the Biondetti territories 1-3; code = region + 100 * side (1 left, 2 right)
PEDUNCLE = 25                       # langley_labels: the JHU cerebral peduncle, Langley's reference (version 2)
STUDY_REF = 10                      # study_labels: the background ROI's three parts are STUDY_REF + 1..3 (MNI x order)
TERRITORIES = {1: "associative", 2: "limbic", 3: "sensorimotor"}   # atlases.BIONDETTI_TERRITORIES
SNCEG_REVISION = "6bcddc5f33d324c3ebef7303b68ff36d0e82257b"   # huggingface.co/lillepeder/SNceg-0.1, MIT (verified 26 September 2026)
SNCEG_FILES = {"SNceg-0.1.pkl": "fd554eed386296a53a44786cced90af73c621a374b72d647559984e91f0674c3",
               "vars_SNceg-0.1.pkl": "a9ef917ffc75cd0216d377a92382fe2ab8bc4ff6fd8ce7009cad5ff032d27c57"}
SNCEG_DIR = Path.home() / ".cache" / "pie" / "snceg"
SNCEG_PYTHON = os.environ.get("PIE_SNCEG_PYTHON", str(Path(__file__).resolve().parents[2] / "third_party" / "snceg_venv" / "bin" / "python"))


# ------------------------------------------------------------------------------------------ regions
def region_labels(biondetti_img=None):
    """MNI label image: Biondetti territory (1-3), the 1 mm search ring around the nigra (RING), the crus parts of the
    background ROI (CRUS) and its midline tegmental part (TEGMENTUM), each + 100 x side (1 left, 2 right; MNI x < 0 is
    left). Background voxels keep their side code. ``biondetti_img`` defaults to ``atlases.biondetti_mni2009c()``."""
    from scipy import ndimage

    if biondetti_img is None:
        from .atlases import biondetti_mni2009c
        biondetti_img = nib.load(biondetti_mni2009c())
    lab = np.asarray(biondetti_img.dataobj).astype(np.int16)
    x = nib.affines.apply_affine(biondetti_img.affine, np.indices(lab.shape).reshape(3, -1).T)[:, 0].reshape(lab.shape)
    sn = np.isin(lab, [1, 2, 3])
    iters = max(1, int(round(SEARCH_DILATE_MM / float(np.abs(biondetti_img.affine[:3, :3]).max()))))
    region = np.where(sn, lab, 0)
    region[ndimage.binary_dilation(sn, iterations=iters) & ~sn] = RING
    parts, n = ndimage.label(lab == 4, structure=np.ones((3, 3, 3)))
    for i in range(1, n + 1):          # the background's lateral parts lie in the crus, its midline part in the tegmentum
        part = (parts == i) & (region == 0)
        region[part] = CRUS if np.abs(x[parts == i].mean()) > 5 else TEGMENTUM
    return nib.Nifti1Image((region + 100 * np.where(x < 0, 1, 2)).astype(np.int16), biondetti_img.affine)


def langley_labels(biondetti_img=None, peduncle_img=None):
    """Version 2 of the Langley regions: ``region_labels``' search region (Biondetti territories + the 1 mm RING) with
    the JHU cerebral peduncle (``atlases.jhu_peduncle_mni2009c``) as the reference, PEDUNCLE + 100 x side, outside the
    search region; the Biondetti crus and tegmental parts are dropped. Langley et al. 2025 and Hwang et al. 2023 use "a
    reference ROI in the cerebral peduncle ... created using the MNI template", whose own geometry is unpublished."""
    if peduncle_img is None:
        from .atlases import jhu_peduncle_mni2009c
        peduncle_img = nib.load(jhu_peduncle_mni2009c())
    codes = np.asarray(region_labels(biondetti_img).dataobj)
    ped = np.asarray(peduncle_img.dataobj) > 0
    if ped.shape != codes.shape:
        raise ValueError("the peduncle and Biondetti labels are not on one grid")
    region, side = codes % 100, codes // 100
    keep = np.isin(region, [1, 2, 3, RING])
    region = np.where(keep, region, np.where(ped, PEDUNCLE, 0))
    return nib.Nifti1Image((region + 100 * side).astype(np.int16), peduncle_img.affine)


def study_labels(biondetti_img=None):
    """The study's native NM regions (manuscript Methods 4.13): Biondetti territory + 100 x side (MNI x < 0 left) and the
    complete background ROI as its three parts, STUDY_REF + 1..3 in MNI x order (left crus, midline, right crus). Unlike
    ``region_labels`` no search ring is cut from the background: the study's reference is the whole published ROI."""
    from scipy import ndimage

    if biondetti_img is None:
        from .atlases import biondetti_mni2009c
        biondetti_img = nib.load(biondetti_mni2009c())
    lab = np.asarray(biondetti_img.dataobj).astype(np.int16)
    ix = np.indices(lab.shape, sparse=True)
    x = np.broadcast_to(biondetti_img.affine[0, 3] + sum(biondetti_img.affine[0, k] * ix[k] for k in range(3)), lab.shape)
    codes = np.where(np.isin(lab, [1, 2, 3]), lab + 100 * np.where(x < 0, 1, 2), 0)
    parts, n = ndimage.label(lab == 4, structure=np.ones((3, 3, 3)))
    if n != 3:
        raise ValueError(f"the published background ROI should have three parts, found {n}")
    for rank, i in enumerate(np.argsort([x[parts == i].mean() for i in (1, 2, 3)]), 1):
        codes[parts == i + 1] = STUDY_REF + rank
    return nib.Nifti1Image(codes.astype(np.int16), biondetti_img.affine)


# ------------------------------------------------------------------------------------------ transforms
def inverse_warp(forward_warp):
    """ANTs inverse of a SyN displacement field on the same grid, cached beside it (``*1InverseWarp*``)."""
    fwd = Path(forward_warp)
    out = fwd.with_name(fwd.name.replace("1Warp", "1InverseWarp") if "1Warp" in fwd.name else "inverse_" + fwd.name)
    if not (out.exists() and out.stat().st_size > 0):
        import ants
        u = ants.image_read(str(fwd))
        tmp = out.with_name("partial_" + out.name)
        ants.image_write(ants.invert_displacement_field(u, u * 0), str(tmp))
        tmp.replace(out)
    return out


def _affine_file(tx, path):
    """Any affine SimpleITK transform written as an ITK AffineTransform file, which antsApplyTransforms reads."""
    import SimpleITK as sitk

    probes = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], float)
    mapped = np.array([tx.TransformPoint(tuple(p)) for p in probes])
    matrix = (mapped[1:] - mapped[0]).T / 50.0
    sitk.WriteTransform(sitk.AffineTransform(tuple(matrix.ravel()), tuple(mapped[0])), str(path))
    return path


def _slice_axis(img):
    """The slab's thin axis (smallest physical extent): its slice direction."""
    return int(np.argmin(np.array(img.shape[:3]) * np.linalg.norm(img.affine[:3, :3], axis=0)))


def pull_labels(label_img, grid_img, tx_t1_grid, warp, affine, codes=None, pad=12, interpolator="genericLabel"):
    """``label_img`` (MNI space) on ``grid_img``'s voxels in one ``interpolator`` (ANTs) resampling: grid -> T1 by the inverse of
    ``tx_t1_grid`` (SimpleITK, T1 point -> grid point, as ``nm.register_slab`` returns), T1 -> MNI by the inverse of
    the SyN (``warp``, ``affine``: the forward ``ants.registration`` pair, fixed = MNI). Returns (labels on the grid,
    coverage): for each name in ``codes`` ({name: [label values]}), the share of that region the grid holds, from the
    same pull onto the grid padded by ``pad`` slices along its thinnest physical extent (the slab normal)."""
    import ants

    axis = _slice_axis(grid_img)
    shape = list(grid_img.shape[:3])
    shape[axis] += 2 * pad
    shift = np.zeros(3)
    shift[axis] = -pad
    ext_aff = grid_img.affine.copy()
    ext_aff[:3, 3] = nib.affines.apply_affine(grid_img.affine, shift)
    with tempfile.TemporaryDirectory(prefix="pie_nm_native_") as d:
        nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), ext_aff), f"{d}/grid.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(label_img.dataobj).astype(np.float32), label_img.affine), f"{d}/labels.nii.gz")
        chain = [str(_affine_file(tx_t1_grid.GetInverse(), f"{d}/grid_to_t1.mat")), str(affine), str(inverse_warp(warp))]
        out = ants.apply_transforms(ants.image_read(f"{d}/grid.nii.gz"), ants.image_read(f"{d}/labels.nii.gz"), chain,
                                    whichtoinvert=[False, True, False], interpolator=interpolator)
        ext = np.rint(out.numpy()).astype(np.int16)
    inside = [slice(None)] * 3
    inside[axis] = slice(pad, pad + grid_img.shape[axis])
    lab = ext[tuple(inside)]
    coverage = {k: float(np.isin(lab, v).sum() / max(np.isin(ext, v).sum(), 1)) for k, v in (codes or {}).items()}
    return lab, coverage


# ------------------------------------------------------------------------------------------ measures
def _side_masks(codes):
    region, side = codes % 100, codes // 100
    return region, {"l": side == 1, "r": side == 2}


def langley_features(nm_img, codes, k=LANGLEY_K, prefix="nml_", ref_region=CRUS):
    """SNc volume (mm^3, total and per side): search-region voxels (nigra + 1 mm ring) above reference mean + k SD
    (``nm.hyperintense_volume``); one threshold from both sides of the reference, as the published cerebral-peduncle
    ROI. ``ref_region``: CRUS (version 1, the Biondetti crus parts) or PEDUNCLE (version 2, ``langley_labels``)."""
    from .nm import hyperintense_volume

    region, sides = _side_masks(codes)
    search = np.isin(region, [1, 2, 3, RING])
    ref = nib.Nifti1Image((region == ref_region).astype(np.uint8), nm_img.affine)
    out = {}
    for name, m in (("", search), ("_l", search & sides["l"]), ("_r", search & sides["r"])):
        r = hyperintense_volume(nm_img, nib.Nifti1Image(m.astype(np.uint8), nm_img.affine), ref, k=k)
        out[f"{prefix}sn_volume{name}_mm3"] = r["volume_mm3"]
    out.update({f"{prefix}threshold": r["threshold"], f"{prefix}ref_mean": r["reference_mean"], f"{prefix}ref_sd": r["reference_sd"],
                f"{prefix}n_ref": r["n_reference"], f"{prefix}k": float(k)})
    return out


def snceg_features(nm_img, sn, codes, prefix="nms_"):
    """Volume (mm^3), contrast ratio (I_sn / I_crus - 1) and CNR ((I_sn - I_crus) / SD_crus) of a segmented SN, per
    side and bilateral; the crus reference excludes segmented voxels (Lillebostad et al. 2025). Each side's reference is
    its own crus on the slices that hold its segmented SN (>= 20 voxels, else NaN), so a through-slab intensity gradient
    cancels; the crus over all slices and both sides (the reference before 11 Oct 2026) is kept as
    ``*_cr_allslices`` and ``crus_mean`` / ``crus_sd`` (diagnostics). ``*_cr_mode`` takes that side's crus mode
    (``nm_template._mode``) in place of its mean, the reference statistic of Cassidy et al. 2019 and Wengler et al. 2020,
    robust to partial-volume voxels at the crus edge. ``sn`` can be any nigral mask: ``process_subject`` also passes
    the CIT168 SNc (``nmf_``)."""
    from .nm_template import _mode

    nm = np.asarray(nm_img.dataobj, float)
    region, sides = _side_masks(codes)
    vox = float(abs(np.linalg.det(nm_img.affine[:3, :3])))
    crus = (region == CRUS) & ~sn & np.isfinite(nm)
    stats = lambda m: (float(nm[m].mean()), float(nm[m].std())) if m.sum() >= 20 else (np.nan, np.nan)
    mu, sd = stats(crus)
    inplane = tuple(a for a in range(3) if a != _slice_axis(nm_img))
    out = {f"{prefix}sn_volume_mm3": float(sn.sum() * vox), f"{prefix}crus_mean": mu, f"{prefix}crus_sd": sd}
    for s, m in sides.items():
        v = sn & m
        ref = crus & m & np.expand_dims(v.any(axis=inplane), inplane)
        smu, ssd = stats(ref)
        mode = _mode(nm[ref]) if np.isfinite(smu) else np.nan
        i = float(nm[v].mean()) if v.any() else np.nan
        out.update({f"{prefix}sn_volume_{s}_mm3": float(v.sum() * vox), f"{prefix}crus_{s}_mean": smu, f"{prefix}crus_{s}_sd": ssd,
                    f"{prefix}crus_{s}_mode": mode, f"{prefix}sn_{s}_cr": i / smu - 1, f"{prefix}sn_{s}_cnr": (i - smu) / ssd,
                    f"{prefix}sn_{s}_cr_mode": i / mode - 1, f"{prefix}sn_{s}_cr_allslices": i / mu - 1})
    for m in ("cr", "cnr", "cr_mode", "cr_allslices"):
        out[f"{prefix}sn_mean_{m}"] = (out[f"{prefix}sn_l_{m}"] + out[f"{prefix}sn_r_{m}"]) / 2
    return out


def _study_measure(nm, sides, reference, support, coverage, acquisition_ok):
    """The study's ``measure`` (``nm_rescue_measurements.py``): a strict bilateral contrast ratio. Returns the values and
    the failed checks; a unilateral result is never promoted."""
    good = np.isfinite(nm) & (nm > 0)
    reference = reference & (sides == 0)
    r, failures = {}, []
    for name, mask in (("reference", reference), ("l", sides == 1), ("r", sides == 2)):
        n, pos = int(mask.sum()), nm[mask & good]
        r["n_" + name], r[name + "_mean"] = n, float(pos.mean()) if len(pos) else np.nan
        if len(pos) < 20:
            failures.append(name + "_fewer_than_20_valid_voxels")
        if (good[mask].mean() if n else 0.0) < 0.95:
            failures.append(name + "_invalid_signal")
        if ((support[mask] >= 0.99).mean() if n else 0.0) < 0.95:
            failures.append(name + "_repeat_support")
    ref = nm[reference & good]
    r["reference_cv"] = float(ref.std() / ref.mean()) if len(ref) else np.nan
    if not np.isfinite(r["reference_cv"]) or r["reference_cv"] >= 0.4:
        failures.append("reference_cv")
    for side, cov in zip("lr", coverage):
        r[side + "_cr"] = r[side + "_mean"] / r["reference_mean"] - 1 if r["reference_mean"] > 0 else np.nan
        if cov < 0.9:
            failures.append(side + "_slab_coverage")
    r["mean_cr"] = float(np.mean([r["l_cr"], r["r_cr"]]))
    if not acquisition_ok:
        failures.append("acquisition")
    if not np.isfinite(r["mean_cr"]):
        failures.append("nonfinite_bilateral_cr")
    return r, failures


def study_features(nm, codes, support, cov, acquisition_ok, prefix="nm_bion_"):
    """The study's native NM measure (Methods 4.13; its ``nonlinear_nm_measurements_v2``): for the whole nigra and each
    Biondetti territory, each side's contrast ratio against the complete background ROI (mean / mean - 1), NaN on both
    sides unless that region passes every check: >= 20 valid voxels per side and in the reference, >= 95 % of them with
    signal and reached by every repeat (``support`` >= 0.99), reference CV < 0.4, >= 90 % slab coverage of each side and
    of each background part, and an acceptable acquisition. ``codes`` is ``study_labels`` on the slab grid and ``cov``
    the ``pull_labels`` coverage of "<region>_<side>" and "ref_<k>"; arrays share one voxel order."""
    region, side = codes % 100, codes // 100
    target, reference = codes > 100, (codes > STUDY_REF) & (codes <= STUDY_REF + 3)
    ref_ok = min(cov[f"ref_{k}"] for k in (1, 2, 3)) >= 0.9
    out = {}
    for name, keep in [("whole", target)] + [(t, target & (region == k)) for k, t in TERRITORIES.items()]:
        r, failures = _study_measure(nm, np.where(keep, side, 0), reference, support, (cov[f"{name}_l"], cov[f"{name}_r"]), acquisition_ok)
        failures += [] if ref_ok else ["reference_component_slab_coverage"]
        for s in ("l", "r", "mean"):
            out[f"{prefix}{name}_{s}_cr"] = r[f"{s}_cr"] if not failures else np.nan
        if name == "whole":
            out.update({f"{prefix}qc_pass": not failures, f"{prefix}qc_failed_checks": ";".join(failures), f"{prefix}n_sn_l": r["n_l"],
                        f"{prefix}n_sn_r": r["n_r"], f"{prefix}n_reference": r["n_reference"], f"{prefix}reference_cv": r["reference_cv"]})
    return out


def repeat_nigral_rms(points, transforms):
    """Largest RMS displacement (mm) of the nigral ``points`` (LPS mm, slab space) under the repeat -> anchor rigids: the
    study's nigral repeat-motion check (< 3 mm). No points: infinite."""
    if not len(points):
        return np.inf
    return max((float(np.sqrt(np.mean(np.sum((np.array([t.TransformPoint(tuple(map(float, p))) for p in points]) - points) ** 2, axis=1))))
                for t in transforms), default=0.0)


def syn_numerical_qc(warp, affine, points, max_roundtrip_mm=0.5):
    """The study's transform QC on a cached SyN at ``points`` (MNI, LPS mm): round trip MNI -> T1 -> MNI (the forward
    warp and affine, then the affine's inverse and the inverse warp), the forward warp's Jacobian determinant, and the
    affine's determinant. Passes when the round trip stays within ``max_roundtrip_mm`` and every Jacobian and the
    determinant are finite and positive."""
    import ants
    import SimpleITK as sitk

    field = lambda path: sitk.DisplacementFieldTransform(sitk.ReadImage(str(path), sitk.sitkVectorFloat64))
    aff = sitk.ReadTransform(str(affine))
    fwd, back = sitk.CompositeTransform(3), sitk.CompositeTransform(3)
    fwd.AddTransform(aff), fwd.AddTransform(field(warp))                       # the last added acts first: warp, then affine
    back.AddTransform(field(inverse_warp(warp))), back.AddTransform(aff.GetInverse())
    pts = [tuple(map(float, p)) for p in points]
    trip = np.array([back.TransformPoint(fwd.TransformPoint(p)) for p in pts]) - np.array(pts)
    probes = np.array([aff.TransformPoint(tuple(v)) for v in np.vstack([np.zeros(3), np.eye(3)])])
    det = float(np.linalg.det((probes[1:] - probes[0]).T))
    jac = ants.create_jacobian_determinant_image(ants.image_read(str(warp)).split_channels()[0], str(warp))
    j = sitk.GetImageFromArray(jac.numpy().T.copy())
    j.SetOrigin(jac.origin), j.SetSpacing(jac.spacing), j.SetDirection(tuple(np.asarray(jac.direction).ravel()))
    idx = [j.TransformPhysicalPointToIndex(p) for p in pts]
    vals = np.array([j.GetPixel(i) if all(0 <= a < b for a, b in zip(i, j.GetSize())) else np.nan for i in idx])
    out = {"syn_roundtrip_mm_max": float(np.linalg.norm(trip, axis=1).max()), "syn_jacobian_min": float(np.min(vals)), "syn_affine_det": det}
    out["syn_numerical_qc_pass"] = bool(out["syn_roundtrip_mm_max"] <= max_roundtrip_mm and np.isfinite(vals).all() and (vals > 0).all() and det > 0)
    return out


# ------------------------------------------------------------------------------------------ snceg
def snceg_model(cache_dir=SNCEG_DIR):
    """Directory holding the pinned snceg weights, downloaded once and sha256-checked."""
    from .atlases import _pinned

    for name, sha in SNCEG_FILES.items():
        _pinned(f"https://huggingface.co/lillepeder/SNceg-0.1/resolve/{SNCEG_REVISION}/{name}", Path(cache_dir) / name, sha)
    return Path(cache_dir)


def snceg_mask(nm_path, out_path, python=None, timeout=900):
    """Boolean SN mask on ``nm_path``'s grid from the snceg model, run by ``snceg_runner.py`` in the snceg environment
    (``PIE_SNCEG_PYTHON``; torch 2.0.1 + fastMONAI 0.4, which PIE's own environment cannot hold)."""
    model = snceg_model()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "2"}
    subprocess.run([python or SNCEG_PYTHON, str(Path(__file__).with_name("snceg_runner.py")), "--input", str(nm_path),
                    "--output", str(out_path), "--model-dir", str(model)], check=True, env=env, timeout=timeout,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    src, pred = nib.load(nm_path), nib.load(out_path)
    if pred.shape[:3] != src.shape[:3] or not np.allclose(pred.affine, src.affine, atol=1e-4):
        raise ValueError("snceg returned a mask on a different grid")
    return np.asarray(pred.dataobj) > 0


# ------------------------------------------------------------------------------------------ TSE input
TSE_PATTERN = r"PD.?T2.*TSE|DUAL.?TSE|T2.?DE\b|PD.?\+.?T2|double.?echo"


def flag_tse(idx):
    """PPMI dual-echo turbo spin echo series (proton-density + T2), the input of Lillebostad et al.'s PPMI analysis."""
    d = idx["desc"].str.replace("_", " ")
    return d.str.contains(TSE_PATTERN, case=False, regex=True) & ~d.str.contains("FLAIR", case=False)


def tse_subject(patno, series_rows, fastsurfer_dir, work_dir, echo=1):
    """Convert a session's dual-echo TSE images (PPMI stores each echo as its own image), keep echo ``echo`` (1 = the
    shortest TE, proton-density weighted) as <work>/<patno>/nm_mean.nii.gz and the rigid T1 -> image transform
    (``nm.register_slab``) as slab_to_t1.tfm, so ``nm_template syn`` and the measures run on it as on an NM slab."""
    import json
    import shutil

    import SimpleITK as sitk

    from . import nm as _nm

    d = Path(work_dir) / str(patno)
    d.mkdir(parents=True, exist_ok=True)
    with _nm.ZipArchiveCache() as zips:
        niis = [f for r in series_rows for f in _nm.convert(r["zip"], r["prefix"], d / "nii", archive_cache=zips)]
    metas = [json.loads(Path(f[:-7] + ".json").read_text()) for f in niis]
    tes = sorted({m["EchoTime"] for m in metas if "EchoTime" in m})
    if len(tes) < echo:
        raise ValueError(f"echo {echo} not found (echo times {tes})")
    pick = next(i for i, m in enumerate(metas) if m.get("EchoTime") == tes[echo - 1])
    src = nib.load(niis[pick])
    img = nib.Nifti1Image(np.asarray(src.dataobj, np.float32), src.affine)
    nib.save(img, d / "nm_mean.nii.gz")
    tx, mi, init, *_ = _nm.register_slab(img, fastsurfer_dir)
    sitk.WriteTransform(tx, str(d / "slab_to_t1.tfm"))
    shutil.rmtree(d / "nii", ignore_errors=True)
    m = metas[pick]
    return {"patno": patno, "tse_echo": echo, "tse_echo_te_s": tes[echo - 1], "manufacturer": str(m.get("Manufacturer", "")),
            "model": str(m.get("ManufacturersModelName", m.get("ManufacturerModelName", ""))),
            "voxel_mm": "x".join(str(round(float(z), 2)) for z in img.header.get_zooms()[:3]), "reg_nm_t1_mi": mi,
            "reg_init": init, "acquisition_date": series_rows[0]["date"], "fs_image_id": Path(fastsurfer_dir).name}


def _tse_job(args):
    patno, rows, fs_dir, work, echo = args
    try:
        return {**tse_subject(patno, rows, fs_dir, work, echo=echo), "error": ""}
    except Exception as e:
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# ------------------------------------------------------------------------------------------ driver
SNC = 7                            # CIT168 label of the SNc (``nm._mni_codes``)


def n4(nm_img):
    """The slab after ANTs N4 bias-field correction inside its head mask (``ants.get_mask``): a receive-field gradient
    across the midbrain would otherwise enter the SN/crus contrast unevenly (vendor asymmetries, Wengler et al. 2021)."""
    import ants

    with tempfile.TemporaryDirectory(prefix="pie_nm_n4_") as d:
        nib.save(nib.Nifti1Image(np.asarray(nm_img.dataobj, np.float32), nm_img.affine), f"{d}/nm.nii.gz")
        img = ants.image_read(f"{d}/nm.nii.gz")
        ants.image_write(ants.n4_bias_field_correction(img, mask=ants.get_mask(img)), f"{d}/n4.nii.gz")
        return nib.Nifti1Image(np.asarray(nib.load(f"{d}/n4.nii.gz").dataobj, np.float32), nm_img.affine)


def process_subject(work_dir, patno, fastsurfer_dir, snceg=False, min_cov=0.9, langley_peduncle=False, image="nm_mean.nii.gz",
                    bias_correct=False):
    """<work>/<patno>/nm_mean.nii.gz + slab_to_t1.tfm (``nm --keep-nifti``) and the cached SyN (``nm_template syn``)
    -> native region labels (saved), coverage, Langley volume and optionally snceg measures. A side whose search region
    or crus is less than ``min_cov`` inside the slab is missing. ``langley_peduncle`` adds version 2 of the Langley
    volume (``nmlp_*``: the JHU cerebral peduncle as reference, ``langley_labels``, regions saved as
    langley_regions_nm.nii.gz): whatever part of the peduncle the slab holds is the reference (>= 20 voxels, else NaN),
    and a side whose search region is less than ``min_cov`` inside the slab is missing.
    Every run adds ``nmf_*``: ``snceg_features`` on the CIT168 SNc (``SNC``) pulled onto the slab in place of a
    segmentation, a fixed-mask contrast with no threshold or model (Shaff et al. 2024, doi:10.1002/hbm.26544).
    ``image``: another slab on nm_mean's grid to measure (``nm --split-half`` writes nm_half_a/b.nii.gz); snceg segments
    it afresh (snceg_sn_<stem>.nii.gz). ``bias_correct``: measure intensities on the ``n4``-corrected slab; snceg
    still segments the uncorrected one, as published."""
    import SimpleITK as sitk

    from .nm import _mni_codes
    from .nm_template import syn_cached, syn_paths

    d = Path(work_dir) / str(patno)
    if not syn_cached(fastsurfer_dir):
        raise FileNotFoundError("no SyN cache; run nm_template syn first")
    nm_path = d / image
    nm_img = nib.load(nm_path)
    warp, affine = syn_paths(fastsurfer_dir)["fwd"]
    tx = sitk.ReadTransform(str(d / "slab_to_t1.tfm"))
    groups = {"search_l": [101, 102, 103, 100 + RING], "search_r": [201, 202, 203, 200 + RING], "crus_l": [100 + CRUS], "crus_r": [200 + CRUS]}
    codes, cov = pull_labels(region_labels(), nm_img, tx, warp, affine, codes=groups)
    if image == "nm_mean.nii.gz":
        nib.save(nib.Nifti1Image(codes, nm_img.affine), d / "native_regions_nm.nii.gz")
    snc = pull_labels(_mni_codes(), nm_img, tx, warp, affine)[0] % 100 == SNC
    nm_img = n4(nm_img) if bias_correct else nm_img
    row = {f"native_cov_{k}": v for k, v in cov.items()}
    ok = {s: cov[f"search_{s}"] >= min_cov and cov[f"crus_{s}"] >= min_cov for s in "lr"}
    row.update(langley_features(nm_img, codes))
    if langley_peduncle:
        pgroups = {"peduncle_l": [100 + PEDUNCLE], "peduncle_r": [200 + PEDUNCLE]}
        pcodes, pcov = pull_labels(langley_labels(), nm_img, sitk.ReadTransform(str(d / "slab_to_t1.tfm")), warp, affine, codes=pgroups)
        nib.save(nib.Nifti1Image(pcodes, nm_img.affine), d / "langley_regions_nm.nii.gz")
        row.update({f"native_cov_{k}": v for k, v in pcov.items()})
        try:
            prow = langley_features(nm_img, pcodes, prefix="nmlp_", ref_region=PEDUNCLE)
        except ValueError:                  # fewer than 20 peduncle voxels in the slab
            prow = {f"nmlp_sn_volume{s}_mm3": np.nan for s in ("", "_l", "_r")}
        for s in "lr":
            if cov[f"search_{s}"] < min_cov:
                prow[f"nmlp_sn_volume_{s}_mm3"] = prow["nmlp_sn_volume_mm3"] = np.nan
        row.update(prow)
    row.update(snceg_features(nm_img, snc, codes, prefix="nmf_"))
    if snceg:
        out = d / ("snceg_sn_nm.nii.gz" if image == "nm_mean.nii.gz" else f"snceg_sn_{image.split('.')[0]}.nii.gz")
        fresh = out.exists() and out.stat().st_mtime > nm_path.stat().st_mtime
        sn = np.asarray(nib.load(out).dataobj) > 0 if fresh else snceg_mask(nm_path, out)
        row.update(snceg_features(nm_img, sn, codes))
    for key in list(row):
        side = "l" if "_l_" in key or key.endswith("_l_mm3") else "r" if "_r_" in key or key.endswith("_r_mm3") else None
        if key.startswith(("nml_sn", "nms_sn", "nmf_sn")) and ((side and not ok[side]) or (side is None and not all(ok.values()))):
            row[key] = np.nan
    return row


def study_subject(work_dir, patno, fastsurfer_dir, labels=None):
    """<work>/<patno> from ``nm --keep-nifti`` (slab, rigid, repeat transforms, repeat support and selection) and the
    cached SyN -> the study's measure (``study_features``) with its acquisition checks (>= 3 identified repeats, repeat
    motion and nigral RMS displacement < 3 mm) and transform checks (``syn_numerical_qc``; a failure blanks every
    contrast). ``labels`` defaults to ``study_labels()``; the regions on the slab are saved as study_regions_nm.nii.gz."""
    import json

    import SimpleITK as sitk

    from .nm_template import syn_paths

    d = Path(work_dir) / str(patno)
    runs = Path(work_dir) / "nm_features.csv"          # nm records its denoising per subject
    if runs.exists():
        den = pd.read_csv(runs, usecols=lambda c: c in ("patno", "denoise")).astype(str)
        if den.get("denoise", pd.Series(dtype=str))[den["patno"].eq(str(patno))].eq("mppca").any():
            raise ValueError("the study's measure is defined on slabs without MP-PCA: run nm without --denoise")
    rt = d / "repeat_transforms"
    nm_img = nib.load(d / "nm_mean.nii.gz")
    labels = study_labels() if labels is None else labels
    warp, affine = syn_paths(fastsurfer_dir)["fwd"]
    groups = {f"{name}_{s}": [100 * k + t for t in ts] for name, ts in [("whole", (1, 2, 3))] + [(n, (t,)) for t, n in TERRITORIES.items()]
              for s, k in (("l", 1), ("r", 2))}
    groups.update({f"ref_{k}": [STUDY_REF + k] for k in (1, 2, 3)})
    codes, cov = pull_labels(labels, nm_img, sitk.ReadTransform(str(d / "slab_to_t1.tfm")), warp, affine, codes=groups)
    nib.save(nib.Nifti1Image(codes, nm_img.affine), d / "study_regions_nm.nii.gz")
    moves = [sitk.ReadTransform(str(p)) for p in sorted(rt.glob("repeat_*_to_anchor.tfm"))]
    motion = max((float(np.linalg.norm(np.array(t.GetParameters())[3:6])) for t in moves), default=0.0)   # as nm.average_repeats
    nigral = repeat_nigral_rms(nib.affines.apply_affine(nm_img.affine, np.argwhere(codes > 100)) * [-1, -1, 1], moves)
    sel = json.loads((d / "repeat_selection.json").read_text())
    ok = len(sel["volumes"]) >= 3 and sel.get("n_missing_acquisition_identity", 0) == 0 and motion < 3 and nigral < 3
    lab = np.asarray(labels.dataobj)
    qc = syn_numerical_qc(warp, affine, nib.affines.apply_affine(labels.affine, np.argwhere(lab > 100)) * [-1, -1, 1])
    row = study_features(np.asarray(nm_img.dataobj, float), codes, np.asarray(nib.load(rt / "repeat_support_fraction.nii.gz").dataobj), cov, ok)
    if not qc["syn_numerical_qc_pass"]:
        row = {k: np.nan if k.endswith("_cr") else v for k, v in row.items()}
        row["nm_bion_qc_pass"] = False
        row["nm_bion_qc_failed_checks"] = ";".join(filter(None, [row["nm_bion_qc_failed_checks"], "syn_numerical_qc"]))
    return {**row, **qc, "repeat_nigral_rms_mm_max": nigral, **{f"nm_bion_cov_{k}": v for k, v in cov.items()}}


def _job(args):
    work, patno, fs_dir, snceg, *rest = args
    study, peduncle, image, bias_correct = (list(rest) + [False, False, "nm_mean.nii.gz", False])[:4]
    try:
        row = process_subject(work, patno, fs_dir, snceg=snceg, langley_peduncle=peduncle, image=image, bias_correct=bias_correct)
        return {"patno": patno, "error": "", **row, **(study_subject(work, patno, fs_dir) if study else {})}
    except Exception as e:
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main(argv=None):
    from .nm_template import _subjects

    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--work-dir", required=True, help="a pie.imaging.nm work dir run with --keep-nifti (and --denoise for Langley)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patnos")
    ap.add_argument("--snceg", action="store_true", help="also segment with snceg (needs PIE_SNCEG_PYTHON)")
    ap.add_argument("--langley-peduncle", action="store_true", help="also version 2 of the Langley volume (nmlp_*): the JHU "
                    "cerebral peduncle as the reference (langley_labels)")
    ap.add_argument("--study", action="store_true", help="also the study's measure (study_subject: Biondetti territories vs "
                    "the complete background, the study's QC); needs nm --keep-nifti without --denoise for the study's quantity")
    ap.add_argument("--image", default="nm_mean.nii.gz", help="measure this slab of each subject's work dir instead "
                    "(on nm_mean's grid, e.g. nm_half_a.nii.gz from nm --split-half); table nm_native_features_<stem>.csv")
    ap.add_argument("--n4", action="store_true", help="measure intensities after N4 bias-field correction (n4); "
                    "table suffix _n4")
    ap.add_argument("--prepare-tse", action="store_true", help="first stage for dual-echo TSE input: convert, pick the echo, "
                    "register; writes <work>/nm_features.csv (then run nm_template syn and this CLI without the flag)")
    ap.add_argument("--zips", nargs="+", help="with --prepare-tse: the PPMI zips holding the TSE series")
    ap.add_argument("--echo", type=int, default=1, help="with --prepare-tse: 1 = proton-density (shortest TE), 2 = T2")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if a.prepare_tse:
        from .batch import fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows, t1_session_dates

        idx = load_index(work / "tse_index.csv", a.zips, flag_tse)
        idx = idx[idx["selected"]]
        fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
        t1d = t1_session_dates(a.sessions, fs)
        out_csv = work / "nm_features.csv"
        done = set(pd.read_csv(out_csv)["patno"]) if out_csv.exists() and out_csv.stat().st_size else set()
        jobs = [(int(p), session_rows(g, t1d.get(int(p))), fs[int(p)], str(work), a.echo) for p, g in idx.groupby("patno") if p not in done and int(p) in fs]
        run_batch(filter_jobs(jobs, a.patnos, a.limit), _tse_job, out_csv, workers=a.workers)
        return
    subjects = _subjects(work, a.sessions, a.fastsurfer_dir, a.patnos)
    region_labels()                    # build the atlas bridge once before the workers start
    if a.langley_peduncle:
        langley_labels()               # and the JHU peduncle bridge
    if a.study and a.image != "nm_mean.nii.gz":
        raise SystemExit("--study measures nm_mean.nii.gz only")
    stem = "" if a.image == "nm_mean.nii.gz" else "_" + a.image.split(".")[0]
    out_csv = work / f"nm_native_features{stem}{'_n4' if a.n4 else ''}.csv"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, [(str(work), p, str(fs), a.snceg, a.study, a.langley_peduncle, a.image, a.n4) for p, fs in subjects]))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"{int((out['error'] == '').sum())}/{len(out)} subjects -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
