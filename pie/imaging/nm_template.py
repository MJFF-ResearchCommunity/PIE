"""
Neuromelanin template pipeline (after Wengler et al. 2020, NeuroImage; Cassidy et al. 2019, PNAS): the averaged
2D GRE-MT slab of every subject is brought into a common 0.5 mm MNI midbrain box through a rigid slab -> T1 and a
deformable (ANTs SyN) T1 -> MNI registration, a study neuromelanin template is built from all normalized slabs, the
substantia-nigra and crus-cerebri masks are defined on that template, and per-subject contrast maps
CNR = (I - mode(crus)) / mode(crus) are read in template space (mean CNR in the template SN and its quadrants).
Unlike the atlas pipeline in ``pie.imaging.nm`` (affine MNI mapping + a T1/T2-defined SN label), the masks lie on the
neuromelanin band itself and the reference sits in the dark peduncle, both several millimetres from the band.

Stages (each resumable, run in order):
    syn        cache an ANTs SyN T1(brain) -> MNI152 1 mm warp per FastSurfer subject (~2 min each; reused by dwi_refine)
    normalize  slab -> T1 (rigid, SimpleITK) -> MNI midbrain box (SyN), saved as <work>/<patno>/nm_mni.nii.gz
    template   mean of the intensity-normalised slabs -> <work>/template/nm_template.nii.gz + sn / crus masks
    features   per subject: crus mode, CNR map, SN mean CNR (+ anterior/posterior/medial/lateral quadrants) ->
               <work>/nm_template_features.csv

    venv_imaging/bin/python -m pie.imaging.nm_template syn --sessions Imaging/derived/sessions.csv \\
        --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SYN_TYPE = "antsRegistrationSyNQuick[s]"


def mni_brain_path():
    """The nilearn MNI152 1 mm template, brain-masked, written once next to nilearn's data (ANTs wants a file)."""
    p = Path.home() / "nilearn_data" / "mni152_brain_1mm.nii.gz"
    if not p.exists():
        from nilearn import datasets

        t = datasets.load_mni152_template(resolution=1)
        m = datasets.load_mni152_brain_mask(resolution=1)
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asanyarray(t.dataobj).astype(np.float32) * (np.asanyarray(m.dataobj) > 0), t.affine), p)
    return p


def syn_paths(fastsurfer_dir):
    tdir = Path(fastsurfer_dir) / "mri" / "transforms"
    return {"fwd": [tdir / "t1_to_mni_syn_1Warp.nii.gz", tdir / "t1_to_mni_syn_0GenericAffine.mat"]}


def crop_warp(path, margin_mm=20.0):
    """Crop a SyN displacement field (defined in MNI space) to the midbrain box plus a margin: the template pipeline only
    resamples inside the box, and a full 1 mm field is ~85 MB per subject. Keeps the NIfTI intent so ANTs reads it."""
    img = nib.load(path)
    inv = np.linalg.inv(img.affine)
    lo = np.array(BOX_ORIGIN_RAS) - margin_mm
    hi = np.array(BOX_ORIGIN_RAS) + np.array(BOX_SHAPE) * BOX_MM + margin_mm
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = (inv @ corners.T).T[:, :3]
    i0 = np.maximum(np.floor(ijk.min(axis=0)).astype(int), 0)
    i1 = np.minimum(np.ceil(ijk.max(axis=0)).astype(int) + 1, np.array(img.shape[:3]))
    if tuple(i1 - i0) == img.shape[:3]:
        return
    cropped = img.slicer[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    nib.save(cropped, path)


def syn_cache(fastsurfer_dir, syn_type=SYN_TYPE):
    """ANTs SyN of the brain-masked conformed T1 to MNI152 (1 mm), cached under <subject>/mri/transforms. Returns the
    forward transform list in ANTs order (``fwd`` warps T1-space images into MNI). The inverse warp is not kept: a
    displacement field is ~70 MB per subject, and MNI -> T1 mappings can be recomputed when a study needs them."""
    paths = syn_paths(fastsurfer_dir)
    if all(p.exists() for p in paths["fwd"]):
        return {k: [str(p) for p in v] for k, v in paths.items()}
    import ants

    mri = Path(fastsurfer_dir) / "mri"
    t1 = ants.image_read(str(mri / "orig.mgz"))
    t1b = t1 * ants.threshold_image(ants.image_read(str(mri / "mask.mgz")), 0.5, 1e9)
    reg = ants.registration(fixed=ants.image_read(str(mni_brain_path())), moving=t1b, type_of_transform=syn_type)
    paths["fwd"][0].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(reg["fwdtransforms"][0], paths["fwd"][0])       # 1Warp
    shutil.copy(reg["fwdtransforms"][1], paths["fwd"][1])       # 0GenericAffine
    crop_warp(paths["fwd"][0])
    return {k: [str(p) for p in v] for k, v in paths.items()}


def _syn_job(fs_dir):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    try:
        syn_cache(fs_dir)
        return {"fastsurfer": Path(fs_dir).name, "error": ""}
    except Exception as e:
        return {"fastsurfer": Path(fs_dir).name, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# ------------------------------------------------------------------------------------------ normalize
BOX_ORIGIN_RAS = (-30.0, -45.0, -30.0)   # MNI midbrain box: x -30..30, y -45..5, z -30..10 mm
BOX_SHAPE = (120, 100, 80)
BOX_MM = 0.5


def box_path():
    """Empty 0.5 mm NIfTI defining the MNI midbrain box (ANTs wants a reference image file)."""
    p = Path.home() / "nilearn_data" / "mni_midbrain_box_0.5mm.nii.gz"
    if not p.exists():
        aff = np.diag([BOX_MM, BOX_MM, BOX_MM, 1.0])
        aff[:3, 3] = BOX_ORIGIN_RAS
        nib.save(nib.Nifti1Image(np.zeros(BOX_SHAPE, np.float32), aff), p)
    return p


def slab_in_t1(nm_img, tx_t1_slab, t1_img, half_mm=(45.0, 45.0, 20.0), mm=BOX_MM):
    """The slab resampled onto a ``mm``-isotropic grid in T1 physical space around the slab centre (T1 -> slab
    transform as returned by ``nm.register_slab``). Returns a NIfTI in the T1's RAS frame."""
    import SimpleITK as sitk

    from .dwi import _sitk_native

    slab = _sitk_native(np.asanyarray(nm_img.dataobj).astype(np.float32), nm_img.affine)
    centre_slab = np.array(slab.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in slab.GetSize()]))
    centre_t1 = np.array(tx_t1_slab.GetInverse().TransformPoint(tuple(float(c) for c in centre_slab)))   # LPS, T1 space
    size = [int(round(2 * h / mm)) for h in half_mm]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing((mm, mm, mm))
    ref.SetDirection((1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    ref.SetOrigin(tuple(float(c - h) for c, h in zip(centre_t1, half_mm)))
    out = sitk.Resample(slab, ref, tx_t1_slab, sitk.sitkLinear, 0.0)
    arr = np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))          # (x, y, z) along the LPS axes
    o = np.array(out.GetOrigin())
    aff = np.diag([-mm, -mm, mm, 1.0])                                  # LPS axes expressed in RAS
    aff[:3, 3] = (-o[0], -o[1], o[2])
    return nib.Nifti1Image(np.ascontiguousarray(arr), aff)


def normalize_subject(work_dir, patno, fastsurfer_dir):
    """<work>/<patno>/nm_mean.nii.gz -> nm_mni.nii.gz (0.5 mm MNI midbrain box) through the rigid slab -> T1 transform
    (cached as slab_to_t1.tfm, recomputed with ``nm.register_slab`` when missing) and the cached SyN T1 -> MNI warp."""
    import SimpleITK as sitk
    import ants

    from .nm import register_slab

    d = Path(work_dir) / str(patno)
    nm_img = nib.load(d / "nm_mean.nii.gz")
    tfm = d / "slab_to_t1.tfm"
    if tfm.exists():
        tx = sitk.ReadTransform(str(tfm))
    else:
        tx = register_slab(nm_img, fastsurfer_dir)[0]
        sitk.WriteTransform(tx, str(tfm))
    t1 = nib.load(Path(fastsurfer_dir) / "mri" / "orig.mgz")
    t1_slab = slab_in_t1(nm_img, tx, t1)
    tmp = d / "_slab_t1.nii.gz"
    nib.save(t1_slab, tmp)
    syn = syn_cache(fastsurfer_dir)
    warped = ants.apply_transforms(fixed=ants.image_read(str(box_path())), moving=ants.image_read(str(tmp)), transformlist=syn["fwd"], interpolator="linear")
    ants.image_write(warped, str(d / "nm_mni.nii.gz"))
    tmp.unlink(missing_ok=True)
    arr = warped.numpy()
    return {"patno": patno, "mni_nonzero_frac": float((arr > 0).mean())}


def _normalize_job(args):
    work, patno, fs_dir = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    try:
        import SimpleITK as sitk
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
        return {**normalize_subject(work, patno, fs_dir), "error": ""}
    except Exception as e:
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def sn_prior():
    """CIT168 SNc + SNr (Pauli 2017, nilearn) resampled (nearest) onto the MNI box: the search prior for the masks."""
    from nilearn.image import resample_to_img

    from .dwi import pauli_atlas

    box = nib.load(box_path())
    lab = resample_to_img(pauli_atlas(), box, interpolation="nearest", force_resample=True, copy_header=True)
    return np.isin(np.asanyarray(lab.dataobj), [7, 9])


# ------------------------------------------------------------------------------------------ template + masks
def _dilate_mm(mask, mm):
    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=max(1, int(round(mm / BOX_MM))))


def _largest(mask):
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == 1 + int(np.argmax(sizes))


def _mode(v, bins=64):
    """Mode of a sample by histogram over its 1st-99th percentile range (Cassidy 2019 uses the crus mode)."""
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 20:
        return np.nan
    lo, hi = np.percentile(v, [1, 99])
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    return float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))


def build_template(work_dir, patnos, min_frac=0.5):
    """Voxel-wise mean of the intensity-normalised MNI slabs (each divided by its own median inside the SN prior
    dilated 10 mm); voxels with data in fewer than ``min_frac`` of the subjects are zero. Returns (template, count)."""
    prior = _dilate_mm(sn_prior(), 10.0)
    acc = np.zeros(BOX_SHAPE, np.float64)
    cnt = np.zeros(BOX_SHAPE, np.int32)
    n = 0
    for p in patnos:
        f = Path(work_dir) / str(p) / "nm_mni.nii.gz"
        if not f.exists():
            continue
        a = np.asanyarray(nib.load(f).dataobj).astype(np.float32)
        med = np.median(a[prior & (a > 0)]) if (prior & (a > 0)).sum() > 100 else 0
        if not med > 0:
            continue
        ok = a > 0
        acc[ok] += a[ok] / med
        cnt += ok
        n += 1
    t = np.where(cnt >= min_frac * max(n, 1), acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
    return t, cnt, n


def template_masks(template, crus_mm=(4.0, 9.0), sn_mm=3.0, cnr_min=0.06):
    """Masks defined on the template (RAS box: +x right, +y anterior). Per side:
    crus  = the darker half of the sector anterior-lateral to the SN prior, ``crus_mm`` from it, largest component;
    sn    = voxels within ``sn_mm`` of the prior whose template CNR (vs the crus mode) is >= max(cnr_min, Otsu), largest
            component. Returns dict of boolean masks: sn_l, sn_r, crus_l, crus_r, plus the template CNR map."""
    from skimage.filters import threshold_otsu

    prior = sn_prior()
    x = np.arange(BOX_SHAPE[0])[:, None, None] * BOX_MM + BOX_ORIGIN_RAS[0]
    y = np.arange(BOX_SHAPE[1])[None, :, None] * BOX_MM + BOX_ORIGIN_RAS[1]
    out = {}
    for side, sel in (("l", x < 0), ("r", x >= 0)):
        pr = prior & np.broadcast_to(sel, BOX_SHAPE)
        if not pr.any():
            continue
        cx, cy = x.ravel()[np.nonzero(pr)[0]].mean(), y.ravel()[np.nonzero(pr)[1]].mean()
        sector = _dilate_mm(pr, crus_mm[1]) & ~_dilate_mm(pr, crus_mm[0]) & (np.broadcast_to(y, BOX_SHAPE) > cy) & (np.abs(np.broadcast_to(x, BOX_SHAPE)) > abs(cx)) & (template > 0)
        if sector.sum() < 50:
            continue
        crus = _largest(sector & (template <= np.median(template[sector])))
        mode = _mode(template[crus])
        if not np.isfinite(mode):
            continue
        out[f"crus_{side}"] = crus
        cnr = template / mode - 1.0
        search = _dilate_mm(pr, sn_mm) & (template > 0)
        thr = max(cnr_min, float(threshold_otsu(cnr[search]))) if search.sum() > 50 else cnr_min
        out[f"sn_{side}"] = _largest(search & (cnr >= thr))
        out[f"sn_thr_{side}"] = thr
    return out


def save_template(work_dir, template, count, masks, n):
    tdir = Path(work_dir) / "template"
    tdir.mkdir(parents=True, exist_ok=True)
    aff = nib.load(box_path()).affine
    nib.save(nib.Nifti1Image(template, aff), tdir / "nm_template.nii.gz")
    nib.save(nib.Nifti1Image(count.astype(np.int16), aff), tdir / "nm_template_count.nii.gz")
    lab = np.zeros(BOX_SHAPE, np.int16)
    for i, k in enumerate(("sn_l", "sn_r", "crus_l", "crus_r"), start=1):
        if k in masks:
            lab[masks[k]] = i
    nib.save(nib.Nifti1Image(lab, aff), tdir / "nm_template_masks.nii.gz")     # 1 sn_l, 2 sn_r, 3 crus_l, 4 crus_r
    (tdir / "template_info.txt").write_text(f"subjects {n}\nsn_thr_l {masks.get('sn_thr_l')}\nsn_thr_r {masks.get('sn_thr_r')}\n" +
                                            "".join(f"{k} {int(v.sum())} voxels\n" for k, v in masks.items() if isinstance(v, np.ndarray)))


def template_figure(work_dir):
    """template/template_qc.png: three axial slices through the SN masks with sn (red) and crus (lime) outlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tdir = Path(work_dir) / "template"
    t = np.asanyarray(nib.load(tdir / "nm_template.nii.gz").dataobj)
    masks = load_masks(work_dir)
    sn = masks["sn_l"] | masks["sn_r"]
    crus = masks["crus_l"] | masks["crus_r"]
    zs = np.unique(np.nonzero(sn)[2])
    zs = zs[[len(zs) // 5, len(zs) // 2, 4 * len(zs) // 5]] if len(zs) >= 3 else zs
    fig, axes = plt.subplots(1, max(len(zs), 1), figsize=(4 * max(len(zs), 1), 4))
    m = np.median(t[t > 0]) if (t > 0).any() else 1.0
    for ax, z in zip(np.atleast_1d(axes), zs):
        ax.imshow(t[:, :, z].T, cmap="gray", origin="lower", vmin=0.8 * m, vmax=1.4 * m)
        ax.contour(sn[:, :, z].T.astype(float), levels=[0.5], colors="red", linewidths=0.7)
        ax.contour(crus[:, :, z].T.astype(float), levels=[0.5], colors="lime", linewidths=0.7)
        ax.set_title(f"template z={BOX_ORIGIN_RAS[2] + z * BOX_MM:.1f} mm", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(tdir / "template_qc.png", dpi=90)
    plt.close(fig)


def load_masks(work_dir):
    lab = np.asanyarray(nib.load(Path(work_dir) / "template" / "nm_template_masks.nii.gz").dataobj)
    return {"sn_l": lab == 1, "sn_r": lab == 2, "crus_l": lab == 3, "crus_r": lab == 4}


# ------------------------------------------------------------------------------------------ features
def template_features(nm_mni, masks, prefix="nmt_"):
    """Per subject in template space: crus mode per side, CNR = I / mode - 1, mean CNR in the template SN and in its
    anterior/posterior and medial/lateral halves; ``*_cov`` = fraction of the SN mask with data. Voxel-wise maps are
    the intended unit of analysis for group studies; a threshold volume is deliberately not reported (noise-driven)."""
    out = {}
    x = np.arange(BOX_SHAPE[0])[:, None, None] * BOX_MM + BOX_ORIGIN_RAS[0]
    y = np.arange(BOX_SHAPE[1])[None, :, None] * BOX_MM + BOX_ORIGIN_RAS[1]
    X, Y = np.broadcast_to(x, BOX_SHAPE), np.broadcast_to(y, BOX_SHAPE)
    for side in ("l", "r"):
        sn, crus = masks.get(f"sn_{side}"), masks.get(f"crus_{side}")
        if sn is None or crus is None:
            continue
        mode = _mode(nm_mni[crus])
        data = nm_mni > 0
        out[f"{prefix}crus_mode_{side}"] = mode
        out[f"{prefix}crus_cv_{side}"] = float(nm_mni[crus & data].std() / mode) if np.isfinite(mode) and (crus & data).sum() > 20 else np.nan
        out[f"{prefix}sn_cov_{side}"] = float((sn & data).mean() / max(sn.mean(), 1e-9))
        if not np.isfinite(mode) or (sn & data).sum() < 20:
            continue
        cnr = nm_mni / mode - 1.0
        m = sn & data
        out[f"{prefix}sn_{side}_cnr"] = float(cnr[m].mean())
        ymed, xmed = np.median(Y[sn]), np.median(np.abs(X[sn]))
        for name, q in (("post", m & (Y <= ymed)), ("ant", m & (Y > ymed)), ("med", m & (np.abs(X) <= xmed)), ("lat", m & (np.abs(X) > xmed))):
            out[f"{prefix}sn_{name}_{side}_cnr"] = float(cnr[q].mean()) if q.sum() >= 10 else np.nan
    for base in ("sn", "sn_post", "sn_ant", "sn_med", "sn_lat"):
        l, r = out.get(f"{prefix}{base}_l_cnr", np.nan), out.get(f"{prefix}{base}_r_cnr", np.nan)
        out[f"{prefix}{base}_mean_cnr"] = float(np.nanmean([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    l, r = out.get(f"{prefix}sn_l_cnr", np.nan), out.get(f"{prefix}sn_r_cnr", np.nan)
    out[f"{prefix}sn_min_cnr"] = float(np.nanmin([l, r])) if not (np.isnan(l) and np.isnan(r)) else np.nan
    out[f"{prefix}sn_asym_cnr"] = float(abs(l - r)) if np.isfinite(l) and np.isfinite(r) else np.nan
    return out


def _features_job(args):
    work, patno, masks = args
    f = Path(work) / str(patno) / "nm_mni.nii.gz"
    if not f.exists():
        return {"patno": patno, "error": "no nm_mni"}
    try:
        return {"patno": patno, "error": "", **template_features(np.asanyarray(nib.load(f).dataobj).astype(np.float32), masks)}
    except Exception as e:
        return {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _subjects(work, sessions, fastsurfer_dir, patnos=None):
    """(patno, fastsurfer dir) for every subject with a successful neuromelanin row."""
    from .batch import fastsurfer_by_patno

    fs = fastsurfer_by_patno(sessions, fastsurfer_dir)
    d = pd.read_csv(Path(work) / "nm_features.csv")
    d = d[d["error"].fillna("") == ""]
    keep = {int(x) for x in Path(patnos).read_text().split()} if patnos else None
    return [(int(p), fs[int(p)]) for p in d["patno"] if int(p) in fs and (keep is None or int(p) in keep)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["syn", "normalize", "template", "features"])
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--work-dir", required=True, help="the pie.imaging.nm work dir (nm_features.csv + per-subject slabs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patnos")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pid-file")
    a = ap.parse_args(argv)
    if a.pid_file:
        Path(a.pid_file).write_text(str(os.getpid()))
    work = Path(a.work_dir)
    subjects = _subjects(work, a.sessions, a.fastsurfer_dir, a.patnos)
    if a.limit:
        subjects = subjects[:a.limit]
    if a.stage == "syn":
        dirs = sorted({str(fs) for _, fs in subjects})
        todo = [d for d in dirs if not all(p.exists() for p in syn_paths(d)["fwd"])]
        print(f"{len(todo)} SyN registrations to run ({len(dirs) - len(todo)} cached)", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(_syn_job, todo), start=1):
                if r["error"] or i % 20 == 0:
                    print(f"{i}/{len(todo)} {r['fastsurfer']} {r['error'] or 'ok'}", flush=True)
        return
    if a.stage == "normalize":
        todo = [(str(work), p, str(fs)) for p, fs in subjects if not (work / str(p) / "nm_mni.nii.gz").exists()]
        print(f"{len(todo)} slabs to normalize ({len(subjects) - len(todo)} done)", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(_normalize_job, todo), start=1):
                rows.append(r)
                if r["error"] or i % 20 == 0:
                    print(f"{i}/{len(todo)} {r['patno']} {r['error'] or 'ok'}", flush=True)
        pd.DataFrame(rows).to_csv(work / "nm_normalize_log.csv", mode="a", header=not (work / "nm_normalize_log.csv").exists(), index=False)
        return
    if a.stage == "template":
        t, cnt, n = build_template(work, [p for p, _ in subjects])
        masks = template_masks(t)
        save_template(work, t, cnt, masks, n)
        template_figure(work)
        print(f"template from {n} subjects -> {work / 'template'}; " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items() if isinstance(v, np.ndarray)), flush=True)
        return
    if a.stage == "features":
        masks = load_masks(work)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_features_job, [(str(work), p, masks) for p, _ in subjects], chunksize=4))
        out = pd.DataFrame(rows)
        out.to_csv(work / "nm_template_features.csv", index=False)
        print(f"{int((out['error'] == '').sum())}/{len(out)} subjects -> {work / 'nm_template_features.csv'}", flush=True)
        return
    raise SystemExit(f"stage {a.stage} not implemented yet")


if __name__ == "__main__":
    main()
