"""
nm_native.py — neuromelanin measured on each subject's own averaged slab (native space). Regions are defined once in
MNI152NLin2009cAsym and pulled onto the slab grid in a single nearest-label resampling (slab -> T1 rigid, T1 -> MNI
through the inverse of the cached ANTs SyN), so the slab itself is never interpolated. Two published measures:

    langley  SNc volume: voxels of the nigral search region brighter than the crus-cerebri mean + 2.8 SD, on
             MP-PCA-denoised repeats (``nm --denoise``). Langley et al. 2025, npj Parkinson's Disease 11:181,
             doi:10.1038/s41531-025-00976-3 (PPMI); Hwang, Langley et al. 2023, PLOS ONE 18:e0282684.
    snceg    the neuromelanin-hyperintense SN segmented by the public snceg Attention U-Net (MIT licence, weights
             pinned), its volume and contrast (CR, CNR) against the crus cerebri. Lillebostad et al. 2025, Imaging
             Neuroscience, doi:10.1162/IMAG.a.158.

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


def pull_labels(label_img, grid_img, tx_t1_grid, warp, affine, codes=None, pad=12):
    """``label_img`` (MNI space) on ``grid_img``'s voxels in one genericLabel resampling: grid -> T1 by the inverse of
    ``tx_t1_grid`` (SimpleITK, T1 point -> grid point, as ``nm.register_slab`` returns), T1 -> MNI by the inverse of
    the SyN (``warp``, ``affine``: the forward ``ants.registration`` pair, fixed = MNI). Returns (labels on the grid,
    coverage): for each name in ``codes`` ({name: [label values]}), the share of that region the grid holds, from the
    same pull onto the grid padded by ``pad`` slices along its thinnest physical extent (the slab normal)."""
    import ants

    axis = int(np.argmin(np.array(grid_img.shape[:3]) * np.linalg.norm(grid_img.affine[:3, :3], axis=0)))   # the slab's thin axis
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
                                    whichtoinvert=[False, True, False], interpolator="genericLabel")
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


def langley_features(nm_img, codes, k=LANGLEY_K, prefix="nml_"):
    """SNc volume (mm^3, total and per side): search-region voxels (nigra + 1 mm ring) above crus mean + k SD
    (``nm.hyperintense_volume``); one threshold from both crus parts, as the published cerebral-peduncle reference."""
    from .nm import hyperintense_volume

    region, sides = _side_masks(codes)
    search = np.isin(region, [1, 2, 3, RING])
    ref = nib.Nifti1Image((region == CRUS).astype(np.uint8), nm_img.affine)
    out = {}
    for name, m in (("", search), ("_l", search & sides["l"]), ("_r", search & sides["r"])):
        r = hyperintense_volume(nm_img, nib.Nifti1Image(m.astype(np.uint8), nm_img.affine), ref, k=k)
        out[f"{prefix}sn_volume{name}_mm3"] = r["volume_mm3"]
    out.update({f"{prefix}threshold": r["threshold"], f"{prefix}ref_mean": r["reference_mean"], f"{prefix}ref_sd": r["reference_sd"],
                f"{prefix}n_ref": r["n_reference"], f"{prefix}k": float(k)})
    return out


def snceg_features(nm_img, sn, codes, prefix="nms_"):
    """Volume (mm^3), contrast ratio (I_sn / I_crus - 1) and CNR ((I_sn - I_crus) / SD_crus) of a segmented SN, per
    side and bilateral; the crus reference excludes segmented voxels (Lillebostad et al. 2025)."""
    nm = np.asarray(nm_img.dataobj, float)
    region, sides = _side_masks(codes)
    vox = float(abs(np.linalg.det(nm_img.affine[:3, :3])))
    ref = (region == CRUS) & ~sn & np.isfinite(nm)
    mu, sd = (float(nm[ref].mean()), float(nm[ref].std())) if ref.sum() >= 20 else (np.nan, np.nan)
    out = {f"{prefix}sn_volume_mm3": float(sn.sum() * vox), f"{prefix}crus_mean": mu, f"{prefix}crus_sd": sd}
    for s, m in sides.items():
        v = sn & m
        i = float(nm[v].mean()) if v.any() else np.nan
        out.update({f"{prefix}sn_volume_{s}_mm3": float(v.sum() * vox), f"{prefix}sn_{s}_cr": i / mu - 1, f"{prefix}sn_{s}_cnr": (i - mu) / sd})
    for m in ("cr", "cnr"):
        out[f"{prefix}sn_mean_{m}"] = (out[f"{prefix}sn_l_{m}"] + out[f"{prefix}sn_r_{m}"]) / 2
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
    niis = [f for r in series_rows for f in _nm.convert(r["zip"], r["prefix"], d / "nii")]
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
def process_subject(work_dir, patno, fastsurfer_dir, snceg=False, min_cov=0.9):
    """<work>/<patno>/nm_mean.nii.gz + slab_to_t1.tfm (``nm --keep-nifti``) and the cached SyN (``nm_template syn``)
    -> native region labels (saved), coverage, Langley volume and optionally snceg measures. A side whose search region
    or crus is less than ``min_cov`` inside the slab is missing."""
    import SimpleITK as sitk

    from .nm_template import syn_cached, syn_paths

    d = Path(work_dir) / str(patno)
    if not syn_cached(fastsurfer_dir):
        raise FileNotFoundError("no SyN cache; run nm_template syn first")
    nm_img = nib.load(d / "nm_mean.nii.gz")
    warp, affine = syn_paths(fastsurfer_dir)["fwd"]
    groups = {"search_l": [101, 102, 103, 100 + RING], "search_r": [201, 202, 203, 200 + RING], "crus_l": [100 + CRUS], "crus_r": [200 + CRUS]}
    codes, cov = pull_labels(region_labels(), nm_img, sitk.ReadTransform(str(d / "slab_to_t1.tfm")), warp, affine, codes=groups)
    nib.save(nib.Nifti1Image(codes, nm_img.affine), d / "native_regions_nm.nii.gz")
    row = {f"native_cov_{k}": v for k, v in cov.items()}
    ok = {s: cov[f"search_{s}"] >= min_cov and cov[f"crus_{s}"] >= min_cov for s in "lr"}
    row.update(langley_features(nm_img, codes))
    if snceg:
        sn = snceg_mask(d / "nm_mean.nii.gz", d / "snceg_sn_nm.nii.gz")
        row.update(snceg_features(nm_img, sn, codes))
    for key in list(row):
        side = "l" if "_l_" in key or key.endswith("_l_mm3") else "r" if "_r_" in key or key.endswith("_r_mm3") else None
        if key.startswith(("nml_sn", "nms_sn")) and ((side and not ok[side]) or (side is None and not all(ok.values()))):
            row[key] = np.nan
    return row


def _job(args):
    work, patno, fs_dir, snceg = args
    try:
        return {"patno": patno, "error": "", **process_subject(work, patno, fs_dir, snceg=snceg)}
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
    ap.add_argument("--prepare-tse", action="store_true", help="first stage for dual-echo TSE input: convert, pick the echo, "
                    "register; writes <work>/nm_features.csv (then run nm_template syn and this CLI without the flag)")
    ap.add_argument("--zips", nargs="+", help="with --prepare-tse: the PPMI zips holding the TSE series")
    ap.add_argument("--echo", type=int, default=1, help="with --prepare-tse: 1 = proton-density (shortest TE), 2 = T2")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    if a.prepare_tse:
        from .batch import fastsurfer_by_patno, filter_jobs, load_index, run_batch, session_rows

        idx = load_index(work / "tse_index.csv", a.zips, flag_tse)
        idx = idx[idx["selected"]]
        fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
        out_csv = work / "nm_features.csv"
        done = set(pd.read_csv(out_csv)["patno"]) if out_csv.exists() and out_csv.stat().st_size else set()
        jobs = [(int(p), session_rows(g), fs[int(p)], str(work), a.echo) for p, g in idx.groupby("patno") if p not in done and int(p) in fs]
        run_batch(filter_jobs(jobs, a.patnos, a.limit), _tse_job, out_csv, workers=a.workers)
        return
    subjects = _subjects(work, a.sessions, a.fastsurfer_dir, a.patnos)
    region_labels()                    # build the atlas bridge once before the workers start
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, [(str(work), p, str(fs), a.snceg) for p, fs in subjects]))
    out = pd.DataFrame(rows)
    out.to_csv(work / "nm_native_features.csv", index=False)
    print(f"{int((out['error'] == '').sum())}/{len(out)} subjects -> {work / 'nm_native_features.csv'}", flush=True)


if __name__ == "__main__":
    main()
