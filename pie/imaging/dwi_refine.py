"""
Refinement pass over `pie.imaging.dwi` outputs (run with --keep-nifti): re-map the CIT168 subcortical atlas with a
deformable ANTs SyN T1 -> MNI registration instead of the affine one, and recompute the ROI features from the saved
FA/MD/FW/FAt maps. The affine-only mapping leaves the 2 mm nigral ROI contaminated by cerebral-peduncle fibres
(posterior-SN FA ~0.45) and interpeduncular CSF; a deformable mapping places the small midbrain nuclei better.

    venv_imaging/bin/python -m pie.imaging.dwi_refine --work-dir Imaging/derived/dwi --sessions Imaging/derived/sessions.csv \
        --fastsurfer-dir Imaging/derived/fastsurfer --workers 6      # -> <work-dir>/dwi_features_syn.csv
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .dwi import PAULI_ROIS, _roi_masks, features, pauli_atlas


def refine_subject(subj_dir, fastsurfer_dir, syn_type="antsRegistrationSyNQuick[s]"):
    """SyN T1->MNI (ANTs) + rigid b0->T1 (ANTs), atlas -> DWI grid, features from the saved maps. Returns a row dict."""
    import ants
    from nilearn import datasets

    subj_dir = Path(subj_dir)
    maps = {k: nib.load(subj_dir / f"{k}.nii.gz") for k in ("fa", "md", "fw", "fat")}
    b0 = ants.image_read(str(subj_dir / "b0.nii.gz"))
    aseg_dwi = np.transpose(np.asanyarray(nib.load(subj_dir / "aseg_dwi.nii.gz").dataobj).astype(np.int32), (2, 1, 0))
    mri = Path(fastsurfer_dir) / "mri"
    t1 = ants.image_read(str(mri / "orig.mgz"))
    mask = ants.image_read(str(mri / "mask.mgz"))
    t1b = t1 * ants.threshold_image(mask, 0.5, 1e9)
    mni_nib = datasets.load_mni152_template(resolution=1)
    mni_mask = datasets.load_mni152_brain_mask(resolution=1)
    mni_path = subj_dir / "_mni_brain.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(mni_nib.dataobj).astype(np.float32) * (np.asanyarray(mni_mask.dataobj) > 0), mni_nib.affine), mni_path)
    mni = ants.image_read(str(mni_path))
    atlas_nib = pauli_atlas()
    atlas_path = subj_dir / "_pauli.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(atlas_nib.dataobj).astype(np.float32), atlas_nib.affine), atlas_path)
    atlas = ants.image_read(str(atlas_path))
    rig = ants.registration(fixed=t1b, moving=b0, type_of_transform="Rigid")
    syn = ants.registration(fixed=mni, moving=t1b, type_of_transform=syn_type)
    # resample the atlas (MNI) onto the b0 grid: b0 -> T1 (rigid inverse) first, then T1 -> MNI (SyN inverse)
    tl = syn["invtransforms"] + rig["invtransforms"]
    inv = [t.endswith(".mat") for t in tl]
    warped = ants.apply_transforms(fixed=b0, moving=atlas, transformlist=tl, whichtoinvert=inv, interpolator="nearestNeighbor")
    pauli_dwi = np.transpose(warped.numpy().astype(np.int32), (2, 1, 0))  # (z, y, x) of the b0 grid
    mask_np = {k: np.transpose(np.asanyarray(v.dataobj).astype(np.float32), (2, 1, 0)) for k, v in maps.items()}
    # physical y (LPS) of every voxel for the anterior/posterior split
    aff = maps["fa"].affine
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in pauli_dwi.shape], indexing="ij")
    M, t = aff[:3, :3], aff[:3, 3]
    phys_y = -(M[1, 0] * xx + M[1, 1] * yy + M[1, 2] * zz + t[1])  # RAS y -> LPS y
    rois = _roi_masks(aseg_dwi, pauli_dwi, phys_y)
    tissue = (mask_np["fa"] < 0.5) & ~(np.nan_to_num(mask_np["fw"], nan=1.0) >= 0.7)
    for name in [k for k in rois if k.startswith(("sn", "snc", "snr", "stn", "vta", "red_nucleus"))]:
        base, side = name.rsplit("_", 1)          # "sn_posterior_l" -> "sn_posterior_t_l"
        rois[f"{base}_t_{side}"] = rois[name] & tissue
    row = {"reg_syn_mi": np.nan}
    row.update(features(mask_np, rois))
    nib.save(nib.Nifti1Image(np.transpose(pauli_dwi, (2, 1, 0)).astype(np.int16), aff), subj_dir / "pauli_dwi_syn.nii.gz")
    for p in (mni_path, atlas_path):
        p.unlink(missing_ok=True)
    return row


def _job(args):
    patno, subj_dir, fs_dir = args
    try:
        row = refine_subject(subj_dir, fs_dir)
        row.update({"patno": patno, "error": ""})
    except Exception as e:
        row = {"patno": patno, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return row


def main(argv=None):
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--patnos")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    sess = pd.read_csv(a.sessions, dtype={"image_id": str})
    fs_root = Path(a.fastsurfer_dir)
    fs_by_patno = {int(r.patno): r.image_id for r in sess.sort_values("session_date").itertuples() if (fs_root / r.image_id / "mri" / "mask.mgz").exists()}
    out_csv = work / "dwi_features_syn.csv"
    done = set(pd.read_csv(out_csv)["patno"]) if out_csv.exists() else set()
    jobs = [(int(d.name), str(d), str(fs_root / fs_by_patno[int(d.name)])) for d in sorted(work.iterdir())
            if d.is_dir() and d.name.isdigit() and (d / "fw.nii.gz").exists() and int(d.name) not in done and int(d.name) in fs_by_patno]
    if a.patnos:
        keep = {int(x) for x in Path(a.patnos).read_text().split()}
        jobs = [j for j in jobs if j[0] in keep]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} subjects to refine ({len(done)} done)", flush=True)
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), start=1):
            rows.append(fut.result())
            if i % 10 == 0 or rows[-1].get("error"):
                print(f"{i}/{len(jobs)} {rows[-1]['patno']} {'ERROR ' + rows[-1]['error'] if rows[-1].get('error') else 'ok'}", flush=True)
    new = pd.DataFrame(rows)
    if out_csv.exists():
        new = pd.concat([pd.read_csv(out_csv), new], ignore_index=True)
    new.to_csv(out_csv, index=False)
    print(f"done -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
