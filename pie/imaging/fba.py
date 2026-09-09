"""
fba.py — fixel-derived measures of the nigrostriatal tract in native diffusion space (MRtrix3).

Answers the reviewer question "would a fibre-specific analysis see what the region averages miss?" without a
population template: per subject, multi-tissue response functions (Dhollander 2019), multi-shell multi-tissue CSD
(WM + GM + CSF; WM + CSF on single-shell data), `mtnormalise`, then probabilistic tractography (iFOD2) seeded in the
atlas substantia nigra and constrained to end in the FastSurfer putamen + caudate of the same hemisphere, with the
contralateral hemisphere and the cerebellum excluded. Per hemisphere:

    nst_afd_{l,r}            apparent fibre density along the tract — the sum of the AFD of every fixel the streamlines
                             traverse divided by the streamline volume (Raffelt et al. 2012, `afdconnectivity`), on the first
                             2,000 accepted streamlines so the value is comparable across subjects (it grows with count)
    nst_seed_success_{l,r}   fraction of 20,000 nigral seeds that produced a streamline reaching the striatum (density proxy)
    nst_n_streamlines_{l,r}  streamlines the AFD was computed on (2,000 unless fewer were accepted; < 500 -> no AFD)
    nst_fa_{l,r}, nst_md_{l,r}   mean FA / MD sampled along the streamlines

Requires MRtrix3 on PATH (mrconvert, dwi2response, dwi2fod, mtnormalise, tckgen, tckinfo, afdconnectivity, tcksample).
AFD is b-value dependent, so compare it within acquisition scheme (the `dwi_batch` of the manifest).

Normally run inside the DWI runner (`python -m pie.imaging.dwi ... --fba --keep-preproc`). To (re)compute the measures for
subjects whose preprocessed DWI was kept (`<work>/<patno>/fba/preproc.*` plus the `aseg_dwi`, `pauli_dwi`, `fa`, `md` maps):

    python -m pie.imaging.fba --work-dir <work> [--patnos file] [--workers 4] [--threads 2]

which rewrites the `nst_*` / `fba_*` columns of `<work>/dwi_features.csv` for those subjects.
"""

import os
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


# MRtrix3 3.0.x Python scripts import the `imp` module that Python 3.12 removed: give them a shim on PYTHONPATH
_ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).parent / "mrtrix_shim") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def _run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def write_preproc(ds, out_dir):
    """Preprocessed DWI as NIfTI + FSL-format bvec/bval + brain mask (what a later fixel or tractography pass needs)."""
    out = Path(out_dir)
    nib.save(nib.Nifti1Image(ds["data"].astype(np.float32), ds["affine"]), out / "preproc.nii.gz")
    np.savetxt(out / "preproc.bval", ds["bvals"][None], fmt="%.0f")
    np.savetxt(out / "preproc.bvec", ds["bvecs"], fmt="%.6f")
    nib.save(nib.Nifti1Image(ds["mask"].astype(np.uint8), ds["affine"]), out / "mask.nii.gz")


def nigrostriatal(ds, rois, maps, work, threads=2, n_seeds=20000, n_select=2000, min_streamlines=500, box_margin=5):
    """Nigrostriatal fixel measures for one subject. ``rois`` / ``maps`` are the (z, y, x) arrays of dwi.process_subject."""
    fb = Path(work) / "fba"
    fb.mkdir(parents=True, exist_ok=True)
    if not (fb / "preproc.nii.gz").exists() or "data" in ds:
        write_preproc(ds, fb)
    T = ["-nthreads", str(threads), "-force", "-quiet"]
    multishell = len(ds["shells"]) >= 2

    def save(name, arr, dtype=np.uint8):
        nib.save(nib.Nifti1Image(np.transpose(arr, (2, 1, 0)).astype(dtype), ds["affine"]), fb / f"{name}.nii.gz")

    # FODs are only needed along the pathway: a box around the nigra, striatum, pallidum and thalamus of both sides, dilated by
    # `box_margin` voxels (~10 mm) and cut to the brain mask, is ~5x fewer voxels than the brain, so CSD and mtnormalise run ~5x
    # faster. Response functions still come from the whole brain (they need single-fibre WM and CSF voxels).
    core = np.zeros_like(rois["sn_l"])
    for name in ("sn", "putamen", "caudate", "pallidum", "thalamus"):
        for side in ("l", "r"):
            if f"{name}_{side}" in rois:
                core |= rois[f"{name}_{side}"]
    box = np.zeros_like(core)
    idx = np.nonzero(core)
    sl = tuple(slice(max(int(i.min()) - box_margin, 0), int(i.max()) + box_margin + 1) for i in idx)
    box[sl] = True
    fod_mask = box & np.transpose(nib.load(fb / "mask.nii.gz").get_fdata() > 0, (2, 1, 0))
    save("fodmask", fod_mask)
    _run(["mrconvert", "preproc.nii.gz", "-fslgrad", "preproc.bvec", "preproc.bval", "dwi.mif", *T], fb)
    _run(["dwi2response", "dhollander", "dwi.mif", "wm.txt", "gm.txt", "csf.txt", "-mask", "mask.nii.gz", *T], fb)
    gm = ["gm.txt", "gm.mif"] if multishell else []
    _run(["dwi2fod", "msmt_csd", "dwi.mif", "wm.txt", "wmfod.mif", *gm, "csf.txt", "csf.mif", "-mask", "fodmask.nii.gz", *T], fb)
    gmn = ["gm.mif", "gm_n.mif"] if multishell else []
    _run(["mtnormalise", "wmfod.mif", "wmfod_n.mif", *gmn, "csf.mif", "csf_n.mif", "-mask", "fodmask.nii.gz", *T], fb)

    for k in ("fa", "md"):
        save(k, maps[k], np.float32)
    out = {"fba_multishell": multishell}
    for s, other in (("l", "r"), ("r", "l")):
        sn, striatum = rois[f"sn_{s}"], rois[f"putamen_{s}"] | rois[f"caudate_{s}"]
        if sn.sum() < 3 or striatum.sum() < 20:
            continue
        save(f"sn_{s}", sn)
        save(f"striatum_{s}", striatum)
        save(f"excl_{s}", rois[f"hemi_{other}"] | rois["cerebellum"])
        _run(["tckgen", "wmfod_n.mif", f"nst_{s}.tck", "-algorithm", "iFOD2", "-seed_image", f"sn_{s}.nii.gz", "-include", f"striatum_{s}.nii.gz",
              "-exclude", f"excl_{s}.nii.gz", "-stop", "-seeds", str(n_seeds), "-select", str(n_seeds), "-minlength", "15", "-maxlength", "90",
              "-cutoff", "0.05", *T], fb)
        n = int(re.search(r"count:\s*(\d+)", _run(["tckinfo", "-count", f"nst_{s}.tck"], fb)).group(1))
        out[f"nst_seed_success_{s}"] = n / n_seeds
        out[f"nst_n_streamlines_{s}"] = min(n, n_select)
        if n < min_streamlines:
            continue
        # AFD along the pathway grows with the number of streamlines (more fringe fixels are touched), so it is computed on the
        # same number of streamlines for every subject; run-to-run CV ~1 % at 2,000 (probabilistic tractography is unseeded)
        _run(["tckedit", f"nst_{s}.tck", "-number", str(n_select), f"nst_{s}_sub.tck", "-force", "-quiet"], fb)
        out[f"nst_afd_{s}"] = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", _run(["afdconnectivity", "wmfod_n.mif", f"nst_{s}_sub.tck", "-quiet"], fb))[-1])
        for k in ("fa", "md"):
            _run(["tcksample", f"nst_{s}.tck", f"{k}.nii.gz", f"{k}_{s}.txt", "-stat_tck", "mean", *T], fb)
            vals = np.loadtxt(fb / f"{k}_{s}.txt", comments="#")
            out[f"nst_{k}_{s}"] = float(np.nanmean(vals))
    for f in ("dwi.mif", "wmfod.mif", "csf.mif", "gm.mif"):       # keep the normalised FODs and tracks, drop the intermediates
        (fb / f).unlink(missing_ok=True)
    return out


def from_saved(subject_dir, threads=2):
    """Recompute the nigrostriatal measures of one subject from the saved preprocessed DWI and label maps."""
    from . import dwi as _dwi
    work = Path(subject_dir)

    def zyx(name):
        img = nib.load(work / name)
        return np.transpose(np.asarray(img.dataobj), (2, 1, 0)), img.affine

    fs_lab, aff = zyx("aseg_dwi.nii.gz")
    pauli_lab, _ = zyx("pauli_dwi.nii.gz")
    fa, _ = zyx("fa.nii.gz")
    md, _ = zyx("md.nii.gz")
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in fs_lab.shape], indexing="ij")
    ras = aff[:3, :3] @ np.stack([xx.ravel(), yy.ravel(), zz.ravel()]) + aff[:3, 3:4]
    phys_y = (-ras[1]).reshape(fs_lab.shape)                       # LPS y, as in dwi.process_subject
    rois = _dwi._roi_masks(fs_lab.astype(int), pauli_lab.astype(int), phys_y)
    left = _dwi._left_mask(fs_lab)
    rois.update({"hemi_l": left, "hemi_r": ~left, "cerebellum": np.isin(fs_lab, [7, 8, 46, 47])})
    bvals = np.loadtxt(work / "fba" / "preproc.bval").ravel()
    shells = sorted(set(int(round(b / 100.0)) * 100 for b in bvals if b > 50))
    return nigrostriatal({"shells": shells, "affine": aff}, rois, {"fa": fa, "md": md}, work, threads=threads)


def _job(args):
    subject_dir, threads = args
    row = {"patno": int(Path(subject_dir).name)}
    try:
        row.update(from_saved(subject_dir, threads))
    except Exception as e:
        row["fba_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def main(argv=None):
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    ap = argparse.ArgumentParser(description="nigrostriatal fixel measures from saved preprocessed DWI")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--patnos", help="text file of PATNOs (default: every subject with fba/preproc.nii.gz)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    subjects = sorted(p.parent for p in work.glob("*/fba/preproc.nii.gz"))
    if a.patnos:
        keep = {x.strip() for x in Path(a.patnos).read_text().split()}
        subjects = [s for s in subjects if s.name in keep]
    print(f"{len(subjects)} subjects", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(_job, [(str(s), a.threads) for s in subjects]), 1):
            rows.append(row)
            print(f"{i}/{len(subjects)} {row['patno']} {row.get('fba_error', '')}", flush=True)
    new = pd.DataFrame(rows).set_index("patno")
    csv = work / "dwi_features.csv"
    if csv.exists():
        feats = pd.read_csv(csv, low_memory=False).set_index("patno")
        for c in new.columns:
            feats.loc[new.index.intersection(feats.index), c] = new.loc[new.index.intersection(feats.index), c]
        if "fba_error" in feats and "fba_error" in new:
            feats.loc[new.index.intersection(feats.index), "fba_error"] = new["fba_error"].reindex(new.index.intersection(feats.index))
        feats.reset_index().to_csv(csv, index=False)
    else:
        new.reset_index().to_csv(csv, index=False)
    print(f"updated {csv}")


if __name__ == "__main__":
    main()
