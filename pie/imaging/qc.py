"""
QC galleries: per-subject overlay montages of ROI contours on the images each pipeline saves (``--keep-nifti``),
so registration and segmentation can be reviewed systematically instead of by ad-hoc plotting.

    venv_imaging/bin/python -m pie.imaging.qc --work-dir Imaging/derived/dwi --modality dwi --out Imaging/derived/qc/dwi [--n 40] [--worst reg_b0_t1_mi]

Modalities and what is drawn (one PNG per subject, three orthogonal views):
  dwi    FA map with striatum (cyan), thalamus (magenta) and substantia nigra (red) contours from aseg_dwi/pauli_dwi
  nm     mean neuromelanin slab with SN (red) and reference ring (lime) contours
  flair  FLAIR resampled onto the T1 grid with the WMH mask (red)
  datscan reconstructed SPECT with the striatal labels, using the stored registration (reg_params) and the FastSurfer labels
``--worst`` sorts subjects by a QC column of the features CSV (ascending) so the poorest registrations come first;
otherwise a random sample. A contact-sheet index (gallery.png) tiles the first 16.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

VIEWS = ("axial", "coronal", "sagittal")


def _slices(img, mask):
    """Index of the centre of ``mask`` (or the image) along each axis, and the three 2-D views of an array."""
    pts = np.argwhere(mask) if mask is not None and mask.any() else np.argwhere(img > 0)
    c = pts.mean(axis=0).round().astype(int) if len(pts) else np.array(img.shape) // 2
    return c


def _view(arr, c, view):
    return {"axial": arr[:, :, c[2]].T, "coronal": arr[:, c[1], :].T, "sagittal": arr[c[0], :, :].T}[view]


def montage(base, contours, out_png, title="", zoom=None):
    """base: (x,y,z) array; contours: list of (mask, colour); zoom: half-width in voxels around the centre (None = full)."""
    centre_mask = contours[0][0] if contours else None
    c = _slices(base, centre_mask)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, view in zip(axes, VIEWS):
        sl = _view(base, c, view)
        vmax = np.percentile(sl[sl > 0], 99.5) if (sl > 0).any() else 1.0
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax, aspect="auto")
        for m, colour in contours:
            mv = _view(m, c, view)
            if mv.any():
                ax.contour(mv.astype(float), levels=[0.5], colors=colour, linewidths=0.7)
        if zoom:
            cy, cx = {"axial": (c[1], c[0]), "coronal": (c[2], c[0]), "sagittal": (c[2], c[1])}[view]
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cy - zoom, cy + zoom)
        ax.set_title(f"{title} {view}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=60)
    plt.close(fig)


def _load(p):
    return np.asanyarray(nib.load(p).dataobj)


def render_subject(modality, subj_dir, out_png, fastsurfer_dir=None, row=None):
    d = Path(subj_dir)
    if modality == "dwi":
        fa, aseg, pauli = _load(d / "fa.nii.gz"), _load(d / "aseg_dwi.nii.gz"), _load(d / "pauli_dwi.nii.gz")
        montage(fa, [(np.isin(pauli, [7, 9]), "red"), (np.isin(aseg, [11, 12, 50, 51]), "cyan"), (np.isin(aseg, [10, 49]), "magenta")], out_png, title=d.name, zoom=40)
    elif modality == "nm":
        nm, pauli, ref = _load(d / "nm_mean.nii.gz"), _load(d / "pauli_nm.nii.gz"), _load(d / "ref_nm.nii.gz")
        montage(nm, [(np.isin(pauli, [7, 9]), "red"), (ref > 0, "lime")], out_png, title=d.name, zoom=90)
    elif modality == "flair":
        fl, wmh = _load(d / "flair_t1.nii.gz"), _load(d / "wmh_t1.nii.gz")
        montage(fl, [(wmh > 0, "red")], out_png, title=d.name)
    elif modality == "datscan":
        import SimpleITK as sitk

        from .datscan import _sitk_from_nib, transform_from_row

        spect = nib.load(d)
        if row is None or fastsurfer_dir is None:
            raise ValueError("datscan galleries need the features row (reg_params/reg_center) and the FastSurfer directory")
        aparc = nib.load(Path(fastsurfer_dir) / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
        tx = transform_from_row(row)
        moving = _sitk_from_nib(spect)
        labels = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc.dataobj).astype(np.float32), aparc.affine))
        lab = np.transpose(sitk.GetArrayFromImage(sitk.Resample(labels, moving, tx.GetInverse(), sitk.sitkNearestNeighbor, 0.0)), (2, 1, 0))
        base = np.transpose(sitk.GetArrayFromImage(moving), (2, 1, 0))
        montage(base, [(np.isin(lab, [11, 12, 50, 51]), "cyan"), (lab > 0, "lime")], out_png, title=Path(d).name.split("_")[0], zoom=45)
    else:
        raise ValueError(modality)


def gallery(pngs, out_png, cols=4):
    pngs = [p for p in pngs if Path(p).exists()][:16]
    if not pngs:
        return
    rows = int(np.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.8 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, p in zip(np.ravel(axes), pngs):
        ax.imshow(plt.imread(p))
        ax.set_title(Path(p).stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--modality", required=True, choices=["dwi", "nm", "flair", "datscan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--worst", help="features column to sort ascending (poorest first), e.g. reg_b0_t1_mi")
    ap.add_argument("--patnos", help="text file of PATNOs to render")
    ap.add_argument("--sessions", help="sessions.csv (datscan only)")
    ap.add_argument("--fastsurfer-dir", help="FastSurfer root (datscan only)")
    a = ap.parse_args(argv)
    work, out = Path(a.work_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats_name = {"dwi": "dwi_features.csv", "nm": "nm_features.csv", "flair": "flair_features.csv", "datscan": "datscan_sbr.csv"}[a.modality]
    feats = pd.read_csv(work / feats_name)
    feats = feats[feats["error"].fillna("") == ""]
    if a.patnos:
        keep = {int(x) for x in Path(a.patnos).read_text().split()}
        feats = feats[feats["patno"].isin(keep)]
    if a.worst and a.worst in feats:
        feats = feats.sort_values(a.worst)
    else:
        feats = feats.sample(frac=1.0, random_state=0)
    fs_by_patno = {}
    if a.modality == "datscan":
        sess = pd.read_csv(a.sessions, dtype={"image_id": str})
        fs_by_patno = {int(r.patno): Path(a.fastsurfer_dir) / r.image_id for r in sess.sort_values("session_date").itertuples()}
    done = []
    for row in feats.head(a.n).itertuples():
        r = row._asdict()
        png = out / f"{r['patno']}.png"
        try:
            if a.modality == "datscan":
                render_subject("datscan", str(work / "nifti" / f"{r['image_id']}_datscan.nii.gz"), png, fastsurfer_dir=fs_by_patno.get(int(r["patno"])), row=r)
            else:
                render_subject(a.modality, work / str(r["patno"]), png)
            done.append(png)
        except Exception as e:  # subjects without saved NIfTIs
            print(f"{r['patno']}: {type(e).__name__}: {str(e)[:100]}")
    gallery(done, out / "gallery.png")
    print(f"{len(done)} montages -> {out}")


if __name__ == "__main__":
    main()
