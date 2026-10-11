"""Montage rendering on synthetic arrays (no data needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.qc import gallery, montage


def test_montage_and_gallery(tmp_path):
    base = np.zeros((30, 40, 20), dtype=np.float32)
    base[5:25, 8:32, 4:16] = 1.0
    mask = np.zeros_like(base, dtype=bool)
    mask[12:18, 18:24, 8:12] = True
    pngs = []
    for i in range(2):
        p = tmp_path / f"s{i}.png"
        montage(base, [(mask, "red")], p, title=f"s{i}", zoom=15)
        assert p.exists() and p.stat().st_size > 1000
        pngs.append(p)
    gallery(pngs, tmp_path / "gallery.png")
    assert (tmp_path / "gallery.png").exists()


def test_t1_gallery_puts_the_robust_outliers_first(tmp_path):
    import nibabel as nib
    import pandas as pd

    from pie.imaging.qc import main

    rng = np.random.default_rng(0)
    n = 25
    rows = {"PATNO": range(n), "IMAGEID": [f"I{i}" for i in range(n)], "Manufacturer": ["GE"] * n, "MagneticFieldStrength": [3] * n,
            **{f"vol_{s}_{r}": rng.normal(5000, 100, n) for s in ("Left", "Right") for r in ("Putamen", "Caudate", "Thalamus")}}
    rows["vol_Left_Thalamus"][7] = 900
    pd.DataFrame(rows).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    mri = tmp_path / "fastsurfer" / "I7" / "mri"
    mri.mkdir(parents=True)
    seg = np.zeros((20, 20, 20), np.int16)
    seg[5:9, 5:9, 5:9], seg[11:15, 5:9, 5:9], seg[2:18, 2:18, 2:4] = 12, 51, 1003
    nib.save(nib.MGHImage(np.ones((20, 20, 20), np.float32), np.eye(4)), mri / "orig.mgz")
    nib.save(nib.MGHImage(seg, np.eye(4)), mri / "aparc.DKTatlas+aseg.deep.mgz")
    main(["--work-dir", str(tmp_path), "--modality", "t1", "--out", str(tmp_path / "qc"), "--n", "1", "--worst", "t1_robust_z_max"])
    assert (tmp_path / "qc" / "I7.png").exists()
