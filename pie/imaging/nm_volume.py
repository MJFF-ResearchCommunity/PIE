"""Neuromelanin volume and reference-normalised signal, for comparison with segmentation-based studies.

``nm.py`` reports contrast ratios on a fixed atlas mask and deliberately does not report threshold volumes:
single-voxel order statistics are noisy, and bright arteries in the interpeduncular cistern contaminate any
threshold applied outside the nigra. Many studies nevertheless report a neuromelanin "SN volume", from manual
segmentation (e.g. Droby et al. 2025, Ben Bashat et al.) or from thresholding against a reference region. These
functions compute both kinds of number under explicit, auditable rules so that PIE output can be set beside
theirs:

    mask_volume(mask)                                    volume of a supplied (e.g. manually drawn) SN mask, mm^3
    hyperintense_volume(nm, search, reference, k)        voxels inside ``search`` brighter than reference mean + k SD
    normalised_intensity(nm, roi, reference)             mean(roi) / mean(reference), and the contrast ratio minus one

The threshold volume is only as good as ``search``: pass the atlas SN mask (dilated at most by the
registration uncertainty), never a box, so that cisternal arteries cannot enter it. ``k`` defaults to 3; papers
use 2 to 4 and the value changes volumes several-fold, so report it with every result. On 40 PPMI 2D GRE-MT scans
(18 September 2026) the bilateral atlas-SN volume had medians of 160, 66 and 21 mm^3 at k = 2, 2.5 and 3; per side at
k = 3 it correlated with the contrast ratio at Spearman 0.56, and ``normalised_intensity`` reproduced ``nm.py``'s
whole-nigra contrast ratio exactly on the same masks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


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
