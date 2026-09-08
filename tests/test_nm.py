"""Unit tests for the neuromelanin module: ROI construction and the contrast / thresholded-volume features on a phantom."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import nm


def _phantom():
    shape = (16, 60, 60)  # (z, y, x) slab
    pauli = np.zeros(shape, dtype=np.int32)
    code = {n: i + 1 for i, n in enumerate(nm.PAULI)}
    pauli[6:10, 26:34, 14:20] = code["SNc"]     # x-low side
    pauli[6:10, 26:34, 40:46] = code["SNr"]     # x-high side
    pauli[6:10, 22:26, 27:33] = code["RN"]
    fs = np.zeros(shape, dtype=np.int32)
    fs[:, 10:50, 5:55] = 16                     # brainstem everywhere around
    fs[2:5, 40:50, 5:15] = 12                   # left putamen marker (x-low = left)
    fs[2:5, 40:50, 45:55] = 51
    return shape, pauli, fs, code


def test_rois_and_features_recover_planted_contrast():
    shape, pauli, fs, code = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, 0.5, 0.5))
    assert rois["sn_l"].sum() == 4 * 8 * 6 and rois["sn_r"].sum() == 4 * 8 * 6
    assert rois["ref"].any() and not (rois["ref"] & (pauli > 0)).any()
    assert rois["search_l"].sum() > rois["sn_l"].sum()
    rng = np.random.default_rng(0)
    img = np.full(shape, 100.0, dtype=np.float32) + rng.normal(0, 2, shape).astype(np.float32)
    img[rois["sn_l"]] = 130.0      # 30 % contrast on the left
    img[rois["sn_r"]] = 110.0      # 10 % on the right
    phys_y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    out = nm.features(img, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), smooth_fwhm_mm=0.0)
    assert abs(out["nm_sn_l_cnr"] - 0.30) < 0.02 and abs(out["nm_sn_r_cnr"] - 0.10) < 0.02
    assert out["n_ref_l"] >= 20 and out["n_ref_r"] >= 20
    assert abs(out["nm_sn_l_cnr_atlas"] - 0.30) < 0.02 and out["nm_sn_shift_mm_l"] < 0.6   # band on the atlas: no shift needed
    assert abs(out["nm_sn_asym_cnr"] - 0.20) < 0.03
    assert np.isfinite(out["nm_sn_posterior_l_cnr"]) and np.isfinite(out["nm_sn_mean_cnr"])


def test_features_do_not_invent_contrast_from_noise():
    """A slab with no neuromelanin band: every contrast measure must sit near zero. Order statistics on noisy voxels
    (brightest-fraction, fixed-threshold volumes) report positive contrast that only tracks the noise level."""
    shape, pauli, fs, code = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, 0.5, 0.5))
    rng = np.random.default_rng(1)
    img = np.full(shape, 100.0, dtype=np.float32) + rng.normal(0, 13, shape).astype(np.float32)   # CV 0.13 as in PPMI slabs
    phys_y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    out = nm.features(img, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5))
    bad = {k: round(v, 3) for k, v in out.items() if "cnr" in k and np.isfinite(v) and abs(v) > 0.03}
    assert not bad, bad


def test_refined_position_recovers_a_band_offset_from_the_atlas():
    """The bright band lies 1.5 mm lateral of the atlas SN (typical affine placement error): the refined CNR must recover
    the planted 20 % contrast while the unrefined atlas CNR is diluted."""
    shape, pauli, fs, code = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, 0.5, 0.5))
    rng = np.random.default_rng(2)
    img = np.full(shape, 100.0, dtype=np.float32) + rng.normal(0, 3, shape).astype(np.float32)
    band_l = np.roll(rois["sn_l"], -3, axis=2)         # 3 voxels = 1.5 mm lateral (x-low side moves to lower x)
    band_r = np.roll(rois["sn_r"], 3, axis=2)
    img[band_l] = 120.0
    img[band_r] = 120.0
    phys_y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    out = nm.features(img, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5))
    assert abs(out["nm_sn_l_cnr"] - 0.20) < 0.04 and abs(out["nm_sn_r_cnr"] - 0.20) < 0.04, (out["nm_sn_l_cnr"], out["nm_sn_r_cnr"])
    assert out["nm_sn_l_cnr_atlas"] < 0.12 and out["nm_sn_r_cnr_atlas"] < 0.12
    assert abs(out["nm_sn_shift_mm_l"] - 1.5) < 0.6 and abs(out["nm_sn_shift_mm_r"] - 1.5) < 0.6
