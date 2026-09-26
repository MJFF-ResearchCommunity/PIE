"""Unit tests for the neuromelanin module: ROI construction and the contrast / thresholded-volume features on a phantom."""

import sys
from pathlib import Path

import nibabel as nib
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


def test_independent_noise_does_not_acquire_contrast_from_localized_masks():
    """Test independent measurements, not a same-image maximum's expected zero.

    Localization on noise can select a positive contrast, especially for small
    subregions. Averaging independent noise in those fixed masks must be unbiased.
    """
    shape, pauli, fs, code = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, 0.5, 0.5))
    rng = np.random.default_rng(1)
    img = np.full(shape, 100.0, dtype=np.float32) + rng.normal(0, 13, shape).astype(np.float32)   # CV 0.13 as in PPMI slabs
    phys_y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    masks = {}
    nm.features(img, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), mask_out=masks)
    contrasts = []
    for _ in range(100):
        independent = 100 + rng.normal(0,13,shape)
        reference = independent[masks['ref']].mean()
        contrasts.append([(independent[masks['sn_'+side]].mean()-reference)/reference for side in ['l','r']])
    assert np.max(np.abs(np.mean(contrasts,axis=0))) < .003


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


def test_reference_excludes_both_final_refined_masks():
    shape, pauli, fs, _ = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, .5, .5))
    img = np.full(shape, 100., np.float32)
    img[np.roll(rois['sn_l'], -3, axis=2)] = 125.
    img[np.roll(rois['sn_r'], 3, axis=2)] = 115.
    masks = {}
    y = np.broadcast_to(np.arange(shape[1])[None,:,None], shape)
    values = nm.features(img, rois, y, mask_out=masks)
    sn = masks['sn_l'] | masks['sn_r']
    assert (rois['ref'] & sn).any(), 'fixture must expose the previous overlap bug'
    assert not (masks['ref'] & sn).any()
    assert values['n_ring'] == masks['ref'].sum()
    assert values['nm_ring_mean'] == img[masks['ref']].mean()
    assert abs(values['nm_sn_l_cnr'] - .25) < .02


def test_mppca_denoising_of_repeats_lowers_noise_and_keeps_nigral_contrast(tmp_path):
    import json
    from scipy import ndimage
    rng = np.random.default_rng(0)
    truth = (100 + 30 * ndimage.gaussian_filter(rng.normal(size=(40, 40, 12)), 2)).astype(np.float32)
    band = np.zeros(truth.shape, bool)
    band[15:25, 15:25, 3:9] = True
    truth[band] += 25                                                      # a bright nigra-like block
    aff = np.diag([0.5, 0.5, 2.0, 1.0])
    paths = []
    for i in range(5):                                                     # PPMI GRE-MT: 5 or 10 measurements
        p = tmp_path / f"rep{i}.nii.gz"
        nib.save(nib.Nifti1Image(truth + rng.normal(0, 15, truth.shape).astype(np.float32), aff), p)
        p.with_suffix("").with_suffix(".json").write_text(json.dumps({"AcquisitionTime": f"10:0{i}:00", "AcquisitionNumber": i + 1}))
        paths.append(p)
    plain = np.asarray(nm.average_repeats(paths, sampling_seed=1)[0].dataobj)
    img, meta, n, _ = nm.average_repeats(paths, sampling_seed=1, denoise=True)
    den = np.asarray(img.dataobj)
    inner = (slice(14, 26), slice(14, 26), slice(2, -2))                  # inside the denoised centre (_mppca_centre)
    rmse = lambda x: float(np.sqrt(((x - truth)[inner] ** 2).mean()))
    contrast = lambda x: float(x[band].mean() - x[ndimage.binary_dilation(band, iterations=3) & ~band].mean())
    assert n == 5 and meta["PIEDenoise"] == "mppca" and rmse(den) < 0.9 * rmse(plain)   # ~0.8 around a bright structure
    assert abs(contrast(den) / contrast(truth) - 1) < 0.1                  # denoising must not flatten the band


def test_mppca_on_the_central_region_equals_whole_slab_mppca_there():
    from dipy.denoise.localpca import mppca
    rng = np.random.default_rng(2)
    stack = (100 + rng.normal(0, 15, (48, 40, 8, 5))).astype(np.float32)
    full = mppca(stack, patch_radius=2, suppress_warning=True)
    part = nm._mppca_centre(stack)
    inner = (slice(12 + 4, 36 - 4), slice(10 + 4, 30 - 4))   # the central half in-plane, two patch radii off its edges
    assert np.allclose(part[inner], full[inner], atol=1e-3)
    assert np.array_equal(part[:10], stack[:10]) and np.array_equal(part[:, :8], stack[:, :8])   # outside: untouched


def test_scanner_clipped_repeats_are_measured_and_fail_nm_qc(tmp_path):
    import pandas as pd

    from pie.imaging.manifest import QC

    rng = np.random.default_rng(3)
    aff = np.diag([0.5, 0.5, 1.5, 1.0])
    clean, clipped = [], []
    for i in range(3):                       # 12-bit magnitude; the clipped scan saturates ~30 % of the slab at 4095
        img = rng.normal(1400, 300, (40, 40, 6)).astype(np.float32)
        img[:, :14] += 3000
        for paths, arr in ((clean, img), (clipped, np.minimum(img, 4095))):
            p = tmp_path / f"{'clip' if arr is not img else 'ok'}_{i}.nii.gz"
            nib.save(nib.Nifti1Image(arr.round().astype(np.int16), aff), p)
            paths.append(p)
    ok = nm.average_repeats(clean, sampling_seed=1)[1]["PIEClippedFraction"]
    bad = nm.average_repeats(clipped, sampling_seed=1)[1]["PIEClippedFraction"]
    assert ok < 0.001 and bad > 0.25                    # ~84 % of the raised third clips
    row = {"n_sn_l": 50, "n_sn_r": 50, "sn_slab_coverage": 1.0, "repeat_motion_mm_max": 0.5,
           "nm_ref_l_sd": 10, "nm_ref_l_mean": 100, "nm_ref_r_sd": 10, "nm_ref_r_mean": 100}
    d = pd.DataFrame([{**row, "nm_clipped_fraction": ok}, {**row, "nm_clipped_fraction": bad}, row])   # last: legacy table
    assert QC["nm"](d).tolist() == [True, False, True]
