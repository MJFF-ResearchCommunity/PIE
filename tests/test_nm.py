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
    img, meta, n, *_ = nm.average_repeats(paths, sampling_seed=1, denoise=True)
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


def test_process_subject_reads_nifti_series_without_a_zip(tmp_path, monkeypatch):
    import pytest

    seen = {}

    def stop(niis, **kw):
        seen["niis"] = niis
        raise RuntimeError("stop after the inputs")

    monkeypatch.setattr(nm, "average_repeats", stop)
    monkeypatch.setattr(nm, "convert", lambda *a: pytest.fail("a NIfTI series must not be converted"))
    with pytest.raises(RuntimeError, match="stop after the inputs"):
        nm.process_subject(1, [{"nifti": "/data/sub-01_mt-GRE.nii.gz", "date": "2000-01-01", "desc": "mt-GRE"}], tmp_path / "fs", tmp_path)
    assert seen["niis"] == ["/data/sub-01_mt-GRE.nii.gz"]


def test_a_subjects_series_read_the_zip_index_once(tmp_path, monkeypatch):
    import zipfile

    import pytest

    zp = tmp_path / "loni.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for i in (1, 2):                    # PPMI stores each NM repeat as its own series
            z.writestr(f"PPMI/1/NM/2000-01-01_09_00_00.0/I00000{i}/a.dcm", b"x")
    real, opened = zipfile.ZipFile, []

    def counting(*a, **k):
        if a[0] == zp or str(a[0]) == str(zp):
            opened.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(zipfile, "ZipFile", counting)
    monkeypatch.setattr(nm, "average_repeats", lambda niis, **kw: (_ for _ in ()).throw(RuntimeError("stop after conversion")))
    rows = [{"zip": str(zp), "prefix": f"PPMI/1/NM/2000-01-01_09_00_00.0/I00000{i}/", "date": "2000-01-01", "desc": "NM"} for i in (1, 2)]
    with pytest.raises(RuntimeError, match="stop after conversion"):
        nm.process_subject(1, rows, tmp_path / "fs", tmp_path / "work")
    assert len(opened) == 1


# ------------------------------------------------------------------------------------------ study ports (27 Sep 2026)
def _grid_points(grid, idx):
    """LPS mm of (x, y, z) voxel indices of a SimpleITK grid."""
    return np.array([grid.TransformContinuousIndexToPhysicalPoint(tuple(map(float, i))) for i in idx])


def _cube(x0, y0=15, z0=5, n=6):
    return np.stack(np.meshgrid(np.arange(x0, x0 + n), np.arange(y0, y0 + n), np.arange(z0, z0 + n), indexing="ij"), -1).reshape(-1, 3)


def _shift(grid, di):
    """T1 point -> slab point transform that moves every point by ``di`` slab voxels."""
    import SimpleITK as sitk

    v = np.array(grid.TransformContinuousIndexToPhysicalPoint(tuple(map(float, di)))) - np.array(grid.GetOrigin())
    return sitk.TranslationTransform(3, tuple(v))


SLAB_AFF = np.diag([0.5, 0.5, 1.5, 1.0])


def test_slab_coverage_is_the_bounded_share_of_each_sides_sn_inside_the_slab():
    """The study's geometric rule: the share of a side's T1-space SN points that the rigid maps inside the slab. The
    old volume ratio exceeded 1 on fine slabs and could pass a slab that missed one side."""
    grid = nm._sitk_native(np.zeros((40, 40, 16), np.float32), SLAB_AFF)
    pts = _grid_points(grid, _cube(4, z0=12))                       # slices 12-17 of 0-15: 4 of 6 inside
    assert nm.slab_coverage(pts, _shift(grid, (0, 0, 0)), grid) == 4 / 6
    assert nm.slab_coverage(pts, _shift(grid, (0, 0, -2)), grid) == 1.0   # the transform maps T1 points into the slab
    assert nm.slab_coverage(pts, _shift(grid, (0, 0, 20)), grid) == 0.0
    assert nm.slab_coverage(pts, _shift(grid, (0, 0, 0.6)), grid) == 3 / 6   # a slice reaches half a voxel past its centre
    assert nm.slab_coverage(np.empty((0, 3)), _shift(grid, (0, 0, 0)), grid) == 0.0


def test_register_slab_fits_both_starts_and_keeps_the_best_fit_that_covers_both_sides(tmp_path, monkeypatch):
    import SimpleITK as sitk

    mri = tmp_path / "mri"
    mri.mkdir()
    for name in ("orig.mgz", "mask.mgz"):
        nib.save(nib.MGHImage(np.ones((8, 8, 8), np.float32), np.eye(4)), mri / name)
    slab = nib.Nifti1Image(np.zeros((40, 40, 16), np.float32), SLAB_AFF)
    grid = nm._sitk_native(np.asarray(slab.dataobj), slab.affine)
    sides = {1: _grid_points(grid, _cube(4)), 2: _grid_points(grid, _cube(30))}   # right SN near the slab's x edge
    monkeypatch.setattr(nm, "_sn_points_in_t1", lambda t1, fs: sides)   # the SyN mapping has its own test; no affine is fitted

    def fits(header, centroid):
        calls = []

        def fake(nm_img, t1, target_center=None, sampling_seed=0):
            calls.append(target_center is None)
            return fits_by_start[target_center is None]
        fits_by_start = {True: header, False: centroid}
        monkeypatch.setattr(nm, "register_nm_to_t1", fake)
        tx, metric, init, cov = nm.register_slab(slab, tmp_path)
        assert calls == [True, False], "both starts are always fitted"
        return metric, init, cov

    # the header fit has the better metric but moves the right SN 6 voxels, 2 of 6 columns out of the slab
    metric, init, cov = fits((_shift(grid, (6, 0, 0)), -0.9), (_shift(grid, (0, 0, 0)), -0.5))
    assert (init, metric, cov) == ("sn_centroid", -0.5, (1.0, 1.0))
    # both cover: the lower MI metric wins, even from the second start
    metric, init, cov = fits((_shift(grid, (0, 0, 0)), -0.5), (_shift(grid, (0, 0, 1)), -0.7))
    assert (init, metric) == ("sn_centroid", -0.7)
    # neither covers: the better-covered fit wins over the better metric
    metric, init, cov = fits((_shift(grid, (8, 0, 0)), -0.9), (_shift(grid, (6, 0, 0)), -0.5))
    assert init == "sn_centroid" and cov == (1.0, 4 / 6)


def test_average_repeats_reports_where_every_repeat_contributes(tmp_path):
    import json
    from scipy import ndimage

    rng = np.random.default_rng(4)
    big = (100 + 400 * ndimage.gaussian_filter(rng.normal(size=(48, 40, 10)), 2)).astype(np.float32)   # texture sd ~20
    paths = []
    for i, x0 in enumerate((4, 4, 0)):                     # the third repeat is displaced 2 mm (4 voxels) in x
        p = tmp_path / f"rep{i}.nii.gz"
        nib.save(nib.Nifti1Image(big[x0:x0 + 40] + rng.normal(0, 2, (40, 40, 10)).astype(np.float32), np.diag([0.5, 0.5, 2.0, 1.0])), p)
        p.with_suffix("").with_suffix(".json").write_text(json.dumps({"AcquisitionTime": f"10:0{i}:00", "AcquisitionNumber": i + 1}))
        paths.append(p)
    img, meta, n, motion, support = nm.average_repeats(paths, sampling_seed=1)
    assert support.shape == img.shape and n == 3 and 1.5 < motion < 2.5
    assert (support[5:35] > 0.99).all()                    # every repeat holds the centre
    assert (support[36:] < 0.99).all()                     # the displaced repeat's field of view ends 4 voxels early


def test_nm_voxels_missing_a_repeat_or_signal_fail_nm_qc():
    import pandas as pd

    from pie.imaging.manifest import QC

    shape, pauli, fs, code = _phantom()
    rois = nm.nm_rois(fs, pauli, (1.5, 0.5, 0.5))
    img = np.full(shape, 100.0, np.float32) + np.random.default_rng(5).normal(0, 2, shape).astype(np.float32)
    img[rois["sn_l"]] = img[rois["sn_r"]] = 125.0
    y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    base = {"n_sn_l": 50, "n_sn_r": 50, "sn_slab_coverage_l": 1.0, "sn_slab_coverage_r": 1.0, "repeat_motion_mm_max": 0.5,
            "nm_ref_l_sd": 10, "nm_ref_l_mean": 100, "nm_ref_r_sd": 10, "nm_ref_r_mean": 100}
    good = nm.features(img, rois, y, spacing_zyx=(1.5, 0.5, 0.5), support=np.ones(shape))
    assert good["nm_support_l"] == good["nm_support_r"] == good["nm_support_ref"] == 1.0 and good["nm_valid_ref"] == 1.0
    partial = np.ones(shape)
    partial[6:8] = 0.8                                     # half the SN's slices lack one of five repeats
    edge = nm.features(img, rois, y, spacing_zyx=(1.5, 0.5, 0.5), support=partial)
    assert 0.4 < edge["nm_support_l"] < 0.6
    holed = img.copy()
    holed[:, :, 31::2] = 0                                 # signal dropouts in every other column of the right half:
    empty = nm.features(holed, rois, y, spacing_zyx=(1.5, 0.5, 0.5), support=np.ones(shape))   # no refined position escapes them
    assert 0.4 < empty["nm_valid_r"] < 0.6 and empty["nm_valid_l"] == 1.0
    d = pd.DataFrame([{**base, **o} for o in (good, edge, empty)])
    assert QC["nm"](d).tolist() == [True, False, False]


def test_nm_qc_requires_each_side_inside_the_slab_and_keeps_the_old_bar_for_old_rows():
    import pandas as pd

    from pie.imaging.manifest import QC

    base = {"n_sn_l": 50, "n_sn_r": 50, "repeat_motion_mm_max": 0.5,
            "nm_ref_l_sd": 10, "nm_ref_l_mean": 100, "nm_ref_r_sd": 10, "nm_ref_r_mean": 100}
    new = pd.DataFrame([{**base, "sn_slab_coverage_l": l, "sn_slab_coverage_r": r} for l, r in ((0.95, 0.92), (0.95, 0.85))])
    old = pd.DataFrame([{**base, "sn_slab_coverage": c} for c in (0.6, 0.4)])     # rows written before 27 Sep 2026
    assert QC["nm"](new).tolist() == [True, False]
    assert QC["nm"](old).tolist() == [True, False]
    assert QC["nm"](pd.concat([new, old], ignore_index=True)).tolist() == [True, False, True, False]   # a resumed table


def _zero_syn_cache(fs, shift_lps=(0.0, 0.0, 0.0)):
    """A SyN cache in ``nm_template``'s layout: a zero warp over the MNI midbrain box and an affine moving every MNI point
    by ``shift_lps`` (mm) into T1 space, as ANTs' forward affine maps fixed (MNI) points to moving (T1) points."""
    import ants
    import SimpleITK as sitk

    from pie.imaging.nm_template import syn_paths
    warp, affine = syn_paths(fs)["fwd"]
    warp.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(ants.from_numpy(np.zeros((40, 50, 40, 3), np.float32), origin=(30.0, 45.0, -30.0), spacing=(1.5, 1.5, 1.5),
                                     direction=np.diag([-1.0, -1.0, 1.0]), has_components=True), str(warp))
    sitk.WriteTransform(sitk.AffineTransform(np.eye(3).ravel().tolist(), list(shift_lps)), str(affine))


def _atlas_sn_lps():
    """LPS mm of the CIT168 SN (SNc + SNr) voxels in MNI space, per side (1 left, 2 right)."""
    atlas = nm.pauli_atlas()
    ras = nib.affines.apply_affine(atlas.affine, np.argwhere(np.isin(np.asarray(atlas.dataobj), [7, 9])))
    return {1: ras[ras[:, 0] < 0] * [-1, -1, 1], 2: ras[ras[:, 0] >= 0] * [-1, -1, 1]}


def test_the_atlas_reaches_t1_and_slab_through_the_subjects_syn(tmp_path):
    """The study's mapping: the CIT168 SN reaches the T1 (slab coverage, second rigid start) and the slab (ROIs,
    hemispheres) through the cached SyN, never an affine that nm fits itself. With a zero warp and an affine that moves
    MNI by a known offset, both land exactly that offset away."""
    import SimpleITK as sitk

    fs = tmp_path / "fs"
    shift = np.array([4.0, -2.0, 3.0])
    _zero_syn_cache(fs, shift)
    t1_aff = np.eye(4)
    t1_aff[:3, 3] = (-40, -60, -40)                                   # 1 mm, RAS x -40..39, y -60..19, z -40..19
    t1 = nib.MGHImage(np.zeros((80, 80, 60), np.float32), t1_aff)
    atlas, got = _atlas_sn_lps(), nm._sn_points_in_t1(t1, fs)
    for side in (1, 2):
        assert abs(len(got[side]) - len(atlas[side])) < 0.1 * len(atlas[side])
        assert np.allclose(got[side].mean(0), atlas[side].mean(0) + shift, atol=0.5), side
    slab_aff = np.diag([0.5, 0.5, 1.5, 1.0])
    slab_aff[:3, 3] = (-30, -40, -30)
    slab = nib.Nifti1Image(np.zeros((120, 80, 20), np.float32), slab_aff)
    pauli, left = nm._atlas_on_slab(slab, sitk.Euler3DTransform(), fs)   # (z, y, x); identity rigid: slab space = T1 space
    sn = np.isin(pauli, [7, 9])
    for side, mask in ((1, sn & left), (2, sn & ~left)):
        ras = nib.affines.apply_affine(slab_aff, np.argwhere(mask)[:, ::-1])
        assert np.allclose((ras * [-1, -1, 1]).mean(0), atlas[side].mean(0) + shift, atol=0.5), side


def test_syn_cache_fits_the_studys_registration_under_its_own_cache_name(tmp_path, monkeypatch):
    """The study's T1 -> MNI SyN (measurement_recovery registration_stability.py): both images on a 2 mm grid, intensities
    scaled to their 99.5th percentile, antsRegistration SyN with a CC (radius 2) metric, affine 1000/500/250/100 and
    deformable 40/20/10 iterations, seeded. On 10 control scans it put the NM masks at Dice 0.84 with the study's, where
    antsRegistrationSyNQuick[s] gave 0.72 and 23 % larger masks. A new cache name keeps the old SyNQuick caches out."""
    import ants

    from pie.imaging import nm_template
    mri = tmp_path / "mri"
    mri.mkdir()
    for name in ("orig.mgz", "mask.mgz"):
        nib.save(nib.MGHImage(np.random.default_rng(0).random((20, 20, 20)).astype(np.float32) * 300, np.eye(4)), mri / name)
    seen = {}

    def fake(fixed, moving, **kw):
        seen.update(kw, fixed=fixed, moving=moving)
        d = tmp_path / "ants_tmp"
        d.mkdir(exist_ok=True)
        ants.image_write(ants.from_numpy(np.zeros((4, 4, 4, 3), np.float32), has_components=True), str(d / "1Warp.nii.gz"))
        (d / "0GenericAffine.mat").write_bytes(b"affine")
        return {"fwdtransforms": [str(d / "1Warp.nii.gz"), str(d / "0GenericAffine.mat")], "invtransforms": []}

    monkeypatch.setattr(ants, "registration", fake)
    monkeypatch.setattr(nm_template, "mni_brain_path", lambda: str(mri / "orig.mgz"))
    monkeypatch.setattr(nm_template, "crop_warp", lambda p: None)
    out = nm_template.syn_cache(tmp_path)
    assert seen["type_of_transform"] == "SyN" and seen["syn_metric"] == "CC" and seen["syn_sampling"] == 2
    assert tuple(seen["reg_iterations"]) == (40, 20, 10) and tuple(seen["aff_iterations"]) == (1000, 500, 250, 100)
    assert seen["fixed"].spacing == (2.0, 2.0, 2.0) and seen["moving"].spacing == (2.0, 2.0, 2.0)
    assert 0.99 < seen["fixed"].max() <= 1.0 and 0.99 < seen["moving"].max() <= 1.0
    assert all("syncc2mm" in Path(p).name for p in out["fwd"]) and nm_template.syn_cached(tmp_path)
