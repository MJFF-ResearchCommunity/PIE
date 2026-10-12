"""Unit tests for the DaTscan module that need no data: photopeak selection, reconstruction round-trip,
NIfTI geometry, and SBR arithmetic on a synthetic phantom."""

import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import datscan


def _window(lo, hi):
    r = Dataset()
    r.EnergyWindowLowerLimit, r.EnergyWindowUpperLimit = lo, hi
    e = Dataset()
    e.EnergyWindowRangeSequence = [r]
    return e


def test_photopeak_window_selection_handles_vendor_units():
    ds = Dataset()
    ds.EnergyWindowInformationSequence = [_window(108, 132), _window(143, 175)]
    assert datscan._photopeak_window(ds) == 2
    ds.EnergyWindowInformationSequence = [_window(14310, 17490)]  # Marconi-style x100 values
    assert datscan._photopeak_window(ds) == 1
    ds.EnergyWindowInformationSequence = []
    assert datscan._photopeak_window(ds) is None


def test_fbp_reconstruction_recovers_hot_spots():
    from skimage.transform import radon

    n, spacing = 64, 3.0
    yy, xx = np.mgrid[:n, :n]
    phantom = np.exp(-((xx - 24) ** 2 + (yy - 32) ** 2) / 18.0) * 5 + np.exp(-((xx - 40) ** 2 + (yy - 32) ** 2) / 18.0) * 2 + 0.3 * (((xx - 32) ** 2 + (yy - 32) ** 2) < 26**2)
    angles = np.arange(0, 360, 3.0)
    sino = radon(phantom, theta=angles, circle=True)           # (n_bins = n, n_angles): no padding, like a camera
    proj = np.zeros((len(angles), 3, sino.shape[0]), dtype=np.float32)
    proj[:, 1, :] = sino.T                                       # one informative axial slice
    vol = datscan.reconstruct(proj, angles, spacing, fwhm_mm=0.0)
    rec = vol[:, :, 1]
    assert rec.shape == (sino.shape[0], sino.shape[0])
    # the two hot spots are recovered at the right positions with the right intensity order
    a, b = rec[30:35, 22:27].mean(), rec[30:35, 38:43].mean()
    assert a > b > rec[5:10, 5:10].mean()
    assert 1.8 < a / b < 3.2


def test_to_nifti_geometry_and_sitk_roundtrip():
    vol = np.zeros((10, 12, 14), dtype=np.float32)
    vol[3, 4, 5] = 1.0
    img = datscan.to_nifti(vol, 2.5)
    assert np.allclose(np.diag(img.affine)[:3], 2.5)
    # reconstruction (row, col, slice) -> NIfTI (col, -row, -slice): rows run posterior, slices run inferior
    assert img.shape == (12, 10, 14) and img.get_fdata()[4, 6, 8] == 1.0
    # the physical point of that voxel must map back to the same voxel through the SimpleITK image
    sitk_img = datscan._sitk_from_nib(img)
    ras = img.affine @ np.array([4, 6, 8, 1.0])
    idx = sitk_img.TransformPhysicalPointToIndex((-float(ras[0]), -float(ras[1]), float(ras[2])))
    assert sitk_img[idx] == 1.0


def test_sbr_arithmetic_with_perfect_registration():
    """ROI arithmetic on co-registered arrays: SBRs equal the planted contrast; dilation keeps the ordering."""
    shape = (60, 60, 60)
    lab = np.zeros(shape, dtype=np.int16)
    lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
    lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
    lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
    lab[10:50, 10:50, 20:40][lab[10:50, 10:50, 20:40] == 0] = 2
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == datscan.PUTAMEN_L] = 3.0    # SBR 2.0
    counts[lab == datscan.PUTAMEN_R] = 1.5    # SBR 0.5
    out = datscan.sbr_from_arrays(counts, lab, dilate=0, ref_dilate=0, search_vox=0)
    assert abs(out["sbr_putamen_l"] - 2.0) < 1e-6 and abs(out["sbr_putamen_r"] - 0.5) < 1e-6
    out2 = datscan.sbr_from_arrays(counts, lab, dilate=1, ref_dilate=2, search_vox=2)
    assert out2["sbr_putamen_l"] > out2["sbr_putamen_r"] > 0 and out2["n_label_voxels"] > out["n_label_voxels"]


def test_registration_on_phantom_keeps_left_right_ordering():
    """End to end on a phantom with anterior/posterior asymmetry: after registration L must stay hotter than R."""
    import nibabel as nib

    shape = (60, 60, 60)
    lab = np.zeros(shape, dtype=np.int16)
    lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
    lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
    lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
    lab[10:50, 10:50, 20:40][lab[10:50, 10:50, 20:40] == 0] = 2
    lab[14:46, 12:20, 20:40] = 3
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == 3] = 0.6
    counts[lab == datscan.PUTAMEN_L] = 3.0
    counts[lab == datscan.PUTAMEN_R] = 1.5
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -60.0
    out = datscan.quantify(nib.Nifti1Image(counts, affine), nib.Nifti1Image(counts * 100, affine), nib.Nifti1Image(lab, affine))
    assert out["sbr_putamen_l"] > out["sbr_putamen_r"] > 0 and out["n_label_voxels"] > 0


def test_anterior_posterior_split_and_white_matter_reference():
    """The A/P split follows the anterior-coordinate map; the white-matter SBR uses WM far from the striatum."""
    shape = (40, 60, 40)
    lab = np.zeros(shape, dtype=np.int16)
    lab[15:21, 20:36, 10:16] = datscan.PUTAMEN_L        # 16 voxels long along axis 1
    lab[15:21, 20:36, 24:30] = datscan.PUTAMEN_R
    lab[10:30, 50:58, 10:30] = datscan.OCCIPITAL[0]
    lab[5:35, 5:45, 5:35][lab[5:35, 5:45, 5:35] == 0] = datscan.WM[0]
    ap = np.broadcast_to(np.arange(shape[1], dtype=float)[None, :, None], shape).copy()   # larger index = more anterior
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == datscan.PUTAMEN_L] = 3.0
    counts[(lab == datscan.PUTAMEN_L) & (ap >= 28)] = 5.0                                  # anterior half hotter
    counts[lab == datscan.WM[0]] = 2.0                                                      # WM reference twice the occipital
    out = datscan.sbr_from_arrays(counts, lab, search_vox=0, ap=ap, spacing_mm=2.0)
    assert abs(out["sbr_putamen_l_ant"] - 4.0) < 1e-6 and abs(out["sbr_putamen_l_post"] - 2.0) < 1e-6
    assert abs(out["sbr_putamen_l"] - 3.0) < 1e-6 and abs(out["sbr_putamen_r_post"] - 0.0) < 1e-6
    assert abs(out["sbrwm_putamen_l_post"] - 0.5) < 1e-6 and out["n_wm_voxels"] > 0
    # WM within 15 mm of the striatum is excluded from the reference (7.5 voxels at 2 mm)
    wm_used = out["n_wm_voxels"]
    assert wm_used < (lab == datscan.WM[0]).sum()
    assert out["n_wm_voxels"] == wm_used
    out0 = datscan.sbr_from_arrays(counts, lab, search_vox=0)                             # without ap: no split columns
    assert "sbr_putamen_l_post" not in out0 and abs(out0["sbr_putamen_l"] - 3.0) < 1e-6


def _ell(p, c, r):
    return sum(((p[i] - c[i]) / r[i]) ** 2 for i in range(3)) <= 1


def _head(p):
    """Analytic head (mm, RAS): labelled brain with striatum and occipital cortex; counts add scalp uptake beyond an
    8 mm skull gap and a neck below the brain, which a head-outline fit would follow."""
    lab = np.zeros(p[0].shape, np.int16)
    brain = _ell(p, (0, 0, 0), (65, 80, 55))
    lab[brain] = 2
    lab[_ell(p, (0, -60, 0), (40, 14, 25)) & brain] = datscan.OCCIPITAL[0]
    for side, x in ((datscan.PUTAMEN_L, -28), (datscan.PUTAMEN_R, 28)):
        lab[_ell(p, (x, 5, 0), (7, 18, 12))] = side
    for side, x in ((datscan.CAUDATE_L, -14), (datscan.CAUDATE_R, 14)):
        lab[_ell(p, (x, 22, 8), (6, 12, 12))] = side
    counts = np.where(lab > 0, 1.0, 0.0)
    counts[_ell(p, (0, 0, 0), (79, 94, 69)) & ~_ell(p, (0, 0, 0), (73, 88, 63))] = 0.6
    counts[(p[0] ** 2 + (p[1] + 10) ** 2 <= 35 ** 2) & (p[2] < -40) & (lab == 0)] = 0.9
    counts[lab == datscan.PUTAMEN_L], counts[lab == datscan.PUTAMEN_R] = 4.0, 2.5
    counts[np.isin(lab, [datscan.CAUDATE_L, datscan.CAUDATE_R])] = 4.5
    return lab, counts


def _grid(shape, sp, centre):
    aff = np.diag([sp, sp, sp, 1.0])
    aff[:3, 3] = np.array(centre) - (np.array(shape) - 1) * sp / 2
    return aff, (aff[:3, :3] @ np.indices(shape).reshape(3, -1) + aff[:3, 3:]).reshape((3,) + tuple(shape))


def _spect(mag, mag_z, tilt_deg=8.0):
    """SPECT of the analytic head on a 4 mm grid, magnified ``mag`` transaxially and ``mag_z`` axially (fan-beam data
    reconstructed as parallel-beam: mag > 1, mag_z = 1), tilted about the left-right axis and offset."""
    import nibabel as nib
    from scipy import ndimage

    aff, q = _grid((64, 64, 64), 4.0, (7, -5, -30))
    t = np.radians(tilt_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    p = rot.T @ (q.reshape(3, -1) / np.array([[mag], [mag], [mag_z]])) + np.array([[3], [-4], [6]])
    counts = ndimage.gaussian_filter(_head(p.reshape(q.shape))[1], 11 / 2.3548 / 4.0)
    return nib.Nifti1Image(counts.astype(np.float32), aff)


def _t1_and_labels():
    import nibabel as nib

    aff, p = _grid((96, 112, 100), 2.0, (0, 0, -20))
    lab = _head(p)[0]
    return nib.Nifti1Image((lab > 0).astype(np.float32) * 100, aff), nib.Nifti1Image(lab, aff)


def test_registration_ignores_scalp_and_neck_and_bounds_fan_beam_scales():
    """Brain-only template: the transaxial fan-beam magnification is recovered with the axial scale at 1 although the
    SPECT carries scalp and neck uptake; an implausible axial scale is clamped at its bound and flagged."""
    t1, aparc = _t1_and_labels()
    out = datscan.quantify(_spect(1.0, 1.0), t1, aparc)                          # parallel-hole: rigid
    assert abs(out["reg_rot_deg"] - 8) < 2 and out["shift_vox"] <= 1 and not out["shift_edge"]
    assert out["sbr_putamen_l"] > out["sbr_putamen_r"] > 0
    assert {"reg_rot_x_deg", "reg_tx_mm", "reg_ty_mm", "reg_tz_mm", "shift_x_mm", "shift_y_mm", "shift_z_mm"} <= set(out)
    out = datscan.quantify(_spect(1.4, 1.0), t1, aparc, scale_fit=True)
    assert abs(out["reg_scale_x"] - 1.4) < 0.05 and abs(out["reg_scale_y"] - 1.4) < 0.05 and abs(out["reg_scale_z"] - 1) < 0.05
    assert not out["reg_scale_at_bound"] and not out["shift_edge"] and out["sbr_putamen_l"] > out["sbr_putamen_r"] > 0
    out = datscan.quantify(_spect(1.4, 0.6), t1, aparc, scale_fit=True)          # axially compressed: not a camera geometry
    assert out["reg_scale_at_bound"] and abs(out["reg_scale_z"] - datscan.SCALE_BOUNDS[2][0]) < 1e-6


def test_placement_search_window_is_in_mm_and_flags_its_edge():
    shape = (60, 60, 60)
    lab = np.zeros(shape, dtype=np.int16)
    lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
    lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
    lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
    lab[10:50, 10:50, 20:40][lab[10:50, 10:50, 20:40] == 0] = 2
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == datscan.PUTAMEN_L], counts[lab == datscan.PUTAMEN_R] = 3.0, 1.5
    truth = datscan.sbr_from_arrays(counts, lab, search_vox=0)
    off = datscan.sbr_from_arrays(counts, np.roll(lab, 3, axis=1), spacing_mm=3.0)      # 9 mm off: beyond the old +-2 voxels
    assert abs(off["sbr_putamen_l"] - truth["sbr_putamen_l"]) < 1e-6 and not off["shift_edge"] and off["shift_y_mm"] == -9.0
    old = datscan.sbr_from_arrays(counts, np.roll(lab, 3, axis=1), search_vox=2)
    assert old["shift_edge"] and old["sbr_putamen_l"] < truth["sbr_putamen_l"]
    far = datscan.sbr_from_arrays(counts, np.roll(lab, 5, axis=1), spacing_mm=3.0)      # 15 mm: past the 12 mm window
    assert far["shift_edge"]


def test_dat_qc_fails_implausible_transforms_and_keeps_old_tables():
    import pandas as pd

    from pie.imaging import manifest

    ok = {"reg_metric": -0.7, "n_label_voxels": 500, "reg_rot_deg": 10.0, "reg_tx_mm": 5.0, "reg_ty_mm": -3.0, "reg_tz_mm": 20.0,
          "reg_scale_at_bound": False, "shift_edge": False}
    rows = [ok, dict(ok, reg_rot_deg=40.0), dict(ok, reg_tz_mm=60.0), dict(ok, reg_scale_at_bound=True), dict(ok, shift_edge=True),
            dict(ok, reg_metric=-0.3), dict(ok, reg_metric=-0.15)]
    # a clamped scale is flagged, not failed; the brain-only correlation bar is 0.2
    assert manifest.QC["dat"](pd.DataFrame(rows)).tolist() == [True, False, False, True, False, True, False]
    legacy = pd.DataFrame([dict(ok, reg_template="legacy_head", reg_metric=-0.3, shift_edge=True),
                           dict(ok, reg_template="legacy_head", reg_metric=-0.5, shift_edge=True), dict(ok, reg_template="brain")])
    assert manifest.QC["dat"](legacy).tolist() == [False, True, True]                   # the fallback keeps the legacy rule
    old = pd.DataFrame({"reg_metric": [-0.7], "n_label_voxels": [500]})                 # tables written before the columns
    assert manifest.QC["dat"](old).tolist() == [True]


def _stack(arr, iop, ipp, px, dz, in_detector_sequence=True):
    from pydicom.dataset import FileMetaDataset

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.set_pixel_data(arr.astype(np.uint16), "MONOCHROME2", 16)
    geo = ds
    if in_detector_sequence:                    # where NM RECON TOMO files keep the patient geometry
        geo = Dataset()
        ds.DetectorInformationSequence = [geo]
    if iop is not None:
        geo.ImageOrientationPatient = list(iop)
    geo.ImagePositionPatient = list(ipp)
    ds.PixelSpacing, ds.SpacingBetweenSlices = list(px), dz
    return ds


def test_read_volume_places_a_rotated_flipped_stack_by_its_dicom_geometry():
    from scipy import ndimage

    arr = np.zeros((12, 16, 20))                  # frames, rows, columns
    arr[3:6, 5:8, 7:10] = 1000                    # blob centred on frame 4, row 6, column 8
    a = np.radians(30)
    row, col = np.array([np.cos(a), np.sin(a), 0.0]), np.array([0.0, 0.0, -1.0])        # oblique, columns run feet-ward
    ipp, px, dz = np.array([10.0, -20.0, 30.0]), (2.0, 3.0), -4.0                        # anisotropic; frames step against the normal
    lps = ipp + 8 * px[1] * row + 6 * px[0] * col + 4 * dz * np.cross(row, col)
    img, sp = datscan.read_volume(_stack(arr, np.r_[row, col], ipp, px, dz))
    assert sp == 2.0 and np.allclose(np.abs(img.affine[:3, :3]), np.diag([2.0] * 3))
    got = img.affine @ np.r_[ndimage.center_of_mass(img.get_fdata()), 1.0]
    assert np.linalg.norm(got[:3] - lps * [-1, -1, 1]) < 1.0                             # RAS = LPS with x, y negated
    img2, _ = datscan.read_volume(_stack(arr, np.r_[row, col], ipp, px, dz, in_detector_sequence=False))
    assert np.allclose(img2.affine, img.affine)
    try:
        datscan.read_volume(_stack(arr, None, ipp, px, dz))
        raise AssertionError("a stack without ImageOrientationPatient must be refused")
    except ValueError as e:
        assert "ImageOrientationPatient" in str(e)


def test_calibration_keeps_the_reference_spread():
    """A linear relation with measurement error in both variables: Deming recovers the slope and equipercentile the
    distribution, where OLS shrinks the mapped values towards the mean."""
    import pandas as pd

    rng = np.random.default_rng(3)
    true = rng.normal(2.0, 0.6, 3000)
    d = pd.DataFrame({"hdr_model": "CAM", "pie": true + rng.normal(0, 0.3, true.size),
                      "ref": 0.3 + 1.2 * true + rng.normal(0, 0.3, true.size)})
    slope, _ = datscan.deming(d["pie"], d["ref"])
    assert abs(slope - 1.2) < 0.06 and np.polyfit(d["pie"], d["ref"], 1)[0] < 1.0       # OLS attenuates it to ~0.96
    for method, lo in (("deming", 0.97), ("equipercentile", 0.999)):
        cal = datscan.calibrate(d, {"pie": "ref"}, method=method)["pie_cal"]
        assert lo < cal.std() / d["ref"].std() < 1.06, method
        assert abs(cal.mean() - d["ref"].mean()) < 0.01


def test_an_uncorrelated_camera_uses_the_pooled_calibration():
    """Deming's slope divides by the covariance: a camera whose values do not track the published ones must not get its own map."""
    import pandas as pd

    rng = np.random.default_rng(4)
    true = rng.normal(2.0, 0.6, 200)
    d = pd.DataFrame({"hdr_model": ["A"] * 180 + ["B"] * 20, "ref": true, "pie": 0.5 * true + rng.normal(0, 0.05, 200)})
    d.loc[180:, "pie"] = rng.permutation(d.loc[180:, "pie"].to_numpy())       # camera B: right range, no relation
    out = datscan.calibrate(d, {"pie": "ref"})
    assert (out.loc[:179, "calibration"] == "A").all() and (out.loc[180:, "calibration"] == "POOLED").all()
    assert out.loc[180:, "pie_cal"].std() < 2 * d["ref"].std()


def test_a_failed_brain_only_fit_falls_back_to_the_legacy_registration(monkeypatch):
    calls = []

    def fake_quantify(spect, t1, aparc, legacy=False, **kw):
        calls.append(legacy)
        return {"reg_metric": -0.6 if legacy else -0.1, "reg_rot_deg": 5.0, "shift_edge": False, "sbr_putamen_l": 2.0 if legacy else 0.5}

    monkeypatch.setattr(datscan, "quantify", fake_quantify)
    q = datscan.quantify_with_fallback(None, None, None)
    assert calls == [False, True] and q["sbr_putamen_l"] == 2.0 and q["reg_fallback"]
    monkeypatch.setattr(datscan, "quantify", lambda *a, legacy=False, **k: {"reg_metric": -0.5, "reg_rot_deg": 5.0, "shift_edge": False})
    assert not datscan.quantify_with_fallback(None, None, None)["reg_fallback"]
