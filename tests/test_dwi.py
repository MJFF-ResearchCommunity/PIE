"""Unit tests for the diffusion module that need no data: run assembly, ROI construction and feature arithmetic."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import dwi


def _run(tmp, name, n_dirs, shape=(8, 8, 6), pe="j", bval=1000):
    n = n_dirs + 1
    img = nib.Nifti1Image(np.random.default_rng(0).random(shape + (n,)).astype(np.float32), np.diag([2.0, 2.0, 2.0, 1.0]))
    nib.save(img, tmp / f"{name}.nii.gz")
    np.savetxt(tmp / f"{name}.bval", np.r_[0, np.full(n_dirs, bval)][None])
    v = np.random.default_rng(1).normal(size=(3, n))
    v[:, 0] = 0
    np.savetxt(tmp / f"{name}.bvec", v / np.maximum(np.linalg.norm(v, axis=0), 1e-9))
    json.dump({"PhaseEncodingDirection": pe, "Manufacturer": "Test"}, open(tmp / f"{name}.json", "w"))
    return (str(tmp / f"{name}.nii.gz"), str(tmp / f"{name}.bval"), str(tmp / f"{name}.bvec"), str(tmp / f"{name}.json"))


def test_assemble_concatenates_same_geometry_and_drops_b0_only_and_opposite_phase(tmp_path):
    runs = [_run(tmp_path, "b700", 10, bval=700), _run(tmp_path, "b1000", 10), _run(tmp_path, "rev", 0, pe="j-"),
            _run(tmp_path, "lr", 12, pe="i"), _run(tmp_path, "other_geom", 30, shape=(9, 9, 6))]
    ds = dwi.assemble(runs)
    # the single 30-direction run beats the two concatenated 10-direction shells (20) and the 12-direction LR run
    assert ds["data"].shape == (9, 9, 6, 31) and ds["n_runs"] == 1
    ds2 = dwi.assemble(runs[:3])
    assert ds2["data"].shape[3] == 22 and ds2["n_runs"] == 2 and ds2["shells"] == [700, 1000]
    assert ds2["bvals"].shape == (22,) and ds2["bvecs"].shape == (3, 22)


def test_roi_masks_split_sn_and_features_average_correctly():
    shape = (10, 12, 14)  # sitk order (z, y, x)
    fs = np.zeros(shape, dtype=np.int32)
    fs[2:5, 3:6, 2:5] = 12    # left putamen (FastSurfer label 12), x low
    fs[2:5, 3:6, 9:12] = 51   # right putamen
    pauli = np.zeros(shape, dtype=np.int32)
    code = {n: i + 1 for i, n in enumerate(dwi.PAULI)}
    pauli[5:7, 4:8, 1:4] = code["Pu"]
    pauli[5:7, 4:8, 10:13] = code["Pu"]
    pauli[6:8, 2:8, 2:4] = code["SNc"]   # SN spanning y 2..7 on the x-low side (same side as FastSurfer's left putamen)
    pauli[6:8, 2:8, 10:12] = code["SNr"]
    phys_y = np.broadcast_to(np.arange(shape[1])[None, :, None].astype(float), shape)
    rois = dwi._roi_masks(fs, pauli, phys_y)
    assert rois["putamen_l"].sum() == 27 and rois["putamen_r"].sum() == 27
    # left = the side of FastSurfer's left putamen (x low here); posterior = larger y
    assert rois["sn_l"].sum() == 24 and rois["sn_r"].sum() == 24 and rois["snc_l"].sum() == 24 and rois["snr_r"].sum() == 24
    assert rois["sn_posterior_l"].sum() + rois["sn_anterior_l"].sum() == rois["sn_l"].sum()
    assert phys_y[rois["sn_posterior_l"]].min() > phys_y[rois["sn_anterior_l"]].max()
    maps = {k: np.full(shape, v, dtype=np.float32) for k, v in zip(dwi.METRICS, (0.4, 0.8, 0.2, 0.5))}
    maps["fw"][rois["sn_posterior_l"]] = 0.6
    out = dwi.features(maps, rois)
    assert abs(out["sn_posterior_l_fw"] - 0.6) < 1e-6 and abs(out["sn_anterior_l_fw"] - 0.2) < 1e-6
    assert abs(out["putamen_mean_fa"] - 0.4) < 1e-6 and out["n_putamen_l"] == 27
    assert np.isnan(out["vta_l_fw"]) and out["n_vta_l"] == 0


def test_single_shell_free_water_recovers_planted_fraction():
    """Bi-tensor phantom (single shell b=1000, 64 directions): the fit should recover f to within ~0.1 and keep
    tissue FA near the planted value."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 64))
    g /= np.linalg.norm(g, axis=0)
    bvals = np.r_[np.zeros(4), np.full(64, 1000.0)]
    bvecs = np.c_[np.zeros((3, 4)), g]
    D = np.diag([1.5e-3, 0.4e-3, 0.4e-3])  # FA ~0.7 tissue
    shape = (2, 2, 2)
    data = np.zeros(shape + (len(bvals),), dtype=np.float32)
    planted = [0.1, 0.3, 0.5, 0.7] * 2
    for n, (i, j, k) in enumerate(np.ndindex(shape)):
        f = planted[n]
        att = np.array([np.exp(-b * (v @ D @ v)) for b, v in zip(bvals, bvecs.T)])
        data[i, j, k] = 1000 * ((1 - f) * att + f * np.exp(-bvals * dwi.D_WATER)) * (1 + 0.01 * rng.normal(size=len(bvals)))
    f_map, fat_map = dwi._fw_single_shell(data, bvals, bvecs, np.ones(shape, bool))
    for n, (i, j, k) in enumerate(np.ndindex(shape)):
        assert abs(f_map[i, j, k] - planted[n]) < 0.12, (planted[n], f_map[i, j, k])
        assert fat_map[i, j, k] > 0.5


def test_labels_to_dwi_applies_the_inverse_registration():
    """A label at a known T1 location must land where the fixed->moving registration maps it (DWI = T1 + 3 mm LPS x)."""
    import SimpleITK as sitk

    lab = np.zeros((40, 40, 40), dtype=np.float32)
    lab[20, 25, 30] = 7
    aff = np.eye(4)
    aff[:3, 3] = -20
    tgt = dwi._sitk_native(np.zeros((40, 40, 40), np.float32), aff)
    tx = sitk.Euler3DTransform()
    tx.SetTranslation((3.0, 0.0, 0.0))
    out = dwi.labels_to_dwi(nib.Nifti1Image(lab, aff), tgt, [tx])
    z, y, x = np.argwhere(out == 7)[0]
    assert (x, y, z) == (17, 25, 30)
