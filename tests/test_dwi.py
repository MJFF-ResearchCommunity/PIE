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


def test_denoise_reduces_noise_and_crop_even_drops_odd_axes():
    rng = np.random.default_rng(0)
    shape = (16, 16, 11)
    zz, yy, xx = np.indices(shape)
    head = ((xx - 8) ** 2 + (yy - 8) ** 2 + (zz - 5) ** 2) < 5.5 ** 2                       # a bright "brain" on a dark background
    core = ((xx - 8) ** 2 + (yy - 8) ** 2 + (zz - 5) ** 2) < 3.0 ** 2
    clean = (np.where(head, 100.0, 3.0)[..., None] * np.linspace(1.0, 0.4, 24)[None, None, None, :]).astype(np.float32)
    noisy = clean + rng.normal(0, 8, clean.shape).astype(np.float32)
    ds = {"data": noisy, "rev_b0": rng.random(shape + (2,)).astype(np.float32), "bvals": np.r_[0, np.full(23, 1000)]}
    out = dwi.denoise_dwi(ds)
    assert out["denoised"] and out["data"].shape == noisy.shape and out["data"].dtype == np.float32
    assert np.abs(out["data"] - clean)[core].std() < 0.6 * np.abs(noisy - clean)[core].std()
    even = dwi._crop_even(ds)
    assert even["data"].shape[:3] == (16, 16, 10) and even["rev_b0"].shape[:3] == (16, 16, 10)
    assert dwi._crop_even(even) is even


def test_multishell_free_water_is_nan_where_the_fit_is_not_physical():
    # the study's rule (freewater_qc): a failed or non-physical NLS solution is missing, never FW = 0
    rng = np.random.default_rng(3)
    g = rng.normal(size=(3, 30))
    g /= np.linalg.norm(g, axis=0)
    bvals = np.r_[0, 0, np.full(30, 700.), np.full(30, 2000.)]
    bvecs = np.c_[np.zeros((3, 2)), g, g]
    sig = lambda D, f: 1000 * ((1 - f) * np.exp(-bvals * np.einsum("in,ij,jn->n", bvecs, D, bvecs)) + f * np.exp(-bvals * dwi.D_WATER))
    data = np.zeros((2, 1, 1, len(bvals)), np.float32)
    data[0, 0, 0] = sig(np.diag([1.5e-3, 0.4e-3, 0.4e-3]), 0.2)
    data[1, 0, 0] = sig(np.diag([1.5e-3, -0.4e-3, 0.3e-3]), 0.0)     # negative tissue diffusivity: no physical fit
    maps = dwi.fit_models({"bvals": bvals, "bvecs": bvecs, "data": data, "mask": np.ones((2, 1, 1), bool)})
    assert abs(maps["fw"][0, 0, 0] - 0.2) < 1e-3
    assert np.isnan(maps["fw"][1, 0, 0]) and np.isnan(maps["fat"][1, 0, 0]) and np.isnan(maps["mdt"][1, 0, 0])


def test_tensor_maps_are_missing_outside_the_brain_mask():
    # an ROI partly outside the mask must average only measured voxels (zeros there once halved nigral MD)
    rng = np.random.default_rng(5)
    g = rng.normal(size=(3, 30))
    g /= np.linalg.norm(g, axis=0)
    bvals, bvecs = np.r_[0, 0, np.full(30, 1000.)], np.c_[np.zeros((3, 2)), g]
    D = np.diag([1.2e-3, 0.6e-3, 0.6e-3])
    data = np.tile(1000 * np.exp(-bvals * np.einsum("in,ij,jn->n", bvecs, D, bvecs)), (2, 1, 1, 1)).astype(np.float32)
    mask = np.array([True, False]).reshape(2, 1, 1)
    maps = dwi.fit_models({"bvals": bvals, "bvecs": bvecs, "data": data, "mask": mask})
    assert np.isnan(maps["md"][1, 0, 0]) and not maps["tensor_physical"][1, 0, 0]
    out = dwi.features(maps, {"roi": np.ones((2, 1, 1), bool)}, min_voxels=1)
    assert abs(out["roi_md"] - 0.8e-3) < 1e-5


def test_scan_without_a_tensor_shell_is_an_explicit_error():
    # e.g. Philips PPMI-2 sessions whose three "b700/b1000/b2000" series were all acquired at b = 2000 (the header
    # and the attenuation agree; only the names differ): no b <= 1050 shell, so no tensor, never a silent FA of 0
    g = np.random.default_rng(4).normal(size=(3, 30))
    bvals, bvecs = np.r_[0, 0, np.full(30, 2000.)], np.c_[np.zeros((3, 2)), g / np.linalg.norm(g, axis=0)]
    ds = {"bvals": bvals, "bvecs": bvecs, "data": np.full((2, 1, 1, 32), 100, np.float32), "mask": np.ones((2, 1, 1), bool)}
    import pytest
    with pytest.raises(ValueError, match="b <= 1050"):
        dwi.fit_models(ds)


def _philips(tmp, name, n_dirs, desc):
    run = _run(tmp, name, n_dirs)
    json.dump({"Manufacturer": "Philips", "PhaseEncodingAxis": "j", "EstimatedTotalReadoutTime": 0.0742, "SeriesDescription": desc},
              open(run[3], "w"))
    return run


def test_philips_polarity_comes_from_the_series_name(tmp_path):
    # Philips exports the PE axis but no polarity; PPMI names it (_PA main, revB0_AP reverse). topup/eddy need only the
    # relative polarity, and on real PPMI-2 sessions it raised the forward/reverse b0 correlation from 0.38-0.52 to 0.97 (3 sessions)
    ds = dwi.assemble([_philips(tmp_path, "b1000", 10, "DTI_b1000_64dir_PA"), _philips(tmp_path, "rev", 0, "DTI_revB0_AP"),
                       _philips(tmp_path, "same", 0, "DTI_revB0_PA")])
    assert ds["meta"]["PhaseEncodingDirection"] == "j" and ds["rev_meta"]["PhaseEncodingDirection"] == "j-"
    assert ds["meta"]["TotalReadoutTime"] == ds["rev_meta"]["TotalReadoutTime"] == 0.0742
    assert ds["meta"]["pe_from_description"] and ds["rev_b0"].shape[3] == 1
    # Philips's other PPMI naming: the fat-shift direction sets the polarity (AX DTI ... FAT SHIFT L / R, ..._FAT_SHIFT_LEFT)
    for a, b in (("AX DTI 32 DIR FAT SHIFT R NO ANGLE", "AX DTI 32 DIR FAT SHIFT L NO ANGLE"), ("2D_DTI_EPI_FAT_SHIFT_RIGHT", "2D_DTI_EPI_FAT_SHIFT_LEFT"),
                 ("DTI_ RL", "DTI_ LR")):
        t = tmp_path / a[:6].replace(" ", "")
        t.mkdir(exist_ok=True)
        ds = dwi.assemble([_philips(t, "main", 10, a), _philips(t, "rev", 10, b)])
        assert {ds["meta"]["PhaseEncodingDirection"], ds["rev_meta"]["PhaseEncodingDirection"]} == {"j", "j-"}, a
    # no direction in the name: nothing is guessed, so no reverse pair and no topup/eddy
    ds = dwi.assemble([_philips(tmp_path, "b1000", 10, "DTI_b1000_64dir"), _philips(tmp_path, "rev", 0, "DTI_revB0")])
    assert "PhaseEncodingDirection" not in ds["meta"] and ds["rev_b0"] is None


# ------------------------------------------------------------------------------------------ the study's nigral measure
def test_assemble_records_where_each_run_sits(tmp_path):
    ds = dwi.assemble([_run(tmp_path, "b700", 10, bval=700), _run(tmp_path, "b1000", 10), _run(tmp_path, "rev", 0, pe="j-")])
    assert ds["run_bounds"] == [(0, 11), (11, 22)]
    assert np.flatnonzero(dwi.study_acquisition(ds["bvals"], ds["run_bounds"])).tolist() == list(range(11, 22))


def test_study_acquisition_is_one_b1000_run_with_its_own_b0():
    b = np.r_[0, 0, [700] * 6, 0, [1000] * 6, 0, [2000] * 6, [1000] * 8].astype(float)
    bounds = [(0, 8), (8, 15), (15, 22), (22, 30)]            # b700+b0, b1000+b0, b2000+b0, b1000 without its own b0
    assert np.flatnonzero(dwi.study_acquisition(b, bounds)).tolist() == list(range(8, 15))
    import pytest
    with pytest.raises(ValueError, match="own b0"):
        dwi.study_acquisition(b, [(22, 30)])
    mixed = np.r_[0, [700] * 6, [1000] * 6].astype(float)       # one run holding two shells: no single acquisition
    with pytest.raises(ValueError):
        dwi.study_acquisition(mixed, [(0, 13)])
    assert dwi.study_acquisition(np.r_[0, [1000] * 6].astype(float), None).all()


def _study_grid():
    """A 2 mm grid (RAS y rising with j) with SNc/SNr codes on both sides: left 71/91 at i 2-4, right 72/92 at i 7-9,
    j 3-8, z 3 SNc and z 4 SNr. Posterior voxels (low j: low RAS y, high LPS y) and anterior ones get different tensors,
    as do the two sides; a b700 run with a different tensor precedes the b1000 run the measure must use."""
    rng = np.random.default_rng(5)
    g = rng.normal(size=(3, 12))
    g /= np.linalg.norm(g, axis=0)
    b = np.r_[0, 0, [700] * 12, 0, 0, [1000] * 12].astype(float)
    v = np.c_[np.zeros((3, 2)), g, np.zeros((3, 2)), g]
    codes = np.zeros((12, 12, 8), int)
    data = np.zeros((12, 12, 8, len(b)), np.float32)
    md = {("l", True): 0.9e-3, ("l", False): 0.7e-3, ("r", True): 0.8e-3, ("r", False): 0.6e-3}
    for side, xs, k in (("l", range(2, 5), 1), ("r", range(7, 10), 2)):
        for i in xs:
            for j in range(3, 9):
                for z, lab in ((3, 7), (4, 9)):
                    codes[i, j, z] = 10 * lab + k
                    D = np.diag([1.5, 1.0, 0.5]) * md[(side, j <= 5)]
                    sig = 1000 * np.exp(-b * np.einsum("in,ij,jn->n", v, D, v))
                    sig[:14] = 1000 * np.exp(-b[:14] * 2e-3)          # the b700 run: a wrong, isotropic tissue
                    data[i, j, z] = sig
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = (-12, -30, -16)
    return data, b, v, aff, [(0, 14), (14, 28)], codes


def test_study_features_split_posterior_by_side_and_use_only_the_b1000_run():
    data, b, v, aff, bounds, codes = _study_grid()
    brain = np.ones(codes.shape, bool)
    row = dwi.study_features(data, b, v, aff, bounds, codes, brain, 0.5)
    assert abs(row["dwis_md_sn_posterior_l"] - 0.9e-3) < 1e-6 and abs(row["dwis_md_sn_posterior_r"] - 0.8e-3) < 1e-6
    assert abs(row["dwis_md_sn_anterior_l"] - 0.7e-3) < 1e-6 and abs(row["dwis_md_sn_posterior_mean"] - 0.85e-3) < 1e-6
    assert row["dwis_n_sn"] == 72 and row["dwis_shell"] == 1000 and row["dwis_qc_pass"], row["dwis_qc_failed_checks"]
    bad = data.copy()
    bad[2:4, 3:5, 3] = 0                                         # 4 of the 18 left posterior voxels have no usable fit
    row = dwi.study_features(bad, b, v, aff, bounds, codes, brain, 2.5)
    assert np.isnan(row["dwis_md_sn_posterior_l"]) and np.isnan(row["dwis_md_sn_posterior_mean"])  # < 80 % valid
    assert set(row["dwis_qc_failed_checks"].split(";")) == {"motion", "bilateral_posterior"}
    row = dwi.study_features(data, b, v, aff, bounds, codes, np.zeros(codes.shape, bool), 0.5)
    assert row["dwis_qc_failed_checks"] == "brain_coverage"


def test_study_nigra_labels_reach_the_dwi_grid_with_their_hemisphere(tmp_path):
    import SimpleITK as sitk
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nm import _zero_syn_cache

    fs = tmp_path / "fs"
    _zero_syn_cache(fs)                                          # T1 = MNI
    a = np.zeros((40, 40, 30), np.int16)
    aff = np.eye(4)
    aff[:3, 3] = (-20, -38, -25)                                 # 1 mm RAS grid over the midbrain
    a[10:13, 18:22, 12:14] = 7                                   # left SNc (RAS x -10..-8)
    a[27:30, 18:22, 12:14] = 9                                   # right SNr (RAS x 7..9)
    a[18:22, 18:22, 12:14] = 8                                   # red nucleus: not nigra
    grid = nib.Nifti1Image(np.zeros((20, 20, 15), np.float32), np.array([[2, 0, 0, -20], [0, 2, 0, -38], [0, 0, 2, -25], [0, 0, 0, 1.0]]))
    codes = dwi.study_nigra_labels(grid, fs, sitk.Euler3DTransform(), atlas=nib.Nifti1Image(a, aff))
    assert set(np.unique(codes)) == {0, 71, 92}
    assert (np.argwhere(codes == 71)[:, 0] < 10).all() and (np.argwhere(codes == 92)[:, 0] >= 10).all()


# ------------------------------------------------------------------------------------------ equivalence Addendum 1 (28 Sep 2026)
def _pe_run(tmp, name, n_dirs, pe, origin=(0.0, 0.0, 0.0), zoom=2.0, bvals=None, **meta):
    """A run with PE, readout, TE and TR recorded (``meta`` overrides) whose every volume holds its voxels' RAS x (mm)."""
    shape = (8, 8, 6)
    aff = np.diag([zoom, zoom, zoom, 1.0])
    aff[:3, 3] = origin
    x = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T)[:, 0].reshape(shape)
    b = np.r_[0, np.full(n_dirs, 1000)] if bvals is None else np.asarray(bvals)
    nib.save(nib.Nifti1Image(np.repeat(x[..., None], len(b), 3).astype(np.float32), aff), tmp / f"{name}.nii.gz")
    np.savetxt(tmp / f"{name}.bval", b[None])
    v = np.random.default_rng(1).normal(size=(3, len(b)))
    v[:, b <= 50] = 0
    np.savetxt(tmp / f"{name}.bvec", v / np.maximum(np.linalg.norm(v, axis=0), 1e-9))
    json.dump({"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05, "EchoTime": 0.08, "RepetitionTime": 9.0, **meta}, open(tmp / f"{name}.json", "w"))
    return tuple(str(tmp / f"{name}{e}") for e in (".nii.gz", ".bval", ".bvec", ".json"))


def test_reverse_b0_is_chosen_and_regridded_as_the_study_chose_it(tmp_path):
    # run_scientific_dwi.prepare: opposite PE with b0s, same shape and axes, same TE and TR, recorded PE and readout, any
    # origin (PPMI's GE L/R - R/L pairs sit 2-3.5 mm apart, Siemens revB0 0.03 mm); the nearest origin; regridded onto the main grid
    main = _pe_run(tmp_path, "main", 12, "i")
    far, near = _pe_run(tmp_path, "far", 12, "i-", origin=(3.0, 0, 0)), _pe_run(tmp_path, "near", 0, "i-", origin=(1.0, 0, 0))
    ds = dwi.assemble([main, far, near])
    assert ds["rev_b0"].shape == (8, 8, 6, 1) and ds["rev_meta"]["PhaseEncodingDirection"] == "i-"
    x = nib.affines.apply_affine(ds["affine"], np.indices((8, 8, 6)).reshape(3, -1).T)[:, 0].reshape(8, 8, 6)
    assert np.allclose(ds["rev_b0"][1:, ..., 0], x[1:], atol=1e-4)          # each voxel holds its own RAS x: on the main grid
    for bad in (_pe_run(tmp_path, "te", 0, "i-", EchoTime=0.09), _pe_run(tmp_path, "zoom", 0, "i-", zoom=2.5),
                _pe_run(tmp_path, "trt", 0, "i-", TotalReadoutTime=None)):
        assert dwi.assemble([main, bad])["rev_b0"] is None, bad[0]


def test_study_dataset_is_the_tensor_acquisition_alone(tmp_path):
    # run_scientific_dwi.choose_tensor_acquisition: one run's b0s and the shell nearest b = 1000 (500-1500, >= 12
    # directions; then more directions, then the path), with that run's reverse b0; the study corrected nothing else
    runs = [_pe_run(tmp_path, "b700", 12, "j", bvals=[0] + [700] * 12), _pe_run(tmp_path, "b1000", 12, "j"),
            _pe_run(tmp_path, "b2000", 12, "j", bvals=[0] + [2000] * 12), _pe_run(tmp_path, "rev", 0, "j-")]
    assert dwi.assemble(runs)["data"].shape[3] == 39                        # PIE's own dataset: every shell
    ds = dwi.study_dataset(runs)
    assert ds["source_nifti"] == [runs[1][0]] and ds["bvals"].tolist() == [0] + [1000] * 12 and ds["bvecs"].shape == (3, 13)
    assert ds["run_bounds"] == [(0, 13)] and ds["data"].shape[3] == 13 and ds["rev_b0"].shape[3] == 1
    ds = dwi.study_dataset([_pe_run(tmp_path, "mixed", 12, "j", bvals=[0] + [1000] * 12 + [2000] * 12)])
    assert ds["bvals"].tolist() == [0] + [1000] * 12 and ds["data"].shape[3] == 13     # one run, two shells: b0s + b1000
    ds = dwi.study_dataset([_pe_run(tmp_path, "z_rl", 12, "i-"), _pe_run(tmp_path, "a_lr", 12, "i")])
    assert ds["source_nifti"] == [str(tmp_path / "a_lr.nii.gz")] and ds["rev_meta"]["PhaseEncodingDirection"] == "i-"


def test_study_nigra_labels_are_sampled_nearest_neighbour(tmp_path):
    # the study sampled its nigra labels nearest-neighbour (nm_transform_chain.sample); genericLabel's per-label linear
    # vote drops a small region's voxel whose nearest grid point is off-centre (0.915 vs 0.967 of the study's voxels on 8 scans)
    import SimpleITK as sitk
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nm import _zero_syn_cache

    fs = tmp_path / "fs"
    _zero_syn_cache(fs)
    a = np.zeros((40, 40, 30), np.int16)
    aff = np.eye(4)
    aff[:3, 3] = (-20, -38, -25)
    a[10, 20, 13] = 7                                            # one left SNc voxel at RAS (-10, -18, -12)
    grid = nib.Nifti1Image(np.zeros((40, 40, 30), np.float32), np.array([[1, 0, 0, -19.6], [0, 1, 0, -37.6], [0, 0, 1, -24.6], [0, 0, 0, 1.0]]))
    codes = dwi.study_nigra_labels(grid, fs, sitk.Euler3DTransform(), atlas=nib.Nifti1Image(a, aff))
    assert (codes == 71).sum() == 1
