"""Small regression fixtures for the imaging audit. No registrations, denoising, downloads or neural fits."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pie.imaging import batch, dwi, labels, manifest, nm


def test_fastsurfer_lookup_keeps_earliest_completed_session(tmp_path):
    for iid in ("early", "late"):
        (tmp_path / iid / "stats").mkdir(parents=True)
        (tmp_path / iid / "stats" / "aseg+DKT.stats").touch()
    sessions = tmp_path / "sessions.csv"
    pd.DataFrame({"patno": [1, 1, 2], "image_id": ["late", "early", "unfinished"],
                  "session_date": ["2022-01-01", "2020-01-01", "2019-01-01"]}).to_csv(sessions, index=False)
    assert batch.fastsurfer_by_patno(sessions, tmp_path) == {1: str(tmp_path / "early")}


@pytest.mark.parametrize("x_sign", [-1, 1])
def test_gradient_rotation_oblique_fsl_frame_preserves_attenuation(x_sign):
    theta = 0.35
    oblique = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    aff = np.eye(4)
    aff[:3, :3] = oblique @ np.diag([2 * x_sign, 2, 3])
    phi = 0.27
    rotation = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    basis = np.diag([-1., -1., 1.]) @ oblique @ np.diag([x_sign, 1., 1.])
    if np.linalg.det(aff[:3, :3]) > 0:
        basis[:, 0] *= -1
    g = np.array([0.2, 0.3, 0.7]); g /= np.linalg.norm(g)
    moved = dwi.rotate_bvec(g, aff, rotation)
    # Synthetic tissue tensor in physical space: moving tensor R D R' must give the same attenuation.
    tissue = np.diag([0.0015, 0.0004, 0.0007])
    physical = basis @ g
    corrected = basis @ moved
    assert physical @ rotation @ tissue @ rotation.T @ physical == pytest.approx(corrected @ tissue @ corrected)
    assert np.linalg.norm(moved) == pytest.approx(1)
    np.testing.assert_allclose(dwi.rotate_bvec(np.zeros(3), aff, rotation), 0)


def test_motion_correction_returns_rotated_gradients(monkeypatch):
    import SimpleITK as sitk
    import dipy.segment.mask
    monkeypatch.setattr(dipy.segment.mask, "median_otsu", lambda b0, **kw: (b0, np.ones_like(b0, dtype=bool)))
    tx = sitk.Euler3DTransform(); tx.SetRotation(0., 0., .1)
    seen_seeds = []
    def registration(*args, **kwargs):
        seen_seeds.append(kwargs.get('sampling_seed'))
        return tx
    monkeypatch.setattr(dwi, "_register_volume", registration)
    gradients = np.array([[0., 1.], [0., 0.], [0., 0.]])
    ds = {"data": np.ones((3, 4, 5, 2), np.float32), "affine": np.eye(4), "bvals": np.array([0., 1000.]), "bvecs": gradients}
    out = dwi.preprocess(ds, sampling_seed=123)
    assert seen_seeds == [123, 123]
    assert out["bvecs_rotated"] and not np.allclose(out["bvecs"][:, 1], gradients[:, 1])
    np.testing.assert_allclose(out["bvecs"][:, 0], 0)
    np.testing.assert_allclose(ds["bvecs"], gradients)  # input dataset not mutated


def test_failed_free_water_fit_is_missing_not_a_prior_value(monkeypatch):
    import scipy.optimize
    def fail(*args, **kwargs):
        raise ValueError("synthetic nonconvergence")
    monkeypatch.setattr(scipy.optimize, "least_squares", fail)
    b = np.r_[0, np.full(6, 1000.)]
    g = np.c_[np.zeros(3), np.eye(3), -np.eye(3)]
    data = np.full((1, 1, 1, 7), 50.); data[..., 0] = 100.
    fw, fa = dwi._fw_single_shell(data, b, g, np.ones((1, 1, 1), bool))
    assert np.isnan(fw).all() and np.isnan(fa).all()


def _run(tmp, name, pe, affine, readout, n=7):
    path = tmp / name
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5, n), np.float32), affine), str(path) + ".nii.gz")
    np.savetxt(str(path) + ".bval", np.r_[0, np.full(n - 1, 1000)][None])
    np.savetxt(str(path) + ".bvec", np.c_[np.zeros(3), np.tile([[1.], [0.], [0.]], n - 1)])
    Path(str(path) + ".json").write_text(json.dumps({"PhaseEncodingDirection": pe, "TotalReadoutTime": readout}))
    return tuple(str(path) + ext for ext in (".nii.gz", ".bval", ".bvec", ".json"))


def test_run_assembly_checks_affines_and_extracts_only_reverse_b0(tmp_path):
    affine = np.diag([2., 2., 2., 1.])
    shifted = affine.copy(); shifted[0, 3] = 20
    runs = [_run(tmp_path, "main", "j", affine, .05, n=8),
            _run(tmp_path, "shifted", "j", shifted, .05),
            _run(tmp_path, "reverse", "j-", affine, .08)]
    ds = dwi.assemble(runs)
    assert ds["n_runs"] == 1 and ds["data"].shape[-1] == 8
    assert ds["rev_b0"].shape[-1] == 1  # six diffusion-weighted reverse volumes must not enter the b0 mean
    assert ds["rev_meta"]["TotalReadoutTime"] == .08
    other = _run(tmp_path, "other_readout", "j", affine, .12)
    assert dwi.assemble([runs[0], other])["n_runs"] == 1


def test_assemble_retains_fixel_features_and_blanks_all_failed_metrics(tmp_path):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01"] * 2,
                  "vol_Left_Putamen": [1., 1.]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    replacement = tmp_path / "dwi_v2"; replacement.mkdir()
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 8.], "n_sn_l": [4, 4], "n_sn_r": [4, 4],
                  "fa_wm_median": [.4, .4], "manufacturer": ["Siemens"] * 2, "shells": ["1000"] * 2,
                  "fw_method": ["singleshell_prior"] * 2, "topup": [True, False], "fs_image_id": ["I1", "I2"],
                  "acquisition_date": ["2022-01-03"] * 2, "sn_l_fw": [.2, np.nan], "sn_r_fw": [.3, .8],
                  "nst_afd_l": [.4, .5], "nst_seed_success_r": [.1, .2], "nst_fa_l": [.4, .5], "nst_n_streamlines_l": [20, 30]})\
        .to_csv(replacement / "dwi_features.csv", index=False)
    f = manifest.assemble_features(tmp_path, modality_dirs={"dwi": replacement}).set_index("PATNO")
    assert f.loc[1, "dwi_nst_afd_l"] == .4
    assert f.loc[1, "dwi_days_from_t1"] == 2 and f.loc[1, "dwi_date_source"] == "recorded"
    assert "fw_method" in f and "dwi_topup" in f
    cols = manifest.feature_blocks(f.columns)["dwi"]
    assert "dwi_nst_seed_success_r" in cols and "dwi_nst_n_streamlines_l" not in f
    assert f.loc[2, cols].isna().all()  # even though its first feature is already missing


def test_concurrent_saa_never_borrows_future_or_conflicting_assays(monkeypatch):
    saa = pd.DataFrame({"PATNO": [1, 1, 2, 3, 4, 4], "CLINICAL_EVENT": ["BL", "V04", "V04", "BL", "BL", "BL"],
                        "SAA_Status": ["Inconclusive", "Positive", "Positive", "Negative", "Positive", "Negative"]})
    monkeypatch.setattr(labels, "_latest", lambda *args: saa)
    sessions = pd.DataFrame({"patno": [1, 2, 3, 4], "image_id": ["I1", "I2", "I3", "I4"], "EVENT_ID": ["BL", "BL", "SC", "BL"]})
    result = labels.saa_labels("unused", sessions).set_index("PATNO")
    assert np.isnan(result.loc[1, "saa_positive"]) and 2 not in result.index
    assert result.loc[3, "saa_match"] == "screening_baseline_pair" and result.loc[3, "saa_positive"] == 0
    assert result.loc[4, "SAA_Status"] == "Conflicting" and np.isnan(result.loc[4, "saa_positive"])
    sensitivity = labels.saa_labels("unused", sessions, allow_unmatched=True).set_index("PATNO")
    assert sensitivity.loc[2, "saa_match"] == "unmatched_visit" and not sensitivity.loc[2, "saa_visit_concurrent"]


def test_saa_empty_result_has_mergeable_schema(monkeypatch):
    monkeypatch.setattr(labels, "_latest", lambda *args: pd.DataFrame(columns=["PATNO", "CLINICAL_EVENT", "SAA_Status"]))
    result = labels.saa_labels("unused", pd.DataFrame({"patno": [1], "image_id": ["I1"], "EVENT_ID": ["BL"]}))
    assert result.empty and {"PATNO", "IMAGEID", "saa_positive"} <= set(result)


def test_nm_hemispheres_do_not_depend_on_slab_voxel_order():
    shape = (5, 12, 14)
    fs = np.full(shape, 16); pauli = np.zeros(shape, dtype=int)
    pauli[2, 5:7, 2:4] = 7; pauli[2, 5:7, 10:12] = 7
    left = np.broadcast_to(np.arange(14)[None, None, :] < 7, shape)
    rois = nm.nm_rois(fs, pauli, (1., 1., 1.), left_mask=left)
    assert np.nonzero(rois["sn_l"])[2].max() == 3
    with pytest.raises(ValueError, match="hemisphere"):
        nm.nm_rois(fs, pauli, (1., 1., 1.))


def test_nm_shift_cannot_wrap_across_slab_edge():
    mask = np.zeros((3, 4, 5), bool); mask[0, 0, 0] = True
    assert not nm._shift_mask(mask, (-1, 0, 0)).any()
    assert nm._shift_mask(mask, (1, 0, 0))[1, 0, 0]


def test_qc_worst_means_least_negative_mi_not_best_registration():
    from pie.imaging.qc import worst_first
    frame = pd.DataFrame({"reg_b0_t1_mi": [-.9, -.1, -.5], "reg_metric": [-.9, -.1, -.5]})
    assert worst_first(frame, "reg_b0_t1_mi").index.tolist() == [1, 2, 0]
    assert worst_first(frame, "reg_metric").index.tolist() == [1, 2, 0]


def test_nested_cnn_fusion_never_trains_on_outer_test(monkeypatch):
    from pie.imaging import cnn
    calls = []
    def fake_train(X, y, tr, va, te, **kwargs):
        calls.append((set(tr), set(va), set(te)))
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        return np.full(len(te), .5), .5, 1
    monkeypatch.setattr(cnn, "train_fold", fake_train)
    n = 120
    result = cnn.cross_validate_fusion(np.zeros((n, 1)), np.tile([0, 1], n // 2), np.arange(n), np.arange(n)[:, None],
                                       n_splits=3, inner_splits=2, val_frac=.25, log=lambda *args: None)
    assert all(p.shape == (n,) and np.isfinite(p).all() for p in result)
    assert len(calls) == 9
    for start in range(0, 9, 3):
        outer_test = calls[start + 2][2]
        for train, valid, predicted in calls[start:start + 2]:
            assert not outer_test.intersection(train | valid | predicted)
