import hashlib
import json
import nibabel as nib
import numpy as np
import pytest

from pie.imaging.viewer.catalog import Scan
from pie.imaging.viewer.fmri import summarize_bold
from pie.imaging.viewer.images import ImageStore
from scripts.prepare_viewer_fmri import bold_description, validate_bold


def test_descriptive_statistics_and_raw_dvars_preserve_all_frames():
    data = np.zeros((5, 6, 7, 3), np.float32)
    data[1:4, 1:5, 1:6, :] = [10, 12, 14]
    before = data.copy()
    maps, summary = summarize_bold(data)
    np.testing.assert_array_equal(data, before)
    assert maps["mean"][2, 2, 2] == 12
    assert maps["sd"][2, 2, 2] == pytest.approx(np.std([10, 12, 14]))
    assert maps["tsnr"][2, 2, 2] == pytest.approx(12 / np.std([10, 12, 14]))
    assert summary["mean_signal"] == [10, 12, 14]
    assert summary["raw_dvars"] == [None, 2, 2]
    assert summary["foreground_voxels"] == 60
    assert not maps["tsnr"][0, 0, 0]
    assert "not an anatomical brain mask" in summary["foreground_definition"]


def test_empty_constant_and_nonfinite_runs_do_not_invent_signal():
    maps, summary = summarize_bold(np.zeros((3, 3, 3, 4)))
    assert summary["mean_signal"] == [None] * 4
    assert summary["raw_dvars"] == [None] * 4
    assert summary["foreground_voxels"] == 0
    data = np.full((3, 3, 3, 4), 20.)
    data[0, 0, 0, 2] = np.nan
    maps, summary = summarize_bold(data)
    assert not maps["tsnr"].any()
    assert summary["excluded_nonfinite_voxels"] == 1
    assert summary["foreground_voxels"] == 26
    assert summary["raw_dvars"] == [None, 0, 0, 0]
    json.dumps(summary, allow_nan=False)


def test_prepared_maps_keep_native_affine_source_and_timing(tmp_path):
    path = tmp_path / "bold.nii.gz"
    affine = np.array([[2, .2, 0, -12], [0, 3, 0, 8], [0, .1, -4, 32], [0, 0, 0, 1.]])
    data = np.broadcast_to([10, 12, 14], (5, 6, 7, 3)).astype(np.int16).copy()
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm", "msec")
    image.header.set_zooms((*image.header.get_zooms()[:3], 2500))
    nib.save(image, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scan = Scan("bold-test", "001", "fMRI", "2020-03-04", "BL", "rsfMRI", path, "native-bold", kind="timeseries")
    store = ImageStore(tmp_path / "cache")
    prepared = store.prepare(scan)
    assert prepared["fmri"]["tr_seconds"] == 2.5
    assert prepared["geometry"]["time_unit"] == "msec"
    assert prepared["geometry"]["frames"] == 3
    assert prepared["scan"]["registration"] == "native"
    assert prepared["regions"] == []
    assert {m["key"] for m in prepared["extra"]} == {"mean", "sd", "tsnr"}
    for m in prepared["extra"]:
        derived = nib.load(store.assets[m["url"].split("/")[-2]])
        np.testing.assert_allclose(derived.affine, affine, atol=1e-6)
        assert derived.shape == data.shape[:3]
    assert store.prepare(scan) == prepared
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_import_requires_actual_bold_candidate_and_consistent_4d_timing(tmp_path):
    assert bold_description("R>L RESTING STATE FMRI ep2d_fid_basic_bold")
    for desc in ("DTI_revB0_AP", "2D GRE_MT", "T1 MPRAGE", "fMRI localizer"):
        assert not bold_description(desc)
    path = tmp_path / "candidate.nii.gz"
    image = nib.Nifti1Image(np.ones((3, 3, 3, 10), np.int16), np.eye(4))
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_zooms((1, 1, 1, 2.5)); nib.save(image, path)
    assert validate_bold(path, {"RepetitionTime": 2.5}).shape[3] == 10
    for metadata in ({}, {"RepetitionTime": 0}, {"RepetitionTime": 3}, {"RepetitionTime": float("nan")}):
        with pytest.raises(ValueError): validate_bold(path, metadata)
    with pytest.raises(ValueError): summarize_bold(np.zeros((3, 3, 3)))
