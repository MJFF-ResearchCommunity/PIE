import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from pie.imaging.fmri import (classify_run, convert_archive_series,
                             framewise_displacement, inspect_nifti, phase_encoding_pair)


@pytest.mark.parametrize("shape,tr,expected", [
    ((64, 64, 40, 10), 2.5, "short_reference_candidate"),
    ((64, 64, 40, 240), 2.5, "rest_candidate"),
    ((64, 64, 40, 240), None, "unknown_timing"),
    ((64, 64, 40, 240), 0, "unknown_timing"),
    ((64, 64, 40, 240), 1, "insufficient_duration"),
    ((64, 64, 40), 2.5, "not_functional_4d"),
    ((64, 64, 40, 10, 2), 2.5, "not_functional_4d"),
])
def test_actual_dimensions_not_dicom_count(shape, tr, expected):
    assert classify_run(shape, tr) == expected


def test_fd_units_and_column_order():
    p = np.zeros((3, 6))
    p[1, 0] = .01
    p[2, 0] = .01
    p[2, 3] = 2
    np.testing.assert_allclose(framewise_displacement(p), [0, .5, 2])


def test_nonfinite_motion_fails():
    with pytest.raises(ValueError):
        framewise_displacement(np.full((4, 6), np.nan))


def test_sidecar_tr_must_agree(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 30)), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1, 1, 1, 2.5))
    nii, js = tmp_path / "test.nii.gz", tmp_path / "test.json"
    nib.save(img, nii)
    js.write_text(json.dumps({"RepetitionTime": 1.0}))
    with pytest.raises(ValueError, match="TR disagreement"):
        inspect_nifti(nii, js)
    js.write_text(json.dumps({"RepetitionTime": 2.5}))
    assert inspect_nifti(nii, js)["tr_seconds"] == 2.5


def test_no_inference_from_description():
    p = {"shape": [2, 2, 2, 240], "affine": np.eye(4), "series_description": "AP"}
    q = dict(p, series_description="PA")
    assert "missing_verified_phase_encoding" in phase_encoding_pair(p, q)
    assert "missing_verified_readout_time" in phase_encoding_pair(p, q)
    p.update(phase_encoding="j", total_readout_time=.03)
    q.update(phase_encoding="j-", total_readout_time=.03)
    assert not phase_encoding_pair(p, q)
    q["affine"] = np.diag([2, 2, 2, 1])
    assert "geometry_requires_explicit_reconciliation" in phase_encoding_pair(p, q)


def test_unsafe_prefix_rejected_before_extraction(tmp_path):
    with pytest.raises(ValueError, match="Unsafe archive prefix"):
        convert_archive_series({"PATNO": "123", "image_id": "I123",
                                "series_prefix": "PPMI/123/../I123/"}, tmp_path)


def test_inventory_mismatch_rejected(tmp_path):
    z = tmp_path / "test.zip"
    with zipfile.ZipFile(z, "w") as archive:
        archive.writestr("PPMI/123/series/I123/file.dcm", b"not dicom")
    with pytest.raises(ValueError, match="member count differs"):
        convert_archive_series({"PATNO": "123", "image_id": "I123", "archive": str(z),
                                "series_prefix": "PPMI/123/series/I123/", "dicom_entries": 2}, tmp_path)


def test_conversion_preserves_all_outputs_and_checks_resume(tmp_path, monkeypatch):
    import io
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    import pie.imaging.fmri as fmri

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, enforce_file_format=True)
    archive_path = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("PPMI/123/series/I123/first.dcm", buffer.getvalue())
    before = fmri.sha256(archive_path)
    executable = tmp_path / "converter"
    executable.write_text("mock converter identity")

    def fake_command(args, log, **kwargs):
        out = Path(args[args.index("-o") + 1])
        for name, n in (("I123_long", 240), ("I123_reference", 10)):
            img = nib.Nifti1Image(np.zeros((2, 2, 2, n), np.float32), np.eye(4))
            img.header.set_xyzt_units("mm", "sec")
            img.header.set_zooms((1, 1, 1, 2.5))
            nib.save(img, out / (name + ".nii.gz"))
            (out / (name + ".json")).write_text(json.dumps({"RepetitionTime": 2.5}))
        Path(log).write_text("test converter output")

    from pathlib import Path
    monkeypatch.setattr(fmri, "command", fake_command)
    row = {"PATNO": "123", "image_id": "I123", "archive": str(archive_path),
           "series_prefix": "PPMI/123/series/I123/", "dicom_entries": 1}
    result = convert_archive_series(row, tmp_path, dcm2niix=str(executable))
    assert {o["run_class"] for o in result["outputs"]} == {"rest_candidate", "short_reference_candidate"}
    assert result["selected_members_crc_checked"]
    assert fmri.sha256(archive_path) == before
    assert not list(tmp_path.glob("fmri-convert-*"))
    assert convert_archive_series(row, tmp_path, dcm2niix=str(executable)) == result
    (tmp_path / "123/I123/I123_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        convert_archive_series(row, tmp_path, dcm2niix=str(executable))
