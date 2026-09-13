import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from pie.imaging.fmri import (classify_run, convert_archive_series,
                             framewise_displacement, inspect_nifti, phase_encoding_pair,
                             validate_motion_resume, command)


def test_command_with_spaced_working_directory(tmp_path):
    import sys
    work = tmp_path / "work with spaces"
    work.mkdir()
    command([sys.executable, "-c", "from pathlib import Path; assert Path.cwd().name == 'work with spaces'"],
            work / "command.log", cwd=work)


def test_resume_verifies_source_and_complete_transforms(tmp_path):
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    for name in ("trimmed", "motion"):
        nib.save(nib.Nifti1Image(data[..., 1:], np.eye(4)), tmp_path / (name + ".nii.gz"))
    np.savetxt(tmp_path / "motion.par", np.zeros((4, 6)))
    matrices = tmp_path / "motion.mat"
    matrices.mkdir()
    for i in range(4):
        np.savetxt(matrices / f"MAT_{i:04d}", np.eye(4))
    assert validate_motion_resume(source, tmp_path, 1)["transform_count"] == 4
    (matrices / "MAT_0003").unlink()
    with pytest.raises(ValueError, match="Incomplete resume motion transforms"):
        validate_motion_resume(source, tmp_path, 1)
    np.savetxt(matrices / "MAT_0003", np.eye(4))
    data[0, 0, 0, 2] += 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    with pytest.raises(ValueError, match="trimmed data differ"):
        validate_motion_resume(source, tmp_path, 1)


def test_resume_accepts_original_integer_header_quantization(tmp_path):
    data = np.arange(120, dtype=np.int16).reshape(2, 3, 4, 5)
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    img = nib.load(source)
    for name in ("trimmed", "motion"):
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32)[..., 1:],
                                img.affine, img.header), tmp_path / (name + ".nii.gz"))
    assert not np.array_equal(np.asarray(nib.load(tmp_path / "trimmed.nii.gz").dataobj), data[..., 1:])
    np.savetxt(tmp_path / "motion.par", np.zeros((4, 6)))
    (tmp_path / "motion.mat").mkdir()
    for i in range(4):
        np.savetxt(tmp_path / "motion.mat" / f"MAT_{i:04d}", np.eye(4))
    assert validate_motion_resume(source, tmp_path, 1)["trimmed_matches_source"]


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
    from pie.imaging.archives import ZipArchiveCache
    other_root = tmp_path / 'cached'
    other_root.mkdir()
    scratch_root = tmp_path / 'configured scratch filesystem'
    scratch_root.mkdir()
    with ZipArchiveCache() as cache:
        cached = convert_archive_series(row, other_root, dcm2niix=str(executable), archive_cache=cache,
                                        scratch_root=scratch_root)
        assert cached['outputs'] == result['outputs']
        assert cached['source_member_index_sha256'] == result['source_member_index_sha256']
        assert Path(cached['command'][-1]).parent == scratch_root
        assert Path(cached['command'][cached['command'].index('-o')+1]).parent.parent == other_root
        assert not list(scratch_root.iterdir())
    with pytest.raises(FileNotFoundError):
        convert_archive_series(row, other_root, dcm2niix=str(executable), scratch_root=tmp_path / 'absent')
    def failing_command(args, log, **kwargs):
        Path(log).write_text('synthetic converter diagnostic')
        raise RuntimeError('conversion failed')
    monkeypatch.setattr(fmri, 'command', failing_command)
    failed_root = tmp_path / 'failed'
    failed_root.mkdir()
    with pytest.raises(RuntimeError, match='retained diagnostic'):
        convert_archive_series(row, failed_root, dcm2niix=str(executable))
    assert not (failed_root / '123/I123').exists()
    logs = list((failed_root / 'conversion_failures/123/I123').glob('*.log'))
    assert len(logs) == 1 and logs[0].read_text() == 'synthetic converter diagnostic'
    assert not list(failed_root.glob('fmri-convert-*'))
    def missing_sidecar(args, log, **kwargs):
        fake_command(args, log, **kwargs)
        (Path(args[args.index('-o')+1]) / 'I123_reference.json').unlink()
    monkeypatch.setattr(fmri, 'command', missing_sidecar)
    invalid_root = tmp_path / 'invalid_metadata'
    invalid_root.mkdir()
    with pytest.raises(fmri.DICOMConversionError, match='retained diagnostic'):
        convert_archive_series(row, invalid_root, dcm2niix=str(executable), scratch_root=scratch_root)
    assert not (invalid_root / '123/I123').exists()
    evidence = list((invalid_root / 'conversion_failures/123/I123').glob('*.json'))
    assert len(evidence) == 1
    saved_failure = json.loads(evidence[0].read_text())
    assert 'required metadata sidecar' in saved_failure['error']
    assert len(list(Path(saved_failure['unvalidated_outputs']).glob('*.nii.gz'))) == 2
    assert not list(scratch_root.iterdir())
    (tmp_path / "123/I123/I123_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        convert_archive_series(row, tmp_path, dcm2niix=str(executable))
