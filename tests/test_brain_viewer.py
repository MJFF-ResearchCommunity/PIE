"""Geometry and provenance guardrails for the local neuroimaging viewer."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pie.imaging.viewer.catalog import Catalog, Scan, iso_date
from pie.imaging.viewer.images import ImageStore, regions_from_atlas, write_volume
from pie.imaging.viewer.server import create_app


@pytest.fixture
def volume(tmp_path):
    # Deliberately asymmetric, oblique RAS geometry catches axis swaps/mirroring.
    affine = np.array([[-1.2, 0, 0, 22], [0, 1.7, -0.3, -11], [0, 0.3, 1.7, 7], [0, 0, 0, 1.]])
    data = np.arange(12 * 14 * 10, dtype=np.float32).reshape(12, 14, 10)
    path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    return path, data, affine


def test_masking_preserves_affine_and_inside_values(volume, tmp_path):
    path, data, affine = volume
    mask = np.zeros(data.shape, np.uint8)
    mask[2:9, 3:11, 1:8] = 1
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    result = write_volume(path, tmp_path / "masked.nii.gz", mask_path)
    np.testing.assert_allclose(result.affine, affine)
    np.testing.assert_array_equal(np.asarray(result.dataobj)[mask > 0], data[mask > 0])
    assert np.count_nonzero(np.asarray(result.dataobj)[mask == 0]) == 0
    np.testing.assert_array_equal(nib.load(path).get_fdata(), data)


def test_misaligned_mask_and_atlas_are_rejected(volume, tmp_path):
    path, data, affine = volume
    shifted = affine.copy()
    shifted[0, 3] += 4
    mask_path = tmp_path / "wrong-grid.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(data.shape, np.uint8), shifted), mask_path)
    with pytest.raises(ValueError, match="mask geometry"):
        write_volume(path, tmp_path / "output.nii.gz", mask_path)
    scan = Scan("s", "001", "MRI", "2020-01-01", "BL", "Test", path, "native", atlas=mask_path)
    with pytest.raises(ValueError, match="Atlas and displayed image"):
        ImageStore(tmp_path / "cache").prepare(scan)


def test_region_volume_uses_affine_determinant_and_world_coordinates(volume, tmp_path):
    _, data, affine = volume
    labels = np.zeros(data.shape, np.uint16)
    labels[1:4, 2:6, 3:5] = 12
    path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), path)
    region = regions_from_atlas(path)[0]
    assert region["id"] == 12
    assert region["voxels"] == 24
    assert region["volume_mm3"] == pytest.approx(24 * abs(np.linalg.det(affine[:3, :3])), abs=.01)
    np.testing.assert_allclose(region["center_mm"], nib.affines.apply_affine(affine, [2, 3.5, 3.5]), atol=.01)


def test_4d_conversion_preserves_frames_and_tr(volume, tmp_path):
    _, data, affine = volume
    source = tmp_path / "bold.nii.gz"
    image = nib.Nifti1Image(np.stack([data, data + 50, data + 100], axis=3), affine)
    image.header.set_zooms((1.2, 1.727, 1.727, 2.4))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, source)
    result = write_volume(source, tmp_path / "converted.nii.gz")
    assert result.shape == (*data.shape, 3)
    assert result.header.get_zooms()[3] == pytest.approx(2.4)
    assert result.header.get_xyzt_units()[1] == "sec"
    np.testing.assert_allclose(result.get_fdata()[..., 2], data + 100)


def test_region_focus_stays_inside_a_concave_label(tmp_path):
    labels = np.zeros((9, 9, 9), np.uint16)
    labels[2:7, 2:7, 2:7] = 12
    labels[3:6, 3:6, 3:6] = 0
    path = tmp_path / "hollow-region.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
    region = regions_from_atlas(path)[0]
    assert labels[tuple(np.rint(region["center_mm"]).astype(int))] == 0
    assert labels[tuple(np.rint(region["focus_mm"]).astype(int))] == 12


def test_masked_or_month_precision_dates_are_not_invented():
    assert iso_date("9999-01-01") is None
    assert iso_date("02/2011") is None
    assert iso_date("2011-02-30") is None
    assert iso_date("2/03/2011") == "2011-02-03"


def test_explicit_non_mm_geometry_is_not_silently_relabelled(volume, tmp_path):
    path, data, affine = volume
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("meter", "sec")
    nib.save(image, path)
    with pytest.raises(ValueError, match="millimetres"):
        write_volume(path, tmp_path / "converted.nii.gz")


def test_conversion_does_not_reduce_float64_precision(tmp_path):
    values = np.full((3, 3, 3), 0.123456789012345, dtype=np.float64)
    source = tmp_path / "precise.nii.gz"
    nib.save(nib.Nifti1Image(values, np.eye(4)), source)
    output = write_volume(source, tmp_path / "precise-copy.nii.gz")
    np.testing.assert_array_equal(np.asarray(output.dataobj), values)


def test_manifest_dates_must_be_iso_not_locale_strings(volume, tmp_path):
    path, _, _ = volume
    with pytest.raises(ValueError, match="real ISO dates"):
        Catalog(tmp_path, manifest=manifest(tmp_path, [{"id":"mri", "subject":"1", "modality":"MRI", "date":"01/02/2020", "path":str(path)}]))


def manifest(tmp_path, scans):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "scans": scans}))
    return path


def test_all_six_modalities_manifest_and_api(volume, tmp_path):
    from fastapi.testclient import TestClient
    path, _, _ = volume
    scans = [{"id": m, "subject": "001", "modality": m, "date": "2020-01-01", "path": str(path)} for m in ["MRI", "CT", "DTI", "SPECT", "PET", "fMRI"]]
    app = create_app(tmp_path, manifest=manifest(tmp_path, scans))
    client = TestClient(app)
    result = client.get("/api/catalog")
    assert result.status_code == 200
    assert len(result.json()["subjects"][0]["modalities"]) == 6
    # Multiple modalities on the same date are one acquisition date, not six visits.
    assert result.json()["subjects"][0]["dates"] == ["2020-01-01"]
    for s in scans:
        detail = client.get(f"/api/scans/{s['id']}")
        assert detail.status_code == 200
        assert detail.json()["scan"]["modality"] == s["modality"]
        url = detail.json()["volumes"][0]["url"]
        binary = client.get(url)
        assert binary.status_code == 200
        assert binary.content == path.read_bytes()
    assert client.get("/api/scans/missing").status_code == 404
    assert client.get("/api/assets/bogus/passwd").status_code == 404
    assert client.get("/api/catalog").headers["cache-control"] == "no-store"


def test_custom_atlas_does_not_inherit_freesurfer_names(volume, tmp_path):
    _, data, affine = volume
    labels = np.full(data.shape, 12, np.uint16)
    path = tmp_path / "custom-labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), path)
    assert regions_from_atlas(path)[0]["name"] == "Label 12"
    lut = tmp_path / "custom.tsv"
    lut.write_text("ID\tLabelName\tR\tG\tB\tA\n12\tCustom region\t100\t100\t200\t0\n")
    assert regions_from_atlas(path, lut)[0]["name"] == "Custom region"


@pytest.mark.parametrize("changes", [{"subject": "002"}, {"space": "wrong"}, {"reference_id": "missing"}])
def test_verified_overlay_requires_same_patient_space_and_reference(volume, tmp_path, changes):
    path, _, _ = volume
    scans = [{"id": "t1", "subject": "001", "modality": "MRI", "date": "2020-01-01", "path": str(path), "space": "subject-t1"},
             {"id": "pet", "subject": "001", "modality": "PET", "date": "2020-01-02", "path": str(path), "space": "subject-t1", "registration": "verified", "reference_id": "t1", **changes}]
    with pytest.raises(ValueError, match="same-subject reference_id"):
        Catalog(tmp_path, manifest=manifest(tmp_path, scans))


def test_projection_manifest_is_rejected(volume, tmp_path):
    path, _, _ = volume
    scans = [{"id": "raw", "subject": "001", "modality": "SPECT", "path": str(path), "kind": "projections"}]
    with pytest.raises(ValueError, match="reconstructed first"):
        Catalog(tmp_path, manifest=manifest(tmp_path, scans))
