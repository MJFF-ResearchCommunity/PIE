"""Alignment previews must not invent a reference, flip, resolution, or QC pass."""
import csv

import nibabel as nib
import numpy as np
import pytest

from pie.imaging.viewer.anatomy import AnatomyPreview, aligned_image, discover_previews, fixed_to_moving_ras, prepare_preview
from pie.imaging.viewer.catalog import Scan
from pie.imaging.viewer.images import ImageStore


def row(**changes):
    return {"reg_params": "0 0 0 10 20 30 1 1 1", "reg_center": "1 2 3", "flip_lr": "False", **changes}


def test_transform_direction_lps_ras_and_voxel_values():
    a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    affine = np.diag([2., 3., 4., 1.])
    source = nib.Nifti1Image(a, affine)
    out = aligned_image(source, row())
    # Stored translation is MRI->SPECT in LPS: invert it to place SPECT in MRI RAS.
    np.testing.assert_allclose(out.affine[:3, 3], [10, 20, -30])
    np.testing.assert_array_equal(np.asarray(out.dataobj), a)
    np.testing.assert_array_equal(np.asarray(source.dataobj), a)
    np.testing.assert_allclose(out.header.get_zooms(), [2, 3, 4])


def test_oblique_transform_maps_back_to_original_voxel_location():
    values = np.zeros((5, 6, 7), np.float32)
    affine = np.array([[0, -2, 0, 13], [3, 0, 0, -7], [0, 0, 4, 19], [0, 0, 0, 1.]])
    r = row(reg_params="0.1 0.2 0.3 12 -4 9 1.1 0.9 1.05")
    out = aligned_image(nib.Nifti1Image(values, affine), r)
    # NIfTI-1 stores sform elements in float32 (sub-micron rounding here).
    np.testing.assert_allclose(fixed_to_moving_ras(r) @ out.affine, affine, atol=1e-5)
    assert out.header["sform_code"] == 1
    assert out.header["qform_code"] == 0


def test_flip_is_only_the_explicit_pipeline_array_flip():
    a = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    source = nib.Nifti1Image(a, np.eye(4))
    np.testing.assert_array_equal(np.asarray(aligned_image(source, row(flip_lr="True")).dataobj), a[::-1])
    with pytest.raises(ValueError, match="explicit"):
        aligned_image(source, row(flip_lr=""))


@pytest.mark.parametrize("r", [row(reg_params="0 0"), row(reg_center="nan 0 0"), row(reg_params="2 0 0 0 0 0 1 1 1"), row(reg_params="0 0 0 0 0 0 -1 1 1")])
def test_invalid_registration_rejected(r):
    with pytest.raises(ValueError):
        fixed_to_moving_ras(r)


@pytest.fixture
def pair(tmp_path):
    path = tmp_path / "Imaging/derived/datscan_full/nifti/spect.nii.gz"
    path.parent.mkdir(parents=True)
    values = np.arange(125, dtype=np.float32).reshape(5, 5, 5) + 1
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)), mask)
    native = Scan("spect-I1", "001", "SPECT", "2020-01-01", "SC", "SPECT", path, "spect-native")
    reference = Scan("mri-I2", "001", "MRI", "2020-01-08", "BL", "MRI", path, "mri-native", mask=mask, atlas=mask)
    r = row(reg_params="0 0 0 0 0 0 1 1 1", image_id="I1", fs_image_id="I2", patno="001", nifti=str(path), error="")
    table = path.parent.parent / "datscan_sbr.csv"
    def write_rows(rows):
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(r))
            writer.writeheader()
            writer.writerows(rows)
    return native, reference, r, write_rows


def test_discovery_requires_exact_reference_same_subject_and_unique_row(pair, tmp_path):
    native, ref, r, write = pair
    scans = {native.id: native, ref.id: ref}
    write([r])
    assert list(discover_previews(tmp_path, scans)) == [native.id]
    write([{**r, "fs_image_id": "I999"}])
    assert not discover_previews(tmp_path, scans)
    write([{**r, "patno": "999"}])
    assert not discover_previews(tmp_path, scans)
    write([r, r])
    assert not discover_previews(tmp_path, scans)


def test_preview_is_unreviewed_keeps_both_dates_and_original_values(pair, tmp_path):
    native, ref, r, _ = pair
    store = ImageStore(tmp_path / "cache")
    result = prepare_preview(store, native, AnatomyPreview(native.path, ref, r))
    assert result["scan"]["registration"] == "unreviewed"
    assert result["scan"]["space"] == ref.space
    assert result["scan"]["date"] == "2020-01-01"
    assert result["context"]["reference_date"] == "2020-01-08"
    assert native.registration == "native" and native.space == "spect-native"
    assert [v["role"] for v in result["volumes"]] == ["anatomy", "primary", "atlas"]
    emitted = next(v for v in result["volumes"] if v["role"] == "primary")
    asset = store.assets[emitted["url"].split("/")[-2]]
    np.testing.assert_array_equal(nib.load(asset).get_fdata(), nib.load(native.path).get_fdata())


def test_preview_rejects_different_participant(pair, tmp_path):
    native, ref, r, _ = pair
    ref.subject = "999"
    with pytest.raises(ValueError, match="same participant"):
        prepare_preview(ImageStore(tmp_path / "cache"), native, AnatomyPreview(native.path, ref, r))


def test_preview_api_and_unknown_reference(pair, tmp_path):
    from fastapi.testclient import TestClient
    from pie.imaging.viewer.server import create_app
    native, ref, r, _ = pair
    app = create_app(tmp_path)
    app.state.catalog.scans.update({native.id: native, ref.id: ref})
    app.state.catalog.anatomy_previews[native.id] = AnatomyPreview(native.path, ref, r)
    client = TestClient(app)
    result = client.get(f"/api/scans/{native.id}/anatomy-preview")
    assert result.status_code == 200
    assert result.json()["context"]["status"] == "unreviewed"
    assert client.get(result.json()["volumes"][1]["url"]).status_code == 200
    assert client.get(f"/api/scans/{ref.id}/anatomy-preview").status_code == 404
    assert client.get("/api/scans/unknown/anatomy-preview").status_code == 404
