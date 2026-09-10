import hashlib
import nibabel as nib
import numpy as np
import pytest
from pie.imaging.viewer.catalog import Scan
from pie.imaging.viewer.comparison import prepare_comparison
from pie.imaging.viewer.images import ImageStore, signal_histogram


def test_histogram_is_bounded_finite_and_does_not_change_signal():
    values = np.array([0, 1, 2, 3, np.nan, np.inf])
    h = signal_histogram(values, "test")
    assert h["sample_count"] == 3 and sum(h["counts"]) == 3
    assert len(h["edges"]) == len(h["counts"]) + 1
    assert values[0] == 0 and np.isnan(values[4])
    assert signal_histogram(np.zeros(12), "test") is None
    assert signal_histogram(np.arange(800_000), "test")["sample_count"] <= 250_000


def test_comparison_rejects_wrong_patient_modality_and_date(tmp_path):
    scan = Scan("a", "001", "MRI", "2020-01-01", "BL", "MRI", tmp_path / "a.nii", "a")
    for changes in ({"subject": "002"}, {"modality": "SPECT"}, {"date": None}, {"date": "2020-01-01"}, {"date": "2019-01-01"}):
        from dataclasses import replace
        other = replace(scan, id="b", date="2021-01-01", **{k:v for k,v in changes.items() if k != "date"})
        if "date" in changes: other.date = changes["date"]
        with pytest.raises(ValueError, match="earlier and later MRI"):
            prepare_comparison(ImageStore(tmp_path / "cache"), scan, other)


def test_rigid_preview_retains_dates_sources_and_unreviewed_status(tmp_path):
    grid = np.indices((48, 48, 48)).astype(float)
    data = (100 * np.exp(-sum((grid[i] - [23, 19, 27][i])**2 for i in range(3))/100)
            + 50 * np.exp(-sum((grid[i] - [13, 31, 16][i])**2 for i in range(3))/50)).astype(np.float32)
    paths = [tmp_path / f"{i}.nii.gz" for i in range(2)]
    for p in paths:
        image = nib.Nifti1Image(data, np.diag([2., 2., 2., 1.])); image.header.set_xyzt_units("mm")
        nib.save(image, p)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    scans = [Scan(str(i), "001", "MRI", f"202{i}-01-01", "visit", "MRI", p, str(i)) for i,p in enumerate(paths)]
    store = ImageStore(tmp_path / "cache")
    result = prepare_comparison(store, *scans)
    assert result["registration"]["status"] == "unreviewed"
    assert result["followup"]["scan"]["date"] == "2021-01-01"
    assert result["followup"]["scan"]["registration"] == "unreviewed"
    assert result["followup"]["regions"] == []
    assert scans[1].registration == "native"
    np.testing.assert_allclose(result["baseline"]["geometry"]["affine"], result["followup"]["geometry"]["affine"])
    assert result == prepare_comparison(store, *scans)
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
