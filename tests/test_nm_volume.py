"""Neuromelanin volume and normalised intensity on a phantom with known answers."""
import nibabel as nib
import numpy as np
import pytest

from pie.imaging import nm as nv


def _phantom():
    rng = np.random.default_rng(0)
    nm = rng.normal(100, 5, (30, 30, 10))
    sn = np.zeros(nm.shape, bool); sn[10:14, 10:15, 4:6] = True          # 40 voxels
    nm[sn] = 160
    search = np.zeros(nm.shape, bool); search[8:16, 8:17, 3:7] = True
    ref = np.zeros(nm.shape, bool); ref[20:28, 5:25, 2:8] = True
    aff = np.diag([0.5, 0.5, 2.0, 1])                                      # 0.5 mm^3 voxels
    img = lambda a: nib.Nifti1Image(a.astype(np.float32), aff)
    return img(nm), img(sn), img(search), img(ref)


def test_hyperintense_volume_recovers_the_bright_region():
    nm, sn, search, ref = _phantom()
    out = nv.hyperintense_volume(nm, search, ref, k=3.0)
    assert out["n_voxels"] == 40 and out["volume_mm3"] == pytest.approx(20.0)
    assert nv.mask_volume(sn) == pytest.approx(20.0)


def test_normalised_intensity_and_grid_guard():
    nm, sn, search, ref = _phantom()
    out = nv.normalised_intensity(nm, sn, ref)
    assert out["normalised_intensity"] == pytest.approx(1.6, abs=0.02)
    with pytest.raises(ValueError, match="grid"):
        nv.hyperintense_volume(nm, nib.Nifti1Image(np.asarray(search.dataobj), np.eye(4)), ref)
