"""Tissue volumes from labels, head-size adjustment, and TIV by warped intracranial map."""
import nibabel as nib
import numpy as np
import pytest

from pie.imaging import features as vol


def test_tissue_volumes_sum_label_groups():
    seg = np.zeros((10, 10, 10), np.int32)
    seg[0:2] = 3; seg[2, :5] = 1005; seg[3] = 2; seg[4, 0, 0] = 16; seg[5, 0, 0:3] = 11
    out = vol.tissue_volumes(nib.Nifti1Image(seg, np.diag([2, 2, 2, 1])))
    assert out["cortical_gm"] == pytest.approx((200 + 50) * 8)
    assert out["white_matter"] == pytest.approx(100 * 8) and out["brainstem"] == pytest.approx(8)
    assert out["caudate_l"] == pytest.approx(24)
    assert out["total_gm"] == pytest.approx(out["cortical_gm"] + 24)


def test_adjustment_removes_head_size_and_fits_on_reference_only():
    rng = np.random.default_rng(0)
    tiv = rng.normal(1.5e6, 1.2e5, 200)
    region = 0.004 * tiv + rng.normal(0, 200, 200)
    adj = vol.adjust_for_head_size(region, tiv, "residual")
    assert abs(np.corrcoef(adj, tiv)[0, 1]) < 0.05 < abs(np.corrcoef(region, tiv)[0, 1])
    ref = np.arange(200) < 100
    shifted = region.copy(); shifted[~ref] += 1e6                           # evaluation rows must not move the fit
    np.testing.assert_allclose(vol.adjust_for_head_size(shifted, tiv, reference=ref)[ref], vol.adjust_for_head_size(region, tiv, reference=ref)[ref])


def test_tiv_scales_with_head_size():
    grid = np.indices((64, 64, 64)).astype(float) - 32
    head = lambda s: ((grid[0] / (20 * s)) ** 2 + (grid[1] / (24 * s)) ** 2 + (grid[2] / (18 * s)) ** 2 <= 1)
    template = nib.Nifti1Image(head(1.0).astype(np.float32) * 100, np.eye(4))
    icv = nib.Nifti1Image(head(0.9).astype(np.float32), np.eye(4))           # intracranial map inside the head
    small = vol.tiv_from_registration(nib.Nifti1Image(head(1.0).astype(np.float32) * 100, np.eye(4)), template, icv)
    large = vol.tiv_from_registration(nib.Nifti1Image(head(1.15).astype(np.float32) * 100, np.eye(4)), template, icv)
    assert small["qc_pass"] and large["qc_pass"]
    assert large["tiv_mm3"] / small["tiv_mm3"] == pytest.approx(1.15 ** 3, rel=0.08)
