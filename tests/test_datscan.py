"""Unit tests for the DaTscan module that need no data: photopeak selection, reconstruction round-trip,
NIfTI geometry, and SBR arithmetic on a synthetic phantom."""

import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import datscan


def _window(lo, hi):
    r = Dataset()
    r.EnergyWindowLowerLimit, r.EnergyWindowUpperLimit = lo, hi
    e = Dataset()
    e.EnergyWindowRangeSequence = [r]
    return e


def test_photopeak_window_selection_handles_vendor_units():
    ds = Dataset()
    ds.EnergyWindowInformationSequence = [_window(108, 132), _window(143, 175)]
    assert datscan._photopeak_window(ds) == 2
    ds.EnergyWindowInformationSequence = [_window(14310, 17490)]  # Marconi-style x100 values
    assert datscan._photopeak_window(ds) == 1
    ds.EnergyWindowInformationSequence = []
    assert datscan._photopeak_window(ds) is None


def test_fbp_reconstruction_recovers_hot_spots():
    from skimage.transform import radon

    n, spacing = 64, 3.0
    yy, xx = np.mgrid[:n, :n]
    phantom = np.exp(-((xx - 24) ** 2 + (yy - 32) ** 2) / 18.0) * 5 + np.exp(-((xx - 40) ** 2 + (yy - 32) ** 2) / 18.0) * 2 + 0.3 * (((xx - 32) ** 2 + (yy - 32) ** 2) < 26**2)
    angles = np.arange(0, 360, 3.0)
    sino = radon(phantom, theta=angles, circle=True)           # (n_bins = n, n_angles): no padding, like a camera
    proj = np.zeros((len(angles), 3, sino.shape[0]), dtype=np.float32)
    proj[:, 1, :] = sino.T                                       # one informative axial slice
    vol = datscan.reconstruct(proj, angles, spacing, fwhm_mm=0.0)
    rec = vol[:, :, 1]
    assert rec.shape == (sino.shape[0], sino.shape[0])
    # the two hot spots are recovered at the right positions with the right intensity order
    a, b = rec[30:35, 22:27].mean(), rec[30:35, 38:43].mean()
    assert a > b > rec[5:10, 5:10].mean()
    assert 1.8 < a / b < 3.2


def test_to_nifti_geometry_and_sitk_roundtrip():
    vol = np.zeros((10, 12, 14), dtype=np.float32)
    vol[3, 4, 5] = 1.0
    img = datscan.to_nifti(vol, 2.5)
    assert np.allclose(np.diag(img.affine)[:3], 2.5)
    # reconstruction (row, col, slice) -> NIfTI (col, -row, -slice): rows run posterior, slices run inferior
    assert img.shape == (12, 10, 14) and img.get_fdata()[4, 6, 8] == 1.0
    # the physical point of that voxel must map back to the same voxel through the SimpleITK image
    sitk_img = datscan._sitk_from_nib(img)
    ras = img.affine @ np.array([4, 6, 8, 1.0])
    idx = sitk_img.TransformPhysicalPointToIndex((-float(ras[0]), -float(ras[1]), float(ras[2])))
    assert sitk_img[idx] == 1.0


def test_sbr_arithmetic_with_perfect_registration():
    """ROI arithmetic on co-registered arrays: SBRs equal the planted contrast; dilation keeps the ordering."""
    shape = (60, 60, 60)
    lab = np.zeros(shape, dtype=np.int16)
    lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
    lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
    lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
    lab[10:50, 10:50, 20:40][lab[10:50, 10:50, 20:40] == 0] = 2
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == datscan.PUTAMEN_L] = 3.0    # SBR 2.0
    counts[lab == datscan.PUTAMEN_R] = 1.5    # SBR 0.5
    out = datscan.sbr_from_arrays(counts, lab, dilate=0, ref_dilate=0, search_vox=0)
    assert abs(out["sbr_putamen_l"] - 2.0) < 1e-6 and abs(out["sbr_putamen_r"] - 0.5) < 1e-6
    out2 = datscan.sbr_from_arrays(counts, lab, dilate=1, ref_dilate=2, search_vox=2)
    assert out2["sbr_putamen_l"] > out2["sbr_putamen_r"] > 0 and out2["n_label_voxels"] > out["n_label_voxels"]


def test_registration_on_phantom_keeps_left_right_ordering():
    """End to end on a phantom with anterior/posterior asymmetry: after registration L must stay hotter than R."""
    import nibabel as nib

    shape = (60, 60, 60)
    lab = np.zeros(shape, dtype=np.int16)
    lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
    lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
    lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
    lab[10:50, 10:50, 20:40][lab[10:50, 10:50, 20:40] == 0] = 2
    lab[14:46, 12:20, 20:40] = 3
    counts = np.where(lab > 0, 1.0, 0.0).astype(np.float32)
    counts[lab == 3] = 0.6
    counts[lab == datscan.PUTAMEN_L] = 3.0
    counts[lab == datscan.PUTAMEN_R] = 1.5
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -60.0
    out = datscan.quantify(nib.Nifti1Image(counts, affine), nib.Nifti1Image(counts * 100, affine), nib.Nifti1Image(lab, affine))
    assert out["sbr_putamen_l"] > out["sbr_putamen_r"] > 0 and out["n_label_voxels"] > 0
