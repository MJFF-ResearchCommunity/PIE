"""3D CNN baseline: volume loading from a synthetic FastSurfer subject, network shape, grouped out-of-fold CV on CPU."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.cnn import SHAPE, cache_volumes, cross_validate, load_volume, sfcn


def _subject(root, image_id, seed):
    rng = np.random.default_rng(seed)
    (root / image_id / "mri").mkdir(parents=True)
    img = np.zeros((256, 256, 256), np.float32)
    mask = np.zeros((256, 256, 256), np.uint8)
    mask[60:200, 70:210, 50:220] = 1                       # brain-sized box (140 x 140 x 170)
    img[mask > 0] = 100 + 20 * rng.standard_normal(int(mask.sum()))
    img[mask == 0] = 5                                      # non-zero background that the mask must remove
    aff = np.array([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128], [0, 0, 0, 1]], float)   # LIA like FastSurfer
    nib.save(nib.MGHImage(img, aff), root / image_id / "mri" / "orig_nu.mgz")
    nib.save(nib.MGHImage(mask, aff), root / image_id / "mri" / "mask.mgz")


def test_volume_and_cache(tmp_path):
    _subject(tmp_path, "I1", 0)
    v = load_volume(tmp_path, "I1")
    assert v.shape == SHAPE and v.dtype == np.float16
    inside = v[v != 0].astype(np.float32)
    assert abs(inside.mean()) < 0.05 and 0.2 < inside.std() < 1.1          # z-scored at 1 mm; 2x2x2 pooling of white noise leaves SD ~0.35
    assert 0.2 < (v != 0).mean() < 0.7                                       # brain occupies the middle of the box (real brains ~0.26)
    _subject(tmp_path, "I2", 1)
    X = cache_volumes(tmp_path, ["I1", "I2"], tmp_path / "vol.npy")
    assert X.shape == (2, *SHAPE) and np.allclose(X[0], v)
    X2 = cache_volumes(tmp_path, ["I1", "I2"], tmp_path / "vol.npy")       # reused, not rebuilt
    assert X2.shape == X.shape and (tmp_path / "vol.npy.ids").read_text().split() == ["I1", "I2"]


def test_network_and_cv():
    import torch
    net = sfcn()
    assert net(torch.zeros(2, 1, *SHAPE)).shape == (2, 1)
    rng = np.random.default_rng(0)
    n = 40
    y = np.tile([0, 1], n // 2)
    X = rng.standard_normal((n, 32, 32, 32)).astype(np.float16)                       # five 2x pools need >= 32 voxels
    X[y == 1, 8:16, 8:16, 8:16] += 3.0                                     # planted signal
    groups = np.arange(n) // 2                                               # two "sessions" per patient
    oof = cross_validate(X, y, groups, n_splits=4, epochs=3, device="cpu", val_frac=0.25, log=lambda *a: None)
    assert oof.shape == (n,) and np.all((oof >= 0) & (oof <= 1)) and np.isfinite(oof).all()


def test_pretrained_architecture_shape():
    import torch
    from pie.imaging.cnn import MNI_SHAPE, sfcn_pretrained
    net = sfcn_pretrained(weights=None)                                     # architecture only (weights may be absent)
    assert net(torch.zeros(1, 1, *MNI_SHAPE)).shape == (1, 1)
