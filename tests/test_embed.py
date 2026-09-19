"""Pretrained-model embeddings: MNI resampling of a synthetic FastSurfer subject, forward-pass shapes of the three
backends with random weights, and (when the weights are present) that the real checkpoints load."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.embed import DIM, EMBED, GRID, WEIGHTS, brainiac_net, columns, load_net, sfcn_net, simclr_net, to_mni

NETS = {"brainiac": brainiac_net, "simclr": simclr_net, "sfcn": sfcn_net}


def _subject(root, image_id, seed=0):
    """A 256^3 LIA FastSurfer-like subject with an ellipsoidal 'brain' roughly where a head sits in the conformed box."""
    rng = np.random.default_rng(seed)
    (root / image_id / "mri").mkdir(parents=True)
    i, j, k = np.mgrid[:256, :256, :256]
    mask = (((i - 128) / 70) ** 2 + ((j - 120) / 85) ** 2 + ((k - 128) / 75) ** 2 <= 1).astype(np.uint8)
    img = np.where(mask > 0, 100 + 20 * rng.standard_normal(mask.shape), 5).astype(np.float32)
    aff = np.array([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128], [0, 0, 0, 1]], float)
    nib.save(nib.MGHImage(img, aff), root / image_id / "mri" / "orig_nu.mgz")
    nib.save(nib.MGHImage(mask, aff), root / image_id / "mri" / "mask.mgz")


def test_to_mni_shape_and_masking(tmp_path):
    _subject(tmp_path, "I1")
    shape, origin = GRID["sfcn"]
    v = to_mni(tmp_path, "I1", shape, origin)
    assert v.shape == shape and v.dtype == np.float32
    from pie.imaging.dwi import mni_cache_path
    assert mni_cache_path(tmp_path / "I1").exists()                                            # affine cached for reuse
    inside = v[v != 0]
    assert 0.3 < (v != 0).mean() < 0.9 and 80 < inside.mean() < 120 and inside.max() < 200      # background stays exactly 0
    v2 = to_mni(tmp_path, "I1", shape, origin)                                                  # cached transform: identical
    assert np.array_equal(v, v2)


@pytest.mark.parametrize("name", list(EMBED))
def test_forward_shapes_random_weights(name):
    shape, _ = GRID[name]
    rng = np.random.default_rng(0)
    vol = np.zeros(shape, np.float32)
    vol[20:-20, 20:-20, 20:-20] = 100 + 10 * rng.standard_normal(tuple(s - 40 for s in shape))
    out = EMBED[name](vol, NETS[name]().eval())
    assert out.shape == (len(columns(name)),) and np.isfinite(out).all()
    if name == "sfcn":
        assert out.shape == (DIM["sfcn"] + 1,) and 42 <= out[-1] <= 82


@pytest.mark.parametrize("name", list(EMBED))
def test_real_weights_load(name):
    if not WEIGHTS[name].exists():
        pytest.skip(f"{name} weights not downloaded")
    net = load_net(name, "cpu")
    assert not net.training
