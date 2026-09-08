"""Unit tests for the neuromelanin template pipeline: mask derivation on a synthetic template and CNR features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import nm_template as T


def _synthetic():
    """Template-like box: tissue 1.0, a bright band (1.2) 1 mm anterior-lateral of the atlas prior, a dark crus (0.85)
    4-8 mm anterior-lateral of it."""
    prior = T.sn_prior()
    x = np.arange(T.BOX_SHAPE[0])[:, None, None] * T.BOX_MM + T.BOX_ORIGIN_RAS[0]
    y = np.arange(T.BOX_SHAPE[1])[None, :, None] * T.BOX_MM + T.BOX_ORIGIN_RAS[1]
    X, Y = np.broadcast_to(x, T.BOX_SHAPE), np.broadcast_to(y, T.BOX_SHAPE)
    img = np.ones(T.BOX_SHAPE, np.float32)
    band = np.zeros(T.BOX_SHAPE, bool)
    for side in (X < 0, X >= 0):
        pr = prior & side
        band |= np.roll(np.roll(pr, 2, axis=1), 2 if (pr & (X >= 0)).any() else -2, axis=0)   # 1 mm anterior, 1 mm lateral
        sector = T._dilate_mm(pr, 8.0) & ~T._dilate_mm(pr, 4.0) & (Y > np.median(Y[pr])) & (np.abs(X) > np.median(np.abs(X[pr])))
        img[sector] = 0.85
    img[band] = 1.2
    return img, band, prior


def test_template_masks_find_band_and_crus():
    img, band, prior = _synthetic()
    m = T.template_masks(img)
    for side in ("l", "r"):
        sn, crus = m[f"sn_{side}"], m[f"crus_{side}"]
        assert sn.sum() > 200 and (sn & band).sum() / sn.sum() > 0.8, (side, sn.sum(), (sn & band).sum())
        assert crus.sum() > 200 and abs(img[crus].mean() - 0.85) < 0.02
        assert not (crus & band).any()


def test_template_features_recover_planted_contrast():
    img, band, prior = _synthetic()
    m = T.template_masks(img)
    rng = np.random.default_rng(0)
    subj = img * 500.0 + rng.normal(0, 25, img.shape).astype(np.float32)      # CV 0.05 around tissue
    subj[band] = 500.0 * 1.2 * 0.9 + rng.normal(0, 25, band.sum())            # this subject's band is 10 % dimmer
    out = T.template_features(subj, m)
    expected = (1.2 * 0.9) / 0.85 - 1                                          # band over crus mode
    assert abs(out["nmt_sn_mean_cnr"] - expected) < 0.05, (out["nmt_sn_mean_cnr"], expected)
    assert out["nmt_sn_cov_l"] > 0.99 and abs(out["nmt_crus_mode_l"] - 425) < 15
    assert np.isfinite(out["nmt_sn_post_mean_cnr"]) and np.isfinite(out["nmt_sn_lat_mean_cnr"])
    noise = np.ones(T.BOX_SHAPE, np.float32) * 500 + rng.normal(0, 60, T.BOX_SHAPE).astype(np.float32)
    out0 = T.template_features(noise, m)
    assert abs(out0["nmt_sn_mean_cnr"]) < 0.03                                 # no contrast invented from noise
