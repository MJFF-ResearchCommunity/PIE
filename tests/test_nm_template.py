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


def _box_xyz():
    x = np.arange(T.BOX_SHAPE[0])[:, None, None] * T.BOX_MM + T.BOX_ORIGIN_RAS[0]
    y = np.arange(T.BOX_SHAPE[1])[None, :, None] * T.BOX_MM + T.BOX_ORIGIN_RAS[1]
    z = np.arange(T.BOX_SHAPE[2])[None, None, :] * T.BOX_MM + T.BOX_ORIGIN_RAS[2]
    return [np.broadcast_to(a, T.BOX_SHAPE) for a in (x, y, z)]


def test_published_masks_split_territories_by_hemisphere():
    import nibabel as nib
    lab = np.zeros((81, 61, 51), np.int16)                   # 1 mm grid, x -40..40, y -50..10, z -35..15
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = (-40, -50, -35)
    for x0 in (-14, 10):                                     # sensorimotor (3) on both sides, 5 mm cubes
        lab[x0 + 40:x0 + 45, 30:35, 21:26] = 3
    lab[-3 + 40:4 + 40, 38:43, 21:26] = 4                    # background straddles the midline
    m = T.published_masks(nib.Nifti1Image(lab, aff))
    X, _, _ = _box_xyz()
    assert m["sensorimotor_l"].any() and (X[m["sensorimotor_l"]] < 0).all() and (X[m["sensorimotor_r"]] > 0).all()
    assert abs(m["sensorimotor_l"].sum() * T.BOX_MM ** 3 - 125) < 30
    assert (m["sn_l"] == (m["associative_l"] | m["limbic_l"] | m["sensorimotor_l"])).all() and not m["limbic_l"].any()
    assert abs(m["bnd"].sum() * T.BOX_MM ** 3 - 175) < 40 and (X[m["bnd"]] < 0).any() and (X[m["bnd"]] > 0).any()


def _territories():
    """Three 4 mm territory slabs per side (anterior to posterior: limbic, associative, sensorimotor) and a midline BND."""
    X, Y, Z = _box_xyz()
    zband = (Z >= -14) & (Z < -10)
    m = {"bnd": ((np.abs(X) < 4) & (Y >= -30) & (Y < -26) | (np.abs(np.abs(X) - 16) < 2) & (Y >= -10) & (Y < -6)) & zband}   # 3 parts
    for side, s in (("l", X < 0), ("r", X > 0)):
        lat = s & (np.abs(X) >= 8) & (np.abs(X) < 12) & zband
        m[f"limbic_{side}"], m[f"associative_{side}"], m[f"sensorimotor_{side}"] = (lat & (Y >= y) & (Y < y + 4) for y in (-14, -18, -22))
        m[f"sn_{side}"] = m[f"limbic_{side}"] | m[f"associative_{side}"] | m[f"sensorimotor_{side}"]
    return m


def test_published_features_read_territory_contrast_against_background():
    m = _territories()
    img = np.full(T.BOX_SHAPE, 90.0, np.float32)
    img[m["bnd"]] = 100
    for side, sm in (("l", 130), ("r", 120)):
        img[m[f"sensorimotor_{side}"]], img[m[f"associative_{side}"]], img[m[f"limbic_{side}"]] = sm, 150, 110
    out = T.published_features(img, m)
    assert np.isclose(out["nmb_sensorimotor_l_cnr"], 0.30) and np.isclose(out["nmb_sensorimotor_r_cnr"], 0.20)
    assert np.isclose(out["nmb_sensorimotor_mean_cnr"], 0.25) and np.isclose(out["nmb_sensorimotor_min_cnr"], 0.20)
    assert np.isclose(out["nmb_sn_l_cnr"], (130 + 150 + 110) / 300 - 1) and np.isclose(out["nmb_bnd_cov"], 1.0)

    cut = img.copy()                                         # half the right sensorimotor territory outside the slab
    X, Y, _ = _box_xyz()
    cut[m["sensorimotor_r"] & (Y < -20)] = 0
    out = T.published_features(cut, m)
    assert np.isnan(out["nmb_sensorimotor_r_cnr"]) and np.isnan(out["nmb_sensorimotor_mean_cnr"])   # both sides, or none
    assert np.isnan(out["nmb_sn_r_cnr"]) and np.isclose(out["nmb_sensorimotor_l_cnr"], 0.30)       # SN coverage 5/6 < 0.9

    cut = img.copy()
    cut[m["bnd"] & (X < 0)] = 0                              # background half missing: no reference, no contrast
    out = T.published_features(cut, m)
    assert all(np.isnan(v) for k, v in out.items() if k.endswith("_cnr"))


def test_published_features_need_every_background_component():
    m = _territories()
    X, Y, _ = _box_xyz()
    img = np.full(T.BOX_SHAPE, 90.0, np.float32)
    img[m["bnd"]] = 100
    img[m["sn_l"] | m["sn_r"]] = 120
    img[m["bnd"] & (X > 10)] = 0                              # one of three background parts outside the slab (~1/3 of BND)
    assert 0.5 < T.published_features(img, m)["nmb_bnd_cov"] < 0.9
    img2 = img.copy()
    img2[m["bnd"] & (X > 10)] = 100
    img2[m["bnd"] & (X > 10) & (Y < -9.5)] = 0               # the same part only nibbled: whole BND still >= 0.9 covered
    out = T.published_features(img2, m)
    assert out["nmb_bnd_cov"] >= 0.9 and np.isnan(out["nmb_sn_mean_cnr"])   # study rule: each part >= 0.9 on its own
