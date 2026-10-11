"""Striatal connectivity: CIT168 striatal regions, seed-to-network z, group ICA + dual regression recovery."""
import warnings

import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from pie.imaging import fmri_connectivity as fs


def test_striatal_rois_split_hemispheres_in_mni():
    img, names = fs.striatal_rois()
    data = np.asarray(img.dataobj)
    assert set(names.values()) == {"caudate_l", "caudate_r", "putamen_l", "putamen_r", "accumbens_l", "accumbens_r"}
    x = lambda k: nib.affines.apply_affine(img.affine, np.argwhere(data == k))[:, 0].mean()
    inv = {v: k for k, v in names.items()}
    assert x(inv["putamen_l"]) < -15 and x(inv["putamen_r"]) > 15 and x(inv["caudate_l"]) < 0 < x(inv["caudate_r"])


def test_seed_network_connectivity_finds_the_coupled_network():
    rng = np.random.default_rng(1)
    t = 300
    drive = rng.normal(size=t)
    seed = drive + rng.normal(0, 0.5, t)
    parcels = np.column_stack([drive + rng.normal(0, 0.7, t) for _ in range(4)] + [rng.normal(size=t) for _ in range(4)])
    out = fs.seed_network_connectivity(seed[:, None], ["caudate_l"], parcels, ["A"] * 4 + ["B"] * 4)
    assert out["caudate_l__A"] > 0.6 and abs(out["caudate_l__B"]) < 0.2


def test_group_ica_finds_striatal_network_and_dual_regression_ranks_subjects():
    rng = np.random.default_rng(2)
    v, t = 600, 150
    striatum = np.zeros(v, bool); striatum[:60] = True
    other = np.zeros(v, bool); other[300:380] = True
    maps_true = np.vstack([striatum * 1.0, other * 1.0])
    strength = np.linspace(0.3, 1.5, 8)                      # per-subject striatal network strength
    data = []
    for s in strength:
        tc = rng.normal(size=(t, 2))
        data.append(tc[:, :1] * s @ maps_true[:1] + tc[:, 1:] @ maps_true[1:] + rng.normal(0, 0.6, (t, v)))
    maps, info = fs.group_ica(data, n_components=4, seed=0)
    ranking = fs.select_component(maps, striatum)
    assert ranking[0]["dice"] > 0.7
    bgn = maps[ranking[0]["component"]]
    weights = [fs.roi_means(fs.dual_regression(d, maps)[1][ranking[0]["component"]],
                            striatum.astype(int), {1: "striatum"})["striatum"] for d in data]
    assert stats.spearmanr(weights, strength).correlation > 0.8
    assert bgn[striatum].mean() > bgn[~striatum].mean()


def test_dual_regression_reproduces_fsl():
    # Values from FSL 6 dual_regression's two calls on these data (fsl_glm -d maps --demean; then -d stage1
    # --demean --des_norm), radiological affine, float32 images as written to disk.
    rng = np.random.default_rng(0)
    shape, t, k = (5, 6, 4), 60, 3
    maps = rng.normal(size=shape + (k,)).astype(np.float32)
    data = (100 + np.einsum('xyzk,tk->xyzt', maps, rng.normal(size=(t, k))) * 3
            + rng.normal(size=shape + (t,))).astype(np.float32)
    mask = np.ones(shape, bool)
    tc, subject_maps = fs.dual_regression(data[mask].T, maps[mask].T)
    np.testing.assert_allclose(tc[[0, 59]], [[-4.073845263, 6.098976553, 2.966006253],
                                             [1.172204026, -3.426790079, -5.019422449]], atol=1e-6)
    volume = np.zeros(shape + (k,))
    volume[mask] = subject_maps.T
    np.testing.assert_allclose(volume[0, 0, 0], [0.4570210576, -0.411555171, 2.1404960155], atol=1e-5)
    np.testing.assert_allclose(volume[4, 5, 3], [0.5494698286, 2.9145517349, 3.0077719688], atol=1e-5)


def test_bold_functions_regress_the_nuisance_design_on_retained_frames():
    rng = np.random.default_rng(3)
    t, v = 120, 40
    keep = np.ones(t, bool)
    keep[[10, 11, 50]] = False
    drift = np.linspace(-1, 1, t)
    data = rng.normal(size=(t, v)) + 5 * drift[:, None]
    data[~keep] += 100                                                # censored frames must not leak in
    design = np.column_stack([np.ones(keep.sum()), drift[keep]])
    residual = data[keep] - design @ np.linalg.lstsq(design, data[keep], rcond=None)[0]
    maps = rng.normal(size=(3, v))
    for got, expected in zip(fs.dual_regression(data, maps, design=design, keep=keep), fs.dual_regression(residual, maps)):
        np.testing.assert_allclose(got, expected, atol=1e-10)
    z = fs.seed_network_connectivity(data[:, :2], ['a', 'b'], data[:, 2:], ['N'] * (v - 2), design=design, keep=keep)
    assert z == pytest.approx(fs.seed_network_connectivity(residual[:, :2], ['a', 'b'], residual[:, 2:], ['N'] * (v - 2)))
    with pytest.raises(ValueError, match='retained'):
        fs.dual_regression(data, maps, design=design)


def test_bgn_maps_from_the_analysed_sample_warn_and_a_frozen_reference_is_used(tmp_path):
    rng = np.random.default_rng(2)
    shape = (6, 10, 10)
    mask_img = nib.Nifti1Image(np.ones(shape, np.uint8), np.eye(4))
    striatum = np.zeros(shape, bool)
    striatum[:, :2] = True
    reference = np.stack([striatum * 3.0, np.roll(striatum, 5, axis=1) * 3.0], axis=-1)
    nib.save(nib.Nifti1Image(reference, np.eye(4)), tmp_path / 'maps.nii.gz')
    template, labels = striatum.ravel(), striatum.ravel().astype(int)
    data = [rng.normal(size=(80, 1)) * s @ striatum.reshape(1, -1) + rng.normal(size=(80, striatum.size))
            for s in (0.5, 1.0, 2.0)]
    loaded = fs.load_reference_maps(tmp_path / 'maps.nii.gz', mask_img)
    assert loaded[0].shape == (2, striatum.size) and loaded[1]['n_components'] == 2
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        weights, provenance = fs.bgn_weights(data, template, labels, {1: 'striatum'}, reference=loaded)
    assert provenance['source'] == 'frozen_reference' and provenance['component'] == 0
    assert provenance['sha256'] == loaded[1]['sha256']
    assert [w['striatum'] for w in weights] == sorted(w['striatum'] for w in weights)
    with pytest.warns(UserWarning, match='circular'):
        _, provenance = fs.bgn_weights(data, template, labels, {1: 'striatum'}, n_components=3)
    assert provenance['source'] == 'analysed_sample'
    with pytest.raises(ValueError, match='checksum'):
        fs.load_reference_maps(tmp_path / 'maps.nii.gz', mask_img, expected_sha256='0' * 64)


def test_putamen_splits_at_the_coronal_plane():
    img, names = fs.striatal_rois(putamen_split_y=0.0)
    data = np.asarray(img.dataobj)
    inv = {v: k for k, v in names.items()}
    y = lambda k: nib.affines.apply_affine(img.affine, np.argwhere(data == k))[:, 1]
    assert set(names.values()) >= {"putamen_anterior_l", "putamen_posterior_l", "putamen_anterior_r", "putamen_posterior_r"}
    assert y(inv["putamen_posterior_l"]).max() < 0 <= y(inv["putamen_anterior_r"]).min()
