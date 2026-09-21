"""Striatal connectivity: CIT168 striatal regions, seed-to-network z, group ICA + dual regression recovery."""
import nibabel as nib
import numpy as np
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
