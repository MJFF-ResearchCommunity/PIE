"""JHU tract measures: label identity, laterality guard, per-tract means, and label mapping by registration."""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pie.imaging import dwi_tracts as dt


def _atlas(shape=(40, 40, 52)):
    """48 labels, one slice each, laid out so every _r label sits at +x (RAS) of its _l partner."""
    lab = np.zeros(shape, np.uint8)
    for k in range(1, 49):
        z = k + 1
        if k <= 6:
            lab[18:22, 5:8, z] = k
        elif k % 2:                     # right: high i under an identity (RAS) affine
            lab[28:32, 10:13, z] = k
        else:
            lab[8:12, 10:13, z] = k
    return nib.Nifti1Image(lab, np.eye(4))


def test_labels_cover_icbm_dti_81_with_named_sides():
    assert len(dt.LABELS) == 48 and dt.LABELS[43] == "superior_fronto_occipital_fasciculus_r"
    assert dt.LABELS[44] == "superior_fronto_occipital_fasciculus_l" and dt.LABELS[3] == "genu_corpus_callosum"


def test_laterality_guard_passes_correct_and_rejects_swapped_atlas():
    atlas = _atlas()
    assert all(v > 0 for v in dt.check_laterality(atlas).values())
    flipped = nib.Nifti1Image(np.asarray(atlas.dataobj)[::-1].copy(), np.eye(4))
    with pytest.raises(ValueError, match="swapped"):
        dt.check_laterality(flipped)


def test_tract_features_means_counts_and_fa_floor():
    atlas = _atlas()
    fa = np.full(atlas.shape, 0.1, np.float32)
    fa[np.asarray(atlas.dataobj) == 44] = 0.5
    fa[28:30, 10:13, 43 + 1] = 0.9          # part of label 43 (right SFOF) bright, the rest 0.1
    out = dt.tract_features({"fa": nib.Nifti1Image(fa, np.eye(4))}, atlas, min_voxels=3)
    assert out["fa_superior_fronto_occipital_fasciculus_l"] == pytest.approx(0.5)
    assert out["n_superior_fronto_occipital_fasciculus_l"] == 12
    floored = dt.tract_features({"fa": nib.Nifti1Image(fa, np.eye(4))}, atlas, min_voxels=3, fa_floor=0.2)
    assert floored["fa_superior_fronto_occipital_fasciculus_r"] == pytest.approx(0.9)
    assert np.isnan(floored["fa_genu_corpus_callosum"])      # all 0.1, below the floor


def test_registration_recovers_labels_on_a_shifted_subject():
    rng = np.random.default_rng(0)
    base = np.zeros((48, 48, 48), np.float32)
    base[10:38, 10:38, 10:38] = 0.3
    base[20:28, 14:34, 14:34] = 0.7                 # a bright "tract" block
    base += rng.normal(0, 0.01, base.shape).astype(np.float32)
    labels = np.zeros(base.shape, np.uint8)
    labels[20:28, 14:34, 14:34] = 44
    atlas_fa, atlas_lab = nib.Nifti1Image(base, np.eye(4)), nib.Nifti1Image(labels, np.eye(4))
    subject = nib.Nifti1Image(np.roll(base, (3, -2, 2), axis=(0, 1, 2)), np.eye(4))
    truth = np.roll(labels, (3, -2, 2), axis=(0, 1, 2)) == 44
    mapped, tx = dt.map_labels_to_subject(subject, atlas_fa, atlas_lab, syn=False)
    got = np.asarray(mapped.dataobj) == 44
    assert 2 * (got & truth).sum() / (got.sum() + truth.sum()) > 0.85
    qc = dt.registration_qc(subject, tx["warped_template_fa"], mapped, atlas_lab,
                            nib.Nifti1Image((np.asarray(subject.dataobj) > 0.15).astype(np.uint8), np.eye(4)))
    assert qc["template_fa_correlation"] > 0.9


@pytest.mark.skipif(not os.environ.get("PIE_ATLAS_CACHE"), reason="set PIE_ATLAS_CACHE to check the downloaded atlas")
def test_downloaded_atlas_verifies():
    labels, fa, prov = dt.fetch_jhu(Path(os.environ["PIE_ATLAS_CACHE"]) / "jhu", download=True)
    assert labels.shape == (182, 218, 182) and prov["laterality_right_minus_left_mm"]["superior_fronto_occipital_fasciculus"] > 30
