"""Regression checks for the native-CIT168/MNI source-space error."""
import json
import shutil
import numpy as np
import pytest
from pie.imaging import atlases, dwi


def test_default_nigral_atlas_is_the_authors_mni2009c_projection():
    image = dwi.pauli_atlas()
    assert image.shape == (193, 229, 193)
    assert np.allclose(image.affine, [[1,0,0,-96],[0,1,0,-132],[0,0,1,-78],[0,0,0,1]])
    # Native atlas has 0.7-mm voxels, 198x263x212 shape and 334 mm3 SNc
    # after naive regridding; this same-version MNI projection has 365 mm3.
    labels = np.asarray(image.dataobj)
    assert (labels == 7).sum() == 365
    assert (labels == 9).sum() == 856
    assert atlases.cit168_metadata()['probability_threshold'] == .25


def test_rejects_a_different_registration_template():
    with pytest.raises(ValueError, match='spaces differ'):
        atlases.cit168_mni2009c(expected_space='MNI152NLin6Asym')


def test_rejects_replaced_atlas_bytes(tmp_path, monkeypatch):
    meta = atlases.cit168_metadata()
    (tmp_path/(atlases.CIT168_STEM+'.json')).write_text(json.dumps(meta))
    (tmp_path/meta['filename']).write_bytes(b'wrong atlas')
    monkeypatch.setattr(atlases, 'ATLAS_DIR', tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        atlases.cit168_mni2009c()
