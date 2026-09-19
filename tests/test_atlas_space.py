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


def test_actual_registration_reference_matches_atlas_space():
    from nilearn import datasets
    template = atlases.mni2009c_template()
    assert atlases.mni2009c_template_metadata()['space'] == atlases.cit168_metadata()['space']
    assert template.shape == (97, 115, 97)
    assert np.allclose(template.affine[:3, 3], [-96.5,-132.5,-78.5])
    # Nilearn's default is a distinct 2009a image; matching the label filename
    # alone cannot validate the reference actually used by registration.
    default = datasets.load_mni152_template(resolution=2)
    assert default.shape != template.shape or not np.allclose(default.affine, template.affine)


def test_legacy_cache_is_rejected_and_preserved(tmp_path):
    import SimpleITK as sitk
    path = tmp_path/'legacy.tfm'
    sitk.WriteTransform(sitk.AffineTransform(3),str(path))
    original=path.read_bytes()
    with pytest.raises(ValueError,match='Unverified'):
        dwi.load_mni_cache(path)
    assert path.read_bytes()==original


def test_cache_checks_reference_transform_and_participant_inputs(tmp_path):
    import SimpleITK as sitk
    path=tmp_path/'current.tfm'
    sitk.WriteTransform(sitk.AffineTransform(3),str(path))
    inputs={'t1_sha256':'a','mask_sha256':'b','sampling_seed':0}
    dwi._write_mni_cache_provenance(path,inputs)
    assert dwi.load_mni_cache(path,expected_inputs=inputs).TransformPoint((1.,2.,3.))==(1.,2.,3.)
    with pytest.raises(ValueError,match='provenance mismatch'):
        dwi.load_mni_cache(path,expected_inputs={**inputs,'t1_sha256':'changed'})
    record=json.loads(path.with_suffix('.json').read_text());record['reference_space']='MNI152NLin2009aSym'
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='provenance mismatch'):
        dwi.load_mni_cache(path)


def test_current_atlas_hash_alone_does_not_authorize_old_nm_mapping(tmp_path):
    from pie.imaging import nm
    from test_imaging_regressions import _nm_subject,_box_sn
    _nm_subject(tmp_path,_box_sn())
    out=nm._refeature_job((str(tmp_path/'nm'),{'patno':1,'error':'','atlas_sha256':atlases.cit168_provenance()['atlas_sha256']}))
    assert 'reference' in out['error']
