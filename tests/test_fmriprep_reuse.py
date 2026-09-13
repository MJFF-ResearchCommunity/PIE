import json

import pytest

from pie.imaging.fmriprep_reuse import snapshot_anatomical_derivatives


def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'sub-A01/anat').mkdir(parents=True)
    (root / 'sub-A01/func').mkdir()
    (root / 'dataset_description.json').write_text(json.dumps({'DatasetType': 'derivative'}))
    (root / 'sub-A01/anat/sub-A01_desc-preproc_T1w.nii.gz').write_bytes(b'anatomical')
    (root / 'sub-A01/func/sub-A01_task-rest_bold.nii.gz').write_bytes(b'functional')
    return root


def test_anatomy_only_exact_copy_and_checked_resume(tmp_path):
    root = source(tmp_path)
    target = tmp_path / 'snapshot'
    record = snapshot_anatomical_derivatives(root, target, ['A01'])
    assert record['anatomical_only'] and not record['scientific_qc_pass']
    assert not list(target.rglob('*bold*'))
    assert snapshot_anatomical_derivatives(root, target, ['A01']) == record
    image = target / 'sub-A01/anat/sub-A01_desc-preproc_T1w.nii.gz'
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        snapshot_anatomical_derivatives(root, target, ['A01'])


def test_functional_contamination_and_symlinks_rejected(tmp_path):
    root = source(tmp_path)
    bad = root / 'sub-A01/anat/sub-A01_from-boldref_to-T1w_xfm.txt'
    bad.write_bytes(b'functional transform')
    with pytest.raises(ValueError, match='Functional'):
        snapshot_anatomical_derivatives(root, tmp_path / 'snapshot', ['A01'])
    bad.unlink()
    (root / 'sub-A01/anat/link').symlink_to(root / 'sub-A01/func/sub-A01_task-rest_bold.nii.gz')
    with pytest.raises(ValueError, match='Symlink'):
        snapshot_anatomical_derivatives(root, tmp_path / 'snapshot', ['A01'])


def test_extra_file_in_snapshot_and_changed_source_rejected(tmp_path):
    root = source(tmp_path)
    target = tmp_path / 'snapshot'
    snapshot_anatomical_derivatives(root, target, ['A01'])
    (target / 'unexpected_bold.nii.gz').write_bytes(b'functional')
    with pytest.raises(ValueError, match='Unexpected'):
        snapshot_anatomical_derivatives(root, target, ['A01'])
    (root / 'sub-A01/anat/sub-A01_desc-preproc_T1w.nii.gz').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='identity'):
        snapshot_anatomical_derivatives(root, target, ['A01'])
