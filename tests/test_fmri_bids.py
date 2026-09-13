import json

import nibabel as nib
import numpy as np
import pytest

from pie.imaging.fmri_bids import export_subject


@pytest.fixture
def inputs(tmp_path):
    paths = []
    for name, shape, pe in [('t1', (4, 4, 4), None),
                            ('bold', (4, 4, 4, 120), 'i'),
                            ('ref', (4, 4, 4, 10), 'i-')]:
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units('mm', 'sec')
        if len(shape) == 4:
            img.header.set_zooms((1, 1, 1, 3))
        nifti, sidecar = tmp_path / (name + '.nii.gz'), tmp_path / (name + '.json')
        nib.save(img, nifti)
        metadata = {} if pe is None else dict(RepetitionTime=3, EchoTime=.03,
                    PhaseEncodingDirection=pe, TotalReadoutTime=.02, SliceTiming=[0, 1, 2, 0])
        sidecar.write_text(json.dumps(metadata))
        paths += [nifti, sidecar]
    return paths


def test_verified_pair_and_raw_preservation(tmp_path, inputs):
    root = tmp_path / 'bids'
    result = export_subject(root, 'example', *inputs)
    assert result['correction_pair_exported']
    bold = root / 'sub-example/func/sub-example_task-rest_bold.nii.gz'
    assert bold.read_bytes() == inputs[2].read_bytes()
    fmap = root / 'sub-example/fmap/sub-example_dir-reverse_epi.nii.gz'
    assert nib.load(fmap).shape == (4, 4, 4)
    expected = nib.load(inputs[4]).get_fdata()[..., 4:10].mean(axis=-1)
    np.testing.assert_allclose(nib.load(fmap).get_fdata(), expected)
    assert len(list((root / 'sub-example/func').glob('*.nii.gz'))) == 1
    assert export_subject(root, 'example', *inputs) == result
    bold.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        export_subject(root, 'example', *inputs)


def test_missing_polarity_never_inferred(tmp_path, inputs):
    meta = json.loads(inputs[-1].read_text())
    del meta['PhaseEncodingDirection']
    meta['SeriesDescription'] = 'PA reverse AP'
    inputs[-1].write_text(json.dumps(meta))
    result = export_subject(tmp_path / 'bids', 'example', *inputs)
    assert not result['correction_pair_exported']
    assert 'missing_verified_phase_encoding' in result['unresolved_reasons']
    assert not (tmp_path / 'bids/sub-example/fmap').exists()


def test_short_reference_cannot_be_bold(tmp_path, inputs):
    with pytest.raises(ValueError, match='full 4D'):
        export_subject(tmp_path / 'bids', 'example', *inputs[:2], *inputs[4:])


def test_reference_exhausted_by_discard_is_not_forced_or_fatal(tmp_path, inputs):
    image = nib.load(inputs[4])
    data = image.get_fdata()
    image.header.set_zooms((1, 1, 1, 1))
    nib.save(nib.Nifti1Image(data, image.affine, image.header), inputs[4])
    meta = json.loads(inputs[5].read_text())
    meta['RepetitionTime'] = 1
    inputs[5].write_text(json.dumps(meta))
    result = export_subject(tmp_path / 'bids', 'example', *inputs)
    assert not result['correction_pair_exported']
    assert 'reference_has_no_post_initialization_volumes' in result['unresolved_reasons']
    assert not (tmp_path / 'bids/sub-example/fmap').exists()
    assert (tmp_path / 'bids/sub-example/func/sub-example_task-rest_bold.nii.gz').read_bytes() == inputs[2].read_bytes()


@pytest.mark.parametrize('kwargs', [{'discard_seconds': -1}, {'discard_seconds': float('nan')},
                                   {'reference_volumes': 0}, {'reference_volumes': 1.2}])
def test_invalid_averaging_settings(tmp_path, inputs, kwargs):
    with pytest.raises(ValueError):
        export_subject(tmp_path / 'bids', 'example', *inputs, **kwargs)
