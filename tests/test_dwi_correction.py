"""Fast correction-contract tests: no FSL execution or patient-image fitting."""
import nibabel as nib
import numpy as np
import pytest

from pie.imaging import dwi_correction as correction


def test_selection_does_not_borrow_b0_or_pool_other_acquisitions():
    runs = [{'path': 'b1000_no_b0', 'bvals': np.full(32, 1000)},
            {'path': 'b700', 'bvals': np.r_[0, np.full(12, 700)]},
            {'path': 'complete', 'bvals': np.r_[0, np.full(12, 1000), np.full(20, 2000)]}]
    selected, mask, shell = correction.choose_tensor_acquisition(runs)
    assert selected['path'] == 'complete' and shell == 1000
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(13))
    with pytest.raises(ValueError, match='own b0'):
        correction.choose_tensor_acquisition(runs[:1])


@pytest.mark.parametrize('shape,expected', [((128, 128, 80), 'b02b0.cnf'),
                                          ((128, 128, 75), 'b02b0_1.cnf')])
def test_topup_preserves_odd_and_even_grids(shape, expected):
    assert correction.topup_config(shape) == expected


def test_command_has_no_guessed_slice_groups_or_second_correction():
    meta = {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': .06,
            'SliceTiming': [0, 1, 2, 0, 1], 'MultibandAccelerationFactor': 2}
    argv, record = correction.build_eddy_command('/fsl/bin', meta, 5,
                                                use_topup=True, raw_is_uncorrected=True)
    assert argv[0] == '/fsl/bin/eddy_cuda'
    assert '--topup=topup' in argv and '--ol_type=sw' in argv and '--mporder=0' in argv
    assert not any(arg.startswith(('--json=', '--mb=', '--slspec=')) for arg in argv)
    assert record['slice_timing'] == 'unused_unverified'
    with pytest.raises(ValueError, match='twice'):
        correction.build_eddy_command('/fsl/bin', meta, 5, raw_is_uncorrected=False)


def test_missing_metadata_does_not_silently_fall_back_or_guess():
    with pytest.raises(ValueError, match='no guessed metadata'):
        correction.build_eddy_command('/fsl/bin', {'PhaseEncodingDirection': 'j',
                                      'EstimatedTotalReadoutTime': .06}, 5, raw_is_uncorrected=True)
    assert correction.slice_options({'SliceTiming': None}, 5)[0] == []
    assert correction.slice_options({'SliceTiming': ['unknown']}, 1)[0] == []
    valid = {'SliceTiming': [0, 1, 2, 0, 1, 2], 'MultibandAccelerationFactor': 2}
    assert correction.slice_options(valid, 6)[0] == ['--json=metadata.json']


def test_corrected_geometry_rejects_shifted_grid_and_bad_gradients():
    raw = nib.Nifti1Image(np.ones((3, 4, 5, 2), np.float32), np.eye(4))
    b = np.array([0., 1000.]); v = np.array([[0., 1.], [0., 0.], [0., 0.]])
    correction.validate_corrected_geometry(raw, raw, b, v)
    affine = np.eye(4); affine[0, 3] = 1
    shifted = nib.Nifti1Image(np.ones(raw.shape, np.float32), affine)
    with pytest.raises(ValueError, match='affine'):
        correction.validate_corrected_geometry(raw, shifted, b, v)
    with pytest.raises(ValueError, match='unit length'):
        correction.validate_corrected_geometry(raw, raw, b, v * 2)
