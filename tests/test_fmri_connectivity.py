import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pie.imaging.fmri_connectivity import (ConnectivityConfig, extract_connectivity,
                                          nuisance_design, residual_connectivity)


def confounds(n=240):
    rng = np.random.default_rng(109)
    values = {}
    for axis in ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'):
        x = rng.normal(size=n)
        derivative = np.r_[np.nan, np.diff(x)]
        values.update({axis: x, axis + '_derivative1': derivative,
                       axis + '_power2': x*x, axis + '_derivative1_power2': derivative**2})
    meta = {}
    for i in range(5):
        key = f'a_comp_cor_{i:02d}'
        values[key] = rng.normal(size=n)
        meta[key] = dict(Mask='combined', Method='aCompCor', Retained=True, SingularValue=10-i)
    values.update(framewise_displacement=np.r_[np.nan, np.full(n-1, .1)],
                  std_dvars=np.r_[np.nan, np.ones(n-1)],
                  cosine00=np.cos(np.linspace(0, np.pi, n)),
                  global_signal=rng.normal(size=n), global_signal_derivative1=rng.normal(size=n))
    return pd.DataFrame(values), meta


def test_censor_neighbors_and_initial_frames():
    table, meta = confounds()
    table.loc[20, 'framewise_displacement'] = .31
    table.loc[30, 'std_dvars'] = 1.51
    table['non_steady_state_outlier00'] = 0
    table.loc[4, 'non_steady_state_outlier00'] = 1
    design, keep, audit = nuisance_design(table, meta, 2.5)
    assert np.flatnonzero(~keep).tolist() == [0, 1, 2, 3, 4, 19, 20, 21, 22, 29, 30, 31, 32]
    assert len(design) == keep.sum()
    assert audit['retained_seconds'] == 227*2.5
    assert audit['temporal_qc_pass']


def test_missing_compcor_and_nans_are_not_imputed():
    table, meta = confounds()
    meta['a_comp_cor_00']['Mask'] = 'WM'
    with pytest.raises(ValueError, match='Insufficient verified'):
        nuisance_design(table, meta, 2.5)
    table, meta = confounds()
    table.loc[20, 'std_dvars'] = np.nan
    with pytest.raises(ValueError, match='Undefined motion'):
        nuisance_design(table, meta, 2.5)
    table, meta = confounds()
    table.loc[20, 'trans_x'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite nuisance'):
        nuisance_design(table, meta, 2.5)


def test_high_motion_fails_temporal_gate():
    table, meta = confounds()
    table.loc[::2, 'framewise_displacement'] = .8
    _, keep, audit = nuisance_design(table, meta, 2.5)
    assert not keep.any()
    assert not audit['temporal_qc_pass']


def test_network_edges_and_simultaneous_regression():
    rng = np.random.default_rng(4)
    design = np.column_stack([np.ones(200), rng.normal(size=200)])
    series = rng.normal(size=(200, 14)) + design[:, 1:2] * 10
    residual, edges, means = residual_connectivity(series, design, [str(i//2) for i in range(14)])
    assert len(edges) == 91
    assert len(means) == 28
    np.testing.assert_allclose(design.T @ residual, 0, atol=1e-10)
    assert means['0__0'] == pytest.approx(np.arctanh(np.corrcoef(residual.T)[0, 1]))


def test_extraction_and_coverage_rejection(tmp_path):
    rng = np.random.default_rng(41)
    atlas = np.zeros((4, 4, 4), np.int16)
    atlas[:2] = 1
    atlas[2:] = 2
    mask = np.ones(atlas.shape, np.uint8)
    data = rng.normal(100, 10, size=(*atlas.shape, 240)).astype(np.float32)
    table, meta = confounds()
    paths = [tmp_path / name for name in ('bold.nii.gz', 'mask.nii.gz', 'confounds.tsv', 'confounds.json', 'atlas.nii.gz')]
    for path, array in [(paths[0], data), (paths[1], mask), (paths[4], atlas)]:
        image = nib.Nifti1Image(array, np.eye(4))
        image.header.set_xyzt_units('mm', 'sec')
        nib.save(image, path)
    table.to_csv(paths[2], sep='\t', index=False)
    paths[3].write_text(json.dumps(meta))
    result = extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / 'result', tr=2.5)
    assert result['numerical_qc_pass']
    assert result['minimum_coverage'] == 1
    assert extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / 'result', tr=2.5) == result
    from pie.imaging.fmri_data import BOLDImageCache
    with BOLDImageCache() as cache:
        for gsr in (False, True):
            c = ConnectivityConfig(global_signal_regression=gsr)
            direct = extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / f'direct-{gsr}', tr=2.5, config=c)
            shared = extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / f'shared-{gsr}', tr=2.5, config=c, bold_cache=cache)
            assert direct == shared
            for key in np.load(tmp_path / f'direct-{gsr}/connectivity.npz').files:
                np.testing.assert_array_equal(np.load(tmp_path / f'direct-{gsr}/connectivity.npz')[key],
                                              np.load(tmp_path / f'shared-{gsr}/connectivity.npz')[key])
        # NIfTI reads are Fortran-strided; compare to the original reader, not
        # a C-strided synthetic array with a different float32 summation order.
        np.testing.assert_array_equal(cache.get(paths[0]).mean(axis=3),
                                      nib.load(paths[0]).get_fdata(dtype=np.float32).mean(axis=3))
        assert cache.loads == 1
        assert not cache.get(paths[0]).flags.writeable
        paths[0].touch()
        with pytest.raises(ValueError, match='changed'):
            cache.get(paths[0])
    with pytest.raises(RuntimeError, match='closed'):
        cache.get(paths[0])
    mask[:2] = 0
    image = nib.Nifti1Image(mask, np.eye(4))
    image.header.set_xyzt_units('mm', 'sec')
    nib.save(image, paths[1])
    result = extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / 'rejected', tr=2.5)
    assert not result['numerical_qc_pass']
    assert 'insufficient_parcel_coverage' in result['exclusion_reasons']
    assert not (tmp_path / 'rejected/connectivity.npz').exists()


def synthetic_inputs(tmp_path):
    rng = np.random.default_rng(41)
    atlas = np.zeros((4, 4, 4), np.int16)
    atlas[:2], atlas[2:] = 1, 2
    table, meta = confounds()
    paths = [tmp_path / name for name in ('bold.nii.gz', 'mask.nii.gz', 'confounds.tsv', 'confounds.json', 'atlas.nii.gz')]
    for path, array in [(paths[0], rng.normal(100, 10, (*atlas.shape, 240)).astype(np.float32)),
                        (paths[1], np.ones(atlas.shape, np.uint8)), (paths[4], atlas)]:
        image = nib.Nifti1Image(array, np.eye(4))
        image.header.set_xyzt_units('mm', 'sec')
        nib.save(image, path)
    table.to_csv(paths[2], sep='\t', index=False)
    paths[3].write_text(json.dumps(meta))
    return paths


@pytest.mark.parametrize('label_networks, message', [({1: 'network', 3: 'network'}, 'Atlas IDs'),
                                                     ({1: 1, 2: 2}, 'strings')])
def test_rejected_inputs_leave_no_output_directory(tmp_path, label_networks, message):
    paths = synthetic_inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        extract_connectivity(*paths, label_networks, tmp_path / 'result', tr=2.5)
    assert not (tmp_path / 'result').exists()


def test_nonstring_network_names_rejected_clearly():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='strings'):
        residual_connectivity(rng.normal(size=(50, 4)), np.ones((50, 1)), [0, 0, 1, 1])


def test_alff_falff_prefer_the_slow_band_and_survive_censoring():
    from pie.imaging.fmri_connectivity import alff_falff
    tr, n = 1.0, 600
    t = np.arange(n) * tr
    rng = np.random.default_rng(0)
    y = np.column_stack([np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.2 * t)]) + 0.1 * rng.normal(size=(n, 2))
    keep = np.ones(n, bool)
    a, f = alff_falff(y, tr, keep)
    assert a[0] > 3 * a[1] and f[0] > 3 * f[1]           # one sine vs broadband noise over ~300 bins
    for spike in rng.choice(np.arange(5, n - 5), 30, replace=False):      # PIE's pattern: spike, 1 frame before, 2 after
        keep[spike - 1:spike + 3] = False
    a2, f2 = alff_falff(np.where(keep[:, None], y, 1e6), tr, keep)       # censored values must not leak in
    assert np.all(a2 < 1) and a2[0] > 2 * a2[1] and f2[0] > 2 * f2[1]


def test_reho_is_high_for_coherent_neighbourhoods_and_low_for_noise():
    from pie.imaging.fmri_connectivity import reho
    rng = np.random.default_rng(0)
    n, shape = 200, (6, 6, 6)
    common = rng.normal(size=n)
    data = rng.normal(size=shape + (n,))
    data[:3] = common + 0.05 * rng.normal(size=(3, 6, 6, n))      # coherent half
    keep = np.ones(n, bool)
    keep[::10] = False
    w = reho(data, np.ones(shape, bool), keep)
    assert w[1, 3, 3] > 0.9 and w[4, 3, 3] < 0.2 and np.isfinite(w).all()


def test_local_measures_use_the_same_censoring_and_find_the_coherent_slow_parcel(tmp_path):
    from pie.imaging.fmri_connectivity import local_measures
    paths = synthetic_inputs(tmp_path)
    rng = np.random.default_rng(3)
    t = np.arange(240) * 2.5
    data = rng.normal(100, 1, (4, 4, 4, 240)).astype(np.float32)
    data[:2] += 5 * np.sin(2 * np.pi * 0.03 * t).astype(np.float32)          # parcel 1: coherent, in the ALFF band
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_xyzt_units('mm', 'sec')
    nib.save(image, paths[0])
    out = local_measures(*paths, tr=2.5)
    ref = extract_connectivity(*paths, {1: 'network', 2: 'network'}, tmp_path / 'fc', tr=2.5)
    assert out['temporal']['retained_frames'] == ref['temporal']['retained_frames']
    assert out['alff_1'] > 2 * out['alff_2'] and out['falff_1'] > out['falff_2'] and out['reho_1'] > out['reho_2']
    assert set(out) >= {'alff_1', 'alff_2', 'falff_1', 'falff_2', 'reho_1', 'reho_2', 'temporal'}


def test_local_alff_is_normalised_to_the_brain_mean_so_intensity_scale_does_not_matter(tmp_path):
    from pie.imaging.fmri_connectivity import local_measures
    paths = synthetic_inputs(tmp_path)
    a = local_measures(*paths, tr=2.5)
    img = nib.load(paths[0])
    scaled = nib.Nifti1Image(np.asarray(img.dataobj) * 10, img.affine)
    scaled.header.set_xyzt_units('mm', 'sec')
    nib.save(scaled, paths[0])
    b = local_measures(*paths, tr=2.5)
    assert abs(a['alff_1'] / b['alff_1'] - 1) < 1e-6 and abs(a['falff_1'] / b['falff_1'] - 1) < 1e-6


def test_falff_is_not_inflated_by_censoring_white_noise():
    from pie.imaging.fmri_connectivity import alff_falff
    rng = np.random.default_rng(1)
    n, tr = 600, 1.0
    y = rng.normal(size=(n, 200))
    keep = np.ones(n, bool)
    a0, f0 = alff_falff(y, tr, keep)
    for spike in rng.choice(np.arange(5, n - 5), 60, replace=False):   # PIE's pattern: spike, 1 frame before, 2 after
        keep[spike - 1:spike + 3] = False
    assert 0.25 < 1 - keep.mean() < 0.45
    a1, f1 = alff_falff(y, tr, keep)
    assert abs(f1.mean() - f0.mean()) < 0.01                             # interpolation gave 0.14 -> 0.22
    # raw ALFF of noise scales with 1/sqrt(retained frames), the same for every voxel of a scan: mALFF cancels it
    assert abs(a1.mean() / a0.mean() - np.sqrt(n / keep.sum())) < 0.03


def test_local_measures_blank_parcels_the_brain_mask_mostly_misses(tmp_path):
    from pie.imaging.fmri_connectivity import local_measures
    paths = synthetic_inputs(tmp_path)
    mask = np.ones((4, 4, 4), np.uint8)
    mask[2:, :2] = 0                                                         # half of parcel 2 outside the brain
    image = nib.Nifti1Image(mask, np.eye(4))
    nib.save(image, paths[1])
    out = local_measures(*paths, tr=2.5)
    assert np.isfinite(out['alff_1']) and np.isfinite(out['reho_1'])
    assert np.isnan(out['alff_2']) and np.isnan(out['falff_2']) and np.isnan(out['reho_2'])   # 50 % < minimum_parcel_coverage
