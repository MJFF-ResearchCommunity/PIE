"""Reusable, opt-in guards for single-acquisition tensor/GPU correction.

These functions do not run FSL or replace ``dwi.preprocess``. They package the
acquisition selection and correction safeguards exercised by the September 2026
reassessment, without importing study scripts or changing a running cohort.
Numerical/metadata validity is not anatomical or biological validation.
"""
from pathlib import Path

import numpy as np

from .dwi_acquisition import pe_row


def choose_tensor_acquisition(runs):
    """Select one complete acquisition near b=1000, retaining only its own b0s.

    Runs use ``dwi_acquisition.load_runs``'s schema. Eligible shells are 500–1500
    s/mm² with at least 12 diffusion volumes. Ties prefer more volumes, then the
    stable source filename. The returned mask indexes this acquisition only.
    This is a tensor strategy, not a claim that single-shell free water is identified.
    """
    candidates = []
    for run in runs:
        b = np.asarray(run['bvals'], dtype=float)
        if b.ndim != 1 or not np.isfinite(b).all() or np.any(b < 0):
            raise ValueError('Finite, nonnegative one-dimensional b-values required')
        shells = sorted(set(np.round(b[b > 50] / 100).astype(int) * 100))
        for shell in shells:
            selected = (b <= 50) | (np.abs(b - shell) <= 50)
            n = int(((b > 50) & selected).sum())
            if 500 <= shell <= 1500 and n >= 12 and (b <= 50).any():
                candidates.append((abs(shell - 1000), -n, str(run['path']), run, selected, int(shell)))
    if not candidates:
        raise ValueError('No complete acquisition with own b0 and >=12 directions in b=500..1500')
    _, _, _, run, selected, shell = min(candidates, key=lambda candidate: candidate[:3])
    return run, selected, shell


def slice_options(metadata, n_slices):
    """Validate slice grouping for volume motion plus single-slice outlier correction.

    Unverified grouping is omitted explicitly. It must not be reused to enable
    slice-to-volume correction (mporder>0) or groupwise outlier replacement.
    """
    reason = 'No complete slice-timing metadata'
    try:
        times = np.asarray(metadata.get('SliceTiming', []), dtype=float)
    except (TypeError, ValueError):
        times = np.array([])
    if times.shape == (n_slices,) and n_slices > 0 and np.isfinite(times).all() and (times >= 0).all():
        _, counts = np.unique(np.round(times, 6), return_counts=True)
        expected = metadata.get('MultibandAccelerationFactor', 1)
        if np.all(counts == counts[0]) and counts[0] == expected:
            return ['--json=metadata.json'], {'slice_timing': 'validated', 'mb': int(counts[0])}
        reason = 'Unequal simultaneous-slice groups or conflict with stated multiband factor'
    return [], {'slice_timing': 'unused_unverified', 'reason': reason,
                'consequence': 'No slice-to-volume or groupwise outlier model; volume motion plus single-slice outliers only'}


def topup_config(shape):
    """Choose a native-grid FSL configuration; never crop odd image dimensions."""
    spatial = tuple(shape[:3])
    if len(spatial) != 3 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in spatial):
        raise ValueError('Three positive integer spatial dimensions required')
    return 'b02b0.cnf' if all(n % 2 == 0 for n in spatial) else 'b02b0_1.cnf'


def build_eddy_command(fsl_bin, metadata, n_slices, *, use_topup=False,
                       raw_is_uncorrected, cuda=True, nthr=1):
    """Build the bounded correction command for a prepared working directory.

    Requires standard raw.nii.gz/mask/gradient/acqparams/index filenames and
    verified acquisition metadata. With ``use_topup``, the caller must supply
    an estimated compatible reverse-PE field named ``topup``. Feed RAW images,
    not images already passed through applytopup or rigid motion correction.
    The caller owns resource scheduling, command/provenance logs and timeouts.
    CPU and CUDA engines are explicit choices, never a silent fallback. ``nthr``: OpenMP threads, which only eddy_cpu
    uses (single-threaded it takes many hours per multi-shell scan).
    """
    if raw_is_uncorrected is not True:
        raise ValueError('Raw uncorrected input required; do not apply correction twice')
    pe_row(metadata)  # Estimated timing or a PE axis without polarity is insufficient.
    if not isinstance(n_slices, (int, np.integer)) or n_slices <= 0:
        raise ValueError('Positive integer slice count required')
    options, record = slice_options(metadata, n_slices)
    binary = Path(fsl_bin) / ('eddy_cuda' if cuda else 'eddy_cpu')
    command = [str(binary), '--imain=raw.nii.gz', '--mask=eddy_mask.nii.gz',
               '--acqp=acqparams.txt', '--index=index.txt', '--bvals=bvals',
               '--bvecs=bvecs', '--out=eddy', '--niter=5', f'--nthr={int(nthr)}', '--repol',
               '--initrand=1', '--cnr_maps', '--ol_type=sw', '--mporder=0', '--verbose', *options]
    if use_topup:
        command.append('--topup=topup')
    return command, record


def validate_corrected_geometry(raw_image, corrected_image, bvals, rotated_bvecs):
    """Reject grid/gradient changes; consume eddy's rotated bvecs exactly once.

    Images can be nibabel objects; only headers are read here. A separate finite-
    intensity check, ROI registration and measurement QC remain mandatory.
    """
    b = np.asarray(bvals, dtype=float)
    v = np.asarray(rotated_bvecs, dtype=float)
    if raw_image.ndim != 4 or raw_image.shape != corrected_image.shape:
        raise ValueError('Corrected image dimensions do not match the raw 4D grid')
    if not np.allclose(raw_image.affine, corrected_image.affine, atol=1e-4, rtol=0):
        raise ValueError('Corrected image affine changed')
    if b.ndim != 1 or len(b) != raw_image.shape[3] or not np.isfinite(b).all() or (b < 0).any():
        raise ValueError('Invalid b-values or volume count')
    if not (b <= 50).any() or not (b > 50).any():
        raise ValueError('Own b0 and diffusion-weighted volumes required')
    if v.shape != (3, len(b)) or not np.isfinite(v).all():
        raise ValueError('Invalid rotated bvec dimensions or values')
    if not np.allclose(np.linalg.norm(v[:, b > 50], axis=0), 1, atol=.01):
        raise ValueError('Rotated diffusion gradients are not unit length')
