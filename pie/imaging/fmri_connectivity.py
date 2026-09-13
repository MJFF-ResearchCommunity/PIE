"""Outcome-independent parcel connectivity from validated fMRIPrep derivatives.

This is downstream denoising, not a replacement for preprocessing or visual QC.
Paths, atlas labels/network membership and QC thresholds are caller supplied.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd

from .fmri import sha256, write_json


@dataclass(frozen=True)
class ConnectivityConfig:
    discard_seconds: float = 10
    fd_threshold: float = .3
    dvars_threshold: float = 1.5
    censor_before: int = 1
    censor_after: int = 2
    minimum_seconds: float = 300
    minimum_residual_dof: int = 60
    minimum_parcel_coverage: float = .9
    compcor_components: int = 5
    global_signal_regression: bool = False


def nuisance_design(confounds, metadata, tr, config=ConnectivityConfig()):
    """Return retained-row design, full-length inclusion mask and audit record.

    No missing confounds are imputed. Undefined first differences may occur only
    on excluded frames. Fit all nuisance regressors together on retained frames.
    """
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError('Need a finite positive TR')
    if (config.discard_seconds < 0 or config.minimum_seconds < 0 or
        min(config.censor_before, config.censor_after, config.minimum_residual_dof) < 0 or
        not 0 <= config.minimum_parcel_coverage <= 1 or config.compcor_components < 1):
        raise ValueError('Invalid connectivity QC configuration')
    n = len(confounds)
    initial = np.arange(n) < math.ceil(config.discard_seconds / tr)
    nonsteady = [c for c in confounds if c.startswith('non_steady_state_outlier')]
    if nonsteady:
        values = confounds[nonsteady].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite nonsteady-state indicators')
        initial |= values.any(axis=1)
    fd, dv = (confounds[c].to_numpy(float) for c in ('framewise_displacement', 'std_dvars'))
    if not np.isfinite(fd[~initial]).all() or not np.isfinite(dv[~initial]).all():
        raise ValueError('Undefined motion/DVARS outside initially discarded frames')
    spikes = (fd > config.fd_threshold) | (dv > config.dvars_threshold)
    censored = spikes.copy()
    for offset in range(1, config.censor_before + 1):
        censored[:-offset] |= spikes[offset:]
    for offset in range(1, config.censor_after + 1):
        censored[offset:] |= spikes[:-offset]
    keep = ~(initial | censored)
    motion = [axis + suffix for axis in ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')
              for suffix in ('', '_derivative1', '_power2', '_derivative1_power2')]
    components = [c for c in confounds if c.startswith('a_comp_cor_') and
                  metadata.get(c, {}).get('Mask') == 'combined' and
                  metadata.get(c, {}).get('Method') == 'aCompCor' and
                  metadata.get(c, {}).get('Retained') is True]
    components.sort(key=lambda c: (-float(metadata[c]['SingularValue']), c))
    if len(components) < config.compcor_components:
        raise ValueError('Insufficient verified combined WM/CSF aCompCor components')
    cosines = sorted(c for c in confounds if c.startswith('cosine'))
    if not cosines:
        raise ValueError('Missing fMRIPrep high-pass cosine regressors')
    columns = motion + components[:config.compcor_components] + cosines
    if config.global_signal_regression:
        columns += ['global_signal', 'global_signal_derivative1']
    design = np.column_stack([np.ones(n), np.linspace(-1, 1, n), confounds[columns].to_numpy(float)])[keep]
    if not np.isfinite(design).all():
        raise ValueError('Nonfinite nuisance regressors on retained frames')
    # Column normalization stabilizes rank/least squares without changing span.
    scales = np.linalg.norm(design, axis=0)
    design = design / np.where(scales > 0, scales, 1)
    rank = int(np.linalg.matrix_rank(design)) if keep.any() else 0
    audit = dict(retained_frames=int(keep.sum()), retained_seconds=float(keep.sum() * tr),
                 initial_discarded=int(initial.sum()), spike_frames=int(spikes.sum()),
                 nuisance_columns=['intercept', 'linear_trend', *columns], nuisance_rank=rank,
                 residual_dof=int(keep.sum()) - rank,
                 mean_fd=float(np.mean(fd[~initial])) if (~initial).any() else None,
                 retained_indices=np.flatnonzero(keep).tolist())
    audit['temporal_qc_pass'] = (audit['retained_seconds'] >= config.minimum_seconds and
                                audit['residual_dof'] >= config.minimum_residual_dof)
    return design, keep, audit


def residual_connectivity(series, design, networks):
    """OLS residualization and Fisher-z edges/network means (self-edges excluded)."""
    series = np.asarray(series, float)
    if series.ndim != 2 or len(networks) != series.shape[1] or len(series) != len(design):
        raise ValueError('Series/design/network dimensions differ')
    if not np.isfinite(series).all():
        raise ValueError('Nonfinite parcel signal')
    residual = series - design @ np.linalg.lstsq(design, series, rcond=None)[0]
    if np.any(np.std(residual, axis=0) <= np.finfo(float).eps * np.maximum(1, np.abs(series).max(axis=0))):
        raise ValueError('Constant residual parcel signal')
    corr = np.corrcoef(residual, rowvar=False)
    upper = np.triu_indices(series.shape[1], 1)
    edges = np.arctanh(np.clip(corr[upper], -1 + 1e-7, 1 - 1e-7))
    groups = sorted(set(networks))
    membership = np.asarray(networks)
    names, values = [], []
    for i, left in enumerate(groups):
        for right in groups[i:]:
            select = ((membership[upper[0]] == left) & (membership[upper[1]] == right)) | (
                (membership[upper[0]] == right) & (membership[upper[1]] == left))
            if not select.any():
                raise ValueError('Network pair has no distinct-parcel edges')
            names.append(left + '__' + right)
            values.append(float(edges[select].mean()))
    return residual, edges, dict(zip(names, values))


def extract_connectivity(bold, brain_mask, confounds_tsv, confounds_json, atlas,
                         label_networks, output_dir, *, tr, config=ConnectivityConfig(), bold_cache=None):
    """Extract features or an explicit QC failure with complete input hashes.

    Atlas and BOLD must already share a physical standard space; resampling only
    reconciles grids, never estimates registration. Keys are positive atlas IDs.
    Coverage is computed on the original atlas grid, including missing-FOV voxels.
    Optional BOLDImageCache shares one immutable 4D read across QC variants;
    it does not alter the scientific calculation or input/output checksums.
    """
    paths = [Path(p).resolve(strict=True) for p in (bold, brain_mask, confounds_tsv, confounds_json, atlas)]
    from . import fmri_data
    identity = dict(inputs={str(p): sha256(p) for p in paths}, config=asdict(config), tr=tr,
                    label_networks={str(k): v for k, v in label_networks.items()},
                    implementation_sha256=sha256(__file__), data_reader_sha256=sha256(fmri_data.__file__))
    out = Path(output_dir)
    completion = out / 'connectivity.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing connectivity inputs/settings changed')
        for name, digest in saved.get('outputs', {}).items():
            if sha256(out / name) != digest:
                raise ValueError('Existing connectivity output checksum mismatch')
        return saved
    out.mkdir(parents=True, exist_ok=True)
    img, mask, atlas_img = (nib.load(p) for p in (bold, brain_mask, atlas))
    if any(image.header.get_xyzt_units()[0] != 'mm' for image in (img, mask, atlas_img)):
        raise ValueError('Explicit millimeter spatial units are required')
    table = pd.read_csv(confounds_tsv, sep='\t')
    if img.ndim != 4 or len(table) != img.shape[3]:
        raise ValueError('BOLD/confounds length mismatch')
    if mask.shape != img.shape[:3] or not np.allclose(mask.affine, img.affine, atol=1e-5, rtol=0):
        raise ValueError('BOLD/mask grids differ')
    labels = sorted(label_networks)
    raw_atlas = atlas_img.get_fdata()
    if set(np.unique(raw_atlas)) - {0} != set(labels) or min(labels) < 1:
        raise ValueError('Atlas IDs and supplied label table differ')
    atlas_bold = resample_from_to(atlas_img, (img.shape[:3], img.affine), order=0).get_fdata().astype(int)
    mask_values = mask.get_fdata()
    if not np.isfinite(mask_values).all():
        raise ValueError('Nonfinite BOLD brain mask')
    brain = mask_values > 0
    selected = (atlas_bold > 0) & brain
    # Loading once avoids repeatedly decompressing a 4D gzip file for each frame.
    data = img.get_fdata(dtype=np.float32) if bold_cache is None else bold_cache.get(bold)
    voxel_series = data[selected]
    valid = np.isfinite(voxel_series).all(axis=1) & (np.ptp(voxel_series, axis=1) > 0)
    valid_mask = np.zeros(selected.shape, np.uint8)
    valid_mask[selected] = valid
    valid_atlas = resample_from_to(nib.Nifti1Image(valid_mask, img.affine), atlas_img, order=0).get_fdata() > 0
    coverage = {str(label): float(np.mean(valid_atlas[raw_atlas == label])) for label in labels}
    design, keep, temporal = nuisance_design(table, json.loads(Path(confounds_json).read_text()), tr, config)
    reasons = []
    if min(coverage.values()) < config.minimum_parcel_coverage:
        reasons.append('insufficient_parcel_coverage')
    if not temporal['temporal_qc_pass']:
        reasons.append('insufficient_retained_time_or_residual_dof')
    result = dict(identity=identity, temporal=temporal, parcel_coverage=coverage,
                  minimum_coverage=min(coverage.values()), numerical_qc_pass=not reasons,
                  visual_qc_required=True, exclusion_reasons=reasons, outputs={})
    if not reasons:
        values = voxel_series[valid]
        voxel_labels = atlas_bold[selected][valid]
        if any(not np.any(voxel_labels == label) for label in labels):
            raise ValueError('Empty parcel on BOLD grid')
        series = np.column_stack([values[voxel_labels == label].mean(axis=0, dtype=np.float64) for label in labels])
        residual, edges, features = residual_connectivity(series[keep], design,
                                                         [label_networks[i] for i in labels])
        np.savez_compressed(out / 'connectivity.npz', parcel_timeseries=series,
                            retained_indices=np.flatnonzero(keep), residual_timeseries=residual,
                            edges=edges, labels=np.array(labels))
        result['network_features'] = features
        result['outputs']['connectivity.npz'] = sha256(out / 'connectivity.npz')
    write_json(completion, result)
    return result
