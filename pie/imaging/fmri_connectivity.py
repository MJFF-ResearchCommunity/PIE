"""Outcome-independent parcel connectivity from validated fMRIPrep derivatives.

This is downstream denoising, not a replacement for preprocessing or visual QC.
Paths, atlas labels/network membership and QC thresholds are caller supplied.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import warnings

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd

from .fmri import framewise_displacement, sha256, write_json

# Respiratory band stopped from the motion parameters before FD at short TRs: XCP-D's recommendation for 65-80
# year olds (12-28 breaths/min, 0.2-0.47 Hz), PPMI's age range. xcpd.py passes the same band to XCP-D.
RESPIRATORY_BAND_BPM = (12, 28)


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
    minimum_segment_frames: int = 5
    motion_filter_max_tr: float = 1.5
    respiratory_low_bpm: float = RESPIRATORY_BAND_BPM[0]
    respiratory_high_bpm: float = RESPIRATORY_BAND_BPM[1]


def filtered_fd(motion, tr, band_bpm=RESPIRATORY_BAND_BPM):
    """Power FD from fMRIPrep motion parameters (frames x trans_x, trans_y, trans_z, rot_x, rot_y, rot_z; radians)
    after a notch filter over the respiratory band, as XCP-D's ``--motion-filter-type notch`` (Fair et al. 2020,
    doi:10.1016/j.neuroimage.2019.116400): at multiband TRs breathing shifts the B0 field, realignment reads it as head
    motion and unfiltered FD censors breathing. Same filter as XCP-D 26.2.0 (``iirnotch`` at the band centre,
    Q = centre / width, one zero-phase ``filtfilt`` pass, constant padding), band folded below Nyquist as XCP-D does.
    First value NaN, as fMRIPrep's."""
    from scipy.signal import filtfilt, iirnotch
    motion = np.asarray(motion, float)
    fs = 1 / tr
    band = np.asarray(band_bpm, float) / 60
    band = np.abs(band - np.floor((band + fs / 2) / fs) * fs)
    centre, width = band.mean(), abs(band[1] - band[0])
    if not (0 < centre < fs / 2 and width > 0) or not np.isfinite(motion).all():
        raise ValueError('Respiratory band unusable at this TR, or nonfinite motion parameters')
    b, a = iirnotch(centre, centre / width, fs=fs)
    filtered = filtfilt(b, a, motion, axis=0, padtype='constant', padlen=len(motion) - 1)
    fd = framewise_displacement(filtered[:, [3, 4, 5, 0, 1, 2]])
    fd[0] = np.nan
    return fd


def nuisance_design(confounds, metadata, tr, config=ConnectivityConfig()):
    """Return retained-row design, full-length inclusion mask and audit record.

    No missing confounds are imputed. Undefined first differences may occur only
    on excluded frames. Fit all nuisance regressors together on retained frames.
    """
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError('Need a finite positive TR')
    if (config.discard_seconds < 0 or config.minimum_seconds < 0 or
        min(config.censor_before, config.censor_after, config.minimum_residual_dof) < 0 or
        not 0 <= config.minimum_parcel_coverage <= 1 or config.compcor_components < 1 or
        config.minimum_segment_frames < 1 or not 0 < config.respiratory_low_bpm < config.respiratory_high_bpm):
        raise ValueError('Invalid connectivity QC configuration')
    n = len(confounds)
    initial = np.arange(n) < math.ceil(config.discard_seconds / tr)
    nonsteady = [c for c in confounds if c.startswith('non_steady_state_outlier')]
    if nonsteady:
        values = confounds[nonsteady].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite nonsteady-state indicators')
        initial |= values.any(axis=1)
    raw_fd, dv = (confounds[c].to_numpy(float) for c in ('framewise_displacement', 'std_dvars'))
    fd = raw_fd
    motion_filtered = tr <= config.motion_filter_max_tr
    if motion_filtered:
        fd = filtered_fd(confounds[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']], tr,
                         (config.respiratory_low_bpm, config.respiratory_high_bpm))
    if not np.isfinite(fd[~initial]).all() or not np.isfinite(dv[~initial]).all():
        raise ValueError('Undefined motion/DVARS outside initially discarded frames')
    spikes = (fd > config.fd_threshold) | (dv > config.dvars_threshold)
    censored = spikes.copy()
    for offset in range(1, config.censor_before + 1):
        censored[:-offset] |= spikes[offset:]
    for offset in range(1, config.censor_after + 1):
        censored[offset:] |= spikes[:-offset]
    keep = ~(initial | censored)
    # Retained stretches shorter than minimum_segment_frames are censored too (Power et al. 2014; XCP-D
    # --censor-between): a few frames between spikes are motion-contaminated and barely estimate anything.
    bounds = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
    short = 0
    for start, stop in zip(bounds[::2], bounds[1::2]):
        if stop - start < config.minimum_segment_frames:
            keep[start:stop] = False
            short += stop - start
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
                 short_segment_frames=short, motion_filtered=motion_filtered,
                 nuisance_columns=['intercept', 'linear_trend', *columns], nuisance_rank=rank,
                 residual_dof=int(keep.sum()) - rank,
                 mean_fd=float(np.mean(raw_fd[~initial])) if (~initial).any() else None,
                 mean_fd_filtered=float(np.mean(fd[~initial])) if motion_filtered and (~initial).any() else None,
                 retained_indices=np.flatnonzero(keep).tolist())
    audit['temporal_qc_pass'] = (audit['retained_seconds'] >= config.minimum_seconds and
                                audit['residual_dof'] >= config.minimum_residual_dof)
    return design, keep, audit


def _lomb_scargle_amplitude(t, y, freq, chunk=20000):
    """One-sided amplitude spectrum of unevenly sampled columns (Lomb 1976; Scargle 1982), scaled so that a sine of
    amplitude A gives A, as the FFT amplitude 2|X|/N does for complete series. ``t``: sample times, ``y``: time x
    columns (mean removed), ``freq``: Hz. Columns are processed in chunks to bound memory."""
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    tau = np.arctan2(np.sin(2 * w * t).sum(axis=1), np.cos(2 * w * t).sum(axis=1)) / (2 * w[:, 0])
    arg = w * (t[None, :] - tau[:, None])
    c, s = np.cos(arg), np.sin(arg)
    cc, ss = (c ** 2).sum(axis=1)[:, None], (s ** 2).sum(axis=1)[:, None]
    out = np.empty((len(freq), y.shape[1]))
    for k in range(0, y.shape[1], chunk):
        yk = y[:, k:k + chunk]
        power = 0.5 * ((c @ yk) ** 2 / cc + np.divide((s @ yk) ** 2, ss, out=np.zeros((len(freq), yk.shape[1])), where=ss > 1e-9))
        out[:, k:k + chunk] = 2 * np.sqrt(power / len(t))
    return out


def alff_falff(series, tr, keep, band=(0.01, 0.08)):
    """ALFF (mean amplitude in ``band``, Zang et al. 2007) and fALFF (band amplitude over the amplitude of (0, Nyquist],
    Zou et al. 2008) per column of a time x columns array of nuisance residuals covering every frame. The spectrum is
    the Lomb-Scargle periodogram of the retained frames (``keep``), at the full series' FFT frequencies: interpolating
    censored frames instead removes high-frequency power and inflates fALFF with the censored fraction (on white noise
    0.14 -> 0.22 at 35 % censored), i.e. with head motion. Uncensored, it equals the FFT amplitude spectrum. Raw ALFF of
    noise grows as 1/sqrt(retained frames), equally for every voxel of a scan, so compare scans by mALFF (ALFF over its
    brain mean, as ``local_measures`` reports it)."""
    keep = np.asarray(keep, bool)
    if keep.sum() < 3:
        raise ValueError('Fewer than three retained frames')
    y = np.asarray(series, float)[keep]
    freq = np.fft.rfftfreq(len(keep), d=tr)[1:]
    amp = _lomb_scargle_amplitude(np.flatnonzero(keep) * float(tr), y - y.mean(axis=0), freq)
    inband = (freq >= band[0]) & (freq <= band[1])
    return amp[inband].mean(axis=0), amp[inband].sum(axis=0) / amp.sum(axis=0)


def reho(data, mask, keep):
    """Regional homogeneity (Zang et al. 2004): Kendall's W of each in-mask voxel with its in-mask 26 neighbours over
    the retained frames. ``data``: (x, y, z, t) array; returns an (x, y, z) float32 map, NaN outside the mask. Works on
    the mask's bounding box in float32 (~1.5 GB for a 2 mm brain and 600 frames)."""
    from scipy import ndimage
    from scipy.stats import rankdata

    mask, keep = np.asarray(mask, bool), np.asarray(keep, bool)
    n = int(keep.sum())
    idx = np.nonzero(mask)
    box = tuple(slice(max(0, int(a.min()) - 1), int(a.max()) + 2) for a in idx)
    m = mask[box]
    ranks = np.zeros(m.shape + (n,), np.float32)
    ranks[m] = rankdata(np.asarray(data)[box][m][:, keep], axis=1)
    k = ndimage.uniform_filter(m.astype(np.float32), size=3, mode="constant") * 27          # in-mask voxels per cube
    rsum = ndimage.uniform_filter(ranks, size=(3, 3, 3, 1), mode="constant") * 27            # summed ranks per frame
    s = ((rsum - k[..., None] * (n + 1) / 2) ** 2).sum(axis=-1)
    w = 12 * s / (np.maximum(k, 1) ** 2 * (n ** 3 - n))
    out = np.full(mask.shape, np.nan, np.float32)
    out[box] = np.where(m & (k > 1.5), w, np.nan)
    return out


def _network_names(networks):
    """Names become "A__B" feature keys: nonstrings would fail late or collide (1 vs "1")."""
    networks = list(networks)
    if not all(isinstance(name, str) and name for name in networks):
        raise ValueError('Network names must be nonempty strings')
    return networks


def network_means(edges, networks):
    """Mean Fisher z per unordered network pair, keyed "A__B" (names sorted), of upper-triangle ``edges``
    (``np.triu_indices`` order) between parcels labelled ``networks``. NaN edges (parcels without coverage) are left
    out; a pair with no edge left, or none at all (a one-parcel network with itself), is NaN."""
    networks = _network_names(networks)
    upper = np.triu_indices(len(networks), 1)
    membership = np.asarray(networks)
    groups = sorted(set(networks))
    out = {}
    for i, left in enumerate(groups):
        for right in groups[i:]:
            select = ((membership[upper[0]] == left) & (membership[upper[1]] == right)) | (
                (membership[upper[0]] == right) & (membership[upper[1]] == left))
            finite = edges[select][np.isfinite(edges[select])]
            out[left + '__' + right] = float(finite.mean()) if finite.size else math.nan
    return out


def residual_connectivity(series, design, networks):
    """OLS residualization and Fisher-z edges/network means (self-edges excluded). An all-NaN column is a parcel
    without coverage: its edges are NaN and it is left out of the network means."""
    networks = _network_names(networks)
    series = np.asarray(series, float)
    if series.ndim != 2 or len(networks) != series.shape[1] or len(series) != len(design):
        raise ValueError('Series/design/network dimensions differ')
    present = ~np.isnan(series).all(axis=0)
    kept = series[:, present]
    if not np.isfinite(kept).all():
        raise ValueError('Nonfinite parcel signal')
    residual = np.full(series.shape, np.nan)
    residual[:, present] = kept - design @ np.linalg.lstsq(design, kept, rcond=None)[0]
    if np.any(np.std(residual[:, present], axis=0) <= np.finfo(float).eps * np.maximum(1, np.abs(kept).max(axis=0))):
        raise ValueError('Constant residual parcel signal')
    corr = np.corrcoef(residual, rowvar=False)
    edges = np.arctanh(np.clip(corr[np.triu_indices(series.shape[1], 1)], -1 + 1e-7, 1 - 1e-7))
    return residual, edges, network_means(edges, networks)


def header_tr(img):
    """TR in seconds from a 4D NIfTI header."""
    zoom, unit = float(img.header.get_zooms()[3]), img.header.get_xyzt_units()[1]
    return zoom / 1000 if unit == 'msec' else zoom / 1e6 if unit == 'usec' else zoom


def _check_tr(img, tr):
    """The caller's TR must be the BOLD file's: PPMI has 2.4, 2.5 and 1.0 s protocols, and a wrong TR mis-scales
    censoring (frames discarded, retained seconds), the motion filter and the ALFF band."""
    seconds = header_tr(img)
    if seconds > 0 and not np.isclose(seconds, tr, rtol=0.01):
        raise ValueError(f'tr={tr} s but the BOLD header says {seconds:g} s')


def extract_connectivity(bold, brain_mask, confounds_tsv, confounds_json, atlas,
                         label_networks, output_dir, *, tr, config=ConnectivityConfig(), bold_cache=None):
    """Extract features or an explicit QC failure with complete input hashes.

    Atlas and BOLD must already share a physical standard space; resampling only
    reconciles grids, never estimates registration. Keys are positive atlas IDs.
    Coverage is computed on the original atlas grid, including missing-FOV voxels.
    A parcel below ``minimum_parcel_coverage`` gets NaN signal and edges and is left
    out of the network means (null when a pair has no edge left); the scan is
    excluded for coverage only when fewer than two parcels remain.
    Optional BOLDImageCache shares one immutable 4D read across QC variants;
    it does not alter the scientific calculation or input/output checksums.
    """
    _network_names(label_networks.values())
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
    img, mask, atlas_img = (nib.load(p) for p in (bold, brain_mask, atlas))
    _check_tr(img, tr)
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
    covered = [label for label in labels if coverage[str(label)] >= config.minimum_parcel_coverage]
    reasons = []
    if len(covered) < 2:
        reasons.append('insufficient_parcel_coverage')
    if not temporal['temporal_qc_pass']:
        reasons.append('insufficient_retained_time_or_residual_dof')
    result = dict(identity=identity, temporal=temporal, parcel_coverage=coverage,
                  minimum_coverage=min(coverage.values()),
                  low_coverage_parcels=[label for label in labels if label not in covered],
                  numerical_qc_pass=not reasons, visual_qc_required=True, exclusion_reasons=reasons, outputs={})
    if not reasons:
        values = voxel_series[valid]
        voxel_labels = atlas_bold[selected][valid]
        if any(not np.any(voxel_labels == label) for label in covered):
            raise ValueError('Empty parcel on BOLD grid')
        series = np.column_stack([values[voxel_labels == label].mean(axis=0, dtype=np.float64) if label in covered
                                  else np.full(len(keep), np.nan) for label in labels])
        residual, edges, features = residual_connectivity(series[keep], design,
                                                         [label_networks[i] for i in labels])
        # JSON has no NaN: a network pair left without covered edges is null
        result['network_features'] = {k: (v if np.isfinite(v) else None) for k, v in features.items()}
    # Created only after every check, so a rejected call leaves nothing behind.
    out.mkdir(parents=True, exist_ok=True)
    if not reasons:
        np.savez_compressed(out / 'connectivity.npz', parcel_timeseries=series,
                            retained_indices=np.flatnonzero(keep), residual_timeseries=residual,
                            edges=edges, labels=np.array(labels))
        result['outputs']['connectivity.npz'] = sha256(out / 'connectivity.npz')
    write_json(completion, result)
    return result


def local_measures(bold, brain_mask, confounds_tsv, confounds_json, atlas, *, tr, config=ConnectivityConfig(), band=(0.01, 0.08)):
    """Parcel means of voxel-wise ALFF (as mALFF: divided by its brain mean), fALFF and ReHo on nuisance residuals, with exactly the censoring and regressors
    of ``extract_connectivity`` (``nuisance_design``). ReHo uses the unsmoothed residuals. Atlas and BOLD must share a
    standard space (e.g. ``atlases.schaefer400_mni2009c()`` on fMRIPrep's MNI152NLin2009cAsym res-2 grid); the atlas is
    only resampled (nearest) onto the BOLD grid. Returns {"alff_<id>", "falff_<id>", "reho_<id>", "temporal"}; numbers
    only, no QC verdict, so check ``temporal["temporal_qc_pass"]`` as for connectivity."""
    img, mask_img, atlas_img = (nib.load(p) for p in (bold, brain_mask, atlas))
    _check_tr(img, tr)
    table = pd.read_csv(confounds_tsv, sep='\t')
    if img.ndim != 4 or len(table) != img.shape[3]:
        raise ValueError('BOLD/confounds length mismatch')
    design, keep, temporal = nuisance_design(table, json.loads(Path(confounds_json).read_text()), tr, config)
    labels = resample_from_to(atlas_img, (img.shape[:3], img.affine), order=0).get_fdata().astype(int)
    data = img.get_fdata(dtype=np.float32)
    brain = np.asarray(mask_img.dataobj) > 0
    series = data[brain].T.astype(float)                                   # time x voxels
    valid = np.isfinite(series).all(axis=0) & (np.ptp(series, axis=0) > 0)
    kept = series[keep][:, valid]
    residual = np.zeros((len(keep), int(valid.sum())))
    residual[keep] = kept - design @ np.linalg.lstsq(design, kept, rcond=None)[0]
    alff, falff = alff_falff(residual, tr, keep, band)
    alff = alff / alff.mean()               # mALFF: raw amplitude is in scanner intensity units (Zang et al. 2007)
    use = np.zeros(brain.shape, bool)
    use[brain] = valid
    res4d = np.zeros(brain.shape + (int(keep.sum()),), np.float32)
    res4d[use] = residual[keep].T
    reho_map = reho(res4d, use, np.ones(int(keep.sum()), bool))
    vox_labels = labels[use]
    out = {"temporal": temporal}
    for k in sorted(set(np.unique(labels)) - {0}):
        sel = vox_labels == k
        ok = sel.sum() >= config.minimum_parcel_coverage * (labels == k).sum()   # as extract_connectivity's parcel coverage
        out[f"alff_{k}"] = float(alff[sel].mean()) if ok else np.nan
        out[f"falff_{k}"] = float(falff[sel].mean()) if ok else np.nan
        out[f"reho_{k}"] = float(np.nanmean(reho_map[use & (labels == k)])) if ok else np.nan
    return out


# ====================================================================================================
# Striatal and basal-ganglia-network connectivity
# Merged from fmri_striatal.py; kept together with the rest of the fmri_connectivity measures.
#
# Striatal and basal-ganglia-network connectivity from preprocessed resting-state fMRI.
#
# Two approaches, the two the Parkinson's literature uses:
#
# 1. Region-based: mean BOLD in anatomical striatal regions (caudate, putamen, nucleus accumbens, per hemisphere,
#    from the bundled CIT168 atlas in MNI152NLin2009cAsym), correlated with cortical parcels and averaged by network.
#    Requires BOLD already in MNI152NLin2009cAsym (fMRIPrep ``--output-spaces MNI152NLin2009cAsym``); grids are
#    reconciled by nearest-neighbour resampling, never by registration.
#
#        striatal_rois(target)                       caudate/putamen/accumbens L/R label image on the BOLD grid
#        roi_timeseries(bold, rois, mask)            mean signal per region, (time x regions)
#        seed_network_connectivity(...)              Fisher-z of each striatal region with each cortical network
#
# 2. Network-based, as in Szewczyk-Krolikowski 2014, Rolinski 2016 and Droby 2025: spatial group ICA on temporally
#    concatenated data, identification of the basal ganglia network (BGN) by overlap with an independent striatal
#    template, back-projection to each subject by dual regression, and mean subject-map weight in anatomical masks.
#
#        group_ica(datasets, n_components)           spatial ICA maps (components x voxels), sign-fixed and z-scored
#        load_reference_maps(path, mask)             a frozen external map set instead, with its checksum
#        select_component(maps, template)            Dice and spatial correlation of every component with a template
#        dual_regression(data, maps)                 FSL dual regression: subject time courses and subject maps
#        roi_means(subject_map, roi_labels, names)   mean weight per anatomical region
#        bgn_weights(...)                            the above end to end, preferring frozen reference maps
#
#    Every function that takes BOLD takes ``design, keep`` from ``nuisance_design``: nuisance regression on the
#    retained frames, all regressors together, as ``extract_connectivity``. Omit them only for input that is
#    already denoised and censored (XCP-D's ``desc-denoised_bold``, see xcpd.py).
#
# CIT168 regions at its 0.25 probability threshold are generous: on the fMRIPrep 2 mm grid each caudate is about
# 860 voxels (6.9 mL) and each putamen about 970 (7.8 mL), so the caudate borders the lateral ventricle. Erode the
# regions or restrict them to a grey-matter mask when partial volume with CSF matters.
#
# Choices that change results are parameters with the literature's defaults, and the template used to pick the
# BGN must be independent of the patients analysed (Griffanti et al. 2016): the CIT168 striatum qualifies.
# Outcome labels are never read here.
# ====================================================================================================

# CIT168 v1 label names (atlases.cit168_metadata()["labels"]) for the striatal regions, and PIE feature names
STRIATAL = {"caudate": "Ca", "putamen": "Pu", "accumbens": "NAC"}


def striatal_rois(target_img=None, regions=STRIATAL, putamen_split_y=None):
    """Caudate, putamen and accumbens per hemisphere from CIT168 (MNI152NLin2009cAsym).

    CIT168 labels are bilateral; hemispheres are split at MNI x = 0 in world coordinates, which is valid only
    because the atlas is verified to be in MNI152NLin2009cAsym. ``putamen_split_y`` (mm) cuts each putamen at that
    coronal plane into ``putamen_anterior_*`` and ``putamen_posterior_*``; y = 0 is close to the anterior commissure,
    behind which lies the sensorimotor putamen, where dopamine loss in PD starts. Returns (label image, {id: name}).
    """
    from .atlases import cit168_mni2009c, cit168_metadata

    atlas = cit168_mni2009c()
    names = cit168_metadata()["labels"]
    data = np.asarray(atlas.dataobj)
    ijk = np.indices(data.shape).reshape(3, -1).T
    x, y = (nib.affines.apply_affine(atlas.affine, ijk)[:, i].reshape(data.shape) for i in (0, 1))
    out = np.zeros(data.shape, np.int16)
    mapping = {}
    k = 1
    for region, code in regions.items():
        label = names.index(code) + 1
        parts = {region: True}
        if region == "putamen" and putamen_split_y is not None:
            parts = {"putamen_anterior": y >= putamen_split_y, "putamen_posterior": y < putamen_split_y}
        for name, part in parts.items():
            for side, sel in (("l", x < 0), ("r", x > 0)):
                out[(data == label) & sel & part] = k
                mapping[k] = f"{name}_{side}"
                k += 1
    img = nib.Nifti1Image(out, atlas.affine)
    if target_img is not None:
        img = resample_from_to(img, (target_img.shape[:3], target_img.affine), order=0)
        img = nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), target_img.affine)
    return img, mapping


def roi_timeseries(bold_img, roi_img, mask_img=None, min_voxels=5):
    """Mean BOLD per region (time x regions) and the voxel count behind each; regions below ``min_voxels`` -> NaN."""
    bold = np.asanyarray(bold_img.dataobj)              # stored type: no float64 copy of the whole 4D image
    rois = np.asarray(roi_img.dataobj)
    if rois.shape != bold.shape[:3]:
        raise ValueError("region image is not on the BOLD grid")
    inside = np.ones(rois.shape, bool) if mask_img is None else np.asarray(mask_img.dataobj) > 0
    labels = [k for k in np.unique(rois) if k > 0]
    series, counts = np.full((bold.shape[-1], len(labels)), np.nan), {}
    for j, k in enumerate(labels):
        voxels = bold[(rois == k) & inside].astype(np.float64)
        voxels = voxels[np.isfinite(voxels).all(axis=1) & (voxels.std(axis=1) > 0)]
        counts[int(k)] = len(voxels)
        if len(voxels) >= min_voxels:
            series[:, j] = voxels.mean(axis=0)
    return series, labels, counts


def _residualise(series, design=None, keep=None):
    """Retained frames (``keep``) residualised on the retained-row nuisance ``design``, all regressors together, as
    ``nuisance_design`` returns them. Without a design the frames are returned as they are (already denoised input).
    All-NaN columns (empty regions) stay NaN."""
    series = np.asarray(series, float)
    if keep is not None:
        series = series[np.asarray(keep, bool)]
    if design is None:
        return series
    if len(design) != len(series):
        raise ValueError("design rows differ from the retained frames")
    out = np.full(series.shape, np.nan)
    ok = ~np.isnan(series).all(axis=0)
    out[:, ok] = series[:, ok] - design @ np.linalg.lstsq(design, series[:, ok], rcond=None)[0]
    return out


def seed_network_connectivity(seed_series, seed_names, parcel_series, parcel_networks, design=None, keep=None):
    """Fisher-z correlation of each seed with each cortical parcel, averaged within network.

    ``design, keep`` are ``nuisance_design``'s: seeds and parcels residualised together on the retained frames.
    Omit both only for denoised, censored input. NaN seeds or parcels give NaN and are left out of the network
    means. Returns {"<seed>__<network>": mean z}.
    """
    seeds = _residualise(seed_series, design, keep)
    parcels = _residualise(parcel_series, design, keep)
    if len(seed_names) != seeds.shape[1] or len(parcel_networks) != parcels.shape[1]:
        raise ValueError("names and series dimensions differ")
    zs = (seeds - seeds.mean(0)) / seeds.std(0)
    zp = (parcels - parcels.mean(0)) / parcels.std(0)
    r = (zs.T @ zp) / len(zs)
    z = np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))
    nets = np.asarray(parcel_networks)
    out = {}
    for i, seed in enumerate(seed_names):
        for net in sorted(set(parcel_networks)):
            values = z[i, nets == net]
            values = values[np.isfinite(values)]
            out[f"{seed}__{net}"] = float(values.mean()) if values.size else math.nan
    return out


def _standardise(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return x / sd


def group_ica(datasets, n_components=20, seed=0, max_iter=2000, designs=None, keeps=None):
    """Spatial group ICA on temporally concatenated, voxel-standardised data (the MELODIC/GIFT design).

    ``datasets``: list of (time x voxels) arrays in one common mask; ``designs``/``keeps``: per-dataset
    ``nuisance_design`` output, applied before standardising. Data are reduced by PCA to ``n_components`` and
    spatially independent sources are estimated with FastICA. Each map is z-scored and its sign fixed so that its
    largest-magnitude tail is positive. Returns (maps: components x voxels, info).
    """
    from sklearn.decomposition import FastICA

    designs, keeps = designs or [None] * len(datasets), keeps or [None] * len(datasets)
    stacked = np.vstack([_standardise(_residualise(d, x, k)) for d, x, k in zip(datasets, designs, keeps, strict=True)])
    if n_components >= min(stacked.shape):
        raise ValueError("more components than the data support")
    # PCA of the concatenated data: V_k spans the principal spatial patterns
    _, s, vt = np.linalg.svd(stacked, full_matrices=False)
    spatial = vt[:n_components].T * s[:n_components]        # voxels x components
    ica = FastICA(n_components=n_components, random_state=int(seed), max_iter=max_iter, whiten="unit-variance")
    sources = ica.fit_transform(spatial).T                  # components x voxels, spatially independent
    maps = (sources - sources.mean(1, keepdims=True)) / sources.std(1, keepdims=True)
    for i in range(len(maps)):
        if abs(maps[i].min()) > abs(maps[i].max()):
            maps[i] = -maps[i]
    explained = float((s[:n_components] ** 2).sum() / (s ** 2).sum())
    return maps, {"n_components": n_components, "pca_variance_retained": explained, "seed": int(seed),
                  "n_iter": int(ica.n_iter_)}


def load_reference_maps(path, mask_img, expected_sha256=None):
    """A frozen external group-ICA map set (4D NIfTI, one component per volume, on the analysis grid) as
    (components x voxels in ``mask_img``, provenance). Maps from outside the analysed sample keep the network
    definition independent of the subjects compared on it. ``expected_sha256`` pins the file: any other bytes raise.
    """
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Reference map set checksum differs from the registered one")
    img = nib.load(path)
    if img.ndim != 4 or img.shape[:3] != mask_img.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-5):
        raise ValueError("Reference maps must be a 4D image on the mask grid")
    return masked(img, mask_img), {"reference_maps": str(Path(path).resolve()), "sha256": digest,
                                   "n_components": int(img.shape[3])}


def select_component(maps, template, z_threshold=2.3):
    """Rank components by overlap with an independent template (boolean over the same voxels).

    Returns a list of {component, dice, spatial_r}, best Dice first. The caller should still inspect the
    chosen map visually, as the published analyses did.
    """
    template = np.asarray(template, bool)
    out = []
    for i, m in enumerate(maps):
        thr = m > z_threshold
        dice = 2 * (thr & template).sum() / max(thr.sum() + template.sum(), 1)
        out.append({"component": i, "dice": float(dice), "spatial_r": float(np.corrcoef(m, template.astype(float))[0, 1])})
    return sorted(out, key=lambda d: -d["dice"])


def dual_regression(data, maps, des_norm=True, *, design=None, keep=None):
    """Dual regression of one subject, as FSL's ``dual_regression`` v0.6 (its two ``fsl_glm`` calls).

    Stage 1 (``fsl_glm -d <maps> --demean``): every volume regressed on the group maps, maps and volume demeaned
    over voxels -> subject time courses (time x components, FSL's ``dr_stage1``). Stage 2 (``--demean
    [--des_norm]``): every voxel regressed on those time courses, both demeaned over time, the time courses scaled
    to unit standard deviation (n - 1, as FSL) when ``des_norm`` (FSL's recommended 1) -> subject maps (components x
    voxels, the parameter estimates of FSL's ``dr_stage2``). The data are not variance-normalised, as in FSL.
    ``data`` is (time x voxels) inside the common mask, which the caller builds (FSL: voxels with nonzero temporal
    SD in every subject). PIE addition: ``design, keep`` residualise the retained frames first; for stage 2 that
    equals fitting the nuisance regressors simultaneously, since stage-1 time courses of residuals are orthogonal
    to the design. Returns (time courses, subject maps).
    """
    y = _residualise(data, design, keep)
    m = np.asarray(maps, float)
    tc = np.linalg.lstsq((m - m.mean(1, keepdims=True)).T, (y - y.mean(1, keepdims=True)).T, rcond=None)[0].T
    x = tc - tc.mean(0)
    if des_norm:
        x = x / x.std(0, ddof=1)
    return tc, np.linalg.lstsq(x, y - y.mean(0), rcond=None)[0]


def bgn_weights(datasets, template, roi_labels, names, *, reference=None, designs=None, keeps=None,
                n_components=20, seed=0):
    """Basal-ganglia-network weight per region and subject: BGN chosen by ``template`` overlap, back-projected by
    ``dual_regression``, averaged by ``roi_means``. ``reference`` is ``load_reference_maps`` output and is used
    whenever given. Without it the maps come from group ICA of ``datasets`` themselves, which is circular (the
    network is defined by the subjects then compared on it) and warns. Returns (per-subject dicts, provenance)."""
    designs, keeps = designs or [None] * len(datasets), keeps or [None] * len(datasets)
    if reference is None:
        warnings.warn("BGN maps derived from the analysed sample are circular; pass a frozen reference map set",
                      stacklevel=2)
        maps, provenance = group_ica(datasets, n_components, seed, designs=designs, keeps=keeps)
        provenance = dict(provenance, source="analysed_sample")
    else:
        maps, provenance = reference
        provenance = dict(provenance, source="frozen_reference")
    best = select_component(maps, template)[0]
    weights = [roi_means(dual_regression(d, maps, design=x, keep=k)[1][best["component"]], roi_labels, names)
               for d, x, k in zip(datasets, designs, keeps, strict=True)]
    return weights, dict(provenance, component=best["component"], template_dice=best["dice"])


def roi_means(subject_map, roi_labels, names):
    """Mean map weight within each anatomical region (both over the same voxel vector)."""
    subject_map, roi_labels = np.asarray(subject_map, float), np.asarray(roi_labels)
    return {name: (float(subject_map[roi_labels == k].mean()) if (roi_labels == k).any() else float("nan"))
            for k, name in names.items()}


def masked(img, mask_img):
    """(time x voxels) array of a 4D image inside a mask on the same grid."""
    data, mask = np.asanyarray(img.dataobj), np.asarray(mask_img.dataobj) > 0
    if data.shape[:3] != mask.shape:
        raise ValueError("image and mask grids differ")
    return data[mask].T.astype(float)


def parcel_centroids(atlas_img, labels):
    """World (mm) centre of mass of each label, in the given order, for ``qc_fc``'s distance dependence."""
    data = np.asarray(atlas_img.dataobj)
    return np.array([nib.affines.apply_affine(atlas_img.affine, np.argwhere(data == k).mean(axis=0)) for k in labels])


def qc_fc(edges, mean_fd, *, covariates=None, centroids=None, alpha=0.05):
    """QC-FC motion benchmark (Ciric et al. 2017, doi:10.1016/j.neuroimage.2017.03.020; Parkes et al. 2018): across
    scans, the correlation of mean FD with every edge, partial on ``covariates`` (scans x k, e.g. age and sex).

    ``edges``: scans x edges (Fisher z, ``np.triu_indices(P, 1)`` order). ``centroids`` (P x 3 mm,
    ``parcel_centroids``) adds the distance dependence: Spearman rho of QC-FC with inter-parcel distance, negative
    when motion inflates short-range coupling (Power et al. 2012). Edges with a NaN in any scan are left out
    (``n_edges_excluded``). Returns the summary and the per-edge ``qcfc`` (NaN where left out).
    """
    from scipy import stats
    from scipy.spatial.distance import pdist

    edges, fd = np.asarray(edges, float), np.asarray(mean_fd, float)
    if edges.ndim != 2 or len(edges) != len(fd) or not np.isfinite(fd).all():
        raise ValueError("Need scans x edges and one finite mean FD per scan")
    design = np.column_stack([np.ones(len(fd))] + ([] if covariates is None else [np.asarray(covariates, float)]))
    usable = np.isfinite(edges).all(axis=0)
    e, f = (a - design @ np.linalg.lstsq(design, a, rcond=None)[0] for a in (edges[:, usable], fd))
    r = e.T @ f / (np.linalg.norm(e, axis=0) * np.linalg.norm(f))
    dof = len(fd) - design.shape[1] - 1
    p = 2 * stats.t.sf(np.abs(r) * np.sqrt(dof / (1 - r ** 2)), dof)
    qcfc = np.full(edges.shape[1], np.nan)
    qcfc[usable] = r
    out = dict(n_scans=len(fd), n_edges=int(usable.sum()), n_edges_excluded=int((~usable).sum()),
               median_abs_qcfc=float(np.median(np.abs(r))), percent_significant=float(100 * np.mean(p < alpha)),
               percent_significant_fdr=float(100 * np.mean(stats.false_discovery_control(p) < alpha)),
               distance_dependence_rho=None, qcfc=qcfc)
    if centroids is not None:
        distance = pdist(np.asarray(centroids, float))
        if len(distance) != edges.shape[1]:
            raise ValueError("centroids do not match the edges")
        out["distance_dependence_rho"] = float(stats.spearmanr(r, distance[usable]).statistic)
    return out
