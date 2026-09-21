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


def _network_names(networks):
    """Names become "A__B" feature keys: nonstrings would fail late or collide (1 vs "1")."""
    networks = list(networks)
    if not all(isinstance(name, str) and name for name in networks):
        raise ValueError('Network names must be nonempty strings')
    return networks


def residual_connectivity(series, design, networks):
    """OLS residualization and Fisher-z edges/network means (self-edges excluded)."""
    networks = _network_names(networks)
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
        result['network_features'] = features
    # Created only after every check, so a rejected call leaves nothing behind.
    out.mkdir(parents=True, exist_ok=True)
    if not reasons:
        np.savez_compressed(out / 'connectivity.npz', parcel_timeseries=series,
                            retained_indices=np.flatnonzero(keep), residual_timeseries=residual,
                            edges=edges, labels=np.array(labels))
        result['outputs']['connectivity.npz'] = sha256(out / 'connectivity.npz')
    write_json(completion, result)
    return result


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
#        select_component(maps, template)            Dice and spatial correlation of every component with a template
#        dual_regression(data, maps)                 subject time courses (stage 1) and subject maps (stage 2)
#        roi_means(subject_map, roi_labels, names)   mean weight per anatomical region
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


def striatal_rois(target_img=None, regions=STRIATAL):
    """Caudate, putamen and accumbens per hemisphere from CIT168 (MNI152NLin2009cAsym).

    CIT168 labels are bilateral; hemispheres are split at MNI x = 0 in world coordinates, which is valid only
    because the atlas is verified to be in MNI152NLin2009cAsym. Returns (label image, {id: name}).
    """
    from .atlases import cit168_mni2009c, cit168_metadata

    atlas = cit168_mni2009c()
    names = cit168_metadata()["labels"]
    data = np.asarray(atlas.dataobj)
    ijk = np.indices(data.shape).reshape(3, -1).T
    x = nib.affines.apply_affine(atlas.affine, ijk)[:, 0].reshape(data.shape)
    out = np.zeros(data.shape, np.int16)
    mapping = {}
    k = 1
    for region, code in regions.items():
        label = names.index(code) + 1
        for side, sel in (("l", x < 0), ("r", x > 0)):
            out[(data == label) & sel] = k
            mapping[k] = f"{region}_{side}"
            k += 1
    img = nib.Nifti1Image(out, atlas.affine)
    if target_img is not None:
        img = resample_from_to(img, (target_img.shape[:3], target_img.affine), order=0)
        img = nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), target_img.affine)
    return img, mapping


def roi_timeseries(bold_img, roi_img, mask_img=None, min_voxels=5):
    """Mean BOLD per region (time x regions) and the voxel count behind each; regions below ``min_voxels`` -> NaN."""
    bold = np.asarray(bold_img.dataobj, np.float64)
    rois = np.asarray(roi_img.dataobj)
    if rois.shape != bold.shape[:3]:
        raise ValueError("region image is not on the BOLD grid")
    inside = np.ones(rois.shape, bool) if mask_img is None else np.asarray(mask_img.dataobj) > 0
    inside &= np.isfinite(bold).all(axis=-1) & (bold.std(axis=-1) > 0)
    labels = [k for k in np.unique(rois) if k > 0]
    series, counts = np.full((bold.shape[-1], len(labels)), np.nan), {}
    for j, k in enumerate(labels):
        sel = (rois == k) & inside
        counts[int(k)] = int(sel.sum())
        if sel.sum() >= min_voxels:
            series[:, j] = bold[sel].mean(axis=0)
    return series, labels, counts


def _residualise(series, design):
    series = np.asarray(series, float)
    if design is None:
        return series - series.mean(axis=0)
    return series - design @ np.linalg.lstsq(design, series, rcond=None)[0]


def seed_network_connectivity(seed_series, seed_names, parcel_series, parcel_networks, design=None):
    """Fisher-z correlation of each seed with each cortical parcel, averaged within network.

    ``design`` is the nuisance matrix (time x regressors, with intercept) applied identically to seeds and
    parcels, e.g. ``fmri_connectivity.nuisance_design``. Returns {"<seed>__<network>": mean z}.
    """
    seeds = _residualise(seed_series, design)
    parcels = _residualise(parcel_series, design)
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
            out[f"{seed}__{net}"] = float(np.nanmean(z[i, nets == net]))
    return out


def _standardise(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return x / sd


def group_ica(datasets, n_components=20, seed=0, max_iter=2000):
    """Spatial group ICA on temporally concatenated, voxel-standardised data (the MELODIC/GIFT design).

    ``datasets``: list of (time x voxels) arrays in one common mask. Data are reduced by PCA to
    ``n_components`` and spatially independent sources are estimated with FastICA. Each map is z-scored and
    its sign fixed so that its largest-magnitude tail is positive. Returns (maps: components x voxels, info).
    """
    from sklearn.decomposition import FastICA

    stacked = np.vstack([_standardise(d) for d in datasets])
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


def dual_regression(data, maps, normalise_timecourses=True):
    """FSL-style dual regression for one subject.

    Stage 1 regresses the group maps on each volume to give subject time courses (time x components);
    stage 2 regresses those time courses (variance-normalised by default, FSL ``--des_norm 1``) on each
    voxel to give subject maps (components x voxels). Returns (timecourses, subject_maps).
    """
    y = _standardise(data)                                  # time x voxels
    m = np.asarray(maps, float)                             # components x voxels
    tc = np.linalg.lstsq(m.T, y.T, rcond=None)[0].T        # time x components
    tc = tc - tc.mean(0)
    if normalise_timecourses:
        tc = tc / tc.std(0)
    subject_maps = np.linalg.lstsq(tc, y, rcond=None)[0]   # components x voxels
    return tc, subject_maps


def roi_means(subject_map, roi_labels, names):
    """Mean map weight within each anatomical region (both over the same voxel vector)."""
    subject_map, roi_labels = np.asarray(subject_map, float), np.asarray(roi_labels)
    return {name: (float(subject_map[roi_labels == k].mean()) if (roi_labels == k).any() else float("nan"))
            for k, name in names.items()}


def masked(img, mask_img):
    """(time x voxels) array of a 4D image inside a mask on the same grid."""
    data, mask = np.asarray(img.dataobj, float), np.asarray(mask_img.dataobj) > 0
    if data.shape[:3] != mask.shape:
        raise ValueError("image and mask grids differ")
    return data[mask].T
