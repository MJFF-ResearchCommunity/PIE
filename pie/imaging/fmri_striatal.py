"""Striatal and basal-ganglia-network connectivity from preprocessed resting-state fMRI.

Two approaches, the two the Parkinson's literature uses:

1. Region-based: mean BOLD in anatomical striatal regions (caudate, putamen, nucleus accumbens, per hemisphere,
   from the bundled CIT168 atlas in MNI152NLin2009cAsym), correlated with cortical parcels and averaged by network.
   Requires BOLD already in MNI152NLin2009cAsym (fMRIPrep ``--output-spaces MNI152NLin2009cAsym``); grids are
   reconciled by nearest-neighbour resampling, never by registration.

       striatal_rois(target)                       caudate/putamen/accumbens L/R label image on the BOLD grid
       roi_timeseries(bold, rois, mask)            mean signal per region, (time x regions)
       seed_network_connectivity(...)              Fisher-z of each striatal region with each cortical network

2. Network-based, as in Szewczyk-Krolikowski 2014, Rolinski 2016 and Droby 2025: spatial group ICA on temporally
   concatenated data, identification of the basal ganglia network (BGN) by overlap with an independent striatal
   template, back-projection to each subject by dual regression, and mean subject-map weight in anatomical masks.

       group_ica(datasets, n_components)           spatial ICA maps (components x voxels), sign-fixed and z-scored
       select_component(maps, template)            Dice and spatial correlation of every component with a template
       dual_regression(data, maps)                 subject time courses (stage 1) and subject maps (stage 2)
       roi_means(subject_map, roi_labels, names)   mean weight per anatomical region

CIT168 regions at its 0.25 probability threshold are generous: on the fMRIPrep 2 mm grid each caudate is about
860 voxels (6.9 mL) and each putamen about 970 (7.8 mL), so the caudate borders the lateral ventricle. Erode the
regions or restrict them to a grey-matter mask when partial volume with CSF matters.

Choices that change results are parameters with the literature's defaults, and the template used to pick the
BGN must be independent of the patients analysed (Griffanti et al. 2016): the CIT168 striatum qualifies.
Outcome labels are never read here.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

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
