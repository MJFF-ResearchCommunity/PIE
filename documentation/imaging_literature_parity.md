# Literature-parity measures (`dwi_tracts`, `fmri_striatal`, `nm_volume`, `volumes`, `stats.small_sample`)

Added 18 September 2026 so that PIE output can be set beside the multimodal imaging literature on
α-synuclein seed status, in particular Droby et al. 2025 (*npj Parkinson's Disease* 11:7), whose measures were
DaTscan striatal binding ratios, CAT12 volumes adjusted for intracranial volume, basal-ganglia-network
connectivity from group ICA, neuromelanin SN volume and intensity, and JHU tract FA, with an SVM classifier.

| Measure in the literature | PIE | Notes |
|---|---|---|
| DaTscan SBR, occipital reference, caudate and putamen | `datscan.py` (existing) | subject-space FastSurfer regions rather than template masks |
| Whole-brain GM and WM, putamen, caudate, pallidum, brainstem volumes | `volumes.tissue_volumes` | from FastSurfer labels |
| Intracranial-volume adjustment | `volumes.adjust_for_head_size`, `volumes.tiv_from_registration` | prefer FastSurfer `--tal_reg` eTIV; the registration estimate is a fallback still to be validated |
| Basal ganglia network from group ICA, caudate and putamen weights | `fmri_striatal.group_ica`, `select_component`, `dual_regression`, `roi_means` | template for BGN selection is the CIT168 striatum, independent of the patients |
| Seed-based striatal connectivity | `fmri_striatal.striatal_rois`, `roi_timeseries`, `seed_network_connectivity` | BOLD must already be in MNI152NLin2009cAsym |
| Neuromelanin SN volume and signal relative to white matter | `nm_volume.hyperintense_volume`, `mask_volume`, `normalised_intensity` | threshold volume depends several-fold on `k`; report it |
| Tract FA on the 48-label JHU ICBM-DTI-81 atlas | `dwi_tracts.fetch_jhu`, `map_labels_to_subject`, `registration_qc`, `tract_features` | atlas FA template registered to subject FA, so no MNI variant is assumed; laterality checked on load; `max_resolution_mm=2.0` registers finer grids (e.g. 1 x 1 x 2 mm reconstructions) on a 2 mm copy and still pulls labels onto the native grid |
| Partial correlations with bootstrap | `stats.small_sample.bootstrap_partial_correlation` | no pingouin dependency |
| SVM with feature-subset search and leave-one-out validation | `stats.small_sample.nested_subset_search`, `naive_subset_search`, `subset_search_null` | the search can run inside each validation fold; the other two help size selection bias in a given sample |

Real-data checks recorded in each module's docstring: JHU tract FA on a PPMI 2 mm scan (template FA
correlation 0.69 with SyN, 0.54 affine; posterior internal capsule 0.67, splenium 0.56), neuromelanin volumes on
40 scans, striatal regions on the fMRIPrep grid, and intracranial volume on 8 FastSurfer subjects.

Downloads (JHU from NeuroVault collection 264, TemplateFlow MNI152NLin2009cAsym res-02 maps) are cached and
checked by sha256; nothing with an unstated licence is bundled.

## Worked examples

Run these in the imaging environment (`venv_imaging/bin/python`). Every one uses synthetic images, so they
need no PPMI data and no downloads, and the printed values are what they actually produce.

### Tissue volumes and head-size adjustment (`volumes`)

`tissue_volumes` counts FreeSurfer/FastSurfer labels. Pass the segmentation FastSurfer already wrote
(`fastsurfer/<IMAGE_ID>/mri/aparc+aseg.mgz` or `aseg.mgz`); here a tiny array stands in for one.

```python
import nibabel as nib, numpy as np
from pie.imaging import volumes as vol

seg = np.zeros((10, 10, 10), np.int32)
seg[0:2] = 3          # left cerebral cortex
seg[3] = 2            # left cerebral white matter
seg[5, 0, 0:3] = 11   # left caudate
out = vol.tissue_volumes(nib.Nifti1Image(seg, np.diag([2, 2, 2, 1])))   # 2 mm voxels = 8 mm^3 each

out["cortical_gm"], out["white_matter"], out["caudate_l"]   # 1600.0, 800.0, 24.0  (mm^3)
out["total_gm"]                                             # 1624.0 = cortical + subcortical + cerebellar
```

Bigger heads have bigger structures, so compare volumes only after adjusting for intracranial volume. Fit
the adjustment on a reference group — controls, or the training fold — so no evaluation row informs it:

```python
rng = np.random.default_rng(0)
tiv = rng.normal(1.5e6, 1.2e5, 200)                     # mm^3
putamen = 0.004 * tiv + rng.normal(0, 200, 200)         # scales with head size
is_control = np.arange(200) < 100

adjusted = vol.adjust_for_head_size(putamen, tiv, method="residual", reference=is_control)
np.corrcoef(putamen, tiv)[0, 1], np.corrcoef(adjusted, tiv)[0, 1]    # 0.91 -> -0.05
```

Get `tiv` from FastSurfer's eTIV when you ran it with `--tal_reg`. PIE's own runs do not, so
`vol.tiv_from_registration(head_img, template_head, template_icv)` is the fallback — validate it on your
data first, and pass a head image with the skull (`orig.mgz` or the raw T1), never a skull-stripped one.

### Neuromelanin volume and intensity (`nm_volume`)

Two numbers the literature reports: the volume of nigral voxels brighter than a reference region, and the
mean nigral signal relative to that reference.

```python
import nibabel as nib, numpy as np
from pie.imaging import nm_volume as nv

rng = np.random.default_rng(0)
nm = rng.normal(100, 5, (30, 30, 10))     # background signal
sn = np.zeros(nm.shape, bool); sn[10:14, 10:15, 4:6] = True     # 40 bright nigral voxels
nm[sn] = 160
search = np.zeros(nm.shape, bool); search[8:16, 8:17, 3:7] = True   # atlas SN mask, slightly dilated
ref = np.zeros(nm.shape, bool); ref[20:28, 5:25, 2:8] = True        # reference region
img = lambda a: nib.Nifti1Image(a.astype(np.float32), np.diag([0.5, 0.5, 2.0, 1]))

v = nv.hyperintense_volume(img(nm), img(search), img(ref), k=3.0)
v["volume_mm3"], v["n_voxels"], round(v["threshold"], 1)    # 20.0, 40, 115.0  (mean + 3 SD of the reference)

i = nv.normalised_intensity(img(nm), img(sn), img(ref))
round(i["normalised_intensity"], 2), round(i["contrast_ratio"], 2)   # 1.6, 0.6
```

`search` decides the answer: pass the atlas SN mask, never a bounding box, or bright cisternal arteries
enter the count. `k` changes the volume several-fold (medians on 40 PPMI scans were 160, 66 and 21 mm^3 at
k = 2, 2.5 and 3), so report the `k` you used with every volume. All three images must be on one grid.

### JHU tract measures (`dwi_tracts`)

`tract_features` averages any maps you give it within each of the 48 ICBM-DTI-81 tracts. The label map
normally comes from `fetch_jhu` plus `map_labels_to_subject`, which needs ANTs; the averaging itself is
plain NumPy. `fa_img`, `md_img` and the label image below are your own, all on one grid:

```python
import nibabel as nib, numpy as np
from pie.imaging import dwi_tracts as dt

len(dt.LABELS), dt.LABELS[5]        # 48, 'splenium_corpus_callosum'

labels = nib.Nifti1Image(my_label_array, affine)        # 0..48, subject grid
feat = dt.tract_features({"fa": fa_img, "md": md_img}, labels, min_voxels=3)

feat["fa_superior_fronto_occipital_fasciculus_l"]   # mean FA in that tract
feat["n_superior_fronto_occipital_fasciculus_l"]    # voxels behind it
len(feat)                                           # 144 = 48 tracts x (fa, md, n)
```

Tracts with fewer than `min_voxels` usable voxels come back as `NaN`, never as a silently thin average.
`fa_floor=0.2` (the TBSS convention) restricts every tract to voxels above that FA, which reduces partial
volume with grey matter and CSF; a tract entirely below the floor becomes `NaN`. The real path is:

```python
labels_img, atlas_fa, provenance = dt.fetch_jhu(cache_dir)          # sha256 + laterality checked on load
subject_labels, tf = dt.map_labels_to_subject(subject_fa, atlas_fa, labels_img,
                                              brain_mask=mask, max_resolution_mm=2.0)
qc = dt.registration_qc(subject_fa, tf["warped_template_fa"], subject_labels, labels_img, mask)
qc["template_fa_correlation"], qc["qc_pass"]        # check before you use the features
```

`max_resolution_mm` defaults to `None`, i.e. registration runs on the native grid; pass `2.0` to register
finer reconstructions on a 2 mm copy while still pulling labels onto the native grid. Read the QC before
modelling: on PPMI 2 mm data SyN gave a template-FA correlation of 0.69 where affine alone gave 0.54.

### Striatal and basal-ganglia-network connectivity (`fmri_striatal`)

BOLD must already be in MNI152NLin2009cAsym (fMRIPrep `--output-spaces MNI152NLin2009cAsym`). The striatal
regions come from the bundled CIT168 atlas, so this needs no download:

```python
from pie.imaging import fmri_striatal as fs

roi_img, names = fs.striatal_rois(target_img=bold_img)      # resampled onto the BOLD grid
names        # {1: 'caudate_l', 2: 'caudate_r', 3: 'putamen_l', 4: 'putamen_r', 5: 'accumbens_l', 6: 'accumbens_r'}

series, labels, counts = fs.roi_timeseries(bold_img, roi_img, mask_img=brain_mask)
z = fs.seed_network_connectivity(series, [names[k] for k in labels],
                                 parcel_series, parcel_networks, design=nuisance)
z["putamen_l__sensorimotor"]        # Fisher-z, averaged over that network's parcels
```

`design` is the nuisance matrix, applied identically to seeds and parcels; `fmri_connectivity.nuisance_design`
builds one. Regions with fewer than `min_voxels` usable voxels give `NaN` rather than a mean over nothing.

The network-based route reproduces the group-ICA analyses. Components are identified by overlap with an
independent template — the CIT168 striatum, never the patients' own data:

```python
maps, info = fs.group_ica(datasets, n_components=20, seed=0)    # datasets: list of (time x voxels)
ranked = fs.select_component(maps, striatum_mask_1d)            # best Dice first
bgn = ranked[0]["component"]

timecourses, subject_maps = fs.dual_regression(one_subject, maps)
fs.roi_means(subject_maps[bgn], roi_labels_1d, names)           # {'caudate_l': ..., 'putamen_l': ...}
```

On a synthetic set of 8 subjects whose striatal-network strength rose from 0.3 to 1.5, the selected
component matched the template (Dice 1.0) and the per-subject weights rose monotonically with it. Inspect
the chosen map visually anyway, as the published analyses did. CIT168 at its 0.25 threshold is generous —
each caudate borders the lateral ventricle — so erode the regions or intersect them with grey matter when
partial volume with CSF matters.

### Keeping selection honest (`stats.small_sample`)

The classifier side of these comparisons belongs to `pie.stats.small_sample`, which runs the feature-subset
search inside each validation fold instead of over the whole sample. It lives in the tabular environment
and is documented with examples in [Statistics](stats.md#small_sample).
