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
| Tract FA on the 48-label JHU ICBM-DTI-81 atlas | `dwi_tracts.fetch_jhu`, `map_labels_to_subject`, `registration_qc`, `tract_features` | atlas FA template registered to subject FA, so no MNI variant is assumed; laterality checked on load |
| Partial correlations with bootstrap | `stats.small_sample.bootstrap_partial_correlation` | no pingouin dependency |
| SVM with feature-subset search and leave-one-out validation | `stats.small_sample.nested_subset_search`, `naive_subset_search`, `subset_search_null` | the nested version runs the search inside each fold; the null shows how high the naive design scores on noise |

Real-data checks recorded in each module's docstring: JHU tract FA on a PPMI 2 mm scan (template FA
correlation 0.69 with SyN, 0.54 affine; posterior internal capsule 0.67, splenium 0.56), neuromelanin volumes on
40 scans, striatal regions on the fMRIPrep grid, and intracranial volume on 8 FastSurfer subjects.

Downloads (JHU from NeuroVault collection 264, TemplateFlow MNI152NLin2009cAsym res-02 maps) are cached and
checked by sha256; nothing with an unstated licence is bundled.
