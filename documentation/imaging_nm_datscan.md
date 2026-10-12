# Neuromelanin MRI and DaTscan SPECT

Part of the [imaging layer](imaging.md). Both read the subject's FastSurfer T1 (`orig.mgz`, `mask.mgz`,
`aparc.DKTatlas+aseg.deep.mgz`) as anatomy and write one row per subject.

| Module | Output | External tools |
|---|---|---|
| `nm.py` | `nm_features.csv`: the averaged slab, its QC, and the atlas-ROI ring contrast (a diagnostic since 11 Oct 2026) | dcm2niix, SimpleITK (the MNI target is bundled with PIE) |
| `nm_template.py` | `nm_template_features.csv`: contrast in a study NM template | + ANTsPy (`ants`), scikit-image, nilearn; the 1 mm MNI152NLin2009cAsym target is fetched from TemplateFlow once |
| `nm_native.py` | `nm_native_features.csv`: published measures on the native slab (Langley SNc volume; snceg segmentation, **PIE's default NM measure**) | + ANTsPy; snceg in its own environment (`PIE_SNCEG_PYTHON`) |
| `datscan.py` | `datscan_sbr.csv`: striatal binding ratios from raw projections | pydicom, scikit-image, SimpleITK |

## Neuromelanin MRI in PIE: methods, evidence and status

Neuromelanin-sensitive MRI shows the pigmented dopaminergic neurons of the substantia nigra pars compacta (SNc) as a bright band [Sasaki 2006; Cassidy 2019]. In PD the band loses signal and shrinks, first in its posterolateral (sensorimotor) part [Biondetti 2020].

**How the field quantifies it.**

| Approach | Status in the field | PIE |
|---|---|---|
| **SNc volume from a reference-based threshold, in native space**: voxels in a nigral search region brighter than a crus/peduncle reference mean + k SD | The most widely used and best-replicated measure. Meta-analyses find volume beats contrast [Cho 2021]; scan–rescan reliability is high [Langley 2016; Wengler 2020]. Published on this PPMI protocol, Siemens only [Langley 2025]: HC 395 ± 117 mm³, de novo PD 309 ± 100 mm³ (d ≈ 0.83). Emory Siemens cohorts: AUROC 0.756 and 0.749 [Hwang 2023] | `nm_native` (`nml_*`) on MP-PCA-denoised repeats (`nm --denoise`), k = 2.8 as published. Reproduction pending (below) |
| **Contrast ratio or CNR of the SN against a crus or tegmental reference** | Standard; slightly less discriminative than volume [Cho 2021] | `nm.py` (atlas ROI, ring reference); `nm_template` (template mask, crus mode); `nm_template published` (Biondetti territories); `nm_native` snceg CR/CNR |
| **Deep-learning segmentation of the NM-hyperintense SN** | State of the art. snceg is public (MIT): a multi-site-trained Attention U-Net, validated on PPMI GRE-MT segmentation. PPMI TSE: PD vs HC AUROC 0.743 with the contrast ratio against the crus [Lillebostad 2025] | `nm_native --snceg` (`nms_*`), pinned weights. Reproduction pending |
| **Voxelwise analysis in template space with CNR against the crus mode** | State of the art for group maps; reproducible [Wengler 2020; Cassidy 2019] | `nm_template` (`nmt_*`) |
| **Functional territories** (associative, limbic, sensorimotor) | State of the art for early-PD localisation [Biondetti 2020] | `nm_template published` (`nmb_*`), `nm_native` region codes |
| **Multi-site harmonisation** | ComBat preserves the age effect while removing scanner effects [Wengler 2021]; vendor matters [Trujillo 2024] | `nm_acquisition_batch` in the manifest, harmonised in-fold (`pie.experiment.prediction.ImageDesign`) |

**What PIE found on PPMI** (September 2026 download, pre-registered tests; see "Validation status" below).
- Mean contrast in fixed masks separates de novo PD from HC only modestly: AUROC 0.54–0.62. This holds for the atlas ROI, the template, and the Biondetti territories on 106 held-out PD.
- A cross-validated 8-territory model reaches 0.67.
- The sensorimotor-territory loss and relative limbic sparing replicated across samples.
- Separation depends strongly on scanner vendor. Unadjusted sensorimotor AUROC is GE 0.53, Siemens 0.70 and Toshiba 0.97 (6 HC / 5 PD). The published PPMI volume study used Siemens only.
- PIE's download held NM for 48 HC and 170 PD, where PPMI's acquisition metadata lists 112 HC and 631 PD. The missing scans, mostly baseline, are the main limit on precision.

**PIE's position.**
- **Default measure (11 Oct 2026): snceg** (`nm_native --snceg`): the SN contrast ratio against each side's crus on the
  slices holding that side's segmented SN (`nms_sn_mean_cr`), and the segmented SN volume (`nms_sn_volume_mm3`),
  assembled as `nm_primary_cr` and `nm_primary_volume_mm3`. It is the only NM measure that reproduced a published PPMI
  result (R2, TSE-FS). On PPMI's 2D GRE-MT it failed the pre-registered Siemens re-test (12 Oct 2026: AUROC 0.631
  [0.555, 0.707] against the 0.743 anchor; [preregistration_nm_snceg.md](preregistration_nm_snceg.md)), so
  `NM_VALIDATED` stays False: on GRE-MT the contrast ratio is a weak marker (above chance), the volume carries no
  signal, and neither is a validated diagnostic measure. `nm.py`'s ring contrast
  (`nm_sn_*_cnr`) is kept, under its old names, as a diagnostic.
- **Native-space threshold volume (`nm_native`, Langley method).** It did not reproduce (R1, R1-v2 and R1b below), so it is exploratory until the authors' regions are available.
- **snceg segmentation.** This is the state-of-the-art alternative, and it reproduced its published PPMI TSE-FS result (R2).
- **The study's contrast ratio** (`nm_native --study`, `nm_bion_*`). It was tested in a pre-registered out-of-sample replication (27 September 2026) on 369 HC/PD that no NM test had used, 298 of them usable (251 PD, 47 HC; 293 Siemens).
  - **The study's control replicated.** The nested NM family over the covariates added +0.121 [+0.027, +0.213] AUROC (study: +0.211). The family reached AUROC 0.635 [0.548, 0.718], and the whole-nigra scalar 0.611 (study: 0.605).
  - **PIE's own gate was not met.** It needs AUROC ≥ 0.70 with a lower bound > 0.55, and a DaT correlation ≥ 0.20; the DaT correlation was ρ 0.14. So `NM_VALIDATED` stays False: the measure detects Parkinson's disease out of sample, but too weakly for PIE to call it validated.
- **Template and territory contrasts.** These remain for voxelwise and regional work.
- **Everything assembled.** `manifest.NM_VALIDATED` stays False until a measure passes.

**Reproductions of published methods** (pre-registered 26–27 September 2026; scripts hashed before any value):

| | Published | Target | PIE result |
|---|---|---|---|
| R1 [Langley 2025] | PPMI Siemens GRE-MT, SNc volume, d ≈ 0.83 | p < 0.05 and PIE's d CI contains 0.83 | **Not reproduced.** 68 HC and 348 de novo PD, 10 Siemens models. HC 711 ± 272 vs PD 761 ± 292 mm³; adjusted d −0.06 [−0.29, 0.18]; AUROC 0.45 [0.38, 0.52]. The 21 HC / 82 PD pilot gave d 0.01 |
| R1-v2 | as R1 | as R1 | **Not reproduced.** Version 2 (JHU ICBM-DTI-81 cerebral peduncle as the reference, as in Langley's text), same slabs: HC 112 ± 73 vs PD 106 ± 62 mm³; d 0.18 [−0.09, 0.46]; AUROC 0.50 |
| R1b [Hwang 2023] | Emory Siemens, AUROC 0.756 / 0.749 | CI contains the published AUROC | **Not reproduced** (v1): 0.475 and 0.513 on the authors' own Dryad release (doi:10.6086/D1709N, CC0). **v2**: cohort 1 0.558 [0.361, 0.737], not reproduced; cohort 2 0.671 [0.540, 0.792], reproduced by the CI rule. Volumes are 4–6 times below the published scale |
| R2 [Lillebostad 2025] | PPMI Siemens TSE-FS, snceg CR vs crus, AUROC 0.743 | CI contains 0.743, lower bound > 0.5 | **Reproduced.** AUROC 0.710 [0.611, 0.799] (TSE-FS echo 1, earliest TSE-FS session, same-day T1) |
| GRE-MT re-test, v7 (pre-registered 11 Oct 2026) | PPMI Siemens 2D GRE-MT (R1 sample), snceg CR vs crus, anchor 0.743 | CI contains 0.743, lower bound > 0.5 | **Failed.** AUROC 0.631 [0.555, 0.707], 66 HC / 342 de novo PD; snceg volume 0.504 (no signal); see [the pre-registration](preregistration_nm_snceg.md#result-2026-10-12-the-single-run) |

Langley's threshold volume is not reproduced with public substitute regions, on PPMI or on the authors' own data. Their SNc atlas, its dilation and their peduncle ROI are not public. They were requested from the authors on 27 September 2026, and a version 3 waits on that material; nothing is tuned against these results in the meantime. snceg on the PPMI TSE-FS scans reproduces its published result. On the R1 GRE-MT scans, the snceg contrast ratio against the crus separated de novo PD from HC at d 0.39 [0.15, 0.65], AUROC 0.60. That is descriptive, because there is no published GRE-MT target.

**Why R1 differs from Langley 2025** (diagnosed on the pilot scans, without group labels).
- PIE's search region is the Biondetti nigra plus a 1 mm ring: about 1,600 mm³ on the native slab.
- After MP-PCA the crus is very uniform (CV 3.7 %), so the mean + 2.8 SD threshold sits only about 10 % above the crus.
- The nigra's mean contrast is about 12 %, so roughly half its voxels fall on either side of the threshold, and the volume tracks noise around it.
- PIE's control volumes (706 mm³) are about 1.8 times Langley's (395 mm³).

With the public Biondetti ROIs, this is therefore not Langley's measure. A faithful reproduction needs their SNc atlas and cerebral-peduncle ROI, which are not public; the corresponding author is X. P. Hu, UC Riverside. PIE has not tuned the search region or k after seeing the pilot. Any such change would be exploratory.

## Neuromelanin-sensitive MRI (`nm.py`)

PPMI-2 acquires a 2D T1-weighted gradient echo with a magnetization-transfer pulse through the midbrain
(0.5 x 0.5 x 1.5 mm, 16 slices, TR 0.45-0.65 s, TE ~5 ms, flip 40), usually five repeats; the study's September
2026 full-MRI download (snapshot) holds them for 654 subjects (433 prodromal, 172 PD, 49 HC). Site descriptions vary
("AX T2 GRE MT", "2D GRE-MT", "AXIAL 2D GRE-MT", "2D GRE-MT_ACPC", "NM-GRE", "NM-MT", …). `flag_nm(idx)` selects
descriptions matching `NM_PATTERN` and not `EXCLUDE` (`MTC-NO|B0|Map|TRACEW|ADC|_FA`).

1. **`compatible_repeats(niis, minimum=1, limit=None)`** — keeps magnitude images only (drops `ImageType` P/PHASE,
   R/REAL, I/IMAGINARY), groups by shape, zooms, TE, TR, flip, `MTState` and reconstruction tags, and takes the
   largest group (NORM wins a tie). Siemens NORM and non-NORM series can be two reconstructions of *one*
   acquisition, so they are never averaged as repeats. A repeated acquisition timestamp/number within one
   reconstruction raises (`Ambiguous NM acquisition identity`): identical times may be anonymisation, not
   duplicates, and a silent collapse would turn five repeats into one. Exact duplicate arrays raise. Missing
   timestamps are counted (`n_missing_acquisition_identity`) and must be resolved before an
   independent-acquisition reliability study.
2. **`average_repeats(niis, sampling_seed=0, transforms_dir=None)`** — rigid MI alignment of each repeat to the
   first, then the coverage-weighted mean: each voxel over the repeats whose field of view reaches it. Until 11 Oct
   2026 the sum was divided by the number of repeats, which darkened a moved repeat's missing edge slices by 1/n per
   missing repeat. Returns `(image, meta, n_repeats, max_motion_mm, support)`: `support` is, per voxel, the
   share of repeats whose field of view reaches it (1 = every repeat; a repeat that moved leaves the slab's edge
   slices averaged over fewer images). `meta["PIERepeatSelection"]` holds the selection record (written to
   `<work>/<patno>/repeat_selection.json`). With `transforms_dir` it also saves `repeat_<i>_to_anchor.tfm` and
   the support as `repeat_support_fraction.nii.gz`.
3. **`register_slab(nm_img, fastsurfer_dir, sampling_seed=0)`** — **`register_nm_to_t1(nm_img, t1_img,
   target_center=None, sampling_seed=0)`** with the slab as fixed image (every metric sample lies inside the
   ~24 mm slab) against the *unmasked* T1: with a brain-masked T1 a thin slab matched several heights equally
   well and settled on the striatum for some subjects, whereas eyes, sinuses and skull pin the height (MI
   metric -0.73 vs -0.43). Both starts are always fitted: the same-session headers, and the slab centred on the
   atlas SN centroid in T1 space. Of the fits whose slab holds at least 90 % of each side's atlas SN
   (`slab_coverage(points, tx_t1_nm, grid)`), the lowest MI metric wins; if neither does, the better-covered one
   (the study's rule, ported 27 Sep 2026; `reg_init` = `header` | `sn_centroid`). T1 → MNI is the subject's SyN
   (`nm_template.syn_cache`, fitted when missing, so `nm` now includes the `syn` stage): the CIT168 atlas
   (`dwi.pauli_atlas()`, MNI2009c v1.0) and the anatomical hemispheres reach the slab in one label resampling
   (`nm_native.pull_labels`, slab → T1 rigid, T1 → MNI inverse SyN), as in the study. Until 27 Sep 2026 the
   mapping was the affine cached by `dwi.register_t1_to_mni` (`reg_t1_mni_mi`). On 24 development slabs, re-mapped
   without reading diagnosis, the SyN put the atlas SN nearer the neuromelanin band: the position refinement
   reached its ±2 mm edge on 35 % of sides against 73 % through the affine, the unrefined left contrast rose
   (median +0.02), and the refined contrast was unchanged (median −0.007 left, −0.004 right). The SyN's SN mask
   is about 20 % larger on the slab, because it follows the subject's own midbrain shape.
4. **`nm_rois(fs_lab, pauli_lab, spacing_mm, left_mask=None)`** — SN = CIT168 SNc + SNr per side; reference ring
   = SN dilated `DILATE_MM` (3 mm, anisotropic) minus the dilated nuclei (SNc, SNr, RN, VTA, STN, PBP), inside
   brainstem / ventral DC / cerebral WM / thalamus labels. Hemispheres come from `atlas_left_mask(tgt, chain)`
   (the MNI left half-space mapped onto the slab), because the slab's voxel x direction is not an anatomical
   label; without it, bilateral putamen landmarks are required or it raises.
5. **`features(nm, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), smooth_fwhm_mm=1.0, mask_out=None, support=None, refine=False)`** —
   the atlas SN is read where the SyN puts it. With `refine=True` (the default until 11 Oct 2026) each side's
   atlas SN is refined by the translation (<= `REFINE_MM` = 2 mm along each in-plane axis, +-1 slice; the
   Euclidean shift can exceed 2 mm) that maximises the ROI mean on a 1 mm-smoothed copy while keeping >= 95 % of
   the ROI in brainstem labels (the cistern lateral to the peduncle is bright). Masks are shifted with zero
   padding, never wrapped. Both final refined masks are excluded from the shared reference before any contrast is
   computed, and that actual reference is saved (`ref_nm.nii.gz`). Contrasts are read from the unsmoothed slab:
   CNR = (SN − ref) / ref, a contrast ratio, not a noise-SD-normalised CNR.

The same-side ring part anterior to the SN (nominally crus cerebri) is kept as `*_cnr_crus`. It is an anatomical
approximation, not an expert-drawn crus-cerebri reference, and it is not the primary reference: it lies next to
the neuromelanin band and is contaminated by it. Same-image intensity refinement can introduce selection bias;
independent-repeat agreement and native anatomical review are needed, and neither conditional repeatability nor
an atlas overlay establishes anatomical accuracy. Selecting the position on the image it then measures added about
+0.023 to the contrast on PPMI slabs, more than the HC–PD gap, so since 11 Oct 2026 refinement is off by default
(`nm_sn_shift_mm_*` = 0, `*_cnr` = `*_cnr_atlas`); `features(refine=True)` keeps it for comparison. Order statistics on
single voxels (brightest-fraction contrast, fixed-threshold volumes) were removed after the 2026-09 audit: they
tracked the slab noise level (per-voxel CV ~0.13 in PPMI's 5-average 1.5 mm protocol; Spearman 0.65-0.71 with the
reference CV) and showed no PD-vs-HC difference, whereas the refined ROI mean was ~9-10 % lower in de novo PD (AUROC
~0.6 on 23 HC vs 103 PD). Those group comparisons were made before the atlas-space correction and are
historical and exploratory; they do not validate the corrected pipeline. The slab does not reach the locus
coeruleus.

`process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, sampling_seed=0,
keep_raw=False)` runs it all. `sampling_seed=0` keeps ITK's legacy wall-clock seed; pass a nonzero seed for
repeatability studies.

### CLI

```bash
venv_imaging/bin/python -m pie.imaging.nm --zips <full-MRI zips> --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/nm --workers 4 [--keep-nifti]
venv_imaging/bin/python -m pie.imaging.nm --zips <zips> --sessions ... --fastsurfer-dir ... --work-dir <derived>/nm --refeature
# -> <derived>/nm/nm_features.csv, nm_index.csv
```

Flags are the shared `batch.add_common_args` set (`--zips`, `--sessions`, `--fastsurfer-dir`, `--work-dir`
required; `--workers 4`, `--limit`, `--patnos`, `--retry-errors`, `--keep-nifti`, `--pid-file`; meanings as in
the [DWI flag table](imaging_dwi.md#cli)) plus:

| Flag | Meaning |
|---|---|
| `--refeature` | Recompute the feature columns of every finished subject from the saved slab and label maps (`refeature_subject(work_dir, patno, fastsurfer_dir=None, atlas_sha256=None, registration_reference_sha256=None)`), without registering. Needs `--keep-nifti` outputs. The saved atlas map is reused only when the row's `atlas_sha256` **and** `registration_reference_sha256` match the bundled atlas and its reference template: the current atlas hash alone does not authorise an old mapping, because the map may have been made against another reference. Otherwise (rows from before 2026-09-15, and affine-mapped rows from before 2026-09-27, whose `registration_reference_sha256` is the 2 mm template's rather than the SyN's 1 mm TemplateFlow T1w) the atlas and hemisphere maps are regenerated from `slab_to_t1.tfm` and the subject's SyN cache (`nm_template syn`) and rewritten, `sn_slab_coverage_{l,r}` and the provenance columns are recomputed, and a subject without those transforms gets an `error` instead of stale-atlas values. Rows are labelled `<PROCESSING_VERSION>-refeatured`. The previous table is kept once as `nm_features.pre_refeature.csv`; `--patnos` restricts the run; `--zips` is required by the parser but unused |

`--keep-nifti` saves, per subject: `nm_mean.nii.gz`, `pauli_nm.nii.gz`, `aseg_nm.nii.gz`, `ref_nm.nii.gz`,
`left_nm.nii.gz`, `sn_refined_nm.nii.gz` (1 = left, 2 = right), `slab_to_t1.tfm` (T1 point → slab point; reused by
`nm_template`) and `repeat_transforms/`.

### Output columns (`nm_features.csv`)

| Column | Meaning |
|---|---|
| `nm_sn_{l,r}_cnr` | Atlas SN vs the whole ring (diagnostic; refined position only with `features(refine=True)`) |
| `nm_sn_{posterior,anterior}_{l,r}_cnr` | Halves of the refined SN, split at its median A-P coordinate |
| `nm_sn_{l,r}_cnr_crus`, `nm_sn_{l,r}_cnr_atlas` | Same-side anterior reference; unrefined atlas SN vs ring |
| `nm_sn_mean_cnr`, `nm_sn_posterior_mean_cnr`, `nm_sn_anterior_mean_cnr`, `nm_sn_min_cnr`, `nm_sn_asym_cnr`, `nm_sn_mean_cnr_atlas` | Bilateral mean, min, \|L−R\| |
| `nm_sn_{l,r}_mean`, `nm_ring_mean`, `nm_ring_sd`, `n_ring`, `nm_ref_{l,r}_mean`, `nm_ref_{l,r}_sd`, `n_ref_{l,r}`, `n_sn_{l,r}` | Raw intensities and voxel counts |
| `nm_sn_shift_mm_{l,r}` | Refinement displacement (mm); 0 unless `refine=True` |
| `n_repeats`, `repeat_motion_mm_max`, `nm_clipped_fraction`, `reg_nm_t1_mi`, `reg_t1_mni_mi` (affine-era rows only), `reg_init`, `sn_slab_coverage_{l,r}`, `nm_valid_{l,r,ref}`, `nm_support_{l,r,ref}` | QC |
| `shape`, `voxel_mm`, `manufacturer`, `model`, `tr_s`, `te_s`, `flip_angle`, `mt_flag`, `series_desc`, `n_series` | Acquisition |
| `patno`, `acquisition_date`, `fs_image_id`, `processing_version`, `atlas_space`, `atlas_version`, `atlas_sha256`, `atlas_probability_threshold`, `registration_reference_space`, `registration_reference_sha256`, `error` | Lineage |

`sn_slab_coverage_{l,r}` is the share of each side's atlas SN voxels (1 mm T1 space) that the chosen rigid maps
inside the slab (voxel centres ± half a voxel): a bounded geometric fraction. Tables written before 27 Sep 2026 have
`sn_slab_coverage` instead, (SN voxels on the slab x voxel volume) / atlas SN voxels in T1 space, a resampled-volume
ratio that exceeds 1 on fine slabs and can pass a slab that misses one side. `nm_valid_{l,r,ref}` is the share of
the measured masks (refined SN per side, ring) with signal (finite, > 0); `nm_support_{l,r,ref}` the share that every
repeat reaches (support ≥ 0.99). New complete runs are labelled `nm.PROCESSING_VERSION` = `2026-10-11-coverage-weighted-unrefined-v7`
(`2026-09-27-study-syn-slab-qc-v6` before coverage-weighted averaging and unrefined ROIs, `…-v5` before the study's slab
checks were ported); `--refeature` rows get that string plus `-refeatured`. `--refeature` applies the unrefined ROIs
but not the averaging, which needs the raw repeats: re-run `nm`.
`manifest.QC["nm"]`: `n_sn_l >= 20`, `n_sn_r >= 20`, `min(sn_slab_coverage_l, sn_slab_coverage_r) >= 0.9` (older
rows: `sn_slab_coverage >= 0.5`), `repeat_motion_mm_max < 3`, and
`nm_ref_{l,r}_sd < 0.4 x nm_ref_{l,r}_mean` (a reference partly outside the slab has CV near 1 and meaningless
CNR; this rejected 7 of 646 Philips scans in the September 2026 snapshot), `nm_clipped_fraction <= 0.01`, and every
`nm_valid_*` and `nm_support_*` ≥ 0.95 (the study's voxel checks; tables without these columns pass them), and
`n_repeats >= 3` (the study's acquisition check; 11 Oct 2026: four single-repeat slabs, three GE and one Siemens, passed
before; tables without the column pass).
`nm_clipped_fraction` is the largest fraction, over the raw repeats, of voxels at that repeat's maximum value.
- A clean slab reaches its maximum in a handful of voxels (about 3 × 10⁻⁷ of the slab).
- A scanner-clipped one saturates the brain. On a PPMI Siemens Prisma_fit scan, 37 % of voxels sat at the 12-bit ceiling of 4095, and snceg found 2 mm³ of nigra.
- Tables written before 26 September 2026 lack the column, and their rows pass this criterion.

The assembled table keeps `nm_*` columns ending `_cnr` (so not `_cnr_crus` / `_cnr_atlas`), plus `nm_vendor` and
`nm_slab_mm` (the slab's thin-axis extent, from `shape` × `voxel_mm`) on every row, QC-failed or not.

Tests: `tests/test_nm.py` (planted contrast, independent noise, band-offset refinement, reference excludes both
refined masks) and the hemisphere / no-wrap cases in `tests/test_imaging_audit.py`; refeature versioning and stale-atlas handling
in `tests/test_imaging_regressions.py`.

### Validation status (26 September 2026): not validated

Pre-registered gate: PD-vs-HC AUROC ≥ 0.70 with the bootstrap 95 % CI lower bound > 0.55, on all 45 QC-passing
healthy controls with NM-MRI and 45 randomly drawn PD patients.

| Measure | HC mean (SD) | PD mean (SD) | AUROC [95 % CI] | Cohen's d |
|---|---|---|---|---|
| `nmt_sn_mean_cnr` (template, crus reference) | 0.181 (0.048) | 0.167 (0.051) | 0.58 [0.46, 0.70] | 0.28 |
| `nmt_sn_post_mean_cnr` | 0.161 (0.037) | 0.144 (0.042) | 0.61 [0.49, 0.72] | 0.44 |
| `nm_sn_mean_cnr` (atlas ROI, ring reference, `nm.py` v5) | 0.102 (0.029) | 0.100 (0.025) | 0.54 [0.42, 0.67] | 0.08 |
| `nm_sn_posterior_mean_cnr` | 0.098 (0.026) | 0.092 (0.023) | 0.60 [0.48, 0.72] | 0.25 |

Neither method passes, so `manifest.NM_VALIDATED` is False and assembled NM columns carry `nm_validated = False`:
treat them as exploratory. Published single-site NM-MRI studies report larger effects; possible reasons here are
PPMI's multi-site 2D GRE-MT protocol, the affine atlas placement of `nm.py`, and a defect found in the template's crus
reference, which lies mostly above the SN (median 7 mm; about 10 % of it at SN levels). Fixing that mask is an
anatomical, outcome-blind change; its effect must then be tested on patients not in this sample. Fixed on 11 Oct 2026
(`template_masks` takes the crus on the SN prior's axial levels); not re-tested.

### Re-test with published nigral territories (26 September 2026): not validated

The re-test replaces PIE's self-derived masks with the published Biondetti et al. 2020 nigral territories and background ROI (`nm_template` stage `published`, below). Both tests were written down before any held-out value was compared.

**Sample.**
- PD: held-out patients, none from the 90 above. 120 were processed, all passing `QC["nm"]`.
- HC: the same 45, whose reuse is declared.
- Primary analysis: GE, Siemens and Toshiba only.
- After mask coverage: 106 PD and 39 HC.

**Gate.**
- AUROC ≥ 0.70;
- CI lower bound > 0.55;
- Spearman ρ ≥ 0.20 with the lowest putamen SBR.

| Test | Measure | AUROC [CI] | ρ with lowest putamen SBR | Result |
|---|---|---|---|---|
| 1 | Bilateral sensorimotor-territory contrast `nmb_sensorimotor_mean_cnr` (where Biondetti found the earliest loss); vendor-, age- and sex-adjusted z against leave-one-out HC | 0.62 [0.53, 0.72] | +0.18 (n = 133); within PD +0.01 | Fail |
| 2 | Eight territory × side contrasts: training-HC z, then L2 logistic regression, 5-fold CV × 20; 97.5 % CI for the second test | 0.67 [0.55, 0.78] | +0.20 (0.196); within PD +0.01 | Fail |

Secondary measures (test 1 design; AUROC, HC > PD):

| Measure | AUROC |
|---|---|
| Whole SN | 0.58 |
| Associative territory | 0.63 |
| Limbic territory (PD higher) | 0.44 |
| Sensorimotor, lower side | 0.59 |

**Replicated findings.** The spatial pattern held in the development sample (0.59 and 0.67) and in the held-out patients:
- PD have lower NM in the sensorimotor territory (0.079 vs 0.089);
- PD have relatively preserved or higher NM in the limbic territory (0.130 vs 0.122);
- the model's weights are negative on sensorimotor and positive on limbic.

This matches the lateral-to-medial gradient of nigral cell loss.

**Scanner vendor.** Separation differs strongly by vendor. Unadjusted sensorimotor AUROC:

| Vendor | AUROC | HC / PD |
|---|---|---|
| GE | 0.53 | 16 / 42 |
| Siemens | 0.70 | 21 / 69 |
| Toshiba | 0.97 | 6 / 5 |

The PD–HC difference is small next to between-subject and between-scanner spread. GE is 41 % of PIE's NM (264 of 649
error-free rows in the September 2026 table), and 189 GE slabs are 20 mm thick against the protocol's 24 mm. The
assembled `nm_vendor` and `nm_slab_mm` columns let an analysis restrict to Siemens or full coverage; no row is dropped
for vendor.

**Status.** NM columns stay exploratory (`manifest.NM_VALIDATED = False`). The held-out patients have now been used, so any further NM hypothesis, for example Siemens-only, needs new data.

**Changes after pre-registration, before any confirmatory value.**
- **Vendor restriction.** The primary analysis uses the vendors with at least 5 HC: GE 16, Siemens 21, Toshiba 6. Philips has 2 HC, too few to estimate its offset.
- **Bridge-QC reference.** The first mapping of the Biondetti atlas failed its check against CIT168 SNc + SNr: centroids were 4.1 and 3.7 mm away, and 61 background voxels fell inside.
  - NM-MRI shows the SNc, and the SNr lies ventrolateral to it.
  - An independent registration of the same atlas gave 1.0–1.4 mm to the SNc against 3.6–3.7 mm to SNc + SNr.
  - The check was therefore re-referenced to the SNc; the visual criterion was unchanged.
- **Test 2.** It was added at the user's request after the development sample gave 0.59 for test 1, and before unblinding.
- **Bug fix.** A bug that left the DaT correlation empty was fixed before unblinding: sessions were de-duplicated by image ID alone.

## Neuromelanin volume and normalised intensity (`nm.hyperintense_volume`, `nm.normalised_intensity`)

The contrast ratios above are computed on a fixed atlas mask, and deliberately exclude threshold
volumes: single-voxel order statistics are noisy, and bright arteries in the interpeduncular cistern
contaminate any threshold applied outside the nigra. Many studies nevertheless report a neuromelanin
"SN volume", either from manual segmentation (Droby et al. 2025, Ben Bashat et al.) or by thresholding
against a reference region. These three functions compute both kinds of number under explicit,
auditable rules, so PIE output can be set beside theirs.

```python
import nibabel as nib, numpy as np
from pie.imaging import nm

# synthetic phantom: 40 bright nigral voxels, 0.5 x 0.5 x 2.0 mm
rng = np.random.default_rng(0)
sig = rng.normal(100, 5, (30, 30, 10))
sn = np.zeros(sig.shape, bool); sn[10:14, 10:15, 4:6] = True
sig[sn] = 160
search = np.zeros(sig.shape, bool); search[8:16, 8:17, 3:7] = True   # atlas SN mask, slightly dilated
ref = np.zeros(sig.shape, bool); ref[20:28, 5:25, 2:8] = True        # reference region
img = lambda a: nib.Nifti1Image(a.astype(np.float32), np.diag([0.5, 0.5, 2.0, 1]))

v = nm.hyperintense_volume(img(sig), img(search), img(ref), k=3.0)
v["volume_mm3"], v["n_voxels"], round(v["threshold"], 1)   # 20.0, 40, 115.0 (reference mean + 3 SD)

i = nm.normalised_intensity(img(sig), img(sn), img(ref))
round(i["normalised_intensity"], 2), round(i["contrast_ratio"], 2)   # 1.6, 0.6

nm.mask_volume(img(sn))      # 20.0 mm^3: volume of a supplied mask, e.g. a manual segmentation
```

`search` decides the answer: pass the atlas SN mask, dilated at most by the registration uncertainty,
never a box, or cisternal arteries enter the count. `k` changes the volume several-fold — medians on
40 PPMI 2D GRE-MT scans were 160, 66 and 21 mm^3 at k = 2, 2.5 and 3 — so report the `k` with every
volume. All three images must be on one grid, and the reference must keep at least 20 voxels outside
the search region. On those 40 scans the per-side volume at k = 3 correlated with the contrast ratio
at Spearman 0.56, and `normalised_intensity` reproduced the whole-nigra contrast ratio exactly on the
same masks.

## Neuromelanin template pipeline (`nm_template.py`)

After Wengler et al. 2020 / Cassidy et al. 2019: instead of an atlas SN label and a ring next to it, masks are
drawn on a study NM template, so they sit on the neuromelanin band and the reference sits in the dark peduncle.
Every averaged slab goes into a 0.5 mm MNI midbrain box (`BOX_ORIGIN_RAS=(-30, -45, -30)`, `BOX_SHAPE=(120, 100,
80)`, `BOX_MM=0.5`) through the rigid slab → T1 transform and a cached ANTs SyN T1 → MNI warp.

| Stage | Function(s) | Writes |
|---|---|---|
| `syn` | `syn_cache(fastsurfer_dir)`: the study's registration (its `registration_stability.py`). The brain-masked T1 and the brain-masked MNI152NLin2009cAsym (`atlases.mni2009c_brain_1mm()`) are both put on a 2 mm grid, with intensities scaled to their 99.5th percentile. It uses ANTs SyN with a CC (radius 2) metric, affine iterations 1000/500/250/100 and deformable 40/20/10, seeded (`SYN_SEED`); about 8 CPU-minutes per subject. On 10 control scans it placed the NM masks at Dice 0.84 with the study's own, with 0.94 of their volume. `antsRegistrationSyNQuick[s]`, used before 27 September 2026, gave 0.72 and 1.23. `crop_warp` then crops the result | `<fastsurfer>/<IMAGE_ID>/mri/transforms/t1_to_MNI152NLin2009cAsym_syncc2mm_1Warp.nii.gz` (cropped to the box + 20 mm) and `…_0GenericAffine.mat`. Caches from before are never reused: `t1_to_mni_syn_*` was fitted to nilearn's 2009a template (before 25 September 2026), and `t1_to_MNI152NLin2009cAsym_syn_*` by SyNQuick |
| `normalize` | `normalize_subject(work_dir, patno, fastsurfer_dir)`, `slab_in_t1` | `<work>/<patno>/nm_mni.nii.gz`, `nm_normalize_log.csv` (`mni_nonzero_frac`) |
| `template` | `build_template(work_dir, patnos, min_frac=0.5)` → `(template, count, n)`; `template_masks(template, crus_mm=(4.0, 9.0), sn_mm=3.0, cnr_min=0.06)`; `save_template`; `template_figure` | `<work>/template/nm_template.nii.gz`, `nm_template_count.nii.gz`, `nm_template_masks.nii.gz` (1 sn_l, 2 sn_r, 3 crus_l, 4 crus_r), `template_info.txt`, `template_qc.png` |
| `features` | `load_masks(work_dir)`, `template_features(nm_mni, masks, prefix="nmt_")` | `<work>/nm_template_features.csv` |
| `published` | `published_masks()` (Biondetti territories + background on the box, via `atlases.biondetti_mni2009c(reference=box_path())`), `published_features(nm_mni, masks, prefix="nmb_", min_cov=0.9)` | `<work>/nm_published_features.csv` |

Template: each normalised slab divided by its median inside the SN prior (bundled CIT168 SNc + SNr, `sn_prior()`)
dilated 10 mm, averaged; voxels with data in fewer than `min_frac` of subjects are zero. Crus = darker half of the
sector 4-9 mm anterior-lateral to the prior, on the prior's own axial levels (since 11 Oct 2026; the 3D shell before
reached the darker peduncle above the SN, median 7 mm, so a through-slab gradient did not cancel). The crus *mode*
still weights those levels by voxel count and noise, so a gradient within the SN's own levels cancels only
approximately (the native snceg reference is a per-side mean on the SN's slices). SN = voxels within 3 mm of the prior
with template CNR >= max(0.06, Otsu). Per subject, CNR = I / mode(crus) − 1. No threshold volume is reported (noise-driven).

```bash
for stage in syn normalize template features; do
  venv_imaging/bin/python -m pie.imaging.nm_template $stage --sessions <derived>/sessions.csv \
      --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/nm --workers 4; done
```

| Flag | Default | Meaning |
|---|---|---|
| `stage` | required | `syn`, `normalize`, `template` or `features`, in that order |
| `--sessions`, `--fastsurfer-dir` | required | As for `nm` |
| `--work-dir` | required | The `pie.imaging.nm` work dir (reads `nm_features.csv`; `normalize` needs `--keep-nifti` slabs) |
| `--workers` | 4 | Processes |
| `--patnos`, `--limit` | none | Restrict / cap subjects |
| `--pid-file` | none | Write the PID |

Subjects are the error-free rows of `nm_features.csv` that have a FastSurfer subject. `syn`, `normalize` resume;
`template` and `features` recompute. Helper files `mni152_brain_1mm.nii.gz` and `mni_midbrain_box_0.5mm.nii.gz`
are written under `~/nilearn_data/`.

Columns: `nmt_sn_{l,r}_cnr`, `nmt_sn_{post,ant,med,lat}_{l,r}_cnr`, `nmt_{sn,sn_post,sn_ant,sn_med,sn_lat}_mean_cnr`,
`nmt_sn_min_cnr`, `nmt_sn_asym_cnr`; QC `nmt_crus_mode_{l,r}`, `nmt_crus_cv_{l,r}`, `nmt_sn_cov_{l,r}` (fraction
of the SN mask with data); `patno`, `error`. `manifest.assemble_features` reads this table from the NM directory and
keeps the `nmt_*_cnr` columns, blanked where `QC["nmt"]` fails (SN coverage >= 0.9 per side, crus CV < 0.3).
Tests: `tests/test_nm_template.py`, and the assembly case in `tests/test_imaging_regressions.py`.

### Published nigral territories (`published` stage)

The `published` stage needs no study template and no data-driven mask. It reads the neuromelanin contrast in the nigral territories that Biondetti et al. published (Brain 2020;143:2757, doi:10.1093/brain/awaa216).

**Territories and reference.**
- The territories are associative, limbic and sensorimotor.
- The reference is the background ROI (BND) the authors used for SNR = 100 × mean(SN) / mean(BND). BND has three parts, in the crus cerebri and the midline tegmentum.
- Per side (MNI x < 0 = left), contrast = mean(region) / mean(BND) − 1, i.e. SNR / 100 − 1.

**Source files.**
- Downloaded at run time from github.com/emmabiondetti/substantia-nigra-neuromelanin at commit `e34cbd5`.
- Checked against pinned sha256 values (`atlases.BIONDETTI_FILES`).
- The repository declares no licence, so PIE never bundles the files.

**Mapping to MNI.** The authors' brain template is registered once to the TemplateFlow MNI152NLin2009cAsym brain:
- antsRegistrationSyN[s];
- about 15 minutes;
- cached under `~/.cache/pie/biondetti/to_MNI152NLin2009cAsym/` together with `qc.json`.

The labels are then pulled directly onto the 0.5 mm box. Where a territory and BND touch, the territory wins.

**Bridge QC** (`atlases.nigral_bridge_qc`). This is an outcome-blind gate, and the bridge raises an error if it fails.
- Criteria:
  - per side, the SN centroid lies within 4 mm of the CIT168 SNc centroid;
  - no BND voxel lies inside the SNc.
- On 26 September 2026 the result was 1.7 mm (left) and 1.2 mm (right), with no BND in the SNc.
  - The SN measured 546 + 544 mm³, and the study's independent registration gave 546 + 539 mm³.
- On the pooled NM template, the SN is brighter than a 1.5 mm shell around it at every level:
  - SN: +3.3 %, +3.3 %, +1.7 % from inferior to superior;
  - sensorimotor territory: +6.0 %, +6.5 %, +1.8 %.
- The uppermost sensorimotor levels, z −12 to −8 mm, have the weakest contrast.

**Coverage and missing values.**
- A region is missing when less than 90 % of its voxels have data.
- The reference is missing when any of BND's three parts is below 90 %; this is the study's rule.
- Bilateral means need both sides.

**Columns.**
- `nmb_{sn,associative,limbic,sensorimotor}_{l,r}_cnr`
- `…_mean_cnr`
- `…_min_cnr`
- coverage: `nmb_*_cov_{l,r}`, `nmb_bnd_cov`, `nmb_bnd_min_part_cov`
- `nmb_bnd_mean`

`assemble_features` keeps the `nmb_*_cnr` columns. It blanks them, like `nmt_*`, wherever the slab fails `QC["nm"]`.

## Native-space published measures (`nm_native.py`)

Every measure is read on the subject's own averaged slab (`nm_mean.nii.gz`). The slab itself is never resampled.
- **Region labels.** They are defined once in MNI152NLin2009cAsym (`region_labels`) and pulled onto the slab grid in one nearest-label resampling (`pull_labels`).
- **Transform chain.** Slab → T1 is the inverse of the rigid `slab_to_t1.tfm` from `nm.register_slab`. T1 → MNI is the inverse of the cached `nm_template syn` warp.
- **Inverse warp.** `inverse_warp` inverts the cropped forward SyN displacement field with ANTs (`invert_displacement_field`) and caches it beside it (`*1InverseWarp*`). A synthetic test checks this against ANTs' own inverse (Dice > 0.95).
- **Region codes.** Each label is region + 100 × side (1 left, 2 right). The regions are:
  - Biondetti territories 1–3;
  - `RING = 5`, the 1 mm search ring around the nigra;
  - `CRUS = 21`, the lateral parts of the Biondetti background ROI;
  - `TEGMENTUM = 23`, its midline part.
- **Coverage.** The share of each region the slab contains comes from the same pull onto the slab grid padded along its normal. A side whose search region or crus is less than 90 % inside the slab is missing.

| Measure | Function | Columns |
|---|---|---|
| **Langley SNc volume** [Langley 2025; Hwang 2023]: search region (nigra + 1 mm) voxels above the crus mean + 2.8 SD (`nm.hyperintense_volume`); run `nm --denoise` first. That is MP-PCA over the repeats before averaging [Veraart 2016], as published, on the slab's central in-plane half, which equals whole-slab MP-PCA in the midbrain at about 2 CPU-min per subject. On a phantom it cut the error of the average about threefold in uniform tissue, by about 20 % around a nigra-like block, and kept that block's contrast within 10 % | `langley_features(nm_img, codes, k=2.8)` | `nml_sn_volume_mm3`, `nml_sn_volume_{l,r}_mm3`, `nml_threshold`, `nml_ref_mean`, `nml_ref_sd`, `nml_n_ref`, `nml_k` |
| **snceg** [Lillebostad 2025], **PIE's default NM measure**: model output on the native slab. Each side's reference is that side's crus on the slices holding that side's segmented SN (the slab's thin axis), excluding segmented voxels, ≥ 20 voxels; a through-slab intensity gradient then cancels. Before 11 Oct 2026 one crus over all slices and both sides was used; it is kept as `*_cr_allslices` and `nms_crus_mean` / `nms_crus_sd` (diagnostics, not assembled) | `snceg_mask(nm_path, out_path)`, `snceg_features(nm_img, sn, codes)` | `nms_sn_volume_mm3`, `nms_sn_volume_{l,r}_mm3`, `nms_sn_{l,r,mean}_cr` (I_SN / I_crus − 1), `nms_sn_{l,r,mean}_cnr` ((I_SN − I_crus) / SD_crus), `nms_crus_{l,r}_mean`, `nms_crus_{l,r}_sd`, `nms_sn_{l,r,mean}_cr_allslices`, `nms_crus_mean`, `nms_crus_sd` |
| QC | `process_subject` | `native_cov_search_{l,r}`, `native_cov_crus_{l,r}` |

**Departures from the publications.**
- Langley's own SNc atlas and cerebral-peduncle ROI are not public. PIE uses the published Biondetti nigral mask, dilated 1 mm as Langley dilate theirs, and the crus parts of Biondetti's background ROI. Absolute volumes can therefore differ from theirs; the effect size is the reproduction target.
- Registration uses ANTs SyN to MNI152NLin2009cAsym with a rigid MI slab-to-T1 step. Langley used FSL FNIRT and boundary-based registration.
- For its GRE analysis, Lillebostad's crus reference is the same Biondetti background ROI.

**snceg setup.**
- Weights: fetched from `huggingface.co/lillepeder/SNceg-0.1` at revision `6bcddc5f`, sha256-pinned (`SNCEG_FILES`), into `~/.cache/pie/snceg/`.
- Model: a pickled fastMONAI learner that needs torch 2.0.1, fastMONAI 0.4.0.2 and numpy 1.26. PIE's own environment cannot hold those, so `snceg_runner.py` (adapted from `snceg.py`, MIT) runs in a separate Python 3.11 environment:

```bash
/usr/bin/python3.11 -m venv third_party/snceg_venv
third_party/snceg_venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 fastMONAI==0.4.0.2 fastai==2.7.12 monai==1.2.0 torchio==0.18.91 \
    huggingface-hub==0.23.4 numpy==1.26.0
export PIE_SNCEG_PYTHON=$PWD/third_party/snceg_venv/bin/python     # the default path
```

A subject takes about 8 s on CPU. The mask must come back on the input grid, or the call raises.

```bash
venv_imaging/bin/python -m pie.imaging.nm --zips … --sessions … --fastsurfer-dir … --work-dir <w> --keep-nifti --denoise
venv_imaging/bin/python -m pie.imaging.nm_template syn --sessions … --fastsurfer-dir … --work-dir <w>
venv_imaging/bin/python -m pie.imaging.nm_native --sessions … --fastsurfer-dir … --work-dir <w> --workers 6 --snceg
```

**Other inputs.**
- **Dual-echo TSE:** `nm_native --prepare-tse --zips … --echo 1|2` converts the session's TSE, keeps the chosen echo as `nm_mean.nii.gz` and registers it rigidly to the T1 that `--sessions` gives for the participant (`batch.fastsurfer_by_patno`: the earliest finished session).
  - To register a later-visit TSE to the T1 from its own day, pass a sessions table holding that T1 only. The Lillebostad reproduction does this (earliest TSE-FS session, same-day T1).
- **NIfTI instead of LONI zips:** `nm.process_subject` reads a series row that carries `"nifti"` as is. A 4D file counts as its repeats.
  - This is how the Emory release of Hwang et al. 2023 (7 repeats per scan) goes through the same `--denoise` → `syn` → `nm_native` chain.

**The study's measure (`--study`, 27 Sep 2026).** This is the neuromelanin quantity of the Study 1 manuscript (Methods 4.13), whose regional family passed that study's PD-vs-HC control (AUROC 0.756, 148 PD / 41 HC).

Regions:
- `study_labels()`: the Biondetti territories + 100 × side, and the *complete* published background ROI as its three parts (`STUDY_REF + 1..3`: left crus, midline, right crus).
- Unlike `region_labels`, no ring is cut from the background.
- The regions are pulled onto the slab in one resampling and saved as `study_regions_nm.nii.gz`.

Measure:
- `study_features(nm, codes, support, cov, acquisition_ok)` ports the study's `measure()`.
- For the whole nigra and each territory, each side's contrast ratio against the background is mean / mean − 1: `nm_bion_{whole,associative,limbic,sensorimotor}_{l,r,mean}_cr`.

A region's contrasts are NaN on both sides unless every one of these holds:
- ≥ 20 valid voxels per side and in the reference;
- ≥ 95 % of them with signal and reached by every repeat (support ≥ 0.99);
- reference CV < 0.4;
- ≥ 90 % slab coverage of each side and of each background part;
- an acceptable acquisition: ≥ 3 identified repeats, repeat motion and nigral RMS displacement (`repeat_nigral_rms`) < 3 mm.

`nm_bion_qc_pass` and `nm_bion_qc_failed_checks` come from the whole-nigra measure. `syn_numerical_qc` adds the study's transform checks on the SyN cache, and a failure blanks every contrast:
- the MNI → T1 → MNI round trip ≤ 0.5 mm at the nigral points (`syn_roundtrip_mm_max`);
- a positive Jacobian there (`syn_jacobian_min`);
- a positive affine determinant.

Run `nm --keep-nifti` *without* `--denoise` for the study's quantity, then `nm_native … --study`.

`manifest.assemble_features` keeps `nml_sn_volume*_mm3`, `nms_sn*_{mm3,cr,cnr}` and `nm_bion_*_cr`, blanked where the slab fails `QC["nm"]`, like `nmt_*` and `nmb_*`, and copies the default measure to `nm_primary_cr` (`nms_sn_mean_cr`) and `nm_primary_volume_mm3` (`nms_sn_volume_mm3`). Tests are in `tests/test_nm_native.py`:
- a real ANTs SyN between synthetic volumes;
- an oblique thin slab;
- coverage;
- region codes;
- the threshold volume per side;
- snceg features;
- the snceg environment contract;
- the study's labels, measure and checks (including each failure mode, and one-sided territories blanked on both sides);
- nigral repeat motion;
- the SyN numerical QC (a real SyN passes, a reflected affine fails);
- `study_subject` end to end on a synthetic subject.

### References (neuromelanin)

- Biondetti E, Gaurav R, Yahia-Cherif L, et al. Spatiotemporal changes in substantia nigra neuromelanin content in Parkinson's disease. *Brain* 2020;143:2757–2770. doi:10.1093/brain/awaa216. Atlas: github.com/emmabiondetti/substantia-nigra-neuromelanin.
- Cassidy CM, Zucca FA, Girgis RR, et al. Neuromelanin-sensitive MRI as a noninvasive proxy measure of dopamine function in the human brain. *PNAS* 2019;116:5108–5117. doi:10.1073/pnas.1807983116.
- Cho SJ, Bae YJ, Kim JM, et al. Diagnostic performance of neuromelanin-sensitive magnetic resonance imaging for patients with Parkinson's disease and factor analysis for its heterogeneity: a systematic review and meta-analysis. *Eur Radiol* 2021;31:1268–1280. doi:10.1007/s00330-020-07240-7.
- Hwang KS, Langley J, Tripathi R, et al. In vivo detection of substantia nigra and locus coeruleus volume loss in Parkinson's disease using neuromelanin-sensitive MRI: replication in two cohorts. *PLOS ONE* 2023;18:e0282684. doi:10.1371/journal.pone.0282684. Data: Dryad doi:10.6086/D1709N (CC0).
- Langley J, Huddleston DE, Liu CJ, Hu X. Reproducibility of locus coeruleus and substantia nigra imaging with neuromelanin sensitive MRI. *MAGMA* 2017;30:121–125. doi:10.1007/s10334-016-0590-z.
- Langley J, Hwang KS, Huddleston DE, Hu XP, PPMI. Nigral volume loss in prodromal, early, and moderate Parkinson's disease. *npj Parkinson's Disease* 2025;11:181. doi:10.1038/s41531-025-00976-3.
- Lillebostad PAG, Njølstad TH, Hogstad S, et al. Deep-learning segmentation of the substantia nigra from multiparametric MRI: application to Parkinson's disease. *Imaging Neuroscience* 2025;3:IMAG.a.158. doi:10.1162/imag.a.158. Code: github.com/lillepeder/snceg (MIT).
- Sasaki M, Shibata E, Tohyama K, et al. Neuromelanin magnetic resonance imaging of locus ceruleus and substantia nigra in Parkinson's disease. *NeuroReport* 2006;17:1215–1218. doi:10.1097/01.wnr.0000227984.84927.a7.
- Trujillo P, Aumann MA, Claassen DO. Neuromelanin-sensitive MRI as a promising biomarker of catecholamine function. *Brain* 2024;147:337–351. doi:10.1093/brain/awad300.
- Veraart J, Fieremans E, Novikov DS. Diffusion MRI noise mapping using random matrix theory. *Magn Reson Med* 2016;76:1582–1593. doi:10.1002/mrm.26059.
- Wengler K, He X, Abi-Dargham A, Horga G. Reproducibility assessment of neuromelanin-sensitive magnetic resonance imaging protocols for region-of-interest and voxelwise analyses. *NeuroImage* 2020;208:116457. doi:10.1016/j.neuroimage.2019.116457.
- Wengler K, Cassidy C, van der Pluijm M, et al. Cross-scanner harmonization of neuromelanin-sensitive MRI for multisite studies. *J Magn Reson Imaging* 2021;54:1189–1199. doi:10.1002/jmri.27679.

## DaTscan SPECT (`datscan.py`)

PPMI's SPECT download holds the **raw tomographic projections** (NM DICOM, `ImageType` TOMO/EMISSION, 60-480
frames = detectors x energy windows x angles), not reconstructed volumes, and PPMI releases SBRs only for some
cohorts. `datscan.py` reproduces the SBR chain with open components and the subject's own FastSurfer
segmentation as the ROI atlas.

1. **`read_projections(dcm)`** (path, bytes or dataset) — keeps the energy window containing 159 keV
   (`_photopeak_window`; windows stored in 1/100 keV are rescaled), assigns an angle to every frame from the NM
   vectors and `RotationInformationSequence`, sums detectors and merges frames at the same angle. Conventions,
   established against PPMI's SBRs and the striatum position across 19 scanner configurations:
   DICOM angles run opposite to scikit-image's (`angle = -start + direction * step * (view - 1)`); dual-head systems
   without per-detector start angles are H-mode (heads 180° apart); no vendor needs a left/right mirror. Broken
   headers (no angular step) are inferred from the frame count. Raises `ValueError` for non-TOMO series. Returns
   `proj`, `angles_deg`, `spacing_mm`, `meta` (the `hdr_*` columns). No PPMI scan is stored reconstructed, but
   consortium sites send such stacks: `read_volume(dcm)` places them by their DICOM geometry
   (`ImageOrientationPatient` / `ImagePositionPatient` at top level or in the first `DetectorInformationSequence`
   item, `PixelSpacing`, signed `SpacingBetweenSlices`), resamples them onto an axis-aligned RAS grid and returns
   `(NIfTI, spacing_mm)`. A stack without that geometry is refused (`ValueError`); until October 2026 this path
   guessed the orientation with the projection flips, which can mirror left and right.
2. **`reconstruct(proj, angles_deg, spacing_mm, fwhm_mm=6.0, filter_name="hann", attenuation=False,
   point_sources=True)`** — filtered back-projection per transaxial slice (scikit-image `iradon`);
   `subtract_point_sources(vol, angles_deg, spacing_mm, filter_name="hann", factor=6.0, max_voxels=200)` removes
   external fiducial markers, whose coherent back-projection otherwise streaks through the volume and flattens
   every SBR; optional `chang_correction(vol, spacing_mm, mu_per_cm=0.11, n_dirs=36, threshold=0.15)`; Gaussian to
   `fwhm_mm`. `attenuation` defaults to off here, as in `process_series` and
   the CLI (until September 2026 this function alone defaulted to on), so a direct call reproduces the pipeline.
   `to_nifti(vol, spacing_mm)` writes patient axes (+y anterior, +z superior), centred at the origin.
3. **`register_to_t1(spect_img, t1_img, flip_lr=False, rz=0.0, search=False, aparc_img=None,
   scale_fit=False)`** — normalised correlation against the brain-only `synthetic_spect(t1_img, aparc_img,
   fwhm_mm=10.0)` (striatum 1.0, rest of brain 0.25, nothing outside the brain, smoothed), sampled only inside the
   brain and a ~7 mm rim, with the SPECT cleaned by `_clean_spect` (largest component of the brightest 4 L of a
   20 mm-smoothed copy, winsorised). Centre-of-moments initialisation, 3-level refinement, `ScaleVersor3DTransform`:
   rigid for parallel-hole cameras, rigid + per-axis scale when `scale_fit` (fan-beam collimators, or no
   collimator/zoom recorded: Marconi and Picker Prism data reconstructed as parallel-beam come out magnified
   transaxially). Scales are bounded by `SCALE_BOUNDS`: 0.85–1.8 transaxially (a fan-beam collimator magnifies by
   F / (F − R), focal length F, radius of rotation R) and 0.85–1.15 axially (the fan converges in the transaxial
   plane only). A fit that leaves them is clamped, refitted rigidly at the bound and flagged (`reg_scale_at_bound`).
   Returns `(transform, metric, moving, {"t0", "scale_at_bound"})`. A striatum-masked second stage was tried and
   removed: it drifted on faint striata (side-wise putamen agreement with PPMI fell from 0.84 to 0.78 pooled, 0.64 to
   0.27 on ADAC).

   **Why it changed (October 2026).** On the 8 September 2026 run the placement search below hit its ±2-voxel edge
   in 81 % of scans, and in 91 % of the lowest putamen-SBR tertile. Putamen agreement with Invicro was r 0.925 inside
   the window and 0.77 on its edge, so the search was correcting real misregistration. The template also held scalp
   and neck (0.12 from the T1 intensity). Its smoothing came out near 30 mm FWHM, not 10, because the voxel size was
   read off the diagonal of FastSurfer's LIA affine, which gives 1/3 mm. The unbounded fan-beam scales had an axial
   median of 0.72, and within each collimator group agreement with Xing fell as the axial scale moved away from 1.
   Putamen r went from 0.96/0.95 to 0.93/0.90 for FANB, and from 0.87/0.81 to 0.74/0.62 where no collimator was
   recorded, comparing the half of scans nearest 1 with the other half. Fits rotated by more than 30° (19 % of scans)
   agreed at r 0.49–0.72, against 0.83–0.85 for the rest.

   On a synthetic head with scalp uptake beyond the skull and a hot neck, the old template reproduced the axial
   shrinkage (0.73). The brain-only fit gave 0.91, and 1.00–1.02 without the hot neck, and it recovered a 1.2 or
   1.4 transaxial magnification to within 1 %.

   **On PPMI (11 Oct 2026).** Against Xing, on a tuning set of 240 random scans outside the validation set, the
   brain-only fit raised the lowest-putamen r from 0.78 to 0.90 and caudate from 0.55 to 0.81. Its correlation bar
   (0.2, the 2nd percentile) was set on that set. On the vendor-stratified validation set it failed on 49 of 237
   scans: HERMES, BrightView, Forte and P3000XP, where the head outline still helps. **`quantify_with_fallback`**
   (what `process_series` runs) therefore re-registers those with **`LEGACY`**: the pre-October head-outline template
   at its effective ~30 mm, correlation over the whole image, ±2-voxel placement. The legacy fit is kept when its own
   correlation reaches 0.4; rows carry `reg_template` (`brain` / `legacy_head`) and `reg_fallback`.

   Results:
   - QC passed 236 of 237 (old pipeline 233).
   - On the 156 scans both pass, putamen r rose from 0.796 to 0.861 and caudate from 0.675 to 0.743.
   - Restricted to brain-only fits, putamen CCC rose from 0.79 to 0.87–0.89.
   - **Full cohort** (`datscan_2026_10`, 1,459 scans): QC passed 1,452. The table below compares it with the September
     run.

   | Against | Scans | Putamen-min r (raw) | Putamen-min CCC | Caudate r (L / R) | Deficit κ (all / prodromal) |
   |---|---|---|---|---|---|
   | Xing, September run | 1,160 | 0.788 | 0.81 | 0.61 / 0.63 | 0.75 / 0.54 |
   | Xing, October run | 1,161 | 0.851 | 0.87 | 0.74 / 0.76 | 0.80 / 0.65 |
   | Invicro, October run | 882 | 0.887 | 0.89 | 0.78 / 0.79 | 0.85 / – |

     The 14 % of scans whose best placement lies beyond the old ±2-voxel window still agree less well with Xing:
     putamen-min r 0.57, against 0.92 for the rest (0.72 against Invicro). These are the scans the old window clipped.
     `manifest.DEFAULT_DIRS["dat"]` now points at this run.

   `manifest.QC["dat"]` applies the legacy rule (correlation ≥ 0.4) to legacy rows. Brain-only rows need correlation
   ≥ 0.2, rotation ≤ 30°, translation ≤ 50 mm and a placement off the window edge. A clamped scale is only flagged:
   those fits agreed at r 0.75, as well as the old pipeline did.
4. **`quantify(spect_img, t1_img, aparc_img, flip_lr=False, rz=0.0, search=False, dilate=0, ref_dilate=0,
   search_vox=None, scale_fit=False)`** → `sbr_with_transform` → **`sbr_from_arrays(S, L, dilate=0,
   ref_dilate=0, search_vox=None, ap=None, spacing_mm=None)`**: DKT labels resampled onto the SPECT grid (through
   the stored, rounded transform, so `--requantify` reproduces a run exactly), a ±`SEARCH_MM` (12 mm; `search_vox`
   overrides it in voxels) translation search maximising striatal counts (mimics hottest-region ROI placement;
   `shift_edge` flags placements that reach the window's edge), SBR = target / occipital
   − 1 with occipital = cuneus, lateral occipital, lingual, pericalcarine (`OCCIPITAL`). Each striatal ROI is also
   split into anterior/posterior halves along the T1 anterior axis (the posterior putamen loses dopamine
   transporter first), and `sbrwm_*` uses cerebral WM >= `WM_MARGIN_MM` (15 mm) from the striatum as reference
   (after the MJFF Research Community DaT pipeline, which references the superior longitudinal fasciculus).
   Minimum voxels: 20 per striatal ROI (10 per half), 50 per reference; otherwise NaN.

**PPMI's own SBRs.** Since 1 December 2024 PPMI's primary SBRs come from the XingImaging core lab, which re-analysed
every earlier scan (`Xing_Core_Lab_-_Quant_SBR_*.csv`; README_SPECT_Quantitative_Analysis_Results, 2025): HERMES HOSEM
reconstruction without attenuation correction or filter, then in MIAKAT a zero-order Chang correction with
scanner-specific μ, a 6 mm Gaussian, a 12-parameter affine to a DaT template in MNI152 and CIC-atlas regions with the
**cerebral white matter** as reference (striatum, caudate, putamen and pre/post-commissural, dorsal/ventral
sub-regions). The Invicro occipital-reference table (`DaTScan_SBR_Analysis_*.csv`) is archived and gets no scans after
that date. `labels.dat_sbr_table` reads the Xing table first; PIE's `sbrwm_*` columns are the ones on its reference.

Agreement of PIE's own SBRs with PPMI's (study run of 8 September 2026, QC-passing scans matched on month):

| PIE | PPMI | Scans | Putamen r (L / R) | Caudate r (L / R) |
|---|---|---|---|---|
| `sbrwm_*` (distant cerebral WM) | Xing, cerebral WM | 1,160 | 0.79 / 0.79 (95 % CI 0.76–0.82) | 0.61 / 0.63 |
| `sbr_*` (occipital) | Invicro, occipital | 880 | 0.81 / 0.81 | 0.66 / 0.65 |
| `sbrwm_putamen_*_post` | Xing post-commissural putamen | 1,160 | 0.79 / 0.78 | |

For scale, PPMI's two core labs agree with each other at r = 0.88 on the putamen (2,963 scans). Lowest-putamen SBR
separated PD from controls with AUROC 0.959 (PIE, WM reference; 474 PD / 126 HC) against 0.990 for Xing on the same
scans, and 0.946 against 0.992 for Invicro (654 / 176). The caudate is PIE's weak region. Use PPMI's SBRs as the label
and reference standard wherever they exist; PIE's reconstruction is for scans PPMI has not quantified and for
analyses that need the subject's own anatomy. Never apply PPMI's percentage cut-offs to PIE SBRs.

The placement search earns its place: re-quantifying 300 of those scans from the saved reconstructions and
transforms with fixed placement (`search_vox=0`) lowered agreement with Xing from r 0.80 to 0.70–0.72 on the putamen
and from 0.60–0.64 to 0.51–0.55 on the caudate, because the search absorbs residual registration error. It also raises
every SBR (putamen by ~0.15 on the WM scale), part of which is the upward bias of taking a maximum over noisy positions,
largest on low-count scans. With perfect registration a noisy synthetic phantom moves it by under half a voxel on
average. Since October 2026 the window is ±12 mm (it was ±2 voxels, 6–10 mm by camera, and 81 % of scans hit its
edge), and a placement on the edge fails QC.

No attenuation correction by default: a Chang implementation exists (`--attenuation`) but lowered agreement with
PPMI's values. Absolute SBRs therefore sit below PPMI's. `datscan.calibrate(d, {pie_column: published_column},
min_ref=15, by="hdr_model", method="deming", delta=1.0)` maps them onto a published scale.
- It fits per camera model on the scans that have both values. A model with fewer than 15 such scans, or whose values correlate with the published ones below r 0.3 (`min_r`), uses all models pooled. Deming's slope divides by the covariance, so an uncorrelated group gets an arbitrary slope: in October 2026, Forte left caudate against Invicro (r 0.02–0.16 on 15 scans per fold) got slope 22, which pulled the region's CCC down to 0.52 (0.80 with the guard).
- `method="deming"` is a linear map with errors in both variables (`datscan.deming(x, y, delta)`; `delta` is the y/x error-variance ratio). `method="equipercentile"` sends each value to the published value at the same percentile, so the mapped reference scans reproduce the published distribution exactly; values outside the reference range go to its ends.
- Both keep the published spread. The OLS fit used until October 2026, which the pipeline never called, regressed towards the mean: the mapped putamen SD was 0.82 of the published SD.
- The map is applied to every scan, so prodromal scans without published SBRs are calibrated too.
- Match the reference regions: `sbrwm_*` to Xing's `*_REF_CWM`, and `sbr_*` to Invicro's occipital-referenced SBRs.

`reg_params` / `reg_center` store the fitted fixed → moving transform so ROI variants can be recomputed without
re-registering: `transform_from_row(row)`, `requantify_row(row, fastsurfer_subject_dir)`.

```python
import numpy as np
from pie.imaging import datscan

lab = np.zeros((60, 60, 60), np.int16)
lab[20:26, 26:34, 28:34] = datscan.PUTAMEN_L
lab[34:40, 26:34, 28:34] = datscan.PUTAMEN_R
lab[24:36, 40:48, 28:34] = datscan.OCCIPITAL[0]
counts = np.where(lab > 0, 1.0, 0.0)
counts[lab == datscan.PUTAMEN_L], counts[lab == datscan.PUTAMEN_R] = 3.0, 1.5
out = datscan.sbr_from_arrays(counts, lab, search_vox=0)
out["sbr_putamen_l"], out["sbr_putamen_r"]            # (2.0, 0.5)
```

### SPECT index (input; not built by PIE)

The CLI reads a CSV the caller builds from the SPECT zips (for example by listing members and probing
`ImageType` / `NumberOfFrames` with pydicom):

| Column | Meaning |
|---|---|
| `zip` | Zip path (if missing on disk, `--zip-dir/<basename>` is tried) |
| `member` | Full DICOM member path, `PPMI/<PATNO>/<desc>/<date>/<IMAGE_ID>/<file>.dcm`; `process_series` takes PATNO, description and image ID from it, and `manifest` takes the date from part 4 |
| `patno`, `image_id` | Identifiers |
| `kind` | Only rows equal to `TOMO` are processed |
| `frames` | Frame count; the largest TOMO series per subject is used |

Split-by-energy-window Philips series (several members per image ID) keep the member whose window contains the
photopeak (`_prefer_photopeak_member`). `manifest.build_manifest` also reads this file as
`<derived>/spect_index.csv` for `dat_date`.

### CLI

```bash
venv_imaging/bin/python -m pie.imaging.datscan --index <derived>/spect_index.csv \
    --sessions <derived>/sessions.csv --fastsurfer-dir <derived>/fastsurfer \
    --out-dir <derived>/datscan_2026_10 --workers 8      # -> datscan_sbr.csv + nifti/<IMAGE_ID>_datscan.nii.gz
```

`manifest` reads `<derived>/datscan_2026_10/datscan_sbr.csv` by default (the September run is `datscan_full`); for another `--out-dir` pass
`modality_dirs={"dat": <out-dir>}` to `build_manifest` / `assemble_features`.

| Flag | Default | Meaning |
|---|---|---|
| `--index` | required | SPECT index CSV (above) |
| `--zip-dir` | `Imaging` | Directory holding the zips when the index path does not exist |
| `--sessions` | required | `sessions.csv`; the earliest session with finished FastSurfer stats is the anatomy (`batch.fastsurfer_by_patno`, the rule every modality uses; before September 2026 this CLI took the latest) |
| `--fastsurfer-dir` | required | FastSurfer subjects dir |
| `--out-dir` | required | Output dir |
| `--workers` | 4 | Series in parallel |
| `--limit` | none | At most N series (with `--requantify`: writes `datscan_sbr_requant_sample.csv` instead of replacing the table) |
| `--patnos` | none | Text file of PATNOs |
| `--flip-lr` | `false` | `true` mirrors the reconstruction left/right (no vendor needed it) |
| `--attenuation` | off | Chang attenuation correction (experimental; file suffix `_ac`) |
| `--requantify` | off | Recompute SBR columns of finished rows from the saved NIfTI and stored transform (`requantify_row`, ~2 s/scan) against the row's `fs_image_id` (older rows: the earliest finished FastSurfer session), with the current placement window; the previous table is kept once as `datscan_sbr.pre_requant.csv`. It does not re-register: rows from before October 2026 keep their old transforms, so re-run them instead |

Resumable: image IDs already in `datscan_sbr.csv` are skipped. Subjects without a FastSurfer T1 are reconstructed
but not quantified. Rows are appended through `batch.run_batch`, so the header widens to every column seen even when
the first series to finish fails.

### Output columns (`datscan_sbr.csv`)

| Column | Meaning |
|---|---|
| `sbr_{caudate,putamen}_{l,r}` | SBR vs occipital |
| `sbr_{caudate,putamen}_{l,r}_{ant,post}` | Anterior / posterior halves |
| `sbrwm_*` | Same ROIs vs distant cerebral WM |
| `mean_<roi>`, `mean_occipital`, `mean_wm` | ROI mean counts |
| `reg_metric` | Negative normalised correlation (more negative is better) |
| `reg_scale_{x,y,z}`, `reg_params`, `reg_center`, `flip_lr` | Transform (ScaleVersor3D: versor xyz, translation xyz, scale xyz) |
| `reg_rot_deg`, `reg_rot_{x,y,z}_deg`, `reg_t{x,y,z}_mm`, `reg_scale_at_bound` | Total rotation and xyz Euler angles about the T1's LPS axes, translation beyond the centre-of-mass start (LPS mm), scale clamped at `SCALE_BOUNDS` |
| `shift_vox`, `shift_{x,y,z}_mm`, `shift_edge` | Hottest-placement shift (norm in voxels; per axis in mm, right / anterior / superior) and whether it reached the window's edge |
| `n_label_voxels`, `n_ref_voxels`, `n_wm_voxels`, `point_source_voxels` | QC |
| `hdr_manufacturer`, `hdr_model`, `hdr_collimator`, `hdr_zoom`, `hdr_scale_fit`, `hdr_n_frames`, `hdr_n_detectors`, `hdr_n_angles`, `hdr_angular_step`, `hdr_scan_arc`, `hdr_rotation_direction`, `hdr_start_angle`, `hdr_energy_window`, `hdr_rows`, `hdr_cols`, `hdr_spacing_mm`, `hdr_counts_total` | Acquisition header |
| `image_id`, `patno`, `series_desc`, `kind` (`projections` / `stored_volume`), `nifti`, `fs_image_id`, `error` | Lineage |

`manifest.QC["dat"]`: `|reg_metric| >= 0.4`, `n_label_voxels >= 100`, `reg_rot_deg <= 30`, translation beyond the
centre-of-mass start <= 50 mm, no clamped scale and no placement on the window's edge. Tables written before October
2026 lack the last four columns and pass those gates, so re-run them. The correlation is now computed inside the brain
only, so its values are not comparable with older tables: check the 0.4 bar on the re-run. The manifest's
`dat_batch` is vendor + first 12 characters of `hdr_model`; assembled features keep `dat_sbr_{caudate,putamen}_{l,r}`.

### Validation

`scripts/validate_datscan.py --sbr <derived>/datscan_2026_10/datscan_sbr.csv --index <derived>/spect_index.csv
--ppmi-dir PPMI [--source xing|invicro] [--method deming|equipercentile] [--folds 5] [--threshold 0.65] [--patnos
file]` (`--patnos` compares two runs on the same participants) matches
QC-passing scans to the core lab on PATNO and month and calibrates per camera model on the other folds (split by
PATNO). Per region it prints r of the raw values (overall, and inside versus at or past the old ±2-voxel edge), and
CCC, ICC(2,1) and Bland–Altman bias and limits of the calibrated values. It then prints Cohen's kappa of the deficit
call in prodromals and overall, and the run's edge, rotation and scale diagnostics. The agreement table above comes
from the 8 September 2026 run, before the October 2026 registration changes. The October full-cohort figures are in the registration section above.

Baseline from the script on that run against Xing (1,160 QC-passing scans, Deming, 5 folds):
- Putamen: CCC 0.81 / 0.82 (L / R), with ICC(2,1) the same. Bland–Altman limits are ±0.50 on Xing's scale.
- Putamen r of the raw values was 0.91 / 0.88 inside the old window (225 scans) and 0.76 / 0.77 on its edge.
- Caudate: CCC 0.69 / 0.67.
- Deficit call at 0.65: κ 0.54 in 560 prodromal scans and 0.75 overall.
- Equipercentile calibration gives the same figures. Against Invicro (880 scans) putamen CCC is 0.83, and r is 0.925 inside the window and 0.77 on its edge.

Tests: `tests/test_datscan.py` (photopeak units, FBP hot spots, NIfTI geometry, SBR arithmetic, left/right
ordering through registration, anterior/posterior split and WM reference, fan-beam scale recovery and bounds with
scalp and neck uptake, the mm placement window and its edge flag, the QC gates, reconstructed-stack geometry and
refusal, calibration spread); table header, T1 choice, attenuation default, signature, per-model calibration and the
validation script end to end in `tests/test_imaging_regressions.py`.
