# Neuromelanin MRI and DaTscan SPECT

Part of the [imaging layer](imaging.md). Both read the subject's FastSurfer T1 (`orig.mgz`, `mask.mgz`,
`aparc.DKTatlas+aseg.deep.mgz`) as anatomy and write one row per subject.

| Module | Output | External tools |
|---|---|---|
| `nm.py` | `nm_features.csv`: nigral contrast ratios on the native NM slab | dcm2niix, SimpleITK (the MNI target is bundled with PIE) |
| `nm_template.py` | `nm_template_features.csv`: contrast in a study NM template | + ANTsPy (`ants`), scikit-image, nilearn; the 1 mm MNI152NLin2009cAsym target is fetched from TemplateFlow once |
| `datscan.py` | `datscan_sbr.csv`: striatal binding ratios from raw projections | pydicom, scikit-image, SimpleITK |

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
   first, then the mean. Returns `(image, meta, n_repeats, max_motion_mm)`; `meta["PIERepeatSelection"]` holds
   the selection record (written to `<work>/<patno>/repeat_selection.json`). With `transforms_dir` it also saves
   `repeat_<i>_to_anchor.tfm` and `repeat_support_fraction.nii.gz`.
3. **`register_slab(nm_img, fastsurfer_dir, sampling_seed=0)`** — **`register_nm_to_t1(nm_img, t1_img,
   target_center=None, sampling_seed=0)`** with the slab as fixed image (every metric sample lies inside the
   ~24 mm slab) against the *unmasked* T1: with a brain-masked T1 a thin slab matched several heights equally
   well and settled on the striatum for some subjects, whereas eyes, sinuses and skull pin the height (MI
   metric -0.73 vs -0.43). Header
   initialisation first; if the atlas SN does not land on the slab, a second start at the atlas SN centroid in T1
   space; the result covering more SN wins (`reg_init` = `header` | `sn_centroid`). The T1 → MNI affine is the
   one cached by `dwi.register_t1_to_mni`, so the CIT168 atlas (`dwi.pauli_atlas()`, MNI2009c v1.0) maps onto the
   slab with the FastSurfer labels.
4. **`nm_rois(fs_lab, pauli_lab, spacing_mm, left_mask=None)`** — SN = CIT168 SNc + SNr per side; reference ring
   = SN dilated `DILATE_MM` (3 mm, anisotropic) minus the dilated nuclei (SNc, SNr, RN, VTA, STN, PBP), inside
   brainstem / ventral DC / cerebral WM / thalamus labels. Hemispheres come from `atlas_left_mask(tgt, chain)`
   (the MNI left half-space mapped onto the slab), because the slab's voxel x direction is not an anatomical
   label; without it, bilateral putamen landmarks are required or it raises.
5. **`features(nm, rois, phys_y, spacing_zyx=(1.5, 0.5, 0.5), smooth_fwhm_mm=1.0, mask_out=None)`** — each side's
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
an atlas overlay establishes anatomical accuracy. A 20-person replication did not establish that independently
anchor-refined ROIs beat same-image-refined ones, so the default ROI method was not changed. Order statistics on
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
| `--refeature` | Recompute the feature columns of every finished subject from the saved slab and label maps (`refeature_subject(work_dir, patno, fastsurfer_dir=None, atlas_sha256=None, registration_reference_sha256=None)`), without registering. Needs `--keep-nifti` outputs. The saved atlas map is reused only when the row's `atlas_sha256` **and** `registration_reference_sha256` match the bundled atlas and its reference template: the current atlas hash alone does not authorise an old mapping, because the map may have been made against another reference. Otherwise (rows from before 2026-09-15) the atlas and hemisphere maps are regenerated from `slab_to_t1.tfm` and the verified T1 → MNI cache and rewritten, `sn_slab_coverage` and the provenance columns are recomputed, and a subject without those transforms (or with an unverified cache) gets an `error` instead of stale-atlas values. Rows are labelled `<PROCESSING_VERSION>-refeatured`. The previous table is kept once as `nm_features.pre_refeature.csv`; `--patnos` restricts the run; `--zips` is required by the parser but unused |

`--keep-nifti` saves, per subject: `nm_mean.nii.gz`, `pauli_nm.nii.gz`, `aseg_nm.nii.gz`, `ref_nm.nii.gz`,
`left_nm.nii.gz`, `sn_refined_nm.nii.gz` (1 = left, 2 = right), `slab_to_t1.tfm` (T1 point → slab point; reused by
`nm_template`) and `repeat_transforms/`.

### Output columns (`nm_features.csv`)

| Column | Meaning |
|---|---|
| `nm_sn_{l,r}_cnr` | Refined SN vs the whole ring (primary) |
| `nm_sn_{posterior,anterior}_{l,r}_cnr` | Halves of the refined SN, split at its median A-P coordinate |
| `nm_sn_{l,r}_cnr_crus`, `nm_sn_{l,r}_cnr_atlas` | Same-side anterior reference; unrefined atlas SN vs ring |
| `nm_sn_mean_cnr`, `nm_sn_posterior_mean_cnr`, `nm_sn_anterior_mean_cnr`, `nm_sn_min_cnr`, `nm_sn_asym_cnr`, `nm_sn_mean_cnr_atlas` | Bilateral mean, min, \|L−R\| |
| `nm_sn_{l,r}_mean`, `nm_ring_mean`, `nm_ring_sd`, `n_ring`, `nm_ref_{l,r}_mean`, `nm_ref_{l,r}_sd`, `n_ref_{l,r}`, `n_sn_{l,r}` | Raw intensities and voxel counts |
| `nm_sn_shift_mm_{l,r}` | Refinement displacement (mm) |
| `n_repeats`, `repeat_motion_mm_max`, `reg_nm_t1_mi`, `reg_t1_mni_mi`, `reg_init`, `sn_slab_coverage` | QC |
| `shape`, `voxel_mm`, `manufacturer`, `model`, `tr_s`, `te_s`, `flip_angle`, `mt_flag`, `series_desc`, `n_series` | Acquisition |
| `patno`, `acquisition_date`, `fs_image_id`, `processing_version`, `atlas_space`, `atlas_version`, `atlas_sha256`, `atlas_probability_threshold`, `registration_reference_space`, `registration_reference_sha256`, `error` | Lineage |

`sn_slab_coverage` is (SN voxels on the slab x voxel volume) / atlas SN voxels in 1 mm T1 space: a resampled-volume
ratio, not a bounded fraction. New complete runs are labelled `nm.PROCESSING_VERSION` = `2026-09-16-explicit-mni2009c-reference-v5`; `--refeature` rows get that string plus `-refeatured`.
`manifest.QC["nm"]`: `n_sn_l >= 20`, `n_sn_r >= 20`, `sn_slab_coverage >= 0.5`, `repeat_motion_mm_max < 3`, and
`nm_ref_{l,r}_sd < 0.4 x nm_ref_{l,r}_mean` (a reference partly outside the slab has CV near 1 and meaningless
CNR; this rejected 7 of 646 Philips scans in the September 2026 snapshot). The assembled table keeps `nm_*` columns ending `_cnr` (so not `_cnr_crus` / `_cnr_atlas`).

Tests: `tests/test_nm.py` (planted contrast, independent noise, band-offset refinement, reference excludes both
refined masks) and the hemisphere / no-wrap cases in `tests/test_imaging_audit.py`; refeature versioning and stale-atlas handling
in `tests/test_imaging_regressions.py`.

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
| `syn` | `syn_cache(fastsurfer_dir, syn_type=SYN_TYPE)` (`antsRegistrationSyNQuick[s]` to the brain-masked 1 mm MNI152NLin2009cAsym, `atlases.mni2009c_brain_1mm()`; ~7 CPU-minutes per subject), `crop_warp` | `<fastsurfer>/<IMAGE_ID>/mri/transforms/t1_to_MNI152NLin2009cAsym_syn_1Warp.nii.gz` (cropped to the box + 20 mm) and `…_0GenericAffine.mat`. Caches named `t1_to_mni_syn_*` were fitted to nilearn's 2009a template before 25 September 2026 and are not reused |
| `normalize` | `normalize_subject(work_dir, patno, fastsurfer_dir)`, `slab_in_t1` | `<work>/<patno>/nm_mni.nii.gz`, `nm_normalize_log.csv` (`mni_nonzero_frac`) |
| `template` | `build_template(work_dir, patnos, min_frac=0.5)` → `(template, count, n)`; `template_masks(template, crus_mm=(4.0, 9.0), sn_mm=3.0, cnr_min=0.06)`; `save_template`; `template_figure` | `<work>/template/nm_template.nii.gz`, `nm_template_count.nii.gz`, `nm_template_masks.nii.gz` (1 sn_l, 2 sn_r, 3 crus_l, 4 crus_r), `template_info.txt`, `template_qc.png` |
| `features` | `load_masks(work_dir)`, `template_features(nm_mni, masks, prefix="nmt_")` | `<work>/nm_template_features.csv` |

Template: each normalised slab divided by its median inside the SN prior (bundled CIT168 SNc + SNr, `sn_prior()`)
dilated 10 mm, averaged; voxels with data in fewer than `min_frac` of subjects are zero. Crus = darker half of the
sector 4-9 mm anterior-lateral to the prior; SN = voxels within 3 mm of the prior with template CNR >=
max(0.06, Otsu). Per subject, CNR = I / mode(crus) − 1. No threshold volume is reported (noise-driven).

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
of the SN mask with data); `patno`, `error`. `manifest.assemble_features` does not read this table; join it on
`patno` yourself. Tests: `tests/test_nm_template.py`.

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
   headers (no angular step) are inferred from the frame count. Raises `ValueError` for non-TOMO series;
   `read_volume(dcm)` handles series stored as a reconstructed stack. Returns `proj`, `angles_deg`, `spacing_mm`,
   `meta` (the `hdr_*` columns).
2. **`reconstruct(proj, angles_deg, spacing_mm, fwhm_mm=6.0, filter_name="hann", attenuation=False,
   point_sources=True)`** — filtered back-projection per transaxial slice (scikit-image `iradon`);
   `subtract_point_sources(vol, angles_deg, spacing_mm, filter_name="hann", factor=6.0, max_voxels=200)` removes
   external fiducial markers, whose coherent back-projection otherwise streaks through the volume and flattens
   every SBR; optional `chang_correction(vol, spacing_mm, mu_per_cm=0.11, n_dirs=36, threshold=0.15)`; Gaussian to
   `fwhm_mm`. `attenuation` defaults to off here, as in `process_series` and
   the CLI (until September 2026 this function alone defaulted to on), so a direct call reproduces the pipeline.
   `to_nifti(vol, spacing_mm)` writes patient axes (+y anterior, +z superior), centred at the origin.
3. **`register_to_t1(spect_img, t1_img, flip_lr=False, rz=0.0, search=False, aparc_img=None,
   scale_fit=False)`** — normalised correlation against `synthetic_spect(t1_img, aparc_img, fwhm_mm=10.0)` (striatum
   1.0, brain 0.25, head 0.12, smoothed), with the SPECT cleaned by `_clean_spect` (largest component of the
   brightest 4 L of a 20 mm-smoothed copy, winsorised). Centre-of-moments initialisation, 3-level refinement,
   `ScaleVersor3DTransform`: rigid for parallel-hole cameras, rigid + per-axis scale when `scale_fit` (fan-beam
   collimators, or no collimator/zoom recorded: Marconi and Picker Prism data reconstructed as parallel-beam come
   out ~1.5 x magnified transaxially). A striatum-masked second stage was tried and removed: it drifted on faint striata
   (side-wise putamen agreement with PPMI fell from 0.84 to 0.78 pooled, 0.64 to 0.27 on ADAC).
4. **`quantify(spect_img, t1_img, aparc_img, flip_lr=False, rz=0.0, search=False, dilate=0, ref_dilate=0,
   search_vox=2, scale_fit=False)`** → `sbr_with_transform` → **`sbr_from_arrays(S, L, dilate=0,
   ref_dilate=0, search_vox=2, ap=None, spacing_mm=None)`**: DKT labels resampled onto the SPECT grid, a ±2-voxel
   translation search maximising striatal counts (mimics hottest-region ROI placement), SBR = target / occipital
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

The ±2-voxel placement search earns its place: re-quantifying 300 of those scans from the saved reconstructions and
transforms with fixed placement (`search_vox=0`) lowered agreement with Xing from r 0.80 to 0.70–0.72 on the putamen
and from 0.60–0.64 to 0.51–0.55 on the caudate, because the search absorbs residual registration error. It also raises
every SBR (putamen by ~0.15 on the WM scale), part of which is the upward bias of taking a maximum over noisy positions,
largest on low-count scans.

No attenuation correction by default: a Chang implementation exists (`--attenuation`) but lowered agreement with
PPMI's values. Absolute SBRs therefore sit below PPMI's; calibrating them per vendor against PPMI's published SBRs
is study code, not part of PIE; the study validated PIE SBRs against PPMI's on 237 reference subjects (September
2026 snapshot; `Parkinsons/study1_virtual_biomarkers/results/datscan_agreement.csv`, not in PIE). `reg_params` / `reg_center` store the fitted fixed → moving transform so ROI
variants can be recomputed without re-registering: `transform_from_row(row)`, `requantify_row(row,
fastsurfer_subject_dir)`.

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
    --out-dir <derived>/datscan_full --workers 8      # -> datscan_sbr.csv + nifti/<IMAGE_ID>_datscan.nii.gz
```

`manifest` reads `<derived>/datscan_full/datscan_sbr.csv` by default; for another `--out-dir` pass
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
| `--requantify` | off | Recompute SBR columns of finished rows from the saved NIfTI and stored transform (`requantify_row`, ~2 s/scan) against the row's `fs_image_id` (older rows: the earliest finished FastSurfer session); the previous table is kept once as `datscan_sbr.pre_requant.csv` |

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
| `shift_vox`, `n_label_voxels`, `n_ref_voxels`, `n_wm_voxels`, `point_source_voxels` | QC |
| `hdr_manufacturer`, `hdr_model`, `hdr_collimator`, `hdr_zoom`, `hdr_scale_fit`, `hdr_n_frames`, `hdr_n_detectors`, `hdr_n_angles`, `hdr_angular_step`, `hdr_scan_arc`, `hdr_rotation_direction`, `hdr_start_angle`, `hdr_energy_window`, `hdr_rows`, `hdr_cols`, `hdr_spacing_mm`, `hdr_counts_total` | Acquisition header |
| `image_id`, `patno`, `series_desc`, `kind` (`projections` / `stored_volume`), `nifti`, `fs_image_id`, `error` | Lineage |

`manifest.QC["dat"]`: `|reg_metric| >= 0.4` and `n_label_voxels >= 100`. The manifest's `dat_batch` is vendor +
first 12 characters of `hdr_model`; assembled features keep `dat_sbr_{caudate,putamen}_{l,r}`.

Tests: `tests/test_datscan.py` (photopeak units, FBP hot spots, NIfTI geometry, SBR arithmetic, left/right
ordering through registration, anterior/posterior split and WM reference); table header, T1 choice, attenuation
default and signature in `tests/test_imaging_regressions.py`.
