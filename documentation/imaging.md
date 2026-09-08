# Imaging layer (`pie/imaging`)

Turns raw PPMI MRI downloads from LONI (zipped DICOM) into a per-visit table of
imaging-derived phenotypes (IDPs) that joins the tabular PIE data on `PATNO` / `EVENT_ID`.

```
LONI zip(s) ──index──► one T1w series per session ──dcm2niix──► NIfTI + JSON sidecar
        ──FastSurfer (segmentation only, GPU)──► aparc.DKTatlas+aseg volumes ──► fastsurfer_idps.csv
```

## Setup

```bash
bash scripts/setup_imaging.sh          # creates venv_imaging/ and third_party/FastSurfer/
```

Requirements: Python 3.12 via `uv`, an NVIDIA GPU with >= 6 GB (FastSurferVINN view
aggregation), ~20 GB disk per 1,000 scans for NIfTI + segmentations. No FreeSurfer licence is
needed because only the segmentation stream is used (no surfaces, hence no cortical thickness).

## Run

```bash
venv_imaging/bin/python -m pie.imaging.run \
    --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
    --ppmi-dir PPMI --work-dir Imaging/derived --workers 4 --threads 4 \
    --priority Imaging/derived/priority_patnos.txt      # optional: PATNOs to do first
```

Pass the collection CSV that LONI writes next to every download with `--loni-csv Imaging/MRI_First_Study_9_07_2026.csv`:
it gives the visit label for every series (used in preference to the date-based link), research group, sex,
age at scan and the real acquisition date (LONI masks some folder dates as year 9999).
If the LONI download was made with "Advanced Download", also pass its metadata zip(s) with
`--ida-metadata Imaging/MRI_First_Study_IDA_Metadata.zip`: the `idaxs` records supply the visit
(overrides the date-based `EVENT_ID` link), research group, age at scan and the protocol terms
(manufacturer, model, field strength, slice thickness, plane, acquisition type), which fill
anonymised DICOM headers. Stub records are ignored. `pie.imaging.index.read_ida_metadata` parses them.

The run is resumable: finished scans are skipped, failures are logged to `failures.csv`, and
`fastsurfer_idps.csv` is rebuilt from whatever has finished on every call
(`--features-only` rebuilds it without processing). Throughput is GPU-bound at roughly one
scan per 30-60 s on an RTX 2080; conversion, N4 bias correction and statistics run in
parallel on the CPU.

## Modules

| Module | What it does |
|---|---|
| `index.py` | Lists every series inside the zips without extracting (`index_zips`), keeps one T1w per session (`select_t1_series`: drops localisers/calibration/non-T1 series, prefers 3D/MPRAGE-family over 2D axial descriptions and non-repeat acquisitions, then the largest series), probes DICOM headers straight from the zip (`probe_headers`), and parses LONI collection CSVs (`read_loni_collection_csv`) and IDA metadata zips (`read_ida_metadata`). |
| `convert.py` | Extracts a series to a temp dir, runs `dcm2niix`, keeps `nifti/<PATNO>/<IMAGEID>_T1w.nii.gz` and the JSON sidecar (scanner vendor, model, field strength, TR/TE/TI, ...). Handles multi-frame DICOM. |
| `link.py` | Maps the DICOM acquisition date to a PPMI `EVENT_ID` through the `Magnetic_Resonance_Imaging` table (same month, else nearest within 3 months, else `UNK`). |
| `fastsurfer.py` | Runs FastSurferVINN inference (GPU, serialised with a file lock), N4 bias correction and partial-volume-corrected `segstats` at 1 mm isotropic; `parse_stats` reads `.stats` files. |
| `features.py` | Builds the wide IDP table: every regional volume (`vol_*`, mm^3), global measures (`MaskVol`, `BrainSegVol`, ...), bilateral sums and left/right asymmetry indices (`sum_*`, `asym_*`), ventricle total, plus scanner metadata and `protocol_phase` (which zip the scan came from). |
| `labels.py` | Session-aligned outcomes: `dat_labels` (closest DaTscan: SBR values, PPMI visual read, and SBR-based deficit = lowest putamen SBR < 65 % of the age/sex expectation fitted on visually-negative controls), `saa_labels` (CSF SAA status at the MRI visit, else baseline), `covariates` (sex, birth month, cohort, LRRK2/GBA/SNCA/APOE). |
| `run.py` | The CLI above. |
| `datscan.py` | DaTscan SPECT: raw projections -> FBP reconstruction -> T1-guided SBR quantification (see below). |
| `dwi.py` | Diffusion MRI: dcm2niix -> motion correction -> tensor + free-water fits -> nigral/subcortical ROI features (see below). |
| `nm.py` | Neuromelanin-sensitive MRI: repeat averaging -> T1/atlas registration -> nigral contrast ratio and neuromelanin volume (see below). |
| `flair.py` | FLAIR: N4 -> rigid registration to T1 -> white-matter-hyperintensity burden by robust threshold (see below). |
| `batch.py` | Shared plumbing for the modality pipelines: LONI series index, dcm2niix conversion, FastSurfer lookup, session choice, resumable parallel runner (CSV append, `--retry-errors`, `--pid-file`). |
| `manifest.py` | `build_manifest` (per subject: session used per modality, dates and intervals to the T1, scanner batch per modality, QC flags) and `assemble_features` (one wide table: FastSurfer IDPs + `dat_*`, `dwi_*`, `nm_*`, `flair_*`, QC-failed values blanked); `feature_blocks` groups columns by modality for block-wise harmonisation / stacking. |
| `qc.py` | QC galleries: per-subject overlay montages (ROI contours on the image) for dwi / nm / flair / datscan outputs, sorted by a QC metric or sampled, plus a contact sheet. |

## DaTscan SPECT (`pie/imaging/datscan.py`)

PPMI's SPECT download contains the **raw tomographic projections** (NM DICOM, `ImageType TOMO/EMISSION`,
60-512 frames = detectors x energy windows x angles), not reconstructed volumes, and PPMI releases
striatal binding ratios (SBR) only for some cohorts. `datscan.py` reproduces the SBR chain with open
components, using the subject's own FastSurfer segmentation as the ROI atlas:

1. `read_projections` — selects the 159 keV photopeak window, assigns an angle to every frame from the DICOM
   NM vectors and rotation information and sums the detectors. Conventions established against PPMI's SBRs
   and the striatum position across 19 scanner configurations: DICOM angles run opposite to scikit-image's
   (`angle = -start + direction * step * view`), dual-head systems without per-detector start angles are
   H-mode (heads 180 degrees apart), and no vendor needs a left/right mirror. Broken headers (no angular
   step) and vendor unit quirks (energy windows in 1/100 keV) are handled. Some Philips series are split into
   one file per energy window; the CLI keeps the photopeak file (`_prefer_photopeak_member`).
2. `reconstruct` — filtered back-projection per transaxial slice (scikit-image, Hann filter); external point
   sources (fiducial markers) are re-projected and subtracted (`subtract_point_sources`); 6 mm Gaussian.
   `to_nifti` writes the volume in patient axes (+y anterior, +z superior).
3. `register_to_t1` — registration to a subject-specific synthetic DaT template (striatum 1.0, brain 0.25,
   head 0.12, smoothed to 10 mm; `synthetic_spect`) with a normalised-correlation metric on the head-only,
   winsorised SPECT (`_clean_spect`: the brightest 4 litres of a 20 mm-smoothed copy, largest component).
   Centre-of-mass initialisation, multi-resolution refinement; rigid for parallel-hole cameras and rigid +
   per-axis scale for fan-beam / unknown-geometry cameras (Marconi and Picker Prism reconstructed as
   parallel-beam come out ~1.5 x magnified transaxially: `hdr_scale_fit`, scales in `reg_scale_*`). A
   striatum-masked second stage was tried and removed (it drifted on faint striata).
4. `quantify` — mean counts in caudate/putamen (left/right) and occipital cortex (cuneus, lateral occipital,
   lingual, pericalcarine) from the DKT labels resampled into SPECT space, after a +-2-voxel translation
   search maximising striatal counts (mimics hottest-region ROI placement); SBR = target/occipital - 1.

```bash
venv_imaging/bin/python -m pie.imaging.datscan --index Imaging/derived/spect_index.csv \
    --sessions Imaging/derived/sessions.csv --fastsurfer-dir Imaging/derived/fastsurfer \
    --out-dir Imaging/derived/datscan --workers 8        # -> datscan_sbr.csv (+ reconstructed NIfTIs)
```

No attenuation correction is applied (a Chang implementation exists, `--attenuation`, but hurt agreement), so
absolute values sit below PPMI's; the study code calibrates PIE SBRs against PPMI's published values per
vendor on the cohorts that have them and applies the mapping to the others (prodromal). QC fields:
`reg_metric` (negative correlation; < 0.4 in magnitude flags a poor fit), `reg_scale_x/y/z`, `shift_vox`,
`n_label_voxels`, `point_source_voxels`; `reg_params`/`reg_center` store the fixed-to-moving transform
(ScaleVersor3D) so ROI variants can be recomputed without re-registering.
Validation against PPMI on 237 reference subjects: see `Parkinsons/study1_virtual_biomarkers/results/datscan_agreement.csv`.
Unit tests: `tests/test_datscan.py`.

## Using the IDPs in the PIE pipeline

```bash
python pie/pipeline.py --data-dir PPMI --output-dir output/with_imaging \
    --target-column COHORT --imaging-features Imaging/derived/fastsurfer_idps.csv
```

The IDP columns enter the reduction/merge step as the `imaging` modality (prefixed
`imaging_`). Scanner effects should be harmonised inside the cross-validation folds with
`endgame.preprocessing.ComBatHarmonizer` (batch = scanner, covariates = age, sex), never on the
full dataset before splitting.

## Limitations / next steps

- Segmentation-only stream: volumes but no cortical thickness or surface area (needs the
  FreeSurfer surface stream and a licence). PPMI's own `FS7_APARC_CTH` tables can supplement.
- No eTIV (needs talairach registration); `MaskVol` is used as the head-size normaliser.
- Only T1w handled; DTI / NM-MRI / QSM pipelines are not wrapped yet.

## Diffusion MRI (`pie/imaging/dwi.py`)

PPMI's DTI download holds two protocol generations: PPMI-1 single-shell (b = 1000; Siemens 64-direction mosaics,
GE 32-direction, Philips 32-direction LR/RL pairs) and PPMI-2 Siemens Prisma three-shell (b = 700/1000/2000, 64
directions each, reverse-phase b0s). Per subject (one session, the FastSurfer T1 of the same subject as anatomy):

1. `convert` — dcm2niix per series (bval/bvec/json); vendor-derived maps (ADC, "Reg -" series) dropped.
2. `assemble` — same-geometry, same-phase-encoding runs concatenated (the three PPMI-2 shells); of opposite-phase
   pairs (Philips LR/RL) the run with more directions is used; b0-only reverse-phase series are not used (no topup).
3. `preprocess` — brain mask (median Otsu on the mean b0) and rigid motion correction of every volume to the
   mean b0 (SimpleITK mutual information, ~0.7 s/volume). With `--fsl` and a reverse-phase b0 series (the
   three-shell PPMI-2 Prisma protocol, 107 subjects; GE "Ax DWI B-0 A/P"), `susceptibility_correct` first runs
   FSL topup on the b0 pair and applytopup (Jacobian) on every volume (~3-7 min/subject; `topup` column records
   it). eddy is not run: eddy_cuda measured 26 min per three-shell subject on the RTX 2080, so eddy-current
   distortion remains uncorrected. FSL lives in `~/fsl` (fslinstaller; the `FSLDIR` env var overrides).
4. `fit_models` — FA/MD from a weighted-least-squares tensor (b <= 1000, whole brain); free-water fraction (FW)
   and tissue FA (FAt) from the bi-tensor model inside the deep-grey/nigral ROI neighbourhood: DIPY's multi-shell
   NLS (Hoy et al. 2014) for PPMI-2, a bounded voxel-wise fit with a tissue-diffusivity prior for single-shell data
   (`fw_method` = `multishell_nls` | `singleshell_prior`; single-shell free-water is ill-posed and behaves closer to MD).
5. `register_b0_to_t1` / `register_t1_to_mni` — rigid b0 -> conformed T1 and affine T1 -> MNI152NLin2009cAsym
   (nilearn template, brain-masked, 2 mm), so that FastSurfer labels and the CIT168 subcortical atlas (Pauli 2017:
   SNc, SNr, RN, STN, VTA, GPe/GPi, NAc) land on the native DWI grid (`labels_to_dwi`).
6. `features` — mean FA/MD/FW/FAt per ROI, left/right, plus bilateral means; the substantia nigra (SNc + SNr) is
   also split into anterior and posterior halves (posterior-SN free-water is the established nigral marker).
   QC columns: `motion_mm_mean/max`, `rotation_deg_max`, `reg_b0_t1_mi`, `reg_t1_mni_mi`, `n_<roi>` voxel counts,
   `fa_wm_median`, `fw_brain_median`, `shells`, `pe_direction`, `readout_s`, `manufacturer`, `model`.

```bash
venv_imaging/bin/python -m pie.imaging.dwi --zips <DTI zips> --sessions Imaging/derived/sessions.csv \
    --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/dwi --workers 8 [--keep-nifti] [--fsl]
# -> Imaging/derived/dwi/dwi_features.csv (one row per subject), dwi_index.csv (series index)
```

Nigral ROIs also come in tissue-restricted variants (`*_t_*`: FA < 0.5 and free water < 0.7) because the affine-mapped
atlas at 2 mm takes in cerebral-peduncle fibres and interpeduncular CSF. `pie/imaging/dwi_refine.py` is an optional
pass over `--keep-nifti` outputs that re-maps the atlas with a deformable ANTs SyN T1 -> MNI registration
(`antsRegistrationSyNQuick[s]`, ~2 min/subject) and recomputes the features (`dwi_features_syn.csv`); on the test
subject it moved the nigral centroid by under a voxel, so it is not in the default path.

Unit tests: `tests/test_dwi.py` (run assembly, ROI construction, label resampling direction, single-shell free-water phantom).

## Neuromelanin-sensitive MRI (`pie/imaging/nm.py`)

PPMI-2 acquires a 2D T1-weighted gradient echo with a magnetization-transfer pulse through the midbrain
(0.5 x 0.5 x 1.5 mm, 16 slices, TR 0.45-0.65 s, TE ~5 ms, flip 40; site descriptions "AX T2 GRE MT",
"2D GRE-MT", "AXIAL 2D GRE-MT", "NM-GRE", "NM-MT", ...), usually five repeats. 654 subjects in the full-MRI
download (433 prodromal, 172 PD, 49 HC). Per subject:

1. `average_repeats` — repeats with the same geometry are rigidly aligned to the first (SimpleITK MI) and averaged
   (`n_repeats`, `repeat_motion_mm_max`).
2. `register_nm_to_t1` — rigid MI registration between the slab (fixed image, so every metric sample lies inside
   the 24 mm slab) and the *full-head* conformed T1: with a brain-masked T1 a thin slab of brain matched several
   heights equally well and settled on the striatum for some subjects, whereas the eyes, sinuses and skull pin its
   height (metric -0.73 vs -0.43). Header initialisation first; if the atlas SN does not land on the slab, a second
   start from the SN centroid in T1 space is tried and the better-covered result kept (`reg_init`). The T1 -> MNI
   affine shared with the diffusion module brings the CIT168 atlas onto the slab.
3. `nm_rois` / `features` — SN (SNc + SNr) left/right and anterior/posterior halves: contrast ratio
   CNR = (SN - ref) / ref against the crus cerebri (the part of a surrounding-midbrain ring, atlas SN dilated 3 mm
   minus the nuclei inside brainstem / ventral DC / peduncle white matter, lying anterior to the SN on the same
   side; whole-ring CNR kept as `*_cnr_ring`). Because the atlas SN can sit 1-2 mm off the thin neuromelanin band,
   two placement-robust measures are computed on a 1 mm-smoothed image inside a search region (dilated SN restricted
   to brainstem / ventral DC labels, away from the other nuclei; outside them arteries are bright on gradient echo):
   `nm_sn_*_top_cnr` = contrast of the brightest half-SN-sized volume, and the neuromelanin volume
   `nm_vol_*_voxels` = search voxels above a fixed 10 % contrast (with their mean CNR). Left/right CNR asymmetry.
   QC: `reg_nm_t1_mi`, `sn_slab_coverage` (fraction of the atlas SN inside the slab), `n_sn_*`, `nm_ref_*`. The slab
   (~24 mm) does not reach the locus coeruleus.

```bash
venv_imaging/bin/python -m pie.imaging.nm --zips <full-MRI zips> --sessions Imaging/derived/sessions.csv \
    --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4 [--keep-nifti]
# -> Imaging/derived/nm/nm_features.csv, nm_index.csv
```

Unit tests: `tests/test_nm.py` (ROI construction and contrast / thresholded-volume arithmetic on a phantom).

## FLAIR white-matter hyperintensities (`pie/imaging/flair.py`)

A vascular covariate, not a synucleinopathy marker: WMH load confounds subcortical volumes and marks the vascular
mimic that gives a normal DaTscan. 1,340 subjects with a T1 have a FLAIR (3D 1 mm or 2D 5 mm; 3D preferred when both
exist, `flair_3d` recorded for harmonisation). No licensed lesion segmenter is available (SAMSEG/LST need a
FreeSurfer or MATLAB licence, BIANCA needs labelled training data), so the classic threshold method is used:

1. dcm2niix -> N4 bias correction (SimpleITK, shrink 2) -> rigid MI registration to the brain-masked conformed T1
   (`reg_flair_t1_mi`) -> FLAIR resampled onto the T1 grid.
2. `wmh` — white matter = FastSurfer cerebral WM + WM-hypointensity labels eroded by one voxel; lesions = WM voxels
   brighter than the median of normal-appearing WM + 3 robust SD (MAD), components < 5 mm^3 removed; split into
   periventricular (<= 10 mm from the lateral ventricles) and deep.
3. Features: `wmh_mm3`, `wmh_log_mm3`, `wmh_pv_mm3`, `wmh_deep_mm3`, `wmh_frac_wm`, `wmh_n_lesions`; QC:
   `flair_wm_median`, `flair_wm_mad`, `wmh_threshold`, `wm_mm3`, registration metric. Sanity references: WMH rises
   with age and agrees with FastSurfer's T1-based WM-hypointensity volume (`validate_flair.py` in the study).

```bash
venv_imaging/bin/python -m pie.imaging.flair --zips <full-MRI zips> --sessions Imaging/derived/sessions.csv \
    --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/flair --workers 6 [--keep-nifti]
```

Unit tests: `tests/test_flair.py` (threshold, minimum size and periventricular split on a phantom).

## Cross-modality manifest, feature assembly and QC galleries

```python
from pie.imaging.manifest import build_manifest, assemble_features, feature_blocks
man = build_manifest("Imaging/derived")        # PATNO, t1_image_id/date, dat_/dwi_/nm_/flair_ date, batch, qc_pass, days_from_t1
df = assemble_features("Imaging/derived")      # manifest + FastSurfer IDPs + modality features (QC-failed values blanked)
blocks = feature_blocks(df.columns)            # {"dat": [...], "dwi": [...], "nm": [...], "flair": [...]}
```

The QC rules live in `manifest.QC` (one lambda per modality) so studies and galleries agree on what "pass" means. The
per-modality batch columns (`dwi_batch` = vendor + shells + free-water method, `nm_batch` = vendor + voxel size,
`flair_batch` = vendor + 2D/3D, `dat_batch` = vendor + camera model) are what block-wise ComBat should use; harmonising
diffusion or neuromelanin features by the *T1* scanner is a mistake the study made before this existed.

```bash
venv_imaging/bin/python -m pie.imaging.qc --work-dir Imaging/derived/dwi --modality dwi --out Imaging/derived/qc/dwi --n 40 --worst reg_b0_t1_mi
venv_imaging/bin/python -m pie.imaging.qc --work-dir Imaging/derived/datscan_full --modality datscan --out Imaging/derived/qc/datscan \
    --sessions Imaging/derived/sessions.csv --fastsurfer-dir Imaging/derived/fastsurfer --worst reg_metric
```

The T1 -> MNI affine used by the diffusion and neuromelanin modules is cached per subject at
`fastsurfer/<IMAGEID>/mri/transforms/t1_to_mni152_affine.tfm` (`dwi.register_t1_to_mni(..., cache_path=...)`), so all
modalities of a subject share one atlas mapping and it is fitted once.
