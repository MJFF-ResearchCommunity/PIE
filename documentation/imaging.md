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
| `labels.py` | Session-aligned outcomes: `dat_labels` (closest DaTscan: SBR values, PPMI visual read, and SBR-based deficit = lowest putamen SBR < 65 % of the age/sex expectation fitted on visually-negative controls), `saa_labels` (CSF SAA status at the MRI visit, else baseline), `covariates` (sex, birth month, cohort, LRRK2/GBA/SNCA/APOE, and the GP2 / META5-without-LRRK2-GBA polygenic scores when the `Polygenic_Risk_Scores` table is present). `saa_labels` also returns `SAA_Type` (Type1/Type2 seeding kinetics). |
| `run.py` | The CLI above. |
| `fmriprep.py` | Configurable native/Apptainer fMRIPrep backend with input/output/work/cache paths, version checking, execution provenance and resumable work; [usage](fmriprep.md). |
| `datscan.py` | DaTscan SPECT: raw projections -> FBP reconstruction -> T1-guided SBR quantification (see below). |
| `dwi.py` | Diffusion MRI: dcm2niix -> motion correction -> tensor + free-water fits -> nigral/subcortical ROI features (see below). |
| `fba.py` | Nigrostriatal fixel measures with MRtrix3 (`--fba` of the DWI runner): multi-tissue CSD, `mtnormalise`, iFOD2 tractography from the atlas SN to the FastSurfer striatum, AFD along the tract, seed success, FA/MD along the streamlines (see below). |
| `cnn.py` | Whole-image 3D CNN baseline (SFCN) on the conformed T1 with patient-grouped out-of-fold predictions — the "did region features miss a pattern?" check (see below). |
| `embed.py` | Fixed-length T1 embeddings from pretrained open-weight models (BrainIAC ViT, 3D-Neuro-SimCLR ResNet-18, SFCN brain age) on the affine-MNI-aligned masked T1, one CSV row per subject (see below). |
| `nm.py` | Neuromelanin-sensitive MRI: repeat averaging -> T1/atlas registration -> nigral contrast ratio and neuromelanin volume (see below). |
| `flair.py` | FLAIR: N4 -> rigid registration to T1 -> white-matter-hyperintensity burden by robust threshold (see below). |
| `batch.py` | Shared plumbing for the modality pipelines: LONI series index, dcm2niix conversion, FastSurfer lookup, session choice, resumable parallel runner (CSV append, `--retry-errors`, `--pid-file`). |
| `manifest.py` | `build_manifest` (per subject: session used per modality, dates and intervals to the T1, scanner batch per modality, QC flags) and `assemble_features` (one wide table: FastSurfer IDPs + `dat_*`, `dwi_*`, `nm_*`, `flair_*`, QC-failed values blanked; `QC["t1"]` = the key subcortical labels are non-empty, so a failed FastSurfer run is flagged `t1_qc_pass=False` and blanked); `feature_blocks` groups columns by modality for block-wise harmonisation / stacking. |
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
   Each striatal ROI is also split into anterior / posterior halves along the T1's anterior axis
   (`sbr_putamen_l_post` etc.; the posterior putamen loses dopamine transporter first), and a second SBR set
   (`sbrwm_*`) uses cerebral white matter at least 15 mm from the striatum as the reference (after the
   MJFF Research Community DaT pipeline, which references the superior longitudinal fasciculus).

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
(ScaleVersor3D) so ROI variants can be recomputed without re-registering: `--requantify` rewrites the SBR
columns of every finished row from the saved NIfTI and stored transform (`requantify_row`; ~2 s/scan; the
previous table is kept as `datscan_sbr.pre_requant.csv`).
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
   distortion remains uncorrected. FSL lives in `~/fsl` (fslinstaller; the `FSLDIR` env var overrides). Odd-sized
   axes are cropped by one voxel before topup (its `b02b0.cnf` subsamples by 2). With `--denoise`, DIPY's
   Marchenko-Pastur PCA denoising and Gibbs-ringing removal run on the raw volumes first (`denoise_dwi`, the
   dwidenoise -> mrdegibbs order of MRtrix; `denoised` column), which lowers the noise floor that biases the
   single-shell free-water fit.
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
    --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/dwi --workers 8 [--keep-nifti] [--fsl] [--denoise]
# -> Imaging/derived/dwi/dwi_features.csv (one row per subject), dwi_index.csv (series index)
```

Nigral ROIs also come in tissue-restricted variants (`*_t_*`: FA < 0.5 and free water < 0.7) because the affine-mapped
atlas at 2 mm takes in cerebral-peduncle fibres and interpeduncular CSF. `pie/imaging/dwi_refine.py` is an optional
pass over `--keep-nifti` outputs that re-maps the atlas with a deformable ANTs SyN T1 -> MNI registration
(`antsRegistrationSyNQuick[s]`, ~2 min/subject) and recomputes the features (`dwi_features_syn.csv`); on the test
subject it moved the nigral centroid by under a voxel, so it is not in the default path.

Unit tests: `tests/test_dwi.py` (run assembly, ROI construction, label resampling direction, single-shell free-water phantom).

### Nigrostriatal fixel measures (`pie/imaging/fba.py`, `--fba`)

Fixel-based analysis without a population template, per subject in native diffusion space (MRtrix3 3.0.4 on PATH):
Dhollander multi-tissue response functions, multi-shell multi-tissue CSD (WM + GM + CSF; WM + CSF on single-shell
PPMI-1 data), `mtnormalise`, then iFOD2 tractography seeded in the (refined) atlas substantia nigra of one hemisphere,
required to reach the FastSurfer putamen or caudate of the same hemisphere, stopped there, and excluded from the
contralateral hemisphere and the cerebellum (20,000 seeds, 15-90 mm). CSD and `mtnormalise` run inside a box around the
nigra, striatum, pallidum and thalamus (dilated ~10 mm, cut to the brain mask: ~25 % of the brain voxels, ~5x faster);
response functions use the whole brain. Features: `nst_afd_{l,r}` = apparent fibre density along the tract (sum of the AFD
of the traversed fixels over the streamline volume, Raffelt 2012, `afdconnectivity`) on the first 2,000 accepted streamlines,
because the value grows with the number of streamlines (fixed count: run-to-run CV ~1 %; fewer than 500 accepted -> no AFD);
`nst_seed_success_{l,r}` = fraction of seeds that produced an accepted streamline (tract-density proxy);
`nst_n_streamlines_{l,r}`; `nst_fa_{l,r}`, `nst_md_{l,r}` = FA / MD sampled along all accepted streamlines. `--keep-preproc` keeps `<work>/<patno>/fba/preproc.nii.gz` +
gradients; the normalised WM FOD and `.tck` files stay in that folder. AFD is b-value dependent: compare within the
acquisition scheme (the manifest's `dwi_batch`). Adds ~2-3 min per subject on 2 threads (~1.5 min of it CSD). The manifest exposes the
features as `dwi_nst_*`; the study's feature sets `X_nigrostriatal_fba_demo_genetics` and `NX_nigral_fw_fba_demo_genetics`
use them.

### Whole-image 3D CNN baseline (`pie/imaging/cnn.py`)

`load_volume` takes the FastSurfer `orig_nu.mgz`, masks it with `mask.mgz`, z-scores inside the brain, crops a
176 x 192 x 176 mm box around the brain centroid and mean-pools to 2 mm (88 x 96 x 88); `cache_volumes` stores every
subject once as a float16 memmap (~1.5 MB each). `sfcn()` is the SFCN of Peng et al. 2021 (five conv-BN-ReLU-maxpool
blocks 32-64-128-256-256, 1x1 conv, global average pooling, dropout, one logit); `cross_validate` gives patient-grouped
stratified out-of-fold probabilities (inner split for early stopping on AUROC, AdamW, weighted BCE, left-right flips,
random shifts, mixed precision; ~0.8 GB GPU at batch 8). CLI: `python -m pie.imaging.cnn --labels labels.csv
--fastsurfer-dir ... --cache vol.npy --out oof.csv`. `--pretrained` instead fine-tunes Peng et al.'s UK Biobank brain-age
SFCN (weights from `pie.imaging.embed`) with a fresh one-logit head on 1 mm MNI volumes (`load_volume_mni`: 160 x 192 x 160,
the authors' normalisation; 9.8 MB per subject as float16, so keep that cache on a large disk; batch 4, lr 1e-4). A network
trained from scratch on ~700 subjects is a weak reviewer baseline; the pretrained backbone is the fair one. The study driver
`dl_baseline.py [--pretrained]` compares the CNN, demographics / genetics, and their late fusion on the same subjects, with
PD-vs-HC as the positive control.

### Pretrained-model embeddings (`pie/imaging/embed.py`)

Complements the CNN baseline with frozen representations from three open-weight models, so the tabular pipeline
can test whether a generic image embedding carries information beyond the region features. `to_mni(fastsurfer_dir,
image_id, shape, origin)` masks `orig_nu.mgz` with `mask.mgz` and resamples it linearly with the per-subject cached
T1 -> MNI affine (`dwi.register_t1_to_mni`, ~6 s when not yet cached) onto the 1 mm grid each model was trained on
(`GRID`); each `embed_<backend>(volume, net)` then reproduces the authors' array order and intensity normalisation:
**brainiac** (Tak et al. 2026; MONAI ViT-B/16, 170 x 206 x 162 mm head-template box in LAS order, trilinear resize
to 96^3, z-score of nonzero voxels; 768-d token 0 of the last layer, which is the first patch token — the checkpoint
has no CLS token), **simclr** (Kaczmarek et al. 2025; 3D ResNet-18, ICBM 2009c box transposed to (z, y, x) =
150 x 192 x 192, masked z-score; 512-d pooled features), **sfcn** (Peng et al. 2021 UK Biobank brain age;
FSL MNI152 182 x 218 x 182 box in LAS order divided by its mean, then the authors' 160 x 192 x 160 centre crop; 64 penultimate channels
plus `brainage_sfcn` = expected age over the 40 one-year bins 42-82). Deviations from the authors' pipelines: affine
instead of rigid alignment for BrainIAC and SimCLR (brain size is normalised away), the FastSurfer mask instead of
HD-BET / SynthStrip, nilearn's MNI152NLin2009cAsym affine target for all three template variants, no WhiteStripe
(redundant under the masked z-score). Weights (BrainIAC research-only license, SimCLR MIT, SFCN MIT) live under
`third_party/weights/` with sources and checksums in `WEIGHTS.md`; a backend without weights is skipped. `run` writes
`emb_<backend>_<k>` columns one subject per row and resumes from the existing CSV (~5-8 s per subject for all three,
peak 3.3 GB GPU for the SimCLR ResNet at 1 mm). CLI: `python -m pie.imaging.embed --fastsurfer-dir ... --ids-csv
dataset.csv --out emb.csv --backends brainiac simclr sfcn --device cuda [--limit N]`.

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
   CNR = (SN - ref) / ref against the surrounding-midbrain ring (atlas SN dilated 3 mm minus the nuclei, inside
   brainstem / ventral DC / peduncle white matter). The part of the ring anterior to the SN (nominally crus cerebri)
   is kept as `*_cnr_crus` but is not the primary reference: the audit showed it is contaminated by the neuromelanin
   band itself (its intensity relative to the ring is lower in PD), which cancelled the group difference. The
   affine-mapped atlas SN sits 1-2 mm off the thin neuromelanin
   band in most subjects, so its position is refined per side by the translation (<= 2 mm in-plane, +-1 slice) that
   maximises the ROI mean on a 1 mm-smoothed copy while keeping >= 95 % of the ROI inside brainstem labels (the
   cistern lateral to the peduncle is bright); `nm_sn_shift_mm_*` records it and `*_cnr_atlas` keeps the unrefined
   value. Left/right CNR asymmetry. Earlier "placement-robust" measures (brightest-fraction contrast, fixed-threshold
   volumes) were removed after the 2026-09 audit: on PPMI slabs (per-voxel CV ~0.13, PPMI's 5-average 1.5 mm protocol)
   they tracked the noise level (Spearman 0.65-0.71 with the reference CV) and showed no PD-vs-HC difference, whereas
   the refined ROI mean is ~9-10 % lower in de novo PD (AUROC ~0.6 on 23 HC vs 103 PD). `--refeature` recomputes the
   feature columns from the saved slabs and label maps (`--keep-nifti` outputs) without registering.
   QC: `reg_nm_t1_mi`, `sn_slab_coverage` (fraction of the atlas SN inside the slab), `n_sn_*`, `nm_ref_*` (a reference
   ring partly outside the slab shows as SD >= 0.4 x mean and is rejected by `manifest.QC`; 7 of 646 Philips scans). The slab
   (~24 mm) does not reach the locus coeruleus.

```bash
venv_imaging/bin/python -m pie.imaging.nm --zips <full-MRI zips> --sessions Imaging/derived/sessions.csv \
    --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4 [--keep-nifti]
# -> Imaging/derived/nm/nm_features.csv, nm_index.csv
```

Unit tests: `tests/test_nm.py` (ROI construction and contrast / thresholded-volume arithmetic on a phantom).

### Neuromelanin template pipeline (`pie/imaging/nm_template.py`)

The atlas pipeline above measures the band through a T1/T2-defined SN label and a reference next to it. The template
pipeline follows Wengler et al. 2020 / Cassidy et al. 2019 instead: every averaged slab is warped into a 0.5 mm MNI
midbrain box through the rigid slab -> T1 transform (`slab_to_t1.tfm`, saved by `--keep-nifti` runs or recomputed) and
a deformable ANTs SyN T1 -> MNI warp cached per FastSurfer subject (`mri/transforms/t1_to_mni_syn_*`, ~2 min each,
reusable by `dwi_refine`); a study neuromelanin template is the mean of the intensity-normalised slabs; the SN mask
(template CNR above an Otsu / 0.06 threshold within 3 mm of the CIT168 prior) and the crus-cerebri reference (the
darker half of the sector 4-9 mm anterior-lateral to the prior) are defined on the template itself; per subject the
CNR map is `I / mode(crus) - 1` and the features are the mean CNR in the template SN and its anterior/posterior and
medial/lateral halves (`nmt_*`, plus coverage and crus CV as QC). No threshold volume is reported (noise-driven).

```bash
for stage in syn normalize template features; do
  venv_imaging/bin/python -m pie.imaging.nm_template $stage --sessions Imaging/derived/sessions.csv \
      --fastsurfer-dir Imaging/derived/fastsurfer --work-dir Imaging/derived/nm --workers 4; done
# -> Imaging/derived/nm/template/nm_template*.nii.gz, Imaging/derived/nm/nm_template_features.csv
```

Unit tests: `tests/test_nm_template.py`.

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

## Measurement follow-up safeguards (September 2026)

Run concatenation now checks TR, receiver bandwidth, and estimated readout timing
when recorded readout timing is absent, in addition to physical affine, PE and TE.
Readout grouping rounds to a microsecond so sub-microsecond JSON rounding does not
split a protocol. Estimated timing is only a mismatch guard: it does not supply
missing PE polarity or establish eligibility for topup/eddy. New complete DWI
processing is labelled `2026-09-09-acquisition-metadata-v3`.

`pie.imaging.dwi_acquisition` is an **opt-in development module**, not the cohort
default. It can regrid runs with different origins but identical axes/spacing in
physical coordinates, retain per-run b0/gradient/metadata lineage, and instrument
DIPY NLS to expose optimizer status, residuals and the un-clipped tissue tensor.
Changed axes are rejected until gradient and PE transformations are explicitly
handled. The diagnostic fit also supports a Cholesky positive-definite sensitivity
arm and an explicitly experimental positive-definite WLS initialization repair.
Direct Cholesky fitting can fail on singular initial tensors; nonfinite failures
are retained per voxel. A positive MINPACK return code alone is insufficient:
require finite outputs/residuals and a nonnegative raw tensor within numerical
roundoff (1e-12 mm^2/s). The initialization repair does not prevent all iteration
limit failures, and none of these checks validates the FW biological model.
Conditional averages over surviving voxels are not validated regional measures.
Do not run its temporary
optimizer instrumentation concurrently with another optimizer in the same Python
process. The earlier bounded CPU follow-up called `eddy_cpu`; the current
reassessment explicitly selects `eddy_cuda` with verified acquisition metadata.
Both consume eddy's rotated gradients once and do not apply topup twice.

NM repeat selection uses one homogeneous reconstruction class: paired Siemens
NORM/non-NORM outputs are not independent acquisitions. Ambiguous acquisition
identity is a visible error, not proof of duplicate scans. Missing metadata is
recorded and must be resolved for independent-acquisition reliability studies.
The new 20-person replication did not establish superiority of independently
anchor-refined over same-image-refined NM ROIs; no default ROI-method change was
made on the initial small pilot's apparent advantage. Neither conditional
repeatability nor an atlas overlay establishes anatomical accuracy.

### What is reusable in PIE, and what remains study-specific

Processing improvements are not evidence of improved SAA prediction. The completed
T1/NM reassessments did not establish incremental prediction benefit. Keep reusable
measurement safeguards separate from the study's labels, models and decision rules.

| Insight | PIE implementation | Integration status |
|---|---|---|
| Match the selected T1/session and avoid future/conflicting SAA labels | `batch.py`, `labels.py`, `manifest.py` | Existing pipeline safeguards; explicit opt-in for unmatched labels |
| Keep affine and acquisition timing/PE differences when assembling DWI | `dwi.py:assemble`, `acquisition_metadata_key` | Existing pipeline safeguards; estimated timing detects mismatches but cannot authorize eddy |
| Rotate FSL gradients in the correct physical frame after rigid motion | `dwi.py:rotate_bvec`, `preprocess` | Existing pipeline; oblique-grid regression fixtures |
| Separate NM reconstructions from independent acquisitions; preserve repeat support and hemisphere coordinates | `nm.py:compatible_repeats`, `average_repeats`, `nm_rois` | Existing pipeline; strict study eligibility thresholds remain study-specific |
| Keep one complete tensor acquisition and its own b0s; validate slice groups; preserve odd-sized grids | `dwi_correction.py` | New reusable opt-in selection/command/geometry helpers; no study imports |
| Inspect raw tensor eigenvalues and nonpositive signal before accepting FA/MD/AD/RD | `dwi_tensor_qc.py:tensor_measurements` | Opt-in measurement API already used by the reassessment, not silently substituted for legacy free-water features |
| Expose free-water optimizer status, residuals and raw tensor failures | `dwi_acquisition.py:diagnostic_multishell` | Development diagnostics only; positive-definite initialization repair is not a validated estimator |

`dwi_correction.build_eddy_command` requires verified PE/readout metadata and an
explicit assertion of raw, uncorrected input. It builds volume-motion/single-slice
outlier correction (`mporder=0`, `ol_type=sw`) and omits unverifiable slice groups.
`topup_config` selects `b02b0_1.cnf` for odd dimensions instead of cropping the scan.
`validate_corrected_geometry` checks image grids and rotated gradients; it does not
check image intensities or replace registration/ROI QC. The command and slice
decisions reproduced all 114 saved study attempts available at integration time.

The package helpers do **not** yet provide an end-to-end GPU cohort CLI. External
process execution, memory admission, restart handling, subject-level QC thresholds,
and the frozen statistical comparisons remain in the study. The current run keeps
its original hashed implementation; it is not hot-swapped to these new helpers.
The maintained follow-up should call these package APIs from one configurable
driver, rather than accumulate more independent scheduling scripts. Historical
script snapshots stay available for reproducibility; they are not alternative
recommended production entry points.

Storage locations and capacity policies are caller-configured; PIE does not
require an external disk or a particular filesystem driver. Use storage that
supports the selected tools and has sufficient space for their working files.
Keep machine-specific mount, backup and recovery procedures in the deployment
or study runner. Verify copied data before retiring a working copy, and do not
move active writers' files. An infrastructure failure is not a scientific scan
exclusion or proof of successful recovery. Preserve original acquisitions.
