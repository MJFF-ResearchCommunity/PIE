# Imaging layer (`pie/imaging`)

Turns raw PPMI imaging downloads from LONI (zipped DICOM) into per-subject or per-visit tables of
imaging-derived phenotypes (IDPs) that join the tabular PIE data on `PATNO` / `EVENT_ID`.

```
LONI MRI zip(s) ──index──► one T1w per session ──dcm2niix──► NIfTI ──FastSurfer (GPU)──► fastsurfer_idps.csv
                                                                          │ (T1 anatomy, labels, MNI affine)
          DTI zips ──dwi──────────────► dwi/dwi_features.csv ◄────────────┤
     full-MRI zips ──nm / flair───────► nm/nm_features.csv, flair/flair_features.csv ◄┤
  SPECT projections ──datscan─────────► datscan_full/datscan_sbr.csv ◄────┘
                           manifest.assemble_features ──► one wide row per subject (QC-failed values blanked)
```

The T1 run comes first: every other modality uses the subject's FastSurfer segmentation as anatomy and its
`sessions.csv` to find it. `<derived>` below is the `--work-dir` of `pie.imaging.run` (default `Imaging/derived`).

| Page | Covers |
|---|---|
| this page | setup, T1 / FastSurfer IDPs, labels, FLAIR, manifest and QC galleries, CNN and embeddings, shared infrastructure |
| [imaging_dwi.md](imaging_dwi.md) | `dwi`, `fba`, `dwi_refine`, `dwi_tensor_qc`, `freewater_qc`, `dwi_acquisition`, `dwi_correction`, `mrtrix_shim` |
| [imaging_nm_datscan.md](imaging_nm_datscan.md) | `nm`, `nm_template`, `datscan` |
| [fmriprep.md](fmriprep.md), [brain_viewer.md](brain_viewer.md), [fmri_viewer.md](fmri_viewer.md) | fMRI (`fmri*.py`) and the viewers |

## Modules

| Module | What it does | CLI |
|---|---|---|
| `index.py` | List series inside LONI zips without extracting; pick one T1w per session; read LONI collection CSVs and IDA metadata | self-check |
| `convert.py` | Extract one series, run dcm2niix, keep NIfTI + JSON sidecar | |
| `link.py` | Map a session date to a PPMI `EVENT_ID` | |
| `fastsurfer.py` | FastSurferVINN segmentation (GPU), N4 + segstats (CPU), `.stats` parser | self-check |
| `features.py` | Wide IDP table from FastSurfer stats | |
| `run.py` | Resumable T1 → IDP CLI | yes |
| `labels.py` | DaTscan-deficit and SAA labels and covariates aligned to an MRI session | |
| `batch.py` | Shared plumbing for the modality runners | |
| `dwi.py`, `fba.py`, `dwi_refine.py` | Diffusion pipeline, tract measures, SyN refinement ([page](imaging_dwi.md)) | yes |
| `dwi_tensor_qc.py`, `freewater_qc.py`, `dwi_acquisition.py`, `dwi_correction.py` | Opt-in DWI measurement and correction safeguards ([page](imaging_dwi.md#opt-in-measurement-apis)) | |
| `nm.py`, `nm_template.py` | Neuromelanin MRI ([page](imaging_nm_datscan.md)) | yes |
| `datscan.py` | DaTscan SPECT reconstruction and SBRs ([page](imaging_nm_datscan.md#datscan-spect-datscanpy)) | yes |
| `flair.py` | White-matter hyperintensity burden | yes |
| `manifest.py` | Per-subject manifest, QC rules, assembled feature table | |
| `qc.py` | Overlay montages and contact sheets for visual QC | yes |
| `cnn.py` | Whole-image 3D CNN baseline (SFCN) with grouped out-of-fold predictions | yes |
| `embed.py` | Embeddings from pretrained open-weight T1 models | yes |
| `atlases.py` + `data/atlases/` | Bundled, checksummed CIT168 atlas in MNI152NLin2009cAsym | |
| `archives.py` | Bounded LRU cache of open ZIP indexes | |
| `dicom_audit.py` | Read-only DICOM header / CRC evidence for conversion-failure review | |
| `staging.py` | Verified, non-destructive copies of work directories | |
| `mrtrix_shim/imp.py` | `imp` stand-in for MRtrix3 3.0.x Python scripts on Python 3.12 | |

`pie.imaging` itself exports `index_zips`, `select_t1_series`, `convert_series`, `link_sessions_to_events`,
`run_fastsurfer`, `parse_stats`, `build_idp_table`, `dat_labels`, `saa_labels`.

## Setup

```bash
bash scripts/setup_imaging.sh          # or: bash scripts/setup_imaging.sh cpu   (default backend cu128)
```

The script (needs [`uv`](https://docs.astral.sh/uv/)) clones FastSurfer into `third_party/FastSurfer`, creates a
Python 3.12 `venv_imaging/` with FastSurfer's requirements (torch for the chosen backend, SimpleITK,
scikit-image, MONAI, …) plus `dcm2niix pydicom nibabel pandas numpy neuroCombat scikit-learn dipy nilearn antspyx
matplotlib pytest` and the modelling libraries, and tries `endgame-ml[tabular]`. DIPY and nilearn are needed by
`dwi`, `nm`, `flair`, `embed` and `nm_template`; ANTsPy (`antspyx`) by `dwi_refine` and `nm_template`.

| Tool | Used by | How PIE finds it |
|---|---|---|
| dcm2niix | `convert`, `batch` (all modalities) | `$PIE_DCM2NIIX`, else `<repo>/venv_imaging/bin/dcm2niix` (`convert.DCM2NIIX`) |
| FastSurfer (segmentation only) | `fastsurfer`, `run` | `$PIE_FASTSURFER_HOME`, else `<repo>/third_party/FastSurfer`; run with `$PIE_FASTSURFER_PYTHON`, else `<repo>/venv_imaging/bin/python` |
| NVIDIA GPU, >= 6 GB | FastSurferVINN; `cnn`, `embed` | `--device` on `run`, `cnn`, `embed` (default `cuda`; `cpu` works, much slower) |
| FSL `topup`/`applytopup` | `dwi --fsl` | `$FSLDIR`, else `~/fsl` |
| MRtrix3 3.0.x | `dwi --fba`, `fba`; `dwidenoise`/`mrdegibbs` for `--denoise` if present | `PATH` |
| ANTsPy | `dwi_refine`, `nm_template` | Python import `ants` |
| Pretrained weights | `embed`, `cnn --pretrained` | `$PIE_WEIGHTS_DIR`, else `<repo>/third_party/weights`, then `<backend>/`; sources and checksums in `third_party/weights/WEIGHTS.md` |

The defaults are relative to the source tree; set the `PIE_*` environment variables (read at import) to run from an
installed package or with tools elsewhere. No FreeSurfer licence is needed: only FastSurfer's segmentation stream is used (no surfaces, so no cortical
thickness). Disk: roughly 20 GB per 1,000 T1 scans for NIfTI + segmentations.

## Structural T1: FastSurfer IDPs (`run.py`)

```bash
venv_imaging/bin/python -m pie.imaging.run \
    --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
    --ppmi-dir PPMI --work-dir Imaging/derived --workers 4 --threads 4 \
    --loni-csv Imaging/MRI_First_Study_9_07_2026.csv --ida-metadata Imaging/MRI_First_Study_IDA_Metadata.zip \
    [--priority patnos.txt] [--limit N] [--device cpu]
```

The file names are those of the study's September 2026 LONI download; LONI writes the collection CSV next to every
download, and the IDA metadata zip comes with "Advanced Download".

| Flag | Default | Meaning |
|---|---|---|
| `--zips` | required | LONI zip(s), laid out `PPMI/<PATNO>/<SeriesDescription>/<yyyy-mm-dd_HH_MM_SS.0>/<IMAGE_ID>/*.dcm`. Their order sets `protocol_phase` (1, 2, …) |
| `--ppmi-dir` | `PPMI` | PPMI tabular download; `Imaging/Magnetic_Resonance_Imaging__MRI__*.csv` (latest file) is used for visit linking |
| `--work-dir` | `Imaging/derived` | Output root |
| `--workers` | 4 | CPU processes for conversion and stats |
| `--threads` | 4 | CPU threads per FastSurfer call |
| `--chunk` | 20 | Scans per FastSurferVINN process (models loaded once per chunk) |
| `--device` | `cuda` | FastSurfer inference device, passed to its `--device` (e.g. `cuda`, `cuda:1`, `cpu`); non-CUDA devices get hour-scale time budgets (step 5) |
| `--loni-csv` | none | LONI collection CSV(s) downloaded with the images (`Image Data ID`, `Visit`, `Group`, `Sex`, `Age`, `Description`, `Acq Date`) |
| `--ida-metadata` | none | LONI "Advanced Download" metadata zip(s) or directories of `idaxs` XML |
| `--priority` | none | Text file of PATNOs to process first |
| `--limit` | none | Process at most N sessions this call |
| `--features-only` | off | Only rebuild `fastsurfer_idps.csv` from what has finished |

What happens:

1. **Index** — `index_zips(zip_paths, cache_csv=None)` reads the zip listings (no extraction) into one row per
   series: `zip, patno, series_desc, session, image_id, n_files, bytes, member_prefix, session_date`. Cached as
   `index.csv`.
2. **Choose the T1** — `select_t1_series(index)` drops localisers, calibration and non-T1 descriptions (`_NOT_T1`),
   then per `(patno, session)` ranks 3D/MPRAGE-family descriptions up, 2D/axial ones down (some sites label a
   2-frame axial T1 as the only "T1", which FastSurfer cannot use), repeats down, and breaks ties by bytes (this
   also handles single-file multi-frame DICOM). Sessions dated 9999 (LONI date masking) are kept with
   `date_masked=True`.
3. **Link visits** — `link_sessions_to_events(sessions, ppmi_dir, max_months=3)`: the MRI table's `INFODT` is a
   month, so a session gets the `EVENT_ID` of the same month, else the nearest within 3 months, else `UNK`
   (`months_off` records the gap). `--loni-csv` (`read_loni_collection_csv`) and then `--ida-metadata`
   (`read_ida_metadata`; `<metadata>` stubs are skipped) override it where their visit label maps through
   `LONI_VISIT_TO_EVENT` / `IDA_VISIT_TO_EVENT` (only Baseline/BL and Screening/SC), and add `loni_*` / `ida_*`
   columns (group, sex, age at scan, acquisition date, and for IDA the protocol terms: manufacturer, model, field
   strength, slice thickness, plane, acquisition type, weighting). The result is cached as `sessions.csv`;
   delete it to apply a newly added CSV or metadata zip.
4. **Convert** — `convert_series(zip_path, member_prefix, patno, image_id, out_dir, dcm2niix=DCM2NIIX)` writes
   `nifti/<PATNO>/<IMAGE_ID>_T1w.nii.gz` + `.json` (the largest output if dcm2niix splits a series) and returns
   the sidecar fields in `SIDECAR_FIELDS` (vendor, model, field strength, TR/TE/TI, flip angle, …). Volumes that
   are not 3D, have a dimension < 40 or a voxel > 2.5 mm are rejected before they can abort a GPU batch.
5. **Segment** — `segment_batch(niftis, subjects_dir, threads=4, device="cuda", batch=4)` runs FastSurferVINN on a
   chunk in one process behind a file lock (`.gpu.lock`: one inference on the GPU at a time). On CUDA a chunk gets
   90 s per scan + 300 s and is killed after `STALL_SECONDS` (420 s) without a new segmentation; a single-scan
   `segment` gets 900 s. Any other device gets `CPU_SCAN_SECONDS` per scan and as the stall limit (3600 s, override
   with `PIE_FASTSURFER_CPU_SECONDS`), because CPU inference takes tens of minutes. Scans the batch skipped are retried alone
   (`segment`, via `run_fastsurfer`). `complete_segmentation` rebuilds `mask.mgz` / `aseg.auto_noCCseg.mgz`,
   which FastSurfer's multi-subject mode can drop; half-written segmentations from a killed batch are swept
   and re-queued.
6. **Stats** — `finish_stats(subjects_dir, sid, threads=4)`: N4 bias correction (`orig_nu.mgz`) and
   partial-volume-corrected `segstats` at 1 mm into `stats/aseg+DKT.stats`, on the CPU pool while the GPU works
   on the next chunk. `parse_stats(path)` returns `{StructName: mm^3}` plus `# Measure` short names.
7. **Table** — `build_idp_table(sessions, subjects_dir)` every call, from whatever has finished.

Resumable: sessions with `stats/aseg+DKT.stats` are skipped. Throughput is GPU-bound, roughly one scan per
30-60 s on an RTX 2080.

Work dir: `index.csv`, `sessions.csv`, `scan_metadata.csv` (`image_id`, `nifti`, `sidecar` + `SIDECAR_FIELDS`),
`nifti/<PATNO>/`, `fastsurfer/<IMAGE_ID>/{mri,stats,scripts}`, `failures.csv` (headerless: patno, image_id,
message), `fastsurfer_idps.csv`.

### IDP columns (`fastsurfer_idps.csv`, one row per processed session)

| Column | Meaning |
|---|---|
| `PATNO`, `EVENT_ID`, `IMAGEID`, `SCAN_DATE`, `protocol_phase` | Identifiers; `protocol_phase` = position of the source zip in `--zips` |
| `Manufacturer`, `ManufacturersModelName`, `MagneticFieldStrength`, `SoftwareVersions`, `InstitutionName`, `RepetitionTime`, `EchoTime`, `InversionTime`, `FlipAngle`, `SliceThickness` | dcm2niix sidecar (`features.META_COLS`); scanner batch for harmonisation |
| `MaskVol`, `BrainSegVol`, … | Global measures whose short name ends in `Vol` keep that name; other measures become `vol_<name>` |
| `vol_<Structure>` | Regional volume in mm^3, name with non-alphanumerics replaced by `_` (`vol_Left_Putamen`, `vol_ctx_lh_precuneus`, `vol_WM_hypointensities`) |
| `sum_<S>`, `asym_<S>` | Left + right and (L − R) / (L + R) for `Putamen Caudate Pallidum Thalamus Hippocampus Amygdala Accumbens_area Lateral_Ventricle Cerebellum_Cortex Cerebellum_White_Matter VentralDC` |
| `sum_Ventricles` | Lateral + inferior-lateral + 3rd + 4th ventricles |

No eTIV (needs talairach registration); use `MaskVol` as the head-size normaliser.

## Labels and covariates (`labels.py`)

Read the PPMI tabular download (latest matching file per table) and align outcomes to MRI sessions. `sessions`
is a frame with `patno`, `image_id`, `session_date`, `EVENT_ID` (e.g. `sessions.csv`).

| Function | Output |
|---|---|
| `covariates(ppmi_dir)` | Per PATNO: `COHORT`, `ENROLL_DATE`, `ENROLL_AGE`, `SEX`, `BIRTHDT`, `HANDED`, `LRRK2`, `GBA`, `SNCA`, `APOE`, `PATHVAR_COUNT`, `{LRRK2,GBA,SNCA}_carrier` (0 for `0`, `0.0`, `"0"`, `"0.0"` whatever the column dtype, 1 for any other value, NaN when blank or missing: untested is not a control), `APOE_e4` (count), and `PRS_GP2`, `PRS_META5_noLRRK2GBA` when `Polygenic_Risk_Scores_*.csv` exists |
| `dat_labels(ppmi_dir, sessions, threshold=0.65, max_months=18)` | Closest analysed DaTscan within 18 months: `PATNO`, `IMAGEID`, `months_to_datscan`, `DATSCAN_DATE`, `DATSCAN_EVENT_ID`, `DATSCAN_{CAUDATE,PUTAMEN}_{R,L}`, `sbr_putamen_min`, `sbr_caudate_min`, `sbr_putamen_mean`, `sbr_pct_expected`, `dat_visual` (PPMI visual read, 1/0), `dat_deficit_sbr` = lowest putamen SBR < `threshold` x the age/sex expectation fitted on visually-negative healthy controls (PPMI's prodromal convention) |
| `saa_labels(ppmi_dir, sessions, allow_unmatched=False)` | `PATNO`, `IMAGEID`, `SAA_EVENT_ID`, `SAA_Status` (`Conflicting` when calls at the chosen visit disagree), `SAA_Type`, `saa_positive` (NaN unless one Positive/Negative call), `saa_match` (`same_visit`, `screening_baseline_pair`, `unmatched_visit`), `saa_visit_concurrent` |

`saa_labels` uses the SAA of the MRI's own visit, treating SC and BL as one occasion. It never borrows a later
assay for an inconclusive concurrent visit: `RUNDATE` is when the laboratory ran the sample, not when CSF was
collected. `allow_unmatched=True` reproduces the historical fallback (earliest SAA visit) for an explicitly
labelled sensitivity analysis only. Tables read: `_Subject_Characteristics/{Participant_Status,Demographics,
iu_genetic_consensus,Polygenic_Risk_Scores}_*.csv`, `Imaging/DaTScan_{SBR_Analysis,Visual_Interpretation_Results}_*.csv`,
`Biospecimen/SAA_Biospecimen_Analysis_Results_*.csv`.

## FLAIR white-matter hyperintensities (`flair.py`)

A vascular covariate, not a synucleinopathy marker: WMH load confounds subcortical volumes and marks the vascular
mimic behind some normal DaTscans. No licensed lesion segmenter is available (SAMSEG/LST need a FreeSurfer or
MATLAB licence, BIANCA needs labelled training data), so this is the classic threshold method. In the study's
September 2026 MRI download (snapshot) 1,340 subjects with a T1 have a FLAIR, 3D 1 mm or 2D 5 mm.

1. Series: `flag_flair` (`FLAIR|dark.?fluid|tirm`, excluding T1 FLAIR and repeats); `flag_flair_3d` = "3D" in
   the description or >= 100 files. 3D is preferred, then the largest series (`flair_3d` is recorded for
   harmonisation).
2. `n4(img_sitk, shrink=2)` → `register_flair_to_t1(flair_sitk, t1_img, t1_mask_img)` (rigid MI to the
   brain-masked T1 at 2 mm) → FLAIR resampled onto the T1 grid.
3. `wmh(flair_t1, aseg, vox_mm=1.0)`: white matter = FastSurfer labels 2, 41, 77 eroded by one voxel; lesions =
   WM voxels brighter than median + `K_MAD` (3) x 1.4826 MAD of normal-appearing WM; components below
   `MIN_LESION_MM3` (5 mm^3) dropped; periventricular = within `PV_MM` (10 mm) of the lateral ventricles.

```bash
venv_imaging/bin/python -m pie.imaging.flair --zips <full-MRI zips> --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/flair --workers 6 [--keep-nifti]
# -> flair_features.csv, flair_index.csv; --keep-nifti adds <patno>/flair_t1.nii.gz, wmh_t1.nii.gz
```

Flags: the shared `batch.add_common_args` set, see [the DWI flag table](imaging_dwi.md#cli) (no extra flags).

Columns: `wmh_mm3`, `wmh_log_mm3` (log1p), `wmh_pv_mm3`, `wmh_deep_mm3`, `wmh_frac_wm`, `wmh_n_lesions`; QC
`reg_flair_t1_mi`, `wm_mm3`, `flair_wm_median`, `flair_wm_mad`, `wmh_threshold`; acquisition `flair_3d`,
`series_desc`, `n_series`, `shape`, `voxel_mm`, `slice_mm`, `manufacturer`, `model`, `tr_s`, `te_s`, `ti_s`;
`patno`, `acquisition_date`, `fs_image_id` (the T1 used; `manifest` fails QC on a mismatch), `error`. `manifest.QC["flair"]`: `reg_flair_t1_mi < -0.2`, `wm_mm3 > 200000`, `flair_wm_mad > 0`.
FastSurfer's T1-based `vol_WM_hypointensities` is the sanity reference. Tests: `tests/test_flair.py`.

## Manifest, feature assembly and QC galleries

```python
from pie.imaging.manifest import QC, build_manifest, assemble_features, feature_blocks

man = build_manifest("<derived>")        # one row per subject: which session of each modality, dates, batch, QC
df = assemble_features("<derived>")      # manifest + FastSurfer IDPs + modality features, QC-failed values blanked
blocks = feature_blocks(df.columns)      # {"dat": [...], "dwi": [...], "nm": [...], "flair": [...]}
```

`build_manifest(derived_dir, modality_dirs=None)` reads `<derived>/fastsurfer_idps.csv` (per subject the
earliest session passing `QC["t1"]`, else the earliest), `datscan_full/datscan_sbr.csv` + `spect_index.csv`,
and `dwi/`, `nm/`, `flair/` feature and index CSVs. `modality_dirs={"dat": path, "dwi": path, …}` relocates any of
the four (`manifest.DEFAULT_DIRS` holds the `datscan_full` default); rows with an `error` are dropped. Columns:
`PATNO`, `t1_image_id`, `t1_date`; per modality `<mod>_date`, `<mod>_batch`, `<mod>_qc_pass`, `<mod>_days_from_t1`;
`dat_image_id`; for DWI/NM/FLAIR also `<mod>_date_source` (`recorded` from the row's `acquisition_date`, else
`index_inferred`: the date with the most files among the index's `selected` series) and, when present,
`<mod>_fs_image_id` (and `dat_fs_image_id`), `<mod>_processing_version`, `<mod>_source_image_ids`, `fw_method`,
`dwi_topup`, `dwi_denoised`, `dwi_bvecs_rotated`. If a modality row records a different FastSurfer T1 than the
manifest's, `<mod>_t1_mismatch` is set and its QC fails; this includes DaTscan, whose stored transform refers to
that T1.

`assemble_features(derived_dir, modality_dirs=None)` adds the IDPs (all blanked with `t1_qc_pass=False` when a
key subcortical label is empty, i.e. FastSurfer failed), `dat_sbr_{caudate,putamen}_{l,r}`, `dwi_*` tensor and
tract features (`manifest.is_dwi_feature`), `nm_*_cnr`, and `flair_wmh_*`. A modality failing QC is blanked
entirely, so a failed left-side metric cannot let a valid-looking right side through.

The QC rules live in `manifest.QC` (one function per modality: `t1 dat dwi nm flair`) so studies and galleries
agree on "pass". The batch columns are what block-wise ComBat should use: `dwi_batch` = vendor + shells +
free-water method, `nm_batch` = vendor + voxel size, `flair_batch` = vendor + 2D/3D, `dat_batch` = vendor +
camera model. Harmonising diffusion or neuromelanin features by the *T1* scanner is the mistake this avoids.

QC galleries (`qc.py`) render three orthogonal views per subject from `--keep-nifti` outputs: DWI FA with SN
(red), striatum (cyan), thalamus (magenta); NM mean slab with refined SN (red), atlas SN (cyan), reference (lime);
FLAIR on the T1 grid with the WMH mask; DaTscan with the labels of the row's `fs_image_id` (older rows: the earliest
finished T1, as `datscan` picks it) through the stored transform.

```bash
venv_imaging/bin/python -m pie.imaging.qc --work-dir <derived>/dwi --modality dwi --out <derived>/qc/dwi --n 40 --worst reg_b0_t1_mi
venv_imaging/bin/python -m pie.imaging.qc --work-dir <derived>/datscan_full --modality datscan --out <derived>/qc/datscan \
    --sessions <derived>/sessions.csv --fastsurfer-dir <derived>/fastsurfer --worst reg_metric
```

| Flag | Default | Meaning |
|---|---|---|
| `--work-dir` | required | Modality work dir (reads its features CSV) |
| `--modality` | required | `dwi`, `nm`, `flair` or `datscan` |
| `--out` | required | Output dir: `<patno>.png` per subject and `gallery.png` (first 16) |
| `--n` | 40 | Subjects to render |
| `--worst` | random sample | QC column to order worst-first (`qc.worst_first`): negative-MI, motion and rotation columns largest first; `reg_metric` weakest \|correlation\| first; other columns smallest first |
| `--patnos` | none | Text file of PATNOs |
| `--sessions`, `--fastsurfer-dir` | none | Required for `datscan` |

Python: `montage(base, contours, out_png, title="", zoom=None)`, `render_subject(modality, subj_dir, out_png,
fastsurfer_dir=None, row=None)`, `gallery(pngs, out_png, cols=4)`. Tests: `tests/test_manifest.py`,
`tests/test_qc.py`.

## Whole-image models

### 3D CNN baseline (`cnn.py`)

Answers "could a network reading the whole image find a pattern the region features miss?".
`load_volume(fastsurfer_dir, image_id, shape=SHAPE)` masks FastSurfer's `orig_nu.mgz` with `mask.mgz`, z-scores
inside the brain, crops a 176 x 192 x 176 mm box around the brain centroid and mean-pools to 2 mm
(`SHAPE = (88, 96, 88)`); `cache_volumes(fastsurfer_dir, image_ids, cache_path, shape=SHAPE, loader=None)` stores
all subjects once as a float16 memmap (~1.5 MB each) with the id list in `<cache>.ids`. `sfcn(channels=(32, 64,
128, 256, 256), dropout=0.5)` is the SFCN of Peng et al. 2021 with one logit.
`cross_validate(X, y, groups, n_splits=5, repeats=1, epochs=30, device="cuda", seed=0, val_frac=0.15, log=print,
model_fn=sfcn, batch=8, lr=1e-3, patience=8)` returns patient-grouped stratified out-of-fold probabilities
(inner split for early stopping on AUROC, AdamW, weighted BCE, left-right flips, random shifts, mixed precision).
`cross_validate_fusion(X, y, groups, demographics, n_splits=5, repeats=1, inner_splits=3, …)` returns (CNN,
demographics, fusion) OOF probabilities with the CNN scores for the meta-model computed only from inner folds:
never fit a meta-model on globally computed OOF scores, whose training features depend on test labels.
`--pretrained` fine-tunes Peng et al.'s UK Biobank brain-age SFCN (`sfcn_pretrained(weights="default")`, weights
from `embed.WEIGHTS`) on 1 mm MNI volumes (`load_volume_mni`, `MNI_SHAPE = (160, 192, 160)`, 9.8 MB per
subject as float16; batch 4, lr 1e-4, patience 6). A network trained from scratch on ~700 subjects (the study's
labelled T1 set, September 2026 snapshot) is a weak reviewer baseline; the pretrained backbone is the fair one. The
study driver `dl_baseline.py [--pretrained]` (study code, not in PIE) compares the CNN, demographics / genetics and
their late fusion on the same subjects, with PD-vs-HC as the positive control.

```bash
venv_imaging/bin/python -m pie.imaging.cnn --labels labels.csv --fastsurfer-dir <derived>/fastsurfer \
    --cache vol.npy --out oof.csv [--folds 5] [--repeats 1] [--epochs 30] [--device cuda] [--pretrained]
```

`--labels` needs `PATNO`, `IMAGEID`, `y` (0/1; rows with missing `y` dropped); `--cache` is created if missing;
`--out` is the labels table plus `p_cnn`. The out-of-fold AUROC is printed. Tests: `tests/test_cnn.py`,
`test_nested_cnn_fusion_never_trains_on_outer_test` in `tests/test_imaging_audit.py`.

### Pretrained-model embeddings (`embed.py`)

Frozen representations from three open-weight models, so the tabular pipeline can test whether a generic image
embedding carries information beyond the region features. `to_mni(fastsurfer_dir, image_id, shape, origin)`
resamples the brain-masked `orig_nu.mgz` with the cached T1 → MNI affine (`dwi.register_t1_to_mni`, ~6 s when not
cached) onto the 1 mm grid each model was trained on (`GRID[backend]`); `embed_<backend>(volume, net)` reproduces
the authors' voxel order and intensity normalisation:

| Backend | Model | Input | Output columns |
|---|---|---|---|
| `brainiac` | Tak et al. 2026, MONAI ViT-B/16 | 170 x 206 x 162 mm box, LAS, trilinear to 96^3, z-score of nonzero voxels | `emb_brainiac_0..767` = token 0 of the last layer (the checkpoint has no CLS token, so this is the first patch token) |
| `simclr` | Kaczmarek et al. 2025, 3D ResNet-18 | ICBM 2009c box as (z, y, x) = 150 x 192 x 192, masked z-score | `emb_simclr_0..511` |
| `sfcn` | Peng et al. 2021, UK Biobank brain age | FSL MNI152 182 x 218 x 182, LAS, `x / x.mean()`, centre crop 160 x 192 x 160 | `emb_sfcn_0..63` + `brainage_sfcn` (expected age over bins 42-82) |

Deviations from the authors' pipelines: affine instead of rigid alignment for BrainIAC and SimCLR, the FastSurfer
mask instead of HD-BET / SynthStrip, nilearn's MNI152NLin2009cAsym affine target for all three, no WhiteStripe
(redundant under the masked z-score). Weights: BrainIAC research-only licence, SimCLR MIT, SFCN MIT; a backend
without weights is skipped with a message (`load_net(name, device="cuda")` raises `FileNotFoundError`).

```bash
venv_imaging/bin/python -m pie.imaging.embed --fastsurfer-dir <derived>/fastsurfer --ids-csv dataset.csv \
    --out emb.csv [--backends brainiac simclr sfcn] [--device cuda] [--workers 4] [--limit N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--fastsurfer-dir` | required | FastSurfer subjects dir |
| `--ids-csv` | required | CSV with an `IMAGEID` column |
| `--out` | required | Merged CSV (`IMAGEID` + columns of every backend) |
| `--backends` | all three | Subset of `brainiac simclr sfcn` |
| `--device` | `cuda` | Torch device |
| `--workers` | 4 | CPU processes for the MNI resampling |
| `--limit` | none | At most N not-yet-embedded subjects |

`run(fastsurfer_dir, image_ids, backends, out_csv, device="cuda", log=print, workers=4)` does one backend at a
time, appending to `<out stem>_<backend>.csv` (resumes from it) and then joins the parts on `IMAGEID`. Roughly
5-8 s per subject for all three; peak ~3.3 GB GPU (SimCLR at 1 mm). Tests: `tests/test_embed.py`.

## Shared infrastructure

### Modality runners (`batch.py`)

`dwi`, `nm` and `flair` share one runner shape: `index_series(zips)` (one row per series: `zip`, `prefix`,
`patno`, `desc`, `date`, `image_id`, `n_files`) → `load_index(index_file, zips, flag_fn)` (cached index with a
`selected` column) → `fastsurfer_by_patno(sessions_csv, fastsurfer_dir, require="stats/aseg+DKT.stats")` (earliest
finished T1 per PATNO) → `session_rows(group)` (the acquisition date with the most DICOM files) →
`run_batch(jobs, job_fn, out_csv, workers=4, log_every=10, pid_file=None)`. `run_batch` appends one row per
subject as it finishes, so a killed run loses at most the subjects in flight; `job_fn` must return a row with an
`error` string rather than raise; when a later row brings new columns the file is rewritten with the union
header. `done_subjects(out_csv, retry_errors=False)`, `filter_jobs(jobs, patnos_file=None, limit=None)` and
`add_common_args(ap)` (the shared CLI flags) complete it. `datscan`, `qc --modality datscan` and `dwi_refine` use
the same `fastsurfer_by_patno` rule, and `datscan` writes its table through `run_batch`. `batch.convert_series(zip_path, prefix, out_dir)`
extracts only `.dcm` members and returns every NIfTI dcm2niix wrote. Tests: `tests/test_batch.py`.

### Bundled atlas (`atlases.py`)

`pie/imaging/data/atlases/CIT168_v1_MNI152NLin2009cAsym_det25.nii.gz` + `.json`: CIT168 v1.0.0 (Pauli, Nili &
Tyszka 2018) in the authors' MNI152NLin2009cAsym 1 mm projection, reduced to a deterministic map (maximum-probability
label, one-based, where that probability >= 0.25). Labels 1-16: `Pu Ca NAC EXA GPe GPi SNc RN SNr PBP VTA VeP HN
HTH MN STH`. The JSON records source URL and SHA-256, file SHA-256, shape, affine, threshold, citation and the
correction date (2026-09-15); `setup.py` ships both files as package data.

| Function | What it does |
|---|---|
| `cit168_metadata()` | The JSON as a dict |
| `cit168_mni2009c(expected_space=MNI_SPACE)` | nibabel image after checking template space, file SHA-256, grid and label set; raises `ValueError` on any mismatch |
| `cit168_provenance()` | `atlas_space`, `atlas_version`, `atlas_sha256`, `atlas_probability_threshold` (written into NM rows) |

Nilearn's `fetch_atlas_pauli_2017` deterministic file is in native CIT168 space, not MNI152: never substitute it,
and never infer template identity from coordinates. `dwi.pauli_atlas()` returns this atlas for DWI, NM and
`nm_template`. Tests: `tests/test_atlas_space.py`.

### ZIP index cache and DICOM audit (`archives.py`, `dicom_audit.py`)

`ZipArchiveCache(max_open=1)` keeps up to `max_open` `zipfile.ZipFile` indexes open (LRU; a million-entry index
can exceed a gigabyte, so keep it small). Use it as a context manager and borrow with `with cache.open(path) as
z:`; leases may not overlap, and the cache is confined to its creating process and thread. It never caches
extracted bytes or disables CRC checks, and an archive changed while cached (including same-size, same-mtime inode
replacement) raises. `stats()` gives hits/misses/evictions. `benchmark_zip_index(path, prefix, repeats=3)` and
`benchmark_zip_workload(requests, capacities=(1, 2))` measure index residency only, not conversion.

`audit_archive_headers(archive, prefix, *, expected_count=None, archive_cache=None, max_members=20000,
max_member_bytes=64 MiB, progress=None)` reads every selected `.dcm` member once (ZIP CRC and SHA-256, no
extraction, no pixel decoding) and returns evidence for a conversion failure: per-member whitelisted header fields
(`dicom_audit.TAGS`), per-tag value summaries, duplicate SOP UIDs, bounded parser-warning examples, and flags
stating that nothing was repaired, no temporal order was assumed and `scientific_qc_pass=False`. `prefix` must
be a relative `…/` path; empty or oversized series raise. Tests: `tests/test_archives.py`,
`tests/test_dicom_audit.py`.

### Verified staging (`staging.py`)

For moving resumable work directories between caller-selected storage without trusting the copy:

- `snapshot_files(source, destination, files, *, max_bytes, guard=None, progress=None)` — copies an explicit
  `{relative_path: sha256}` manifest; each file is published (hard-link from `.partial`) only after its digest
  matches; existing destinations must match; interrupted partial files are kept and refused.
- `snapshot_tree(source, destination, *, max_bytes, guard=None, progress=None, include=None)` — copies a
  quiesced tree preserving bytes, timestamps, modes, symlinks (as links) and hardlinks, with a journal at
  `<destination>.snapshot.json` so an interrupted copy resumes only after byte verification; never deletes.
  Membership or metadata changes raise.

`guard()` is called between blocks (raise from it to stop). Writers must be quiesced first; an infrastructure
failure is not a scan exclusion. Tests: `tests/test_staging.py`, `tests/test_staging_files.py`.

## Using the IDPs in the PIE pipeline

```bash
python pie/pipeline.py --data-dir PPMI --output-dir <output> \
    --target-column COHORT --imaging-features <derived>/fastsurfer_idps.csv
```

The table enters the reduction/merge step as the `imaging` modality: `IMAGEID`, `SCAN_DATE` and text columns
(scanner strings) are dropped and the rest are prefixed `imaging_`. Harmonise scanner effects inside the
cross-validation folds (e.g. `endgame.preprocessing.ComBatHarmonizer(batch=…, covariates=[age, sex])`), never
on the full dataset before splitting. For multi-modality work use `manifest.assemble_features` and
[`pie.experiment`](experiment.md) instead.

## End-to-end example

```bash
bash scripts/setup_imaging.sh
D=Imaging/derived
venv_imaging/bin/python -m pie.imaging.run --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
    --ppmi-dir PPMI --work-dir $D --loni-csv Imaging/MRI_First_Study_9_07_2026.csv
venv_imaging/bin/python -m pie.imaging.dwi   --zips Imaging/<DTI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/dwi --fsl
venv_imaging/bin/python -m pie.imaging.nm    --zips Imaging/<MRI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/nm --keep-nifti
venv_imaging/bin/python -m pie.imaging.flair --zips Imaging/<MRI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/flair
venv_imaging/bin/python -m pie.imaging.qc --work-dir $D/nm --modality nm --out $D/qc/nm --worst reg_nm_t1_mi
```

```python
from pie.imaging.manifest import assemble_features, feature_blocks
from pie.imaging.labels import saa_labels
import pandas as pd

df = assemble_features("Imaging/derived")
sessions = pd.read_csv("Imaging/derived/sessions.csv", dtype={"image_id": str})
saa = saa_labels("PPMI", sessions)                                   # PATNO, IMAGEID, saa_positive, saa_match, ...
frame = df.merge(saa, left_on=["PATNO", "t1_image_id"], right_on=["PATNO", "IMAGEID"], how="left")
```

The pieces below run without external tools or PPMI data:

```python
import pandas as pd
from pathlib import Path
from pie.imaging.index import select_t1_series
from pie.imaging.features import build_idp_table

idx = pd.DataFrame({"zip": "<zip>", "patno": 1, "series_desc": ["MPRAGE", "MPRAGE_Repeat", "Localizer"],
                    "session": "2000-01-01_10_00_00.0", "image_id": ["IMG_A", "IMG_B", "IMG_C"],
                    "n_files": [176, 176, 3], "bytes": [9e7, 9e7, 1e5], "member_prefix": "p/"})
select_t1_series(idx)["image_id"].tolist()          # ['IMG_A']: 3D description, not a repeat, localiser dropped

stats = Path("fs/IMG_A/stats"); stats.mkdir(parents=True, exist_ok=True)
(stats / "aseg+DKT.stats").write_text("# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n"
                                      "  1  12  5000  5000.0  Left-Putamen\n"
                                      "  2  51  4800  4800.0  Right-Putamen\n")
sessions = pd.DataFrame({"patno": [1], "image_id": ["IMG_A"], "session_date": ["2000-01-01"],
                         "EVENT_ID": ["BL"], "protocol_phase": [1]})
idps = build_idp_table(sessions, "fs")
idps[["PATNO", "MaskVol", "vol_Left_Putamen", "sum_Putamen", "asym_Putamen"]]   # 1, 1500000.0, 5000.0, 9800.0, 0.0204
```

## FSL alternatives (`misc/`)

Standalone FSL loops over a BIDS-like `sub-*/anat/sub-*_T1w.nii.gz` layout; not used by `pie.imaging` (FSL on
`PATH`):

```bash
bash misc/run_first.sh  <dataset_dir> <output_dir>                    # -> <output_dir>/first_volumes.csv
bash misc/run_sienax.sh <dataset_dir> <output_dir> ["-f 0.2 -g 0.02"]  # -> <output_dir>/<subject>/report.sienax
```

`run_first.sh` runs `run_first_all` per subject and reads each of its 15 structures from the combined segmentation
`<prefix>_all_<method>_firstseg.nii.gz` with `fslstats -l <label-0.5> -u <label+0.5> -V`, writing
`Subject,Structure,Label,Volume_mm3`. `run_sienax.sh` runs `sienax` with the given BET options (default
`-f 0.2 -g 0.02`). Missing T1s and failed subjects are reported and skipped. Both are tested with stub FSL
binaries in `tests/test_imaging_regressions.py`, not against real FSL runs.

## Measurement follow-up safeguards (September 2026)

The September 2026 reassessment moved several safeguards into the package. The DWI ones (run concatenation keyed
on affine, PE, TE, TR, bandwidth and readout; gradient rotation; the opt-in `dwi_acquisition` and
`dwi_correction` helpers) are described in [imaging_dwi.md](imaging_dwi.md); the NM repeat-selection and
reference rules in [imaging_nm_datscan.md](imaging_nm_datscan.md).

### What is reusable in PIE, and what remains study-specific

Processing improvements are not evidence of improved SAA prediction. The completed T1/NM reassessments did not
establish incremental prediction benefit. Keep reusable measurement safeguards separate from the study's labels,
models and decision rules.

| Insight | PIE implementation | Integration status |
|---|---|---|
| Match the selected T1/session and avoid future/conflicting SAA labels | `batch.py`, `labels.py`, `manifest.py` | Existing pipeline safeguards; explicit opt-in for unmatched labels |
| Keep affine and acquisition timing/PE differences when assembling DWI | `dwi.py:assemble`, `acquisition_metadata_key` | Existing pipeline safeguards; estimated timing detects mismatches but cannot authorize eddy |
| Rotate FSL gradients in the correct physical frame after rigid motion | `dwi.py:rotate_bvec`, `preprocess` | Existing pipeline; oblique-grid regression fixtures |
| Separate NM reconstructions from independent acquisitions; preserve repeat support and hemisphere coordinates | `nm.py:compatible_repeats`, `average_repeats`, `nm_rois` | Existing pipeline; strict study eligibility thresholds remain study-specific |
| Keep one complete tensor acquisition and its own b0s; validate slice groups; preserve odd-sized grids | `dwi_correction.py` | New reusable opt-in selection/command/geometry helpers; no study imports |
| Inspect raw tensor eigenvalues and nonpositive signal before accepting FA/MD/AD/RD | `dwi_tensor_qc.py:tensor_measurements` | Opt-in measurement API already used by the reassessment, not silently substituted for legacy free-water features |
| Expose free-water optimizer status, residuals and raw tensor failures | `dwi_acquisition.py:diagnostic_multishell` | Development diagnostics only; positive-definite initialization repair is not a validated estimator |
| Verify atlas space, geometry, labels and exact bytes before nigral mapping | `atlases.py:cit168_mni2009c` | Shared DWI/NM default corrected on 2026-09-15; uses the authors' MNI2009c v1.0 projection |
| Reject failed or nonphysical multi-shell fits before eigenvalue clipping hides the failure | `freewater_qc.py:fit_multishell_checked` | Opt-in checked NLS API used by corrected control runs; no initialization repair or changed solver |

Corrected study controls also require both hemispheres, complete signal coverage of the new mask, and a single
tensor acquisition with its own b0 images. The NM recovery runner calculates geometric slab coverage from
transformed source-atlas points; the legacy API's resampled-volume ratio is not a bounded coverage fraction.
Anatomical correctness, repeat stability and biological validity remain separate checks. Physical fit acceptance
alone does not establish any of them.

The package helpers do **not** yet provide an end-to-end GPU cohort CLI. External process execution, memory
admission, restart handling, subject-level QC thresholds, and the frozen statistical comparisons remain in the
study. The current run keeps its original hashed implementation; it is not hot-swapped to these new helpers. The
maintained follow-up should call these package APIs from one configurable driver, rather than accumulate more
independent scheduling scripts. Historical script snapshots stay available for reproducibility; they are not
alternative recommended production entry points.

Storage locations and capacity policies are caller-configured; PIE does not require an external disk or a
particular filesystem driver. Use storage that supports the selected tools and has sufficient space for their
working files. Keep machine-specific mount, backup and recovery procedures in the deployment or study runner.
Verify copied data before retiring a working copy (`staging.py`), and do not move active writers' files. An
infrastructure failure is not a scientific scan exclusion or proof of successful recovery. Preserve original
acquisitions.

## Limitations

- Segmentation-only FastSurfer: volumes, no cortical thickness or surface area (needs the surface stream and a
  FreeSurfer licence). PPMI's own `FS7_APARC_CTH` tables can supplement. No eTIV.
- FastSurfer on CPU (`run --device cpu`) works but takes tens of minutes per scan.
- DWI: eddy-current and slice-outlier correction are not in the default path; susceptibility correction only
  where a reverse-PE b0 exists (`--fsl`). Single-shell free water is ill-posed.
- DaTscan: no attenuation correction by default, so SBRs are not on PPMI's scale without a calibration step.
- QSM is not wrapped.

## Tests

```bash
venv_imaging/bin/python -m pytest -q tests/test_imaging.py tests/test_imaging_audit.py tests/test_archives.py \
    tests/test_atlas_space.py tests/test_batch.py tests/test_cnn.py tests/test_datscan.py tests/test_dicom_audit.py \
    tests/test_dwi.py tests/test_dwi_correction.py tests/test_embed.py tests/test_flair.py tests/test_freewater_qc.py \
    tests/test_manifest.py tests/test_nm.py tests/test_nm_template.py tests/test_qc.py tests/test_staging.py \
    tests/test_staging_files.py tests/test_imaging_regressions.py
```

`tests/test_imaging_regressions.py` holds one regression test per bug fixed after the September 2026 audit
(DaTscan header, T1 choice, manifest dates and DaTscan lineage, NM refeature, carrier coding, defaults, device
budgets, tool-path overrides, the `misc/` scripts).

All synthetic (phantoms and generated tables): no PPMI data, GPU, FastSurfer, FSL or MRtrix3 needed, but DIPY
and nilearn must be installed. `tests/test_embed.py::test_real_weights_load` is skipped unless the weights are
under `third_party/weights/`. Self-checks:
`python -m pie.imaging.index`, `python -m pie.imaging.fastsurfer`.
