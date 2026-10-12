# Imaging layer (`pie/imaging`)

Turns raw PPMI imaging downloads from LONI (zipped DICOM) into per-subject or per-visit tables of
imaging-derived phenotypes (IDPs) that join the tabular PIE data on `PATNO` / `EVENT_ID`.

```
LONI MRI zip(s) ──index──► one T1w per session ──dcm2niix──► NIfTI ──FastSurfer (GPU)──► fastsurfer_idps.csv
                                                                          │ (T1 anatomy, labels, MNI affine)
          DTI zips ──dwi──────────────► dwi/dwi_features.csv ◄────────────┤
     full-MRI zips ──nm / flair───────► nm/nm_features.csv, flair/flair_features.csv ◄┤
  SPECT projections ──datscan──────► datscan_2026_10/datscan_sbr.csv ◄────┤
   AV-133 PET zips ──pet─────────────► pet/pet_av133.csv ◄───────────────┘
                           manifest.assemble_features ──► one wide row per subject (QC-failed values blanked)
```

The T1 run comes first: every other modality uses the subject's FastSurfer segmentation as anatomy and its
`sessions.csv` to find it. `<derived>` below is the `--work-dir` of `pie.imaging.run` (default `Imaging/derived`).

| Page | Covers |
|---|---|
| this page | setup, T1 / FastSurfer IDPs, labels, FLAIR, manifest and QC galleries, CNN and embeddings, shared infrastructure |
| [imaging_dwi.md](imaging_dwi.md) | `dwi`, `fba`, `dwi_refine`, `dwi_tensor_qc`, `freewater_qc`, `dwi_acquisition`, `dwi_correction`, `mrtrix_shim` |
| [imaging_nm_datscan.md](imaging_nm_datscan.md) | `nm`, `nm_template`, `datscan` |
| [fmriprep.md](fmriprep.md), [brain_viewer.md](brain_viewer.md), [fmri_viewer.md](fmri_viewer.md) | fMRI (`fmri*.py`, `xcpd.py`) and the viewers |

## Modules

| Module | What it does | CLI |
|---|---|---|
| `index.py` | List series inside LONI zips without extracting; pick one T1w per session; read LONI collection CSVs and IDA metadata | self-check |
| `convert.py` | Extract one series, run dcm2niix, keep NIfTI + JSON sidecar | |
| `link.py` | Map a session date to a PPMI `EVENT_ID` | |
| `fastsurfer.py` | FastSurferVINN segmentation (GPU), N4 + segstats (CPU), `.stats` parser | self-check |
| `features.py` | Wide IDP table from FastSurfer stats; tissue volumes, intracranial volume and head-size adjustment from the segmentation image | |
| `run.py` | Resumable T1 → IDP CLI | yes |
| `labels.py` | DaTscan-deficit and SAA labels and covariates aligned to an MRI session | |
| `batch.py` | Shared plumbing for the modality runners | |
| `dwi.py`, `fba.py`, `dwi_refine.py` | Diffusion pipeline, nigrostriatal and JHU tract measures, SyN refinement ([page](imaging_dwi.md)) | yes |
| `dwi_tensor_qc.py`, `freewater_qc.py`, `dwi_acquisition.py`, `dwi_correction.py` | Opt-in DWI measurement and correction safeguards ([page](imaging_dwi.md#opt-in-measurement-apis)) | |
| `nm.py`, `nm_template.py`, `nm_native.py` | Neuromelanin MRI: contrast ratios, template and territory contrasts, native-space SNc volume (Langley method) and snceg segmentation ([page](imaging_nm_datscan.md)) | yes |
| `datscan.py` | DaTscan SPECT reconstruction and SBRs ([page](imaging_nm_datscan.md#datscan-spect-datscanpy)) | yes |
| `pet.py` | PET: [18F]AV-133 (VMAT2) striatal SBRs against occipital and cerebellar cortex; SV2A and PI-2620 tau SUVRs (`--tracer`); frame timing and motion QC, time-weighted mean, MI registration to the T1, agreement with PPMI ([section](#av-133-vmat2-pet-petpy)) | yes |
| `flair.py` | White-matter hyperintensity burden | yes |
| `manifest.py` | Per-subject manifest, QC rules, assembled feature table | |
| `freesurfer.py` | Environment and runner for FreeSurfer tools that FastSurfer's eTIV, surface and longitudinal stages call | |
| `normative.py` | Deviation (z) scores against a reference group, e.g. PPMI controls | |
| `cohorts.py` | One integer participant key across cohorts (PPMI PATNO, ADNI RID range, hashed keys for other LONI projects and BIDS) | |
| `harmonize.py` | ComBat (optionally with a spline age term), longitudinal ComBat, batch × label confounding report | |
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
matplotlib pytest` and the modelling libraries, and tries `endgame-ml[tabular]`. Of those, DIPY is what the
diffusion fits need, scikit-image the DaTscan reconstruction and the NM template masks, and nilearn and ANTsPy the
two optional deformable passes (`dwi_refine`, `nm_template`). The MNI target of the default registrations is
bundled with PIE, so no template is downloaded.

| Tool | Used by | How PIE finds it |
|---|---|---|
| dcm2niix | `convert`, `batch` (all modalities) | `$PIE_DCM2NIIX`, else `<repo>/venv_imaging/bin/dcm2niix` (`convert.DCM2NIIX`) |
| FastSurfer (segmentation only) | `fastsurfer`, `run` | `$PIE_FASTSURFER_HOME`, else `<repo>/third_party/FastSurfer`; run with `$PIE_FASTSURFER_PYTHON`, else `<repo>/venv_imaging/bin/python` |
| NVIDIA GPU, >= 6 GB | FastSurferVINN; `cnn`, `embed` | `--device` on `run`, `cnn`, `embed` (default `cuda`; `cpu` works, much slower) |
| FSL `topup`/`applytopup` | `dwi --fsl` | `$FSLDIR`, else `~/fsl` |
| MRtrix3 3.0.x | `dwi --fba`, `fba`; `dwidenoise`/`mrdegibbs` for `--denoise` if present | `PATH` |
| ANTsPy | `dwi_refine`, `nm_template` | Python import `ants` |
| FreeSurfer 7.4.1 + license (optional) | `run --etiv`, `run --long`, `fastsurfer.surfaces` | `$PIE_FREESURFER_HOME` (a path without spaces: FreeSurfer's scripts break on them, so link an install on a drive like "Seagate Portable Drive" to e.g. `~/freesurfer-7.4.1`), `$PIE_FS_LICENSE` (else `$FS_LICENSE`); `talairach_avi` also needs `/bin/tcsh` |
| TrUE-Net + FSL `fsl_anat` | `flair` (default `--method truenet`) | `truenet` 2.0.2 and `prepare_truenet_data` in `venv_imaging` (Apache-2.0); weights from `$TRUENET_PRETRAINED_MODEL_PATH`, else `$PIE_WEIGHTS_DIR/truenet`, else `<repo>/third_party/weights/truenet` (FSL licence, non-commercial); FSL from `$FSLDIR`, else `~/fsl` |
| Pretrained weights | `embed`, `cnn --pretrained` | `$PIE_WEIGHTS_DIR`, else `<repo>/third_party/weights`, then `<backend>/`; sources and checksums in `third_party/weights/WEIGHTS.md` |

The defaults are relative to the source tree; set the `PIE_*` environment variables (read at import) to run from an
installed package or with tools elsewhere. The default T1 run needs no FreeSurfer licence: it uses only FastSurfer's segmentation stream (no surfaces, so no
cortical thickness); `run --etiv`, `run --long` and `fastsurfer.surfaces` do need FreeSurfer and its licence. Disk: roughly 20 GB per 1,000 T1 scans for NIfTI + segmentations.

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
| `--zips` | required | LONI zip(s), laid out `PPMI/<PATNO>/<SeriesDescription>/<yyyy-mm-dd_HH_MM_SS.0>/<IMAGE_ID>/*.dcm` |
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
   `index.csv`; a zip added later (a follow-up download into the same work dir) is indexed and appended, and
   `sessions.csv` is rebuilt whenever the T1 selection it holds no longer matches the zips (it is kept as it is
   when a zip is not on disk, e.g. a backfill with the download drive unmounted).
2. **Choose the T1** — `select_t1_series(index)` drops localisers, calibration and non-T1 descriptions (`_NOT_T1`),
   then per participant and visit day (LONI timestamps every series separately, so an MPRAGE and its repeat minutes
   later are one visit; before 11 Oct 2026 they were two sessions, 29 subject-days in the September 2026 tables) ranks 3D/MPRAGE-family descriptions up, 2D/axial ones down (some sites label a
   2-frame axial T1 as the only "T1", which FastSurfer cannot use) and repeats down. Where a visit has more than one
   candidate (`t1_candidates`), one DICOM header per candidate is read from the zip (`run.t1_headers`, cached in
   `t1_headers.csv`) and `header_quality` adds +2 for a 3D acquisition (-2 for 2D), +1 for an `ORIGINAL` image (-1 for
   derived, reformatted, MPR or projection images), +1 for an inversion time, and -1 for in-plane spacing under
   0.75 mm. Ties are then broken by bytes, which also handles single-file multi-frame DICOM; before 11 Oct 2026 bytes
   alone decided, which favoured 512-matrix interpolated reconstructions. Sessions dated 9999 (LONI date masking)
   are kept with `date_masked=True`.
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
   chunk in one process behind a file lock (`.gpu.lock`: one inference on the GPU at a time). On CUDA a chunk is
   killed after `STALL_SECONDS` (420 s) without a new segmentation. Its wall cap is only a backstop:
   `STALL_SECONDS` per scan + 300 s.
   - A thermally throttled laptop GPU (95 °C, 300 MHz) took about 4 min per scan.
   - The earlier cap of 90 s per scan killed such batches mid-scan while they were still progressing.

   A single-scan
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

Three optional stages use FreeSurfer tools through FastSurfer (FastSurfer stays the segmentation engine; FreeSurfer's
own `recon-all` is never run):

| Stage | What it adds | Cost |
|---|---|---|
| `run --etiv` | FreeSurfer eTIV for every finished subject: FastSurfer's talairach registration (`talairach_avi` on `orig_nu`, the `--tal_reg` step; `fastsurfer.talairach_etiv`), eTIV = 1948106 / det(`talairach.xfm`); then the table is rebuilt with an `eTIV` column | ~2 CPU-min per scan |
| `run --surfaces` | FastSurfer's surface stream (`fastsurfer.surfaces`, `--fsaparc`) for every finished subject, then the table is rebuilt with PIE's own DK thickness and area, `cth_{lh,rh}_<region>`, `area_{lh,rh}_<region>`, `cth_{lh,rh}_mean` (`features.thickness_columns`). One pinned FastSurfer version for every scan, unlike mixing FS7 releases, and the regions match PPMI's `fs7_cth_*` for agreement checks | ~1 CPU-hour per scan |
| `run --sclimbic` | FreeSurfer 7.4.1 ScLimbic (`fastsurfer.sclimbic`; Greve 2021): `vol_sclimbic_*` for the **basal forebrain** (the nucleus basalis of Meynert / Ch4 region; smaller NbM was the best-replicated SAA-positivity imaging marker, Spotorno/Wisse 2024), hypothalamus, fornix, mammillary bodies, septal nuclei and accumbens, plus `sclimbic_qa_z_max` (largest QA z against the tool's population) | ~1 CPU-min, ~3.3 GB per scan |
| `run --subfields` | Hippocampal subfields and amygdala nuclei (`fastsurfer.hippocampal_subfields`: FreeSurfer's Python `segment_subregions hippo-amygdala`, no MATLAB): `vol_hsf_{lh,rh}_*`, `vol_amyg_{lh,rh}_*`. Needs `--surfaces` first (norm, aseg, wmparc) | ~20–40 CPU-min per scan |
| `run --synthseg` | SynthSeg-robust on the raw T1 (`fastsurfer.synthseg`; Billot 2023): `synthseg_*` volumes and `synthseg_qc_min` (below ~0.65 is suspect). The robustness path for legacy 1.5 T and thick-slice scans (the 2012 prodromal cohort) and a cross-check of FastSurfer | ~2 CPU-min per scan |
| `run --long` | FastSurfer's longitudinal stream (within-subject template) for every subject with >= 2 finished, real-dated sessions, into `<work>/fastsurfer_long/` (`run.long_timepoints`, `fastsurfer.longitudinal`), then `fastsurfer_long_idps.csv` (`run.write_long_idps`: same columns, `stream=long`, `long_template`) | hours per subject, GPU |

Work dir: `index.csv`, `sessions.csv`, `scan_metadata.csv` (`image_id`, `nifti`, `sidecar` + `SIDECAR_FIELDS`),
`nifti/<PATNO>/`, `fastsurfer/<IMAGE_ID>/{mri,stats,scripts}`, `failures.csv` (headerless: patno, image_id,
message), `fastsurfer_idps.csv`.

### IDP columns (`fastsurfer_idps.csv`, one row per processed session)

| Column | Meaning |
|---|---|
| `PATNO`, `EVENT_ID`, `IMAGEID`, `SCAN_DATE`, `protocol_phase` | Identifiers; `protocol_phase` = 1 for the PPMI clinical protocol (scans before 2020), 2 for PPMI 2.0 (2020 on), missing for masked dates (`index.protocol_phase`; until 11 Oct 2026 it was the zip's position in `--zips`) |
| `Manufacturer`, `ManufacturersModelName`, `MagneticFieldStrength`, `SoftwareVersions`, `InstitutionName`, `RepetitionTime`, `EchoTime`, `InversionTime`, `FlipAngle`, `SliceThickness` | dcm2niix sidecar (`features.META_COLS`); scanner batch for harmonisation |
| `MaskVol`, `BrainSegVol`, … | Global measures whose short name ends in `Vol` keep that name; other measures become `vol_<name>` |
| `vol_<Structure>` | Regional volume in mm^3, name with non-alphanumerics replaced by `_` (`vol_Left_Putamen`, `vol_ctx_lh_precuneus`, `vol_WM_hypointensities`) |
| `sum_<S>`, `asym_<S>`, `ai_<S>` | Left + right, (L − R) / (L + R), and the ENIGMA asymmetry index (L − R) / ((L + R) / 2) (twice `asym_`) for `Putamen Caudate Pallidum Thalamus Hippocampus Amygdala Accumbens_area Lateral_Ventricle Cerebellum_Cortex Cerebellum_White_Matter VentralDC` |
| `sum_Ventricles` | Lateral + inferior-lateral + 3rd + 4th ventricles |
| `eTIV` | FreeSurfer eTIV from `mri/transforms/talairach.xfm` when `run --etiv` has made it, else NaN |

eTIV is optional: the default segmentation-only run does not compute it; `run --etiv` adds it afterwards (below). Do not normalise by
`MaskVol` or `BrainSegVol` — the first is a dilated brain mask, not an intracranial measurement, and the
second shrinks with atrophy, so dividing by it removes part of the effect you are trying to measure.
Adjust with `features.adjust_for_head_size` and a true intracranial volume: FastSurfer's `--tal_reg`
eTIV if you ran it that way, or PPMI's FreeSurfer 7 eTIV (`EstimatedTotalIntraCranialVol` in
`FS7_ASEG_VOL_*.csv`, baseline scans, keyed on `PATNO` + `EVENT_ID`); `features.tiv_from_registration` is a weak
fallback (below).

### Tissue volumes and head-size adjustment (`features.py`)

`build_idp_table` above reads FastSurfer's `.stats` text. These functions work on the segmentation
image itself, for the whole-tissue volumes the literature reports (CAT12-style grey and white matter
totals adjusted for intracranial volume, e.g. Droby et al. 2025).

```python
import nibabel as nib, numpy as np
from pie.imaging import features

seg = np.zeros((10, 10, 10), np.int32)
seg[0:2] = 3          # left cerebral cortex
seg[3] = 2            # left cerebral white matter
seg[5, 0, 0:3] = 11   # left caudate
out = features.tissue_volumes(nib.Nifti1Image(seg, np.diag([2, 2, 2, 1])))   # 2 mm voxels = 8 mm^3

out["cortical_gm"], out["white_matter"], out["caudate_l"]   # 1600.0, 800.0, 24.0 mm^3 (floats: 1599.99...)
out["total_gm"]                                             # 1624.0 = cortical + subcortical + cerebellar
```

Bigger heads have bigger structures, so compare volumes only after adjusting for intracranial volume,
and fit the adjustment on a reference group — controls, or the training fold — so that no evaluation
row informs it:

```python
rng = np.random.default_rng(0)
tiv = rng.normal(1.5e6, 1.2e5, 200)                     # mm^3
putamen = 0.004 * tiv + rng.normal(0, 200, 200)         # scales with head size
is_control = np.arange(200) < 100

adjusted = features.adjust_for_head_size(putamen, tiv, method="residual", reference=is_control)
np.corrcoef(putamen, tiv)[0, 1], np.corrcoef(adjusted, tiv)[0, 1]    # 0.91 -> -0.05
```

For analyses, `manifest.assemble_features` gives one intracranial volume per subject, `tiv_mm3`, with
`tiv_source`: PIE's own eTIV (`pie_talairach`, same T1 as the volumes), else PPMI's FreeSurfer 7 eTIV for the same
visit (`ppmi_fs7`, when `ppmi_dir` is passed), else missing (`none`). Adjust with that.

`features.tiv_from_registration(head_img, template_head, template_icv)` is superseded; it was the fallback when there was
no eTIV: it warps the TemplateFlow intracranial map into subject space by an affine head-to-head
registration and sums it, returning `qc_pass` beside the volume. The registration must be head to
head, skull included — pass FastSurfer's conformed `orig.mgz` or the raw T1, never a skull-stripped
image. Against FreeSurfer 7.3.2 eTIV from PPMI's `FS7_ASEG_VOL` table on 40 random baseline scans (25 September
2026) it correlated at r = 0.81 (0.82 for the 33 passing its QC) and read 9 % low, while FastSurfer's `MaskVol`
correlated at r = 0.85: in this early-disease cohort it is no better a head-size proxy than `MaskVol`. `features.fetch_template(cache_dir)`
downloads and sha256-checks the TemplateFlow maps it needs.

## Labels and covariates (`labels.py`)

Read the PPMI tabular download (latest matching file per table) and align outcomes to MRI sessions. `sessions`
is a frame with `patno`, `image_id`, `session_date`, `EVENT_ID` (e.g. `sessions.csv`).

| Function | Output |
|---|---|
| `covariates(ppmi_dir)` | Per PATNO: `COHORT`, `ENROLL_DATE`, `ENROLL_AGE`, `SEX`, `BIRTHDT`, `HANDED`, `LRRK2`, `GBA`, `SNCA`, `APOE`, `PATHVAR_COUNT`, `{LRRK2,GBA,SNCA}_carrier` (0 for `0`, `0.0`, `"0"`, `"0.0"` whatever the column dtype, 1 for any other value, NaN when blank or missing: untested is not a control), `APOE_e4` (count), and `PRS_GP2`, `PRS_META5_noLRRK2GBA` when `Polygenic_Risk_Scores_*.csv` exists |
| `dat_labels(ppmi_dir, sessions, threshold=0.65, max_months=18, source="auto", xing_threshold=None)` | Closest analysed DaTscan within 18 months: `PATNO`, `IMAGEID`, `months_to_datscan`, `days_datscan_after_mri` (signed: positive when the DaTscan came later, which for a converter can carry a later disease state), `DATSCAN_DATE`, `DATSCAN_EVENT_ID`, `DATSCAN_{CAUDATE,PUTAMEN}_{R,L}`, `sbr_putamen_min`, `sbr_caudate_min`, `sbr_putamen_mean`, `sbr_pc_ratio_{l,r}` (putamen/caudate), `sbr_ai_{putamen,caudate}` (\|L − R\| / mean x 100), `sbr_pct_expected` (lowest putamen SBR / the age + sex expectation fitted on visually-negative healthy controls), `sbr_putamen_min_z` (same, as a z-score against the controls' residual SD), `dat_visual` (PPMI visual read, 1/0), `dat_deficit_sbr` = `sbr_pct_expected <= threshold` (0.65: PPMI prodromal / PARS rule, Jennings et al. 2017, doi:10.1001/jamaneurol.2017.0985, derived on Invicro's occipital scale; 0.65–0.80 is their indeterminate band; Xing rows use `xing_threshold` when given), `dat_deficit_threshold` (the cut-off the row used), `dat_d_nsdiss` = `sbr_pct_expected <= 0.75` (the NSD-ISS D+ rule as applied in PPMI, Dam et al. 2024, doi:10.1038/s41531-024-00789-w), `sbr_source` |
| `xing_deficit_threshold(ppmi_dir, invicro_threshold=0.65)` | The Xing percent-of-expected cut-off that calls as many deficits as `invicro_threshold` does on Invicro, on the scans both labs analysed: `threshold`, `n_overlap`, `n_deficit`, `invicro_rate`, `kappa` |
| `dat_sbr_scored(ppmi_dir, source)`, `percent_expected(sbr, value)`, `kappa(a, b)` | One table's scans with covariates and `sbr_pct_expected` / `sbr_putamen_min_z`; the control-norm ratio and z-score; Cohen's kappa |
| `dat_sbr_table(ppmi_dir, source="auto")` | PPMI's SBR table and visual reads. `xing`: `Xing_Core_Lab_-_Quant_SBR_*` (PPMI's primary SBRs since December 2024, all earlier scans re-analysed; cerebral white-matter reference; sub-regional `*_REF_CWM` columns kept); `invicro`: the archived `DaTScan_SBR_Analysis_*` (occipital reference, nothing after 1 December 2024); `auto` = Xing when present. The scales differ: never pool them. 99mTc-TRODAT-1 scans are dropped |
| `pet_labels(ppmi_dir)` | PPMI's `PET_SBR_Analysis`, one row per scan: `tracer` and `pipeline` columns that must never be pooled across, `PET_SCAN_DATE` (month), `EVENT_ID`, `analysed`, `sbr_<region>`, `cb_occ`, `sbrcb_<region>`; see [the PET section](#av-133-vmat2-pet-petpy) |
| `saa_labels(ppmi_dir, sessions, allow_unmatched=False)` | `PATNO`, `IMAGEID`, `SAA_EVENT_ID`, `SAA_Status` (`Conflicting` when calls at the chosen visit disagree), `SAA_Type`, `saa_positive` (NaN unless one Positive/Negative call), `saa_match` (`same_visit`, `screening_baseline_pair`, `unmatched_visit`), `saa_visit_concurrent` |

On the September 2026 download the Xing table labels 1,447 MRI sessions (1,398 subjects) where the archived Invicro table labels 988. The Xing release lacks about 716 PPMI-1 participants whom Invicro analysed (504 PD, 136 HC, 76 SWEDD).
- `auto` therefore takes the Xing match where one exists and the Invicro match otherwise, per session. Each table keeps its own control norm, and `sbr_source` records which table was used.
- Raw SBRs cannot be pooled across the two, and neither can deficit labels at one shared cut-off.
- Refitting the expectation within each table does not make percent-of-expected scale-free (SBR = target / reference − 1, and the references differ). On the 2,963 scans both labs analysed (September 2026; PD, HC and SWEDD only, since Invicro has no prodromal values), 0.65 calls 88.9 % deficits on Invicro and 85.5 % on Xing. `xing_deficit_threshold` matches Invicro's rate at Xing 0.76 (κ 0.86 with Invicro's calls). Matched within one cohort the cut-off is 0.70 for HC (7 deficits in 246) and 0.78 for PD. On Xing's prodromal scans the cut-off decides a lot: 24 % of participants' first scans fall at or below 0.65, and 43 % at or below 0.76.
- `dat_labels` still applies 0.65 to Xing rows by default, so exported labels have not changed. Pass `xing_threshold` to change that, and record the choice: `dat_deficit_threshold` holds the cut-off each row used. The NSD-ISS 0.75 rule (`dat_d_nsdiss`) has the same caveat; `xing_deficit_threshold(ppmi_dir, 0.75)` gives 0.83.
- On Xing alone, PD + SWEDD visual reads fall to about 5 negatives. PPMI 2.0 eligibility used < 0.80 of an age-only expectation on the Invicro
scale, which `sbr_pct_expected < 0.80` approximates but does not reproduce. Tables are chosen by the release date in
their names (`viewer.catalog.latest_table`), not by text order, in which `18Mar2025` sorts after `08Sep2026`.

`saa_labels` uses the SAA of the MRI's own visit, treating SC and BL as one occasion. It never borrows a later
assay for an inconclusive concurrent visit: `RUNDATE` is when the laboratory ran the sample, not when CSF was
collected. `allow_unmatched=True` reproduces the historical fallback (earliest SAA visit) for an explicitly
labelled sensitivity analysis only. Tables read: `_Subject_Characteristics/{Participant_Status,Demographics,
iu_genetic_consensus,Polygenic_Risk_Scores}_*.csv`, `Imaging/DaTScan_{SBR_Analysis,Visual_Interpretation_Results}_*.csv`,
`Biospecimen/SAA_Biospecimen_Analysis_Results_*.csv`.

## FLAIR white-matter hyperintensities (`flair.py`)

A vascular covariate, not a synucleinopathy marker: WMH load confounds subcortical volumes and marks the vascular
mimic behind some normal DaTscans. In the study's September 2026 MRI download (snapshot) 1,340 subjects with a T1
have a FLAIR, 3D 1 mm or 2D 5 mm.

The default method is **TrUE-Net** (October 2026). The intensity threshold it replaces tracked age at only ρ 0.18
(validity table), below FastSurfer's T1-only `vol_WM_hypointensities`. TrUE-Net (Sundaresan et al. 2021, MedIA,
doi:10.1016/j.media.2021.102184) was trained on vascular WMH. It recovered the WMH–age relation in an independent
comparison (Strain et al. 2024, NeuroImage, doi:10.1016/j.neuroimage.2023.120494). LST-AI and SAMSEG lesions were
not chosen as primary because both are MS-trained. **TrUE-Net is not yet validated in PPMI** (plan below).

1. Series: `flag_flair` (`FLAIR|dark.?fluid|tirm`, excluding T1 FLAIR and repeats); `flag_flair_3d` = "3D" in
   the description or >= 100 files. 3D is preferred, then the largest series (`flair_3d` is recorded for
   harmonisation).
2. Both methods: `n4(img_sitk, shrink=2)` → `register_flair_to_t1(flair_sitk, t1_img, t1_mask_img)` (rigid MI to the T1
   *head* at 2 mm from two starts, the scanner header and the centre of mass, the better final pose winning; the
   reported `reg_flair_t1_mi` is the brain-masked MI at that pose) → FLAIR resampled onto the T1 grid. Until 25
   September 2026 the fixed image was the brain-masked T1 with a centre-of-mass start, which put partial-coverage 2D
   FLAIR (5 mm slices over the upper head) centimetres off: 177 of 1,338 scans failed QC, and their WMH volumes were
   inflated (median 11.1 mL against 2.5 mL once registered). On 30 of those scans the new registration passed QC for
   21; the other 9 look aligned in the gallery but stay below the MI threshold, so their QC rule is still under review.
3. `--method truenet` (default; `run_truenet`), TrUE-Net 2.0.2 run as distributed, on the FLAIR and FastSurfer's `orig.mgz`:

   ```bash
   prepare_truenet_data --FLAIR=<out>/flair --T1=<out>/t1 --outname=<out>/sub --nodistmaps
   truenet evaluate -i <out> -m mwsc -o <out>/pred
   ```

   - `prepare_truenet_data` is TrUE-Net's own preprocessing: `fslreorient2std`, BET and FAST bias correction of the
     FLAIR, `fsl_anat` on the T1, rigid FLIRT of T1 to FLAIR, and the `make_bianca_mask` white-matter mask. The
     distance maps it skips are only used for training.
   - Model `mwsc`: MICCAI WMH Challenge, FLAIR + T1, three scanners with 3 mm 2D and 3D FLAIR. `ukbb` (UK Biobank,
     one protocol) is also installed.
   - Lesions: the probability map masked by the WM mask, then >= 0.5 (`WMH_PROB`; the paper's post-processing,
     section 2.1.4). No minimum lesion size is applied, as in the paper. Voxels are counted on the prepared FLAIR grid.
   - Periventricular: within 10 mm of FastSurfer's lateral ventricles. The distance map is computed on the T1 grid
     and carried to the FLAIR grid by step 2's registration, which also supplies the QC metric. TrUE-Net's own FLIRT
     is used only for its input.
   - Runtime: `fsl_anat` (FNIRT to MNI, for the WM mask) dominates. On PPMI FLAIRs, sharing the CPU with five other
     jobs, a subject took 41–60 min, against 9–13 min with `--truenet-prep fastsurfer`; both include `truenet evaluate`
     on the CPU. `truenet` uses the GPU when there is one; `CUDA_VISIBLE_DEVICES=` forces the CPU.
   - `--truenet-prep fastsurfer` (`_prepare_from_fastsurfer`; rows record `truenet_prep`) skips `fsl_anat`. The
     FLAIR is prepared exactly as the script does it. The T1 is FastSurfer's N4-corrected `orig_nu` in its brain mask,
     with the same rigid FLIRT to the FLAIR. The WM mask is FastSurfer's cerebral WM plus WM hypointensities
     (aseg 2, 41, 77) instead of `make_bianca_mask`.
     - Comparison on 10 random PPMI participants (9 with 3D FLAIR), same FLAIRs (`truenet_prep_check.py`, 12 Oct
       2026): lesion-mask Dice median 0.875 (0.82–0.92). Volumes agree at r 0.993 and CCC 0.971. The FastSurfer
       preparation is smaller on all 10, by 53 mm³ on average (11 %; limits −126 to +20 mm³).
     - Use one preparation for a whole cohort. Resuming a work directory with the other one stops with an error.
   - Needs FSL (`$FSLDIR`, else `~/fsl`) and a work directory without spaces, because the FSL script breaks on them.
     Put it on the NVMe; the zips may stay on the Seagate.
   - Weights: `third_party/weights/truenet/{mwsc,ukbb}` from the FSL conda package `fsl-truenet_data` 2508.0
     (download commands and sha256 in `third_party/weights/truenet/README.md`). `$TRUENET_PRETRAINED_MODEL_PATH`,
     else `$PIE_WEIGHTS_DIR/truenet`, overrides the location. **Licence:** the truenet code is Apache-2.0; the weights
     are an FSL component under the FSL licence, so non-commercial use only.
4. `--method threshold` (`wmh(flair_t1, aseg, vox_mm=1.0)`): white matter = FastSurfer labels 2, 41, 77 eroded by one
   voxel; lesions = WM voxels brighter than median + `K_MAD` (3) x 1.4826 MAD of normal-appearing WM; components
   below `MIN_LESION_MM3` (5 mm^3) dropped; periventricular = within `PV_MM` (10 mm) of the lateral ventricles.

WMH-SynthSeg, the planned second opinion for 5 mm 2D FLAIR and for cohorts without FLAIR, is not wrapped.
`mri_WMHsynthseg` is not in FreeSurfer 7.4.1: it is in FreeSurfer's development tree and needs its 790 MB model
`WMH-SynthSeg_v10_231110.pth` in `$FREESURFER_HOME/models`.

```bash
venv_imaging/bin/python -m pie.imaging.flair --zips <full-MRI zips> --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/flair_truenet --workers 4 [--method threshold] [--keep-nifti]
# -> flair_features.csv, flair_index.csv; --keep-nifti adds <patno>/flair_t1.nii.gz, wmh_t1.nii.gz and, for TrUE-Net,
#    <patno>/truenet/ (TrUE-Net's inputs and outputs, including the unthresholded probability map)
```

`assemble_features(derived, modality_dirs={"flair": "<derived>/flair_truenet"})` reads the TrUE-Net table. Use
`--keep-nifti` only on subsets, because the disk is nearly full.

Flags: the shared `batch.add_common_args` set, see [the DWI flag table](imaging_dwi.md#cli), plus `--method`
(`truenet` default, `threshold`). One work directory holds one method: resuming into a directory written by the
other method stops with an error (rows from before `wmh_method` existed count as threshold rows).

Columns: `wmh_mm3`, `wmh_log_mm3` (log1p), `wmh_pv_mm3`, `wmh_deep_mm3`, `wmh_frac_wm` (WMH / eroded FastSurfer WM
volume), `wmh_n_lesions`. Method: `wmh_method`; TrUE-Net rows add `truenet_model`, `truenet_version`, `fsl_version`
and `wmh_prob_threshold`, threshold rows add `wmh_threshold`. QC: `reg_flair_t1_mi`, `wm_mm3`, `flair_wm_median`,
`flair_wm_mad`, `flair_si_mm` (S–I extent: slices x spacing along the axis nearest S–I). Acquisition: `flair_3d`,
`series_desc`, `n_series`, `shape`, `voxel_mm`, `slice_mm`, `manufacturer`, `model`, `tr_s`, `te_s`, `ti_s`. Also
`patno`, `acquisition_date`, `fs_image_id` (the T1 used; `manifest` fails QC on a mismatch) and `error`.

`manifest.QC["flair"]`: `reg_flair_t1_mi < -0.2`, `wm_mm3 > 200000`, `flair_wm_mad > 0`, `flair_si_mm >= 130`.
27 of 486 PPMI 2D slabs cover less than 130 mm S–I, which leaves white matter outside the slab; 3D scans all pass.
Tables written before October 2026 lack `flair_si_mm` and pass that rule. **QC pass rates differ by acquisition:**
72 % of 2D and 95 % of 3D FLAIRs passed (threshold-method table, September 2026). Blanked WMH is therefore not
missing at random: report it by `flair_batch` (vendor x 2D/3D), adjust for 2D/3D, and run a 3D-only sensitivity
analysis. FastSurfer's T1-based `vol_WM_hypointensities` is the sanity reference. Tests: `tests/test_flair.py`.

### TrUE-Net validation plan (before WMH is used as more than an exploratory covariate)

Fix the thresholds below before anyone traces or rates. Sample all subsets stratified by 2D/3D, vendor and TrUE-Net
volume tertile. Readers are blind to TrUE-Net output and diagnosis. `--keep-nifti` on these subjects (`--patnos`)
keeps the prepared FLAIR and the probability map.

1. **Manual tracing, 30–40 scans**, traced on the prepared FLAIR. Report voxel Dice and the volume ICC(2,1) with
   Bland–Altman, overall and by 2D/3D. Suggested pass: median Dice >= 0.6 and ICC >= 0.85 on 3D. Report 2D separately;
   a 2D failure restricts the covariate to 3D. Threshold sweeps on the saved probability maps are secondary analyses.
   They do not re-tune the 0.5 threshold.
2. **Fazekas ratings, about 150 scans** (periventricular and deep, 0–3), with a second rater on 30 for weighted κ.
   Report Spearman ρ of `wmh_pv_mm3` against the periventricular score and of `wmh_deep_mm3` against the deep score.
3. **Age as positive control**: Spearman ρ of `wmh_log_mm3` with age in QC-passed scans, overall and by 2D/3D. It
   should clearly exceed the threshold method's 0.18 and at least match FastSurfer's `vol_WM_hypointensities` (0.46).
   Also report the QC pass rate by 2D/3D under TrUE-Net.

## AV-133 (VMAT2) PET (`pet.py`)

[18F]AV-133 binds the vesicular monoamine transporter in dopaminergic terminals: a presynaptic striatal measure like
DaTscan. PPMI's `PET_SBR_Analysis` (September 2026) holds 237 AV-133 rows (233 analysed, 106 participants, 2012–2024)
plus 10 [18F]C05-05 and 9 [18F]DPA-714 rows. **Status: implemented and phantom-tested, not yet run: no AV-133 images
are downloaded, so agreement with PPMI is unmeasured.**

What PPMI did (IND processing methods, rev. 25 April 2014; PET Technical Operations Manual for protocol 004, v4.0 2022):
- 222 MBq bolus; 2 × 5 min frames from 80 ± 5 min after injection (median injection-to-scan 75–80 min per protocol in
  `PET_Acquisition_Metadata`); attenuation, scatter, randoms and decay corrected; iterative reconstruction in
  quantitative units, submitted without post-filtering.
- Frames checked for motion and corrected where necessary, time-weighted mean, co-registration to the MRI, MNI
  normalisation, IND's small striatal VOIs (caudate, anterior and posterior putamen) adjusted per subject; follow-ups
  registered to the baseline PET.
- SBR = target / occipital − 1 (`PET_OCCIP` is 0 by construction; the DPA-714 rows sit near 0).

The table has no method column. The source (a blank `PET_LIGAND` marks the legacy IND export) and `PROTOCOL` separate
three pipelines, which `labels.pet_labels` names:

| `pipeline` | Rows (Sep 2026) | Regions | Median SBR, left posterior putamen / left caudate |
|---|---|---|---|
| `ind_2014` | protocols 001 (84) and T011 (20) | six small VOIs | 0.66 / 1.38 |
| `core_lab_6voi` | 004 (94 of 97 analysed) | six small VOIs | 0.72 / 1.46 |
| `subregional` | 015 (35 of 36); also the C05-05 and DPA-714 rows | six VOIs, ten sub-regions, cerebellum (`PET_CBM`), occipital | 2.26 / 1.81 |

Never pool them: the sub-regional posterior putamen reads about three times higher. On the ten people scanned in T011
and again in 004 a year later, 004 values ran 12–23 % lower, and progression cannot be told from method, so those two
stay apart as well. `sbrcb_*` re-references sub-regional rows to the cerebellum: (1 + SBR) / (1 + `PET_CBM`) − 1. There
is no Xing-style re-analysis of AV-133 (that exists for DaTscan only).

PIE measures every PET visit on the participant's earliest finished FastSurfer T1:
1. `index_pet`, `select_series`: LONI series plus a first-file DICOM probe. Candidates are modality PT with no other
   PPMI tracer named. The scanner's FDG default does not exclude a series; the PPMI join decides the tracer. Per
   participant-day the unfiltered reconstruction wins (`--recon filtered` reverses this), then most frames, then most
   files; single-frame series of one class are stacked as sequential statics. Choices are kept in `pet_index.csv`.
2. `load_scan`: dcm2niix gives the 4D image and a PET-BIDS sidecar (`TracerName`, `InjectedRadioactivity`, `Units`,
   `DecayCorrection`, `ReconMethodName`). Frame timing comes from every DICOM header: `FrameReferenceTime` − duration / 2
   within a series, anchored at its earliest `AcquisitionTime`, relative to `RadiopharmaceuticalStartTime`.
   Overlapping frames are refused, and the volume count and dcm2niix's durations must agree.
3. `mean_image`: frames starting ≥ 60 min after injection (all of them when timing is unknown). Each is registered to
   the first (rigid MI). If any corner of a ±50 mm cube moves more than 1 mm (`motion_mm`), frames are resampled onto
   the first. Then the duration-weighted mean.
4. `register_pet_to_t1`: rigid Mattes MI to `orig.mgz`, the metric restricted to the FastSurfer brain dilated 10 mm,
   started from the centres of mass, seeded with `datscan.ITK_SEED`.
5. `quantify`: the PET is resampled onto the T1, and voxels outside its field of view leave every ROI.
   `datscan.sbr_from_arrays` then measures caudate and putamen, each also split at the median anterior coordinate,
   against the DKT occipital cortex; `sbrcb_*` uses cerebellar cortex (aseg 8 / 47).

On a phantom with known uptake, a 9° / 14 mm misalignment and 3 mm of inter-frame motion, the SBRs come back within
10 % (+0.05). With the registration's optimiser switched off they miss by up to 0.45.

```bash
venv_imaging/bin/python -m pie.imaging.pet --zips <AV-133 PET zips> --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/pet --workers 4 [--keep-nifti] [--recon filtered]
venv_imaging/bin/python -m pie.imaging.pet <same arguments> --agreement PPMI    # -> pet_agreement.csv; no processing
```

Flags: the shared `batch.add_common_args` set, plus `--recon` and `--agreement PPMI_DIR`. `pet_av133.csv` gets one row
per visit and resumes by `image_id`. `--keep-nifti` keeps `<patno>/<date>/` (4D NIfTI, `pet.json` with `TimeZero` =
injection, `pet_mean.nii.gz`, `t1_to_pet.tfm`).

Columns:
- Outcomes: `sbr_{caudate,putamen}_{l,r}` with `_ant` / `_post` halves (occipital reference), the same set as
  `sbrcb_*` (cerebellum), `cb_occ`, `mean_*`, `suv_reference` (body-weight SUV of the reference region).
- QC: `reg_pet_t1_mi`, `reg_template_r` (correlation with `datscan.synthetic_spect`'s striatum-hot template over the
  head), `motion_mm`, `motion_corrected`, `n_frames`, `n_frames_used`, `start_min`, `end_min`, `frame_minutes`,
  `timing_ok` (first frame at 75–85 min), `units`, `units_ok`, `cov_{striatum,occipital,cerebellum}`, `n_<roi>`.
- Acquisition and provenance: `tracer`, `tracer_av133`, `manufacturer`, `model`, `recon`, `postfilter`,
  `decay_correction`, `injected_mbq`, `voxel_mm`, `series_desc`, `patno`, `image_id`, `acquisition_date`,
  `fs_image_id`, `processing_version`, `error`.

`pet.qc_pass(d, reference="occipital")` passes rows with no error, an activity or count scale (`units_ok`: Bq/mL, SUV or
Philips `CNTS`, since the ratio cancels the scale), `reg_template_r >= 0.4` (the
DaTscan rule), `cov_striatum >= 0.99`, reference coverage ≥ 0.95, `motion_mm <= 5` and all six PPMI-comparable SBRs.
Confirm the thresholds on the first real batch.

`agreement_table` (`--agreement`) matches QC-passed rows to analysed AV-133 rows by PATNO and scan month. It reports per
pipeline × reference × region: n, Pearson r, ICC(2,1) (Shrout & Fleiss), Lin's CCC, and the Bland–Altman bias (PIE −
PPMI) with 95 % limits. Whole-structure ROIs without partial-volume correction should read below IND's small VOIs:
judge r first, and CCC / ICC after a per-pipeline calibration like `datscan.calibrate`.

Not done:
- GTM partial-volume correction (needs each scanner's resolution).
- PPMI's ten sub-regions (no FastSurfer equivalent).
- Follow-ups anchored on the baseline PET.
- Enhanced (single-file, multi-frame) PET DICOM: timing is read per file, so such a series stops with a frame-count
  error rather than being guessed.

### SV2A and tau PET (`pet.py --tracer sv2a | tau`)

PPMI's tables:
- **SV2A** (synaptic density; [18F]SynVesT-1): 18 scans of 15 participants on one Siemens Biograph, 88 Hammers-atlas
  regions plus Braak composites, SUVR against the centrum semiovale (`SV2A_PET_SUVR_Analysis`).
- **Tau** ([18F]PI-2620; the form is named MK-6240): 23 scans, Braak VI SUVR against the ventral cerebellum
  (`TAU_PET_Analysis`).
- Static scans start about 55–60 minutes after injection.

The same ingest, frame QC and registration as AV-133 apply. Frames from 40 minutes on are averaged (`timing_ok` for a
start within 45–70 minutes). `quantify_suvr` then gives `suvr_<region>` over FastSurfer aseg/DKT regions named after
the PPMI region they approximate (`SUVR_REGIONS`):
- striatum, pallidum, thalamus, amygdala, hippocampus and accumbens;
- insula, fusiform, cuneus, lingual, precentral, postcentral, superior frontal and superior parietal;
- anterior and posterior cingulate, parahippocampal, brainstem;
- the Braak VI composite (Schöll 2016: pericalcarine, postcentral, precentral, paracentral).

The references are:
- `cs`, the centrum semiovale: cerebral white matter at least 8 mm from any other label, above the lateral
  ventricles;
- `cerebellum_ventral`: cerebellar cortex below its median height, at least 6 mm from the cerebrum.

Outputs and checks:
- `pet_sv2a.csv` / `pet_tau.csv` hold the values; `qc_pass_suvr` checks units, reference coverage and motion;
- `--agreement` compares each region with PPMI's of the same name (`labels.suvr_labels`, `agreement_table_suvr`);
- **status:** phantom-tested; no SV2A or tau images downloaded yet.

Tests: `tests/test_pet.py`.

## Manifest, feature assembly and QC galleries

```python
from pie.imaging.manifest import QC, build_manifest, assemble_features, feature_blocks

man = build_manifest("<derived>")        # one row per subject: which session of each modality, dates, batch, QC
df = assemble_features("<derived>")      # manifest + FastSurfer IDPs + modality features, QC-failed values blanked
blocks = feature_blocks(df.columns)      # {"dat": [...], "dat_core": [...], "dwi": [...], "nm": [...], "flair": [...]}
```

`build_manifest(derived_dir, modality_dirs=None)` reads `<derived>/fastsurfer_idps.csv` (per subject the
earliest session passing `QC["t1"]`, else the earliest), `datscan_2026_10/datscan_sbr.csv` + `spect_index.csv`,
and `dwi/`, `nm/`, `flair/` feature and index CSVs. `modality_dirs={"dat": path, "dwi": path, …}` relocates any of
the four (`manifest.DEFAULT_DIRS` holds the `datscan_2026_10` default: the October 2026 registration); rows with an `error` are dropped. Columns:
`PATNO`, `t1_image_id`, `t1_date`; per modality `<mod>_date`, `<mod>_batch`, `<mod>_qc_pass`, `<mod>_days_from_t1`;
`dat_image_id`; for DWI/NM/FLAIR also `<mod>_date_source` (`recorded` from the row's `acquisition_date`, else
`index_inferred`: the date with the most files among the index's `selected` series) and, when present,
`<mod>_fs_image_id` (and `dat_fs_image_id`), `<mod>_processing_version`, `<mod>_source_image_ids`, `fw_method`,
`dwi_topup`, `dwi_denoised`, `dwi_bvecs_rotated`. If a modality row records a different FastSurfer T1 than the
manifest's, `<mod>_t1_mismatch` is set and its QC fails; this includes DaTscan, whose stored transform refers to
that T1.

`assemble_features(derived_dir, modality_dirs=None, single_shell_fw=False)` adds the IDPs. All of them are blanked,
with `t1_qc_pass=False`, when `QC["t1"]` fails:
- a key subcortical label is empty (FastSurfer failed);
- `BrainSegVol` is under 0.7 L, FastSurfer's own segmentation-volume check;
- the scan is rated `fail` in `<derived>/t1_visual_qc.csv` (`IMAGEID, rating` with pass, borderline or fail; column
  `t1_visual_qc`).

`t1_robust_z_max` and `t1_outlier` (`manifest.t1_outliers`) flag scans for visual review but never fail them,
because atrophy is an outlier too. They are the robust z of log key volumes within vendor × field strength, and
`t1_outlier` means |z| > 4. Review the outliers first, then a random 100, with
`python -m pie.imaging.qc --work-dir <derived> --modality t1 --worst t1_robust_z_max --out <dir>`, and record the
ratings in the CSV. The assembled table also adds every `dat_sbr_*` (occipital) and
`dat_sbrwm_*` (cerebral WM, PPMI's current reference) SBR with its anterior/posterior halves, putamen/caudate ratios and
asymmetry indices, `dwi_*` tensor and tract features (`manifest.is_dwi_feature`; single-shell free water blanked
unless `single_shell_fw=True`, see [imaging_dwi.md](imaging_dwi.md#single-shell-free-water-is-not-assembled-by-default)),
`nm_*_cnr` (`nm.py`'s ring contrast, a diagnostic), and `flair_wmh_*`. The default NM measure is snceg
(`nm_primary_cr` = `nms_sn_mean_cr`, the SN contrast against each side's crus on the SN's own slices, and
`nm_primary_volume_mm3` = `nms_sn_volume_mm3`; aliases, left out of `feature_blocks`), read from
`nm/nm_native_features.csv` (`nm_native --snceg`). `nm_vendor` and `nm_slab_mm` (the slab's thin-axis extent:
PPMI's protocol is 24 mm, most GE slabs 20 mm) are kept for every NM row, QC-failed or not, so analyses can restrict
to Siemens or full coverage explicitly. LONI-masked dates (year 9999) become missing, so no `*_days_from_t1` is computed from them. A modality failing QC is blanked
entirely, so a failed left-side metric cannot let a valid-looking right side through.

`assemble_features(..., ppmi_dir="PPMI")` also joins PPMI's own FreeSurfer 7 and MRIQC tables (`features.fs7_tables`:
`fs7_cth_<DK region>`, `fs7_sa_<region>`, `fs7_<aseg volume>` including `fs7_EstimatedTotalIntraCranialVol`, and
`mriqc_<metric>`), but only where the T1 PIE used is from the table's visit (baseline in the 2025 release; 1,080 of 1,804
subjects on the September 2026 download). It also adds **PPMI's core-lab SBRs for the DaTscan PIE reconstructed**
(`manifest.core_lab_sbr`, matched on participant and scan month): `dat_core_{caudate,putamen}_{l,r}`, their ratios and
asymmetry indices, and `dat_core_source` (`xing_cwm` where Xing analysed the scan, else `invicro_occipital`). These are
PPMI's endpoint values and the ones to analyse for PPMI; they take no PIE DaT QC. PIE's own `dat_sbr_*` reconstruction
is a screening measure for cohorts without a core lab. The two core-lab scales differ, so stratify or harmonise on
`dat_core_source` (`feature_blocks` gives them as their own `dat_core` block).

With `pet/pet_av133.csv` present (`pet.py`), the manifest takes the AV-133 visit nearest the T1:
- columns `pet_av133_image_id`, `_date`, `_days_from_t1`, `_batch`, `_fs_image_id` and `_qc_pass`;
- QC is `pet.qc_pass`, and it fails when the stored registration refers to another T1;
- the assembled table adds `pet_av133_sbr_*` (occipital reference, PPMI's scale), `pet_av133_sbrcb_*` (cerebellum,
  under its own coverage rule) and `pet_av133_cb_occ`, as the `pet_av133` block;
- `pet_av133_validated` is False until `pet --agreement` has been run.

FLAIR rows carry `flair_wmh_method` (`truenet`, or `threshold` for tables written before 11 Oct 2026). Never pool the
two methods. With `nm/nm_template_features.csv` present it adds the template-space NM
contrasts `nmt_*_cnr` under their own QC (`QC["nmt"]`: SN mask >= 90 % covered on both sides, crus CV < 0.3; flag
`nmt_qc_pass`).

The QC rules live in `manifest.QC` (one function per modality: `t1 dat dwi nm nmt flair`) so studies and galleries
agree on "pass". The batch columns are what in-fold harmonisation should use:
- `dwi_batch` = vendor + shells + free-water method (+ `_eddy`);
- `dwi_acquisition_batch` = vendor | shells | voxel size | number of volumes;
- `dwi_correction` = eddy / topup / rigid;
- `nm_batch` = vendor + voxel size;
- `nm_acquisition_batch` = vendor | voxel | TR | TE | MT preparation;
- `flair_batch` = vendor + 2D/3D;
- `dat_batch` = vendor + camera model.

The two `*_acquisition_batch` keys are the study's definitions. Its scanner-sensitivity analysis also adds scanner model and flip angle, which are in `nm_features.csv`.

Harmonising diffusion or neuromelanin features by the *T1* scanner is the mistake this avoids.

QC galleries (`qc.py`) render three orthogonal views per subject from `--keep-nifti` outputs: DWI FA with SN
(red), striatum (cyan), thalamus (magenta); NM mean slab with refined SN (red), atlas SN (cyan), reference (lime);
FLAIR on the T1 grid with the WMH mask; DaTscan with the labels of the row's `fs_image_id` (older rows: the earliest
finished T1, as `datscan` picks it) through the stored transform. `--modality t1` draws FastSurfer's segmentation
on the conformed T1. `--modality report` (`--work-dir` = the derived root) writes one self-contained HTML page per
participant (`qc.subject_report`): each modality's montage beside its manifest columns (QC flag, batch, dates, QC
metrics), like fMRIPrep's subject reports.

```bash
venv_imaging/bin/python -m pie.imaging.qc --work-dir <derived> --modality report --out <derived>/qc/reports --patnos ids.txt
venv_imaging/bin/python -m pie.imaging.qc --work-dir <derived>/dwi --modality dwi --out <derived>/qc/dwi --n 40 --worst reg_b0_t1_mi
venv_imaging/bin/python -m pie.imaging.qc --work-dir <derived>/datscan_2026_10 --modality datscan --out <derived>/qc/datscan \
    --sessions <derived>/sessions.csv --fastsurfer-dir <derived>/fastsurfer --worst reg_metric
```

| Flag | Default | Meaning |
|---|---|---|
| `--work-dir` | required | Modality work dir (reads its features CSV) |
| `--modality` | required | `t1`, `dwi`, `nm`, `flair`, `datscan`, or `report` (HTML per participant) |
| `--out` | required | Output dir: `<patno>.png` per subject and `gallery.png` (first 16) |
| `--n` | 40 | Subjects to render |
| `--worst` | random sample | QC column to order worst-first (`qc.worst_first`): negative-MI, motion and rotation columns largest first; `reg_metric` weakest \|correlation\| first; other columns smallest first |
| `--patnos` | none | Text file of PATNOs |
| `--sessions`, `--fastsurfer-dir` | none | Required for `datscan` |

Python: `montage(base, contours, out_png, title="", zoom=None)`, `render_subject(modality, subj_dir, out_png,
fastsurfer_dir=None, row=None)`, `gallery(pngs, out_png, cols=4)`. Tests: `tests/test_manifest.py`,
`tests/test_qc.py`.

## Deviation scores against controls (`normative.py`)

`normative.fit(frame, features, covariates, reference)` fits, per feature, a linear model on the reference rows only
(intercept, the numeric covariates you pass, and age² when one of them is `age`); `normative.zscores(frame, model)`
returns `z_<feature>` = (observed − predicted) / the reference residual SD. Use PPMI healthy controls (or, inside a
cross-validation fold, the training controls) as the reference and `tiv_mm3`, sex and a one-hot scanner batch as
covariates.

Options for a multi-site reference (the consortium; PPMI's ~50 sites):
- `site=` adds a per-site offset: the site's reference residual mean, shrunk by empirical Bayes; a site without
  controls gets none.
- `age_terms="spline"` uses a natural cubic spline in age.
- `heteroscedastic=True` lets the residual SD change with age.
- `crossfit_zscores(...)` gives controls out-of-fold z, so their z is a fair null for patients'.

`mtl_preservation(z)` gives `mtl_z`, the mean z of hippocampi and amygdalae, and `mtl_preserved` (`mtl_z` > −1). This
is relative medial-temporal preservation, the DLB-supportive biomarker (McKeith 2017). Pretrained lifespan models
(Rutherford et al. 2022) need pcntoolkit and are not wrapped. Tests: `tests/test_normative.py`.

```python
from pie.imaging import normative
model = normative.fit(df, ["vol_Left_Putamen", "vol_Right_Putamen"], ["age", "sex", "tiv_mm3"], reference=df.COHORT.eq("Healthy Control"))
z = normative.zscores(df, model)          # z_vol_Left_Putamen, z_vol_Right_Putamen
z = normative.crossfit_zscores(df, features, ["age", "sex", "tiv_mm3"], df.COHORT.eq("Healthy Control"),
                               site="scanner_id", age_terms="spline", heteroscedastic=True)
```

## Other cohorts: ADNI, any LONI project, BIDS (`cohorts.py`)

PIE's tables are keyed on an integer `patno`, so other cohorts get a key in their own range, computed from the
identifiers alone. That way every pipeline and work dir derives the same key without a shared registry
(`cohorts.subject_key`):
- PPMI keeps its PATNO.
- ADNI gets 10^8 + RID (`002_S_0295` → 100000295).
- Any other LONI project, and any BIDS dataset, gets 10^15 + 12 hex digits of sha1("<cohort>/<id>").

The ranges never overlap, so cohorts can be pooled. `index.csv`, `sessions.csv` and the modality indexes keep
`cohort` and `participant_id`.
- **LONI zips of any project:** `--zips` takes them as they come; the zip's top folder (`PPMI/`, `ADNI/`, …) names
  the cohort. ADNI sessions get ADNIMERGE's `VISCODE` as `EVENT_ID` with `run --adnimerge ADNIMERGE.csv`
  (`link.link_adni`: nearest exam date within 90 days). Without PPMI sessions the PPMI MRI table is not needed.
- **BIDS datasets (T1):** a folder in `--zips` is read as BIDS (`index.index_bids`):
  - every `sub-*/[ses-*/]anat/*_T1w.nii.gz` becomes a candidate series, with the sidecar's `AcquisitionDateTime`
    (else `scans.tsv` `acq_time`) as the session time;
  - undated sessions are kept apart by session label, LONI-masked style (9999-01-01);
  - images are copied, not converted, with their sidecar (`convert.convert_series`).
  PPP's data can only be reached inside its PEP environment, so PIE runs there from the container (`docker/`).
- **BIDS for DWI, NM and FLAIR:** the same folders work in their `--zips` (`batch.bids_series`):
  - `*_dwi` with its `.bval`/`.bvec` reads as `BIDS_dwi`;
  - `*_FLAIR` reads as `BIDS_FLAIR`, with `_3D` at slices of 1.5 mm or less;
  - neuromelanin (`acq-NM`, `neuromelanin`, `_NM`, `_MTw`) reads as `NM - BIDS …`, so the existing selectors pick
    them;
  - images are copied with their sidecars and gradients.
- **Still PPMI-only:** clinical labels (`labels.py`) and the DaT core-lab reader.

**Sharing across cohorts.** `manifest.export_long(df, out_dir)` writes the assembled table (cross-sectional or
per-visit) as a BIDS-Derivatives-style long table: `dataset_description.json` and `imaging_features.tsv` with one row
per participant, session and feature. Its columns are:
- `participant_id`, `sub-<cohort><native id>`;
- `session_id`, `ses-<EVENT_ID>`, else the T1 date;
- `cohort`, `block`, `feature` and `value`;
- the T1's vendor, field strength, serial number and `fastsurfer_version`.

QC-blanked values are dropped. This is the format to pool or to hand to another site.

## Longitudinal T1 (`manifest.t1_visits`)

The modality manifest (`build_manifest`, `assemble_features`) still has one row per participant: the earliest usable
T1 and the modality sessions nearest to it. For change over time, `t1_visits(derived_dir, stream="cross"|"long")`
returns **every** T1 visit (one per participant and visit day) from `fastsurfer_idps.csv` or the longitudinal stream's
`fastsurfer_long_idps.csv`. It applies `QC["t1"]` (measures blanked, identifiers kept) and the outlier flags, and adds:
- `years_from_bl` (from the earliest usable real-dated visit) and `visit_index`;
- `scanner_id` (`DeviceSerialNumber`, else `StationName`, else vendor and model), with `scanner_changed_since_bl` and
  `field_changed_since_bl`;
- `tiv_subject_mm3`, the median eTIV over the participant's usable visits (one head size per participant).

Visits are labelled with PPMI's `EVENT_ID`:
- `link_sessions_to_events` links by month;
- `index.visit_to_event` reads LONI collection-CSV codes (`BL`, `V04`, `U01`, …) and IDA names (`Month 24` → `V06`,
  `Unscheduled Visit 01` → `U01`, `Symptomatic Therapy` → `ST`) on PPMI's schedule (`MONTH_TO_EVENT`).

Recommended analysis:
- linear mixed models (`pie.stats.longitudinal.linear_mixed_model`) rather than change scores;
- the longitudinal stream for atrophy rates;
- scanner changes harmonised with `harmonize.long_combat`, with same-scanner follow-up as a sensitivity analysis.

The pilot's acceptance test is the 29 same-day repeat sets: they should show about zero change, and they give the
minimal detectable change.

**Every modality, every visit.** The DWI, NM and FLAIR CLIs take `--per-visit`, in a separate `--work-dir`:
- `batch.visit_jobs` runs one modality session per finished T1 session.
- Each dated session goes to the participant's T1 session nearest in date, within `--visit-window-days` (default 180).
- A T1 session keeps only its nearest modality session.
- Each session is measured on its own visit's anatomy (`fs_image_id`); resuming is by (patno, `fs_image_id`).

`build_manifest(per_visit=True)` and `assemble_features(per_visit=True)` then give one row per T1 session:
- each modality row attaches to the T1 session it was measured on;
- AV-133 PET attaches to the nearest T1 session; every PET visit is measured on the earliest T1;
- rows from tables written before `fs_image_id` existed attach to the baseline;
- the QC rules are the same as cross-sectional.

The cross-sectional reads prefer the row measured on the baseline T1 when a table holds several per participant.
DaT longitudinal values should come from PPMI's core-lab tables (`dat_sbr_table`), not PIE's reconstruction.

## Harmonisation (`harmonize.py`)

| Function | What it does |
|---|---|
| `combat(df, features, batch, covariates=(), nonlinear=("age",), label=None, eb=True, spline_df=4)` | ComBat with empirical Bayes and protected covariates. Covariates named in `nonlinear` enter as a cubic B-spline, like ComBat-GAM (Pomponio 2020) but unpenalised. Returns `(harmonised copy, model)`. With `nonlinear=()` it reproduces neuroCombat 0.2.12 to 1e-12 |
| `combat_apply(model, df)` | Applies a fitted model to new rows. Raises for batches the model has not seen. Spline covariates are clipped to the fitted range |
| `long_combat(df, features, batch, subject, time, covariates=(), label=None, ranef="1", niter=30)` | Longitudinal ComBat (Beer 2020): a REML mixed model with a subject random intercept is the location model, then empirical-Bayes batch location and scale on its residuals. A participant's scanner may change between visits. Ported from the R package `longCombat` (method `REML`), with statsmodels `MixedLM` in place of lme4 |
| `confounding_report(df, batch, label, covariates=(), min_per_class=5)` | Batch × label counts, label proportions per batch and Cramér's V. Flags batches that are single-label or have fewer than `min_per_class` of a class. Also gives covariate means (or level proportions) per batch |

Feature values must be complete, every batch needs at least 2 rows, and covariates are column names or patsy terms
(wrap numerically coded multi-level categories in `C()`).

**Inference mode** (group differences across cohorts): fit on every scan with the outcome protected, e.g.
`combat(df, f, "site", ["age", "sex"], label="saa")` or `long_combat(..., label="saa")`. Every such call logs the
confounding report (logger `pie.imaging.harmonize`, WARNING) and keeps it in `model["confounding"]`. These values
depend on every participant's label, so they must not feed a classifier evaluated on the same participants.
`combat_apply` refuses such a model.

**Prediction mode** (classifiers): fit inside each training fold with nuisance covariates only, never the outcome,
then `combat_apply` the model to the held-out fold. The held-out rows need covariates but no labels. Pool or drop
rare scanners first: a batch the training fold never saw is an error. `pie.experiment.prediction` does not call
this module. Its in-fold batch handling is still the study's ridge residualisation on batch one-hot codes (levels
seen fewer than 8 times pooled), applied when `residualize=True`.

**Harmonisation cannot fix complete confounding.** ComBat estimates each batch's shift after removing the protected
covariates. In a batch that holds only one label class, that shift is identified only by assuming the label effect
estimated in the other batches holds there too, and nothing in that batch can check this. If the label is constant
within every batch, the design is singular and `combat` / `long_combat` raise. In this consortium SAA positivity is
about 88 % in PPMI PD but about 22 % in ADNI, so cohort and label are entangled. Read `cramers_v` and `flagged`
before interpreting harmonised SAA differences, and check flagged cohorts with within-cohort contrasts. For
`long_combat`, the within-subject slope is protected only if scanner changes do not coincide with one visit for
everyone (some participants stay on a scanner over the same interval, or change dates vary). Otherwise the scanner
change and time are confounded.

Tests (`tests/test_harmonize.py`, simulated data only) cover:
- linear `combat` against neuroCombat, with and without empirical Bayes;
- recovery of known additive and multiplicative site effects when sites differ in age range and age acts on a
  curve. Spline ComBat brings the site-mean spread from 3.7 to 0.14 and the site SD ratio to 1.07; linear ComBat
  leaves 1.2 and 4.2;
- in-fold fit and apply, unseen batches and complete confounding;
- the confounding report on the consortium's prevalences;
- a longitudinal study where 60 % of participants change scanner at year 2. Their raw within-subject slope is
  −1.4 against a true −0.5; `long_combat` recovers −0.50 and removes 98 % of the scanner-mean spread.

No R reference numbers: R is not installed here, so the lme4 → `MixedLM` swap is unchecked against R itself.

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
mask instead of HD-BET / SynthStrip, the bundled MNI152NLin2009cAsym 2 mm affine target for all three, no WhiteStripe
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

`brain_age_gap(predicted, age, reference=None)` turns `brainage_sfcn` into the bias-corrected brain-age gap of de Lange &
Cole 2020 (doi:10.1016/j.nicl.2020.102229): predicted age is regressed on chronological age in the `reference` rows
only (controls or the training fold) and the gap is the residual. The raw difference falls with age by regression to
the mean. In PPMI the gap predicts clinical progression and DaT decline (Hähnel et al. 2025,
doi:10.1038/s41531-025-01232-4); SFCN was trained on ages 42–82. Before claiming anything from the embeddings, check
how well scanner or site can be predicted from them: on a travelling-heads set BrainIAC and 3D-Neuro-SimCLR had
between-scanner ICCs of 0.45 and 0.25, against 0.93 for FreeSurfer volumes (a 2026 preprint).

`run(fastsurfer_dir, image_ids, backends, out_csv, device="cuda", log=print, workers=4)` does one backend at a
time, appending to `<out stem>_<backend>.csv` (resumes from it) and then joins the parts on `IMAGEID`. Roughly
5-8 s per subject for all three; peak ~3.3 GB GPU (SimCLR at 1 mm). Tests: `tests/test_embed.py`.

## Shared infrastructure

### Modality runners (`batch.py`)

`dwi`, `nm` and `flair` share one runner shape: `index_series(zips)` (one row per series: `zip`, `prefix`,
`patno`, `desc`, `date`, `image_id`, `n_files`) → `load_index(index_file, zips, flag_fn)` (cached index with a
`selected` column) → `fastsurfer_by_patno(sessions_csv, fastsurfer_dir, require="stats/aseg+DKT.stats")` (earliest
finished T1 per PATNO) → `session_rows(group, t1_date)` (the acquisition date nearest the T1 session, then the most DICOM files) →
`run_batch(jobs, job_fn, out_csv, workers=4, log_every=10, pid_file=None)`. `run_batch` appends one row per
subject as it finishes, so a killed run loses at most the subjects in flight; `job_fn` must return a row with an
`error` string rather than raise; when a later row brings new columns the file is rewritten with the union
header. `done_subjects(out_csv, retry_errors=False, key="patno")` (`pet` keys on `image_id`: one row per scan), `filter_jobs(jobs, patnos_file=None, limit=None)` and
`add_common_args(ap)` (the shared CLI flags) complete it. `datscan`, `qc --modality datscan` and `dwi_refine` use
the same `fastsurfer_by_patno` rule, and `datscan` writes its table through `run_batch`. `batch.convert_series(zip_path, prefix, out_dir, archive_cache=None)`
extracts only `.dcm` members and returns every NIfTI dcm2niix wrote.
- `nm`, `nm_native --prepare-tse` and `dwi` hold one `ZipArchiveCache` per subject job, so a session's series read the zip index once.
  - PPMI stores each NM repeat as its own series.
  - A LONI zip with 2.8 million members takes about 13 s and 2.8 GB to index.
- The cache is not kept per worker process: eight resident indexes of that size would exceed a 24 GB memory cap.

Tests: `tests/test_batch.py`, `tests/test_nm.py`.

### Bundled atlas (`atlases.py`)

`pie/imaging/data/atlases/CIT168_v1_MNI152NLin2009cAsym_det25.nii.gz` + `.json`: CIT168 v1.0.0 (Pauli, Nili &
Tyszka 2018) in the authors' MNI152NLin2009cAsym 1 mm projection, reduced to a deterministic map (maximum-probability
label, one-based, where that probability >= 0.25). Labels 1-16: `Pu Ca NAC EXA GPe GPi SNc RN SNr PBP VTA VeP HN
HTH MN STH`. The JSON records source URL and SHA-256, file SHA-256, shape, affine, threshold, citation and the
correction date (2026-09-15). The same directory holds the registration reference the atlas belongs to,
`MNI152NLin2009cAsym_brain_2mm.nii.gz` + `.json` + `_LICENSE.txt`. `setup.py` ships all five files as package data.

| Function | What it does |
|---|---|
| `cit168_metadata()` | The JSON as a dict |
| `cit168_mni2009c(expected_space=MNI_SPACE)` | nibabel image after checking template space, file SHA-256, grid and label set; raises `ValueError` on any mismatch |
| `cit168_provenance()` | `atlas_space`, `atlas_version`, `atlas_sha256`, `atlas_probability_threshold`, `registration_reference_space`, `registration_reference_sha256` (written into NM rows) |
| `mni2009c_template_metadata()`, `mni2009c_template()` | The bundled 2 mm registration reference and its metadata, checked the same way (space, SHA-256, grid). `dwi.register_t1_to_mni` registers to this image |
| `mni2009c_brain_1mm(cache_dir=CACHE_DIR)` | Path of the brain-masked 1 mm MNI152NLin2009cAsym T1w, the SyN target of `dwi_refine` and `nm_template` |
| `schaefer400_mni2009c(cache_dir=CACHE_DIR)` | Schaefer 2018 400 parcels / 7 networks on fMRIPrep's 2 mm MNI152NLin2009cAsym grid, with `{parcel: network}` ([fmriprep.md](fmriprep.md)) |
| `templateflow_file(name, cache_dir=CACHE_DIR)` | One of `TEMPLATEFLOW_FILES`, downloaded once to `~/.cache/pie/templateflow` and checked against its pinned SHA-256 |

Nilearn's `fetch_atlas_pauli_2017` deterministic file is in native CIT168 space, not MNI152: never substitute it,
and never infer template identity from coordinates. The registration target is checked for the same reason --
nilearn's default 2 mm `load_mni152_template` is a distinct 2009**a** image, so registering to it and then mapping
a 2009c atlas through the result silently mixes two spaces. `dwi.pauli_atlas()` returns this atlas for DWI, NM and
`nm_template`. Tests: `tests/test_atlas_space.py` (atlas bytes and space, the reference template, and the cache
provenance below).

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
cross-validation folds, never on the full dataset before splitting. The study's final method, which PIE ships as `pie.experiment.prediction.ImageDesign(residualize=True, batch_cols=…)`, is a training-fold ridge residualisation on an age spline, sex (plus ICV for T1) and batch one-hot codes, chosen against the unadjusted features in the inner loop. ComBat ([`harmonize.py`](#harmonisation-harmonizepy)) is the alternative; with a few dozen scans per batch it can over-shrink. For multi-modality work use `manifest.assemble_features` and
[`pie.experiment`](experiment.md) instead.

## End-to-end example

```bash
bash scripts/setup_imaging.sh
D=Imaging/derived
venv_imaging/bin/python -m pie.imaging.run --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip \
    --ppmi-dir PPMI --work-dir $D --loni-csv Imaging/MRI_First_Study_9_07_2026.csv
venv_imaging/bin/python -m pie.imaging.dwi   --zips Imaging/<DTI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/dwi --fsl
venv_imaging/bin/python -m pie.imaging.nm    --zips Imaging/<MRI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/nm --keep-nifti
venv_imaging/bin/python -m pie.imaging.flair --zips Imaging/<MRI>.zip  --sessions $D/sessions.csv --fastsurfer-dir $D/fastsurfer --work-dir $D/flair_truenet
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

## Measures matched to the published literature

PIE's imaging measures are meant to be comparable with what the multimodal literature on α-synuclein
seed status reports, in particular Droby et al. 2025 (*npj Parkinson's Disease* 11:7), whose measures
were DaTscan striatal binding ratios, CAT12 volumes adjusted for intracranial volume,
basal-ganglia-network connectivity from group ICA, neuromelanin SN volume and intensity, and JHU tract
FA, with an SVM classifier. Each measure is documented on its own modality page; this table is only
the map between the two vocabularies.

| Measure in the literature | PIE | Notes |
|---|---|---|
| DaTscan SBR, occipital reference, caudate and putamen | `datscan.py` ([page](imaging_nm_datscan.md#datscan-spect-datscanpy)) | subject-space FastSurfer regions rather than template masks |
| Whole-brain GM and WM, putamen, caudate, pallidum, brainstem volumes | `features.tissue_volumes` | from FastSurfer labels |
| Intracranial-volume adjustment | `features.adjust_for_head_size`, `features.tiv_from_registration` | prefer eTIV (FastSurfer `--tal_reg`, or PPMI's `FS7_ASEG_VOL`); the registration estimate correlates with it at r = 0.81 |
| Basal ganglia network from group ICA, caudate and putamen weights | `fmri_connectivity.bgn_weights` (`load_reference_maps`, `select_component`, `dual_regression`, `roi_means`) ([page](fmriprep.md)) | dual regression matches FSL's; prefer a frozen external map set (maps from the analysed sample warn as circular); the CIT168 striatum picks the BGN |
| Seed-based striatal connectivity | `xcpd.xcpd_striatal_features` on XCP-D's denoised BOLD; `fmri_connectivity.striatal_rois`, `roi_timeseries`, `seed_network_connectivity` ([page](fmriprep.md)) | BOLD must already be in MNI152NLin2009cAsym; optional anterior/posterior putamen split |
| Neuromelanin SN volume and signal relative to white matter | `nm.hyperintense_volume`, `nm.mask_volume`, `nm.normalised_intensity` ([page](imaging_nm_datscan.md)) | the threshold volume depends several-fold on `k`; report it |
| Tract FA on the 48-label JHU ICBM-DTI-81 atlas | `dwi.fetch_jhu`, `dwi.map_labels_to_subject`, `dwi.registration_qc`, `dwi.tract_features` ([page](imaging_dwi.md)) | the atlas FA template is registered to the subject FA, so no MNI variant is assumed; laterality is checked on load |
| Partial correlations with bootstrap | `stats.small_sample.bootstrap_partial_correlation` ([page](stats.md#small_sample)) | no pingouin dependency |
| SVM with feature-subset search and leave-one-out validation | `stats.small_sample.nested_subset_search`, `naive_subset_search`, `subset_search_null` ([page](stats.md#small_sample)) | the search runs inside each validation fold; the other two size the selection bias in a given sample |

Real-data checks are recorded with each measure and in the module docstrings: JHU tract FA on a PPMI
2 mm scan (template FA correlation 0.69 with SyN, 0.54 affine), neuromelanin volumes on 40 scans,
striatal regions on the fMRIPrep grid, and intracranial volume on 8 FastSurfer subjects. Downloads
(JHU from NeuroVault collection 264, TemplateFlow MNI152NLin2009cAsym res-02 maps) are cached and
sha256-checked; nothing with an unstated licence is bundled.

## Coverage against the gold standard and the state of the art (September 2026 review)

A review of PPMI's own methods documents (core-lab SPECT, CIND DTI, the McGill FreeSurfer/MRIQC release, the PPMI
2.0 MRI manual) and the 2023–2026 literature. "Gold" is what PPMI or the field treats as the reference measurement,
"SOTA" what a current imaging working group would expect to see discussed.

| Modality | Measure | Standing | In PIE |
|---|---|---|---|
| DaT-SPECT | Caudate/putamen SBR, occipital and cerebral-WM reference | Gold (PPMI core lab) | PPMI's values via `labels.dat_sbr_table` (Xing first, Invicro for scans Xing never analysed); PIE's own reconstruction `datscan.py` for unquantified scans, mapped to a published scale by `datscan.calibrate` |
| | Putamen/caudate ratio, asymmetry index, z-score vs controls | Gold (EANM, DaTQUANT) | Yes: `labels.dat_labels`, `manifest.assemble_features` |
| | % of age/sex-expected lowest putamen; PARS ≤ 65 %, NSD-ISS D+ ≤ 75 % | Gold (PPMI staging) | Yes: `sbr_pct_expected`, `dat_deficit_sbr`, `dat_d_nsdiss` |
| | Pre/post-commissural, dorsal/ventral putamen | SOTA | PPMI's Xing columns kept; PIE's own: anterior/posterior halves |
| | Annualised SBR change | Gold trial endpoint | PPMI's longitudinal SBRs + `stats.longitudinal`; PIE's own reconstruction takes one scan per subject |
| T1 | aseg/DKT volumes, head-size adjustment | Gold | Yes (FastSurfer, release pinned in `setup_imaging.sh`, version on every row); eTIV from `run --etiv` (FreeSurfer 7.4.1 talairach, validated r 0.992 vs FS7 on 40 scans; cohort run pending); the registration TIV is unvalidated |
| | Cortical thickness and area (DK) | Gold; PPMI releases FS7 tables for 1,716 baseline scans | `run --surfaces` (`cth_*`, `area_*`; needs FreeSurfer 7.4.1 and its licence; not yet run on the cohort); else PPMI's `FS7_APARC_CTH/SA` |
| | Basal forebrain / nucleus basalis (Ch4) volume | SOTA SAA marker (smaller with SAA positivity in BioFINDER and ADNI) | `run --sclimbic` (ScLimbic basal forebrain, a broader region than the atlas-defined Ch4 the SAA papers used; agreement with a Ch4 atlas volume not yet checked); not yet run on the cohort |
| | Hippocampal subfields, amygdala nuclei | Gold (FreeSurfer) | `run --subfields` after `--surfaces`; not yet run |
| | Legacy 1.5 T / thick-slice robustness | SOTA (SynthSeg) | `run --synthseg` (volumes + QC scores); not yet run |
| | Longitudinal within-subject stream | Gold for atrophy rates | `run --long` writes `fastsurfer_long_idps.csv`; `manifest.t1_visits` gives every visit with time, scanner change and one TIV per participant; not yet piloted |
| | Brain-age gap | SOTA, replicated in PPMI | SFCN + `embed.brain_age_gap` |
| | Foundation-model embeddings | SOTA, exploratory | BrainIAC, 3D-Neuro-SimCLR, SFCN; check scanner reliability first |
| | Normative centiles; NBM Ch4 volume; MRPI; SynthSeg+ | SOTA / specialised | No |
| DWI | FA, MD, AD, RD in nigral, subcortical and JHU tract ROIs | Gold | Yes (AD/RD added September 2026) |
| | topup + eddy with outlier replacement | Gold preprocessing | topup opt-in (`--fsl`), eddy as a command builder only (`dwi_correction`) |
| | Posterior-SN free water | SOTA PPMI marker | Multi-shell (PPMI 2.0) yes; single-shell quarantined, see [imaging_dwi.md](imaging_dwi.md#single-shell-free-water-is-not-assembled-by-default) |
| | Free-water-corrected FA / MD; DKI mean kurtosis | SOTA | FAt all scans; MDt and MK multi-shell |
| | Nigrostriatal tractography, fixel AFD | SOTA | Yes (`--fba`) |
| | TBSS skeleton; NODDI | ENIGMA standard; SOTA | No (b ≤ 2000 limits NODDI) |
| NM-MRI | SNc volume by reference-based threshold, native space | Gold (most replicated; Cho 2021, Langley 2025 on PPMI) | Implemented: `nm_native` (`nml_*`), MP-PCA-denoised repeats (`nm --denoise`). With the public Biondetti ROIs in place of Langley's unpublished atlas, the Siemens pilot did **not** reproduce the published PPMI effect (d 0.01 vs 0.83) |
| | SN contrast vs a crus/tegmental reference | Gold | Yes: **default** `nm_native` snceg CR vs the crus on the SN's slices (`nm_primary_cr`); `nm_template`; `nm.py` ring contrast (diagnostic) |
| | Voxel-wise CNR vs crus in template space (Cassidy 2019, Wengler 2020) | SOTA; cited by the PPMI 2.0 MRI manual | Yes (`nm_template.py`) |
| | Deep-learning SN segmentation (snceg, Lillebostad 2025) | SOTA | Yes: `nm_native --snceg` (pinned public weights, separate environment), PIE's default NM measure. Reproduced on PPMI TSE-FS (AUROC 0.710 vs 0.743); GRE-MT re-test pre-registered ([preregistration_nm_snceg.md](preregistration_nm_snceg.md)) |
| | Voxelwise template CNR, functional territories, multi-site harmonisation | SOTA (Wengler 2020/2021, Biondetti 2020) | Yes: `nm_template`, `published` stage, `nm_acquisition_batch` |
| | NM SN volume loss rate (longitudinal) | SOTA progression marker | Cross-sectional volume only; no longitudinal pipeline |
| | Locus coeruleus | SOTA in prodromal work | Not in PPMI's 24 mm slab |
| Iron | QSM, R2*, nigrosome-1 on SWI | SOTA | Not acquired by PPMI-1 or PPMI 2.0 (no multi-echo GRE or SWI) |
| FLAIR | WMH volume | Gold (vascular covariate) | TrUE-Net (Oct 2026), not yet validated in PPMI; until then prefer FastSurfer's `vol_WM_hypointensities` (below) |
| rs-fMRI | fMRIPrep + XCP-D (36P, censoring, respiratory notch filter at TR 1 s) | Best practice | Yes: `xcpd.py` pins XCP-D 26.2.0 ([page](fmriprep.md#5c-xcp-d-denoising-and-parcellation-pieimagingxcpd)); PIE's own 24P + aCompCor path kept as a cross-check |
| | QC-FC and its distance dependence | Expected by reviewers | `fmri_connectivity.qc_fc`; validation plan on the [page](fmriprep.md#5e-validation-plan), not yet run |
| | Schaefer cortex + CIT168 subcortex | SOTA default ("4S") | Yes (`atlases.schaefer400_mni2009c`, `striatal_rois`) |
| | Striatal seed FC, basal-ganglia network (ICA + dual regression) | Canonical, poorly reproducible | Yes |
| | ALFF / fALFF / ReHo | SOTA | XCP-D's ALFF (as mALFF) and ReHo via `xcpd_features`; `fmri_connectivity.local_measures` adds fALFF (unvalidated) |
| PET | AV-133 (VMAT2) striatal SBR; SV2A SUVR; tau (PI-2620) SUVR | PPMI core-lab tables (`PET_SBR_Analysis`, `SV2A_PET_SUVR_Analysis`, `TAU_PET_Analysis`) | AV-133: `pet.py` follows PPMI's 2014 IND method on the subject's FastSurfer ROIs (occipital and cerebellar references); `labels.pet_labels` reads PPMI's values per pipeline. Not yet run: no AV-133 images downloaded, no agreement measured. No partial-volume correction. SV2A and tau: `pet.py --tracer sv2a|tau` SUVRs (phantom-tested; images not yet downloaded) |
| CT | Safety head CT before the screening lumbar puncture | Not an imaging endpoint: the `Safety_Head_CT_Scan` form (`CTSCAN`, yes/no) records whether one was done; PPMI does not distribute the images | Nothing to process |
| All | In-fold harmonisation, per-modality batch | Gold | Batch columns in the manifest; training-fold ridge residualisation on age, sex and batch in `pie.experiment.prediction.ImageDesign` (the study's method); ComBat, spline-age ComBat and longitudinal ComBat in `harmonize.py` |

### Real-data validity checks (September 2026)

On the study's derived tables (written 7–8 September 2026, before the 9–21 September fixes: re-run DWI, NM and FLAIR
before quoting their values), with PPMI cohort labels:

| Measure | Check | Result |
|---|---|---|
| DaT SBR, lowest putamen | PD vs HC | d = −2.1; AUROC 0.95–0.96 (PPMI's own SBRs 0.99 on the same scans) |
| DaT SBR | Agreement with PPMI | putamen r 0.79–0.81, caudate r 0.61–0.66 (PPMI's two core labs: 0.88) |
| DaT labels | Coverage of MRI sessions | 1,447 with the Xing table, 988 with the archived Invicro table; `auto` now adds the Invicro-only PPMI-1 sessions (count not yet re-measured) |
| T1 volumes | Age | thalamus r −0.42, hippocampus −0.36, ventricles +0.40 (n = 1,800) |
| T1 volumes | vs PPMI's FreeSurfer 7.3.2 `FS7_ASEG_VOL` (1,080 baseline scans) | subcortical r 0.81–0.96, Lin's concordance 0.77–0.95 (putamen 0.92 / 0.85, lateral ventricle 0.96 / 0.95); WM-hypointensities r 0.19 (the two methods differ) |
| Registration TIV | vs FreeSurfer eTIV (40 scans) | r 0.81, 9 % low; FastSurfer `MaskVol` r 0.85 |
| PIE eTIV (`run --etiv`, FreeSurfer 7.4.1 talairach) | vs PPMI's FreeSurfer 7 eTIV, same 40 scans (26 Sep 2026) | r 0.992, mean difference +0.3 %, 0 of 40 rejected by `talairach_afd`: passes the pre-registered gate (r ≥ 0.95, < 5 %) |
| DWI multi-shell FW | Age | brain r +0.49, posterior SN +0.28 (n = 68) |
| DWI single-shell FW | Age; PD vs HC | brain r +0.02, posterior SN AUROC 0.50: quarantined |
| DWI nigral FA / MD | vs PPMI's hand-drawn SN ROIs (248 scans) | FA ρ 0.39 (PIE reads higher: 0.43 vs 0.32, peduncle partial volume), MD ρ −0.01 |
| NM atlas ROI CNR | PD vs HC (166 / 45) | AUROC 0.54 overall, 0.60 posterior half |
| NM, pre-registered gate (AUROC ≥ 0.70, CI lower bound > 0.55), 45 HC + 45 PD, 26 Sep 2026 | PD vs HC | template CNR (`nmt_sn_mean_cnr`) 0.58 [0.46, 0.70], posterior 0.61; atlas CNR re-run with current `nm.py` (`nm_sn_mean_cnr`) 0.54 [0.42, 0.67]. **Neither passes: NM measures are exploratory** (`manifest.NM_VALIDATED = False`, column `nm_validated`). The template's crus reference sat mostly above the SN (median 7 mm); restricted to the SN's axial levels on 11 Oct 2026 |
| NM re-test, published Biondetti territories, pre-registered (same gate plus ρ ≥ 0.20 with the lowest putamen SBR); 106 held-out PD + the 39 usable HC (reused), GE/Siemens/Toshiba | PD vs HC | sensorimotor contrast (`nmb_sensorimotor_mean_cnr`, vendor/age/sex-adjusted) 0.62 [0.53, 0.72], ρ 0.18; 8-territory model (cross-validated) 0.67 [97.5 % CI 0.55, 0.78], ρ 0.20. Both fail; NM stays exploratory. Sensorimotor NM is lower and limbic NM higher in PD in both samples. Unadjusted sensorimotor AUROC by vendor: GE 0.53, Siemens 0.70, Toshiba 0.97 (6 HC / 5 PD) |
| FLAIR registration | 30 scans that failed QC + 10 that passed, re-run 25 Sep 2026 | 21 / 30 now pass, 10 / 10 still pass; former failures' median WMH 11.1 → 2.5 mL |
| FLAIR WMH (threshold) | Age; vs FastSurfer WM-hypointensities | ρ +0.18 (3D FLAIR +0.24); ρ 0.52 with the T1 measure, which itself tracks age at ρ +0.46 |

The DWI nigral values agree poorly with manual ROIs because a 2 mm, affine-mapped, distortion-uncorrected nigra
is several millimetres from the anatomy on PPMI-1 data; free-water results should come from PPMI 2.0 multi-shell
scans with topup (`--fsl`), and ideally eddy.

### Real-data validity checks (October 2026)

| Measure | Check | Result |
|---|---|---|
| T1 volumes (FastSurfer) | Same-day MPRAGE + repeat (29 pairs; ICC(2,1), SEM, 95 % minimal detectable change) | putamen ICC 0.98 (MDC95 5.1–5.5 %), caudate 0.99 (3.7–4.6 %), thalamus 0.98–0.99 (2.3–3.9 %), hippocampus 0.99 (2.7–3.2 %), amygdala 0.97–0.98 (5.7–6.2 %), lateral ventricles 1.00 (1.7–2.4 %), brainstem 0.98 (4.3 %). `BrainSegVol` 0.87 (10 %): coverage differs between repeats |
| T1 volumes | vs PPMI FS7, by vendor and field strength (1,080 baseline scans; median CCC over 15 structures) | all 0.87 (r 0.90); Siemens 0.88, Philips 0.82, GE 0.80; 3 T 0.86, 1.5 T 0.82. PIE's volumes run larger (putamen +235–246 mm³) |
| T1 normative z (`normative.crossfit_zscores`, age + sex + TIV; 165 HC, 543 PD, 318 prodromal) | Controls' out-of-fold z | plain model: mean within 0.013 of 0, SD 1.01–1.03, residual age r ≤ 0.016, 5.0 % beyond ±2; with `site` + spline + heteroscedastic SD: SD 1.08–1.16 (too many parameters for 165 HC across 6 scanner groups; use it when sites have more controls). PD mean z: putamen −0.08 to −0.10 (ENIGMA-PD d −0.14), hippocampus −0.13 to −0.21. `mtl_preserved`: HC 90 %, PD 86 % |
| DaT registration (Oct 2026): brain-only template, legacy fallback | vs Xing, raw r | **Tuning set** (240 random Xing-matched scans outside the validation set): old pipeline putamen-min 0.78 / caudate 0.55 → brain-only 0.90 / 0.81 (INFINIA 0.53 → 0.97); the head-outline template variants scored 0.71–0.83. Brain-only correlation bar set there: 0.2 (2nd percentile 0.23). **Validation set** (236 participants, vendor-stratified): brain-only fit passes 188 of 237; on the 128 Xing-matched passers putamen 0.80 → 0.87–0.90 (CCC 0.79 → 0.87–0.89), caudate 0.58–0.61 → 0.84–0.86, deficit κ 0.71 → 0.74. Its 31 matched failures (HERMES, BrightView, Forte, P3000XP) agree at 0.35 but at 0.75–0.79 with the legacy registration, so failures fall back to it (`quantify_with_fallback`): 236 of 237 pass, and on the 156 scans both pipelines pass putamen 0.796 → 0.861, caudate 0.675 → 0.743. **Full cohort** (1,459 scans, 1,452 pass QC; now the manifest default, `datscan_2026_10`): vs Xing (1,161 matched) putamen-min r 0.788 → 0.851, CCC 0.81 → 0.87, caudate 0.61–0.63 → 0.74–0.76, deficit κ 0.75 → 0.80 (prodromal 0.54 → 0.65); vs Invicro (882) putamen-min r 0.887, CCC 0.89, caudate CCC 0.80–0.82, deficit κ 0.85 |

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
| Separate NM reconstructions from independent acquisitions; preserve repeat support and hemisphere coordinates | `nm.py:compatible_repeats`, `average_repeats`, `nm_rois` | Existing pipeline |
| Bounded per-side slab coverage ≥ 0.9; rigid fit from both starts, lowest MI among fits covering both sides; ≥ 95 % of the measured voxels with signal and reached by every repeat | `nm.py:slab_coverage`, `register_slab`, `features`; `manifest.QC["nm"]` | Ported 27 Sep 2026; older tables keep the old coverage bar |
| Map the atlas onto the NM slab through the subject's nonlinear (SyN) T1 → MNI registration, not an affine | `nm.py:_atlas_on`, `register_slab`, `refeature_subject` | Ported 27 Sep 2026; `refeature` regenerates affine-era maps |
| Keep one complete tensor acquisition and its own b0s; validate slice groups; preserve odd-sized grids | `dwi_correction.py` | New reusable opt-in selection/command/geometry helpers; no study imports |
| Inspect raw tensor eigenvalues and nonpositive signal before accepting FA/MD/AD/RD | `dwi_tensor_qc.py:tensor_measurements` | Opt-in measurement API already used by the reassessment, not silently substituted for legacy free-water features |
| Expose free-water optimizer status, residuals and raw tensor failures | `dwi_acquisition.py:diagnostic_multishell` | Development diagnostics only; positive-definite initialization repair is not a validated estimator |
| Verify atlas space, geometry, labels and exact bytes before nigral mapping | `atlases.py:cit168_mni2009c` | Shared DWI/NM default corrected on 2026-09-15; uses the authors' MNI2009c v1.0 projection |
| Reject failed or nonphysical multi-shell fits before eigenvalue clipping hides the failure | `freewater_qc.py:fit_multishell_checked` | Opt-in checked NLS API used by corrected control runs; no initialization repair or changed solver |

Corrected study controls also require both hemispheres, complete signal coverage of the new mask, and a single
tensor acquisition with its own b0 images. The NM recovery runner calculates geometric slab coverage from
transformed source-atlas points; `nm.py` does the same since 27 Sep 2026 (`sn_slab_coverage_{l,r}`), in place of
its earlier resampled-volume ratio, which was not a bounded coverage fraction.
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
  FreeSurfer licence). PPMI's own `FS7_APARC_CTH` / `FS7_APARC_SA` tables (FreeSurfer 7.3.2, DK atlas, baseline scans)
  can supplement; `fastsurfer.surfaces` runs the surface stream (pilot pending). eTIV comes from `run --etiv`
  (validated, r 0.992 vs FS7). The longitudinal stream (`run --long`) exists but has not been piloted.
- FastSurfer on CPU (`run --device cpu`) works but takes tens of minutes per scan.
- ANTs registrations are seeded through `ants.config` (`features._seed_ants`). Until 26 September 2026 they were not: ANTsPy 0.6.3 silently ignores a `random_seed=` keyword.
  - Even seeded and single-threaded, Mattes/MI registrations in ANTsPy 0.6.3 are not bit-reproducible. On a phantom, repeat affine runs differed by about 0.01 mm in translation and 0.5 % in registration-based TIV.
  - Treat values from ANTs-registered masks as reproducible to that order. ANTs' Repro mode (GC metric) would be exact but is not used.
  - The Biondetti bridge is computed once and cached, so NM masks do not vary between runs on one machine.
- DWI: eddy-current and slice-outlier correction are not in the default path; susceptibility correction only
  where a reverse-PE b0 exists (`--fsl`, PPMI 2.0). Single-shell free water is not assembled by default.
- DaTscan: PIE's own reconstruction (FBP, no attenuation correction, subject-space ROIs, hottest-placement search) is
  not on PPMI's scale; use PPMI's SBRs where they exist. One scan per subject.
- FLAIR: TrUE-Net is the default but unvalidated in PPMI (validation plan in the FLAIR section); the threshold
  method tracks age more weakly than FastSurfer's T1 WM-hypointensity volume. Use the latter as the vascular
  covariate until TrUE-Net passes. QC passed 72 % of 2D against 95 % of 3D FLAIRs (threshold-method table), so
  blanked WMH is not missing at random. WMH-SynthSeg is not available in FreeSurfer 7.4.1.
- AV-133 PET: whole anatomical ROIs, no partial-volume correction and no sub-regional ROIs, so absolute SBRs are not on PPMI's scale until `--agreement` has been run; each visit is measured on the earliest T1 (atrophy at late visits is not modelled).
- QSM, R2* and SWI are not possible: PPMI does not acquire multi-echo GRE or SWI.
- Before 25 September 2026 every `ants.registration` call (`nm_template syn`, `dwi_refine`, `tiv_from_registration`, `map_labels_to_subject`) left its transforms in the temp directory, ~140 MB per SyN. They are now deleted after use; files named `tmp*Warp.nii.gz`, `tmp*InverseWarp.nii.gz` and `tmp*GenericAffine.mat` left in `/tmp` by older runs are safe to remove.

## Tests

```bash
venv_imaging/bin/python -m pytest -q tests/test_imaging.py tests/test_imaging_audit.py tests/test_archives.py \
    tests/test_atlas_space.py tests/test_batch.py tests/test_cnn.py tests/test_datscan.py tests/test_dicom_audit.py \
    tests/test_dwi.py tests/test_dwi_correction.py tests/test_embed.py tests/test_flair.py tests/test_freewater_qc.py \
    tests/test_manifest.py tests/test_nm.py tests/test_nm_template.py tests/test_qc.py tests/test_staging.py \
    tests/test_staging_files.py tests/test_imaging_regressions.py \
    tests/test_dwi_tracts.py tests/test_fmri_striatal.py tests/test_nm_volume.py tests/test_volumes.py tests/test_normative.py tests/test_harmonize.py \
    tests/test_fmri_connectivity.py tests/test_xcpd.py tests/test_brain_viewer.py tests/test_pet.py tests/test_cohorts.py tests/test_analyse_dwi_fw.py
```

`tests/test_imaging_regressions.py` holds one regression test per bug fixed after the September 2026 audit
(DaTscan header, T1 choice, manifest dates and DaTscan lineage, NM refeature, carrier coding, defaults, device
budgets, tool-path overrides).

All synthetic (phantoms and generated tables): no PPMI data, GPU, FastSurfer, FSL or MRtrix3 needed, but DIPY
and nilearn must be installed. `tests/test_embed.py::test_real_weights_load` is skipped unless the weights are
under `third_party/weights/`. Self-checks:
`python -m pie.imaging.index`, `python -m pie.imaging.fastsurfer`.
