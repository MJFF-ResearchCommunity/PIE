# fMRI processing (fMRIPrep, connectivity, QC)

Resting-state BOLD processing in `pie.imaging`, from archived DICOM to one row of network
connectivity per scan. PIE orchestrates the steps and records what it did. It does not
vendor fMRIPrep, pick acquisitions for you, or declare a scan scientifically usable.

```
IDA ZIP ──convert_archive_series──► NIfTI + conversion.json          (fmri.py)
          └─assemble_temporal_volumes (converter-split series only)   (fmri_assembly.py)
       ──export_subject──► BIDS sub-<label>/{anat,func,fmap}          (fmri_bids.py)
       ──run_fmriprep──► fMRIPrep derivatives + pie_provenance/       (fmriprep.py)
       ──visual review──► montages, report-layer PNGs                 (fmri_qc.py)
       ──run_xcpd──► XCP-D denoised BOLD, matrices, ALFF/ReHo, QC     (xcpd.py, primary path)
          └─xcpd_features / xcpd_striatal_features──► feature row
       ──extract_connectivity──► connectivity.json + .npz             (fmri_connectivity.py, PIE's own path)
       ──caller──► table keyed on PATNO / EVENT_ID                    (recipe below)
```

Denoising and parcellation are handed to a pinned XCP-D (section 5c). PIE's own path (section 5) stays for
what XCP-D lacks: basal-ganglia-network dual regression, the QC-FC report, and a cross-check.

| Module | What it does |
|---|---|
| `fmri.py` | CRC-checked DICOM→NIfTI conversion that keeps every converter output. Run classification, phase-encoding pair checks, FD, and an FSL motion/alignment pilot. |
| `fmri_assembly.py` | Rebuilds a 4D run from converter-split 3D volumes, only when DICOM headers prove the temporal order. |
| `fmri_bids.py` | Single-session BIDS export: T1w, one rest run and an optional verified reverse-PE fieldmap pair. |
| `fmriprep.py` | fMRIPrep backend (native or Apptainer) with config validation, version pinning, execution identity and resume. |
| `fmriprep_reuse.py` | Anatomical-only snapshot of earlier fMRIPrep derivatives, passed back in as precomputed input. |
| `fmri_qc.py` | Physical-space montages and both layers of fMRIPrep's animated SVG comparisons, for human review. |
| `xcpd.py` | XCP-D 26.2.0 backend (same config/identity/resume contract as `fmriprep.py`) and parsers of its outputs into `fmri_xcpd_*` feature rows, including striatal connectivity on its denoised BOLD. |
| `fmri_connectivity.py` | Censoring, nuisance regression, parcel coverage, Fisher-z network connectivity, striatal seeds, group ICA / dual regression, QC-FC. |
| `fmri_data.py` | `BOLDImageCache`: one read-only 4D array shared across QC variants. |

Every step shares four properties:

- **No outcomes are read.** Inputs are images, sidecars and confounds only.
- **Completion records, not overwrites.** Each step writes a JSON record holding input and
  output SHA-256 hashes. On a rerun with the same inputs, the step verifies the saved
  outputs and returns the record. Changed inputs or settings raise an error; an
  incomplete destination raises `FileExistsError` for you to inspect.
- **`scientific_qc_pass` is always `False`.** A process exiting 0 or a numerical gate
  passing does not certify distortion correction, registration or coverage. Someone has
  to look at them.
- **Nothing is inferred.** Phase-encoding polarity, readout time, slice timing and temporal
  order come from metadata or the step fails. Series descriptions such as "AP"/"PA" are
  never used as polarity.

## Requirements

| Need | For | Notes |
|---|---|---|
| `dcm2niix` | conversion | `pie.imaging.convert.DCM2NIIX` defaults to `venv_imaging/bin/dcm2niix` (pip `dcm2niix`, installed by `scripts/setup_imaging.sh`). |
| `pydicom`, `nibabel`, `numpy`, `pandas`, `matplotlib` | all | Present in `venv_imaging`. |
| fMRIPrep | preprocessing | Native install ([upstream instructions](https://fmriprep.org/en/stable/installation.html)) or a local Apptainer `.sif`. Contract tested against **25.2.5**. The pip package alone lacks the external tools. Docker is not a backend. |
| FreeSurfer license | fMRIPrep | fMRIPrep 25.2.5 checks it even with `--fs-no-reconall`. Free from [FreeSurfer registration](https://surfer.nmr.mgh.harvard.edu/registration.html). |
| XCP-D | denoising, parcellation | Local Apptainer `.sif` of `pennlinc/xcp_d:26.2.0` (pull command in 5c). A native install needs AFNI, ANTs and Workbench; use the container. |
| FSL (`fsl_dir`) | pilot only | `motion_pilot`, `alignment_pilot` (`mcflirt`, `bet`, `flirt`). Not needed for the fMRIPrep path. |
| `cairosvg` | `render_report_states` only | Optional; not installed in `venv_imaging`. |
| Atlas + network table | connectivity | Any atlas in the BOLD derivative's standard space, or the default: Schaefer 2018 400 parcels / 7 networks in MNI152NLin2009cAsym (`atlases.schaefer400_mni2009c()`, fetched from TemplateFlow and sha256-pinned). |

No external drive, mount point or machine path is assumed. Every path is a caller
argument. Capacity limits and mount/recovery policy belong to the deployment.

## 1. Conversion (`pie.imaging.fmri`)

```python
from pie.imaging.archives import ZipArchiveCache
from pie.imaging.fmri import convert_archive_series

record = {"PATNO": "1", "image_id": "<IMAGE_ID>", "archive": "/data/ida/download.zip",
          "series_prefix": "PPMI/1/<SERIES>/<DATE>/<IMAGE_ID>/", "dicom_entries": 240}
with ZipArchiveCache(max_open=1) as cache:
    result = convert_archive_series(record, "/data/converted", archive_cache=cache)
[(o["nifti"], o["run_class"]) for o in result["outputs"]]
```

`convert_archive_series(record, output_root, *, dcm2niix=DCM2NIIX, archive_cache=None, scratch_root=None, extra_args=())`

| `record` key | Rule |
|---|---|
| `PATNO` | Digits only. |
| `image_id` | `I` followed by digits (LONI image ID). |
| `series_prefix` | Relative, ends in `/`, no `..`, contains the PATNO component and ends with `image_id`. |
| `archive` | ZIP path. It is opened read-only and must not change during conversion (size/mtime are rechecked). |
| `dicom_entries` | Optional. Selected `.dcm` member count must equal it. |

- Members are extracted in CRC-checked form to numbered files (`00000000.dcm`, …), because
  member filenames are ignored. That rules out path traversal and basename collisions.
- The converter runs as `dcm2niix -g i -z i -b y -ba y <extra_args> -f <image_id>_%s -o … …`
  with `OMP_NUM_THREADS=1`. `-g i` ignores per-user converter defaults. Timeout 3600 s.
- **All outputs are kept.** A BOLD series often converts to a full run plus a short
  reference, and choosing "the largest file" silently discards one of them.
- Free space is checked per filesystem. The temporary DICOM copies need DICOM bytes + 1 GiB
  on the scratch filesystem, and converter staging needs 3 × DICOM bytes + 1 GiB on the
  `output_root` filesystem. When both roots share a filesystem the two are summed
  (4 × + 2 GiB). `scratch_root` moves only the temporary DICOM copies (e.g. to a faster
  filesystem). It must already exist; a missing scratch directory is an error, never a
  fallback. Staging and atomic publication stay on `output_root`.
- `extra_args=("-m", "y")` asks dcm2niix to merge split temporal volumes. The flags are
  recorded in the saved command.

Output under `output_root/<PATNO>/<image_id>/`: `<image_id>_<series>.nii.gz` + `.json` for each
converter output, plus `conversion.json` with `source` (archive identity),
`dicom_count`, `source_member_index_sha256`, `study_uid`, `series_uid`,
`converter_sha256`, `code_sha256`, `command` and `outputs[]`. Each output row holds the
file hashes and the `inspect_nifti` fields. On failure, the converter log, its unvalidated
products and a JSON record go to `output_root/conversion_failures/<PATNO>/<image_id>/`, and
`DICOMConversionError` is raised. Nothing is published as completed.

`ZipArchiveCache` reuses ZIP central-directory indexes, not extracted bytes. A
million-entry index can exceed 1 GB, so keep `max_open` small and use one owner
thread/process.

### Inspection helpers

| Function | Returns |
|---|---|
| `inspect_nifti(nifti, sidecar)` | `shape`, `voxel_sizes`, `axis_codes`, `affine`, `spatial_units`, `tr_seconds` (sidecar), `nifti_tr_seconds` (header), `duration_seconds`, `run_class`, `phase_encoding`, `total_readout_time`, `slice_timing_count`, `manufacturer`, `image_type`, `series_description`. Raises when NIfTI and JSON TR disagree (1e-4 tolerance) or the affine is singular. |
| `classify_run(shape, tr, *, min_volumes=100, min_seconds=300)` | `not_functional_4d` (not 4D or <2 volumes), `unknown_timing`, `short_reference_candidate` (≤20 volumes), `insufficient_duration` (<100 volumes or <300 s), `rest_candidate`. Uses actual dimensions, not DICOM file counts. |
| `phase_encoding_pair(first, second)` | Sorted reasons a reverse-PE pair is unusable: `missing_verified_phase_encoding`, `not_opposite_phase_encoding`, `missing_verified_readout_time`, `geometry_requires_explicit_reconciliation`. `[]` means compatible metadata, which is not the same as a verified correction. |
| `framewise_displacement(parameters, radius_mm=FD_RADIUS_MM)` | Power-style FD from an N×6 MCFLIRT array (rotations in radians first, then mm). First value 0. Module constants `FD_RADIUS_MM = 50` and `FD_THRESHOLD_MM = 0.3` are shared by the pilot, its review plot and its verifier. |
| `sha256(path)`, `write_json(path, value)`, `command(args, log, *, env=None, timeout=3600, cwd=None)` | Shared helpers. `write_json` is atomic and rejects NaN. `command` raises `RuntimeError` on a nonzero exit. |

```python
from pie.imaging.fmri import classify_run, framewise_displacement
classify_run((64, 64, 40, 240), 2.5)   # 'rest_candidate'
classify_run((64, 64, 40, 10), 2.5)    # 'short_reference_candidate'
classify_run((64, 64, 40, 80), 2.5)    # 'insufficient_duration'
framewise_displacement(params)         # params: N x 6 -> array of N, first 0.0
```

### Split temporal volumes (`pie.imaging.fmri_assembly`)

Some classic (non-enhanced) DICOM series convert to one 3D file per time point.
`assemble_temporal_volumes` rebuilds the 4D run only when a header audit proves the
temporal order. Filenames and series numbers never define it.

```python
from pie.imaging.dicom_audit import audit_archive_headers
from pie.imaging.fmri import write_json
from pie.imaging.fmri_assembly import assemble_temporal_volumes

audit = audit_archive_headers("/data/ida/download.zip", record["series_prefix"],
                              expected_count=record["dicom_entries"])
write_json("/data/audit/<IMAGE_ID>.json", audit)
assemble_temporal_volumes("/data/converted/1/<IMAGE_ID>", "/data/audit/<IMAGE_ID>.json",
                          "/data/assembled/1/<IMAGE_ID>")
```

`assemble_temporal_volumes(conversion_dir, header_audit, output_dir, *, max_memory_bytes=1024**3, timing_tolerance_seconds=0.001, allow_verified_trigger_time_partition=False)`

Required evidence (any failure raises and writes nothing):

- The audit and the conversion refer to the same CRC-checked members, with no duplicate SOP UIDs.
- Every header has a positive integer `TemporalPositionIdentifier`. Positions are 1…N with
  none missing, and `NumberOfTemporalPositions` equals N. Enhanced/multiframe input is rejected.
- Geometry and timing constants (UIDs, orientation, pixel spacing, rows/columns, slice
  thickness/spacing, TR, TE) are identical across all headers. Each volume is a complete,
  regularly spaced slice stack with the same coverage.
- One `AcquisitionTime` per volume, unique across volumes. Consecutive acquisition
  timestamps must differ by the DICOM TR within `timing_tolerance_seconds` (maximum 1 ms).
- Converted volumes map one-to-one to temporal positions by acquisition time. Their
  sidecars agree apart from `SeriesNumber`, `AcquisitionTime` and `BidsGuess`, and their
  TR/TE match the DICOM headers.
- Peak memory, estimated as `(2N+2) × voxels × 8` bytes, stays within `max_memory_bytes`.

Outputs in `output_dir` (which must not already exist): `assembled.nii.gz` (float64,
scale 1/0, physical voxel values checked exactly after writing), `assembled.json`,
`conversion.json` (the original conversion record with `outputs` replaced, so downstream
code treats it like a conversion), and `assembly.json` (ordered inputs, identity, output
hashes). Missing phase-encoding sign, readout time or slice timing is left missing.
Original conversion outputs are untouched.

`allow_verified_trigger_time_partition=True` is for a reviewed series whose converter
split volumes on `TriggerDelayTime`. It additionally requires each converter value to
match that volume's raw DICOM `TriggerTime` (ms, ±0.51). The per-volume values stay in
`assembly.json`, not as a run-level scalar. See
[dcm2niix issue 395](https://github.com/rordenlab/dcm2niix/issues/395).

### FSL technical pilot (optional)

A bounded check that a run is worth preprocessing. It is **not** analysis-ready BOLD:
there is no slice timing, susceptibility correction, nuisance regression or connectivity.

| Function | Does | Writes (under `output`) |
|---|---|---|
| `motion_pilot(nifti, sidecar, output, *, fsl_dir, discard_seconds=10, resume=False)` | Requires `rest_candidate` with ≥100 volumes after discarding `ceil(discard_seconds/TR)`. Runs MCFLIRT, then BET (`-f 0.3`) on the mean. Mask must reach ≥1000 voxels. | `trimmed.nii.gz`, `motion.nii.gz`, `motion.par`, `motion.mat/`, `mean.nii.gz`, `mean_brain_mask.nii.gz`, `motion_qc.tsv` (`framewise_displacement_mm`, `dvars_raw`), `qc.json` (mean/median/max FD, `fd_radius_mm`, `fd_threshold_mm`, `fraction_fd_gt_0p3`, `seconds_fd_le_0p3_excluding_first`, median native tSNR, median raw DVARS). FD summaries other than max exclude frame 0, whose FD is fixed at 0. |
| `validate_motion_resume(nifti, output, discard)` | For `resume=True`: proves the existing trimmed and motion images and all per-volume transforms match the source. | nothing |
| `alignment_pilot(t1, motion_dir, output, *, fsl_dir)` | BET on T1, then 6-DOF normalised-MI FLIRT of the mean BOLD brain. Not BBR. | `t1_brain*.nii.gz`, `bold_in_t1.nii.gz`, `bold_to_t1.mat`, `bold_mask_in_t1.nii.gz`, `alignment.png`, `alignment.json` (`mask_dice`, `t1_mask_covered_fraction`) |
| `render_pilot_review(motion_dir, alignment_dir, output, *, title="")` | Multi-slice anatomy/BOLD panel and FD/DVARS traces, with the threshold line at the record's `fd_threshold_mm`. | `spatial_review.png`, `motion_review.png` |
| `verify_pilot_metrics(motion_dir, alignment_dir)` | Recomputes FD, DVARS, tSNR, Dice and transform orthogonality from saved files, without rerunning FSL. Uses the `fd_radius_mm`/`fd_threshold_mm` recorded in `qc.json`. Records written before those keys existed keep the legacy `seconds_fd_le_0p3` (frame 0 included) and are verified under that definition. `motion_pilot` never rewrites a completed `qc.json`. | nothing |

## 2. BIDS export (`pie.imaging.fmri_bids`)

```python
from pie.imaging.fmri_bids import export_subject

record = export_subject("/data/bids", "P001",
                        "t1.nii.gz", "t1.json", "bold.nii.gz", "bold.json",
                        reference="ref.nii.gz", reference_sidecar="ref.json")
record["correction_pair_exported"], record["unresolved_reasons"]
```

`export_subject(bids_root, subject, t1, t1_sidecar, bold, bold_sidecar, reference=None, reference_sidecar=None, *, discard_seconds=10, reference_volumes=6)`

- `subject` is an alphanumeric BIDS label without `sub-`. The export is **single-session**,
  with no `ses-` level. For several visits, use one BIDS root per `EVENT_ID` or a label
  that encodes the visit (e.g. `P001BL`), and keep the label → (PATNO, EVENT_ID) mapping yourself.
- The BOLD must be a `rest_candidate` and the T1 must be 3D. The BOLD file is copied
  byte-for-byte, and its sidecar gains `TaskName: "rest"`.
- A reverse-PE reference becomes a fieldmap pair only when `phase_encoding_pair` returns
  no reasons, the reference has volumes left after `discard_seconds`, and both echo times
  are present and equal within 1e-5 s. Otherwise the reasons are recorded and no `fmap/`
  is written. Other reasons: `no_reference`, `reference_has_no_post_initialization_volumes`,
  `missing_echo_time`, `different_echo_time`.
- Each fieldmap image is the mean of up to `reference_volumes` volumes after the first
  `ceil(discard_seconds/TR)`, taken from the BOLD (`dir-forward`) and the reference
  (`dir-reverse`). Both carry `B0FieldIdentifier` and `IntendedFor`, and the BOLD gets
  `B0FieldSource`. A short reference never becomes a resting run.

Output:

```
bids/dataset_description.json            (written once: BIDSVersion 1.10.0, raw)
bids/sourcedata/P001.json                (identity, reasons, reference_means, output hashes)
bids/sub-P001/anat/sub-P001_T1w.{nii.gz,json}
bids/sub-P001/func/sub-P001_task-rest_bold.{nii.gz,json}
bids/sub-P001/fmap/sub-P001_dir-{forward,reverse}_epi.{nii.gz,json}   (only if verified)
```

The identity includes the **resolved source paths**. Moving the raw inputs makes a rerun
fail with "source/settings differ" instead of silently re-exporting.

## 3. fMRIPrep (`pie.imaging.fmriprep`)

### Configuration

JSON keys are exactly the `FMRIPrepConfig` fields. An unknown key raises `TypeError`.

| Field | Default | Meaning / rule |
|---|---|---|
| `bids_dir` | required | Must contain `dataset_description.json` and each `sub-<label>/`. |
| `output_dir`, `work_dir`, `cache_dir` | required | All four directories must be distinct and none nested in another. |
| `participants` | required | Unique alphanumeric labels **without** `sub-`. |
| `backend` | `"native"` | `"native"` or `"apptainer"`. |
| `executable` | `"fmriprep"` | fMRIPrep binary (native) or Apptainer binary. |
| `container_image` | `None` | Existing local `.sif`; required for Apptainer. PIE never pulls. |
| `expected_version` | `None` | Must appear as a whole token in `--version` output (`25.2.5` does not match `25.2.50`). |
| `nprocs` | `4` | `--nprocs`. |
| `omp_nthreads` | `1` | `--omp-nthreads`, between 1 and `nprocs`. Also sets `OMP_NUM_THREADS` and `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`. |
| `memory_mb` | `12000` | `--mem-mb`. ≥ 1. |
| `output_spaces` | `["MNI152NLin2009cAsym:res-2", "T1w"]` | `--output-spaces`. The MNI output is what connectivity uses. |
| `surface_reconstruction` | `false` | `false` adds `--fs-no-reconall`. `true` requires `fs_license_file`. |
| `fs_license_file` | `None` | Must exist if set. Bound read-only at `/license.txt` in Apptainer. |
| `random_seed` | `42` | `--random-seed`. |
| `extra_args` | `[]` | Appended verbatim. Rejects `-w`, `--work-dir`, `--participant-label`, `--config-file`, `--fs-license-file` (and `_` / `=value` spellings), so the recorded configuration stays authoritative. |
| `derivatives` | `{}` | `{name: path}` for fMRIPrep `--derivatives` (precomputed input). Each path needs `dataset_description.json` and must not equal or nest with the BIDS input, output, work or cache directory. Cannot be combined with `-d`/`--derivatives` in `extra_args`. |

```json
{
  "bids_dir": "/data/my-study/bids",
  "output_dir": "/data/my-study/derivatives/fmriprep",
  "work_dir": "/scratch/my-study/work",
  "cache_dir": "/scratch/my-study/cache",
  "participants": ["P001", "P002"],
  "backend": "native",
  "executable": "fmriprep",
  "expected_version": "25.2.5",
  "nprocs": 4,
  "omp_nthreads": 1,
  "memory_mb": 12000,
  "output_spaces": ["MNI152NLin2009cAsym:res-2", "T1w"],
  "fs_license_file": "/opt/freesurfer/license.txt",
  "random_seed": 42
}
```

For Apptainer, merge in:

```json
{"backend": "apptainer", "executable": "/usr/bin/apptainer",
 "container_image": "/data/software/fmriprep-25.2.5.sif"}
```

### CLI

```bash
python -m pie.imaging.fmriprep --config settings.json --dry-run   # validate, print {"argv": [...]}
python -m pie.imaging.fmriprep --config settings.json             # run, print the run record
```

| Flag | Meaning |
|---|---|
| `--config CONFIG` | Required. JSON file with `FMRIPrepConfig` fields. |
| `--dry-run` | Validate and print the argument list. Creates no directories and runs nothing. The BIDS directory and participants must still exist. |

Native dry run for the config above, abbreviated (paths are single argv entries, so
spaces are safe):

```
fmriprep /data/my-study/bids /data/my-study/derivatives/fmriprep participant
  --participant-label P001 P002 -w /scratch/my-study/work --nprocs 4 --omp-nthreads 1
  --mem-mb 12000 --output-spaces MNI152NLin2009cAsym:res-2 T1w --random-seed 42 --notrack
  --fs-no-reconall --fs-license-file /opt/freesurfer/license.txt
```

### Python API

```python
import json
from pie.imaging.fmriprep import FMRIPrepConfig, build_command, run_fmriprep

config = FMRIPrepConfig(**json.load(open("settings.json")))
argv, env = build_command(config)          # validate(); no writes
result = run_fmriprep(config)              # execute or return the completed record
```

| Call | Behaviour |
|---|---|
| `FMRIPrepConfig.validate()` | Enforces every rule in the table above. |
| `FMRIPrepConfig.paths()` | Resolved `bids_dir`, `output_dir`, `work_dir`, `cache_dir`. |
| `build_command(config, *, version_only=False)` | Returns `(argv, env)`. No shell and no writes. `env` sets `TMPDIR=<work>/tmp`, `TEMPLATEFLOW_HOME=<cache>/templateflow`, `XDG_CACHE_HOME=<cache>/xdg`, BLAS threads 1. Apptainer adds `APPTAINER_CACHEDIR=<cache>/apptainer` and `APPTAINER_TMPDIR=<work>/tmp`. |
| `run_fmriprep(config)` | See below. Returns the run record. |
| `container_runtime`, `execute_recorded`, `tree_hashes` | The argv/environment builder, the once-per-identity execution and the file hashing behind `run_fmriprep`, shared with `xcpd.run_xcpd`. |

Apptainer mapping: `exec --cleanenv --containall --home <cache>/home:/home/pie`,
BIDS → `/data:ro`, output → `/out`, work → `/work`, `<work>/tmp` → `/tmp`,
cache → `/cache`, license → `/license.txt:ro`, each derivative → `/derivatives/<name>:ro`.
TemplateFlow and XDG caches point inside `/cache`. PIE does not download images, install
packages, mount disks or configure Docker.

`run_fmriprep`:

1. Creates output, work, `<work>/tmp` and `<cache>/{home,templateflow,xdg,apptainer}`.
2. Runs `--version` and checks `expected_version`.
3. Builds an identity from the config, the version string, the container SHA-256 and (when
   `derivatives` is set) the SHA-256 of every derivative file. `derivatives` is left out
   when empty, so checkpoints created before that field existed keep their identity.
4. `<work>/pie_fmriprep_identity.json` must match that identity. **A work directory
   belongs to one configuration.** Changing any setting, the version or a derivative file
   requires a new `work_dir`.
5. If `<output>/pie_provenance/<identity-sha256>/execution_completed.json` exists, returns
   it without running anything.
6. Otherwise writes `configuration.json` (identity + argv), streams output to
   `attempt-<UTC>.log` and writes `attempt-<UTC>.json` (return code, times). A nonzero
   exit raises `RuntimeError`; logs and the work directory stay for fMRIPrep's native resume.
7. Requires `<output>/sub-<label>.html` for every participant, then writes
   `execution_completed.json` with `reports` and `scientific_qc_pass: false`.

The completion record prevents accidental duplicate runs. It does not certify image
bytes or quality: inspect the reports, SDC, registration, coverage and confounds.

### Reusing anatomical derivatives

```python
from pie.imaging.fmriprep_reuse import snapshot_anatomical_derivatives

snapshot_anatomical_derivatives("/data/prior/fmriprep", "/data/reuse/anatomy", ["P001"])
# then in the config: "derivatives": {"anatomy": "/data/reuse/anatomy"}
```

`snapshot_anatomical_derivatives(source, target, participants)` copies only
`sub-<p>/anat/` and `sub-<p>/ses-*/anat/`, plus `dataset_description.json` (which must say
`DatasetType: derivative`). It rejects symlinks and any anatomical-folder file whose
name contains `_bold`, `_boldref`, `_epi`, `_confounds`, `from-bold` or `to-bold`. Its
manifest is `target/sourcedata/pie_anatomical_snapshot.json`, and an existing target is
re-verified file by file.

Byte identity is not a QC pass. Before reuse, establish identical source T1, compatible
preprocessing and a successful anatomical review. Keep the snapshot limited to the
participants being run, because `run_fmriprep` hashes the whole derivative dataset into
the identity. Keep it outside the BIDS root as well (not under `bids/derivatives/`), because
validation rejects a derivative that equals or nests with the BIDS input. Use a separate
work/output directory per processing variant.

### Staging inputs and checkpoints (`pie.imaging.staging`)

| Function | Use |
|---|---|
| `snapshot_files(source, destination, files, *, max_bytes, guard=None, progress=None)` | Stage a `{relative_path: sha256}` manifest without a directory scan. Existing destinations must match, and mismatched `.partial` files are kept for inspection. |
| `snapshot_tree(source, destination, *, max_bytes, guard=None, progress=None, include=None)` | Copy a quiesced tree (e.g. a work directory) with verified bytes, timestamps, hardlinks and unfollowed symlinks. A journal (`<destination>.snapshot.json`) allows verified resume. `include` limits the copy to named immediate children. |

`max_bytes` is the caller's budget (distinct inodes for `snapshot_tree`). The optional
`guard()` callback enforces the caller's storage policy. Neither function mounts disks or
makes a live, changing workflow consistent: stop writers first. A migrated
fMRIPrep/Nipype checkpoint also needs compatible saved paths and configuration.

On I/O-sensitive Apptainer deployments, moving `work_dir` does not move reads from an
image, runtime or `SESSIONDIR` on slow storage (`apptainer buildcfg` shows them). See the
upstream [filesystem guidance](https://apptainer.org/docs/admin/1.5/installation.html#filesystem-support-limitations).

## 4. Visual review (`pie.imaging.fmri_qc`)

```python
from pie.imaging.fmri_qc import spatial_montage

spatial_montage(mask_path, {"atlas": ("atlas.nii.gz", 0.5, "lime")},
                "review/sub-P001_mask.png", title="sub-P001")
```

| Function | Output |
|---|---|
| `spatial_montage(background, overlays, output, *, title='', mask=None)` | 3×5 PNG (axial, coronal, sagittal at 12/30/50/70/88 % of the support extent). Each title gives the world (RAS mm) coordinate of the displayed support centre on that slice, so it stays correct on oblique grids, where a voxel slice has no single world coordinate. `overlays` maps a legend label to `(path, contour_level, color)`. Overlays are resampled to the background grid (linear), and `mask` (nearest) defines the support, falling back to background > 0. Registration is never estimated. Returns the PNG path. |
| `render_report_states(svg, output_dir, *, output_width=1600)` | fMRIPrep's before/after SVGs animate between two layers, so a static render hides one. Writes `<stem>_background.png` and `<stem>_foreground.png` (or `<stem>_static.png`). Hidden layers stay in the tree because reports share resources between them. Needs `cairosvg`. |

Neither assigns a QC pass.

## 5. Connectivity (`pie.imaging.fmri_connectivity`)

Downstream denoising of fMRIPrep output. It is not a replacement for preprocessing or
visual QC. Supply the MNI-space BOLD, its brain mask, the confounds TSV/JSON, an atlas
already in the **same physical standard space**, and the atlas-ID → network mapping.
Resampling only reconciles voxel grids. It is not registration.

```python
from pie.imaging.fmri_connectivity import ConnectivityConfig, extract_connectivity
from pie.imaging.fmri_data import BOLDImageCache

func = "derivatives/fmriprep/sub-P001/func/"
inputs = [func + "sub-P001_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
          func + "sub-P001_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
          func + "sub-P001_task-rest_desc-confounds_timeseries.tsv",
          func + "sub-P001_task-rest_desc-confounds_timeseries.json",
          "atlas_MNI152NLin2009cAsym_res-2.nii.gz"]
label_networks = {1: "DMN", 2: "DMN", 3: "SMN", 4: "SMN"}   # every nonzero atlas ID

with BOLDImageCache() as cache:            # one 4D decompression for both variants
    primary = extract_connectivity(*inputs, label_networks, "connectivity/sub-P001/primary",
                                   tr=2.5, bold_cache=cache)
    gsr = extract_connectivity(*inputs, label_networks, "connectivity/sub-P001/gsr", tr=2.5,
                               config=ConnectivityConfig(global_signal_regression=True),
                               bold_cache=cache)
primary["numerical_qc_pass"], primary["exclusion_reasons"], primary["network_features"]
# True, [], {'DMN__DMN': ..., 'DMN__SMN': ..., 'SMN__SMN': ...}
```

With no study atlas, use the default cortical parcellation, already on fMRIPrep's `MNI152NLin2009cAsym` `res-2`
grid (97 x 115 x 97): Schaefer et al. 2018, 400 parcels in 7 networks (`Vis SomMot DorsAttn SalVentAttn Limbic Cont
Default`), the cortical part of XCP-D's "4S" atlas set. `striatal_rois` (5b) adds the CIT168 striatum in the same space.

```python
import nibabel as nib
from pie.imaging.atlases import schaefer400_mni2009c

atlas, label_networks = schaefer400_mni2009c()       # (label image, {1..400: network}); downloaded once, sha256-checked
nib.save(atlas, "schaefer400_7net_MNI152NLin2009cAsym_res-2.nii.gz")   # pass this path as the atlas input above
```

### `ConnectivityConfig` (frozen dataclass)

The defaults are explicit so that they get justified per study; they are not universal. They follow the
benchmarks: 24 motion parameters + aCompCor + censoring is among the best-performing strategies on motion–connectivity
(QC-FC) correlation and its distance dependence at a moderate loss of temporal degrees of freedom (Ciric et al. 2017,
doi:10.1016/j.neuroimage.2017.03.020; Parkes et al. 2018, doi:10.1016/j.neuroimage.2017.12.073), and global signal regression is the variant that removes
the most residual motion (Wang et al. 2024, doi:10.1371/journal.pcbi.1011942), so run it as the sensitivity analysis.
ICA-AROMA left fMRIPrep in 23.1.0 and is not used. Before reporting, check QC-FC (`qc_fc`, below) and lost degrees of
freedom on your own data. PPMI 2.0 rs-fMRI is multiband (MB4, TR 1 s): breathing contaminates its motion parameters
and inflates FD (Fair et al. 2020, doi:10.1016/j.neuroimage.2019.116400), so at TR ≤ 1.5 s FD is computed from
notch-filtered motion parameters, as XCP-D does. FD per frame still scales with TR; treat protocol as a batch variable.

| Field | Default | Why |
|---|---|---|
| `discard_seconds` | `10` | Drops `ceil(10/TR)` frames before T1 equilibrium, plus any frame flagged in fMRIPrep's `non_steady_state_outlier*`. Must cover row 0: fMRIPrep writes `n/a` for the first FD, DVARS and motion derivatives, and a retained NaN is an error, not imputed. |
| `fd_threshold` | `0.3` mm | Frames with FD above this are motion spikes. A common Power-style scrubbing threshold. FD is fMRIPrep's column, or the filtered FD below. |
| `motion_filter_max_tr` | `1.5` s | At this TR or shorter, FD comes from `filtered_fd`: motion parameters notch-filtered over the respiratory band, then Power FD (50 mm radius). `0` disables. |
| `respiratory_low_bpm` / `respiratory_high_bpm` | `12` / `28` | The notch band in breaths/min (0.2–0.47 Hz): XCP-D's recommendation for ages 65–80. Same filter as XCP-D 26.2.0 (`iirnotch` at the centre, Q = centre/width, one `filtfilt` pass, band folded below Nyquist). |
| `minimum_segment_frames` | `5` | Retained stretches shorter than this are censored too (Power et al. 2014; XCP-D `--censor-between 4`). |
| `dvars_threshold` | `1.5` | Frames with `std_dvars` above this are spikes, which catches signal jumps FD misses. |
| `censor_before` / `censor_after` | `1` / `2` | Also censors 1 frame before and 2 after each spike. Motion contaminates the BOLD signal for several seconds after the movement (Power et al. 2014 convention). |
| `minimum_seconds` | `300` | At least 5 min of retained data. Shorter scans give unreliable correlation estimates. |
| `minimum_residual_dof` | `60` | Retained frames minus design rank. Prevents a large nuisance model from "explaining" a short series. |
| `minimum_parcel_coverage` | `0.9` | A parcel needs ≥90 % of its voxels covered by valid BOLD signal, or it is NaN (its edges too) and left out of the network means. A truncated field of view (cerebellum, inferior temporal lobe, brainstem) would otherwise yield a parcel mean from its remaining edge. One such parcel no longer excludes the scan. |
| `compcor_components` | `5` | Top 5 aCompCor components from the **combined** WM+CSF mask, by singular value. |
| `global_signal_regression` | `False` | Adds `global_signal` and `global_signal_derivative1`. Off by default because GSR changes the sign distribution of correlations. Run it as a separate sensitivity variant. |

### Steps

`nuisance_design(confounds, metadata, tr, config=ConnectivityConfig())` → `(design, keep, audit)`

- `keep` is a boolean mask over all frames. A frame is dropped if it is initial or
  non-steady, if it is a spike (`FD > fd_threshold` or `std_dvars > dvars_threshold`), if it
  falls inside a spike's censoring window, or if its retained stretch is shorter than
  `minimum_segment_frames`. At TR ≤ `motion_filter_max_tr` the FD is `filtered_fd(motion, tr)`.
- Design columns: intercept, linear trend, 24 motion regressors (`trans_*`/`rot_*`, each
  with `_derivative1`, `_power2` and `_derivative1_power2`), top `compcor_components`
  `a_comp_cor_*`, every `cosine*` (fMRIPrep's high-pass basis), and optionally GSR.
- aCompCor columns count only if the confounds JSON records `Method: aCompCor`,
  `Mask: combined` and `Retained: true`. Too few, or no `cosine*` column, raises. The
  denoising model is never silently weakened.
- The design is returned **for retained frames only** and fitted on them. Regressing
  across censored frames would let spikes set the nuisance betas. Columns are
  norm-scaled, which doesn't change their span.
- `audit` keys: `retained_frames`, `retained_seconds`, `initial_discarded`, `spike_frames`,
  `short_segment_frames`, `motion_filtered`, `nuisance_columns`, `nuisance_rank`, `residual_dof`,
  `mean_fd` (fMRIPrep's unfiltered FD, all non-initial frames, as XCP-D's `mean_fd`), `mean_fd_filtered`
  (when filtered), `retained_indices`, `temporal_qc_pass`.

`residual_connectivity(series, design, networks)` → `(residual, edges, network_means)`

- OLS residuals of `series` (retained frames × parcels) on `design`. All regressors are
  fitted simultaneously, not sequentially.
- `edges`: Fisher z (`arctanh`, r clipped to ±(1−1e-7)) of the upper triangle in
  `np.triu_indices(P, 1)` order.
- `network_means`: mean z per unordered network pair, keyed `"<A>__<B>"` with names
  sorted (the same helper, `network_means(edges, networks)`, averages XCP-D's matrices). Within-network
  means exclude self-edges. An all-NaN parcel column (no coverage) gives NaN edges that are left out; a
  pair with no edge left is NaN. `networks` must be nonempty strings; anything else raises `ValueError`,
  since `1` and `"1"` would otherwise collide in the keys. A constant residual parcel raises.

```python
design, keep, audit = nuisance_design(confounds_df, confounds_json_dict, tr=2.5)
residual, edges, means = residual_connectivity(parcel_series[keep], design,
                                               ["DMN", "DMN", "DMN", "SMN", "SMN", "SMN"])
edges.shape, sorted(means)   # (15,), ['DMN__DMN', 'DMN__SMN', 'SMN__SMN']
```

`extract_connectivity(bold, brain_mask, confounds_tsv, confounds_json, atlas, label_networks, output_dir, *, tr, config=ConnectivityConfig(), bold_cache=None)`

- Checks: all images have mm units; confounds rows equal BOLD volumes; mask grid equals the
  BOLD grid; atlas nonzero IDs equal `label_networks` keys exactly (positive ints); every
  network name is a nonempty string. All checks run before `output_dir` is created, so a
  rejected call leaves nothing on disk.
- **Coverage** is computed on the atlas's own grid. It is the fraction of each parcel's
  voxels whose nearest BOLD voxel is inside the brain mask with a finite, non-constant
  time series. Voxels outside the field of view count as uncovered.
- Parcel signal is the mean over valid voxels (all frames, stored). Edges are computed from
  retained frames after residualisation. A parcel below `minimum_parcel_coverage` is NaN and listed in
  `low_coverage_parcels`; network pairs left without an edge are `null` in `network_features`.
- A scan with fewer than two covered parcels, or failing the temporal gate, gets
  `numerical_qc_pass: false`, an `exclusion_reasons` entry (`insufficient_parcel_coverage`,
  `insufficient_retained_time_or_residual_dof`) and no features. Excluded scans are
  recorded, never imputed.

Outputs in `output_dir`:

| File | Content |
|---|---|
| `connectivity.json` | `identity` (input paths + SHA-256, `config`, `tr`, `label_networks`, `implementation_sha256`, `data_reader_sha256`), `temporal` (the audit above), `parcel_coverage` `{id: fraction}`, `minimum_coverage`, `low_coverage_parcels`, `numerical_qc_pass`, `visual_qc_required: true`, `exclusion_reasons`, `outputs`, and on pass `network_features` `{"A__B": z or null}`. |
| `connectivity.npz` | Only on pass: `parcel_timeseries` (T×P, all frames; NaN columns for uncovered parcels), `retained_indices`, `residual_timeseries`, `edges`, `labels` (sorted atlas IDs). |

Rerunning with an identical identity verifies the `.npz` hash and returns the saved record.
The identity includes the hashes of `fmri_connectivity.py` and `fmri_data.py`, so **editing
either file invalidates existing outputs** ("inputs/settings changed"). Write to a new
`output_dir`; do not delete old records to get past the check.

`BOLDImageCache()` holds at most one read-only float32 4D image, confined to the creating
process/thread. `get(path)` fails if the file changed since it was read. `close()` and
context exit release it, and `.loads` counts reads. It changes I/O only: results are
byte-identical with and without it (tested). The same cache can serve mean-image
rendering via `cache.get(bold_path)`.

### Local measures: ALFF, fALFF, ReHo

`local_measures(bold, brain_mask, confounds_tsv, confounds_json, atlas, *, tr, config=ConnectivityConfig(),
band=(0.01, 0.08))` returns parcel means of voxel-wise ALFF (mean amplitude in `band`, divided by its brain-wide mean: mALFF, since raw amplitude
is in scanner intensity units; Zang et al. 2007), fALFF (band
over whole-spectrum amplitude; Zou et al. 2008) and ReHo (Kendall's W over each voxel's 27-voxel neighbourhood; Zang
et al. 2004) as `alff_<id>`, `falff_<id>`, `reho_<id>`, plus the same `temporal` audit as `extract_connectivity`: the
censoring and nuisance regressors are identical. The spectrum is the Lomb-Scargle periodogram of the retained frames:
interpolating censored frames instead inflated fALFF with the censored fraction (0.14 → 0.22 on white noise at 35 %
censored), i.e. with motion. ReHo uses the unsmoothed residuals of the retained frames. The building blocks are `alff_falff(series, tr,
keep, band)` and `reho(data, mask, keep)`. Memory: several GB for a 2 mm brain and 600 frames (the BOLD, the residuals
and the ReHo rank array are all held). On 4 PPMI fMRIPrep scans (16–40 s each) fALFF and ReHo were lowest in the limbic
network (susceptibility dropout) and highest in control, attention and default networks.

## 5b. Striatal and basal-ganglia-network connectivity

The same module also carries the two approaches the Parkinson's literature uses for the striatum. Both
need BOLD already in MNI152NLin2009cAsym (`--output-spaces MNI152NLin2009cAsym`); grids are reconciled
by nearest-neighbour resampling, never by registration. Outcome labels are never read here.

**Region-based.** Mean BOLD in anatomical striatal regions from the bundled CIT168 atlas, correlated
with cortical parcels and averaged within network:

```python
from pie.imaging import fmri_connectivity as fc

roi_img, names = fc.striatal_rois(target_img=bold_img)     # resampled onto the BOLD grid
names  # {1: 'caudate_l', 2: 'caudate_r', 3: 'putamen_l', 4: 'putamen_r', 5: 'accumbens_l', 6: 'accumbens_r'}

design, keep, audit = fc.nuisance_design(confounds_df, confounds_json_dict, tr=2.5)
series, labels, counts = fc.roi_timeseries(bold_img, roi_img, mask_img=brain_mask)
z = fc.seed_network_connectivity(series, [names[k] for k in labels],
                                 parcel_series, parcel_networks, design=design, keep=keep)
z["putamen_l__SomMot"]             # Fisher-z, averaged over that network's parcels
```

`design, keep` are `nuisance_design`'s: seeds and parcels are residualised together on the retained
frames, as `extract_connectivity` does. `group_ica(..., designs=, keeps=)` and
`dual_regression(..., design=, keep=)` take the same. Omit them only for input that is already denoised
and censored (XCP-D's `desc-denoised_bold`). Regions with fewer than `min_voxels` usable voxels, and NaN
parcels, give `NaN` and are left out of network means. `striatal_rois(putamen_split_y=0.0)` cuts each
putamen at the coronal plane y = 0 (close to the anterior commissure) into `putamen_anterior_*` and
`putamen_posterior_*`; the posterior, sensorimotor putamen is where PD dopamine loss starts. CIT168 at its
0.25 threshold is generous (each caudate borders the lateral ventricle), so erode the regions or intersect
them with a grey-matter mask when partial volume with CSF matters.

**Network-based**, as in Szewczyk-Krolikowski 2014, Rolinski 2016 and Droby 2025: spatial group ICA on
temporally concatenated data, the basal ganglia network identified by overlap with an independent
template, then back-projected to each subject by dual regression.

```python
reference = fc.load_reference_maps("bgn_reference_maps.nii.gz", mask_img,
                                   expected_sha256="<sha256 registered with the analysis plan>")
weights, provenance = fc.bgn_weights(datasets, striatum_mask_1d, roi_labels_1d, names,
                                     reference=reference, designs=designs, keeps=keeps)
weights[0]   # {'caudate_l': ..., 'putamen_l': ...};  provenance: path, sha256, component, template_dice

# the steps, for one subject at a time (memory):
ranked = fc.select_component(reference[0], striatum_mask_1d)   # best Dice first
timecourses, subject_maps = fc.dual_regression(one_subject, reference[0], design=design, keep=keep)
fc.roi_means(subject_maps[ranked[0]["component"]], roi_labels_1d, names)
```

**Maps.** Group ICA of the analysed sample defines the network from the same subjects later compared on
it, which is circular. `bgn_weights` uses a frozen external map set whenever `reference` is given and
warns otherwise. `load_reference_maps(path, mask_img, expected_sha256=None)` reads a 4D NIfTI (one
component per volume) already on the analysis grid and records its path and SHA-256; a different
checksum raises. Candidates: a published group-ICA set (e.g. UK Biobank's rfMRI group maps, resampled once
to MNI152NLin2009cAsym 2 mm with TemplateFlow's MNI152NLin6Asym→2009cAsym transform and then hashed), or
maps from a PPMI subsample held out of every analysis. The template that picks the BGN must be
independent of the patients too (Griffanti et al. 2016); the CIT168 striatum qualifies. Inspect the
chosen map visually, as the published analyses did.

**Dual regression** is FSL's `dual_regression` v0.6, checked against `fsl_glm` (stage 1 within 1e-9, stage
2 within 4e-7 on synthetic float32 data; the values are pinned in `test_fmri_striatal.py`). Stage 1
regresses every volume on the maps with maps and volume demeaned over voxels (`fsl_glm --demean`) and
returns those time courses (`dr_stage1`). Stage 2 regresses every voxel on them, both demeaned over time,
the time courses scaled to unit SD (n − 1) when `des_norm=True` (FSL `<des_norm> 1`), and returns the
parameter estimates (`dr_stage2`). The data are not variance-normalised; earlier PIE versions z-scored
every voxel, which FSL does not do. Deliberate differences: the caller supplies the common mask (FSL
builds it from voxels with nonzero temporal SD in every subject), FSL's z-statistics and `randomise` stage
are not reproduced, and `design, keep` residualise the retained frames first, which for stage 2 equals
fitting the nuisance regressors simultaneously because the stage-1 time courses of residuals are already
orthogonal to the design.

## 5c. XCP-D denoising and parcellation (`pie.imaging.xcpd`)

The primary path. XCP-D (PennLINC) is the community post-processor of fMRIPrep output; PIE pins one
release, records its identity the way `run_fmriprep` does, and parses its outputs.

```bash
# once; the image is several GB, so put Apptainer's cache and build space on a disk with room
export APPTAINER_CACHEDIR=/data/apptainer-cache APPTAINER_TMPDIR=/data/apptainer-tmp
apptainer build /data/software/xcp_d-26.2.0.sif docker://pennlinc/xcp_d:26.2.0

python -m pie.imaging.xcpd --config xcpd_ppmi2.json --dry-run   # validate, print argv
python -m pie.imaging.xcpd --config xcpd_ppmi2.json             # run, print the run record
```

```json
{"fmriprep_dir": "/data/ppmi/derivatives/fmriprep", "output_dir": "/data/ppmi/derivatives/xcp_d_tr1",
 "work_dir": "/scratch/xcpd_tr1/work", "cache_dir": "/scratch/xcpd_tr1/cache",
 "participants": ["P001", "P002"], "repetition_time": 1.0,
 "backend": "apptainer", "executable": "apptainer", "container_image": "/data/software/xcp_d-26.2.0.sif",
 "nprocs": 8, "omp_nthreads": 2, "memory_mb": 24000}
```

**One protocol per configuration.** `repetition_time` must match every participant's fMRIPrep
`desc-preproc_bold.json` (validated), because the motion filter depends on it: run PPMI-1 (TR 2.4 and
2.5 s) and PPMI 2.0 (TR 1 s) as separate configurations and output directories. Fields, paths, backends,
`expected_version` (default `26.2.0`), `extra_args` and the work-directory identity follow `FMRIPrepConfig`
(section 3). The identity also holds the SHA-256 of every file under each participant's fMRIPrep folder, so
a rerun fMRIPrep needs a new XCP-D work directory. `extra_args` cannot repeat a flag PIE sets. In the
container, TemplateFlow points at the templates baked into the image (`/home/xcp_d/.cache/templateflow`),
so nothing is downloaded; the image's atlases stay at its hardcoded `/home/xcp_d/.cache/xcp_d`, which PIE's
`--home` bind (at `/home/pie`) leaves visible.

Settings (every result-changing flag is passed explicitly, so a mode default changing between releases
cannot move results):

| Setting | Value | Why |
|---|---|---|
| Release | 26.2.0 (17 Aug 2026) | Latest stable; first with `--censor-between`. 26.1.x changed only multi-run concatenation, which single-run PPMI sessions do not use. |
| `--mode linc --file-format nifti` | | Single-run adult data, censored outputs, no run concatenation, LINC QC. NIfTI because PIE runs fMRIPrep with `--fs-no-reconall` (no surfaces, no CIFTI). `abcd` mode forces interpolated outputs and run concatenation. |
| `--nuisance-regressors` | `36P` | 6 motion + WM + CSF + global signal, each with derivative and squares. With censoring, among the best on QC-FC and its distance dependence (Ciric et al. 2017); GSR removes the most residual motion (Wang et al. 2024). Sensitivity: `acompcor` (no GSR), closest to PIE's own 24P + aCompCor path. |
| `--despike y` | | AFNI despiking before regression (linc default). |
| Motion filter | TR ≤ 1.5 s: `notch`, 12–28 bpm, order 4; otherwise `none` | Breathing aliases into multiband realignment parameters and inflates FD (Fair et al. 2020). 12–28 bpm is XCP-D's band for ages 65–80, PPMI's range. At TR 2.4/2.5 s the band folds onto most of the 0–0.2 Hz spectrum, so no filter there. PIE's own path uses the same band and filter. |
| `--fd-thresh` | `0.3` mm | Same threshold as PIE's own path, on filtered FD at TR 1 s and raw FD otherwise. FD per frame scales with TR: check QC-FC per protocol. |
| `--censor-between` | `4` | Retained stretches of ≤ 4 frames are censored: Power et al. 2014's "fewer than 5 contiguous frames". |
| `--min-time` | `300` s | 5 minutes retained, as PIE's own gate. Below it XCP-D writes no BOLD outputs for the run. |
| `--output-type censored` | | High-motion frames are removed from every output; correlations use retained frames only. |
| Band-pass | 0.01–0.08 Hz, order 2 | Conventional; needed for ALFF. |
| `--smoothing` | `6` mm | Affects only the smoothed copies (`desc-denoisedSmoothed`, smoothed ALFF). PIE parses unsmoothed outputs, so the striatum is not blurred into the ventricles. |
| `--atlases` | `4S456Parcels 4S156Parcels` | 4S456: Schaefer 400 (PIE's default cortex) + CIT168 subcortex, HCP thalamus, cerebellum, amygdala/hippocampus. 4S156: Schaefer 100, a coarser and more reliable network level. |
| `--min-coverage` | `0.5` | XCP-D default: parcels below are NaN, never a scan exclusion. |
| `--dummy-scans auto`, `--head-radius 50` | | fMRIPrep's non-steady-state frames; adult FD radius. |

**Features.** `xcpd_features(xcpd_dir, subject, *, atlases=("4S456Parcels",))` returns one row:

| Columns | Content |
|---|---|
| `fmri_xcpd_available` | `False` (and nothing else) when XCP-D wrote no BOLD for the run (`--min-time`, failure): a visible exclusion. |
| `fmri_xcpd_qc_*` | XCP-D's LINC QC TSV: `mean_fd` (unfiltered), `mean_fd_post_censoring`, `mean_relative_rms`, `max_relative_rms`, `mean_dvars_initial/final`, `num_dummy_volumes`, `num_censored_volumes`, `num_retained_volumes`, `fd_dvars_correlation_initial/final`, plus `fmri_xcpd_qc_retained_seconds`. BIDS entity columns are dropped. |
| `fmri_xcpd_fc_<atlas>_<A>__<B>` | Mean Fisher z per network pair of XCP-D's Pearson matrix. Groups are the atlas table's `network_label`, else `atlas_name` (CIT168Subcortical, ThalamusHCP, ...), as XCP-D's own plots. |
| `fmri_xcpd_malff_<atlas>_<group>`, `fmri_xcpd_reho_<atlas>_<group>` | Parcel ALFF divided by its mean across parcels (XCP-D's ALFF is in scanner units), and ReHo, averaged per group. |
| `fmri_xcpd_coverage_min_<atlas>`, `fmri_xcpd_parcels_nan_<atlas>` | Lowest parcel coverage; parcels XCP-D blanked for coverage (left out of every mean). |

`xcpd_striatal_features(xcpd_dir, subject, *, atlas="4S456Parcels", putamen_split_y=0.0)` recomputes
PIE's striatal seed connectivity on XCP-D's unsmoothed, censored `desc-denoised_bold`: CIT168 caudate,
anterior and posterior putamen and accumbens per hemisphere against the atlas's cortical networks (XCP-D's
parcel time series of the same BOLD), keyed `fmri_xcpd_striatal_<seed>__<network>`. Nothing is regressed again:
XCP-D's denoising is the model. It needs the BOLD in MNI152NLin2009cAsym (fMRIPrep must not also output
MNI152NLin6Asym, which XCP-D would prefer) and about 2.6 GB of memory for 600 frames. The basal-ganglia
network on XCP-D output is `dual_regression(masked(denoised_img, mask_img), reference_maps)` per subject,
without `design, keep`. `xcpd_edges(xcpd_dir, subject, atlas=...)` returns (parcel names, Fisher-z edges)
for `qc_fc`.

Limitations: output names and QC columns were checked against the 26.2.0 source, not a real run; an
upgrade means rechecking them. XCP-D censors on FD only (PIE's own path also uses DVARS). Tian and the other
built-in atlases are not parsed.

## 5d. Motion benchmark (QC-FC)

```python
from pie.imaging.fmri_connectivity import parcel_centroids, qc_fc
from pie.imaging.xcpd import xcpd_edges, xcpd_features

edges = np.vstack([xcpd_edges(root, s)[1] for s in subjects])
mean_fd = [xcpd_features(root, s)["fmri_xcpd_qc_mean_fd"] for s in subjects]
report = qc_fc(edges, mean_fd, covariates=np.column_stack([age, sex]),
               centroids=parcel_centroids(atlas_img, atlas_index_in_matrix_order))
report["percent_significant_fdr"], report["median_abs_qcfc"], report["distance_dependence_rho"]
```

`qc_fc(edges, mean_fd, *, covariates=None, centroids=None, alpha=0.05)`: across scans, the (partial)
correlation of mean FD with every edge (Ciric et al. 2017; Parkes et al. 2018). Returns `n_scans`,
`n_edges`, `n_edges_excluded` (edges with a NaN in any scan), `median_abs_qcfc`, `percent_significant`
(uncorrected), `percent_significant_fdr` (Benjamini–Hochberg), `distance_dependence_rho` (Spearman rho of
QC-FC with inter-parcel distance; negative when motion inflates short-range coupling, Power et al. 2012)
and the per-edge `qcfc`. Run it per protocol. The atlas image for the centroids is XCP-D's
`atlases/.../*atlas-4S456Parcels*_dseg.nii.gz`, labels in `index` order.

## 5e. Validation plan

None of this is a PD-vs-HC gate: resting-state group differences in PPMI did not reproduce (Badea et al.
2017), so a null group effect neither validates nor fails the pipeline.

1. **End to end, 100 scans per protocol** (PPMI-1 TR 2.4 s and 2.5 s, PPMI 2.0 TR 1 s), spread over sites
   and scanners, from IDA ZIP to feature row. Count every exclusion by reason (no verified reverse-PE pair,
   fMRIPrep failure, XCP-D `--min-time`, coverage) and review every fMRIPrep and XCP-D report (distortion
   correction, registration, coverage, carpet plots). Report pass rates and retained time per protocol.
2. **QC-FC per protocol** on the 4S456 matrices, partial on age and sex, with distance dependence. Target:
   no worse than the 36P-with-censoring results of Ciric et al. 2017 (few significant edges, distance
   dependence near zero), reported with the retained time it cost. On PPMI 2.0, compare with and without
   the notch filter: filtering should restore retained time without raising QC-FC.
3. **Split-half reliability on PPMI 2.0**: split each scan's retained frames into first and second halves,
   compute matrices and features per half, and report edge-wise and feature-wise ICC(3,1) and subject
   identification accuracy. Adjacent halves overstate between-session reliability; use short-interval
   repeat visits where they exist. Features with ICC < 0.4 are not used as predictors.
4. **Known biology, not a gate**: posterior putamen–sensorimotor coupling
   (`fmri_xcpd_striatal_putamen_posterior_{l,r}__SomMot`) against the same hemisphere's posterior-putamen DaT SBR
   (PIE's posterior half or PPMI's Xing post-commissural putamen), partial on age, sex, protocol and mean FD,
   within PD and across all participants. Lower coupling with lower SBR is expected; report the effect and
   its interval either way.

## 6. Table keyed on PATNO / EVENT_ID

PIE has no fMRI table builder. `connectivity.json` is per scan and carries no PATNO or
visit. Build the table from your own label mapping. Keep the QC columns and let excluded
scans have NaN features rather than dropping them, so the exclusion stays visible.

```python
import json
from pathlib import Path
import pandas as pd

records = {("sub-P001", "BL"): Path("connectivity/sub-P001/primary/connectivity.json")}
patno_of = {"sub-P001": 1}      # caller-owned mapping from BIDS label to PATNO

rows = []
for (subject, event_id), path in records.items():
    result = json.loads(path.read_text())
    row = {"PATNO": patno_of[subject], "EVENT_ID": event_id,
           "fmri_numerical_qc_pass": result["numerical_qc_pass"],
           "fmri_exclusion_reasons": ";".join(result["exclusion_reasons"]),
           "fmri_retained_seconds": result["temporal"]["retained_seconds"],
           "fmri_mean_fd": result["temporal"]["mean_fd"],
           "fmri_min_parcel_coverage": result["minimum_coverage"]}
    row |= {"fmri_" + pair: z for pair, z in result.get("network_features", {}).items()}
    rows.append(row)
fmri = pd.DataFrame(rows)
# PATNO, EVENT_ID, fmri_numerical_qc_pass, ..., fmri_DMN__DMN, fmri_DMN__SMN, fmri_SMN__SMN
```

From XCP-D the row is already a dict; add the keys the same way:

```python
from pie.imaging.xcpd import xcpd_features, xcpd_striatal_features
row = {"PATNO": patno_of[subject], "EVENT_ID": event_id}
row |= xcpd_features(root, "P001") | xcpd_striatal_features(root, "P001")
```

To obtain `EVENT_ID` from a scan date, use
`pie.imaging.link.link_sessions_to_events(sessions, ppmi_dir, max_months=3)` (columns
`patno`, `session_date`). It matches the MRI table's visit month and returns `"UNK"`
beyond `max_months`. Visual QC decisions are a separate caller-owned column; join them
before modelling.

`pie.experiment.prediction` residualises every feature on intracranial volume unless its
prefix is in `NON_ICV_PREFIXES` (`nm_`, `dwi_`, `new_nm_`, `new_dti_`, `fmri_`). `fmri_` is
in that list, so connectivity features are not residualised on intracranial volume. If your
analysis wants that covariate on them, change the constant or rename the columns.

## Viewer ingestion (`scripts/prepare_viewer_fmri.py`)

This converts selected raw BOLD series for [Brain Explorer](fmri_viewer.md) display. It is
not an input to the pipeline above.

```bash
venv_imaging/bin/python scripts/prepare_viewer_fmri.py \
    --archives /data/ida/part1.zip /data/ida/part2.zip \
    --collection /data/ida/collection.csv --images <IMAGE_ID> <IMAGE_ID> \
    --output /data/viewer_fmri
```

| Flag | Meaning |
|---|---|
| `--archives` | One or more IDA ZIPs (member layout `<root>/<PATNO>/<series>/<date>/<IMAGE_ID>/<file>`). |
| `--collection` | IDA collection CSV (`Image Data ID`, `Description`, `Acq Date`, `Subject`, `Visit`, `Group`, `Sex`, `Age`). |
| `--images` | Explicit unique image IDs. Each must have a BOLD-like description (`fmri`/`bold`/`resting`, not DTI/diffusion/revb0/localizer/GRE-MT) and a date. |
| `--output` | Output root. |

Unlike `convert_archive_series`, this runs dcm2niix with `-m n` and **requires exactly one
output** per series. Writes `<Subject>/<IMAGE_ID>/<IMAGE_ID>_bold.{nii.gz,json}` with a
`conversion.json` receipt, merges scan entries into `manifest.json`, and writes
`fmri_archive_inventory.json` (archive reconciliation against the CSV). The uncompressed
selection is capped at 3 GB (`max_bytes`, not a CLI flag).

## Tests

```bash
venv_imaging/bin/python -m pytest tests/test_fmri.py tests/test_fmri_assembly.py \
    tests/test_fmri_bids.py tests/test_fmri_connectivity.py tests/test_fmriprep.py \
    tests/test_fmriprep_reuse.py tests/test_fmri_qc.py tests/test_fmri_striatal.py tests/test_xcpd.py -q
```

All synthetic. No fMRIPrep, XCP-D, FSL, dcm2niix run or participant data is needed; external
commands are mocked, and XCP-D outputs are a synthetic tree in the 26.2.0 layout.
