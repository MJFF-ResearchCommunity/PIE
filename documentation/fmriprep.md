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
       ──extract_connectivity──► connectivity.json + .npz             (fmri_connectivity.py)
       ──caller──► table keyed on PATNO / EVENT_ID                    (recipe below)
```

| Module | What it does |
|---|---|
| `fmri.py` | CRC-checked DICOM→NIfTI conversion that keeps every converter output. Run classification, phase-encoding pair checks, FD, and an FSL motion/alignment pilot. |
| `fmri_assembly.py` | Rebuilds a 4D run from converter-split 3D volumes, only when DICOM headers prove the temporal order. |
| `fmri_bids.py` | Single-session BIDS export: T1w, one rest run and an optional verified reverse-PE fieldmap pair. |
| `fmriprep.py` | fMRIPrep backend (native or Apptainer) with config validation, version pinning, execution identity and resume. |
| `fmriprep_reuse.py` | Anatomical-only snapshot of earlier fMRIPrep derivatives, passed back in as precomputed input. |
| `fmri_qc.py` | Physical-space montages and both layers of fMRIPrep's animated SVG comparisons, for human review. |
| `fmri_connectivity.py` | Censoring, nuisance regression, parcel coverage checks and Fisher-z network connectivity. |
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
| FSL (`fsl_dir`) | pilot only | `motion_pilot`, `alignment_pilot` (`mcflirt`, `bet`, `flirt`). Not needed for the fMRIPrep path. |
| `cairosvg` | `render_report_states` only | Optional; not installed in `venv_imaging`. |
| Atlas + network table | connectivity | Supplied by the caller in the BOLD derivative's standard space. PIE ships no cortical network parcellation. |

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

### `ConnectivityConfig` (frozen dataclass)

The defaults are explicit so that they get justified per study; they are not universal.

| Field | Default | Why |
|---|---|---|
| `discard_seconds` | `10` | Drops `ceil(10/TR)` frames before T1 equilibrium, plus any frame flagged in fMRIPrep's `non_steady_state_outlier*`. Must cover row 0: fMRIPrep writes `n/a` for the first FD, DVARS and motion derivatives, and a retained NaN is an error, not imputed. |
| `fd_threshold` | `0.3` mm | Frames with FD above this are motion spikes. A common Power-style scrubbing threshold. |
| `dvars_threshold` | `1.5` | Frames with `std_dvars` above this are spikes, which catches signal jumps FD misses. |
| `censor_before` / `censor_after` | `1` / `2` | Also censors 1 frame before and 2 after each spike. Motion contaminates the BOLD signal for several seconds after the movement (Power et al. 2014 convention). |
| `minimum_seconds` | `300` | At least 5 min of retained data. Shorter scans give unreliable correlation estimates. |
| `minimum_residual_dof` | `60` | Retained frames minus design rank. Prevents a large nuisance model from "explaining" a short series. |
| `minimum_parcel_coverage` | `0.9` | Every parcel needs ≥90 % of its voxels covered by valid BOLD signal. A truncated field of view (cerebellum, inferior temporal lobe, brainstem) would otherwise yield a parcel mean from its remaining edge. |
| `compcor_components` | `5` | Top 5 aCompCor components from the **combined** WM+CSF mask, by singular value. |
| `global_signal_regression` | `False` | Adds `global_signal` and `global_signal_derivative1`. Off by default because GSR changes the sign distribution of correlations. Run it as a separate sensitivity variant. |

### Steps

`nuisance_design(confounds, metadata, tr, config=ConnectivityConfig())` → `(design, keep, audit)`

- `keep` is a boolean mask over all frames. A frame is dropped if it is initial or
  non-steady, if it is a spike (`FD > fd_threshold` or `std_dvars > dvars_threshold`), or
  if it falls inside a spike's censoring window.
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
  `nuisance_columns`, `nuisance_rank`, `residual_dof`, `mean_fd` (all non-initial frames),
  `retained_indices`, `temporal_qc_pass`.

`residual_connectivity(series, design, networks)` → `(residual, edges, network_means)`

- OLS residuals of `series` (retained frames × parcels) on `design`. All regressors are
  fitted simultaneously, not sequentially.
- `edges`: Fisher z (`arctanh`, r clipped to ±(1−1e-7)) of the upper triangle in
  `np.triu_indices(P, 1)` order.
- `network_means`: mean z per unordered network pair, keyed `"<A>__<B>"` with names
  sorted. Within-network means exclude self-edges. `networks` must be nonempty strings; anything
  else raises `ValueError`, since `1` and `"1"` would otherwise collide in the keys. A constant
  residual parcel raises.

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
  retained frames after residualisation.
- A scan failing coverage or the temporal gate gets `numerical_qc_pass: false`, an
  `exclusion_reasons` entry (`insufficient_parcel_coverage`,
  `insufficient_retained_time_or_residual_dof`) and no features. Excluded scans are
  recorded, never imputed.

Outputs in `output_dir`:

| File | Content |
|---|---|
| `connectivity.json` | `identity` (input paths + SHA-256, `config`, `tr`, `label_networks`, `implementation_sha256`, `data_reader_sha256`), `temporal` (the audit above), `parcel_coverage` `{id: fraction}`, `minimum_coverage`, `numerical_qc_pass`, `visual_qc_required: true`, `exclusion_reasons`, `outputs`, and on pass `network_features` `{"A__B": z}`. |
| `connectivity.npz` | Only on pass: `parcel_timeseries` (T×P, all frames), `retained_indices`, `residual_timeseries`, `edges`, `labels` (sorted atlas IDs). |

Rerunning with an identical identity verifies the `.npz` hash and returns the saved record.
The identity includes the hashes of `fmri_connectivity.py` and `fmri_data.py`, so **editing
either file invalidates existing outputs** ("inputs/settings changed"). Write to a new
`output_dir`; do not delete old records to get past the check.

`BOLDImageCache()` holds at most one read-only float32 4D image, confined to the creating
process/thread. `get(path)` fails if the file changed since it was read. `close()` and
context exit release it, and `.loads` counts reads. It changes I/O only: results are
byte-identical with and without it (tested). The same cache can serve mean-image
rendering via `cache.get(bold_path)`.

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

To obtain `EVENT_ID` from a scan date, use
`pie.imaging.link.link_sessions_to_events(sessions, ppmi_dir, max_months=3)` (columns
`patno`, `session_date`). It matches the MRI table's visit month and returns `"UNK"`
beyond `max_months`. Visual QC decisions are a separate caller-owned column; join them
before modelling.

`pie.experiment.prediction` residualises every feature on intracranial volume unless its
prefix is in `NON_ICV_PREFIXES` (`nm_`, `dwi_`, `new_nm_`, `new_dti_`). `fmri_` is not in
that list, so decide whether connectivity should carry that covariate.

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
    tests/test_fmriprep_reuse.py tests/test_fmri_qc.py -q
```

All synthetic. No fMRIPrep, FSL, dcm2niix run or participant data is needed; external
commands are mocked.
