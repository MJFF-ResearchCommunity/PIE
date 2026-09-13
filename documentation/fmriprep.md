# fMRIPrep preprocessing in PIE

PIE supports fMRIPrep as a configurable preprocessing backend through
`pie.imaging.fmriprep`. It accepts a valid BIDS dataset and runs either an
installed fMRIPrep executable or a local Apptainer image. It records version,
configuration, command, container checksum, attempt logs and report locations.
No personal paths, drive labels, UUIDs, dataset IDs or host mount operations are
part of the backend.

An external drive is **not required**. Callers choose the BIDS, output, work and
cache directories on storage compatible with their selected runtime, including
ordinary local storage. The paths below are examples, not required mount points.
Machine-specific capacity limits, drive identity checks and mount/recovery policy
belong in the calling application or deployment, not in PIE's processing API.

## Native installation

Install fMRIPrep and its required external neuroimaging tools according to the
[upstream installation instructions](https://fmriprep.org/en/stable/installation.html).
Installing only the Python package does not install all of those tools.

Save a JSON configuration with your own paths:

```json
{
  "bids_dir": "/data/my-study/bids",
  "output_dir": "/data/my-study/derivatives",
  "work_dir": "/scratch/my-study/work",
  "cache_dir": "/scratch/my-study/cache",
  "participants": ["001", "002"],
  "backend": "native",
  "executable": "fmriprep",
  "expected_version": "25.2.5",
  "nprocs": 4,
  "omp_nthreads": 1,
  "memory_mb": 12000,
  "output_spaces": ["MNI152NLin2009cAsym:res-2", "T1w"],
  "surface_reconstruction": false,
  "random_seed": 42
}
```

```bash
python -m pie.imaging.fmriprep --config settings.json --dry-run
python -m pie.imaging.fmriprep --config settings.json
```

Use participant labels without `sub-`. The four directories must be distinct
and not nested inside one another. BIDS validation remains enabled. The dry run
checks configuration and prints an argument list; it creates no directories and
does not execute fMRIPrep. Paths containing spaces remain single arguments.

## Apptainer installation

With an installed Apptainer executable and an already acquired local image,
change the configuration to:

```json
{
  "backend": "apptainer",
  "executable": "/opt/apptainer/bin/apptainer",
  "container_image": "/data/software/fmriprep-25.2.5.sif"
}
```

Merge these fields into the full configuration above. PIE binds BIDS read-only
at `/data`, derivatives at `/out`, scratch at `/work` and `/tmp`, and caches at
`/cache`. Container home, TemplateFlow and runtime caches use configured paths.
PIE does not download images, install packages, mount disks, create filesystem
images or reconfigure Docker. Those are deployment choices outside this API.
The runtime must be able to use the configured storage; filesystems that reject
Linux filenames may need an administrator-provided compatible scratch location.

For I/O-sensitive deployments, consider the location of the Apptainer executable,
helpers, image and session directory as well as BIDS/work/cache paths. Moving
only `work_dir` does not remove reads from an image or runtime installed on slow
storage. `apptainer buildcfg` reports the actual runtime paths, including
`SESSIONDIR`; PIE's cache settings do not relocate that installation-owned path.
These locations can be ordinary directories. PIE requires neither a separate
partition nor a filesystem image. See the upstream
[filesystem guidance](https://apptainer.org/docs/admin/1.5/installation.html#filesystem-support-limitations).

To stage only a caller-frozen set of inputs, avoiding recursive scans and old
working-directory transfers:

```python
from pie.imaging.staging import snapshot_files

verified = snapshot_files(
    source_root, destination_root, relative_path_to_sha256,
    max_bytes=byte_budget, guard=check_available_storage,
)
```

The manifest maps relative filenames to expected SHA256 digests. Existing
destinations must match; interrupted or mismatched partial files are preserved
and require inspection. This verifies input bytes, not scientific image quality.
The caller owns the storage budget and can omit the optional `guard` callback.

Provide a valid FreeSurfer license with `fs_license_file` (or an existing valid
license in the native/container installation). fMRIPrep 25.2.5 checks this license
even with `surface_reconstruction: false`; `--fs-no-reconall` does not remove the
license requirement. Obtain your own free license from
[FreeSurfer registration](https://surfer.nmr.mgh.harvard.edu/registration.html).
For surfaces, also set `surface_reconstruction` to true.
Additional scientific fMRIPrep switches may be supplied in
`extra_args`; work-directory, participant and license overrides are rejected so
the recorded configuration remains authoritative. See
[fMRIPrep usage](https://fmriprep.org/en/stable/usage.html) for their meanings.

## Anatomical reuse and archive conversion performance

Set `derivatives` to a mapping such as
`{"anatomy": "/data/precomputed/anatomy-only"}` to pass fMRIPrep's precomputed
derivatives input. Native paths are passed directly; Apptainer binds each package
read-only under `/derivatives/<name>`. PIE hashes the derivative inputs into the
execution identity, so changing them requires separate work. Configurations
without this optional field retain their previous checkpoint identities.

`pie.imaging.fmriprep_reuse.snapshot_anatomical_derivatives(source, target,
participants)` creates a verified independent anatomical-only dataset, excluding
functional and fieldmap folders. It rejects symlinks and misplaced functional
files. Before reuse, callers must establish identical source anatomy, compatible
preprocessing and successful anatomical review. Byte identity is not a QC pass.
Keep snapshots limited to the participants being run: execution integrity checks
hash the supplied derivative dataset. fMRIPrep may reference native anatomical
files from that read-only dataset instead of duplicating them in new outputs;
downstream review should retain the explicit anatomical-source provenance.
Use a separate work/output directory for each processing variant; do not supply
corrected functional derivatives to an uncorrected sensitivity run.

For repeated archive conversions, explicitly own a bounded ZIP index cache:

```python
from pie.imaging.archives import ZipArchiveCache
from pie.imaging.fmri import convert_archive_series

with ZipArchiveCache(max_open=1) as cache:
    for record in selected_series:
        convert_archive_series(record, output_root, archive_cache=cache)
```

The cache reuses indexes, not extracted bytes. Archive identity and member CRC
checks remain enabled; output hashes still govern resume. A million-entry ZIP
index can exceed a gigabyte, so use a small capacity and one owner thread/process.
Converter failures retain diagnostic logs under the configured output root.
Pass `scratch_root="/scratch/my-study/raw-dicom"` to `convert_archive_series`
when small-file extraction should use a separate configured filesystem. The
directory must already exist with sufficient space; PIE never substitutes
another disk. Only temporary DICOMs use it. Output staging and atomic publication
remain under `output_root`, even when the two roots are on different filesystems.

Primary/GSR QC and mean-image rendering may share a
`pie.imaging.fmri_data.BOLDImageCache` context by passing `bold_cache=cache` to
`extract_connectivity` and calling `cache.get(bold_path)` for rendering. The
cache holds one read-only float32 image, checks source identity and releases it
at context exit. This changes I/O, not denoising or coverage definitions. The new
reader is recorded in QC provenance; old QC records are preserved and must not
be overwritten to bypass an implementation-identity mismatch.

## Python API and completion semantics

`pie.imaging.fmri_bids.export_subject` exports explicitly selected anatomical,
resting-state and optional reference acquisitions. It checks full-run dimensions,
preserves raw resting images and source hashes, and exports a correction pair
only when signed phase encoding, readout, geometry and echo time are compatible.
Reference acquisitions never become resting-state feature runs. Acquisition
selection and same-session verification belong to the calling study/application.

After preprocessing and visual review,
`pie.imaging.fmri_connectivity.extract_connectivity` provides configurable
censoring, nuisance regression, atlas coverage checks and parcel/network
connectivity. Supply the atlas, its network mapping, confounds, TR and all paths.
It checks aCompCor provenance in the confounds JSON, fits nuisance regressors on
retained frames only, and records numerical QC exclusions instead of imputing
missing brain signal. The atlas must already be in the BOLD derivative's physical
standard space; grid resampling is not registration. Defaults are explicit in
`ConnectivityConfig` and must be justified for each application, not assumed
universally appropriate. Numerical QC does not replace anatomical/SDC inspection.

`pie.imaging.fmri_qc.spatial_montage` renders physical-coordinate multiview
overlays for that inspection without assigning an automatic scientific QC pass.

```python
import json
from pie.imaging.fmriprep import FMRIPrepConfig, run_fmriprep

with open("settings.json") as stream:
    config = FMRIPrepConfig(**json.load(stream))
result = run_fmriprep(config)
```

Failed attempts preserve logs and work files for fMRIPrep's native resume.
Changing settings requires a different work directory. The backend does not
declare scientific QC passed when a process exits successfully: researchers
must inspect the reports, correction, alignment, coverage and confounds, then
apply their separately specified denoising, feature extraction and evaluation.
Existing execution records prevent accidental duplicate execution under the
same configuration; they do not certify unchanged image bytes or image quality.

## Recovery of split temporal volumes

`pie.imaging.fmri_assembly.assemble_temporal_volumes(conversion_dir,
header_audit, output_dir)` can create a separate 4D image from converter-split
3D files only when a CRC-checked classic DICOM audit supplies complete temporal
positions, uniform geometry and unique acquisition-time matches. It validates
input hashes, declared TR against acquisition timestamps, metadata consistency,
and exact physical voxel preservation after serialization. Filenames never
define temporal order. Short references remain references; missing signed
phase-encoding, readout or slice timing is not inferred. Unresolved inputs fail
closed. Caller parameters control paths, memory budget and timing tolerance;
no particular disk or mount is required. Recovery alone is not preprocessing
or scientific QC approval. Original conversion outputs are preserved.

For a reviewed trigger-time partition, an explicit
`allow_verified_trigger_time_partition=True` additionally requires each
converter trigger value to match the corresponding raw DICOM millisecond tag.
Those per-volume values remain in provenance rather than an ambiguous run-level
scalar. The option does not relax acquisition timestamp/TR or voxel checks.
See [the upstream Philips split-volume report](https://github.com/rordenlab/dcm2niix/issues/395).

## Verified checkpoint staging

`pie.imaging.staging.snapshot_tree(source, destination, max_bytes=...)` stages
a quiesced working tree at caller-selected paths. It copies and verifies regular
file bytes, preserves timestamps and hardlinks, and records symlinks without
following them. Optional `include` selects explicitly named immediate children;
unrelated files are not silently copied. Its journal supports verified reuse of
completed copies after interruption. Partial files require inspection, and a
completed snapshot cannot silently grow when new source files appear.

The byte limit counts distinct regular-file inodes; an optional `guard` callback
can enforce the caller's overall storage policy. This function does not mount
filesystems, impose an external-drive requirement, or choose fallback storage.
It does not make a live, changing workflow into a consistent checkpoint. Stop
source writers first, retain the original, and separately verify required input
and container hashes. A migrated fMRIPrep/Nipype checkpoint also needs compatible
saved paths and configuration; copying a directory alone does not establish
that compatibility or scientific QC.
