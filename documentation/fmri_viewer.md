# fMRI inspection in Brain Explorer

Raw, native-EPI inspection of 4D BOLD runs inside the [Brain Explorer](brain_viewer.md):
frame playback, fixed-voxel signal traces, descriptive temporal maps and run
diagnostics. Nothing here is preprocessing or analysis. No motion, slice-timing or
susceptibility correction, denoising, EPI-to-T1 registration, activation model or
connectivity model is applied.

## Getting BOLD into the viewer

| Route | Summaries (mean/SD/tSNR, diagnostics) | Persistent |
|---|---|---|
| `scripts/prepare_viewer_fmri.py` → collection manifest | yes | yes |
| Hand-written manifest entry with `"modality": "fMRI", "kind": "timeseries"` (any source, including open data; see [Non-PPMI data](brain_viewer.md#non-ppmi-data)) | yes | yes |
| **Import scan** in the browser (4D file, modality fMRI) | no, frames only | no |

### `scripts/prepare_viewer_fmri.py`

Converts explicitly named IDA BOLD series and merges them into a viewer manifest. IDA
can split one collection's download into several zips; pass them all:

```bash
venv_imaging/bin/python scripts/prepare_viewer_fmri.py \
  --archives /path/to/PPMI/Imaging/First_Study_fMRI_1.zip \
             /path/to/PPMI/Imaging/First_Study_fMRI_1_dataset.zip \
  --collection /path/to/PPMI/Imaging/First_Study_fMRI_1_9_09_2026.csv \
  --images <IMAGE_ID> [<IMAGE_ID> ...] \
  --output Imaging/derived/viewer_collection
```

| Flag | Meaning |
|---|---|
| `--archives` | One or more IDA zips, read-only. A series may appear in only one archive/session. |
| `--collection` | IDA collection CSV (`Image Data ID`, `Subject`, `Description`, `Acq Date`, `Visit`, `Group`, `Sex`, `Age`). |
| `--images` | Unique `I<digits>` image IDs to convert. |
| `--output` | Manifest folder. Use `Imaging/derived/viewer_collection` for automatic loading by `serve`. |

Why each refusal exists:

- **The description must match `fmri|bold|resting` and must not match
  `dti|diffusion|revb0|localizer|gre.?mt`, and the series must be dated.** IDA's fMRI
  tag also covers diffusion and MT series, so the checkbox is not evidence of BOLD.
- **The archive participant must equal the collection's `Subject`, and a series must not
  be duplicated across archives or sessions.** Otherwise the wrong participant's run
  could be picked silently.
- **The selection is capped at 3 GB uncompressed.** Explicit examples, not bulk ingestion.
- **`dcm2niix` (`-z y -b y -ba y -m n`) must produce exactly one NIfTI.** A split output
  is refused rather than taking the largest file. Opposite phase-encoding runs are never
  concatenated.
- **The JSON `RepetitionTime` must be positive and equal the NIfTI frame step in
  seconds.** Frame times shown in the viewer come from the header.
- **An existing output folder without a `conversion.json` receipt, or whose receipt
  names different source members, is refused.** Re-running with identical sources skips
  conversion and rewrites the same manifest entries. A changed archive never overwrites a
  converted run.

Output per series: `<output>/<PATNO>/<IMAGE_ID>/<IMAGE_ID>_bold.nii.gz`, the
`_bold.json` sidecar and `conversion.json` (source fingerprint, DICOM member count,
converter version and log). Temporary DICOM copies are deleted.

Scans (`bold-<IMAGE_ID>`) and their subjects (`collection: "PPMI"`) are merged into
`manifest.json` through the same `catalog.merge_manifest` used by
`prepare_viewer_followup.py`. Either script can run first; an existing ID with a
different source fingerprint aborts the merge without writing.

Both scripts build their entries from the same four `pie.imaging.viewer.catalog` helpers:
`read_manifest(path)` returns the existing version-1 document, or `{"version": 1, "subjects":
[], "scans": []}` when the file does not exist (invalid JSON, or any other version, raises);
`iso_date(value)` keeps only day-precision dates, returning `None` for masked or month-only
ones so the viewer never invents a date; `source_fingerprint(archive, members)` identifies
which archive members a scan came from; and `merge_manifest(path, subjects, scans)` writes the
result. Reuse them if you write your own ingestion script.

`fmri_archive_inventory.json` reconciles the archives' central directories against the
collection CSV: series per archive, overlaps, missing and extra IDs. That checks the
inventory, not pixel QC. Selected members pass ZIP CRC checks during extraction.

Snapshot, fMRI collection 1 as downloaded in September 2026:
- 387 distinct series in `First_Study_fMRI_1.zip` plus 531 in
  `First_Study_fMRI_1_dataset.zip` gave 918 series, with no duplicates and none missing
  or extra versus `First_Study_fMRI_1_9_09_2026.csv`.
- The collection also contained diffusion and MT series despite its fMRI tag.
- Full resting runs had 240 frames (TR 2.5 s, 64 × 64 × 40, 3.5 mm isotropic, span
  0–597.5 s). The short opposite-encoding references had 10 frames.

Each scan carries `metadata.example: true`, `run_role` and `short_reference`. A run with
fewer than 30 frames is labelled **Short EPI reference candidate**: typically an
opposite-phase-encoding reference, not a resting run. It is labelled a *candidate*
because using it for distortion correction still needs timing, coverage and protocol
review.

Restart the API after the manifest changes. Choose output and cache locations to suit
your storage before converting more than a few runs.

## Opening a run

Scans with `metadata.example` get **fMRI EXAMPLES** shortcuts in the participant
browser. Each is labelled with the participant's own cohort and opens the full run (a
short reference only if that is all there is). Otherwise select the participant and
the **fMRI** modality. On one date, a full run is preferred over a short reference;
**Series on this date** switches between them. A BOLD run opens in **Axial** with the
crosshair on.

A same-day MRI does not establish EPI-to-T1 alignment. No fMRI overlay is added to MRI
unless a manifest declares a `verified` registration (see
[fusion rules](brain_viewer.md#geometry-registration-and-fusion-rules)).

## Controls

- **Raw BOLD frames**: Play/Pause, previous/next frame, a frame slider showing
  within-scan seconds, and display speed of 1, 2 or 4 frames/s. Display speed is not
  the acquisition TR. Playback is opt-in, pauses when the page is hidden or the
  workspace changes, and stops at the last frame; Play there restarts from frame 1.
  The same transport sits inside fullscreen.
- **Selected voxel**: click a slice to plot every acquired sample at that fixed voxel.
  Optionally show percent from the voxel's temporal mean; this is unavailable when that
  mean is zero or negative rather than inventing a baseline. Charts show their numeric
  range and gaps for missing values. Clicking a chart seeks that frame, and the frame
  slider gives keyboard access to the same action.
- **Temporal mean**, **Temporal standard deviation**, **Temporal signal-to-noise ratio**:
  maps in the native EPI grid with their own windows (tSNR starts at 0). They summarize
  all frames and replace frame playback while selected.
- **Run diagnostics**: foreground mean signal and raw DVARS across the run, plus the
  foreground voxel count, method text and any nonfinite voxels excluded. No pass/fail,
  motion estimate or clinical interpretation is given.

Header text shows frames, TR and the sidecar's `PhaseEncodingDirection` (from
`metadata.sidecar`). Phase encoding uses NIfTI axis notation (`i`, `j-`, …), not
patient left/right.

## What the summaries compute

`pie/imaging/viewer/fmri.py`, run once per run and cached in the viewer cache as
`bold_{mean,sd,tsnr}_v1.nii.gz` and `bold_summary_v1.json`:

| Quantity | Definition |
|---|---|
| Temporal mean | Arithmetic mean over all frames (Welford; no second full-size 4D copy). |
| Temporal SD | Population SD, `ddof=0`. |
| Foreground | Largest connected component with temporal mean > 20% of the 95th percentile of positive means, finite at every frame. An intensity heuristic, not a brain mask. |
| tSNR | mean / SD inside the foreground; 0 outside it and where SD ≈ 0. |
| Foreground mean signal | Mean over the fixed foreground at each frame. |
| Raw DVARS | RMS of successive-frame differences over the same foreground; frame 1 is null. Unstandardized. |
| TR | NIfTI frame step × time-unit factor (`sec`, `msec`, `usec`); null if unknown. |

No filtering, detrending, frame censoring or dummy-frame removal: initial frames are
included. Voxels with any nonfinite sample are excluded and counted. Runs whose
`voxels × frames × 4 bytes` exceeds 1.5 GB are not summarized
(`fmri_unavailable` explains why); frame inspection still works.

The prepared scan's `fmri` object holds `frames, tr_seconds, mean_signal, raw_dvars,
foreground_voxels, excluded_nonfinite_voxels, foreground_threshold,
foreground_definition, method, warning, source_fingerprint`. The three maps are
`extra` entries with keys `mean`, `sd` and `tsnr`.

## Exports

| Export | File | Contains |
|---|---|---|
| PNG (camera) | `PIE-<subject>-fMRI-<date>.png` | Date, representation and units, and either `BOLD frame i/N · t s from first frame` or `Temporal summary of all N frames`. |
| **Export voxel signal** | `PIE-<subject>-<scan>-voxel-signal.json` | `scan_id, date, fingerprint, ras_mm, selected_location_mm, ras_voxel, tr_seconds`, and every frame's `{frame, seconds, signal, percent_from_temporal_mean}`. |
| **Export descriptive diagnostics** | `PIE-<scan>-descriptive-fmri.json` | Scan record, geometry and the full `fmri` summary including method and foreground definition. |

`ras_mm` is the centre of the exact voxel that was sampled. A click can land between
voxel centres, so the potentially fractional crosshair position is recorded separately
as `selected_location_mm`. Match exported traces to source data at `ras_mm`.

## Visits versus frames

The acquisition timeline changes **dated series**. The frame slider moves **within one
series**. Frame seconds are never visit time, and a baseline-only run's frames do not
represent progression. **Compare visits** is MRI-only; there is no quantitative
longitudinal fMRI comparison.

## Scientific limits

A fixed scanner voxel can sample different tissue over time when the participant moves.
Signal variability and tSNR are not maps of neuronal activity. Functional analysis needs
an explicit preprocessing and QC workflow, validated anatomical alignment and an
appropriate design, all outside this viewer. Full session restoration is not
implemented.

Conventions: [BIDS MRI specification](https://bids-specification.readthedocs.io/en/v1.11.0/modality-specific-files/magnetic-resonance-imaging-data.html)
(TR in seconds, NIfTI-axis phase encoding); rendering uses
[NiiVue's 4D frame API](https://niivue.com/docs/api/niivue/classes/Niivue/).

## Tests

```bash
venv_imaging/bin/python -m pytest tests/test_viewer_fmri.py tests/test_viewer_scripts.py -q
npm --prefix brain-viewer test
```

- `tests/test_viewer_fmri.py`: the summary formulas and raw DVARS over all frames;
  empty, constant and nonfinite runs; native affine, source immutability and timing of
  prepared maps; and refusal of non-BOLD descriptions and inconsistent 4D timing in the
  import script.
- `tests/test_viewer_scripts.py`: the shared, order-independent collection manifest.
- `brain-viewer/src/fmriDisplay.test.ts`: frame seconds and unknown units, RAS-buffer
  voxel sampling, percent-from-mean without a zero baseline, and export labels.
- `model.test.ts`: a full run is preferred over a short reference.
- `catalogLabels.test.ts`: example shortcuts use each participant's own cohort.
