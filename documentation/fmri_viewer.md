# fMRI inspection in Brain Explorer

## Open the examples

Refresh http://127.0.0.1:8765 and use **fMRI examples** in the participant browser.
The shortcuts open the full run rather than the short reference candidate.

| Participant / cohort | Date | Full resting run | Short opposite-encoding candidate |
|---|---|---|---|
| 101685 / Prodromal | 2021-07-21 | I1491269, 240 frames, PE `i` | I1491270, 10 frames, PE `i-` |
| 218968 / Prodromal | 2023-06-15 | I10253746, 240 frames, PE `i` | I10253745, 10 frames, PE `i-` |

All four converted series are 64 × 64 × 40, with TR 2.5 seconds and 3.5 mm
isotropic voxels. The 240-frame runs span frame onsets from 0 to 597.5 seconds.
The two directions are **separate acquisitions**, not concatenated, registered,
or automatically treated as equivalent resting runs. The short-run label is a
reference *candidate*, not confirmation that a particular distortion-correction
pipeline can use it without reviewing timing, coverage and protocol metadata.
Phase encoding uses NIfTI axis notation, not patient anatomical left/right.

218968 already has a same-day MRI in the viewer (I10253749). Its existence alone
does not establish EPI-to-T1 alignment. No fMRI overlay is silently added to MRI.

## Controls

- **Raw BOLD frames:** play/pause, previous/next frame, slider, display speed and
  within-scan seconds. Playback is opt-in, pauses when the page is hidden or the
  workspace changes, and stops at the last frame. Press Play at the end to restart.
  The same transport is available inside fullscreen. Display frames/second are
  playback speed, not a change to acquisition TR.
- **Selected voxel:** click a measured slice to plot all acquired samples at a
  fixed voxel. Optionally show percent from that voxel's temporal mean. No percent
  series is invented for zero/nonpositive mean. Charts retain their numerical
  axis ranges, missing values and units; clicking a chart seeks a frame. The
  named frame slider provides keyboard access to the same action.
- **Temporal mean, temporal SD and tSNR:** descriptive maps in the native EPI
  grid, with independent display windows. The maps summarize all acquired frames;
  they are not activation, connectivity, or motion-corrected tissue measurements.
- **Run diagnostics:** fixed-foreground mean signal and raw, unstandardized DVARS
  (RMS successive-frame signal differences). The foreground is a heuristic
  intensity mask, explicitly not an anatomical segmentation. No automated
  pass/fail, motion estimate, or disease interpretation is supplied.
- **Exports:** PNGs include date, representation, units and the actual frame/time
  or all-frame-summary label. Voxel JSON includes every sample, frame number,
  timing, source fingerprint, and the **sampled voxel-center RAS coordinate**.
  The potentially fractional click/crosshair coordinate is recorded separately.
  Diagnostic JSON retains methods and foreground definition.

### Visits versus frames

The acquisition timeline changes **dated series** in one viewer. **Compare visits**
in the top navigation (also linked beside the timeline) opens two MRI acquisitions
side by side. Select participant **116869** to try it. Choose baseline/follow-up
MRI and optionally prepare an **unreviewed** rigid alignment preview. Native
slices remain independent; the aligned preview can link positions. This comparison
currently supports MRI, not quantitative longitudinal fMRI comparison.

The new fMRI examples are baseline-only. Their within-scan frame sliders do not
represent disease progression or additional study visits.

## Source preservation and ingestion

The read-only inventory reconciles **387** distinct image IDs in
`First_Study_fMRI_1.zip` with **531** in `First_Study_fMRI_1_dataset.zip`: **918 total**,
no duplicate IDs, no missing/extra IDs versus `First_Study_fMRI_1_9_09_2026.csv`.
This verifies the series inventory, not the pixel integrity/QC of every archive
member. Selected DICOM files passed ZIP CRC checks while being extracted.
The collection includes diffusion and MT descriptions despite its fMRI tags;
these are not used as BOLD examples.

Only four explicit series were converted with dcm2niix. Sources are unchanged;
temporary extracted copies were cleaned up after conversion. The new helper
retains converter logs/JSON, validates 4D/TR consistency, refuses ambiguous split
outputs or conflicting existing IDs, and merges the optional local manifest
without removing the previous MRI pair. Outputs are in
`Imaging/derived/viewer_collection/{participant}/{image_id}`; descriptive maps are
cached in `Imaging/derived/viewer_cache`. No data leave this machine.

To reproduce these examples (idempotent for matching sources):

```bash
venv_imaging/bin/python scripts/prepare_viewer_fmri.py \
  --archives '/path/to/PPMI/Imaging/First_Study_fMRI_1.zip' \
    '/path/to/PPMI/Imaging/First_Study_fMRI_1_dataset.zip' \
  --collection '/path/to/PPMI/Imaging/First_Study_fMRI_1_9_09_2026.csv' \
  --images I1491269 I1491270 I10253746 I10253745 \
  --output Imaging/derived/viewer_collection
```

Replace the example paths with your own; no external drive is required.
Restart the service after changing its manifest. The example service indexes
**1,806 participants / 3,053 scans**. Only examples, not every archive series,
are converted/indexed. Choose an ingestion scope and output/cache locations
appropriate to your available storage before attempting bulk conversion.

## Scientific limits and next work

No new motion correction, susceptibility-distortion correction, slice-timing
correction, denoising, segmentation, EPI-to-T1 registration, activation model or
connectivity model has been applied. A fixed scanner voxel can contain different
tissue over time when the participant moves. Signal variation/tSNR is not a map
of neuronal activity. Descriptive maps use all frames, including initial frames;
there is no hidden frame censoring or detrending.

No more downloads are required for this raw-data inspection workflow. Advanced
functional analysis requires an explicit preprocessing/QC workflow, validated
anatomical alignment and appropriate analysis design. Existing same-day anatomy
for 218968 is a useful next input; additional cohort and longitudinal examples
should be selected separately. Full session restoration is still future work.

Metadata conventions: [BIDS MRI specification](https://bids-specification.readthedocs.io/en/v1.11.0/modality-specific-files/magnetic-resonance-imaging-data.html)
defines functional TR in seconds and NIfTI-axis phase-encoding metadata. Rendering
uses [NiiVue's 4D frame API](https://niivue.com/docs/api/niivue/classes/Niivue/).

## Verification

The combined viewer suite passes 36 Python and 25 TypeScript tests, including
native geometry/source immutability, summary formulas, missing/zero signal,
TR consistency, time-unit conversion, frame bounds, exports and preference for
full runs over short references on a matching date. The production build passes.

Firefox browser checks covered playback, both examples, short-run switching,
temporal maps, trace export and the MRI comparison shortcut. All 240 values in a
browser-exported voxel trace were independently matched to the source NIfTI at
the exported voxel-center RAS coordinate. This caught and corrected a fractional
click/voxel-center ambiguity. Chromium additionally verified fullscreen rendering,
last-frame selection and playback restart. The automated native Firefox window
exited fullscreen immediately for both MRI and fMRI; no user browser preferences
were changed, and native Firefox fullscreen persistence was not validated here.
