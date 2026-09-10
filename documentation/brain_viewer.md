# PIE Brain Explorer

A local research workstation for participant-specific neuroimaging: interactive
3D volume rendering, anatomical slices, regional segmentation, acquisition-date
navigation and explicitly registered overlays. The browser uses
[NiiVue](https://niivue.com/docs/); the Python service reads existing PIE products.
No external service receives images, no template brain substitutes for a
participant, and no generative 3D modeling is used for anatomical geometry.

## Run

From the repository root:

```bash
# One-time dependencies (the existing imaging environment is suitable).
venv_imaging/bin/python -m pip install -r pie/imaging/viewer/requirements.txt
npm --prefix brain-viewer ci

# Build the frontend and serve everything on localhost.
bash scripts/run_brain_viewer.sh
```

Open **http://127.0.0.1:8765**. Stop with Ctrl+C. The server binds to loopback;
this is a single-user local viewer, without a multiuser authentication layer.
For a fresh environment: `python3 -m venv venv_imaging`, then the dependency command
above. A GPU is not needed by the API; browser rendering requires WebGL2. Use a
current Chrome/Chromium or Firefox with hardware acceleration enabled.

## September 9 exploration and review upgrade

**fMRI update:** four native BOLD/EPI examples, frame playback, voxel signal
traces, descriptive temporal maps and diagnostics are now available. See
[fMRI controls, source inventory and limits](fmri_viewer.md). The current service
inventory is **1,806 participants / 3,053 scans**. Use **Compare visits** in the
top navigation or the shortcut beside the acquisition timeline for two-pane MRI
inspection; participant 116869 supplies the existing two-visit MRI example.

- **Inside the brain:** load cached, participant-specific DKT/aseg boundary meshes;
  independently fade left/right hemisphere shells and cerebellum/brainstem, select
  deep nuclei, or focus on the striatum. These are native-voxel segmentation
  boundaries, not validated pial surfaces or cortical-thickness estimates. MRI
  volume mode remains available. Meshes are hidden in measured slice review.
- **Four-view:** explicit 2×2 layout, named panel-enlargement buttons, linked
  crosshairs, double-click to enlarge/return and all three single-plane controls.
  Integer label intent is now preserved in prepared atlases, fixing scalar
  rendering of labels; the atlas outline toggle uses the label shader.
- **SPECT presets:** Anatomy context, Signal only and Alignment review. Manual
  hide/show of SPECT permits layer comparison without automatic flashing.
  Numeric color scales and a bounded sampled histogram show the display cutoff;
  histogram scope is explicit (native nonzero FOV versus MRI-mask samples).
- **Review notebook:** timestamped reviewer observations and inspected planes,
  stored in the local browser and exportable as JSON. Entries are bound to the
  acquisition/processing fingerprint and retain geometry/transform provenance.
  Notes do not change `registration`, grant fusion approval, or constitute
  clinical validation. Browser imports without a stable fingerprint do not enable it.
- **Compare visits:** same-participant two-date MRI selection, side-by-side native
  viewing, independent windows, and optional cached 6-DOF rigid alignment.
  Native slices are not linked. In the explicitly unreviewed alignment preview,
  follow-up is linearly resampled into baseline geometry and positions can be
  linked. No scaling, deformable registration, intensity normalization or disease
  progression estimate is introduced. Per-pane errors/retry and comparison
  provenance exports are provided.
- **Exports:** PNGs retain dates, unreviewed status, modality/units, active
  representation, cutoff/hidden-signal status and anatomical-label context.
  View JSON also records the selected structure representation and fingerprint.

The optional local manifest at `Imaging/derived/viewer_collection/manifest.json`
is loaded automatically by the service; an explicit `--manifest` takes precedence.
`scripts/prepare_viewer_followup.py` prepared **only** participant 116869's existing
T1 series I1548259 (2022-01-24) and I1678052 (2023-02-22). Source archives are
unchanged; these are native head MRIs without new brain extraction/segmentation.
That earlier addition brought the service to 1,805 participants / 3,049 scans.

The two parts of fMRI collection 1 now reconcile exactly: 387 + 531 = 918 image
IDs, with no missing or duplicate IDs versus its CSV. Selected BOLD runs are
indexed. The separate collection 2 still needs its own reconciliation before
requesting duplicate downloads. See the
[next download checklist](viewer_next_downloads.md), also downloadable in Sample
plan. Real PET/CT validation remains data-limited. Regional trend charts require
appropriate longitudinal processing and review. Full saved-session restoration
and broader modality ingestion remain future work, not completed features.

Additional verification:

```bash
venv_imaging/bin/python -m pytest tests/test_brain_viewer.py tests/test_viewer_anatomy.py tests/test_viewer_structures.py tests/test_viewer_comparison.py -q
npm --prefix brain-viewer test
npm --prefix brain-viewer run build
```

The expanded suite includes 32 Python and 20 TypeScript tests. New tests cover
mesh RAS coordinates and edge padding, label intent, source immutability,
histogram bounds, comparison subject/date guards, unreviewed rigid previews,
independent mesh visibility, and export warnings. These are software checks,
not clinical validation of the acquired data, segmentation or alignment.

The fMRI update adds `tests/test_viewer_fmri.py` and frontend frame/trace/selection
tests: the current combined viewer suite has **36 Python and 25 TypeScript tests**.

Development, in two terminals:

```bash
venv_imaging/bin/python -m pie.imaging.viewer serve
npm --prefix brain-viewer run dev
```

The development frontend runs on **http://127.0.0.1:5178**, proxying `/api` to
port 8765. To choose another PPMI directory, pass `--ppmi-dir /path/to/study_data`.
`--manifest /path/to/scans.json` adds a persistent explicit collection. Restart
the API after changing a manifest or finishing additional processing.

## What the viewer does

- Search all locally indexed participants; filter by current clinical cohort.
  Archive research group and current cohort are displayed separately.
- Rotate with a drag, zoom with the wheel or buttons, and rotate the focused
  canvas with arrow keys. Reset restores the camera. Orientation labels use
  neurological convention and world coordinates in scanner RAS millimetres.
- Enter fullscreen with the expand button beside the camera, or press **F**
  while the viewer has keyboard focus. The same button or **Escape** exits.
  Rotation, zoom, layout controls and image capture remain available. The canvas
  resizes without reloading the scan or resetting the camera.
- Click the 3D brain to depth-pick a voxel and its anatomical label. When a surface
  voxel lies just outside the segmentation, a nearby label within 3 mm is shown
  with its explicit distance; the original voxel signal is unchanged. Slice picks
  use exact labels without this fallback. Four-view displays linked sagittal,
  coronal, axial and 3D views. The accessible region list navigates to an actual
  labelled voxel nearest the region's centroid, even for curved regions.
- Adjust window bounds, colormap, opacity, lighting, atlas opacity and a cutaway
  plane. Region volume is **voxel count × affine determinant**, an uncorrected
  segmentation volume; it is not FreeSurfer's partial-volume-corrected estimate.
- Standard 3D rendering starts with optional surface lighting **off**. Firefox
  on the workstation's NVIDIA driver was observed losing its WebGL context
  during NiiVue's Sobel/gradient lighting pass (unsupported framebuffer). Turning
  lighting off after context loss cannot restore the canvas. The viewer now
  checks lighting render-target support, detects context loss, and automatically
  creates a fresh renderer with lighting off once. Further failures show a retry
  message instead of a silent black canvas; lighting stays disabled after a reset
  for the current page session. This changes appearance, not anatomy or intensity.
- Select **actual acquisition dates** with the timeline. Repeated sequences on
  one date do not create extra timepoints. Switching modality selects the nearest
  acquired scan and updates the displayed date. Unknown/masked dates remain
  unknown. No interpolation creates fictional patient states between visits.
- View 4D frames with a separate within-scan control. BOLD/PET frame indices are
  independent of longitudinal visit dates. Raw BOLD is not an activation map.
- Export a PNG of the imaging canvas or JSON of the current view settings and
  provenance. View-state JSON does not embed patient images and is not an import
  manifest or a full saved session.

## Modalities and representations

| Modality | Supported representation | Interpretation |
|---|---|---|
| MRI | NIfTI / MGZ 3D anatomy, optional brain mask and integer segmentation | Surface appearance is volume-rendered MRI signal; no synthetic anatomy or estimated cortical thickness. |
| DTI | Scalar FA/MD/FW/FAt, mean b0 reference; `.tck` / `.trk` via a manifest | Native b0 and scalar maps share a voxel grid. FA is not tract density. Raw diffusion direction frames are not fitted tensors. |
| CT | Reconstructed NIfTI, adjustable brain/bone windows | HU labels require correctly calibrated source values. Attenuation-correction CT differs from diagnostic head CT. |
| PET | Reconstructed 3D/4D NIfTI, tracer and units metadata | Raw counts, SUV, SUVR and binding maps are not interchangeable. Keep tracer, dose, timing and processing metadata. |
| SPECT | Reconstructed 3D NIfTI | Original NM projection angles are not anatomical slices. Reconstruction is required. Local legacy PIE reconstruction is exploratory, not a quantitatively calibrated SBR volume. |
| fMRI | 4D BOLD or a separately computed 3D statistical map | Frame time is not visit time. Functional localization requires an analysis, thresholding and validated registration. |

### SPECT background and anatomical context

**MRI + SPECT preview** now supplies the missing anatomical context. When an
existing `datscan_full/datscan_sbr.csv` row names an exact `fs_image_id` belonging
to the same participant, the viewer can show that patient's masked MRI and its
region atlas with the corresponding SPECT reconstruction. It is selected by
default where available and is visibly marked **ALIGNMENT NOT REVIEWED**.
The native mode remains one click away. Missing, duplicate, invalid or
cross-participant references do not enable a preview; no nearest-date reference
or left/right convention is guessed.

The preview uses the **datscan_full reconstruction and its own stored transform**,
not the older `datscan_v5` volume used by Native SPECT. These are two processing
outputs of the same acquisition, not two visits. No new registration is estimated.
The stored ScaleVersor parameters map fixed MRI to moving SPECT in LPS. The
viewer converts that mapping to RAS and inverts it to place the SPECT voxel grid
in MRI space. A recorded pre-registration left/right array flip is applied only
when explicitly specified. The affine copy preserves source voxel values and
voxel count; GPU display sampling does not increase the effective SPECT resolution.

Gray folds and region labels are **MRI anatomy**. Colored signal is internal
SPECT viewed through the MRI in a see-through volume composite, **not SPECT
painted onto the cortical surface**. MRI opacity and SPECT opacity are independent.
The preview's color-window maximum is the 99.5th percentile of positive SPECT
samples within the MRI brain mask, avoiding contrast estimation dominated by
the large noisy field of view. Its initial 3D cutoff is 70% of that window and
MRI opacity is 80%; these are adjustable appearance presets, not diagnostic
thresholds. The 3D composite clips emission to MRI foreground while MRI is
visible. At zero MRI opacity clipping is disabled so SPECT remains visible.
Four-view and
slices remove that extra cutoff/clipping to permit full-signal alignment review.

The UI, PNG exports and view-state JSON retain the **unreviewed** status and
both acquisition dates. View-state JSON also records source and reference
geometry, the transform, processing source and flip flag. This preview does not
pass the normal `registration: verified` fusion gate and does not provide regional
SPECT quantification. Review correspondence in multiple planes before using it
for anatomical interpretation. For participant 3176, the recorded pair is SPECT
I306187 (2012-04-24) and MRI I305942 (2012-05-07), **13 days apart**.

#### Standalone SPECT

The legacy filtered-back-projection files contain low-intensity signal and
reconstruction streaks throughout the rectangular field of view. Rendering every
nonzero voxel makes the scan look like a box. The **3D signal** view therefore
starts with a **50% background cutoff relative to the selected color window**,
with a short 5%-of-window opacity ramp above it. This is a display preset, not a
validated tissue boundary, disease threshold or reconstruction correction.
The controls show the corresponding source-intensity cutoff. RGB colors and the
selected color window are unchanged; only the display transfer function's alpha
changes. Source files, affines, voxel readouts and quantitative values are untouched.
The camera is centered on above-cutoff signal using bounded, intensity-capped
sampling at load time; this is camera framing only, not a spatial transformation.

Set **3D background cutoff** to zero to remove this additional suppression, or
choose **Inspect full-signal slices**. Axial, coronal, sagittal and Four-view use
the original transfer function without this extra cutoff, including Four-view's
3D panel. MRI, DTI, PET, CT and fMRI keep their existing rendering behavior.
Cutoff state is recorded in exported view-state JSON.

A SPECT intensity boundary is not a cortical surface. Anatomical context needs
a separately acquired, correctly registered MRI/CT; the v5 transforms without
explicit references are still not used. Do not interpret apparent shapes or
intensity changes introduced by a display cutoff as anatomy or pathology.
The [EANM/SNMMI guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC7300075/)
specifically cautions that inappropriate contrast/background thresholding can
create artifacts and recommends checking the anatomical images.

### Existing PIE data

The index reads `Imaging/derived/sessions.csv` and finished FastSurfer MRI products,
`Imaging/derived/dwi` for retained diffusion maps, and reconstructed images listed
in `Imaging/derived/datscan_v5/datscan_sbr.csv`. It does not recursively ingest
every intermediate or failed research run on the external disk.

The implementation inventory contained **1,804 participants and 3,047 scans**:
1,853 MRI, 957 diffusion and 237 reconstructed SPECT. Local CT/PET/fMRI images were
not indexed; their formats are supported through import but require real sample
downloads before modality-specific validation on PPMI scans. Missing combinations
are shown as unavailable, not populated with synthetic patient images.

Prepared volumes and region tables are cached under
`Imaging/derived/viewer_cache/` (gitignored). Source files are read only. Brain
masking zeroes extracranial signal without warping, left/right flips or intensity
normalization. DTI's mean b0 is masked by its aligned segmentation when present.
LUT labels are from FreeSurfer/FastSurfer DKT + aseg, with provenance in
`pie/imaging/viewer/ATLAS_NOTICE.md`. Automated segmentations still need visual QC.

## Importing scans

**Quick import:** use **Import scan**, select `.nii`, `.nii.gz`, or `.mgz`, and
enter participant, modality and actual acquisition date. PET/SPECT also require
a tracer. Files stay in browser memory; imports disappear on reload. The browser
file-size limit is 512 MB. For larger acquisitions or persistence use a manifest.
This import path does not run skull stripping, registration, or DICOM conversion.

**DICOM:** convert one consistent acquisition series with the existing PIE
conversion pipeline or `dcm2niix -z y -b y -o /output /series_directory`. Preserve
the JSON sidecar and diffusion `.bval/.bvec` files. Never run this as a substitute
for reconstructing raw SPECT projections.

**Persistent manifest:** create a JSON file outside version control with real,
local image paths. Relative paths resolve against the manifest's directory.
All date values must be actual `YYYY-MM-DD` dates or null.

```json
{
  "version": 1,
  "subjects": [
    {"id": "example-001", "group": "Imported", "cohort": "Imported"}
  ],
  "scans": [
    {
      "id": "example-t1-baseline",
      "subject": "example-001",
      "modality": "MRI",
      "date": "2024-01-15",
      "visit": "Baseline",
      "description": "T1-weighted anatomical MRI",
      "path": "images/T1w.nii.gz",
      "mask": "images/brain_mask.nii.gz",
      "atlas": "images/labels.nii.gz",
      "atlas_name": "Participant segmentation",
      "atlas_lut": "images/labels.tsv",
      "space": "sub-example-001:baseline-T1",
      "units": "arbitrary intensity",
      "provenance": "Describe source acquisition and preprocessing here"
    },
    {
      "id": "example-pet-registered",
      "subject": "example-001",
      "modality": "PET",
      "date": "2024-01-16",
      "visit": "Baseline PET",
      "path": "images/PET_in_T1w.nii.gz",
      "space": "sub-example-001:baseline-T1",
      "registration": "verified",
      "reference_id": "example-t1-baseline",
      "tracer": "18F-FDG",
      "units": "SUV (body-weight normalized)",
      "provenance": "Record the actual calibration, transform, registration software and visual QC; do not label uncalibrated counts SUV"
    }
  ]
}
```

These are schema examples, **not provided patient data**. Omit optional mask,
atlas and LUT fields if they do not exist. A LUT is tab-delimited with columns
`ID`, `LabelName`, `R`, `G`, `B`, `A` (RGB values 0–255). Custom atlases without an
explicit LUT get numeric labels; the viewer never guesses a FreeSurfer label
namespace. Atlas and base image must share a voxel grid; resample labels with
nearest-neighbor interpolation before importing.

Declared metre or micron spatial units must be converted to millimetres before
import. Undeclared spatial units follow the conventional millimetre assumption,
with an explicit warning; verify the header before interpreting measurements.

For tractography, use `"modality": "DTI", "kind": "tracts"` with an actual `.tck`
or `.trk` file. Put it in a documented RAS coordinate frame; set the same space,
reference ID and verified registration to associate it with its anatomical MRI.
The streamlines are inferred pathways, not directly observed axons.

### Fusion rules

The viewer trusts NIfTI affines to describe geometry but **does not equate a valid
affine with successful registration**. The Registered overlay menu includes only
images with the same participant, same explicit space, `registration: verified`,
and a `reference_id` that points to the active scan. Register the images outside
the viewer, save the transformed image, and record your QC first. This is a
declaration of previously completed QC, not automatic validation by the viewer.

No transform is estimated just to make an overlay look plausible. The legacy
SPECT v5 table lacks an unambiguous reference image ID, so its transform is not
automatically applied. Native diffusion maps use their measured b0 geometry.
Across-date overlays retain both acquisition dates in the UI. Deep nuclei are
inspected as volumes/slices rather than projecting their signal onto cortex.

## Sample download plan

```bash
venv_imaging/bin/python -m pie.imaging.viewer sample-plan \
  --ppmi-dir '/media/cameron/Seagate Portable Drive/PPMI/study_data'
```

Open `Imaging/derived/viewer_sample_plan/DOWNLOAD_PLAN.md`, or the **Sample plan**
tab. `download_candidates.csv` records the evidence for each suggested PATNO;
`MRI_subject_ids.txt`, `DTI_subject_ids.txt`, `PET_subject_ids.txt`,
`SPECT_subject_ids.txt`, `CT_subject_ids.txt` and `fMRI_subject_ids.txt` can be pasted
into IDA's Subject ID field. Clinical-table month dates are search hints, not
exact acquisition timestamps.

In the supplied IDA interface, choose the modality and archive group separately,
leave visit unrestricted initially, then select baseline plus 12/24-month follow-up
where present. Use anatomical T1 alongside DTI/BOLD/PET/SPECT. Download collection
CSV and Advanced Download metadata with the images. Prefer processed/reconstructed
SPECT; "Original" can return raw projection data. For CT, the plan excludes
`CTSCAN=0` and flags diagnostic versus attenuation-correction series for review.

Current clinical cohorts and archive group checkboxes are not interchangeable.
The local tables do not establish a full CT/PET/etc. × cohort matrix. Some
legacy/genetic groups, Volunteer, Phantom and AV133 require targeted IDA queries;
Phantom represents scanner QA, not patient anatomy. Do not request CT or another
new scan for a participant just to fill this software test matrix.

## Verification

```bash
venv_imaging/bin/python -m pytest tests/test_brain_viewer.py tests/test_viewer_anatomy.py -q
npm --prefix brain-viewer test
npm --prefix brain-viewer run build
```

Tests exercise asymmetric/oblique image geometry, label volumes, mismatched
geometry rejection, 4D frame timing, all six modality manifest routes and API
file access, reference-specific fusion and exact-date navigation. Synthetic
arrays are used only as explicitly identified software test fixtures; they are
not displayed as real patient examples. Browser verification artifacts belong in
`output/playwright/` (gitignored because they can contain participant information).

Implementation verification: 26 Python tests and 16 TypeScript tests pass, and the
production bundle builds. Browser checks covered real MRI/DTI/SPECT, 3D rotation
and picking, linked region navigation, date selection, atlas/cutaway controls,
PNG export, and desktop/mobile layouts. Separate synthetic fixtures exercised
CT windows, PET fusion, tractography, 4D timing and browser-local import. The
final production smoke test reported no JavaScript errors or external requests.
The black-canvas regression was also reproduced and the standard-rendering fix
verified in Firefox on the actual NVIDIA display, not just in an isolated
software-rendered browser. Forced context-loss recovery and diffusion-metric
switching are included in the browser checks.

SPECT/fullscreen regression checks on native Firefox/NVIDIA cover participant
3167 plus PD, Control and SWEDD examples, the cutoff-off reversal, unthresholded
slice/Four-view modes, colormap changes, camera framing, fullscreen buttons and
F/Escape keys, focus restoration, resizing, and MRI/DTI switching. Unit tests
check alpha-only thresholding, modality isolation and camera-target sampling.
An acquisition retired while it is still loading no longer renders into a
destroyed context: drawing is disabled immediately, and GL release waits for
the pending startup to settle.

Anatomy-preview tests cover exact-reference discovery, duplicate/cross-subject
rejection, LPS/RAS transform direction, oblique affines, explicit flips, native
voxel-value preservation, separate acquisition dates, unreviewed status and API
access. Native Firefox/NVIDIA checks cover the actual 3176 preview, independent
layer opacity, region navigation, standalone/preview switching, fullscreen and
PNG export with provenance. These are software checks, **not clinical validation
of reconstruction quality or registration accuracy**.

Sources: [NiiVue geometry and rendering](https://niivue.com/docs/),
[PPMI data and imaging resources](https://www.ppmi-info.org/access-data-specimens/download-data),
[PPMI cohort definitions](https://www.ppmi-info.org/study-design/study-cohorts).
