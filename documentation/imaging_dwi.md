# Diffusion MRI (`pie.imaging.dwi` and friends)

Part of the [imaging layer](imaging.md). Turns PPMI's DTI download into one row per subject of tensor and
free-water measures in subcortical and nigral ROIs, in native diffusion space, using the subject's FastSurfer T1
as anatomy.

```
DTI zip(s) ──index/convert (dcm2niix)──► assemble runs ──[denoise]──[topup]──► rigid motion + bvec rotation
   ──► WLS tensor (FA/MD) + free-water (FW/FAt) ──► b0→T1 rigid, T1→MNI affine ──► CIT168 + FastSurfer ROIs ──► dwi_features.csv
                                                                          └─[--fba]─► MRtrix3 nigrostriatal tract (nst_*)
```

| Module | Role | Default path? |
|---|---|---|
| `dwi.py` | The cohort pipeline and CLI | yes |
| `fba.py` | MRtrix3 nigrostriatal fixel measures (`--fba`), standalone re-run CLI | opt-in |
| `mrtrix_shim/imp.py` | Stand-in for the `imp` module MRtrix3 3.0.x scripts need on Python 3.12 | used by `fba` |
| `dwi_refine.py` | Re-maps the atlas with ANTs SyN and recomputes ROI features from saved maps | opt-in |
| `dwi_tensor_qc.py` | WLS tensor with explicit pre-clipping physicality QC | opt-in API |
| `freewater_qc.py` | Multi-shell free water with solver-status and raw-tensor rejection | opt-in API |
| `dwi_acquisition.py` | Acquisition-aware run grouping/regridding, instrumented DIPY NLS | development only |
| `dwi_correction.py` | Single-acquisition selection, eddy command builder, geometry checks | opt-in API |

External tools: `dcm2niix` (installed into `venv_imaging`; `PIE_DCM2NIIX` overrides), SimpleITK, DIPY (the MNI152NLin2009cAsym target is bundled with PIE, so nothing is downloaded). Optional:
FSL `topup`/`applytopup` for `--fsl` (found via `$FSLDIR`, else `~/fsl` if `~/fsl/bin/topup` exists); MRtrix3
`dwidenoise`/`mrdegibbs` for `--denoise` (DIPY fallback when absent); MRtrix3 3.0.x on `PATH` for `--fba`
(`mrconvert dwi2response dwi2fod mtnormalise tckgen tckedit tckinfo afdconnectivity tcksample`); ANTsPy (`ants`)
for `dwi_refine`.

## Protocols handled

PPMI-1 single-shell (b = 1000; Siemens 64-direction mosaics, GE 32-direction, Philips 32-direction LR/RL pairs)
and PPMI-2 Siemens Prisma three-shell (b = 700/1000/2000, 64 directions each, reverse-phase b0s).

## Pipeline (`dwi.process_subject`)

1. **`convert(zip_path, prefix, out_dir)`** — `batch.convert_series` (dcm2niix) on one series; returns
   `(nii, bval, bvec, json)` tuples, dropping vendor-derived maps (dcm2niix suffixes `_ADC _FA _TRACEW _ColFA
   _TENSOR _EXP`) and anything without a `.bval`. The series index (`index_dwi` / the CLI's `dwi_index.csv`)
   already deselects descriptions containing `ADC` or starting `Reg_`, `dReg`, `eReg` (`_flag_dwi`).
2. **`assemble(runs)`** — runs with >= 6 volumes at b > 50 are grouped by shape, full affine (rounded to 1e-4)
   and `acquisition_metadata_key(meta)` = (PE direction, PE axis, readout, TE, TR, `PixelBandwidth`); the group
   with the most diffusion-weighted volumes is concatenated (the PPMI-2 shells). The full affine is in the key
   because shifted or oblique grids cannot be concatenated voxel-wise. Readout uses `TotalReadoutTime`, else
   dcm2niix's `EstimatedTotalReadoutTime` rounded to 1 µs so sub-microsecond JSON rounding does not split a
   protocol; an estimate only guards against mixing protocols, it never supplies PE polarity or qualifies a
   scan for topup/eddy. One reverse-PE acquisition on the same grid (a b0-only series or a full opposite-PE run,
   e.g. Philips LR/RL) is kept as `rev_b0` for topup. Raises on no DW run, no b0, or missing/zero gradients.
3. **`denoise_dwi(ds, work_dir=None)`** (`--denoise`) — Marchenko-Pastur PCA denoising then Gibbs-ringing
   removal on the raw volumes, before any interpolation. MRtrix3 `dwidenoise -extent 5,5,5` + `mrdegibbs` inside
   a dilated brain mask when both are on `PATH` (the default MRtrix patch reaches 7^3 on 200-volume data and costs
   CPU-hours); otherwise DIPY `mppca` + `gibbs_removal`. Lowers the noise floor that biases the free-water fit,
   single-shell most of all. Sets `denoised`.
4. **`susceptibility_correct(ds, work_dir, threads=2)`** (`--fsl`) — FSL topup on the mean main b0 and mean
   reverse b0, then `applytopup --method=jac` on every volume. Needs `PhaseEncodingDirection` and
   `TotalReadoutTime` for both acquisitions and a reverse-PE b0 (the three-shell PPMI-2 Prisma protocol: 107 subjects
   in the study's September 2026 DTI download, snapshot; GE "Ax DWI B-0 A/P"), ~3-7 min per subject; otherwise
   returns the data unchanged with
   `topup=False`. Odd-sized axes are cropped by one voxel first (`_crop_even`: `b02b0.cnf` subsamples by 2).
   eddy is not run in this path (eddy_cuda measured ~26 min per three-shell subject on an RTX 2080); see
   `dwi_correction` for the opt-in eddy command.
5. **`preprocess(ds, sampling_seed=0)`** — median-Otsu brain mask on the mean b0 and rigid Mattes-MI
   registration of every volume to the mean b0 (SimpleITK, 2-level pyramid, ~0.7 s/volume). Each DW gradient is
   reoriented with `rotate_bvec(bvec, affine, fixed_to_moving_lps)`, which converts the FSL voxel-frame bvec
   through physical LPS and back (FSL flips x for positive-determinant affines) and uses the inverse of ITK's
   fixed→moving rotation. A missing per-volume rotation biases the fit; a single common reflection does not. Sheared
   affines raise. Returns `motion_mm_mean/max`, `rotation_deg_max`, `bvecs_rotated=True`. This is volume-to-b0
   alignment, not eddy-current or slice-outlier correction.
6. **`fit_models(ds, fw_mask=None)`** — FA/MD from a DIPY WLS tensor on b <= 1050 over the brain mask. Free water
   `fw` and tissue FA `fat` only inside `fw_mask` (ROIs dilated by 2 voxels, cut to the brain): DIPY's
   `FreeWaterTensorModel` NLS (Hoy et al. 2014) on b <= 2050 when there are >= 2 non-zero shells
   (`fw_method="multishell_nls"`), else `_fw_single_shell`, a bounded voxel-wise bi-tensor fit (Cholesky tensor,
   f in [0, 0.95], `D_WATER=3e-3`) with a weak prior pulling tissue MD towards `MD_TISSUE=0.7e-3`
   (`fw_method="singleshell_prior"`). Single-shell free water is ill-posed and behaves closer to MD. A failed
   optimisation is NaN, never the prior value; `fw_fit_valid_fraction` records the fitted fraction.
7. **`register_b0_to_t1(b0_img, t1_img, t1_mask_img, sampling_seed=0)`** — rigid MI, mean b0 → brain-masked
   conformed T1 at 2 mm; returns (T1→b0 transform, metric). **`register_t1_to_mni(t1_img, t1_mask_img,
   cache_path=None, sampling_seed=0)`** — affine MI, brain-masked T1 → the bundled, checksummed
   MNI152NLin2009cAsym 2 mm template (`atlases.mni2009c_template()`, not nilearn's default, which is a 2009a
   image); with `cache_path` (`mni_cache_path(fs_subject)` =
   `<fastsurfer>/<IMAGE_ID>/mri/transforms/t1_to_MNI152NLin2009cAsym_affine_v2.tfm`, versioned so a legacy cache
   is not reused by name) it is fitted once and shared by NM and
   `embed` (a cached read returns metric NaN). The cache carries a `.json` sidecar recording the reference space
   and SHA-256, a registration version, the transform's own hash and the T1/mask hashes and sampling seed;
   **`load_mni_cache(path, expected_inputs=None)`** raises on anything unverified or mismatched and leaves the file
   in place, so a legacy 2009a cache is never silently reused.
8. **`labels_to_dwi(label_img, target, chain)`** — nearest-neighbour resampling of FastSurfer labels (`[T1→DWI]`)
   and the CIT168 atlas (`[MNI→T1, T1→DWI]`) onto the native DWI grid; the chain is inverted and applied in
   reverse, as ITK composites need. **`pauli_atlas()`** returns `atlases.cit168_mni2009c()`: the authors' CIT168
   v1.0 projection into MNI152NLin2009c, bundled with a checksum and space metadata. Nilearn's
   `fetch_atlas_pauli_2017` deterministic file is in native CIT168 space and must not enter this transform chain
   (changed 2026-09-15; see [imaging.md](imaging.md#bundled-atlas-atlasespy)).
9. **`_roi_masks` / `features(maps, rois, min_voxels=3)`** — per-ROI means (NaN below 3 finite voxels), see
   the column table below. Hemispheres come from FastSurfer's left/right putamen centroids (`_left_mask`), not
   voxel order. The SN is split at its median anterior-posterior coordinate; posterior-SN free water is the
   established nigral marker.

`process_subject(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, fsl=False, denoise=False,
fba=False, keep_preproc=False)` runs all of it and returns a flat dict; SimpleITK is limited to 2 threads per
worker. Raw conversions (`<work>/<patno>/nii`) are deleted afterwards; they are reproducible from the zips.

## CLI

```bash
venv_imaging/bin/python -m pie.imaging.dwi --zips <DTI zips> --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/dwi --workers 8 [--keep-nifti] [--fsl] [--denoise] [--fba]
# -> <derived>/dwi/dwi_features.csv (one row per subject), dwi_index.csv (series index, 'selected' column)
```

| Flag | Default | Meaning |
|---|---|---|
| `--zips` | required | LONI DTI zip(s) (`PPMI/<PATNO>/<desc>/<date>/<IMAGE_ID>/*.dcm`) |
| `--sessions` | required | `sessions.csv` from `pie.imaging.run` (PATNO → FastSurfer `image_id`) |
| `--fastsurfer-dir` | required | FastSurfer subjects dir; the earliest session with `stats/aseg+DKT.stats` is used (`batch.fastsurfer_by_patno`) |
| `--work-dir` | required | Output dir; `dwi_index.csv` is cached here |
| `--workers` | 4 | Subjects in parallel |
| `--limit` | none | Process at most N subjects this call |
| `--patnos` | none | Text file of PATNOs to process (whitespace-separated) |
| `--retry-errors` | off | Drop rows with a non-empty `error` and redo those subjects |
| `--keep-nifti` | off | Save `fa md fw fat b0 pauli_dwi aseg_dwi .nii.gz` per subject (needed by `dwi_refine`, `qc`, `fba` re-runs) |
| `--pid-file` | none | Write the runner's PID (stop a run without pattern-matching `ps`) |
| `--fsl` | off | topup/applytopup where a reverse-PE b0 and readout times exist |
| `--denoise` | off | MP-PCA + Gibbs removal before motion correction |
| `--fba` | off | Add the MRtrix3 nigrostriatal measures (`nst_*`) |
| `--keep-preproc` | off | Keep `<work>/<patno>/fba/preproc.{nii.gz,bval,bvec}` + `mask.nii.gz` |
| `--priority` | none | Text file of PATNOs to run first |

One session per subject: among selected series, the acquisition date with the most DICOM files
(`batch.session_rows`). The run is resumable: rows are appended as subjects finish and finished PATNOs are
skipped. Failures are rows with an `error` string, not crashes.

## Output columns (`dwi_features.csv`)

ROI names: FastSurfer `thalamus caudate putamen pallidum cerebellum_wm cerebral_wm` (`_l`/`_r`) and `brainstem`;
CIT168 `snc snr sn`(=SNc+SNr) `red_nucleus stn vta gpe gpi nac` (`_l`/`_r`); `sn_posterior_{l,r}`,
`sn_anterior_{l,r}`. Tissue-restricted variants `<roi>_t_<side>` (voxels with FA < 0.5 and FW < 0.7) exist for
every ROI starting `sn`, `stn`, `vta`, `red_nucleus`: the affine-mapped 2 mm atlas SN takes in cerebral-peduncle
fibres and interpeduncular CSF.

| Pattern | Meaning |
|---|---|
| `<roi>_{fa,md,fw,fat}` | ROI mean (e.g. `sn_posterior_l_fw`, `sn_t_r_fat`) |
| `<base>_mean_{fa,md,fw,fat}` | Mean of left and right for `sn_posterior sn snc snr putamen caudate sn_posterior_t sn_t snc_t snr_t` |
| `n_<roi>` | Voxel count (QC; excluded from the assembled features) |
| `nst_afd_{l,r}`, `nst_seed_success_{l,r}`, `nst_n_streamlines_{l,r}`, `nst_fa_{l,r}`, `nst_md_{l,r}`, `fba_multishell`, `fba_error` | `--fba` only, see below |
| `motion_mm_mean`, `motion_mm_max`, `rotation_deg_max`, `bvecs_rotated` | Motion QC |
| `reg_b0_t1_mi`, `reg_t1_mni_mi` | Registration metrics (negative MI; more negative is better; NaN when the MNI affine came from cache) |
| `fa_wm_median`, `fw_brain_median`, `fw_fit_valid_fraction` | Global sanity checks |
| `fw_method`, `shells`, `n_volumes`, `n_series`, `n_runs_used`, `n_rev_b0`, `topup`, `denoised`, `voxel_mm` | Acquisition / processing |
| `manufacturer`, `model`, `pe_direction`, `readout_s`, `series_desc` | Scanner metadata |
| `patno`, `acquisition_date`, `fs_image_id`, `source_image_ids`, `processing_version`, `error` | Lineage; `fs_image_id` lets `manifest` flag a T1 mismatch |

`processing_version` for new complete runs is `2026-09-09-acquisition-metadata-v3`. The manifest's QC rule
(`manifest.QC["dwi"]`): `motion_mm_max < 6`, `n_sn_l >= 3`, `n_sn_r >= 3`, `fa_wm_median > 0.25`.

## Nigrostriatal fixel measures (`fba.py`, `--fba`)

Fixel-based analysis without a population template, per subject in native diffusion space: Dhollander response
functions (whole brain), multi-shell multi-tissue CSD (WM + GM + CSF; WM + CSF on single-shell data) and
`mtnormalise` inside a box around the nigra, striatum, pallidum and thalamus (`box_margin=5` voxels, cut to the
brain mask; several times fewer voxels than the brain), then iFOD2 seeded in one hemisphere's atlas SN, required
to reach that hemisphere's FastSurfer putamen or caudate (`-include … -stop`), excluded from the other hemisphere
and the cerebellum; `-seeds 20000 -select 20000 -minlength 15 -maxlength 90 -cutoff 0.05`.

`nigrostriatal(ds, rois, maps, work, threads=2, n_seeds=20000, n_select=2000, min_streamlines=500, box_margin=5)`:

| Column | Meaning |
|---|---|
| `nst_seed_success_{l,r}` | Accepted streamlines / `n_seeds` (tract-density proxy) |
| `nst_n_streamlines_{l,r}` | `min(accepted, n_select)` |
| `nst_afd_{l,r}` | `afdconnectivity` on the first `n_select` streamlines; omitted below `min_streamlines`. AFD grows with streamline count, hence the fixed count (run-to-run CV ~1 % at 2,000 on study data; tractography is unseeded) |
| `nst_fa_{l,r}`, `nst_md_{l,r}` | Mean FA/MD sampled along all accepted streamlines (`tcksample -stat_tck mean`) |
| `fba_multishell` | Whether the 3-tissue model was used |

A hemisphere with < 3 SN voxels or < 20 striatum voxels is skipped. `--fba` adds ~2-3 min per subject on 2 threads
(~1.5 min of it CSD). AFD is b-value dependent: compare within the
manifest's `dwi_batch`. Normalised WM FODs and `.tck` files stay in `<work>/<patno>/fba/`. `fba` puts
`mrtrix_shim/` on `PYTHONPATH` for its subprocesses only: MRtrix3 3.0.x Python scripts (`dwi2response`, …)
import `imp`, removed in Python 3.12; the shim provides `find_module`, `load_module`, `load_source`. An `fba`
failure is recorded in `fba_error`; the tensor features stand.

Re-run from saved outputs (needs `--keep-nifti --keep-preproc` outputs; rewrites the `nst_*`/`fba_*` columns of
`<work>/dwi_features.csv` for those subjects):

```bash
venv_imaging/bin/python -m pie.imaging.fba --work-dir <derived>/dwi [--patnos file] [--workers 4] [--threads 2]
```

| Flag | Default | Meaning |
|---|---|---|
| `--work-dir` | required | DWI work dir |
| `--patnos` | every subject with `fba/preproc.nii.gz` | Text file of PATNOs |
| `--workers` | 4 | Subjects in parallel |
| `--threads` | 2 | MRtrix threads per subject |

Python: `fba.from_saved(subject_dir, threads=2)`, `fba.write_preproc(ds, out_dir)`.

## JHU tract measures (`dwi.fetch_jhu`, `dwi.tract_features`)

Two different tract measurements live in `dwi.py`, and they answer different questions. The
nigrostriatal fixel measures above (`--fba`) follow one tract PIE tracks itself, per hemisphere. The
JHU measures here average a map over each of the 48 labelled white-matter tracts of the ICBM-DTI-81
atlas (Mori 2005, Wakana 2007, Hua 2008), the atlas most tract-level FA studies report against.

The atlas is fetched at run time from its NeuroVault release (collection 264) rather than bundled,
because that release states no licence. Downloads are cached and sha256-checked, and the left/right
labels are verified on load.

Template space is never assumed. The atlas ships on a generic "MNI" grid whose exact flavour is
undeclared, so labels reach each subject by registering the atlas's *own* FA template — which shares
the label grid voxel for voxel — to the subject's FA map (affine, then SyN with cross-correlation). No
MNI152 variant is involved at any step, which removes the class of error in which an atlas is read in
the wrong template space.

```python
from pie.imaging import dwi

len(dwi.LABELS), dwi.LABELS[5]        # 48, 'splenium_corpus_callosum'

labels_img, atlas_fa, provenance = dwi.fetch_jhu(cache_dir)      # sha256 + laterality checked on load
subject_labels, tf = dwi.map_labels_to_subject(subject_fa, atlas_fa, labels_img,
                                               brain_mask=mask, max_resolution_mm=2.0)
qc = dwi.registration_qc(subject_fa, tf["warped_template_fa"], subject_labels, labels_img, mask)
qc["template_fa_correlation"], qc["qc_pass"]        # check this before you use the features
```

Then average any maps you like within the tracts. `fa_img`, `md_img` and the label image are your own,
all on one grid:

```python
feat = dwi.tract_features({"fa": fa_img, "md": md_img}, subject_labels, min_voxels=3)

feat["fa_superior_fronto_occipital_fasciculus_l"]   # mean FA in that tract
feat["n_superior_fronto_occipital_fasciculus_l"]    # voxels behind it
len(feat)                                           # 144 = 48 tracts x (fa, md, n)
```

Tracts with fewer than `min_voxels` usable voxels come back as `NaN`, never as a silently thin
average. `fa_floor=0.2` (the TBSS convention) restricts every tract to voxels above that FA, which
reduces partial volume with grey matter and CSF; a tract entirely below the floor becomes `NaN`.
`max_resolution_mm` defaults to `None`, i.e. registration on the native grid; pass `2.0` to register
finer reconstructions (e.g. 1 x 1 x 2 mm) on a 2 mm copy while still pulling labels onto the native
grid.

Read the QC before modelling. On PPMI 2 mm data SyN gave a template-FA correlation of 0.69 where
affine alone gave 0.54, with correspondingly higher tract FA (posterior internal capsule 0.67,
splenium 0.56). `dwi.write_provenance(path, provenance, qc)` records both beside the features.

## Deformable atlas refinement (`dwi_refine.py`)

Optional pass over `--keep-nifti` outputs: ANTs rigid b0 → T1 and SyN T1 → MNI (1 mm nilearn template,
`antsRegistrationSyNQuick[s]`, ~2 min/subject), the CIT168 atlas warped onto the b0 grid, ROI features recomputed
from the saved `fa md fw fat` maps (FastSurfer labels from the saved `aseg_dwi`). Writes
`<work>/<patno>/pauli_dwi_syn.nii.gz` and appends to `<work>/dwi_features_syn.csv` (same column names as the
feature part of `dwi_features.csv`, plus `reg_syn_mi`, which is always NaN). On a test subject it moved the nigral
centroid by under a voxel, so it is not in the default path. The T1 is the one the DWI run used
(`batch.fastsurfer_by_patno`: the earliest finished session).

```bash
venv_imaging/bin/python -m pie.imaging.dwi_refine --work-dir <derived>/dwi --sessions <derived>/sessions.csv \
    --fastsurfer-dir <derived>/fastsurfer [--workers 6] [--limit N] [--patnos file]
```

`--workers` defaults to 6. Subjects already in `dwi_features_syn.csv` are skipped. Python:
`refine_subject(subj_dir, fastsurfer_dir, syn_type="antsRegistrationSyNQuick[s]")`.

## Opt-in measurement APIs

These do not change `dwi.process_subject`. They package safeguards from the September 2026 reassessment for
callers that want explicit failure reporting instead of clipped or silently-defaulted values.

**`dwi_tensor_qc.tensor_measurements(signal, bvals, bvecs)`** — `signal` is `(n_voxels, n_volumes)`, `bvecs`
`(3, n_volumes)`. Requires an own b0 (b <= 50) and >= 12 DW volumes. DIPY log-WLS tensor; returns `md fa ad rd`
(NaN unless accepted), `accepted`, `relative_rmse`, `raw_min_eigenvalue`, `nonpositive_signal`,
`nonfinite_signal`. A voxel is accepted only with finite, positive signal, finite parameters, raw minimum
eigenvalue >= -1e-12 mm²/s and finite RMSE. Non-positive signal is floored for the solver but never accepted.
It does not estimate single-shell free water.

**`freewater_qc.fit_multishell_checked(signal, bvals, bvecs)`** — unchanged DIPY NLS on b <= 2050 (needs b0 and
>= 2 non-zero shells), via `dwi_acquisition.diagnostic_multishell`. Returns `f`, `tissue_md`, `tissue_fa`,
`physical` (positive finite signal and an accepted solve), `raw_min_eigenvalue`, `solver_status`, `diagnostics`,
`shells`. Rejects failed or nonphysical fits before eigenvalue clipping can hide them. Use one optimizer thread per
process (it temporarily patches scipy's `leastsq`). A DIPY threshold shortcut with no NLS solve is not accepted.

**`dwi_acquisition`** (development path, not the cohort default):

| Function | What it does |
|---|---|
| `load_runs(files)` | `(nii, bval, bvec, json)` tuples → run dicts; rejects dimension mismatches, non-finite or non-unit gradients |
| `acquisition_key(run)` | Shape, axes (rounded 1e-5) and `dwi.acquisition_metadata_key`; raises on sheared grids |
| `choose_group(runs, minimum_directions=12)` | Largest compatible group; every acquisition must have its own b0 |
| `regrid_volume(array, source_affine, reference_shape, reference_affine)` | Resample onto an origin-shifted grid with identical axes/spacing; changed axes raise (gradients and PE would need transforming) |
| `prepare_group(runs)` | Concatenate the chosen group, regridding origin shifts; `run_records` keeps per-run b0 indices, affines and metadata |
| `pe_row(meta)` | topup/eddy `acqparams` row; requires verified `PhaseEncodingDirection` in `i j k` (+`-`) and positive `TotalReadoutTime` |
| `diagnostic_multishell(signal, bvals, bvecs, cholesky=False, cholesky_init_floor=None)` | Per-voxel DIPY NLS exposing MINPACK status, nfev, relative RMSE and the un-clipped tissue tensor's minimum eigenvalue |
| `acceptable_tensor_fit(diag, fw, error='')` | status 1-4, finite RMSE and eigenvalue, eigenvalue >= `TENSOR_EIGENVALUE_TOLERANCE` (1e-12), 0 <= fw <= 1 |
| `positive_definite_initial_tensor(tensor_elements, floor=1e-6)` | Experimental repair of a singular WLS *initialisation* (Cholesky arm only); not a validated estimator |

A positive MINPACK return code alone is insufficient: finite outputs and residuals and a non-negative raw tensor
are also required. Direct Cholesky fitting can fail on singular initial tensors; failures are retained per voxel.
The initialisation repair does not prevent all iteration-limit failures, and none of these checks validates the
free-water biological model. Conditional averages over surviving voxels are not validated regional measures. Do
not run the instrumentation concurrently with another optimizer in the same Python process.

**`dwi_correction`** (builds commands; does not run FSL):

| Function | What it does |
|---|---|
| `choose_tensor_acquisition(runs)` | One complete acquisition nearest b = 1000 (shells 500-1500, >= 12 DW volumes, own b0); ties → more volumes, then filename. Returns `(run, selected_mask, shell)`; never borrows b0s from another acquisition |
| `slice_options(metadata, n_slices)` | `['--json=metadata.json']` only when `SliceTiming` is complete and its simultaneous-slice groups match `MultibandAccelerationFactor`; otherwise `[]` plus the reason. Unverified grouping must not enable `mporder>0` |
| `topup_config(shape)` | `b02b0.cnf` for even grids, `b02b0_1.cnf` for any odd dimension; never crops |
| `build_eddy_command(fsl_bin, metadata, n_slices, *, use_topup=False, raw_is_uncorrected, cuda=True)` | `eddy_cuda`/`eddy_cpu` (explicit, no fallback) with `--repol --ol_type=sw --mporder=0 --niter=5 --cnr_maps …`; `raw_is_uncorrected` must be `True` (no correction applied twice); needs verified PE/readout (`pe_row`) |
| `validate_corrected_geometry(raw_image, corrected_image, bvals, rotated_bvecs)` | Rejects changed shape/affine, bad b-values, missing b0 or DW volumes, non-unit rotated bvecs. Headers only |

`build_eddy_command` expects `raw.nii.gz`, `eddy_mask.nii.gz`, `acqparams.txt`, `index.txt`, `bvals`, `bvecs`
(and `topup_*` with `use_topup`) in the working directory. Feed raw images, not ones already through
applytopup or rigid motion correction; consume eddy's rotated bvecs once. The command and slice decisions
reproduced all 114 saved study attempts available at integration time. Scheduling, logs, timeouts and memory
admission stay with the caller: PIE has no end-to-end eddy cohort CLI.

```python
from pie.imaging import dwi_correction as dc

dc.topup_config((96, 96, 61))                          # 'b02b0_1.cnf'
meta = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}
cmd, record = dc.build_eddy_command("/opt/fsl/bin", meta, n_slices=60, raw_is_uncorrected=True, cuda=False)
record["slice_timing"]                                 # 'unused_unverified' (no SliceTiming)
```

```python
import numpy as np
from pie.imaging.dwi_tensor_qc import tensor_measurements

rng = np.random.default_rng(0)
g = rng.normal(size=(30, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
bvals = np.r_[0.0, 0.0, np.full(30, 1000.0)]
bvecs = np.vstack([np.zeros((2, 3)), g]).T                          # (3, n_volumes)
D = np.diag([1.5e-3, 0.4e-3, 0.4e-3])
signal = 1000 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs.T, D, bvecs.T))[None]
out = tensor_measurements(signal, bvals, bvecs)
out["accepted"], round(float(out["fa"][0]), 3)                      # (array([ True]), 0.686)
```

## Python usage

```python
from pie.imaging import dwi

ds = dwi.assemble(runs)                       # runs = [(nii, bval, bvec, json), ...] from dwi.convert
ds = dwi.preprocess(ds)                       # motion correction + rotated bvecs
maps = dwi.fit_models(ds)                     # {'fa', 'md', 'fw', 'fat', 'fw_method'}
```

Full subject: `dwi.process_subject(1, series_rows, "<derived>/fastsurfer/<IMAGE_ID>", "<derived>/dwi")`, where
`series_rows` are `dwi_index.csv` records (`zip`, `prefix`, `desc`, `date`, …) of one session.

## Tests

`tests/test_dwi.py` (run assembly, ROI construction and averaging, label-resampling direction, single-shell
free-water phantom, denoising and even cropping), `tests/test_dwi_correction.py`, `tests/test_freewater_qc.py`,
`tests/test_atlas_space.py`, and the gradient-rotation / failed-fit / reverse-b0 cases in
`tests/test_imaging_audit.py`; `test_dwi_refine_uses_the_earliest_finished_t1` in `tests/test_imaging_regressions.py`.
