# Pre-registration: TrUE-Net white-matter hyperintensity volume (PIE FastSurfer preparation) as a vascular covariate

**Status: FROZEN 2026-10-15** (design approved 15 Oct: G1 + G2, G3 optional), by the commit that changes this line,
before any gate statistic was computed, any challenge file was downloaded, or any TrUE-Net WMH volume was related to
age or vascular risk. Hashes are in the [run log](#run-log). Items still open at the freeze, and how each is handled,
are in [Open items at the freeze](#open-items-at-the-freeze). **Caveat:** whether the distributed `mwsc` weights saw
any challenge test subject is not yet known; if they did, G1 is void and the fallback applies.

## Why

WMH volume is a vascular covariate in PIE, not a synucleinopathy marker. It confounds subcortical volumes and marks
the vascular mimic behind some normal DaTscans. PIE's first WMH measure, an intensity threshold, tracked age at only
ρ 0.18 in PPMI and was superseded. The default is now TrUE-Net (Sundaresan et al. 2021, Medical Image Analysis
73:102184, doi:10.1016/j.media.2021.102184), model `mwsc`, with PIE's FastSurfer-based input preparation
(`flair --truenet-prep fastsurfer`, `flair._prepare_from_fastsurfer`) in place of TrUE-Net's FSL `fsl_anat` script.

What is known about this exact pipeline:
- **TrUE-Net with its own (FSL) preparation**, trained on the 60 MICCAI WMH Challenge training subjects and
  submitted to the challenge, scored on the 110 unseen test subjects: Dice 0.77, H95 6.95 mm, AVD 20.49 %, lesion
  recall 0.74, lesion F1 0.70 (subject-weighted averages; Sundaresan 2021 §3.4.2).
- **PIE's FastSurfer preparation against the FSL preparation**, 60 PPMI FLAIRs: median Dice 0.872, volume r 0.991,
  ICC 0.984. FastSurfer-prepared volumes are about 10 % smaller (bias −110 mm³).
- **Nothing yet** on its accuracy against manual labels with PIE's preparation, or on PPMI biology.

This pre-registration supplies both, in two gates:
- **G1, technical:** accuracy against manual labels on independent public data.
- **G2, biological:** the expected associations with age and vascular risk in PPMI.

G3, visual ratings on PPMI, is optional.

## Disclosure: what has already been seen

- **The threshold method:** age ρ 0.18 (3D FLAIR 0.24); ρ 0.52 with FastSurfer's T1-based `vol_WM_hypointensities`.
- **FastSurfer `vol_WM_hypointensities`** tracks age at ρ 0.46 in PPMI.
- **The preparation comparisons** (10, then 60 PPMI FLAIRs) compared TrUE-Net volumes between the two preparations
  only. No comparison with age, cohort or vascular risk is known (not yet confirmed by the user; open item 4).
- **The cohort run** (`Imaging/derived/flair_truenet_fs`) had written 402 of 1,340 participants when this draft was
  written, all 2D slabs so far because of run order. While drafting, only non-outcome columns of that table were read,
  as aggregate counts: `error`, vendor, 2D/3D, slice thickness, and linkage to the T1 table. The G2 loader was checked
  on the real schema with every `wmh_*` column replaced by random numbers. It printed only that the code path ran and
  that the needed columns were present.
- **Vascular-risk prevalences** (below) come from PPMI clinical tables and dates only.
- **No TrUE-Net output exists for any challenge subject.** No challenge image or label has been downloaded. The
  published challenge results (Kuijf 2019) and TrUE-Net's own test-set result are public and set G1's context.

## Measure

`wmh_mm3` from `pie.imaging.flair` (code at the freeze commit), `--method truenet --truenet-prep fastsurfer`:
- the FLAIR reoriented, brain-extracted and FAST bias-corrected exactly as `prepare_truenet_data` does;
- FastSurfer's N4 T1 (`orig_nu`) in its brain mask, rigidly registered to the FLAIR (`flirt -dof 6`);
- `truenet evaluate -m mwsc`, giving a probability map;
- the map masked by FastSurfer's cerebral white matter and WM hypointensities (aseg 2, 41, 77), then thresholded at
  ≥ 0.5 (`WMH_PROB`), with no minimum lesion size;
- volume counted on the prepared FLAIR grid.

The analysed value is `wmh_log_mm3` = log(1 + WMH mm³). Model weights: `third_party/weights/truenet/mwsc`
(`fsl-truenet_data` 2508.0; sha256 in its README).

## G1. Technical accuracy against manual labels (primary)

### Reference data (verified 15 Oct 2026)

**MICCAI WMH Segmentation Challenge 2017, test set.** Kuijf et al., "Data of the White Matter Hyperintensity (WMH)
Segmentation Challenge", DataverseNL v1.0, published 21 Dec 2022, doi:10.34894/AECRSD
(https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD). Challenge paper: Kuijf et al. 2019, IEEE
TMI 38(11):2556–2568, doi:10.1109/TMI.2019.2905770.

- **Licence:** CC BY-NC 4.0. Non-commercial use, as for the TrUE-Net weights. Cite the paper and the dataset.
- **Access:** all 1,791 files are public, with no access request. A terms-of-access click-through applies ("By
  downloading or otherwise accessing the materials, the downloader represents his/her acceptance of the Terms of
  Use"), and a guestbook may ask for name and e-mail at download. **The user accepts these terms; nothing has been
  downloaded.**
- **Size:** about 8.2 GB in all; `test/` is 5.67 GB. Only `orig/FLAIR.nii.gz`, `orig/3DT1.nii.gz` and `wmh.nii.gz`
  per subject are needed.
- **Contents:** every test subject has the manual labels (`wmh.nii.gz`: 0 background, 1 WMH, 2 other pathology)
  and a 3D T1 (`orig/3DT1.nii.gz`), which PIE's FastSurfer preparation needs. Also included: the FLAIR, a 2D T1 in
  FLAIR space, face masks, the elastix 3DT1→FLAIR transform and SPM12-bias-corrected `pre/` copies.
- **Reference standard:** STRIVE criteria. Observer O1 delineated, and observer O2 peer-reviewed in consensus. A
  voxel counts as WMH if more than 50 % of it lies inside the contour. Label 2 (lacunes, infarcts, haemorrhages) was
  dilated 1 voxel in-plane and is excluded from scoring.
- **Subjects:** memory-clinic patients (Utrecht, Amsterdam) and a memory-and-ageing cohort (Singapore). Across the
  challenge, age 70.1 ± 9.3, 50 % male.

| Site, scanner | Test n | FLAIR | 3D T1 voxel (mm) | In TrUE-Net training? |
|---|---|---|---|---|
| UMC Utrecht, 3 T Philips Achieva | 30 | 2D, 0.96 × 0.95 × 3.0 mm | 1.00 × 1.00 × 1.00 | scanner yes, subjects no |
| NUHS Singapore, 3 T Siemens TrioTim | 30 | 2D, 1.0 × 1.0 × 3.0 mm | 1.00 × 1.00 × 1.00 | scanner yes, subjects no |
| VU Amsterdam, 3 T GE Signa HDxt | 30 | 3D 0.98 × 0.98 × 1.2 mm, resampled to 3 mm axial | 0.94 × 0.94 × 1.00 | scanner yes, subjects no |
| VU Amsterdam, 1.5 T GE Signa HDxt | 10 | 3D 1.21 × 1.21 × 1.3 mm, resampled to 3 mm axial | 0.98 × 0.98 × 1.50 | **no** |
| VU Amsterdam, 3 T Philips Ingenuity PET/MR | 10 | 3D 1.04 × 1.04 × 0.56 mm, resampled to 3 mm axial | 0.87 × 0.87 × 1.00 | **no** |

**Independence.**
- The challenge assigned subjects to training (60) or test (110) at random per scanner.
- TrUE-Net's challenge submission was trained on the training set. The test labels were secret until December 2022.
- **Not yet verified (open item 1):** FSL's distributed weights (`fsl-truenet_data` 2508.0, 2025) are documented
  only as trained on "MICCAI WMH Segmentation Challenge Data". The documentation does not say whether they were
  retrained after the test labels became public. If any test subject was used, G1 is void and the fallback applies.

**Fallback if the test set cannot be used.**
- The 60 training subjects are **not** independent (they trained `mwsc`). They could give only a resubstitution
  upper bound, never a gate.
- A search on 15 Oct 2026 found no other public set with FLAIR, T1 and manual vascular-WMH labels. ISBI 2015 and
  MSSEG are MS lesions, the wrong pathology. MRBrainS13/18 include a white-matter-lesion class in a handful of
  labelled subjects; that is unverified and too small to gate on.
- The fallback is therefore in-domain manual tracing of 30–40 PPMI FLAIRs (the plan in `imaging.md`, "TrUE-Net
  validation plan" step 1). It needs a rater and is pre-registered separately.

### Procedure

The runner is written after the freeze and before any challenge file is downloaded (open item 3); it implements
exactly this procedure. For each of the 110 test subjects:
1. **T1:** PIE's pinned FastSurfer on `orig/3DT1.nii.gz`, as for PPMI T1s.
2. **TrUE-Net:** `flair.run_truenet(orig/FLAIR.nii.gz, t1, <work>, prep="fastsurfer", fastsurfer_dir=…)`, then
   probability ≥ 0.5: the cohort run's lesion definition, unchanged.
3. **Back to the label grid:** the binary prediction is resampled nearest-neighbour onto `wmh.nii.gz`'s grid
   (`fslreorient2std` only permutes axes, so this is exact). The runner asserts that `orig/FLAIR` and `wmh.nii.gz`
   share the grid.
4. **Metrics:** `analyse_flair_wmh.mask_metrics(wmh, prediction, voxel mm³)` writes one row per subject.
   `mask_metrics` follows the challenge's own `evaluation.py` (github.com/hjkuijf/wmhchallenge, commit cd4f691,
   sha256 d3366e911d0ec015ae2bef5da5b0b57858dd6b3ae08611eaedab85a63559e31b):
   - label-2 voxels are removed from the prediction;
   - Dice;
   - AVD = |V_pred − V_ref| / V_ref × 100;
   - lesion recall and lesion F1 on 3D 26-connected components.

   H95 is not computed: it is a boundary metric, and PIE uses volume. Before the freeze, `mask_metrics` is checked
   against `evaluation.py` on the training set's extra observer annotations (O3 against the reference), which are not
   test data. They must agree to 1e-6.

A subject that fails to process is kept, with its error.

Per-subject table (`--metrics`), one row per test subject:

| Column | Meaning |
|---|---|
| `subject` | challenge folder, e.g. `Utrecht/1` (public challenge numbering) |
| `scanner` | `Utrecht_PhilipsAchieva3T`, `Singapore_SiemensTrioTim3T`, `Amsterdam_GE3T`, `Amsterdam_GE1T5`, `Amsterdam_PhilipsIngenuityPETMR3T` |
| `dice`, `lesion_recall`, `lesion_f1`, `avd_pct` | `mask_metrics` |
| `vol_ref_mm3`, `vol_pred_mm3` | manual and TrUE-Net WMH volume on the label grid |
| `error` | empty, or the exception (`Type: message`) |

### Statistic and pass rule

On all 110 test subjects:
- **Median Dice ≥ 0.70.** A subject that failed to process counts as Dice 0.
- **ICC(2,1) of WMH volume ≥ 0.85**, TrUE-Net against manual, in mL (`pie.stats.agreement.icc21`). Failed subjects
  are left out and counted.
- Both conditions, on the point estimates. The 95 % CIs come from 2,000 subject-bootstrap draws (seed 20261015) and
  are reported.

**Why these values.**
- Dice > 0.70 is the conventional "excellent" level for lesion overlap (Zijdenbos et al. 1994, IEEE TMI 13:716).
  It is 0.07 below TrUE-Net's own 0.77 with the FSL preparation, which leaves room for the change of preparation
  (agreement between preparations: Dice 0.87, volumes about 10 % smaller).
- On the same test set, the challenge's top four methods averaged Dice 0.77–0.80 and ranks 6–7 averaged 0.72–0.73.
  Two extra human observers scored 0.77 and 0.79 on the training images (Kuijf 2019, Table II).
- ICC ≥ 0.85 is the volume bar fixed in `imaging.md`'s earlier plan. Volume is the quantity PIE exports.
- Precision: with 110 subjects the median Dice is estimated to about ±0.02–0.03.

**Reported regardless (no gate):**
- the challenge-style means of Dice, lesion recall, lesion F1 and AVD, beside Kuijf 2019 Table II and TrUE-Net's
  0.77 / 0.74 / 0.70 / 20.49 %;
- volume agreement: Pearson r, CCC, Bland–Altman bias and limits, and the same on log(1 + mL);
- every metric by scanner, and for the two unseen scanners pooled (n 20);
- failures by reason.

**Limits of G1.**
- The subjects are older memory-clinic patients with heavier WMH than PPMI.
- The challenge 3D FLAIRs were resampled to 3 mm slices. PIE processes PPMI 3D FLAIR at its native 1–1.2 mm, and
  PPMI 2D slabs are 5 mm.
- Only the segmentation path is tested: preparation, TrUE-Net and threshold. PIE's FLAIR series selection,
  FLAIR→T1 registration QC and periventricular/deep split are not.

G2 covers PPMI itself.

## G2. Biology on PPMI

### Sample

- **Rows:** every participant in the TrUE-Net cohort run (`flair --truenet-prep fastsurfer`, work dir
  `Imaging/derived/flair_truenet_fs`, one row per participant). Each row is the FLAIR session nearest the
  participant's earliest finished FastSurfer T1 (`batch.modality_jobs`): the baseline scan.
- **Cohorts:** all PPMI cohorts (PD, prodromal, HC, SWEDD).
- **When:** G2 runs once, after the cohort run has attempted every participant in its index with a finished T1
  (1,340 participants with a FLAIR at drafting). No partial-cohort analysis.

**Exclusions, in order, each counted by reason:**
1. a processing error (counted by exception type);
2. a row not made by `truenet` / `fastsurfer` / `mwsc`;
3. `manifest.QC["flair"]` failure, counted by rule, since a scan can fail several:
   - registration MI ≥ −0.2;
   - white matter < 200,000 mm³;
   - normal-appearing-WM MAD of 0;
   - S–I coverage < 130 mm;
4. T1 QC failure (`manifest.QC["t1"]`) or no eTIV;
5. missing sex or cohort, or an age outside 18–100.

QC pass rates are reported by FLAIR batch. Participants without a vascular-risk score leave G2b only (16 of 1,340 at
drafting).

### Variables

- **Outcome:** `wmh_log_mm3`.
- **Age:** (FLAIR date − birth month) / 365.25 (`labels._age_at`).
- **Sex:** PPMI Demographics.
- **eTIV:** FreeSurfer eTIV of the T1 used (validated: r 0.992 against PPMI's FreeSurfer 7).
- **FLAIR batch:** vendor × 2D/3D (as `manifest`'s `flair_batch`).
- **Cohort:** PPMI cohort label.
- **Vascular-risk score `vrs`** = hypertension + diabetes + hyperlipidaemia, 0–3, each item counted as 1 if
  documented on or before the scan date:
  - **Medical Conditions Log:** `MHTERM` matches the item's pattern, with diagnosis year (`MHDIAGYR`) at or before
    the scan year, or missing;
  - **or Concomitant Medication Log:** coded indication `CMINDC` (14 hypertension, 11 diabetes, 13 hyperlipidaemia),
    or, where the older EDC left the code blank, `CMINDC_TEXT` matching the pattern; start month (`STARTDT`) at or
    before the scan, or missing.
  - **Patterns** (case-insensitive; exact regexes in the script):
    - hypertension: `hypertens|htn|high/elevated blood pressure`;
    - diabetes: `diabet|t1dm|t2dm|niddm|iddm|dm|dm2`;
    - hyperlipidaemia: `hyperlipid|hypercholest|dyslipid|cholesterol|hypertriglycerid`.
  - **Exclusions:** terms naming pulmonary, ocular, portal, intracranial, white-coat, gestational or pregnancy
    conditions, family history, pre-hypertension, pre-diabetes, diabetes insipidus or "borderline".
  - An item is 0 when the participant has records in either log but no match. All items are missing when the
    participant has no record in either log.
  - Equal weights, as in the UK Biobank risk-factor count (Cox et al. 2019).
  - Smoking, BMI and waist–hip ratio are not items: smoking is answered by only 35 % (PPMI Online), and height and
    weight are missing on 38 % of vital-signs rows.

Prevalence at drafting, from labels and dates only (1,340 participants with a FLAIR; earliest FLAIR date):

| Hypertension | Diabetes | Hyperlipidaemia | vrs 0 / 1 / 2 / 3 / missing | Supine SBP within 365 d | Smoking answered (ever) |
|---|---|---|---|---|---|
| 36.0 % | 7.9 % | 41.0 % | 571 / 442 / 250 / 61 / 16 | 1,328 | 464 (35 %) |

Analysed n after QC is expected near 1,100: about 1,340 with a FLAIR, minus the 72 % (2D) and 95 % (3D) QC pass
rates of the threshold-method table.

### Statistics and pass rules

Partial Spearman correlations, rank-based, with percentile bootstrap 95 % CIs (2,000 draws, seed 20261015;
`pie.stats.small_sample.bootstrap_partial_correlation`). The adjustment set Z is sex, eTIV, FLAIR batch (dummies)
and cohort (dummies). Both hypotheses are one-sided, positive.

**G2a, age.** ρ(age, `wmh_log_mm3` | Z).
- **Pass: ρ ≥ 0.35 and CI lower bound > 0.18.**
- **Expected value, about 0.5.** TrUE-Net tracked age at r 0.70 (semi-manual reference 0.71) in 160 adults aged
  63.7 ± 13.7 (Strain et al. 2024, NeuroImage 285:120494). Corrected for PPMI's narrower age range (SD 8.7 years in
  the 1,336 FLAIR participants with plausible ages; Thorndike case 2), that gives about 0.53. PIE's T1-based
  FastSurfer WM-hypointensity volume tracks age at ρ 0.46 in PPMI.
- **The 0.35 bar** is about two-thirds of that expectation. It is the bar written into the 25 Sep 2026 plan before
  any TrUE-Net value existed (`docs/superpowers/plans/2026-09-25-imaging-weak-points.md`).
- **The CI floor** is the threshold method's ρ 0.18. The new measure must beat the one it replaces by more than
  sampling error.

**G2b, vascular risk.** ρ(`vrs`, `wmh_log_mm3` | age, Z).
- **Pass: CI lower bound > 0** (one-sided α 0.025).
- **Expected value, about 0.05–0.10.** In UK Biobank (Cox et al. 2019, Eur Heart J 40:2290; n 9,722, aged 44–79),
  a 6-item count gave a standardised β of 0.110 on log WMH, adjusted for age, sex and head size. The items were
  self-reported hypertension, diabetes and hypercholesterolaemia, ever smoking, BMI > 25 and high waist–hip ratio.
  Single-item β: hypertension 0.097, diabetes 0.065, hypercholesterolaemia 0.023 (0.003 with the others in the
  model). PIE's three-item count lacks smoking, BMI and waist–hip ratio and weights the weakest item equally, so a
  smaller value is expected.

**`FLAIR_WMH_VALIDATED`** needs G1 and both parts of G2.

**Power** (Fisher z, Spearman SE inflated 1.03, 10 adjustment terms):

| Analysed n | G2a P(pass) at ρ 0.30 / 0.35 / 0.40 / 0.45 | G2b P(pass) at ρ 0.05 / 0.08 / 0.10 / 0.12 |
|---|---|---|
| 800 | 0.06 / 0.50 / 0.94 / 1.00 | 0.28 / 0.59 / 0.78 / 0.91 |
| 1,100 | 0.04 / 0.50 / 0.97 / 1.00 | 0.36 / 0.73 / 0.89 / 0.97 |

G2b is adequately powered only if the PPMI association is near UK Biobank's. A G2b failure at ρ ≈ 0.05 would be
reported as uninformative, not as evidence of no association. Either way the gate stays failed.

**Reported regardless (no gate):**
- the participant flow, QC by batch and the `vrs` distribution;
- the unadjusted Spearman ρ with age;
- the G2a statistic by cohort (HC, PD, prodromal; n ≥ 30) and by 2D/3D;
- the same age statistic for FastSurfer `vol_WM_hypointensities` (log) on the same participants, as the benchmark;
- periventricular and deep log volume against age;
- each risk item separately (adjusted as G2b);
- supine systolic BP at the vital-signs visit nearest the scan, within 365 days;
- ever-smoking (PPMI Online: ≥ 100 cigarettes; where answered);
- G2b restricted to 3D FLAIR.

## G3 (optional). Visual ratings on PPMI

Only if a rater is available, and pre-registered as an addendum before rating:
- **Sample:** 100 baseline QC-passing PPMI FLAIRs, sampled by FLAIR batch × age tertile (seed 20261015), not by WMH
  volume.
- **Ratings:** Fazekas periventricular and deep scores (0–3 each). The rater is blind to WMH volume, age and cohort.
  A second rater scores 30 (weighted κ reported).
- **Statistics:** Spearman ρ of the total score with `wmh_log_mm3`; periventricular score against `wmh_pv_mm3`;
  deep score against `wmh_deep_mm3`.
- **Proposed bar:** ρ ≥ 0.60. **Its literature source is not yet verified**; fix it, with a source, in the addendum.

## What a pass would mean, and would not

**A pass** (G1 and G2): TrUE-Net WMH volume with PIE's FastSurfer preparation becomes **Validated, with limits**, as a
vascular covariate for group-level analyses of PPMI.
- It is adjusted for FLAIR batch, with a 3D-only sensitivity analysis, because QC pass rates differ between 2D and 3D.
- It replaces FastSurfer `vol_WM_hypointensities` as PIE's default vascular covariate.

**If G1 passes and G2 fails:** accuracy has been shown outside PPMI, but not the expected biology inside it. The
measure stays **Exploratory**, and the T1 measure stays the default.

**If G1 fails:** the measure stays **Exploratory**, whatever G2 shows. The failure is reported with the per-scanner
breakdown.

**A pass would not mean:**
- **Lesion-level accuracy on PPMI scans.** No PPMI manual labels exist unless G3 or tracing is done.
- **Accuracy on 5 mm 2D slabs.** G1 has 3 mm slices; the by-acquisition results are reported but not gated.
- **Validity of the periventricular/deep split, or of longitudinal change.**
- **Validity in other cohorts, or of the FSL preparation.**
- **That WMH is a marker of synucleinopathy.**

## Deviations

Any change after the freeze is recorded here, dated, with its reason, before the affected result is read. Where the
original analysis can still be run, it is run too and both are reported. A gate is run once. It is not repeated until
it passes, and every number is reported.

## Analysis code

- `scripts/analyse_flair_wmh.py`:
  - `g1 --metrics <table>`;
  - `g2 --features <flair_truenet_fs>/flair_features.csv --idps <derived>/fastsurfer_idps.csv --ppmi PPMI`;
  - `--synthetic --out <dir>`, the dry run, which reads nothing real: every value, label and identifier is generated.

  Results are written to `wmh_g1_result.json` / `wmh_g2_result.json`.
- `tests/test_analyse_flair_wmh.py`:
  - `mask_metrics` on tiny arrays: identical masks, disjoint masks, a missed lesion, a label-2 detection,
    26-connectivity;
  - G1 and G2 passing on synthetic effects and failing on synthetic nulls or weak effects.
- **Hashes:** sha256 at the freeze in the run log; the G1 runner's is added there before it runs. Synthetic dry run
  only before the freeze.

## Open items at the freeze

The design, gates and analysis script were frozen on 15 Oct 2026 with these items open. None changes a gate,
threshold or statistic.
1. **Weights' training data (documented caveat).** Ask TrUE-Net's authors (V. Sundaresan, L. Griffanti) whether
   `fsl-truenet_data` 2508.0 `mwsc` saw any challenge test subject. The FSL docs and README say only "MICCAI WMH
   Segmentation Challenge Data". The answer is recorded in the run log before G1 is read. If any test subject was
   used, or no answer comes, G1 is reported with that caveat; if one was used, G1 is void and the fallback applies.
2. **Data.** The user accepts DataverseNL's terms (CC BY-NC 4.0; terms click-through, guestbook) and downloads the
   110 test subjects' three files to the NVMe (check free space first).
3. **G1 runner.** Written after the freeze, before any challenge file is downloaded, to the procedure above. Its
   `mask_metrics` is checked against the challenge's `evaluation.py` on the training-set observer annotations (to
   1e-6), and both hashes go in the run log before G1 runs. GPU time: FastSurfer on 110 T1s, after the current GPU
   jobs.
4. **Seen results.** The user confirms that no TrUE-Net WMH value has been compared with age, cohort or vascular
   risk, before G2 runs. The confirmation is recorded in the run log.
5. **G3.** Rater availability, and a verified source for its bar (an addendum, pre-registered before rating).

## Run log

| Date | Gate | Commit | Result |
|---|---|---|---|
| 2026-10-15 | — | the freeze commit | **Frozen.** sha256: `analyse_flair_wmh.py` `6fb59500a5650d3d10abfc34eae961f55252ef7859fcc9bcbaf0230a68827b5b`; `flair.py` `7ca4bc11609ae349cd5dd377e613a54955577c6f519f21bf9d80dd8b97741899`. Open items 1–5 as listed. No challenge file downloaded; no gate statistic computed |
