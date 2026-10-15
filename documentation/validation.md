# Imaging validation: status, evidence and log

What PIE's imaging measures have been validated against, how, with what result, and what is still being validated.
Updated **15 October 2026**. Detailed methods and tables live in the module pages
([imaging.md](imaging.md), [imaging_nm_datscan.md](imaging_nm_datscan.md), [imaging_dwi.md](imaging_dwi.md),
[fmriprep.md](fmriprep.md)) and in the pre-registrations linked below. This page is the summary and the running log.
It is updated whenever a validation result or a pre-registration changes.

All numbers are aggregates from PPMI data. No participant-level data appear here or in the linked pages.

## How to read it

Each measure family ends in one of five statuses:

| Status | Meaning | In the assembled table |
|---|---|---|
| **Validated** | Passed its pre-specified checks on PPMI data | Used by default |
| **Validated, with limits** | Passed, within a stated context of use (named below) | Used by default, with the limit documented |
| **Exploratory** | Failed a pre-specified check, or no fair check exists yet; the failure is reported | Exported, flagged, not used by default |
| **In validation** | A frozen test is running or waiting for data | Exported, not yet relied on |
| **Not offered** | Out of scope for now | Not exported |

The rules behind these statuses:
- **Pre-registered:** each test's question, sample rule, measure and pass rule are frozen in a dated document before the outcome is computed.
- **Run once:** a test is not repeated until it passes.
- **Every number reported:** failures stay in the record next to every attempt.
- **Fully synthetic dry runs:** dry runs of a frozen analysis use fake values and labels only.

Evidence is graded in three layers, used where they make sense:
- **technical:** repeatability, and agreement with a reference measurement;
- **biological:** known associations such as age and dopaminergic deficit;
- **clinical:** known groups or prognosis.

Manifest flags that encode a status today: `NM_VALIDATED` (False), `DWI_NIGRAL_VALIDATED` (False) and
`PET_AV133_VALIDATED` (True), all in `pie/imaging/manifest.py`. They are exported as `nm_validated`,
`dwi_nigral_validated` and `pet_av133_validated`.

## Summary (15 October 2026)

| Modality | Measure | Status | Key evidence | Next |
|---|---|---|---|---|
| T1 | Subcortical volumes (FastSurfer, pinned build) | **Validated** | vs PPMI FreeSurfer 7: r 0.81–0.96, median CCC 0.87 (1,080 scans); same-day ICC 0.97–1.00 (29 pairs) | Visual QC ratings of flagged scans |
| T1 | eTIV (FreeSurfer 7.4.1) | **Validated** | r 0.992 vs PPMI FS7 eTIV, +0.3 % (40 scans; pre-registered gate r ≥ 0.95, < 5 %) | — |
| T1 | Normative z-scores (age, sex, TIV) | **Validated** | Controls' out-of-fold z: mean within 0.013 of 0, SD 1.01–1.03 | — |
| T1 | Basal forebrain, ScLimbic | In use; descriptive checks only | Age r −0.43 in controls; no PD–HC difference | Convergence gate within the Ch4 test |
| T1 | Nucleus basalis (Ch4), CAT12.9 | **In validation** | FSL-FAST version failed (no age effect); rebuilt on CAT12, pre-registered 13 Oct | Result expected mid-October |
| T1 | Cortical thickness, hippocampal subfields, SynthSeg (legacy 1.5 T), medial-temporal index | Not yet run | Code and synthetic tests | Agreement with PPMI's `FS7_APARC_CTH`; same-day ICC |
| DaT-SPECT | PPMI core-lab SBRs (Xing; Invicro fallback) | **Validated** (reference values) | PPMI's own endpoint; PD vs HC AUROC 0.99 | — |
| DaT-SPECT | PIE's own reconstruction (`datscan_2026_10`) | **Validated** | 1,452 of 1,459 scans pass QC. vs Xing: putamen r 0.851, CCC 0.87, deficit κ 0.80. vs Invicro: r 0.887. The two core labs agree with each other at r 0.88 | Non-PPMI reconstructed stacks |
| NM-MRI | snceg contrast ratio (`nm_primary_cr`) | **Exploratory**, 2 of 3 stages passed | Stage 1 technical: projected ICC 0.945. Stage 2b: prodromal DaT-deficient vs normal AUROC 0.693 [0.625, 0.758] (scans largely shared with an earlier analysis; see below). PD vs HC on Siemens GRE-MT: 0.631, at the published like-for-like level (0.633) | Stage 2a (longitudinal), Stage 3 (fresh HC/PD); downloads pending |
| DWI | Stripped-gradient recovery | **Validated** | 36 series rebuilt across 5 protocols; leave-one-out error ≤ 0.42°; MD r > 0.9999 | — |
| DWI | Nigral FA/MD (single-shell tensor) | **Exploratory** | Pre-registered PPMI 2.0 re-test failed: AUROC 0.574 (0.495–0.652); MD ρ −0.01 vs PPMI's hand-drawn ROIs | — |
| DWI | Single-shell free water | Quarantined | Age r +0.02; PD vs HC AUROC 0.50; ill-posed on one shell | — |
| DWI | Multi-shell posterior-SN free water (`dwisf_*`) | **In validation** | Pre-registered 14 Oct (prodromal DaT contrast); run in progress | Result expected late October; QSIPrep comparator |
| FLAIR | WMH volume, TrUE-Net | **In validation** | Preprocessing agreement on 60 FLAIRs: Dice 0.872, volume ICC 0.984 | Accuracy vs manual labels; biology on PPMI |
| FLAIR | WMH volume, threshold method | Exploratory (superseded) | Age ρ 0.18 | — |
| rs-fMRI | fMRIPrep + XCP-D connectivity | **Not yet validated** | No real-data run since the October fixes | Pre-registered plan (below) |
| PET | AV-133 striatal SBR | **Validated, with limits** | vs all three PPMI pipelines: posterior putamen r 0.92–0.97; caudate r 0.79–0.95 with a fixed offset; no partial-volume correction | Remaining scans |
| PET | SV2A, tau (PI-2620) | Not validated (planned: Not offered) | Too few scans in PPMI (18 SV2A, 23 tau) to validate a pipeline. PPMI's own SUVRs are read by `labels.suvr_labels` | Revisit if PPMI publishes its method |
| — | CT, α-synuclein PET, DPA-714, iron (QSM/SWI) | **Not offered** | CT is a safety form without images; PPMI does not acquire QSM/SWI | — |
| Longitudinal | Per-visit runs, longitudinal FastSurfer stream | Not yet piloted | Same-day MDC95 2–6 % gives the noise floor | Pilot on 100 participants |
| Multi-cohort | BIDS/ADNI input, ComBat harmonisation | Synthetic tests only | — | Harmonisation across PPMI's own scanners; then a public BIDS dataset and ADNI |

## How each modality was validated

### T1 structural (FastSurfer)

- **Agreement with PPMI's reference.** PIE's volumes were compared with PPMI's FreeSurfer 7.3.2 `FS7_ASEG_VOL` on 1,080 baseline scans:
  - subcortical r 0.81–0.96;
  - Lin's CCC median 0.87 (Siemens 0.88, Philips 0.82, GE 0.80; 3 T 0.86, 1.5 T 0.82);
  - PIE's volumes run larger (putamen about +240 mm³).

  White-matter hypointensities agree poorly (r 0.19), because the two methods differ.
- **Repeatability.** 29 same-day MPRAGE pairs; ICC(2,1) with 95 % minimal detectable change (MDC95):
  - putamen 0.98 (5.1–5.5 %);
  - caudate 0.99;
  - thalamus 0.98–0.99;
  - hippocampus 0.99;
  - amygdala 0.97–0.98;
  - lateral ventricles 1.00;
  - brainstem 0.98.
- **Biology.** Age associations in 1,800 scans: thalamus r −0.42, hippocampus −0.36, ventricles +0.40.
- **eTIV.** Pre-registered gate (r ≥ 0.95, mean difference < 5 %) against PPMI's FreeSurfer 7 eTIV on 40 scans: r 0.992, +0.3 %, none rejected. The registration-based TIV (r 0.81) is not used for adjustment.
- **Normative z-scores** (`normative.crossfit_zscores`; 165 HC, 543 PD, 318 prodromal):
  - controls' out-of-fold z had mean within 0.013 of 0, SD 1.01–1.03, residual age r ≤ 0.016, and 5.0 % beyond ±2;
  - PD putamen mean z −0.08 to −0.10, beside the ENIGMA-PD d of −0.14.
- **Still open:** visual QC ratings of every robust-z outlier plus 100 random scans, to measure the QC rule's miss rate.

### Basal forebrain and nucleus basalis (Ch4)

- **ScLimbic basal forebrain** (`run --sclimbic`, whole cohort): age r −0.24 / −0.28 overall and −0.43 in controls; no PD–HC difference (d ≈ −0.04). It is PIE's basal forebrain measure while Ch4 is in validation.
- **Ch4, first version (FSL FAST grey matter in the Julich-Brain Ch4 map), 13 Oct: not validated.**
  - Grey matter was only 26 % of the map.
  - There was no age effect (controls r −0.05).
  - ρ with ScLimbic was 0.17.

  The published nucleus basalis results that link it to α-synuclein seed amplification used SPM/CAT12 modulated grey matter, so the stage was rebuilt that way.
- **Ch4, CAT12.9 r2665, in validation.** Pre-registered in [preregistration_ch4_cat12.md](preregistration_ch4_cat12.md), frozen 13 Oct. All three gates are needed:
  - controls' age r ≤ −0.30 with CI upper bound < 0;
  - ρ ≥ 0.40 with ScLimbic basal forebrain;
  - same-day ICC(2,1) ≥ 0.75.

  504 scans, run once.
- **Software version.** CAT26 was released on 12 Oct 2026. The pre-registered test stays on CAT12.9. Moving to CAT26 would need a version-bridging comparison first (on hold).

### DaT-SPECT

- **PPMI's core-lab SBRs** are PIE's reference values (`labels.dat_sbr_table`: Xing first, Invicro where Xing never analysed a scan). PD vs HC on the lowest putamen: AUROC 0.95–0.96 with PIE's reconstruction, 0.99 with PPMI's SBRs on the same scans.
- **PIE's own reconstruction**, for scans PPMI never quantified and for other cohorts:
  - **Tuning and validation:** the October registration (brain-only template with a legacy fallback) was tuned on 240 random scans and checked on a separate, vendor-stratified validation set of 236 participants.
  - **Full cohort** (1,459 scans, 1,452 pass QC; now the default `datscan_2026_10`):
    - vs Xing (1,161 matched): putamen-min r 0.788 → 0.851, CCC 0.87, caudate 0.74–0.76, deficit κ 0.80;
    - vs Invicro (882): r 0.887, CCC 0.89, deficit κ 0.85.

    PPMI's two core labs agree with each other at 0.88.

### Neuromelanin MRI

**Attempts, in order** (all pre-registered; all results kept):

| Date | Test | Result |
|---|---|---|
| 26 Sep 2026 | Template and atlas CNR, PD vs HC, gate AUROC ≥ 0.70 with CI lower bound > 0.55 (45 + 45) | 0.58 and 0.54: failed |
| late Sep 2026 | Published Biondetti territories (106 held-out PD, 39 HC) | Sensorimotor 0.62, 8-territory model 0.67: failed |
| 26–27 Sep 2026 | Langley 2025 threshold volume (PPMI Siemens; and the authors' own release) | Not reproduced: their nigral atlas and peduncle ROI are unpublished, and PIE's substitute regions gave control volumes 1.8× theirs |
| early Oct 2026 | snceg (Lillebostad 2025) on PPMI TSE-FS | Reproduced: 0.710 vs published 0.743 |
| 12 Oct 2026 | snceg on PPMI Siemens GRE-MT, target 0.743 ([preregistration_nm_snceg.md](preregistration_nm_snceg.md)) | 0.631 [0.555, 0.707]: failed against that target |
| 12 Oct 2026 | Staged validation, Stage 1, technical ([preregistration_nm_staged_validation.md](preregistration_nm_staged_validation.md)) | Passed: primary `nms_sn_mean_cr`, projected ICC 0.945 |
| 12 Oct 2026 | Stage 2b, biological: prodromal DaT-deficient vs DaT-normal (267 participants) | Passed: AUROC 0.693 [0.625, 0.758]. Corrected 15 Oct: 224 of the 267 scans had been used by the Study 1 manuscript's NM analysis, so this was not a test on unseen scans. On 178 shared participants, Study 1's whole-nigra contrast performed alike (0.637 vs PIE 0.648) once slabs under 24 mm were excluded. The slab-coverage QC, not the measure, explains most of the gain |

**Why PD vs HC stays modest (literature review, 15 Oct 2026).**
- **The like-for-like benchmark.** The only published estimate for the same comparison (PPMI Siemens GRE-MT, newly diagnosed PD) is an AUROC of 0.633, rising to 0.755 at four years (Langley, Hwang, Huddleston & Hu, ISMRM 2023, abstract 3195). PIE's 0.631 sits at it.
- **Why the 0.743 target didn't apply.** It came from a different sequence (TSE-FS). Neuromelanin contrast ratios from different protocols correlate poorly in the same patients (r 0.23; Hirose et al. 2026, doi:10.1007/s00702-026-03179-z).
- **Why pooled accuracies are higher.** Pooled diagnostic accuracies (AUC about 0.85–0.87) come mainly from patients with longer disease (Cho et al. 2021, doi:10.1007/s00330-020-07240-7).
- **Biology.** Nigral neuromelanin loss lags striatal DaT loss at diagnosis, and PPMI required a DaT deficit to enrol PD.
- **No reference standard.** PPMI publishes no core-lab neuromelanin values, so agreement cannot be tested as it was for DaT and AV-133. The field calls for standardisation before neuromelanin MRI is used as a multisite biomarker (Trujillo et al. 2024, doi:10.1093/brain/awad300).

**Slab coverage matters more than the measure.** In prodromal participants, slabs under 24 mm through-plane (median 20 mm, almost all GE) carry no DaT signal: AUROC 0.398 against 0.634 on full-coverage slabs, for the same contrast measure. PIE's NM QC requires ≥ 24 mm, which removed 27 % of prodromal sessions.

**What is left:**
- **Stage 2a:** decline in PD on follow-up scans.
- **Stage 2c:** side-matched DaT; exploratory.
- **Stage 3:** fresh HC and PD, from NM in no PIE download, and checked against every NM scan used before. Pass rule fixed 13 Oct, before any Stage 3 data: vendor-stratified AUROC with CI lower bound > 0.50 and CI upper bound ≥ 0.633.

If Stage 3 passes, NM becomes **validated, with limits**: Siemens GRE-MT, snceg contrast, research use at group level, not diagnosis.

### Diffusion MRI

- **Gradient recovery** (`dwi_gradients.py`, 11 Oct): 36 series with stripped gradient tags rebuilt across 5 protocols.
  - Leave-one-out direction error ≤ 0.42°; MD r > 0.9999 against intact references.
  - 16 series in four protocols with no intact reference were left alone.
- **Nigral tensor measures.**
  - vs PPMI's hand-drawn nigral ROIs (248 scans): FA ρ 0.39, MD ρ −0.01.
  - Pre-registered PPMI 2.0 re-test of posterior-nigral MD (218 PD, 82 HC; 29 Sep): AUROC 0.574 (0.495–0.652), against a gate of ≥ 0.60 with lower bound > 0.50. **Failed.**
- **Single-shell free water.** Quarantined: age r +0.02, PD vs HC 0.50. It is ill-posed on one shell.
- **Multi-shell posterior-SN free water** (`dwi --eddy --study`, PPMI 2.0 three-shell). Brain free water tracks age (r +0.49, 68 scans).
  - **Pre-registered test** ([preregistration_dwi_fw.md](preregistration_dwi_fw.md), frozen 14 Oct): prodromal DaT-deficient vs DaT-normal, as NM Stage 2b. PD vs HC is infeasible because PPMI holds three-shell DWI for only 13 controls.
  - **Freeze counts:** 275 DaT-normal, 184 DaT-deficient; power about 0.91 at AUROC 0.60.
  - **Running.** Sessions without a usable b0 reference are counted as exclusions.
- **Planned:** a QSIPrep/QSIRecon comparator (agreement of nigral free water between pipelines), once the containers are set up.

### FLAIR white-matter hyperintensities

- **Old method:** the threshold method tracked age at only ρ 0.18. It has been superseded.
- **TrUE-Net preprocessing agreement.** TrUE-Net (`mwsc` model) on 60 PPMI FLAIRs, PIE's FastSurfer preprocessing vs TrUE-Net's own:
  - Dice median 0.872;
  - volume r 0.991, ICC 0.984;
  - FastSurfer preprocessing gives volumes about 10 % smaller.

  The cohort run is in progress.
- **Planned test** (approved 15 Oct, pre-registration to follow):
  - accuracy against manual WMH labels on the public MICCAI WMH 2017 challenge test set, which is independent of TrUE-Net's training data;
  - associations with age and vascular risk in PPMI.

### Resting-state fMRI

- **No real-data validation yet.**
- **October audit fixes.** Two fixes change outputs:
  - PPMI 2.0 scans had been run without distortion correction;
  - the repetition time was never checked.

  Earlier fMRIPrep outputs predate both.
- **Plan** ([fmriprep.md §5e](fmriprep.md#5e-validation-plan), to be pre-registered):
  - end to end on 100 scans per protocol (TR 2.4 s, 2.5 s and 1 s), with every exclusion counted;
  - QC-FC benchmarked against Ciric et al. 2017;
  - split-half reliability, dropping features with ICC < 0.4;
  - posterior putamen–sensorimotor coupling against DaT, reported but not a gate.
- **No PD-vs-HC gate:** PPMI resting-state group differences have not reproduced.
- **Blockers:** fMRIPrep and XCP-D containers, and PPMI-1 fMRI downloads.

### PET

- **AV-133 (VMAT2)**, validated 12 Oct 2026, following PPMI's 2014 IND method on the participant's FastSurfer regions, against all three PPMI pipelines:
  - posterior putamen r 0.92–0.97;
  - caudate r 0.79–0.95, with a fixed offset (the stated limit);
  - no partial-volume correction.
- **SV2A and tau (PI-2620): run, not validated.**
  - SV2A: PIE's values run about 0.45× PPMI's scale on 3 matched scans.
  - Tau: unfiltered 1 mm frames and real motion.
  - PPMI holds 18 SV2A and 23 tau scans in all. That is too few to validate a pipeline, and tuning on the handful of matched scans would choose the method on its outcome.
  - Use PPMI's own SUVRs (`labels.suvr_labels`).

### Longitudinal and multi-cohort

- **Longitudinal.** Per-visit runs, the longitudinal FastSurfer stream and longitudinal ComBat are coded and tested on synthetic data. The same-day MDC95 (2–6 %) is the noise floor.
  - **Planned pilot:** 100 participants with two or more visits.
  - **Zero-change control:** same-day pairs.
  - **Plausibility:** change in controls against published rates.
- **Multi-cohort.** BIDS and ADNI input, cohort-neutral keys, ComBat-GAM with a confounding report, and a BIDS-Derivatives long table are coded and tested on synthetic data. Planned real-data tests, in order:
  1. harmonisation across PPMI's own scanners and protocol phases;
  2. a public BIDS dataset;
  3. ADNI, once data access is in place.

## Log

Newest first.

- **2026-10-15**
  - NM Stage 2b correction: 224 of its 267 scans had been used by the Study 1 manuscript's NM analysis. Disclosed in the pre-registration; the result is unchanged. Most of its margin comes from the ≥ 24 mm slab rule.
  - NM literature review. The like-for-like PPMI GRE-MT benchmark is 0.633, and PIE's 0.631 sits at it. There is no public PPMI reference standard for neuromelanin. Recorded above.
  - CAT26 noted and put on hold.
  - This page added.
- **2026-10-14**
  - DWI multi-shell free-water pre-registration frozen (prodromal DaT contrast). Run started.
  - TrUE-Net preprocessing agreement on 60 FLAIRs.
  - T1 selection and session-date fixes (regression tests added).
- **2026-10-13**
  - FAST-based Ch4 failed.
  - Ch4 rebuilt on CAT12.9 and its pre-registration frozen. Run started (504 scans).
  - NM Stage 3 pass rule fixed.
  - Tau PET frames diagnosed.
- **2026-10-12**
  - NM Stage 1 passed (ICC 0.945) and Stage 2b passed (0.693).
  - snceg GRE-MT test failed against 0.743 (0.631).
  - AV-133 validated.
  - SV2A and tau run, not validated.
  - Normative z-scores validated.
- **2026-10-11**
  - T1 same-day repeatability and agreement by vendor and field strength.
  - DaT October registration validated and made the default.
  - DWI gradient recovery (36 series).
  - snceg pre-registration frozen.
  - Audit fixes, including fMRI distortion correction and TR checks.
- **2026-09-29:** DWI nigral MD pre-registered re-test failed (0.574).
- **2026-09-26 to 27**
  - eTIV passed (r 0.992).
  - NM CNR gate failed (0.58 / 0.54).
  - Biondetti-territory re-test failed.
  - Langley threshold volume not reproduced.
- **2026-09 (early):** first real-data checks: T1 vs FS7 r 0.81–0.96; DaT PD vs HC AUROC 0.95–0.96; DWI and FLAIR checks as in [imaging.md](imaging.md#real-data-validity-checks-september-2026).
