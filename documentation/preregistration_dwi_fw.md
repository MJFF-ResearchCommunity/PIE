# Pre-registration: multi-shell posterior-SN free water in PPMI 2.0 (PD vs HC)

**Status: DRAFT, 11 Oct 2026.** It is frozen, with a timestamp and the commit hash of the code, after three things:
- the user has reviewed it;
- the participant counts below are filled in from QC flags and cohort labels only;
- and before any `dwisf_*` value exists for the sample.

This is a **post hoc** analysis of a measure chosen after related measures were seen (disclosure below), and it will be labelled that way.

## Why this endpoint

PIE's first nigral DWI endpoint failed its pre-registered PPMI 2.0 gate. That endpoint was posterior-SN mean diffusivity (`dwis_md_sn_posterior_mean`); it scored AUROC 0.574 [0.495, 0.652] on 218 PD / 82 HC (29 Sep 2026). The result matches the meta-analytic nigral MD effect (Schwarz 2013, d ≈ 0.26). In PPMI-1, nigral MD did not separate the groups (Ofori 2015).

The PPMI-supported nigral DWI marker is **posterior-SN free water**:
- Ofori 2015: PD > HC.
- Burciu 2017 (Brain, 10.1093/brain/awx146): PPMI-1, PD > HC with d ≈ 0.4–0.5, increasing over 1 and 4 years in PD and not in HC.

Free water needs a second shell to be well-posed (Golub 2021, MRM 10.1002/mrm.28599). PPMI 2.0's multi-shell protocol provides one. **User decision, 11 Oct 2026:** multi-shell posterior-SN free water is PIE's pre-registered DWI endpoint.

## Disclosure: what has already been seen

- **The study's nigral free water:** AUROC 0.688 PD vs HC on PPMI-1 Siemens, and 0.55 on PPMI 2.0 (`2026-09-27-dwi-fw-plan.md`). That was a different estimator and ROI pipeline.
- **PIE's nigral MD gate:** 0.574 (above).
- **PIE's whole-brain median multi-shell free water:** rises with age, r +0.49.
- **PIE's generic-path free-water columns** (`dwi_sn_posterior_*_fw`, affine CIT168, scanner-frame split) exist on disk for earlier runs. No PD-vs-HC comparison of them is known to have been run, and none will be before freezing.
- **No `dwisf_*` value exists yet.** The measure was implemented on 11 Oct 2026 and has only been run on synthetic data.

## Measure (one scalar)

`dwisf_fw_sn_posterior_mean`, from PIE `dwi --eddy --study` (`dwi.study_fw_features`; code at the freeze commit):
- **Model and fit:** DIPY's multi-shell free-water NLS (Hoy 2014) on the whole multi-shell dataset. Runs are scaled to the first run's b0 and regridded onto its grid. A non-physical fit is missing, never FW = 0.
- **ROI:** the CIT168 SNc + SNr, through the subject's T1 → MNI SyN and a rigid b0 → T1, sampled nearest-neighbour onto the native DWI grid.
- **Posterior half, per side:** the voxels behind that side's median MNI y. This anatomical split approximates the hand-drawn posterior SN of Ofori/Burciu.
- **Region value:** the mean of physically fitted voxels, when there are ≥ 3 and ≥ 80 % of the region.
- **Bilateral value:** the mean of the two sides, so both sides are required.

**QC** (`dwisf_qc_pass`): a multi-shell fit; ≥ 20 nigral voxels; ≥ 95 % in the brain mask; ≥ 80 % physical free-water fits; motion ≤ 2 mm (eddy mean RMS, or rigid maximum translation); both posterior halves measured.

## Sample

All PPMI 2.0 baseline PD and HC multi-shell DWI sessions available locally (acquired 2020 or later) that have a finished FastSurfer T1. This is the 29 Sep gate's candidate set restricted to multi-shell sessions, plus any later downloads. Diagnosis comes from PPMI's cohort labels. Philips sessions use the name-derived PE polarity.

Counts are filled in at freeze, from QC flags only:

| | HC | PD |
|---|---|---|
| Multi-shell candidates | _ | _ |
| Passing `dwisf_qc_pass` | _ | _ |

The 28 Siemens sessions with stripped gradient tags are excluded from the primary analysis; if recovered, they enter only the sensitivity analysis.

## Harmonisation (primary)

As the 28 Sep gate: HC-referenced z-scores within vendor × correction tier.
- Age and sex are regressed out in all HC, with batch intercepts.
- Each batch is scaled on its own HC residual SD.
- A vendor whose tiers have fewer than 5 HC each is one batch. A vendor with fewer than 5 HC in total has no batch, and its participants are excluded and counted.
- Only HC labels enter the harmonisation.

## Estimand and reading (no pass/fail gate)

**Primary:** the standardised PD − HC difference of the harmonised scalar, Cohen's d with the pooled SD. Its 95 % bootstrap CI uses 2,000 draws, resampled within group with the HC reference refitted each draw, seed 20261011.

The anchor is the published PPMI-1 effect, d = 0.40–0.50. The CI is read as follows:

| 95 % CI of d | Reading |
|---|---|
| Lower bound > 0 and the CI overlaps 0.40–0.50 | Consistent with the published PPMI effect |
| Lower bound > 0, upper bound < 0.40 | Present but smaller than published |
| Lower bound > 0.50 | Larger than published |
| Upper bound < 0.40, lower bound ≤ 0 | Not detected; excludes the published effect |
| Otherwise | Inconclusive |

**Power** (normal approximation; SE(d) ≈ √(1/n_PD + 1/n_HC + d²/2N)):

| n PD / n HC | CI half-width | P(lower bound > 0 \| d = 0.45) | P(CI contains 0.45 \| d = 0.45) | P(lower bound > 0 \| d = 0.30) |
|---|---|---|---|---|
| 100 / 40 | 0.37 | 0.66 | 0.95 | 0.36 |
| 150 / 60 | 0.30 | 0.83 | 0.95 | 0.50 |
| 218 / 82 | 0.26 | 0.93 | 0.95 | 0.64 |

With fewer than about 40 HC, the analysis can say little beyond "inconclusive". That will be stated before anything is unblinded if the freeze counts fall below it.

**Reported regardless:**
- AUROC with its CI;
- Glass's Δ (HC-SD units);
- the unadjusted AUROC;
- AUROC by vendor;
- the participant flow;
- batch sizes;
- HC age ρ, the positive control (expected > 0);
- anterior-SN free water d, the specificity check (expected smaller than posterior);
- posterior-SN tissue FA d;
- in PD, Spearman ρ with the lowest putamen SBR, using PPMI's core-lab values and each source separately (expected negative).

**Sensitivity analyses (they never replace the primary):**
1. Single-scanner batches (`DeviceSerialNumber` with ≥ 5 HC and ≥ 5 PD).
2. Recovered Siemens sessions added.
3. ComBat by vendor × tier, with diagnosis, age and sex protected (`pie.imaging.harmonize`, once merged).

**Afterwards, not part of this test:**
- longitudinal change in PD vs HC, as in Burciu 2017, within one correction tier;
- SAA status in prodromals, analysed only after this reading and pre-registered separately.

## Analysis code

`scripts/analyse_dwi_fw.py`:
- `--synthetic` is the dry run. It reads nothing real: every value, label and identifier is generated.
- The real run is `--tables <dwi_features.csv>[,...] --ppmi PPMI --out <dir>`.
- `tests/test_analyse_dwi_fw.py` checks it on synthetic data.
