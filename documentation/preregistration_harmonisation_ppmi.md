# Pre-registration: ComBat harmonisation across PPMI's own T1 scanners and protocol phases (first stage of K4)

**Status: FROZEN 2026-10-15**, before any gate statistic was computed on real data (hashes in the run log). No harmonisation, classifier or effect estimate has been computed on real
volumes. The counts below come from the script's loader only (cohort, batch labels, missingness).

## Why

PIE will pool T1 volumes across the MRI CSF-SAA consortium (PPMI, ADNI, BEAT-PD, PPP, Stanford). `harmonize.combat`
has been tested on simulated data only (documentation/imaging.md, Harmonisation): it matches neuroCombat, and recovers
known site effects when sites differ in age range.

Before any cross-cohort use, it has to work across PPMI's own acquisition batches. These batches are:
- PPMI-1 (before 2020) and PPMI 2.0 (2020 on);
- three vendors;
- 1.5 T and 3 T.

PIE's T1 volumes already differ by batch: agreement with PPMI's FreeSurfer 7 runs from CCC 0.80 (GE) to 0.88
(Siemens), and from 0.82 (1.5 T) to 0.86 (3 T). PPMI is the easy case: one protocol family, one consortium, and
controls in every batch. If harmonisation fails here, it is not used on ADNI.

## Disclosure: what has already been seen

- **Batch differences in agreement:** PIE's agreement with FS7 by vendor and field strength (above).
- **Age associations:** in 1,800 scans, thalamus r −0.42, hippocampus −0.36, ventricles +0.40.
- **Normative z-scores with a 6-group site term:** controls' z SD 1.08–1.16, too many parameters for 165 controls.
- **No ComBat run on real PPMI volumes:** none known.
- **The batch counts below** (15 Oct 2026), from cohort labels, scanner tags and missingness only.

## Method

- **Features (91):** FastSurfer volumes (`fastsurfer_idps.csv`, pinned build), from each participant's earliest QC-passing, real-dated T1:
  - left and right lateral and inferior lateral ventricles, cerebellar white matter and cortex, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC and cerebral white matter (26);
  - 3rd and 4th ventricles and brainstem (3);
  - the 62 DKT cortical parcel volumes.

  Not included: WM hypointensities, choroid plexus, CSF, and the ScLimbic and CAT12 measures (other pipelines, not every scan).
- **Batch:** vendor × protocol phase × field strength (`Manufacturer`, `protocol_phase`, `MagneticFieldStrength`).
  - Vendor × phase alone would mix 1.5 T and 3 T scanners within PPMI-1 batches.
  - Scanner model is reported in the confounding table, not used: only 5 models have ≥ 10 controls.
- **Minimum 10 controls per batch.** ComBat needs ≥ 2 rows per batch. With 10, every training fold of a 5-fold split keeps ≥ 8 per batch and every test fold ≥ 2. Smaller batches are excluded and counted, with their PD.
- **Harmonisation:** `harmonize.combat(df, features, "batch", ["age", "sex", "eTIV"], nonlinear=("age",), spline_df=4)`.
  - Empirical Bayes.
  - Age is a cubic B-spline: ComBat-GAM in the sense of Pomponio et al. 2020, though unpenalised.
  - Sex and FreeSurfer eTIV (validated, r 0.992 against FS7) are linear.

## Sample

Baseline T1 of every healthy control and PD participant in PIE (PPMI cohort labels). Counts on 15 Oct 2026, after
excluding scans without a batch, age, sex, eTIV or volume:

| Batch (vendor, phase, field) | HC | PD |
|---|---|---|
| GE, PPMI-1, 1.5 T | 27 | 64 |
| GE, PPMI-1, 3 T | 14 | 45 |
| GE, PPMI 2.0, 3 T | 17 | 73 |
| Philips, PPMI-1, 1.5 T | 11 | 41 |
| Philips, PPMI-1, 3 T | 12 | 17 |
| Philips, PPMI 2.0, 3 T | 11 | 77 |
| Siemens, PPMI-1, 1.5 T | 13 | 69 |
| Siemens, PPMI-1, 3 T | 86 | 243 |
| Siemens, PPMI 2.0, 3 T | 24 | 90 |
| **Used (9 batches)** | **215** | **719** |
| Excluded, < 10 HC: other vendor, PPMI 2.0, 1.5 T; GE PPMI-1 0.7 T; Siemens PPMI 2.0 1.5 T | 6 | 12 |

Exclusions before batching:
- HC: 2.
- PD: 197, of which 193 have no eTIV. These are T1s processed after the eTIV run. `run --etiv` is run before this analysis, and whatever is still excluded at the run is counted.

The controls are 61.5 ± 11.6 years old (31–83): 60.3 in PPMI-1 and 64.9 in PPMI 2.0. The age difference is why the
classifier sees covariate-residualised volumes (below).

## Gates (all three for `HARMONISATION_PPMI_T1` = pass)

| Gate | Statistic | Threshold | Source |
|---|---|---|---|
| 1. Batch removed | Macro one-vs-rest AUROC of predicting batch from harmonised volumes, out of fold; mean of 10 repeats of stratified 5-fold CV (seeds 20261015 + r) | **≤ 0.60 for both classifiers** (0.50 = chance) | Fixed at the freeze. No published standard threshold |
| 2. Age kept | Controls' age β per decade of bilateral hippocampus, thalamus and lateral ventricles (OLS with sex and eTIV), harmonised against the unharmonised within-batch estimate (batch fixed effects) | **\|β_h − β_raw\| / \|β_raw\| ≤ 0.10 for all three** | Fixed at the freeze |
| 3. No inflation | PD − HC Cohen's d per volume (OLS with age, sex, eTIV; β / residual SD), ComBat protecting diagnosis against the unharmonised within-batch estimate | **\|d_protected − d_within-batch\| ≤ 0.10 for all 91** | Fixed at the freeze. Rationale: Nygaard et al. 2016 |

**Gate 1, how the classifiers are run.** In every training fold:
1. ComBat is fitted on the training controls (no label) and applied to the held-out fold (`combat_apply`).
2. Volumes are residualised on age, age², sex and eTIV, fitted on the training rows.
3. The classifier is trained on the training rows and scored on the held-out rows.

Step 2 is there because ComBat keeps covariate effects on purpose. Without it, a classifier would detect batch from
the PPMI-1/2.0 age difference rather than the scanner. The two classifiers are fixed:
- multinomial logistic regression (L2, C = 1, standardised inputs);
- random forest (500 trees, seed as the fold seed).

**Gate 1 may fail even if ComBat works as designed.** ComBat removes each feature's location and scale, not batch
differences in covariance. Chen et al. 2022 showed that machine learning still identified scanner manufacturer after
ComBat in ADNI, and that CovBat removed it. A linear classifier on residuals cannot see covariance; the random forest
can. Requiring both is the stricter reading, chosen so that a pass supports multivariate (classifier) use. If only
the forest fails, the reading is "location and scale harmonised, covariance not". CovBat is then the next step before
any classifier pools PPMI batches or cohorts.

**Gate 3: why it is measured against a within-batch reference.** Protecting the label keeps group differences, but
when the PD share differs by batch it can overstate them. The 9 batches here run from 59 % to 88 % PD. The
unharmonised regression with batch fixed effects is the reference: it estimates the within-batch difference without
any harmonisation step.

**Reported regardless:**
- the flow, the batch × diagnosis table and the excluded batches;
- the confounding report of the label-protecting fit (`harmonize.confounding_report`): counts, PD share and covariate means per batch, Cramér's V, flagged batches;
- gate 1 per classifier: mean, minimum and maximum over the repeats, for harmonised and unharmonised volumes, plus the covariates-only AUROC (age, sex, eTIV) as the demographic floor;
- gate 2: the three βs per structure, and the pooled unharmonised β without batch terms;
- gate 3: for every volume, d within-batch, d protected and d unprotected; the median |d| of each. Unprotected d is expected to shrink toward 0 where the PD share differs by batch.

## What a pass would and would not mean

- **A pass would mean** that, for these 91 FastSurfer volumes in PPMI healthy controls:
  - PIE's spline-age ComBat removes PPMI's own scanner and protocol-phase differences, so that neither a linear nor a nonlinear classifier can tell the batch;
  - it keeps the age effects;
  - it does not inflate PD–HC differences when diagnosis is protected.

  The next stages are a public BIDS dataset, then ADNI.
- **A pass would not mean:**
  - cross-cohort harmonisation works: other scanners and populations, and labels entangled with cohort (SAA about 88 % positive in PPMI PD against about 22 % in ADNI);
  - other modalities (DWI, NM, FLAIR) are harmonised;
  - harmonised features predict better.
- **A failure** is reported with the gate that failed.
  - Gate 1 failing for the forest only points to covariance (CovBat).
  - Gate 1 failing for the logistic model, or gate 2 or 3 failing, means ComBat is not used on PIE's T1 volumes until it is understood.
  - A rerun is a new, dated pre-registration.

## Deviations policy

After the freeze, any change is logged in the run log below, with its date, before the script is run.
- A bug fix that changes no feature, batch rule, threshold, covariate or statistic is a logged deviation. The synthetic dry run must pass again and the new sha256 is recorded.
- Anything else is a new pre-registration.

The script is run once. Every number it writes is reported.

## Analysis code

`scripts/analyse_harmonisation_ppmi.py` (`venv_imaging`):
- `PYTHONPATH=. venv_imaging/bin/python scripts/analyse_harmonisation_ppmi.py --derived Imaging/derived --ppmi PPMI --out <dir>` is the real run.
- `--synthetic` is the dry run, on fully generated data: 9 batches with additive and multiplicative effects, a curved age effect, and a PD share that differs by batch.
- `tests/test_analyse_harmonisation_ppmi.py` checks it on synthetic data:
  - known batch effects are detected before harmonisation (AUROC > 0.9) and not after;
  - a batch under 10 controls is dropped;
  - all gates pass;
  - unharmonised volumes fail gate 1.

sha256 at the freeze: in the run log.

## Run log

- 2026-10-15: **frozen**, gates as drafted (approved). sha256: `analyse_harmonisation_ppmi.py` `e8ecc147a7c1fe52c76be6cbb32b10150224e5a91c814b00bbf1726db447b592`; `harmonize.py` `b67d0bb66da724fdf4b5fffe633b306a15a290c18e4cac8eb0bfeff8bf4f6245`.
- 2026-10-15: drafted. The synthetic dry run passed:
  - batch AUROC, logistic: 1.000 raw → 0.483 harmonised;
  - batch AUROC, forest: 0.997 → 0.495;
  - age relative difference ≤ 0.017;
  - maximum \|d gap\| 0.031.

  Batch counts were taken from the real tables (counts only).

## References

- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* 2007;8:118–127. doi:10.1093/biostatistics/kxj037.
- Fortin J-P, et al. Harmonization of cortical thickness measurements across scanners and sites. *NeuroImage* 2018;167:104–120. doi:10.1016/j.neuroimage.2017.11.024.
- Pomponio R, et al. Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. *NeuroImage* 2020;208:116450. doi:10.1016/j.neuroimage.2019.116450.
- Nygaard V, Rødland EA, Hovig E. Methods that remove batch effects while retaining group differences may lead to exaggerated confidence in downstream analyses. *Biostatistics* 2016;17:29–39.
- Chen AA, et al. Mitigating site effects in covariance for machine learning in neuroimaging data. *Hum Brain Mapp* 2022;43:1179–1195.
