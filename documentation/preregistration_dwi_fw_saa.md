# Pre-registration: posterior-SN free water and CSF seed status (SAA) in PPMI prodromal participants

**Status: FROZEN 30 Oct 2026** (user-approved, all three decisions below as recommended), **amended 30 Oct 2026
before the run** (user-approved; see Amendment 1). Drafted 25 Oct; frozen
before the frozen DaT test (`preregistration_dwi_fw.md`) was run. The commit that adds this line is the frozen code.
At the freeze:
- No `dwisf_*` value had been looked at.
- The counts below come from the DWI series index, the paper's frozen clinical-cohort tables, and labels and dates only.

This is the follow-on that `preregistration_dwi_fw.md` names: "SAA status in prodromal participants, pre-registered
separately". It is **post hoc** with respect to the paper, and it will be labelled that way.

## Why

The paper's question is whether imaging adds to a clinical model for concurrent CSF SAA positivity in prodromal
participants. Its diffusion answer has a gap:
- Nigral free water failed its gate and entered no prodromal model.
- Of the 229 clinical-cohort diffusion participants, only 41 had three-shell scans, and only their b ≈ 1000 shell was
  used (Supplement S5).

PIE's multi-shell estimate, `dwisf_fw_sn_posterior_mean`, now exists for PPMI 2.0's three-shell protocol. Supplement S5
says a pre-specified multi-shell analysis "is under way in PIE". This document is that analysis, run on the paper's
own cohort and engine.

## Gate: the outcome-free control comes first

The analysis runs only if the frozen DaT test passes (`DWI_FW_VALIDATED`: DaT-deficient vs DaT-normal prodromal
participants, CI lower bound > 0.50). This follows the paper, where every measure first passed an outcome-free control
and failed measures entered no prodromal model, and the DWI re-test (`dwi_retest_20260929`, "if Part 1 fails, Part 2
does not run").

If the DaT test fails, this analysis reports "not run" and nothing else. `--dat-result` enforces this.

## Disclosure: what has already been seen

- **No `dwisf_*` value** of any participant has been looked at.
- **An earlier exploratory PIE multi-shell fit** covered the 41 clinical-cohort three-shell participants. It was made
  before the atlas correction, with rigid motion correction only. Supplement S5 says the superseded values entered no
  reported analysis. A search of the paper's repository (25 Oct) found no record of a seed-status comparison on them.
  Even so, treat at most 41 participants as possibly exposed.
- **The paper's seed-status results** for other measures in these participants:
  - Seed-positive participants were about 4 years older and more often male.
  - Nigral neuromelanin was lower in seed-positive participants (adjusted OR 0.79, 0.62–1.00).
  - Tract FA and caudate network weight differed unadjusted, but not after age and sex adjustment.
  - No imaging family added to the clinical model.
- **PIE's whole-brain free water rises with age** (r +0.49). An unadjusted seed-status difference would therefore be
  expected from age alone, which is why the primary is adjusted.
- **The DaT test's result will be known before this runs**, because it is the gate. It is the same scalar on largely
  the same participants. This is disclosed, not avoidable.

## Measure, sessions and QC

These are identical to the DaT test (`analyse_dwi_fw.load`, imported unchanged):
- `dwisf_fw_sn_posterior_mean`, from each participant's earliest three-shell PPMI 2.0 session;
- `dwisf_qc_pass` and no error;
- the same values, not re-processed.

## Sample

Prodromal participants who pass the DaT test's QC and are in the **paper's clinical cohort**, all three roles. The
cohort comes from the frozen `saa_triage_20260909/features.parquet` and its label files:
- PPMI `COHORT` 4;
- an unambiguous baseline CSF SAA result;
- no SAA-based eligibility;
- no definite-disease diagnosis;
- all four clinical scores.

The DWI session must also fall within **18 months of the CSF collection month**, the same window the DaT test used for
DaT labels. The outcome is the paper's `saa_prodromal`.

Counts at the draft, from labels and dates only; QC counts come after the run:

| Step | Participants | SAA+ | SAA− |
|---|---|---|---|
| Prodromal with a three-shell session (DaT test sample) | 466 | | |
| … in the paper's clinical cohort | 342 | 232 | 110 |
| … DWI within 18 months of CSF | 332 | 227 | 105 |
| … within 6 months (sensitivity) | 261 | 179 | 82 |
| Passing QC with a finished T1 (after the run) | _ | _ | _ |

- **Clinical-cohort roles:** 209 of the 342 are in the paper's held-out test set, 72 in calibration and 61 in
  development. Within 18 months: test 202 (139 SAA+ / 63 SAA−), calibration 70 (47 / 23), development 60 (41 / 19).
  The analyses use all roles. ~~The prediction analysis below cross-validates in all of them, as the paper's own
  MRI-versus-clinical comparisons did (`mri_clinical_comparator_20260918`).~~ **Corrected 30 Oct (Amendment 1):** the
  paper's MRI-versus-clinical comparisons were development-only (274 NM and 229 DTI participants, all development
  role; their PROTOCOL.md and Methods say so). This is the first structural or diffusion comparison in the paper's
  cohort to use calibration- and test-role participants.
- **Gap from the CSF month:** median 0 months, 90th percentile 13.5 months.

## Primary analysis (biology) and decision rule

**Statistic:** age- and sex-adjusted posterior-SN free water.
- The adjustment is a linear age + sex fit (`nm_staged._adjust`) on every QC-passing prodromal participant of the DaT
  test, so no SAA label enters the fit.
- The AUROC is P(adjusted free water of an SAA+ participant > that of an SAA− one), comparing pairs only within vendor ×
  correction tier and pooling.
- The 95 % CI comes from 2,000 bootstrap draws within stratum × group (`nm_staged.auc_with_ci`, the DaT test's
  statistic).

**Associated (`FW_SAA_ASSOCIATED`) if the CI lower bound > 0.50.** The expected direction is higher free water in SAA+.

**Expected size:** small. SAA positivity is broader biology than a DaT deficit, since many SAA+ prodromal participants
have no deficit yet, so the AUROC should be below the DaT test's.

**Power** (Hanley–McNeil):

| SAA+ / SAA− | CI half-width at 0.58 | P(associated \| 0.55) | P(associated \| 0.58) | P(associated \| 0.60) | P(associated \| 0.65) |
|---|---|---|---|---|---|
| 227 / 105 (all within 18 months) | 0.065 | 0.32 | 0.68 | 0.87 | 1.00 |
| 182 / 84 (~80 % pass QC and have a T1) | 0.072 | 0.27 | 0.58 | 0.79 | 0.99 |

## Prediction analysis: the paper's increment

**Engine:** the paper's nested engine, unchanged, called through its own fold driver
(`mri_clinical_comparator_20260918/run.py:fold`, the route `dwi_retest_20260929/analyse_dwi.py:estimand_b` used). It
compares the clinical model + free water (one feature, `dwi_fw_sn_posterior`) against the clinical model alone.
- **Clinical model:** age, sex, LRRK2, GBA, SNCA, APOE ε4, UPSIT, RBD 12-item, SCOPA-AUT 21-item and UPDRS-III, from the
  frozen triage table.
- **Cross-validation:** the same grid, seeds (20260909/19/29), 5 × 3 outer folds and 3-fold inner selection.
- **Interval:** 2,000-draw paired bootstrap, giving 95 % and 98.75 % intervals.
- **Runs:**
  - unadjusted;
  - scanner-adjusted, with batch columns manufacturer, scanner model and correction tier.
- **Minimum:** the paper's 80 participants with 20 per class.

**Increment demonstrated only under the paper's rule:** all three must hold.
- The 98.75 % lower bound is above 0.
- The difference is positive in all three repeats.
- There is no reversal under scanner adjustment.

**Reported regardless:**
- the ΔAUROC for the selected model and for the free-water-required model;
- both intervals;
- whether the 95 % upper bound is below +0.05, the paper's detectability benchmark.

## Reported regardless, no gate

- the participant flow, strata and their sizes;
- the unadjusted stratified AUROC and a one-sided Mann–Whitney P, in the form of the paper's post hoc single-measure
  tests;
- the odds ratio per SD, from a logistic model of SAA on z(free water) + age + sex, in the form of the paper's adjusted
  odds ratios;
- within DaT-normal participants only, the adjusted AUROC SAA+ vs SAA− (NSD-ISS stage 1 vs 0: does free water track
  synuclein biology before dopaminergic loss?);
- SAA × DaT: n and median adjusted free water per cell (the paper's neuromelanin table);
- DWI within 6 months of the CSF month: the adjusted AUROC;
- anterior-SN free water, adjusted AUROC (specificity; expected smaller);
- by clinical-cohort role (Amendment 1): the analysed counts by role × SAA, and the adjusted AUROC within test-role
  participants and within development + calibration participants.

## What a result would mean

- **Associated, with the increment demonstrated:** posterior-SN free water carries seed-status information that the
  clinical model does not.
- **Associated, with no increment:** the biology is there, but it adds nothing over smell and the other clinical
  measures. This would be the paper's pattern for neuromelanin.
- **Not associated:** in this cohort, multi-shell posterior-SN free water does not separate seed status after age and
  sex. The CI bounds the size of any effect.
- None of these would be a claim of diagnostic accuracy.

## Analysis code

**PIE code:**
- `scripts/analyse_dwi_fw_saa.py`, with `tests/test_analyse_dwi_fw_saa.py` (synthetic only).
- It imports `analyse_dwi_fw.load` and `nm_staged` unchanged.
- It reads the paper's repository read-only.

**Real run** (once, after the DaT test, with the same feature tables):
```
PYTHONPATH=. python scripts/analyse_dwi_fw_saa.py --tables <dwi_features.csv>[,...] --ppmi PPMI \
    --study ~/Parkinsons/study1_virtual_biomarkers --dat-result <DaT out>/fw_result.json \
    --out ~/pie_validation/20261025_dwi_fw_saa
```

**Dry run** (`--synthetic`): every value, label and identifier is generated. It ran end to end through the paper's real
engine (37 s).

**sha256 at the freeze (30 Oct 2026)**, every input re-hashed; all unchanged since the 25 Oct draft except the new script:

| File | sha256 |
|---|---|
| `analyse_dwi_fw_saa.py` | 00c6cd25ea243deec7ca87349e98a89dc01b7118116a4a29d7cf4b8e87ffd358 (Amendment 1; frozen 30 Oct as 292c3c43d844dc39912ac679a8314391101dd098cd03e04aedb622cf4cd66a1c) |
| `analyse_dwi_fw.py` | ef22dda5d559c9f65b117faa10e9bbc3f4c110ca9714a26a25912993fe557d17 (the DaT test's frozen hash) |
| `nm_staged.py` | 3ab1bc26e088b5e46551a280b31754ce8842a18e68fcbc9e80069c061b4fee3f |
| Paper: `scientific_prediction.py` | f27d8cff353988a5222de255597b927cc98fc11d768963e3bd2fc2dc50fae2ff (as in the paper's run) |
| Paper: `run_scientific_prediction.py` | e9e4e137ad5b490625ef7cdd224de749a09cefea6a9f1248de7810de365dbbf5 |
| Paper: `mri_clinical_comparator_20260918/run.py` | 9ff717a28360cdc1915011cbb0a97aabdf22a3197ff569aa2e552cdbf394459c |
| Paper: `saa_triage_20260909/features.parquet` | 11a4e03e10046cf22931df36ecb99d337ee24f94a9a2b636998d7cb1be08eb8a |
| Paper: `development_labels.csv` | b06b123ea9488447bfd200a9356515b20454d10320a5c79c1a92a1617abef670 |
| Paper: `calibration_labels.csv` | f7230308f8adb024c6a7bef2ca303a3a7e2de59eb317e1bb506faa18160ed1b5 |
| Paper: `test_labels.csv` | a32b70c4f92683dd69c33394774c69345bf134d923066141d8fdee5994e79299 |

The paper's `run.py` is its 18 Sep amended version: the hash differs from `plan_as_run.json`'s e3b9c934…. The paper's
`AMENDMENT.md` records the change as computation-identical (a comparator patch for the public deposit).

Run once; every number reported. The result goes to the paper session.

## Decisions (user, 30 Oct 2026: all three as recommended)

1. **Gate on the DaT test** (recommended, as the paper and the DWI re-test did). The alternative is to run regardless
   and label the result as descriptive.
2. **Primary adjusted for age and sex** (recommended, because seed-positive participants are about 4 years older and
   free water rises with age). The alternative is unadjusted, like the paper's post hoc single-measure tests, which are
   reported anyway.
3. **The increment uses the paper's 98.75 % rule** (recommended, so free water meets the same bar as T1, NM and DTI).
   The alternative is the 95 % interval, as the paper's Amendment 3 used.

## Amendment 1 (30 Oct 2026, before the run; user-approved)

The paper's session found that the frozen text wrongly said the paper's MRI-versus-clinical comparisons used all
clinical-cohort roles. They used development-role participants only. Recounted from the frozen tables, NM had 274
participants and DTI 229, all in the development role.

**Changes:**
- The sentence is corrected in place, with the old text struck through.
- All roles are kept. Development + calibration alone would leave at most 130 participants within 18 months, too few
  for this test.
- No gate, sample, statistic or decision rule changed.
- Added, reported with no gate: the analysed counts by role × SAA, and the adjusted AUROC within test-role participants
  and within development + calibration participants.
  - They were added before any `dwisf_*` value was seen, so they are prospective.
  - The new code computes only these additions. It has a synthetic test, and the synthetic dry run was repeated.

At the amendment, no `dwisf_*` value had been looked at, and neither the DaT test nor this analysis had run.
