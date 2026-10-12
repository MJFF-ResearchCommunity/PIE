# Pre-registration (draft): snceg neuromelanin contrast on PPMI 2D GRE-MT, Siemens only

**Status: FROZEN 2026-10-11T19:13:49-04:00** (user decision, 11 Oct 2026: freeze as drafted, with the R1 reuse declared). Frozen at
commit `7fb36073764b780288225a31404aba69a60c5474`, which holds this text and `scripts/analyse_nm_snceg.py` (sha256 `52582786b455f09b707cd978c41921fb2ec8f9a79d775370241d1a3114fbc0da`), before any v7 feature table
of the analysis set was written. Counts below are aggregates from tables PIE already holds; no outcome of this test
has been computed. The synthetic dry run (`--synthetic`) printed only group sizes and the decision on fake data.

## Question

Does PIE's default NM measure, the snceg contrast ratio, separate PD from healthy controls on PPMI's Siemens 2D GRE-MT
neuromelanin slabs as well as published PPMI results?

## Measure

- **Primary:** `nms_sn_mean_cr`, the bilateral mean of each side's contrast ratio I_SN / I_crus − 1
  (`nm_native.snceg_features`). The SN is the snceg segmentation, with the pinned public weights (Lillebostad 2025).
  Each side's reference is that side's crus (the lateral parts of the Biondetti background ROI) on the slices that hold
  that side's segmented SN, excluding segmented voxels. Higher in HC.
- Slabs: `nm --keep-nifti` with coverage-weighted repeat averaging and no MP-PCA (snceg was published on plain images).
  Then `nm_template syn` and `nm_native --snceg`, all at `nm.PROCESSING_VERSION` `2026-10-11-coverage-weighted-unrefined-v7`.
- **Secondary (descriptive; they do not decide the outcome):**
  - `nms_sn_volume_mm3`;
  - `nms_sn_mean_cr_allslices` (the reference before 11 Oct 2026, on the same slabs; isolates the reference change);
  - the primary measure adjusted for age and sex;
  - all PD in place of de novo PD.

## Anchors

| Source | Data | Effect |
|---|---|---|
| Lillebostad et al. 2025 (primary anchor) | PPMI TSE, snceg CR vs crus | AUROC 0.743 |
| Langley et al. 2025 (secondary anchor) | PPMI Siemens GRE-MT, SNc volume | d ≈ 0.83, i.e. AUROC ≈ Φ(0.83 / √2) = 0.72 |
| PIE R2 (26 Sep 2026) | PPMI Siemens TSE-FS, snceg CR vs crus (all-slices reference) | AUROC 0.710 [0.611, 0.799]: reproduced |

## Decision rule

AUROC of the primary measure, de novo PD vs HC, with a DeLong 95 % CI (a stratified bootstrap CI, 2,000 resamples
with a fixed seed, is reported beside it).
- **Pass** if the CI contains 0.743 **and** its lower bound is > 0.50 (the R2 rule).
- The secondary rule, that the CI contains 0.72, is reported but does not decide the outcome.

A pass means snceg reproduces the published PPMI effect on GRE-MT within this sample's precision. It does not set
`manifest.NM_VALIDATED`. That still needs PIE's own gate (AUROC ≥ 0.70, lower bound > 0.55, ρ ≥ 0.20 with the
lowest putamen SBR) on participants no NM analysis has seen. The result is reported whatever it is, and the
pipeline is not changed and re-run on this sample afterwards.

## Sample

Siemens 2D GRE-MT, HC and de novo PD, as R1 defined them: one NM session per participant (the earliest passing QC)
with its own T1.

**What has already been seen.** Almost every Siemens HC/PD slab PIE holds was in R1, where snceg was looked at
descriptively.
- R1 (27 Sep 2026) ran snceg on 72 HC and 386 PD Siemens slabs. Its analysed set was 68 HC and 348 de novo PD. On
  MP-PCA slabs with the all-slices crus it gave d 0.39 [0.15, 0.65], AUROC 0.60.
- The NM replication's Siemens set (353) overlaps R1 in 351 participants.
- PIE's main GRE-MT table (649 slabs) holds 23 Siemens HC and 90 Siemens PD. Only 1 HC and 4 PD of them are outside
  R1 and the replication.
- PPMI's metadata lists 112 HC and 631 PD with NM. After R1's 72 Siemens HC and the 24 non-Siemens HC in PIE's table,
  at most about 16 HC remain to download, of any vendor (by subtraction; to be confirmed from LONI).

**Analysis sets.**
1. **Primary: the R1 Siemens sample, re-processed with v7 (declared reuse).** The analyst knows the old-pipeline
   estimate (AUROC 0.60). The v7 changes came from a code audit (2026-10-11 imaging readiness review), not from group
   results:
   - coverage-weighted repeat averaging;
   - the crus restricted to the SN's slices;
   - no MP-PCA;
   - the `n_repeats >= 3` check.

   No further change is allowed on this sample.
2. **Independent subset.** Siemens HC/PD GRE-MT sessions on LONI that are in no PIE download are fetched and reported
   separately. They are decisive on their own only with ≥ 50 HC; below that they are descriptive (see Power).

**Exclusions**, counted at each step and reported:
- `manifest.QC["nm"]` fails (motion, slab coverage, reference homogeneity, clipping, voxel support, `n_repeats >= 3`);
- `nm_vendor` is not SIEMENS;
- `nm_slab_mm` < 24 (the protocol is 16 × 1.5 mm);
- a side whose nigral search region or crus is < 90 % inside the slab, or whose crus on the SN's slices has < 20 voxels
  (`nm_native` sets that side to NaN);
- an empty snceg SN on either side.

## Power

The decision rule's pass probability, by simulation: binormal data, DeLong CI, 2,000 runs per cell.

| HC / PD | CI half-width at 0.743 | P(pass) if true AUROC 0.60 | 0.65 | 0.70 | 0.743 |
|---|---|---|---|---|---|
| 72 / 386 (R1 snceg rows) | 0.054 | 0.01 | 0.22 | 0.77 | 0.95 |
| 68 / 348 (R1 analysed) | 0.067 | 0.03 | 0.26 | 0.78 | 0.94 |
| 50 / 250 | 0.082 | 0.08 | 0.40 | 0.83 | 0.94 |
| 30 / 150 | 0.100 | 0.28 | 0.63 | 0.90 | 0.93 |
| 16 / 150 | 0.099 | 0.28 | 0.56 | 0.76 | 0.87 |

- "CI contains the published effect" passes a true effect about 95 % of the time at any size. What the sample size
  buys is rejecting a smaller true effect.
- At R1's size, a true AUROC of 0.60 (the old estimate) passes 1–3 % of the time, and 0.65 about a quarter of the time.
- With about 16 fresh HC, a true AUROC of 0.60 would pass 28 % of the time. So no fresh-only design PIE can assemble
  is informative, and the primary set is the declared-reuse R1 sample.
- HC are the binding constraint: the PD count matters little once it is several times the HC count.

## Procedure

1. Merge the v7 pipeline. Commit this file and the analysis script. Record the hash and the script's sha256.
2. Dry-run the script on fully synthetic values **and** labels: no real feature value or cohort label may enter a dry
   run.
3. Run the pipeline (commands in the 11 Oct 2026 hand-off). Record the sha256 of `nm_native_features.csv` before any
   cohort label is joined.
4. Run the script once, and report every number it prints, including the exclusion counts and all secondaries.

Any deviation is logged here, with its date, before the script is run.

## Freeze record and deviations

- 2026-10-11T19:13:49-04:00: frozen as above. The analysis script was written at the freeze (the draft referred to it before it existed).
  It implements the rule as drafted. De novo PD = PD whose NM session is PPMI's BL visit, the R1 definition
  (`analyse_r1.py`); the age/sex adjustment uses residuals from an HC-only fit, so no PD label enters it.

- 2026-10-11T20:01: the first start of the pipeline used the 27 Sep T1 tables of the new downloads only (the NM
  hand-off's command), not R1's combined tables (`pie_validation/20260927_nm_final`). It was stopped after 3 slabs,
  before snceg; nothing was inspected, and its work dir was deleted. The run restarted on R1's own inputs
  (sessions, FastSurfer links, Siemens list), as `20260927_nm_final/run_r1.sh` used them.

## References

- Lillebostad PAG et al. Deep-learning segmentation of the substantia nigra from multiparametric MRI. *Imaging
  Neuroscience* 2025;3:IMAG.a.158. doi:10.1162/imag.a.158.
- Langley J, Hwang KS, Huddleston DE, Hu XP, PPMI. Nigral volume loss in prodromal, early, and moderate Parkinson's
  disease. *npj Parkinson's Disease* 2025;11:181. doi:10.1038/s41531-025-00976-3.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves. *Biometrics*
  1988;44:837–845.
