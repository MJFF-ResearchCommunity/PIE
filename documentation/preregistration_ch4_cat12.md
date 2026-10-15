# Pre-registration: nucleus basalis (Ch4) grey-matter volume with CAT12 on PPMI

**Status: FROZEN 2026-10-13; RUN 2026-10-15: NOT VALIDATED** (age and convergence gates failed; reliability passed). Frozen before any CAT12 value existed for a selected scan. One scan outside the selection (a
prodromal participant) was run first to check the plumbing (output names, report fields, run time); no gate statistic
can be computed from it. It took 13.5 min on one thread. It found two bugs, both fixed before the freeze: a missing
`stats/` folder, and the basal forebrain names (CAT12.9 writes "Left/Right Basal Forebrain"). The checks printed
pass/fail only: TIV, Ch4, basal forebrain and total grey matter in plausible ranges; mwp1's grey matter within 5 % of
CAT12's own; the IQR conversion reproduces the rating in CAT12's log (IQR, not SIQR).

## Why

PIE's first Ch4 measure (the Julich-Brain Ch4 map pulled through the NM study's SyN, grey matter from FSL FAST) failed
plausibility on 354 PPMI subjects: no age effect in controls (r −0.05) where ScLimbic's basal forebrain showed −0.43,
ρ 0.17 with ScLimbic, 26 % of the map counted as grey matter (documentation/imaging.md, October validity checks). The
12 Oct 2026 literature review found that every PD, PPMI and SAA Ch4 result used SPM or CAT12 segmentation, a
high-dimensional warp (DARTEL or Shooting) and modulated grey matter summed in a cytoarchitectonic Ch4 map, with head
size from TIV: Grothe et al. 2012 (Cereb Cortex 22:2245, Zaborszky maps at 50 %); Ray et al. 2018 (Brain 141:165, PPMI);
Wisse, Spotorno et al. 2024 (SAA). The method below is that one, chosen from the literature and not from PIE's data.

## Method (`pie.imaging.cat12`, `run --ch4`)

- CAT12.9 r2665 standalone (compiled SPM25, R2023b build of 27 Mar 2025; MATLAB Runtime R2023b Update 10) on the raw T1 (`Imaging/derived/nifti/<patno>/<image>_T1w.nii.gz`),
  CAT12's default segmentation and Shooting registration to its MNI152NLin2009cAsym template (1.5 mm). Changes to the
  shipped batch only drop outputs PIE does not use (surfaces, WM, warps, report figure; `cat12.OVERRIDES`).
- Grey matter modulated for the full warp (`mwp1`, native mm³ per voxel) summed inside the Julich-Brain Ch4
  probability maps (Zaborszky et al. 2008; `atlases.ch4_maps`, pinned by sha256), resampled linearly to the mwp1 grid.
- **Primary measure:** bilateral Ch4 grey matter at p ≥ 0.5 (`vol_ch4_lh_gm + vol_ch4_rh_gm`). Bilateral, because the
  maps themselves are asymmetric (left 261 mm³, right 111 mm³ at p ≥ 0.5 in MNI space). Head size: CAT12's TIV.
- Secondary measures (no gate): probability-weighted Ch4 grey matter; the neuromorphometrics atlas's basal forebrain
  from CAT12's own ROI output.

## Sample (`scripts/ch4_cat12_validation.py select`, reads no Ch4 value)

Baseline (earliest QC-passing) T1 per participant from `fastsurfer_idps.csv`:
- every healthy control: 223;
- a random 223 of the 736 PD (pandas `sample`, seed 20261013);
- both scans of each of the 29 same-day MPRAGE + repeat pairs used for T1 reliability.

504 scans in all. Exclusions, counted: CAT12 failure; CAT12 image-quality rating (IQR) below 60 % (grades E and F,
"critical" and "unacceptable"); missing age, sex, TIV, eTIV or ScLimbic. Sensitivity: IQR below 70 % excluded.

## Gates (all three for `CH4_VALIDATED`)

1. **Age (construct).** In controls, Pearson r between age and the primary measure residualised for TIV and sex (fit
   in controls): **r ≤ −0.30 and the 95 % CI upper bound < 0.** Literature: Ch4 volume falls with age in healthy adults
   (r ≈ −0.3 to −0.5; Ray 2018 in PPMI controls |r| > 0.46). ScLimbic's basal forebrain gave −0.43 in PIE's controls.
2. **Convergence.** Spearman ρ, all analysed participants, between the primary measure residualised for TIV and
   ScLimbic's basal forebrain (left + right) residualised for eTIV: **ρ ≥ 0.40.** The two define different extents (Ch4
   only against the whole basal forebrain), so agreement is expected to be moderate, not high.
3. **Reliability.** ICC(2,1) of the primary measure across the same-day pairs (both scans passing): **≥ 0.75**,
   with a 2,000-resample pair bootstrap CI (seed 20261013) reported.

Reported, with no gate:
- the PD − control difference (Cohen's d on residuals for age, sex and TIV fit in controls); expected near 0 in de
  novo PD (three PPMI studies);
- median volumes against the literature (CAT12 Ch4 about 270 mm³);
- the left/right ratio;
- the IQR distribution;
- Spearman ρ with the failed FAST measure;
- every gate statistic for the secondary measures.

## Limits

Validation here is technical and construct validity. Prognostic validity (baseline Ch4 against later cognitive decline
in PD, Ray 2018) and the SAA association (Wisse/Spotorno 2024) are later stages. Controls and the first Ch4's 57 HC
overlap; the method was not chosen on them.

Script: `scripts/ch4_cat12_validation.py`, sha256 `49c79ed5e1f5122a343cd78e8110ff6c2a4975c5117552dcdf54c8269d54bb9e` at the freeze (synthetic dry run only). Run once; every
number reported.

## Run log and result (15 Oct 2026)

- **Code:** `scripts/ch4_cat12_validation.py` sha256 `49c79ed5…` (as frozen); `pie/imaging/cat12.py` unchanged since the
  freeze commit (1d63c66). `analyse` run once on 15 Oct.
- **Deviation (scheduling only):** the first run stopped on 14 Oct after an out-of-memory kill. One 0.5 mm scan took
  13 GB in CAT12, and three ran at once. A resume script called the frozen `_run_one` on every selected scan without a
  result: 1 mm scans 6 at a time, then 0.5 mm scans 2 at a time. The method was unchanged. Result: 251/251 and 30/30
  completed, 0 failed.
- **Flow:** 223 HC and 223 PD selected and analysed; 0 missing CAT12 results; 0 below the IQR minimum; 58 repeat scans
  (29 same-day pairs), 0 missing.

| Primary: Ch4 GM, p ≥ 0.5 | Result | Gate |
|---|---|---|
| Controls' age r (TIV, sex residualised) | −0.299 [−0.414, −0.174], n 223 | r ≤ −0.30 and CI upper < 0: **fail** (misses by 0.001) |
| Spearman ρ with ScLimbic basal forebrain | 0.102, n 446 | ≥ 0.40: **fail** |
| Same-day ICC(2,1) | 0.761 [0.525, 0.925], 29 pairs | ≥ 0.75: **pass** |

**Verdict: not validated.** Ch4 stays out of PIE's validated measures. ScLimbic basal forebrain remains the basal
forebrain measure.

Reported, no gate:
- **PD − control:** d 0.204 (primary), 0.175 (probability-weighted), −0.035 (neuromorphometrics basal forebrain).
- **Median volume:** 157 mm³ (0.0106 % of TIV), against about 270 mm³ for CAT12 Ch4 in the literature.
- **Left/right ratio:** median 2.04. A roughly symmetric structure should give about 1. This is not a gate, but it
  points to a problem in how the Ch4 region is defined (the left and right probability maps, their resampling, or the
  p ≥ 0.5 cut), not only to a weak biomarker. It is the first thing to examine before any revised, newly
  pre-registered version.
- **IQR:** median 87.3 %; 22.4 % below 70 %. The sensitivity analysis excluding them gave the same age r (−0.299),
  ρ 0.105 (n 445) and ICC 0.761 [0.562, 0.921].
- **Spearman ρ with the failed FAST Ch4:** 0.485.
- **Secondary measures, every gate statistic:**
  - **Probability-weighted Ch4:** age r −0.413 [−0.516, −0.298]; ρ 0.217; ICC 0.867 [0.745, 0.963]; median 358 mm³.
  - **Neuromorphometrics basal forebrain:** age r −0.462 [−0.560, −0.352]; ρ 0.210; ICC 0.956 [0.911, 0.975];
    median 701 mm³.

  Both would meet the age and reliability bars but not convergence. No measure reached ρ 0.40 with ScLimbic.

