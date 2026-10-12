# Pre-registration: staged validation of PIE's neuromelanin measure (technical, biological, clinical)

**Status: Stage 1 FROZEN 2026-10-12** (the commit that adds this file, before any split-half value existed; the
split-half slabs were being written when it was committed, and no measure had been computed on them). **Stage 2b
FROZEN 2026-10-12**, before its prodromal slabs were built. Stages 2a, 2c and 3 are DRAFTS until their data are
downloaded; each is frozen, with its script hash, before its data are opened.

## Why staged

PIE's single NM gate (de novo PD vs HC AUROC ≥ 0.70, lower bound > 0.55, ρ ≥ 0.20 with DaT) failed four times on PPMI
2D GRE-MT (0.54–0.67). The 12 Oct 2026 literature review found that this gate sits at the field's ceiling, not at a
realistic target:

- The only published AUROC on the same data (PPMI Siemens GRE-MT, de novo PD vs controls) is **0.633** (Langley,
  Hwang, Huddleston, Hu, ISMRM 2023 abstract #3195); PIE's snceg contrast ratio gave 0.631. The d ≈ 0.83 of Langley et
  al. 2025 (npj Parkinson's Disease 11:181) comes from covariate-adjusted means on 50 controls; no AUROC was reported.
  This was found after the 11 Oct pre-registration, whose failure stands as recorded.
- Automated, multi-site or externally tested early-PD studies reach 0.64–0.85; external tests lose 0.10–0.21
  (Roh 2026 npj PD, doi:10.1038/s41531-026-01286-y: 0.76 internal, 0.64–0.66 external; Lillebostad 2025: 0.86 → 0.74).
- Early-PD NM–DaT correlations are r ≈ 0.25–0.33 on the matched more-affected side in single-site 3D data (Park 2023,
  doi:10.3988/jcn.2022.0147); several studies find none.
- Trujillo et al. 2024 (Brain 147:337, doi:10.1093/brain/awad300): NM-MRI "does not yet have the validation necessary
  to be considered a quantitative imaging biomarker"; there is no consensus method. They separate technical,
  biological and clinical validation, as do the FDA-NIH BEST framework and QIBA.

So PIE validates NM in three stages against a stated **context of use: a group-level research measure of nigral
neuromelanin, not an individual diagnostic test.**

## Changes to the pipeline (12 Oct 2026), all decided without outcomes

| Change | Evidence | Columns |
|---|---|---|
| Crus **mode** as the reference statistic (Gaussian kernel density peak; was the mean) | Cassidy 2019; Wengler 2020; Lesh 2025 (manual ICC −0.14 → 0.56 with the mode) | `nms_sn_*_cr_mode`, `nmf_sn_*_cr_mode` |
| **Fixed-mask contrast**: the CIT168 SNc pulled onto the slab, no model, no threshold | Shaff 2024 HBM (doi:10.1002/hbm.26544): template SNc signal beat thresholds and manual tracing on PPMI's protocol | `nmf_sn_*` |
| **N4 bias-field correction** of the slab before intensities are read (`nm_native --n4`) | Wengler 2021: vendor L/R asymmetry from the receive field; Trujillo 2024 | `nm_native_features*_n4.csv` |
| **Split-half slabs**: repeats 1 + 3 and 2 + 4 averaged separately on the full average's grid (`nm --split-half`) | technical reliability without a rescan | `nm_half_{a,b}.nii.gz` |

The threshold volumes (Langley, snceg volume) are kept as diagnostics: the review confirmed that an SD threshold
moves with the noise level, and both carried no signal on GRE-MT.

## Stage 1: technical validation (frozen; reads no label)

**Data.** The R1 Siemens GRE-MT sessions already processed at `nm.PROCESSING_VERSION` v7
(`Imaging/derived/nm_r1_v7`, 459 slabs; 449 with 5 repeats). Exclusions, counted: `QC["nm"]` fails, not Siemens,
slab < 24 mm, halves not made (fewer than 4 repeats), and any candidate missing on either half.

**Candidates (8).** `nms_sn_mean_cr` (the current default), `nms_sn_mean_cr_mode`, `nmf_sn_mean_cr`,
`nmf_sn_mean_cr_mode`, each without and with N4.

**Metric.** ICC(A,1) between the two halves, projected to the 5-repeat average by Spearman–Brown (m = 2.5), with a
2,000-resample subject bootstrap CI (seed 20261012). Also reported: per-candidate missing rates, ICC on each
candidate's own rows, Spearman agreement between candidates on the full slab, and the share of variance explained by
scanner model (η², descriptive).

**Choice of the primary measure.** The highest projected ICC. The current default is kept unless another candidate
beats it by more than 0.02; otherwise, among the candidates within 0.02 of the best, the earliest in the order above
wins (without N4 before with).

**Technical gate.** Projected ICC ≥ 0.75 ("good", Koo & Li 2016) and its CI lower bound > 0.50.

**Limits, stated in advance.** Halves share one session, head position, slab → T1 registration and scanner, so this
is noise-plus-segmentation reliability, an upper bound on scan–rescan reliability. A measure can be reliable without
tracking neuromelanin; Stages 2 and 3 test that.

Script: `scripts/nm_split_half.py`, sha256 `f3ca5807815c5c920b711e912b358ac7d4502555a89ed4f49c787d76a6ce9079` at the freeze (synthetic dry run only before the real run). Run once; every number reported.

## Stage 2: biological validation (draft; new data)

PPMI's acquisition metadata (Sep 2026) lists NM at ≥ 2 visits for 546 PD and 1,150 prodromal participants, but only
5 HC; PIE holds one NM session per participant.

- **2a, decline in PD (primary).** Annualised change of the Stage 1 primary measure across each PD participant's NM
  visits, every visit registered to the participant's baseline T1; pairs with a scanner change excluded
  (`SCANNER_CHANGE_AFTER_BASELINE`). Pass: mean annual change < 0 with the 95 % CI excluding 0. Anchor: −2 %/yr for the
  contrast ratio in PPMI PD (Lillebostad, Lundervold, Tzoulis, EAN 2025 ePoster EPO-131). Prodromal change is
  descriptive.
- **2b, NM against DaT within prodromal participants (FROZEN 2026-10-12, before any prodromal slab was built).**
  PIE holds 433 prodromal NM sessions that no NM analysis has used. They are processed at v7 like R1 (job
  `nm_prodromal.sh`: slabs, SyN, every candidate with and without N4); the feature table's sha256 is recorded before
  any label is joined.
  - Groups: PPMI's DaT SBR nearest the NM session (`labels.dat_labels`, within 18 months), DaT-deficient by the
    NSD-ISS D+ rule (lowest putamen ≤ 75 % of the age/sex expectation, `dat_d_nsdiss`) or not.
  - Statistic: AUROC (DaT-normal > DaT-deficient) of the Stage 1 primary measure, stratified by vendor (pairs only
    within a vendor), bootstrap 95 % CI (2,000 resamples within vendor × group, seed 20261012).
  - **Pass: CI lower bound > 0.50.** Exclusions as Stage 1 (QC, slab < 24 mm, measure missing), any vendor.
  - Secondary: the 0.65 deficit rule (`dat_deficit_sbr`); age- and sex-adjusted (fit on all prodromal rows, no DaT
    label); Spearman ρ with percent-of-expected SBR.
  - Rationale: nigral neuromelanin loss should accompany striatal dopamine-transporter loss before diagnosis (within
    iRBD, NM separated DaT-abnormal from DaT-normal at AUROC 0.73–0.75, Kim 2026, doi:10.1007/s11682-026-01176-0); no
    HC or PD participant is reused.
  - Script: `scripts/nm_staged.py prodromal-dat`, sha256 `3ab1bc26e088b5e46551a280b31754ce8842a18e68fcbc9e80069c061b4fee3f` at the freeze (synthetic dry run only).
- **2c, side-matched DaT (exploratory, reused data).** Each hemisphere's NM against the same hemisphere's posterior
  putamen SBR, and the more-affected side, as Park 2023.

## Stage 3: clinical validation (draft; fresh participants)

PPMI's metadata lists NM for 41 HC and 411 de novo PD (NM at BL/SC) that no PIE download holds (lists written to
`~/pie_validation/20261012_nm_staged/`, not in the repository). Only 10 of these HC and 75 of these PD have a
processed T1 in PIE, so their T1s are downloaded with them.

- **Measure:** the Stage 1 primary measure, unchanged. Script: `scripts/nm_staged.py fresh-clinical` (draft).
- **Statistic:** AUROC (HC > de novo PD) stratified by vendor (pairs compared only within a vendor, pooled), with a
  stratified bootstrap 95 % CI; Siemens-only AUROC as a secondary.
- **Proposed pass rule (the user decides before the data are opened):** CI lower bound > 0.50, the point estimate
  reported beside the like-for-like benchmark 0.633 and the Langley 2025 equivalent of about 0.71. With about 35
  usable HC and 300 PD, a true AUROC of 0.63 passes 74 % of the time (simulated, DeLong; 30 HC / 250 PD: 67 %).
- **What `NM_VALIDATED` would mean:** technically reliable (Stage 1), biologically plausible (Stage 2a), and above
  chance out of sample (Stage 3), for the stated context of use. It would not mean diagnostic accuracy.
