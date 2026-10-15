# Pre-registration: staged validation of PIE's neuromelanin measure (technical, biological, clinical)

**Status: Stage 1 FROZEN 2026-10-12** (the commit that adds this file, before any split-half value existed; the
split-half slabs were being written when it was committed, and no measure had been computed on them) **and RUN the same
day: technical gate passed, primary `nms_sn_mean_cr`.** **Stage 2b FROZEN 2026-10-12**, before its prodromal slabs were
built, **and RUN the same day: passed (AUROC 0.693 [0.625, 0.758]).** Stages 2a, 2c and 3 are DRAFTS until their data
are downloaded; each is frozen, with its script hash, before its data are opened. **Amendment 1 FROZEN 2026-10-15**
(Stage 3 and Stage 2b-R, `scripts/nm_stage3.py`), before either download was opened. Stages 2a and 2c stay drafts.

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

### Stage 1 result (run once, 12 Oct 2026; script hash verified; result JSON sha256 `13176314cea1ada7f4820f0eeca71642ad819cc81ace2a616b39b20caef3e4e7`)

**Technical gate PASSED. Primary measure: `nms_sn_mean_cr` (the current default, kept by the tie rule).**

Flow: 459 slabs; excluded 37 `QC["nm"]`, 0 not Siemens, 2 slab < 24 mm, 4 halves not made, 10 a candidate missing on
a half; **406 analysed**.

| Candidate | ICC(A,1) halves | Projected (m = 2.5) [95 % CI] | Own rows | Missing | η² scanner model |
|---|---|---|---|---|---|
| `nms_sn_mean_cr` | 0.873 | **0.945** [0.928, 0.957] | 0.945 | 0.024 | 0.014 |
| `nms_sn_mean_cr_mode` | 0.506 | 0.719 [0.641, 0.773] | 0.719 | 0.024 | 0.012 |
| `nmf_sn_mean_cr` | 0.899 | 0.957 [0.943, 0.966] | 0.958 | 0.022 | 0.011 |
| `nmf_sn_mean_cr_mode` | 0.653 | 0.825 [0.781, 0.859] | 0.829 | 0.022 | 0.010 |
| `nms_sn_mean_cr_n4` | 0.847 | 0.933 [0.900, 0.954] | 0.933 | 0.024 | 0.012 |
| `nms_sn_mean_cr_mode_n4` | 0.514 | 0.726 [0.656, 0.778] | 0.726 | 0.024 | 0.009 |
| `nmf_sn_mean_cr_n4` | 0.876 | 0.947 [0.922, 0.963] | 0.948 | 0.022 | 0.009 |
| `nmf_sn_mean_cr_mode_n4` | 0.651 | 0.823 [0.776, 0.856] | 0.826 | 0.022 | 0.009 |

The best projected ICC (`nmf_sn_mean_cr`, 0.957) beat the default by 0.012, inside the 0.02 tie margin, so the
default stands. Full-slab Spearman agreement with the default: 0.88 (`nmf_sn_mean_cr`), 0.91 (with N4), 0.76–0.83 (the
mode variants).

Unexpected, reported as found: the crus-mode reference **lowered** reliability (0.72–0.83 against 0.93–0.96 with the
mean), the opposite of Lesh 2025, whose gain was for manual tracing. A kernel-density peak over a small crus mask is a
noisier statistic than its mean when the mask is automated. N4 did not help either. Scanner model explains about 1 %
of the variance of every candidate.

Limit (as stated above): halves share the session, registration and segmentation, so 0.945 is an upper bound on
scan–rescan reliability. Stage 2b uses `nms_sn_mean_cr`.

## Stage 2: biological validation (draft; new data)

PPMI's acquisition metadata (Sep 2026) lists NM at ≥ 2 visits for 546 PD and 1,150 prodromal participants, but only
5 HC; PIE holds one NM session per participant.

- **2a, decline in PD (primary).** Annualised change of the Stage 1 primary measure across each PD participant's NM
  visits, every visit registered to the participant's baseline T1 (`nm --session-index k`, one work dir per k, then
  `nm_template syn` and `nm_native` per work dir); pairs with a scanner change excluded
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
  - **Result (run once, 12 Oct 2026; script hash verified; features sha256 `4010ad32…c193` recorded before the
    label join; result JSON sha256 `9cec65d0…fe2`): PASSED.** Primary `nms_sn_mean_cr` (from Stage 1).
    - Flow: 433 sessions; 8 slabs failed (7 ambiguous acquisition identity, 1 empty reference region); 425 measured;
      excluded 36 `QC["nm"]`, 117 slab < 24 mm, 1 measure missing; 271 prodromal; **267 with a DaT scan within 18
      months**: 141 DaT-normal, 126 DaT-deficient (NSD-ISS D+). Vendor strata (normal / deficient): Siemens 114 / 107,
      GE 23 / 16, Philips 4 / 3.
    - **AUROC 0.693 [0.625, 0.758]** (DaT-normal > DaT-deficient, vendor-stratified).
    - Secondary: 0.65 rule 0.615 [0.541, 0.688] (195 / 72); age- and sex-adjusted 0.685 [0.616, 0.750]; Spearman ρ
      0.264 with percent-of-expected SBR (n 267), inside Park 2023's 0.25–0.33.
    - Note: the slab ≥ 24 mm rule removed 117 sessions (27 %), more than in R1; the result holds for the sessions with
      full nigral coverage.
  - **Correction, 15 Oct 2026 (disclosure; the computed result is unchanged).** The freeze text above says the 433
    sessions were ones "no NM analysis has used". That was wrong. 224 of the 267 analysed participants have the same NM
    scans that the Study 1 manuscript's NM analysis used (also produced with PIE), and 43 are new to it. Study 1 had
    already compared its own whole-nigra contrast with DaT deficit in those participants, with a null result, before
    this stage was frozen. So Stage 2b is a pre-registered test of a new measure, but not on unseen scans.
    The Study 1 session then ran a post hoc comparison (aggregates in that study's `summary_nm_coverage.json`) on the
    178 participants shared with its DaT analysis:
    - Study 1's whole-nigra contrast gave AUROC 0.637 against DaT deficit; PIE's `nms_sn_mean_cr` gave 0.648. The
      two measures correlate at Spearman 0.84.
    - Study 1's null came from slabs under 24 mm (median 20 mm, almost all GE). There, its contrast gave AUROC 0.398;
      on slabs with full coverage it gave 0.634.

    Most of Stage 2b's margin over Study 1 therefore comes from the slab-coverage rule (≥ 24 mm through-plane), not
    from the snceg measure. Stage 3 draws only participants whose NM is in no PIE download, so it does not share this
    problem. Before it runs, its list is also checked against every NM scan used by Study 1 or by earlier PIE NM
    tests, and any overlap is excluded and counted.
- **2c, side-matched DaT (exploratory, reused data).** Each hemisphere's NM against the same hemisphere's posterior
  putamen SBR, and the more-affected side, as Park 2023.

## Stage 3: clinical validation (draft; fresh participants)

PPMI's metadata lists NM for 41 HC and 411 de novo PD (NM at BL/SC) that no PIE download holds (lists written to
`~/pie_validation/20261012_nm_staged/`, not in the repository). Only 10 of these HC and 75 of these PD have a
processed T1 in PIE, so their T1s are downloaded with them.

- **Measure:** the Stage 1 primary measure, unchanged. Script: `scripts/nm_stage3.py fresh-clinical` (Amendment 1;
  it replaces `nm_staged.py fresh-clinical`, which coded only half the pass rule).
- **Statistic:** AUROC (HC > de novo PD) stratified by vendor (pairs compared only within a vendor, pooled), with a
  stratified bootstrap 95 % CI; Siemens-only AUROC as a secondary.
- **Pass rule (decided 13 Oct 2026, before any Stage 3 data were downloaded):** both of these must hold.
  1. The vendor-stratified 95 % CI lower bound is > 0.50: above chance out of sample.
  2. The CI upper bound is ≥ 0.633, the like-for-like published PPMI GRE-MT benchmark (Langley/Hu ISMRM 2023): PIE
     is not shown to be worse than the published method on the same protocol.

  The point estimate is reported beside 0.633 and the Langley 2025 equivalent of about 0.71.

  The user asked for the rule most likely to hold up to PPMI/MJFF review. Requiring the point estimate to reach 0.633
  was rejected: with about 35 HC it would fail about half the time even if PIE matched the published method. The
  test would be uninformative, and the threshold would look picked to pass or fail. Requiring the lower bound to
  exceed 0.633 cannot be powered.

  With about 35 usable HC and 300 PD, a true AUROC of 0.63 passes about 74 % of the time (simulated, DeLong; 30 HC /
  250 PD: 67 %). Condition 2 removes almost nothing at that effect, because an upper bound below 0.633 needs an
  estimate near 0.54.
- **What `NM_VALIDATED` would mean:** technically reliable (Stage 1), biologically plausible (Stage 2a), and above
  chance out of sample (Stage 3), for the stated context of use. It would not mean diagnostic accuracy.

## Amendment 1 (FROZEN 2026-10-15, before either download was opened)

Written before any Stage 3 or Stage 2b-R data were downloaded. Nothing below uses an outcome.

**Why.**
1. **The pass rule was not fully coded.** `scripts/nm_staged.py fresh-clinical` coded only condition 1 of the 13 Oct pass rule (CI lower bound > 0.50), not condition 2 (CI upper bound ≥ 0.633). Stage 3 therefore runs with `scripts/nm_stage3.py`, which applies both conditions.
2. **`nm_staged.py` is not edited.** It stays byte-identical (sha256 `3ab1bc26…`), because the frozen DWI free-water analysis imports from it. The new script imports its loading, labelling and statistics unchanged.
3. **Stage 2b reused scans** (correction above): 224 of its 267 participants had been analysed by Study 1. Stage 2b is therefore repeated on prodromal participants whose NM no analysis has used (Stage 2b-R).
4. **Overlap is checked.** Every Stage 3 and Stage 2b-R participant is checked against a list of every participant whose NM was analysed before: Study 1's NM inventory, every PIE NM feature table, and the held prodromal sessions. Those participants are excluded and counted (`--exclude`). Built on 15 Oct, the list holds 1,005 participants. It overlaps the Stage 3 lists in 1 HC and 2 PD, and the Stage 2b-R list in none. The list itself stays outside the repository.

**Stage 2b-R: prodromal NM against DaT, fresh participants.**
- **Candidates.** PPMI's acquisition metadata lists NM for 2,534 prodromal participants. 1,266 of them were never analysed. Of those, 1,084 have NM at SC or BL, and 1,027 also have core-lab DaT at SC or BL.
- **Sample.** 400 of the 1,027 were drawn with `numpy.random.default_rng(20261015)` (list outside the repository). They are downloaded with their T1s.
- **Measure, statistic, exclusions and pass rule:** as Stage 2b. Stage 1 primary; vendor-stratified AUROC, DaT-normal > DaT-deficient (NSD-ISS D+); CI lower bound > 0.50. Exclusions include slab < 24 mm.
- **Power.** With about a third excluded (QC, thin slabs, no DaT within 18 months), about 260 remain. A true AUROC of 0.65 then gives a CI lower bound above 0.50 with power above 0.95.

**Stage 3, unchanged except:**
- the overlap exclusion above;
- the pass rule is now coded as decided on 13 Oct (`nm_stage3.stage3_pass`).

**Descriptive secondaries, Stage 3 and Stage 2b-R (no gate, no multiplicity claim):**
- **The lower of the two sides' contrast** (`min(nms_sn_l_cr, nms_sn_r_cr)`). Loss is greater on the more affected side, and "lower side" is defined for controls too.
- **snceg nigral volume** (`nms_sn_volume_mm3`, unadjusted). Meta-analyses find volume discriminates better than signal (Cho 2021).

**Considered and not added: ComBat on the primary.** The primary statistic already compares pairs only within vendor. ComBat on a two-group test would either remove group signal (diagnosis not protected) or let the label into the harmonisation (diagnosis protected).

**Script.** `scripts/nm_stage3.py` (`fresh-clinical`, `prodromal-fresh`, `--synthetic`), tests in `tests/test_nm_stage3.py`. Synthetic dry run only before the freeze.

**Freeze record (15 Oct 2026).** No Stage 3 or Stage 2b-R image had been downloaded. sha256:
- `scripts/nm_stage3.py` `688bb8bebdf0821418a44d36cdc32aab1bd99693327a324e44526e7380269a1c`;
- `scripts/nm_staged.py` `3ab1bc26e088b5e46551a280b31754ce8842a18e68fcbc9e80069c061b4fee3f` (byte-identical to its 12 Oct freeze);
- the lists, outside the repository (`~/pie_validation/20261012_nm_staged/`): HC `loni_nm_hc_unheld_patnos.txt` `95cd70cb3393f560dc427d4520340f6d31aa50d77b3ecac602b34bbc52ffbb11`;
  de novo PD `loni_nm_pd_denovo_unheld_patnos.txt` `f1ef14fbf6bfa08cb23c68f2d2ceefbb7e6352d098b3c535b7b2ee041ce094a9`; prodromal `loni_nm_prodromal_fresh400_patnos.txt`
  `37e775dfb299f3cf24d78c486033ab688ed42f27c169fdca94327680554c0afe`; exclusion `nm_used_before_20261015_patnos.txt` `7226a015917494d06a5f7e48954fa449caed9addca8b722e49bd72e3916c75ee`.

**What `NM_VALIDATED` would mean: unchanged** (Stage 1, Stage 2a, Stage 3). Stage 2b-R is supporting biological evidence.
