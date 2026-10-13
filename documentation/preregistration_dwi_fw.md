# Pre-registration: multi-shell posterior-SN free water in PPMI 2.0 prodromal participants (DaT-deficient vs DaT-normal)

**Status: DRAFT, contrast changed 13 Oct 2026 (user decision).** Freezing was approved; it happens, with a timestamp
and the commit hash of the code, once the multi-shell download is in and the candidate counts below are filled in
from the DWI series index, `sessions.csv`, cohort and DaT labels only. It must be frozen before `dwi --eddy --study`
produces any `dwisf_*` value for the sample. No `dwisf_*` value exists yet.

**Why the contrast changed.** The first draft (11 Oct) compared PD with HC. A count from series names and labels only
(13 Oct) found PPMI 2.0's three-shell protocol (Siemens Prisma, b = 700/1000/2000) in at most 5 healthy controls (4
held, 1 more on LONI), against 80 PD and 242 prodromal. That is fewer than the 5 HC per vendor the HC-referenced
harmonisation needs, so PD − HC could not be estimated. The new contrast mirrors NM Stage 2b
(`preregistration_nm_staged_validation.md`), which passed: within prodromal participants, those with a DaT deficit
against those without.

This is a **post hoc** analysis of a measure chosen after related measures were seen (disclosure below), and it will
be labelled that way.

## Why this endpoint

Posterior-SN free water is the PPMI-supported nigral DWI marker: PD > HC in PPMI-1 (Ofori 2015), d ≈ 0.4–0.5 and rising
over 1 and 4 years in PD but not HC (Burciu 2017, Brain, 10.1093/brain/awx146). PIE's first nigral DWI endpoint,
posterior-SN mean diffusivity, failed its PPMI 2.0 gate (AUROC 0.574 [0.495, 0.652], 218 PD / 82 HC, 29 Sep 2026).
Free water needs a second shell to be well-posed (Golub 2021, MRM 10.1002/mrm.28599). **User decision, 11 Oct 2026:**
multi-shell posterior-SN free water is PIE's pre-registered DWI endpoint.

A nigral marker should track nigrostriatal loss before diagnosis. A DaT deficit in a prodromal participant is
nigrostriatal loss (NSD-ISS stage ≥ 2 biology), so the expected direction is higher posterior-SN free water with a
deficit. PD − HC (d 0.4–0.5) corresponds to AUROC ≈ 0.61–0.64; DaT-deficient prodromal participants are earlier in
the disease, so a smaller separation, about 0.55–0.62, is expected.

## Disclosure: what has already been seen

- **The study's nigral free water:** AUROC 0.688 PD vs HC on PPMI-1 Siemens, and 0.55 on PPMI 2.0 (`2026-09-27-dwi-fw-plan.md`). That was a different estimator and ROI pipeline.
- **PIE's nigral MD gate:** 0.574 (above).
- **PIE's whole-brain median multi-shell free water:** rises with age, r +0.49.
- **PIE's generic-path free-water columns** (`dwi_sn_posterior_*_fw`, affine CIT168, scanner-frame split) exist on disk for earlier runs. No PD-vs-HC comparison of them is known to have been run, and none will be before freezing.
- **No DWI measure of any kind has been compared between DaT-deficient and DaT-normal prodromal participants** (none known).
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

Every PPMI 2.0 prodromal participant (PPMI cohort label) with a three-shell DWI session and a finished FastSurfer T1,
earliest such session per participant. DaT labels come from PPMI's core-lab tables through `labels.dat_labels`
(`dat_d_nsdiss`: the NSD-ISS DaT deficit), matched to the DWI session within 18 months, as in NM Stage 2b. PD
participants with the same protocol enter the secondary analyses only.

Held on 13 Oct: 67 prodromal and 26 PD sessions. LONI lists 175 prodromal and 54 PD more (the 23 Aug search export),
most without a PIE T1, so they are downloaded with their T1s and run through FastSurfer first. Candidate counts are
filled in at freeze; QC counts come after the run:

| | DaT-normal prodromal | DaT-deficient prodromal | PD |
|---|---|---|---|
| Three-shell candidates with a FastSurfer T1 and a DaT label (at freeze) | _ | _ | _ |
| Passing `dwisf_qc_pass` (after the run, in the flow) | _ | _ | _ |

The 28 Siemens sessions with stripped gradient tags are single-shell PPMI-1 and not in this sample.

## Primary analysis and pass rule

**Statistic:** AUROC, P(free water of a DaT-deficient participant > that of a DaT-normal one), with pairs compared only
within a stratum and pooled (`nm_staged.stratified_auc`). Strata are vendor × correction tier (`dwisf_tier`:
topup + eddy, eddy without reverse PE, rigid). The 95 % CI comes from 2,000 bootstrap draws resampled within stratum ×
group (`nm_staged.auc_with_ci`).

**Pass (`DWI_FW_VALIDATED` for this context of use): CI lower bound > 0.50**, the Stage 2b rule. The point estimate is
reported beside the expected 0.55–0.62 and the PD − HC equivalent 0.61–0.64.

**Power** (Hanley–McNeil normal approximation; the deficit share assumed near Stage 2b's 47 %):

| n deficient / n normal | CI half-width at 0.60 | P(pass \| 0.58) | P(pass \| 0.60) | P(pass \| 0.65) |
|---|---|---|---|---|
| 24 / 26 (held only) | 0.16 | 0.16 | 0.24 | 0.48 |
| 60 / 65 | 0.10 | 0.35 | 0.50 | 0.86 |
| 85 / 100 (after the download) | 0.08 | 0.47 | 0.67 | 0.96 |

On the held sessions alone the test is close to uninformative. It is run only after the download.

**Reported regardless, no gate:**
- the participant flow, strata and their sizes;
- the AUROC with age and sex regressed out (fitted on all prodromal; no DaT label enters the fit);
- the AUROC with the 0.65 SBR deficit label in place of NSD-ISS;
- Spearman ρ with percent-expected putamen SBR in prodromal participants (expected negative);
- the positive control: Spearman ρ of free water with age in DaT-normal prodromal participants (expected > 0);
- the specificity check: anterior-SN free water AUROC (expected smaller);
- posterior-SN tissue FA (FAt) AUROC;
- PD against DaT-normal prodromal participants, stratified AUROC (expected > the primary);
- in PD, Spearman ρ with the lowest putamen SBR, each SBR source separately (expected negative).

**Afterwards, not part of this test:** longitudinal change within one correction tier (Burciu 2017); SAA status in
prodromal participants, pre-registered separately; PD − HC if PPMI acquires multi-shell controls.

## Analysis code

`scripts/analyse_dwi_fw.py`:
- `--synthetic` is the dry run. It reads nothing real: every value, label and identifier is generated.
- The real run is `--tables <dwi_features.csv>[,...] --ppmi PPMI --out <dir>`.
- `tests/test_analyse_dwi_fw.py` checks it on synthetic data.
