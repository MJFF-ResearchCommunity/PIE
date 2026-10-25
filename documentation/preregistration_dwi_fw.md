# Pre-registration: multi-shell posterior-SN free water in PPMI 2.0 prodromal participants (DaT-deficient vs DaT-normal)

**Status: FROZEN 2026-10-14** (user-approved; contrast changed 13 Oct by user decision), before `dwi --eddy --study`
produced any `dwisf_*` value for the sample. The commit that adds this line is the frozen code. The candidate counts
below come from the DWI series index, `sessions.csv`, cohort and DaT labels only. T1s for most participants were still
to be downloaded at the freeze; the sample is defined by the rule below, not by who is processed first.

**Why the contrast changed.** The first draft (11 Oct) compared PD with HC. A count from series names and labels only
(13 Oct) found PPMI 2.0's three-shell protocol (Siemens Prisma, b = 700/1000/2000) in only 4 held healthy controls.
A census of every PPMI DTI image on LONI (14 Oct, search export, no images read) confirmed it: three-shell sessions
exist for 13 controls, 240 PD and 466 prodromal participants. That is fewer than the 5 HC per vendor the HC-referenced
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

Every PPMI 2.0 prodromal participant (PPMI cohort label) with a three-shell DWI session, the earliest such session
per participant (`scripts/dwi_fw_index.py`: two or more of DTI_B700/B1000/B2000_64dir on one date, every series of
that date kept). Each session is measured on the participant's finished FastSurfer T1 nearest in date within 180 days
(`dwi --per-visit`, `batch.visit_jobs`). Without one, the participant is excluded and counted. DaT labels come from PPMI's core-lab tables through `labels.dat_labels`
(`dat_d_nsdiss`: the NSD-ISS DaT deficit), matched to the DWI session within 18 months, as in NM Stage 2b. PD
participants with the same protocol enter the secondary analyses only.

Availability (LONI census, 14 Oct): three-shell sessions for 466 prodromal and 240 PD participants. At the freeze all
466 prodromal sessions are held (`First_Study_DTI_Refined*`, `dwi_validation_multishell*`,
`remaining_prodromal_dwi_R*` zips; 500 participants in the index with 4 HC and 30 PD). The 210 other PD sessions and
the same-visit T1s of 613 participants (399 prodromal, 214 PD) are still to be downloaded. The T1s go through
FastSurfer before their sessions run. An earlier LONI export (23 Aug) proved unreliable for DTI and is not used.

Counts at the freeze (14 Oct), from labels and dates only; QC counts come after the run:

| | DaT-normal prodromal | DaT-deficient prodromal | No DaT label (excluded) | PD (held) |
|---|---|---|---|---|
| Three-shell session held | 275 | 184 | 7 | 30 |
| … with a finished T1 within 180 days at the freeze | 45 | 22 | 0 | 26 |
| Passing `dwisf_qc_pass` (after the run, in the flow) | _ | _ | _ | _ |

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
| 27 / 31 (held only, 73 prodromal) | 0.15 | 0.18 | 0.26 | 0.54 |
| 147 / 220 (the 184 / 275 at the freeze, ~80 % passing QC) | 0.06 | 0.75 | 0.91 | 1.00 |

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

The run: `scripts/dwi_fw_index.py --zips <the zips above> --out <derived>/dwi_fw/dwi_index.csv`, then
`dwi --zips <same> --sessions <derived>/sessions.csv --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/dwi_fw
--eddy --study --per-visit --workers 1`, prodromal first (`--priority`). It is repeated as T1s and the PD download
arrive (finished participants are skipped).

`scripts/analyse_dwi_fw.py`:
- `--synthetic` is the dry run. It reads nothing real: every value, label and identifier is generated.
- The real run is `--tables <dwi_features.csv>[,...] --ppmi PPMI --out <dir>`.
- `tests/test_analyse_dwi_fw.py` checks it on synthetic data.

sha256 at the freeze (14 Oct 2026): `analyse_dwi_fw.py` ef22dda5d559c9f65b117faa10e9bbc3f4c110ca9714a26a25912993fe557d17;
`dwi_fw_index.py` 8f253245ff856e7ef74277cf921bab904ed58f332be260e26ec4735fc584d2df; `nm_staged.py` (its AUROC and CI)
3ab1bc26e088b5e46551a280b31754ce8842a18e68fcbc9e80069c061b4fee3f. Synthetic dry run only before the freeze. Run once;
every number reported.

## Run log (processing only; no `dwisf_*` value was looked at)

- **14–15 Oct 2026: eddy failures while the GPU was shared.**
  - From about 00:30 on 15 Oct, FastSurfer (the new-T1 run) shared the 8 GB GPU with three `eddy_cuda` workers. From then on, almost every eddy run failed with an FSL exception in eddy's outlier-replacement step: 79 participants, against about 20 successes before.
  - The failures stopped when the GPU was no longer shared. One failed participant was re-run alone with full eddy logging, and it completed normally.
  - The failed participants are re-run with the same frozen code (`--retry-errors`, which drops the error rows before re-running them). A pre-retry copy of the table is kept.
  - The GPU now runs one job type at a time: the DWI passes, then the PD arm, then TrUE-Net.
- **15 Oct 2026: registration-cache check.** 7 participants failed the registration-cache provenance check at the end of the new-T1 run, most likely a timing clash while that run was writing.
  - All 496 cached T1→MNI registrations were then checked against their current T1s. All matched, and none was stale.
  - The 30 participants finished before the incident were all measured on their current T1.
  - The 7 are re-run.
- **15 Oct 2026: PD arm queued.** `pd_dwi*.zip` (210 PD, all with a three-shell session; 209 with a finished T1 within 180 days, 1 without any T1) runs after the prodromal pass, with the same frozen command. The restricted index is rebuilt with `dwi_fw_index.py` over all zips, because `load_index` never adds new zips to an existing index.
- **16–20 Oct 2026: pause (scheduling only).** All analysis units were frozen from 16 to 20 Oct.
  - 4 runs that were mid-flight hit their `eddy_cuda` / `topup` time limit on thaw, because the limit counts wall-clock
    time.
- **24 Oct 2026: stale registration caches.**
  - Cause: 35 participants in pass 4, and 31 of the PD arm's first 110, failed the registration-cache provenance check.
    - 461 cached T1→MNI registrations were fitted from 25 to 29 Sep with sampling seed 0.
    - b1a6327 (11 Oct, before the freeze) set the frozen code's seed to `ITK_SEED` (20260909), so the check correctly
      refused to reuse them.
    - Every cache was fitted on the current `orig.mgz`, and the template and transform hashes all matched. The 15 Oct
      check compared the T1s but not the seed.
  - Action, as the check's message prescribes:
    - The 461 were set aside, not deleted, as `*.seed0.tfm` / `*.seed0.json`, so the frozen code refits them.
    - Pass 5 (`--retry-errors`) re-runs every error row with the same frozen command after the PD arm. The GPU runs
      one job type at a time, so TrUE-Net now waits for pass 5 (reordered 24 Oct so the paper's analysis comes first). A pre-retry copy of the table is kept.
  - No `dwisf_*` value had been looked at, and no value was ever computed on a seed-0 registration under the frozen
    code.
