# Pre-registration: posterior-SN free water from PIE's preprocessing against QSIPrep 26.0.0 (technical agreement)

**Status: DRAFT, 15 Oct 2026. Not frozen.** It is to be frozen, with the script's sha256, before `select` runs and before
any QSIPrep output exists for a PPMI session. At the draft, no QSIPrep or QSIRecon container has been run on PPMI data
and no `dwiq*` value exists. The code change listed under **Required before freezing** landed on 15 Oct.

## Why

PIE's DWI endpoint, `dwisf_fw_sn_posterior_mean`, comes from PIE's own preprocessing. Its biological test (prodromal
DaT-deficient against DaT-normal) is frozen and running ([preregistration_dwi_fw.md](preregistration_dwi_fw.md)).
QSIPrep is the community-standard diffusion preprocessing (Cieslak et al. 2021, Nature Methods 18:775). This test asks a
technical question only: **does the endpoint depend on PIE's preprocessing choices?** The free-water model, the nigral
regions, the region rule and the QC thresholds are held fixed, and only the preprocessing changes. The agreement between
the two paths bounds the share of measurement error that preprocessing contributes.

What differs between the two paths:

| Step | PIE (`dwi --eddy --study`) | QSIPrep 26.0.0 (`dwi_qsiprep run`, pinned by digest) |
|---|---|---|
| Denoising, unringing | None (the FW run is without `--denoise`) | MP-PCA `dwidenoise`; `mrdegibbs` (`rpg` when `PartialFourier` < 1) |
| Run scaling | Each run scaled to the first run's b0 | QSIPrep's per-run b0 scaling |
| Susceptibility | topup from the reverse-PE b0s (tier `topup_eddy`); none otherwise | TOPUP from the same reverse-PE b0s; SyN-SDC otherwise |
| Motion and eddy currents | FSL `eddy_cuda` (GPU): `--repol --ol_type=sw --mporder=0`, `--slm=linear` below 60 directions | FSL eddy on CPU, QSIPrep's defaults (`dwi_qsiprep.EDDY_PARAMS`: `use_cuda` False, `repol`, `flm` quadratic, `slm` linear, 5 iterations) |
| Grid | Native DWI grid | T1w (ACPC) grid at the finest native voxel size: one more interpolation |
| DWI → T1 | PIE's rigid b0 → FastSurfer T1 | QSIPrep's b0 → T1w coregistration, then a rigid QSIPrep T1w → FastSurfer T1 (`compare`) |
| Motion value for QC | eddy mean RMS displacement (mm) | QSIPrep mean framewise displacement (mm) |

Held fixed:
- the free-water fit (`freewater_qc.fit_multishell_checked`: DIPY multi-shell NLS, b ≤ 2050, a non-physical fit is missing);
- the CIT168 SNc + SNr through the same cached T1 → MNI SyN, pulled by nearest neighbour onto each pipeline's own DWI grid;
- the anatomical (MNI y) posterior split;
- the region rule (≥ 3 voxels and ≥ 80 % of the region) and the bilateral mean;
- the `dwisf_qc_pass` checks and thresholds.

Each pipeline is measured in its own preprocessed DWI space: PIE on the native grid, QSIPrep on its resampled T1w grid.
Neither moves DWI data into MNI space.

## Required before freezing: `compare` must measure the endpoint as `dwisf_*` does

`dwi_qsiprep.compare_subject` (code at 1f097a1) does not yet produce a like-for-like value:
- **Wrong split.** Its `dwiq_fw_*` regions come from `dwi.study_rois`, which splits the SN on the scanner-frame y (the `dwis_*` tensor rule). `dwisf_*` uses the anatomical split (`study_nigra_labels(..., posterior=True)`, MNI y).
- **No free-water QC.** `dwiq_qc_pass` is the tensor QC on the b = 1000 run. No FAt or MDt is written.
- **T1 not pinned.** The FastSurfer T1 is chosen per participant (`batch.fastsurfer_by_patno`), not taken from PIE's row (`fs_image_id`), and it is not recorded. The export takes PIE's session through `batch.session_rows`, which matches PIE only when both use the same T1 date.

**Done (15 Oct 2026, before any QSIPrep run):**
- `compare_subject` pulls the `posterior=True` codes once on QSIPrep's grid; the plain study labels are those codes mod 100, the same nearest-neighbour pull.
- It calls `dwi.study_fw_features` on the multi-shell fit's `f`, `tissue_fa` and `tissue_md`, with QSIPrep's mean FD as the motion value.
- It writes `dwiqf_{fw,fat,mdt}_{sn,sn_posterior,sn_anterior}_{l,r,mean}`, `dwiqf_qc_pass` and `dwiqf_qc_failed_checks`. The legacy scanner-split `dwiq_fw_*` columns stay.
- With `--pie-features`, `compare` measures on the FastSurfer subject named by PIE's row (`fs_image_id`) and records it as `dwiq_fs_image_id`.
- The export reads PIE's restricted series index (`dwi_fw_index.py`: one three-shell session per participant), so it exports the session PIE measured.
- Synthetic test: `tests/test_dwi_qsiprep.py::test_compare_subject_on_synthetic_qsiprep_and_qsirecon_trees`.

`scripts/analyse_dwi_qsiprep_agreement.py` reads these columns.

## Sample (`select`, outcome-blind)

**Candidates.** Every PPMI 2.0 session (acquired 2020 or later) in PIE's free-water tables (`<derived>/dwi_fw/dwi_features.csv`, after the frozen
FW run finishes) that meets three conditions:
- no `error`;
- `dwisf_qc_pass`;
- correction tier `topup_eddy` or eddy without reverse PE (`eddy_no_reverse_PE`, "eddy-only").

The earliest such session per participant is used. Rigid tiers are not drawn: QSIPrep always runs eddy, so they would
confound the comparison.

**Draw.** Candidates are sorted by participant, then given a seeded random order within each tier (numpy
`default_rng(20261015)`). The sample is the first 20 of each tier. If a tier has fewer than 20, all of it is taken and
the other tier makes up the 40.

**Replacement.** A technical failure is replaced by the next session in the same tier's order and counted by exception type:
- export fails (no T1, or a phase encoding PIE does not know, where QSIPrep stops by design);
- QSIPrep fails;
- `compare` raises.

A QSIPrep QC failure is **not** replaced (below). Replacing it would select on agreement.

**Why the selection is outcome-blind.**
- **No measured value or label is read.** `select` reads five columns only (`patno`, `acquisition_date`, `error`, `dwisf_qc_pass`, `dwisf_tier`; pandas `usecols`). No free-water value, cohort or DaT label enters the selection.
- **Labels cannot enter the result.** Agreement compares two measurements of the same scan. Labels play no part, so the draw cannot favour the DaT result and the DaT result cannot shape the draw.
- **The frozen test is untouched.** It is not re-run or re-analysed with QSIPrep values.

**Counts at the draft (15 Oct 2026).** The FW run is in progress: 33 sessions written, 28 passing `dwisf_qc_pass` in
PPMI 2.0, all Siemens `topup_eddy`. Eddy-only sessions may therefore be rare in the three-shell protocol. The candidate
counts per tier and the quotas are recorded in the run log at `select`.

## Measure and analysis

**Primary pair:** `dwisf_fw_sn_posterior_mean` (PIE) against `dwiqf_fw_sn_posterior_mean` (QSIPrep). It is computed
over sessions where both free-water QCs pass and both runs used the same FastSurfer T1.

**Statistics** (`pie.stats.agreement`):
- **Primary:** ICC(2,1) (two-way random effects, absolute agreement, single measurement; Shrout & Fleiss 1979), with a 95 % CI from 2,000 pair-bootstrap resamples (seed 20261015).
- **Lin's concordance:** CCC (Lin 1989).
- **Bland–Altman:** bias of QSIPrep − PIE with 95 % limits of agreement, and the bias as a percentage of PIE's mean.
- **Pearson's r.**

| Gate | Statistic | Threshold | Source |
|---|---|---|---|
| Agreement (`QSIPREP_AGREEMENT`) | ICC(2,1) point estimate, primary pair | **≥ 0.75** | **Proposed.** The lower edge of "good" reliability (0.75–0.90; Koo & Li 2016, J Chiropr Med 15:155). It is the same bar as the Ch4 same-day gate ([preregistration_ch4_cat12.md](preregistration_ch4_cat12.md)). No published agreement value for nigral free water between preprocessing pipelines was found |

**Power** (simulation on synthetic pairs, n = 40; "true ICC" is the consistency ICC; a systematic bias of 0.3
between-session SD lowers absolute agreement):

| True ICC | P(pass), no bias | P(pass), bias 0.3 SD |
|---|---|---|
| 0.65 | 0.10 | 0.07 |
| 0.75 | 0.51 | 0.37 |
| 0.80 | 0.78 | 0.64 |
| 0.85 | 0.96 | 0.87 |
| 0.90 | 1.00 | 0.98 |

Forty sessions cannot separate 0.75 from 0.80 well. The CI is reported beside the point estimate for that reason.

**Reported regardless, no gate:**
- **The flow:** candidates and quotas per tier; technical failures by type; QSIPrep QC failures by failed check; QC concordance between the pipelines.
- **The primary pair per tier.**
- **Secondary nigral measures,** each with ICC, CCC, bias and limits:
  - left and right posterior SN free water;
  - anterior SN free water;
  - whole-SN free water;
  - posterior-SN FAt and MDt;
  - the tensor FA and MD of the posterior SN (`dwis_*` against `dwiq_*`, both tensor QCs passed).

**Not part of this test:**
- **JHU tract measures.** `compare` does not compute them; adding them would be a separate change before the freeze.
- **QSIRecon maps.** NODDI ISOVF comes from a different model, and the TORTOISE tensor pools b = 700 and 1000.

## Exclusions, by reason (each counted)

1. **Technical failure:** export, QSIPrep or `compare`. These sessions are replaced in draw order.
2. **Different T1:** `dwiq_fs_image_id` ≠ `fs_image_id`. Excluded, not replaced. Once the change above lands, this should be 0.
3. **QSIPrep free-water QC fail:** `dwiqf_qc_pass` False, by failed check. Excluded, not replaced.

A PIE QC failure cannot occur, because the selection requires `dwisf_qc_pass`.

## What passing would and would not mean

Passing would mean that, on PPMI 2.0 three-shell Siemens data in the drawn tiers, PIE's posterior-SN free water agrees
with the value from community-standard preprocessing at least at the "good" level. The frozen DaT result would then not
be an artefact of PIE-specific preprocessing.

It would not show:
- accuracy (neither pipeline is ground truth; there is no phantom or histology);
- biological validity (that is the DaT test's job);
- agreement on other vendors, single-shell or PPMI-1 data, the rigid tiers, or QSIRecon's models.

**If it fails:**
- **PIE stays the primary pipeline.** The frozen DWI FW result stands as run; it is not re-run or re-analysed with QSIPrep values.
- **The failure is investigated:** by tier; bias against scatter (Bland–Altman); and step by step where it can be isolated (denoising, distortion correction, grid and coregistration).
- **It is recorded** in [validation.md](validation.md) as a DWI limitation.
- **Any re-test is a new pre-registration** on sessions not drawn here.

## Deviations

Any change after the freeze is dated here with its reason, before any `dwiq*` value is seen where possible. A change made
after values were seen is labelled as such, and the frozen analysis is reported beside it. Running eddy on the GPU
(`run --cuda`) instead of the planned CPU eddy is a deviation.

## Compute and order

- **Images.** Both are pinned by digest: QSIPrep (about 10.6 GB compressed) and QSIRecon (6.3 GB; not needed here). They must be pulled first, and a FreeSurfer licence is needed.
- **Cost.** CPU eddy on three-shell data takes hours per session (not yet timed on this machine) for 40 sessions plus replacements.
- **Timing.** Runs after the DWI FW runs (GPU eddy) and the CAT12 run finish, never alongside them.

## Analysis code

```bash
S=scripts/analyse_dwi_qsiprep_agreement.py; T=<derived>/dwi_fw/dwi_features.csv; O=<out>; Q=<derived>/qsiprep
PYTHONPATH=. venv_imaging/bin/python $S select --tables $T --out $O          # -> $O/qsiprep_order.csv (identifiers stay in $O)
M="venv_imaging/bin/python -m pie.imaging.dwi_qsiprep"                       # export, run, compare per imaging_dwi.md, in draw order
$M compare --qsiprep-dir $Q/out --sessions <derived>/sessions.csv --fastsurfer-dir <derived>/fastsurfer --work-dir $Q
PYTHONPATH=. venv_imaging/bin/python $S analyse --tables $T --dwiq $Q/dwiq_features.csv --out $O
```

- `--synthetic` is the dry run. It reads nothing real: every value, tier and identifier is generated.
- `tests/test_analyse_dwi_qsiprep_agreement.py` checks the script on synthetic data:
  - a good agreement passes and a poor one fails;
  - `select` reads no measured value;
  - the draw is seeded and stratified;
  - a short tier is made up from the other;
  - technical failures are replaced in order.
- sha256 at the freeze: `analyse_dwi_qsiprep_agreement.py` _to be recorded_; `dwi_qsiprep.py` _to be recorded_ (after the change above).
- Synthetic dry run only before the freeze. Run once; every number reported.

## Run log

| Date | Step | What happened (aggregates only) |
|---|---|---|
| 2026-10-15 | Draft | Script, test and synthetic dry run (ICC 0.80 on fake data). No PPMI value read |
