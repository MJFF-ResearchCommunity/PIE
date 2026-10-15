# Pre-registration: cortical thickness from FastSurfer surfaces (against PPMI's FreeSurfer 7, same-day repeats, age) and the T1 QC flag against MRIQC

**Status: DRAFT, 15 Oct 2026. Not frozen.** It is to be frozen, with the script's sha256, before `select` runs. At the
draft no PIE surface exists: none of the FastSurfer subjects has `stats/lh.aparc.stats` (checked 15 Oct). The counts
below come from identifiers, dates, cohort labels and the presence of table rows only; no thickness value was read.

## Why

Cortical thickness is listed as "Not yet run" in [validation.md](validation.md). PPMI releases FreeSurfer 7.3.2
Desikan–Killiany thickness for baseline scans (`FS7_APARC_CTH`, 1,716 rows). PIE computes its own thickness for three
reasons:
- one pinned version for every scan;
- any visit, not only baseline;
- cohorts other than PPMI.

PIE's subcortical volumes passed the same three layers (technical agreement and repeatability, biology). Thickness needs
its own evidence.

Part B is descriptive. It relates PIE's T1 outlier flag to PPMI's MRIQC image-quality metrics, as a first look at what
the flag catches before the visual ratings that will measure its miss rate.

## Part A, measure (`fastsurfer.surfaces`, as `run --surfaces`)

- **Segmentation.** FastSurfer at the pinned commit d0d7d39 (2.6.0-dev0). Its corpus-callosum module runs first; recon-surf then runs with `--fsaparc` on FreeSurfer 7.4.1.
- **Values.** From the DK `?h.aparc.stats` (`features.thickness_columns`):
  - `cth_{lh,rh}_<region>`: ThickAvg;
  - `area_{lh,rh}_<region>`: SurfArea;
  - `cth_{lh,rh}_mean`: the `# Measure Cortex, MeanThickness` line.
- **Regions:** the 68 DK regions present in both PIE and PPMI's table (34 per hemisphere).
- **Mean cortical thickness:** (`cth_lh_mean` + `cth_rh_mean`) / 2.
- **Reference:** `FS7_APARC_CTH_08Sep2026.csv` (FreeSurfer 7.3.2 `recon-all`; the 2025 McGill/Nipoppy release, baseline only), joined on PATNO + EVENT_ID. PPMI chose its own T1 for the visit, and the table carries no image ID. Its series may differ from PIE's, so the agreement includes series choice, as the volume comparison did.

## Part A, sample (`select`, reads no thickness)

**Baseline scans.** The earliest QC-passing PIE T1 per participant (`ch4_cat12_validation.baseline`, the Ch4 test's rule)
with an `FS7_APARC_CTH` row on PATNO + EVENT_ID: **1,283 scans** (15 Oct).

| Split | Counts |
|---|---|
| Cohort | PD 639, prodromal 425, HC 165, SWEDD 54 |
| Vendor × field strength | Siemens 3 T 772, GE 3 T 188, Siemens 1.5 T 107, GE 1.5 T 77, Philips 1.5 T 72, Philips 3 T 55 |
| Protocol | PPMI-1 826, PPMI 2.0 457 |

**Same-day pairs.** The 29 MPRAGE + repeat pairs of the T1 reliability analysis (`ch4_cat12_validation.same_day_pairs`,
from `fastsurfer_idps.pre_20261011.csv`, both scans passing `QC["t1"]`): 58 scans (Philips 3 T 50, Siemens 3 T 6,
GE 3 T 2).

**Age.** The 165 HC of the baseline sample (age 31–83, mean 60.1, SD 11.4).

**Processing order** (`sids.csv`):
1. every HC;
2. both scans of each pair;
3. 250 of the other baseline scans, drawn in a seeded random order (pandas `sample`, `random_state` 20261015, first 250).

The test uses only these scans, about 470 in all. The other baseline scans run later for the feature table, outside this test. If compute forces an earlier stop, the analysed set is the finished prefix of this order. That rule is fixed now, without any value; the stop itself is a deviation.

**Compute.** About one CPU-hour per scan (imaging.md), so about 470 CPU-hours: roughly 3 days at 6–8 workers. Runs after the CAT12 run finishes, on CPU only.

## Part A, gates (all three for "Validated")

| # | Layer | Statistic | Threshold | Source |
|---|---|---|---|---|
| 1 | Agreement with PPMI | Median over the 68 regions of Lin's CCC (PIE against FS7), all analysed baseline scans; 2,000-resample scan bootstrap CI | **≥ 0.75** | **Proposed** (lowered from 0.80 on 15 Oct, before any thickness value). PIE's subcortical volumes reached a median CCC of 0.87 against the same release (validation.md). Thickness agrees less well across FreeSurfer versions than volumes do: v7.1 thickness was the least compatible with v6.0/5.3, with cingulate ICC 0.37–0.61 (Haddad et al. 2023, Hum Brain Mapp, PMC9921222). Here the segmentation engine and the FreeSurfer version both differ, and possibly the series, so the bar sits below the volumes' 0.87 |
| 2 | Repeatability | Median over the 68 regions of ICC(2,1) across the same-day pairs (both scans with surfaces); 2,000-resample pair bootstrap CI | **≥ 0.85** | **Proposed** (lowered from 0.90 on 15 Oct, before any thickness value: 0.90 sat at the best published value and would fail a pipeline as reliable as FastSurfer's published one about a third of the time). FastSurfer test-retest thickness ICC: mean 0.92, lowest region 0.78, against FreeSurfer's 0.87 (OASIS-1, 20 participants; Henschel et al. 2020, NeuroImage 219:117012). FreeSurfer DK thickness on visually approved scans: 0.81 (Iscan et al. 2015, Hum Brain Mapp 36:3472). The bar is FreeSurfer's published level. PPMI's repeats are within one session, which favours it |
| 3 | Age (biological) | In HC, Pearson r between age and mean cortical thickness residualised for sex and scanner batch (vendor × field strength, fitted in HC) | **r ≤ −0.30 and 95 % CI upper bound < 0** (Fisher z) | **Proposed.** Thinning with age is consistent across the cortex and across samples (Salat et al. 2004, Cereb Cortex 14:721; Fjell et al. 2009, Cereb Cortex 19:2001, 883 participants in six samples). No primary source giving r for a healthy 30–85-year sample was found, so the bar is the one used for the Ch4 test. PPMI's own FS7 mean thickness in the same HC is reported beside it as the reference r |

**Power:**
- **Gate 3** (n = 165, normal approximation). P(pass) is 0.25, 0.50, 0.76, 0.93 and 1.00 at true r −0.25, −0.30, −0.35, −0.40 and −0.50.
- **Gate 2.** Simulated with 29 pairs and 68 regions whose true ICCs spread with SD 0.04: against the original 0.90 bar, P(pass) was 0.02, 0.63 and 1.00 at a median true ICC of 0.88, 0.90 and 0.92. Against 0.85, the same spread gives approximately 0.02, 0.6 and 1.00 at 0.83, 0.85 and 0.87 (sampling error is slightly larger at lower ICC), so a pipeline at FastSurfer's published 0.92 passes.
- **Gate 1.** With about 415 baseline scans the CI of the median CCC is about ±0.02 wide. The true value decides the gate.

**Reported regardless, no gate:**
- **The flow.**
- **Per region:** r, CCC, and the bias of PIE − FS7 with 95 % limits.
- **Median CCC subsets** (200-resample CIs): by vendor, by field strength, and without `t1_outlier` scans.
- **Hemispheric mean thickness agreement.**
- **Surface area:** median CCC against `FS7_APARC_SA`.
- **Same-day repeats:**
  - per-region ICC;
  - the lowest region and the number below 0.75;
  - the median MDC95 %.
- **Age r:** unadjusted and adjusted, PIE's and FS7's, in the same HC.

**Exclusions (counted):**
- **Surfaces failed:** no `?h.aparc.stats`. Errors go to `<out>/surfaces_errors.csv`, outside the repository.
- **Missing FS7 value:** any region's FS7 value missing.
- **Incomplete pair:** a same-day pair with either scan missing is dropped.
- **Missing age or sex:** HC without them are excluded.

No outlier is removed; the `t1_outlier` exclusion is a sensitivity analysis only.

## Part B: T1 QC flag against MRIQC (descriptive, no gate)

**PIE's rule.** `QC["t1"]` fails only a broken segmentation, a brain under 0.7 L or a visual "fail". `manifest.t1_outliers`
computes the robust z (median and 1.4826 × MAD) of log key volumes within vendor × field strength, over every session.
It flags |z| > 4 to be reviewed first; a flag is not a failure.

**Reference.** PPMI's MRIQC table (`MRIQC_08Sep2026.csv`, baseline; MRIQC per Esteban et al. 2017, PLoS One
12:e0184661). The metrics are oriented so that higher means worse:

| Metric | Direction |
|---|---|
| CJV | Higher is worse |
| CNR | Lower is worse |
| EFC | Higher is worse |
| FBER | Lower is worse |
| SNR (`snr_total`) | Lower is worse |

**Sample.** The earliest PIE T1 per participant matched to MRIQC on PATNO + EVENT_ID, whatever its QC: **1,303 scans**
(15 Oct). Among them:
- 10 are flagged (|z| > 4) and 37 have |z| > 3;
- none fails `QC["t1"]`;
- by vendor × field strength: Siemens 3 T 774, GE 3 T 191, Siemens 1.5 T 107, Philips 1.5 T 81, GE 1.5 T 78, Philips 3 T 55, Toshiba 1.5 T 5.

**Comparisons:**
1. The AUROC of each metric for PIE's flag, with pairs compared within vendor × field strength (`nm_staged.auc_with_ci`, 2,000 within-stratum resamples). With 10 flagged scans the CIs will be wide; that is stated now.
2. Spearman ρ of each metric with `t1_robust_z_max`, over all scans.
3. The flag rate by within-batch quintile of each metric.
4. The flag rate among scans in the worst 5 % (within batch) of any metric, against the rest.

**Expected.** The robust z flags anatomy (atrophy, failed segmentation), while MRIQC measures image quality. Modest
agreement is expected and would not by itself argue for changing PIE's rule. The visual ratings of flagged plus 100
random scans, still open in validation.md, remain the measure of the rule's miss rate. MRIQC's series may differ from
PIE's.

## What passing would and would not mean

Passing all three gates would make PIE's DK thickness **Validated** for PPMI T1s:
- it agrees with PPMI's FreeSurfer 7 at a stated level;
- it repeats within a session;
- it carries the expected age effect.

It would not show:
- accuracy against histology or manual tracing;
- longitudinal sensitivity (the longitudinal pilot covers that);
- validity of the hippocampal subfields, which need surfaces but have their own checks;
- validity on other cohorts.

**If a gate fails:**
- **Status.** Thickness stays Exploratory: exported, flagged and not used by default.
- **Reporting.** The failure is reported per gate and per region.
- **PPMI's FS7 values** remain usable for PPMI baseline analyses.
- **Any re-test is a new pre-registration.**

Part B changes no status.

## Deviations

Any change after the freeze is dated here with its reason. A change made after any thickness value was seen is labelled
as such, and the frozen analysis is reported beside it.

## Analysis code

```bash
S=scripts/analyse_t1_surfaces.py; O=<out>                       # PIE_FREESURFER_HOME (7.4.1) and the licence set
PYTHONPATH=. venv_imaging/bin/python $S select --out $O          # -> $O/sids.csv (identifiers stay in $O)
PYTHONPATH=. venv_imaging/bin/python $S run --out $O --workers 4 --threads 2
PYTHONPATH=. venv_imaging/bin/python $S analyse --out $O         # -> $O/t1_surfaces_result.json
```

- `run` calls the same `fastsurfer.surfaces` as `run --surfaces`, on the selected scans only and in their order. `run --surfaces` would take every finished subject, about 2,900 sessions.
- `--synthetic` is the dry run. It reads nothing real: every value, label and identifier is generated.
- `tests/test_analyse_t1_surfaces.py` checks the script on synthetic data:
  - synthetic thickness passes all three gates;
  - a flat age effect fails only the age gate;
  - the vectorised CCC matches `pie.stats.agreement.ccc`;
  - the MRIQC orientation recovers flagged scans.
- sha256 at the freeze: `analyse_t1_surfaces.py` _to be recorded_. The sample rules are imported unchanged from the frozen `ch4_cat12_validation.py` (sha256 in its pre-registration).
- Synthetic dry run only before the freeze. Run once; every number reported.

## Run log

| Date | Step | What happened (aggregates only) |
|---|---|---|
| 2026-10-15 | Draft | Script, test and synthetic dry run (median CCC 0.90, same-day ICC 0.94, HC age r −0.42, all on fake data). Counts above from identifiers and labels |
