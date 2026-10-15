# Pre-registration: survival-layer positive control (baseline DaT predicts phenoconversion in PPMI prodromal participants)

**Status: FROZEN 2026-10-15**, before any gate statistic was computed on real data (hashes in the run log). No model has been fitted to real data. The counts below come from the
script's loader and flow step only (dates, diagnosis codes, vital status, and whether a DaT value exists). No DaT
value was summarised or tabulated against outcome.

## Why

PIE's interval-censored and competing-risk survival functions (`pie.stats.survive`: `visit_intervals`,
`illness_death`, `cause_specific_cox`, `fine_gray`, `cumulative_incidence`, `interval_censored_regression`) have been
validated on simulated data only (documentation/stats.md). They are meant for the retrospective 2012 prodromal cohort
(up to 15 years of follow-up) and the MRI CSF-SAA consortium. Before then, they have to recover an effect on real
data whose direction is not in doubt.

Striatal DaT loss in prodromal participants predicts conversion to clinical PD. In the PARS hyposmic cohort, a DaT
deficit (≤ 65 % of the age-expected lowest putamen SBR) gave a 4-year relative risk of 17.5 (95 % CI 7.0–43.5):
14 of 21 converted, against 3 of 109 without a deficit (Jennings et al. 2017). PPMI's prodromal cohort was built on
the same design.

**This is a positive control, not a hypothesis test.** Failure means a problem in the pipeline: the outcome coding,
the time origin, the DaT join, or the model. It is investigated before the survival layer is used for anything else.
Failure is not evidence about DaT.

## Disclosure: what has already been seen

- The DaT–conversion association in PPMI and PARS is published (above). Its direction and rough size are known.
- PIE has never fitted a survival model to PPMI conversion data, and has never tabulated DaT against conversion (none known).
- The flow counts below were produced on 15 Oct 2026 by `load()` and `frame()` of the draft script, printing counts only.
- No hazard ratio, concordance or cumulative incidence was computed.

## Outcome, predictor and time

- **Conversion** comes from PPMI's research diagnosis form (`Primary_Research_Diagnosis`, `PRIMDIAG`, one record per visit; `INFODT` month).
  - The event is a first record of 1 (idiopathic PD) or 5 (DLB).
  - Diagnosis is absorbing (`visit_intervals`): `left` is the last record without the event, `right` the first record with it.
  - Month-only dates enter as the window from the first to the last day of the month.
  - Other codes count as not converted (17 none; 23/24 prodromal non-motor/motor; 25 prodromal synucleinopathy; 7, 97 other).
  - `Clinical_Diagnosis` (`NEWDIAG`, `DIAGDT`) is not used: its date is the clinician's estimate and is filled for few records.
- **Death** is the competing event.
  - The date is taken from FOUND's `FODOD` (`FOVITAL` = 1, full date), else the conclusion form's `WDDT` (`WDDEATH` = 1), else `Participant_Status.STATUS_DATE` (`Withdraw Deceased`).
  - Month-only dates are set to mid-month.
  - A death with no date is censored at the last record, and counted.
- **Time 0** is the participant's first dated diagnosis record. Time is in years.
- **Last contact** is the last diagnosis record. A death time that falls before the last record (or before the conversion window) is clipped to it, and counted.
- **Predictor** `dat_low_sd`: baseline DaT per SD lower. It is computed as follows.
  1. Take PPMI's core-lab lowest putamen SBR as a fraction of its age- and sex-expected value (`labels.dat_sbr_scored`: `sbr_pct_expected`, fitted on visually normal controls of the same table).
  2. Use the analysed scan nearest time 0 within ±6 months. Xing is used first; Invicro only where Xing has no scan in the window.
  3. Standardise within core lab (the two scales differ), with the sign flipped so that HR > 1 means lower DaT, higher risk.
- **Covariates:** age at time 0 (per decade, centred at 65) and sex.

## Sample

PPMI participants whose cohort (`Participant_Status.COHORT_DEFINITION`) is Prodromal. This includes the PPMI-1-era
hyposmic and RBD participants (enrolled 2013–2019) and the PPMI 2.0 prodromal cohort (2021 on).

Flow (PPMI tables of 8 Sep 2026; counted 15 Oct 2026):

| Step | n |
|---|---|
| Prodromal cohort in `Participant_Status` (2,836 of them screen failures) | 6,411 |
| With ≥ 1 dated research-diagnosis record | 3,156 |
| Excluded: PD or DLB at the first record (prevalent) | 95 |
| Excluded: alive with a single record (no follow-up) | 877 |
| Excluded: no core-lab DaT within ±6 months of time 0 | 192 |
| Excluded: age or sex missing | 0 |
| **Analysed** | **1,992** |
| … conversions (PD or DLB) | 218 |
| … deaths, of which after conversion | 30, 6 |
| … follow-up, median (IQR; max) | 2.2 y (1.4–3.2; 13.0) |
| … baseline DaT from Xing / Invicro | 1,992 / 0 |
| … conversion later contradicted by a non-PD/DLB record ("reverted") | 51 |
| Enrolled before 2020: n, conversions, deaths, median follow-up | 350, 60, 12, 8.3 y (IQR 7.3–10.0) |
| Enrolled 2020 on: n, conversions, deaths, median follow-up | 1,642, 158, 18, 2.1 y (max 5.0) |

**Power.** These are approximations: Schoenfeld-type, one standardised covariate, before the loss from interval censoring and adjustment.
- 218 events give SE(log HR) ≈ 0.07–0.08. The CI excludes 1 with probability > 0.9 from a per-SD HR of 1.3, and with probability ≈ 1 from 1.5 upward.
- The enrolled-before-2020 subset (60 events) has SE ≈ 0.14. That is power ≈ 0.8 at HR 1.5.
- Given the PARS relative risk of 17.5 for a dichotomised deficit, a per-SD HR well above 1.5 is expected.

## Primary analysis and pass rule

**Model:** `illness_death(d, "left", "right", "time", "died", covariates=["dat_low_sd", "age10", "sex"],
death_covariates=["age10"], hazard="weibull")`.
- It is an interval-censored, three-state Markov model (healthy → PD/DLB, healthy → dead, PD/DLB → dead).
- A participant who dies undiagnosed may have converted unseen before death, and the likelihood integrates over that.
- **Pre-specified fallback:** if the fit reports `converged = False`, it is refitted once without `death_covariates`, and the fallback is reported. With 6 deaths after conversion, the PD → dead intensity is thin.

| Gate | Statistic | Threshold | Source |
|---|---|---|---|
| **Positive control** | HR (healthy → PD/DLB) per SD lower DaT, adjusted for age and sex | Fit converged **and 95 % CI lower bound > 1** | Direction: Jennings et al. 2017 (PARS RR 17.5) |

**Comparator** (reported beside the gate, not gated): `cause_specific_cox` with the same covariates.
- Converters' event time is the first diagnosed record; deaths are a competing cause; everyone else is censored at the last record.
- It is reported with Harrell's C of the model and its Schoenfeld PH test.
- Expected: the same direction, and an HR somewhat smaller than the illness-death HR. Naive Cox is attenuated on visit-dated data (documentation/stats.md: 1.16 against a true 2 in simulation).

**Discrimination** (reported, not gated):
- Harrell's C (lifelines) and Uno's C, both truncated at τ = 5 years (Uno et al. 2011; censoring KM at T−, deaths counted as censored);
- for DaT alone, against time to conversion.

**Reported regardless:**
- the flow table above, recomputed at the run;
- every coefficient of the primary fit (the age and sex HRs; the healthy → dead and PD → dead HRs for age);
- the Weibull baseline intensities;
- the absolute risk of conversion at 2, 4 and 6 years (mean DaT, age 65, female), with deaths accounted for;
- the illness-death HR without age and sex;
- the Fine–Gray subdistribution HR;
- the interval-censored Weibull AFT HR with deaths censored (the "ignore death" analysis);
- the Aalen–Johansen cumulative incidence of conversion at 2, 4 and 6 years by NSD-ISS DaT deficit (≤ 75 % of expected);
- sensitivity analyses (illness-death HR each; skipped if < 10 conversions):
  - MSA (`PRIMDIAG` 11) counted as conversion;
  - reverted conversions counted as not converted;
  - Xing rows only;
  - participants enrolled before 2020, the subset closest to the 2012 cohort (median follow-up 8.3 years).

## What a pass would and would not mean

- **A pass would mean:**
  - PIE's outcome coding from PPMI's tables, its interval construction, its death handling and its illness-death fit recover a known strong prognostic effect on real visit-based data;
  - the survival layer may be used on the 2012 prodromal cohort, with the same coding rules adapted to that cohort's tables.
- **A pass would not:**
  - validate DaT as anything new;
  - calibrate absolute risks;
  - show that weaker predictors (MRI measures, SAA) will be estimated without bias;
  - show that the 2012 cohort's diagnosis records behave like PPMI's.
- **A failure** (non-convergence after the fallback, or a CI that includes 1 or points the wrong way) means a pipeline problem until shown otherwise. The run is reported, the cause found, and any fix is a new, dated pre-registration. The survival layer is not used on the 2012 cohort until a positive control passes.

## Deviations policy

After the freeze, any change is logged in the run log below, with its date, before the script is run.
- A bug fix that changes no outcome code, window, covariate, threshold or statistic is a logged deviation. The synthetic dry run must pass again and the new sha256 is recorded.
- Anything else is a new pre-registration, and the run is labelled post hoc.

The script is run once. Every number it writes is reported.

## Analysis code

`scripts/analyse_survival_positive_control.py` (run with `venv_py311`: lifelines needs pandas < 3):
- `PYTHONPATH=. venv_py311/bin/python scripts/analyse_survival_positive_control.py --ppmi PPMI --out <dir>` is the real run.
- `--synthetic` is the dry run. It reads nothing real: visits, codes, dates, deaths, DaT and identifiers are all generated, with a known HR of 2.5 per SD.
- It imports `pie.imaging.labels` without the imaging package's `__init__` (which needs nibabel, absent from `venv_py311`).
- `tests/test_analyse_survival_positive_control.py` checks the script on synthetic data:
  - recovery of the known HR with deaths before and after onset;
  - each sensitivity moves the conversion count the right way;
  - a null effect is not found;
  - Uno's C.

sha256 at the freeze: in the run log.

## Run log

- 2026-10-15: **frozen**, gates as drafted (approved). sha256: `analyse_survival_positive_control.py` `a13b405bcbbd7ce3644997353cc58950dce34abcc5f1dd6aa6c729826aff3f44`; `survive.py` `446fd965d713b089de10890097c15e7f81fdaeccf8d5056a89df30eddd70dcec`.
- 2026-10-15: drafted.
  - Synthetic dry run: HR 3.00 [2.52, 3.57] against a true 2.5 (n 1,183, 146 conversions, 96 deaths). Across three more seeds the estimates were 2.66, 2.28 and 2.56: sampling variation, not bias.
  - The flow counts above were taken from the real tables (counts only).

## References

- Jennings D, Siderowf A, Stern M, et al. Conversion to Parkinson disease in the PARS hyposmic and dopamine transporter–deficit prodromal cohort. *JAMA Neurol* 2017;74(8):933–940. doi:10.1001/jamaneurol.2017.0985.
- Touraine C, Gerds TA, Joly P. SmoothHazard: an R package for fitting regression models to interval-censored observations of illness-death models. *J Stat Softw* 2017;79(7).
- Leffondré K, Touraine C, Helmer C, Joly P. Interval-censored time-to-event and competing risk with death: is the illness-death model more accurate than the Cox model? *Int J Epidemiol* 2013;42:1177–1186.
- Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med* 2011;30:1105–1117.
- NSD-ISS DaT rule (≤ 0.75 of expected): Dam et al. 2024, as implemented in `labels.dat_labels`.
