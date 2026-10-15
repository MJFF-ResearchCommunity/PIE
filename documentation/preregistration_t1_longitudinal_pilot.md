# Pre-registration: pilot of the longitudinal FastSurfer stream on 100 PPMI participants (J1)

**Status: FROZEN 2026-10-15**, before any gate statistic was computed on real data (hashes in the run log). No longitudinal-stream output exists. The counts below come from dates, QC
flags and cohort labels only.

## Why

PIE's change measures for the 2012 prodromal cohort and the consortium will come from FastSurfer's longitudinal
stream. In that stream, every time point is segmented from a within-subject template, which lowers the noise of
change (Reuter et al. 2012). The stream is coded (`run --long`, `fastsurfer.longitudinal`, `run.write_long_idps`,
`manifest.t1_visits(stream="long")`) but has never run on real data.

This pilot checks two things before a cohort-wide run, which takes hours per participant on the GPU:
- the stream reports no change where there is none;
- it reports plausible change in healthy controls.

## Disclosure: what has already been seen

- **Cross-sectional same-day repeatability** (29 MPRAGE + repeat pairs, documentation/imaging.md): ICC(2,1) 0.97–1.00. MDC95 by structure:

  | Structure | MDC95 |
  |---|---|
  | Putamen | 5.1–5.5 % |
  | Caudate | 3.7–4.6 % |
  | Thalamus | 2.3–3.9 % |
  | Hippocampus | 2.7–3.2 % |
  | Amygdala | 5.7–6.2 % |
  | Lateral ventricles | 1.7–2.4 % |
  | Brainstem | 4.3 % |

  These pairs are reused here as the zero-change control.
- **Cross-sectional age associations:** thalamus r −0.42, hippocampus −0.36, ventricles +0.40.
- **No annualised change has been computed from PIE's T1 volumes** (none known), cross-sectional or longitudinal.
- **The availability counts below** (15 Oct 2026).

## Sample

**Pilot participants.**
- **Eligible:**
  - ≥ 2 QC-passing (`QC["t1"]`), real-dated T1 visits in `fastsurfer_idps.csv`, first to last ≥ 1.0 year apart;
  - PPMI cohort healthy control, PD or prodromal.
- **Selection** (`select`):
  - a seeded random sample within cohort (pandas `sample`, seed 20261015, participants sorted by ID first): 40 HC, 30 PD, 30 prodromal;
  - an HC shortfall goes to PD and prodromal, PD first;
  - with fewer than 20 eligible HC, `select` refuses, because the plausibility gate could not be judged.
- **Time points:** every usable visit of the participant.
- **Outcome-blind:** selection reads dates, QC flags and cohort, never a volume.

**Zero-change control.**
- The 29 same-day MPRAGE + repeat pairs (27 prodromal, 2 PD; `ch4_cat12_validation.same_day_pairs`, from the pre-11-Oct IDP table where both scans of a day were rows).
- Each pair is its own template with two time points.
- It is kept in a separate subjects directory, so a scan in both sets never shares a template.
- All 58 NIfTIs and cross-sectional outputs are on disk.

**Availability (15 Oct 2026):**

| | HC | PD | Prodromal |
|---|---|---|---|
| PIE: ≥ 2 usable dated T1 visits | **0** | 161 | 194 |
| … first to last ≥ 1 year (eligible) | **0** | 138 | 162 |
| PPMI MRI form (`MRICMPLT` = 1): participants with ≥ 2 MRI visits | 73 | 779 | 1,361 |
| … HC spanning ≥ 1 year | 62 | | |

- **Visits** per PIE participant with ≥ 2: 2 visits for 236 participants, 3 for 103, 4 for 16.
- **Scanner changes:** the scanner changed after baseline for 30 PD and 27 prodromal participants.
- **HC follow-up.** PPMI's controls had follow-up MRI mostly at V04 (year 1), a few at V06 and V10. PPMI's own FS7 table holds only their baseline.

**Precondition:** PIE holds no control with a follow-up T1. The controls' follow-up T1s (about 62 eligible by PPMI's
MRI form) are downloaded and processed cross-sectionally before `select` runs. The choice of whom to download uses
dates and cohort only.

## Method

- **Stream:** `fastsurfer.longitudinal` (`long_fastsurfer.sh`, FastSurfer's pinned build; FreeSurfer 7.4.1 and its licence; GPU).
  - One template per participant or per same-day pair, into `<out>/long` and `<out>/long_sameday`.
  - The run time per template is not yet known. The first template's time is recorded in the run log. 129 templates in all.
- **Volumes:** each time point's `stats/aseg+DKT.stats` (`fastsurfer.STATS_FILE`, what `run.write_long_idps` reads).
  - That the stream writes this file is unverified until the first template. If it writes another stats file, the reader is fixed and logged as a deviation.
  - Cross-sectional comparators come from `fastsurfer_idps.csv`, and from the pre-11-Oct table for the same-day pairs.
- **Structures (15):** left and right putamen, caudate, thalamus, pallidum, hippocampus, amygdala and lateral ventricle, and the brainstem. These are the reliability analysis's set.
- **Annualised change:** per participant, (last − first) / first / years × 100, on left + right, between the first and last usable visits.

## Gates (both for `T1_LONG_PILOT` = pass)

| Gate | Statistic | Threshold | Source |
|---|---|---|---|
| 1. Zero change | Share of same-day pair × structure values (29 × 15 = 435) whose long-stream difference \|a − b\| lies within that structure's cross-sectional MDC95. MDC95 is in mm³: 1.96 × SD of the 29 cross-sectional same-day differences | **≥ 95 %** | PIE's same-day MDC95 (2–6 %, documentation/imaging.md). About 95 % of the cross-sectional differences fall within their own MDC95 by construction, so the long stream must do at least as well |
| 2a. Plausibility, hippocampus | HC median annualised change, left + right | **−1.70 to −0.27 %/yr** | Fraser et al. 2015 meta-analysis, studies with mean age 55 to < 70: 0.98 %/yr loss (95 % CI 0.27–1.70) |
| 2b. Plausibility, putamen | HC median annualised change, left + right | **−1.20 to −0.10 %/yr** | Fjell et al. 2009: −0.43 %/yr (−0.92 % over 2 years; 60–91 y, automated). Raz et al. 2003: −0.73 %/yr (20–77 y, manual, 5 years) |

**Hippocampus window.** The window is Fraser's 55 to < 70 interval; PPMI's controls average 61.5 years. The
neighbouring estimates are inside it:
- all ages 0.85 %/yr (0.63–1.07);
- ≥ 70 years 1.12 (0.86–1.38);
- Fjell 2009 −0.84 %/yr.

Fraser also found lower rates with automated segmentation.

**Putamen window.** It spans the two published estimates, widened by about 0.5 %/yr toward faster loss. The slower
end is capped at −0.10 %/yr so that "no measurable atrophy" fails.

**What the two gates guard against.** A template can over-regularise: it pulls the time points together and could
pass gate 1 by suppressing real change. Gate 2 guards against that. Gate 1 guards against a stream whose change is
mostly noise.

**Precision** (a normal approximation, not a simulation of PIE's data). Most control follow-up is about 1.1 years.
Assumptions:
- the noise of a difference is the cross-sectional MDC95 / 1.96, scaled by 0.6 to 1.0 for the long stream;
- the between-subject SD of true change is 0.6 %/yr.

With 40 controls:

| Structure | SE of the median | P(estimate in window) |
|---|---|---|
| Hippocampus (true −0.9 %/yr) | 0.2–0.3 %/yr | ≥ 0.98 |
| Putamen (true −0.43 to −0.73 %/yr) | 0.3–0.5 %/yr | 0.85–0.91 with long-stream noise 0.6 × cross-sectional; 0.68–0.72 if no lower |

The putamen arm is the weaker one. A putamen-only failure is still a failure. It is reported with its CI, and the
next step is more controls or longer intervals, not a wider window.

**Reported regardless (no gate):**
- the flow: eligible, selected and analysed per cohort, long-stream failures by role;
- the same-day pairs, per structure: ICC(2,1) and MDC95 % for the long and cross-sectional streams, the share within the cross-sectional MDC95, and the number of structures where the long-stream MDC95 is lower;
- within-subject variability: SD of annualised change across pilot participants, long against cross-sectional, per structure (expected lower for the long stream);
- the HC median (bootstrap 95 % CI, 2,000 resamples) for all 7 bilateral structures. Thalamus is set beside Fjell's −0.69 %/yr and lateral ventricles beside +4.40 %/yr;
- the HC subset without a scanner change;
- hippocampus and putamen median change by cohort, and PD − HC (bootstrap CI). Not gated: the pilot is not powered for it.

## Exclusions, by reason (counted at each step)

- Not eligible: fewer than 2 usable visits, a span under 1 year, a masked date, or a cohort other than HC, PD or prodromal.
- Long-stream failure: any time point of a template without all 15 volumes (`long_failures.csv` has the errors). The whole template is excluded, from both streams.

## What a pass would and would not mean

- **A pass would mean:**
  - PIE's longitudinal stream runs end to end on PPMI T1s;
  - it shows no change where none exists, at least as tightly as the cross-sectional stream;
  - it shows control atrophy rates in the published range.

  The stream can then run cohort-wide. Scanner changes are handled with `harmonize.long_combat`, with same-scanner follow-up as a sensitivity analysis.
- **A pass would not mean:**
  - PD or prodromal change is measured accurately;
  - change across a scanner change is unbiased (most pilot pairs share a scanner; reported separately);
  - the stream suits 1.5 T or legacy scans in other cohorts.
- **A failure** is reported with the gate that failed, and the stream is not run cohort-wide until it is understood. A rerun is a new, dated pre-registration.

## Deviations policy

After the freeze, any change is logged in the run log below, with its date, before `analyse` runs.
- A bug fix that changes no selection rule, structure, threshold or statistic is a logged deviation. This includes the stats-file name above. The synthetic dry run must pass again and the new sha256 is recorded.
- Anything else is a new pre-registration.

`analyse` runs once. Every number it writes is reported.

## Analysis code

`scripts/analyse_t1_longitudinal_pilot.py` (`venv_imaging`):
- `select --out <dir>`: the selection;
- `run --out <dir>`: the stream;
- `analyse --out <dir>`: the gates;
- `analyse --synthetic --out <dir>`: the dry run. Every scan, value and label is generated, with known change and long-stream noise of 0.6 × cross-sectional.

`tests/test_analyse_t1_longitudinal_pilot.py` checks the script on synthetic data:
- the gates pass on known change;
- a failed template is excluded and counted;
- a noisy long stream fails gate 1;
- no atrophy fails gate 2.

sha256 at the freeze: in the run log.

## Run log

- 2026-10-15: **frozen**, gates as drafted (approved). sha256: `analyse_t1_longitudinal_pilot.py` `b94b599190fb2041c95262644e9f567d695be8aa66a6ff3e8404496bf00fcaf3`; `fastsurfer.py` `fb5fb32af9006426934a097d414e90d2fa04182edef42878123654fad76ace07`.
- 2026-10-15: drafted.
  - The synthetic dry run passed: 100 % of 435 values within MDC95; HC hippocampus −0.90 and putamen −0.49 %/yr against true −0.9 and −0.5.
  - `select` was run once on the real tables to check the code path, printing counts only. It found 0 eligible HC, 138 PD and 162 prodromal, and its output was deleted. It now refuses with fewer than 20 eligible HC.

## References

- Reuter M, Schmansky NJ, Rosas HD, Fischl B. Within-subject template estimation for unbiased longitudinal image analysis. *NeuroImage* 2012;61:1402–1418.
- Fraser MA, Shaw ME, Cherbuin N. A systematic review and meta-analysis of longitudinal hippocampal atrophy in healthy human ageing. *NeuroImage* 2015;112:364–374.
- Fjell AM, Walhovd KB, Fennema-Notestine C, et al. One-year brain atrophy evident in healthy aging. *J Neurosci* 2009;29:15223–15231. doi:10.1523/JNEUROSCI.3252-09.2009 (Table 2).
- Raz N, Rodrigue KM, Kennedy KM, Head D, Gunning-Dixon F, Acker JD. Differential aging of the human striatum: longitudinal evidence. *AJNR Am J Neuroradiol* 2003;24:1849–1856.
