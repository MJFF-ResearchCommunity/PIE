# Pre-registration: resting-state fMRI pipeline validation on PPMI (throughput, motion, reliability)

**Status: FROZEN 2026-10-15**, before any scan in the sample was run through the pipeline under test and before the
PPMI-1 census or download. Decisions at the freeze, and what is still to be written before the run, are listed in
[At the freeze](#at-the-freeze). The commit that changes this line is the frozen analysis code; hashes are in the
[run log](#run-log).

## Why

PIE's resting-state path has never been validated on real data. The pipeline is:
1. IDA zip → NIfTI → BIDS, with a reverse phase-encoding (PE) fieldmap pair only when its metadata verify;
2. fMRIPrep 25.2.5 (Apptainer);
3. XCP-D 26.2.0 (36P, despiking, censoring, respiratory notch filter at TR ≤ 1.5 s, censor-between 4);
4. feature rows: 4S456 and 4S156 network connectivity, striatal seeds.

Two October audit fixes change every output (`b1a6327`, 11 Oct 2026), so no earlier output stands for the current
pipeline:
- **Distortion correction.** Every PPMI 2.0 scan had run without it. The 10-volume reverse-PE run (TR 1 s) fell
  entirely inside the 10 s dummy discard. A short reference now gives its last volumes.
- **TR.** The repetition time (TR) was never checked against the BOLD header. Multiband data run as TR 2.5 s
  overstated the retained time 2.5× and mis-scaled the ALFF band.

This test is **technical**: does the pipeline turn PPMI scans into feature rows, with residual motion no worse than
the published benchmark and features stable enough to use? It has **no PD-vs-HC gate**. Resting-state group
differences in PPMI did not reproduce (Badea et al. 2017, PLoS One, 10.1371/journal.pone.0188196). A null group
effect would neither validate nor fail the pipeline. Known biology is reported, not gated.

## Disclosure: what has already been seen

- **The PPMI 2.0 pool was used by Study 1.** The held PPMI 2.0 zips (`First_Study_fMRI_1*`, `First_Study_fMRI_2*`)
  are Study 1's fMRI download. In September 2026 Study 1 processed them with its own pipeline, which predates both
  fixes above:
  - 641 scans were converted.
  - **Throughput:** 336 scans (52 %) were analysed. Of the rest, 281 failed Study 1's retained-time rule (median
    109 s retained), 14 had nuisance-model errors and 10 were left out for other reasons. By TR: 134 of 300 at
    ≈ 1 s (45 %) and 202 of 341 at ≈ 2.5 s (59 %) were analysed.
    - Study 1 censored on FD and DVARS and had no respiratory filter at TR 1 s (the 11 Oct audit lists that filter as
      missing). XCP-D censors on FD only, after the notch filter at TR ≤ 1.5 s.
    - The 80 % throughput gate was written with these rates known.
  - **QC-FC** (15 Sep 2026; 336 scans pooled across TR and vendor; 4,371 edges from 94 parcels): none
    significant after FDR. 189 were nominally significant (4.3 %, about the 5 % expected by chance). Median |partial r|
    was 0.037, the value expected with no motion effect at that N.
  - **Split-half reliability** (312 scans; contiguous halves; 21 Schaefer network-pair features, Limbic excluded):
    median absolute-agreement ICC 0.583 (features 0.51–0.62). By vendor: Siemens 0.595, GE 0.509, Philips 0.459.
  - Study 1 also related these scans' connectivity to SAA status (primary, GSR and DaTscan-matched experiments).
- **Thresholds were set after those results.** The ICC < 0.4 export rule (`fmriprep.md` §5e, 11 Oct) and the 0.40
  reliability gate below both came after Study 1's ICC. The 0.40 cut is Cicchetti's (1994) conventional
  bound for "poor" agreement, not a value fitted to Study 1.
- **Overlap with this sample.** The draw run on 15 Oct under the rule below (seeded, so the run reproduces it):
  | Stratum | Drawn | In Study 1's fMRI table | In Study 1's QC-FC / half-ICC set |
  |---|---|---|---|
  | P2_TR1 | 100 | 88 | 43 |
  | P2_TR2.5 | 100 | 72 | 47 |

  The PPMI 2.0 results are therefore **not** on unseen scans. Two alternatives were considered and rejected:
  - Drawing outside Study 1's analysed set (189 / 246 eligible) would over-sample scans that failed Study 1's QC,
    because inclusion there was a QC outcome.
  - Drawing outside Study 1's whole table leaves only 45 eligible P2_TR1 participants.
- **PPMI-1 scans have not been seen.** No PPMI-1 rs-fMRI has been downloaded or processed, by PIE or by Study 1.
- **No posterior putamen–SomMot vs DaT SBR association** is known to have been computed on any of these scans.
  Checked 15 Oct: no Study 1 document (`study1_virtual_biomarkers/*.md` and its fMRI folder) mentions SomMot.
- **XCP-D has not run on PPMI data.** It has run only on a synthetic output tree.

## Pipeline under test (code at the freeze commit)

- **Conversion and BIDS export:**
  - `fmri.convert_archive_series`; temporal assembly where needed.
  - `fmri_bids.export_subject` (`discard_seconds` 10). A fieldmap pair is written only when `phase_encoding_pair`
    returns no reasons and the echo times match. A short reverse-PE reference keeps its last volumes.
- **fMRIPrep 25.2.5**, Apptainer, `--fs-no-reconall`, output spaces `MNI152NLin2009cAsym:res-2` and `T1w`.
- **XCP-D 26.2.0** with the settings in `fmriprep.md` §5c:
  - `linc` mode, `36P`, despiking;
  - motion filter: notch 12–28 bpm, order 4 at TR ≤ 1.5 s, none above;
  - censoring: FD > 0.3 mm (head radius 50 mm), `--censor-between 4`, censored outputs;
  - `--min-time 300`;
  - band-pass 0.01–0.08 Hz;
  - atlases `4S456Parcels` and `4S156Parcels`, `--min-coverage 0.5`.

  There is one XCP-D configuration per exact header TR (its validator allows 1 %), so 1.0, 1.03, 1.05 and 1.103 s
  run separately and are pooled in their stratum.
- **Features:** `xcpd.xcpd_features` (both atlases) and `xcpd.xcpd_striatal_features` (CIT168 seeds, putamen split at
  y = 0).

## Protocol strata

| Stratum | Era (scan date) | Header TR | Naming | Source |
|---|---|---|---|---|
| P1_TR2.4 | PPMI-1 (before 1 Jul 2020) | 2.4 s | — | LONI download, not yet made |
| P1_TR2.5 | PPMI-1 | 2.5 s | — | LONI download, not yet made |
| P2_TR1 | PPMI 2.0 (from 1 Jul 2020) | ≈ 1 s (1.0–1.103), multiband | rsfMRI AP / PA | held zips |
| P2_TR2.5 | PPMI 2.0 | 2.5 s (2.50–2.54), single-band | rsfMRI RL / LR, ep2d, BOLD_RS | held zips. Added at the freeze |

**Why a fourth stratum (kept at the freeze).** The plan assumed PPMI 2.0 means TR 1 s. It does not:
- LONI's search export for the first held zip (408 participants, 9 Sep 2026) gives TR ≈ 1 s only for the AP/PA series
  (Siemens 1.0 and 1.03 s, GE 1.0 s, Philips 1.103 s; 2 Siemens series at 2.5 s).
- The RL/LR series run at TR 2.5 s on all three vendors.
- Study 1's header TRs for its 641 converted scans agree: 300 at ≈ 1 s (1.0 s 244, 1.03 s 45, 1.05 s 1, 1.1 s 10) and
  341 at ≈ 2.5 s (2.5 s 335, 2.52 s 1, 2.54 s 5).

The TR 2.5 s PPMI 2.0 scans are the earliest rs-fMRI session for 418 of the 715 held participants. Without their own
stratum they would be exported but never validated. They run under the TR 2.5 s XCP-D settings (no notch filter).
Unlike PPMI-1, they carry a reverse-PE reference.

The era boundary comes from PPMI's `MRI_Acquisition_Metadata`. It records rs-fMRI under protocol 001 up to 2020 Q1
and under protocol 002 from 2020 Q4, with none in between.

## Availability (aggregate counts; no identifiers)

- **Held PPMI 2.0 zips** (series names only, 15 Oct):
  - 720 participants, 715 with an rs-fMRI series, all scanned 2021–2024. The collection CSV of the first zip lists
    all 408 of its participants as prodromal.
  - Earliest rs-fMRI session: RL/LR 418 (414 with two or more rs-fMRI series that day, the name-level sign of a
    reverse-PE reference); AP/PA 297 (296).
  - 20 participants have both variants; 71 have two or more rs-fMRI sessions.
  - **Eligible under the rule below:** P2_TR1 315, P2_TR2.5 420.
- **PPMI-1** (PPMI's `MRI_Acquisition_Metadata`, `MRI_SEQ_RS = Yes`): 214 participants (118 PD, 20 HC, 62 prodromal,
  14 SWEDD), 418 visits, 2012 to 2020 Q1.
  - Counts per TR and whether a reverse-PE reference was acquired are **pending** a LONI fMRI census.
  - 100 per PPMI-1 stratum may not be reachable. A stratum with fewer eligible participants is taken whole.
- **PPMI 2.0 overall** (same table, for scale): 2,090 participants (632 PD, 110 HC, 1,348 prodromal). Only prodromal
  scans are held.

## Sample rule (`scripts/analyse_fmri_validation.py select`)

1. **Eligibility** comes from series names and census metadata only, never from images, QC or outcomes. A session is
   eligible if a series description matches `rs.?fmri|resting|bold`.
2. **Stratum:**
   - Era is set by scan date.
   - TR comes from the LONI census. The held zips carry no TR, so there AP/PA naming means P2_TR1 and anything else
     P2_TR2.5.
   - The header TR at conversion decides. A scan whose header TR is outside its stratum is an acquisition exclusion
     (`header_tr_outside_stratum`), counted and not moved.
3. **One scan per participant per stratum:** the earliest eligible session, with all of that session's rs-fMRI series.
4. **Up to 100 per stratum**, spread over sites:
   - Allocation is proportional over naming signatures: the session's set of normalised rs-fMRI descriptions, a
     stand-in for site protocol, since site and scanner are not in the names. Remainders go to the largest fractions.
   - Each signature is then drawn at random (pandas `sample`, seed **20261015**).
   - A stratum with 100 or fewer eligible participants is taken whole.
5. **No replacement.** Every selected scan stays in the flow table, whatever happens to it.
6. **A participant may be drawn in two strata** (different visits). Each stratum is analysed on its own. The pooled
   biology analysis keeps that participant's earlier scan.

**Freezing the list.** `fmri_selection.csv` is written outside the repository. Its path and sha256 go in the run log
at the freeze; only counts go in this document.
- PPMI 2.0 is drawn from the held zips listed above.
- PPMI-1 is drawn from a fresh LONI fMRI census (Modality fMRI, with the Imaging Protocol column for TR). Exactly the
  drawn scans are then downloaded.

**PPMI 2.0 draw (15 Oct, held zips, this rule):**

| Stratum | Eligible | Naming signatures | Drawn |
|---|---|---|---|
| P2_TR1 | 315 | 6 | 100 |
| P2_TR2.5 | 420 | 16 | 100 |

## Exclusions, by reason

Every selected scan ends with one of two outcomes: a feature row, or exactly one exclusion reason (the first one
met).

| Reason | Category | Rule |
|---|---|---|
| `no_verified_reverse_pe_pair` | acquisition | No `fmap/` pair from `export_subject`. Applies only to protocols that acquire a reverse-PE reference: both PPMI 2.0 strata, and PPMI-1 if the census shows one. A protocol that acquires none runs without distortion correction, which becomes part of its context of use. |
| `too_few_volumes` | acquisition | `classify_run` does not return `rest_candidate`: fewer than 100 volumes or under 300 s. |
| `header_tr_outside_stratum` | acquisition | Header TR outside the stratum's range. |
| `no_t1` | acquisition | No usable 3D T1 at the session. |
| `conversion_error` | pipeline | Conversion or temporal assembly raised. |
| `fmriprep_error` | pipeline | fMRIPrep exited non-zero or wrote no report. |
| `xcpd_error` | pipeline | XCP-D exited non-zero. |
| `xcpd_min_time` | motion | Under 300 s retained after censoring: XCP-D wrote no BOLD. |
| `visual_review_fail` | pipeline | fMRIPrep/XCP-D report review failed: distortion correction, registration or coverage. The reviewer sees only the reports. |
| `coverage` | pipeline | More than 10 % of 4S456 parcels (over 45) blanked by XCP-D for coverage. |

**Acquisition exclusions** are counted and reported by reason, and leave the throughput denominator.

**Motion losses (`xcpd_min_time`) are reported separately** and also leave the throughput denominator: their count
and their share of acquisition-eligible scans, per stratum, beside throughput. Decided at the freeze (15 Oct): too
little retained time reflects how much participants moved more than what the pipeline did. Study 1 lost 44 % of
scans to its retained-time rule under stricter censoring. Whether the notch filter restores retained time at TR 1 s
is reported by the no-notch variant below.

**Pipeline failures count against throughput.**

A feature row without age or sex leaves QC-FC only, and is counted.

## Gates, per protocol

A protocol can pass while another fails. All values below are fixed at the freeze.

| Gate | Statistic | Pass | Source of the threshold | Status |
|---|---|---|---|---|
| **1. Throughput** | feature rows / (selected − acquisition exclusions − motion losses) | **≥ 0.80** | PIE plan (`fmriprep.md` §5e; roadmap WS-D) | fixed |
| **2a. Motion: QC-FC edges** | % of 4S456 edges with FDR-significant QC-FC (Benjamini–Hochberg, q 0.05) | **< 7 %** | Ciric et al. 2017, Results (verified, below) | fixed |
| **2b. Motion: median QC-FC** | median \|QC-FC\| | **≤ the 95th percentile of its permutation null** | Permutation null (Ciric's value is figure-only; below) | fixed |
| **2c. Motion: distance dependence** | \|Pearson r\| of QC-FC with inter-parcel distance | **≤ the 95th percentile of its permutation null's \|r\|** | Permutation null (Ciric's value is figure-only; below) | fixed |
| **3. Reliability** (PPMI 2.0 strata only) | median split-half ICC(3,1) over the 4S456 network-pair features | **≥ 0.40** | Cicchetti 1994 "poor" bound; see disclosure | fixed |

**QC-FC** follows `fmri_connectivity.qc_fc`:
- Across a protocol's feature rows, the partial correlation of each 4S456 edge (XCP-D's Pearson matrix, Fisher z)
  with mean FD (XCP-D `mean_fd`), on age and sex.
- Edges with a blanked parcel in any scan are left out and counted.
- Distances are Euclidean, between parcel centroids of XCP-D's 4S456 atlas image (`parcel_centroids`). Pearson is used
  as in Ciric 2017; `qc_fc`'s Spearman is reported beside it.

**The benchmark, Ciric et al. 2017**, NeuroImage 154:174–187, doi:10.1016/j.neuroimage.2017.03.020 (author
manuscript PMC5483393, read 15 Oct 2026):
- **Study:** 393 young adults, 14 confound models. QC-FC is a partial correlation on age and sex, with motion as mean
  relative RMS displacement, in 264-node (Power) and 333-node (Gordon) networks.
- **Results, verbatim:** "Beyond this base 36-parameter model, all censoring techniques provided provided some
  additional benefit, reducing the number of edges that were significantly related to motion to less than 7%."
  Also: "The top-performing method, GSR+spike regression, yielded < 1% of edges".
- **Figure-only values.** The median |QC-FC| and the distance dependence of each model are given only in Figs. 3 and
  4, and could not be retrieved. Gates 2b and 2c therefore do not use Ciric's values (decided at the freeze, 15 Oct).
  Ciric's 36P+scrub model (FD > 0.2 mm, retained stretches under 5 frames censored, frames excised) remains the
  closest published comparator to XCP-D's censoring (FD > 0.3 mm, `--censor-between 4`, censored output); gate 2a
  uses its verified Results value.

**Why a permutation null.** At N ≈ 100 the median |QC-FC| of a motion-free pipeline is not a fixed number:
- It is centred near 0.068 (Ciric's 393 scans: 0.034).
- It moves by several hundredths with the dependence between edges. In motion-free synthetic data at N ≈ 55 it lay
  between −0.016 and +0.028 of the analytic null.
- A fixed threshold would fail clean pipelines by chance.

The null is built from mean FD permuted across the protocol's scans: 1,000 permutations, seed 20261015, covariates
staying with their scans. A protocol fails 2b or 2c when its statistic exceeds that null's 95th percentile: a
motion–connectivity association detectable at this N (one-sided α 0.05). A clean pipeline fails each by chance about
5 % of the time, so up to about 10 % of clean protocols fail 2b or 2c. At N ≈ 100 only a moderate artefact is
detectable, and the protocols' N is reported beside each statistic.

**Gate 2a is lenient at N = 100**, because fewer edges reach FDR significance than at N = 393 for the same artefact.
Gates 2b and 2c carry most of the weight. In synthetic data, a moderate short-range artefact passed 2a with 0 % of
edges and failed 2c.

**Gate 3 and the export rule.**
- **Split:** each scan's retained (censored) frames, from XCP-D's `stat-mean_timeseries.tsv`, into first and second
  halves in time order.
- **Per half:** Pearson, Fisher z, and network-pair means with XCP-D's grouping (`network_label`, else `atlas_name`),
  for 4S456 and 4S156.
- **Striatal seed features** are recomputed on each half of the denoised BOLD by the run step.
- **ICC(3,1):** two-way mixed, consistency, single measure (Shrout & Fleiss 1979), per feature across scans.

The gate is the median over the 4S456 network-pair features. Network level is the level the literature supports: edge
ICC averages 0.29 (95 % CI 0.23–0.36) across test-retest studies (Noble et al. 2019, NeuroImage,
10.1016/j.neuroimage.2019.116157).

**Feature families, per protocol** (an export rule, not a module gate, applied to every protocol including PPMI-1):
- `fc_4S456`, `fc_4S156` and `striatal`: a family whose median split-half ICC(3,1) is below 0.40 is not exported for
  that protocol.
- Families with no split-half estimate here (ALFF, ReHo, basal-ganglia-network dual regression) are not covered by
  `FMRI_VALIDATED` and stay Exploratory.
- If the run step cannot produce striatal halves, the striatal family is not exported.

**Status:** `FMRI_VALIDATED[protocol]` is True when every gate of that protocol passes. Gate 3 applies to the PPMI 2.0
strata only; it is computed for PPMI-1 and reported.

### What `FMRI_VALIDATED[protocol]` means

**It means**, for that protocol's PPMI scans processed by the pinned pipeline above:
- at least 80 % of acquisition-eligible scans not lost to motion become a feature row (motion losses reported beside
  it);
- fewer than 7 % of edges carry an FDR-significant motion association (the published 36P-with-censoring benchmark),
  and no median or distance-dependent motion association is detectable beyond a permutation null at this N;
- (PPMI 2.0) network-level connectivity is at least fair within a session;
- the exported feature families are those that met 0.40.

**It does not mean:**
- between-session or longitudinal reliability (adjacent halves overstate it);
- sensitivity to PD, prodromal status, SAA or any clinical outcome;
- that edge-level features, ALFF, ReHo or the basal-ganglia network are usable;
- validity outside PPMI's protocols and scanners;
- that an individual scan is good: `scientific_qc_pass` stays False and reports are still reviewed.

PPMI-1 without a reverse-PE reference would be validated *without* distortion correction.

**In the manifest:** a per-protocol mapping `FMRI_VALIDATED = {stratum: bool}` in `pie/imaging/manifest.py`, exported
per row as `fmri_validated` from the row's protocol. A failed protocol's features stay exported, flagged Exploratory.

## Reported whatever the result (no gate)

- **Flow table per stratum:** selected, each exclusion reason, feature rows, motion losses (count and share of
  eligible), throughput (as gated, of eligible including motion losses, and of selected).
  Also header vendor and model, retained seconds (median and IQR), and mean FD.
- **Biology:** per hemisphere, the partial Pearson r (95 % CI, Fisher) of posterior putamen–SomMot coupling
  (`fmri_xcpd_striatal_putamen_posterior_{l,r}__SomMot`) with the same side's post-commissural putamen SBR.
  - SBR: Xing core lab `POSCOMMISSURAL_PUTAMEN_{L,R}_REF_CWM`, the DaTscan nearest the fMRI session within 18 months.
    Xing only, never pooled with Invicro; scans without one are counted.
  - Partial on age, sex, protocol and mean FD.
  - Across all participants (one scan each), and within PD when n ≥ 20.
  - Expected positive: lower SBR goes with lower coupling. The held PPMI 2.0 scans are prodromal, so PD comes from
    PPMI-1.
- **Notch filter (P2_TR1):** the same scans rerun through XCP-D with the motion filter off. Reported: throughput,
  retained seconds and the QC-FC statistics. Expected: the filter restores retained time without raising QC-FC.
- **QC-FC detail:** with mean relative RMS (Ciric's motion measure) in place of FD; nominal percentages; Spearman
  distance dependence.
- **Reliability detail:** every family's median ICC for every stratum, median edge ICC, and identification accuracy
  (Finn et al. 2015; half 1 against half 2, both directions).

## Analysis code

**Selection:** `scripts/analyse_fmri_validation.py select --zips <held PPMI 2.0 zips> --census <LONI fMRI
census.csv> --out <dir outside the repo>`. It reads series names and census TRs only, and prints counts only.

**Run step:** written after the freeze and before any scan runs (see [At the freeze](#at-the-freeze)), as one systemd unit outside the harness (fMRIPrep, then XCP-D, per participant;
large BOLD derivatives deleted after features). It writes one row per selected scan:
- `scan` (BIDS label), `patno`, `session_date`, `protocol`, `exclusion` (empty or one reason), `xcpd_dir`;
- the `xcpd_features` and `xcpd_striatal_features` columns;
- `<striatal column>__h1` / `__h2` halves.

The no-notch variant has rows with protocol `P2_TR1:no_notch`.

**Analysis:** `scripts/analyse_fmri_validation.py analyse --scans <scans.csv> --ppmi PPMI --out <dir>`.
- It recomputes edges from XCP-D's parcel series and stops if they differ from XCP-D's own matrix on any scan
  (1e-6).
- `--synthetic` is the dry run: every value, label and identifier is generated, and nothing real is read.
- `tests/test_analyse_fmri_validation.py` checks on synthetic data that:
  - the gates flip: a strong motion artefact fails 2a and 2b, and a short-range one fails 2c; clean data passes;
  - halves without stable individual connectivity fail gate 3;
  - acquisition exclusions and motion losses leave the throughput denominator while pipeline failures count;
  - ICC(3,1) ignores a constant offset;
  - the draw is one earliest scan per participant and stratum, and is reproducible.

The sha256 of the script (and of `fmri_connectivity.py`, `xcpd.py`, `fmri_bids.py`) at the freeze is in the run log.
Synthetic dry run only before the freeze. Run once; every number reported.

## Deviations

A change after the freeze is recorded here, dated, before the statistic it affects is computed. Anything decided
after seeing a result is labelled post hoc.

Gates are not rerun with new settings after a failure. A bug found during the run is fixed and documented, with the
scans it affected. Throughput is gated on each scan's first-run outcome, with the post-fix rate reported beside it.

## At the freeze

**Decided 15 Oct 2026** (the user approved the recommendations as a whole):
- the P2_TR2.5 stratum is kept;
- gate 3 at 0.40, and the coverage rule (10 % of parcels);
- `xcpd_min_time` is a motion loss, reported separately and outside the throughput denominator;
- gates 2b and 2c are judged against the permutation null only (Ciric's figure values could not be retrieved);
- gate 2a stays at < 7 % (Ciric 2017 Results, verified);
- PPMI 2.0 is drawn from all held scans despite the Study 1 overlap (disclosed above);
- the biology disclosure was checked (no Study 1 document mentions SomMot).

**Still to do before any scan runs** (execution, not analysis; none changes a rule, threshold or statistic above):
1. **PPMI-1 census.** A LONI fMRI search export. The frozen `select` rule then gives the PPMI-1 eligible counts, the
   reverse-PE question and the PPMI-1 draw, which is downloaded. Counts go in the run log.
2. **Run-step code:**
   - the `scans.csv` writer;
   - striatal features on each half of the denoised BOLD;
   - an `XCPDConfig` switch that turns the motion filter off for the no-notch variant (today the flag is reserved and
     set from the TR).

   Its sha256, and those of `fmri_bids.py`, `xcpd.py` and `fmri_connectivity.py` at run start, go in the run log
   before the first scan runs. Any other change to those modules after the freeze is a deviation.
3. **The selection file's sha256** (`fmri_selection.csv`, outside the repository) goes in the run log when it is
   written.

## Run log

| Date | Event |
|---|---|
| 2026-10-15 | Draft written. Availability counted from zip member names, LONI metadata, PPMI's MRI metadata and Study 1's summary tables (aggregates only). Draft draw run on the held zips (counts above; list kept outside the repository). Synthetic dry run and tests pass. |
| 2026-10-15 | **Frozen** with the decisions in [At the freeze](#at-the-freeze). Script changed before the freeze: motion losses reported apart; gates 2b and 2c against the permutation null only; census rows restricted to PPMI-1, so the census cannot change the PPMI 2.0 draw (as the sample rule says). Tests and synthetic dry run pass. sha256: `analyse_fmri_validation.py` `a9f8b37ef340be003910c58f98453ca5edda68d10573c8d093881d69a0a6dfea`; `fmri_connectivity.py` `a411f5e39af098f53b6c5b53ef3a333b5013b18ee424592be34ececfee1c3cc1`; `xcpd.py` `619732baa9f1646ebb1df5fd45a28fb6e5837baa1f686d95b018afde0212c434`; `fmri_bids.py` `a4888b0de17db84fd8dd5f98703ed460d6a22a88090565bc259d454cbfabc45d`. |
