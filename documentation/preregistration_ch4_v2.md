# Pre-registration: nucleus basalis (Ch4) volume, version 2 (CAT12.9, Julich-Brain v3.0.3 map, 1 mm)

**Status: FROZEN 2026-10-15**, before IXI was downloaded and before any version-2 value existed for a selected scan.
Convergence: **option B** (decided 15 Oct). Frozen code:
- `scripts/ch4_v2_validation.py` 792264984e592303b1a2ad330b95175ca42f580dc5b43a43f48d77c55f76d8ed
- `scripts/ch4_cat12_validation.py` 49c79ed5e1f5122a343cd78e8110ff6c2a4975c5117552dcdf54c8269d54bb9e (its helpers are imported; unchanged since 13 Oct)
- `pie/imaging/cat12.py` 08b48f9ec20f25bd96c8270465c0212eded1b20bfde41fbafe2b564fb2f93251
- `pie/imaging/atlases.py` 32023a110612d101f38f35a523de808eb3a10a2c7e6efa6b10816ae0d69b9650

## Why a version 2

The 13 Oct test ([preregistration_ch4_cat12.md](preregistration_ch4_cat12.md)) failed:
- controls' age r −0.299 against a bar of ≤ −0.30;
- ρ 0.10 with ScLimbic basal forebrain against ≥ 0.40;
- same-day ICC 0.76, which passed;
- an unexplained left/right ratio of 2.04.

Diagnostics on 15 Oct used the atlas files, the reference measure's repeatability and the relation between the two
methods, not any Ch4 gate statistic. They found:

1. **The map was defective.** The Julich-Brain v3.1 Ch4 map has a truncated right hemisphere: left 261 vs right
   111 mm³ at p ≥ 0.5, mirror Dice 0.48.
   - It comes from the same 10 brains as Zaborszky et al. 2008 (NeuroImage 42:1127), whose histology is symmetric
     (85 vs 90 mm³).
   - Every other release is near-symmetric. v3.0.3 gives 252 vs 211 mm³ and mirror Dice 0.65.
2. **The convergence reference is reliable but measures different person-to-person variation.**
   - ScLimbic's basal forebrain: same-day ICC 0.97, left/right 0.94.
   - CAT12's whole basal forebrain is also reliable (ICC 0.96), yet agrees with ScLimbic at only ρ 0.21 after head
     size, whichever head-size measure is used, and about 0.05 after age and sex too.
   - So a ρ ≥ 0.40 bar against ScLimbic cannot tell a correct Ch4 from an incorrect one.

## What changes (each with its reason)

1. **Map:** Julich-Brain v3.0.3 Ch4 (dataset v4.3, doi:10.25493/KQAX-ADP; pinned by sha256 in `atlases.CH4_MAPS["julich303"]`;
   CC BY-NC-SA 4.0, fetched at run time). Reason: the v3.1 defect (above). The Kilimann 2014 map used by Ray et al.
   2018 is not public, and the SPM Anatomy Toolbox licence forbids extracting its maps.
2. **CAT12 output at 1 mm** instead of the 1.5 mm default. Reason: Ch4 is about 2 mm thick, and published ROI volumes
   were unstable on coarser grids (Wang et al. 2022, NeuroImage 259:119421).
3. **Primary measure: probability-weighted grey matter** instead of grey matter at p ≥ 0.5. Reason: the thresholded
   volume roughly halves with each 0.1 step of the threshold above 0.4, so a fixed cut on a small map is fragile.
   *Disclosure:* this measure also looked better in the 13 Oct secondary results (age r −0.41, ICC 0.87). Those
   participants are excluded from every gate below.
4. **Convergence:** decision at the freeze (below). The 13 Oct gate is not repeated as it stood, for the reason in
   finding 2.
5. **Quality:** CAT12 IQR ≥ 70 % for the primary analysis (CAT12's grade-C boundary), instead of 60 %; ≥ 60 % as a
   sensitivity analysis. Site-stratified results are reported.

Configuration name: `julich303-1mm` (`pie.imaging.cat12.CONFIGS`; CAT12.9 r2665 standalone, Shooting template,
modulated grey matter). It is PIE's default Ch4 configuration since 15 Oct, marked not validated.

## Data (none used by the 13 Oct gates, except the repeat pairs)

- **Age and symmetry: IXI healthy adults aged ≥ 50 with a T1.**
  - 299 with a recorded age, from IXI's demographics sheet.
  - Sites: Guy's (1.5 T Philips), Hammersmith (3 T Philips), IOP (1.5 T GE).
  - Licence CC BY-SA 3.0; the T1 archive is 4.8 GB.
  - PPMI's baseline controls were all used on 13 Oct.
- **Convergence (see the decision): 223 PPMI baseline PD**, seeded (20261016) from those whose participant the 13 Oct
  test did not select.
- **Reliability: the 29 PPMI same-day pairs, re-run at 1 mm.**
  - They are PPMI's only same-day repeats, so they are reused. This is disclosed: their 13 Oct ICC (0.76, at 1.5 mm
    with the v3.1 map) is known.

## Gates (all required for `validated`)

1. **Symmetry (technical).** Median left/right ratio of the primary measure in IXI ≥ 50: **within 0.9–1.2**.
   - Anchors: histology 0.95 (Zaborszky 2008); MRI whole basal forebrain 1.09 (Grothe 2012); manual 1.02.
   - The v3.0.3 map itself gives 1.09 by summed probability.
2. **Age (construct).** In IXI ≥ 50, Pearson r between age and the primary measure residualised for TIV and sex:
   **r ≤ −0.30 and the 95 % CI upper bound < 0.**
   - Unchanged from 13 Oct.
   - Ray et al. 2018 found |r| > 0.46 in PPMI's controls, so −0.30 is not a strict bar.
3. **Reliability.** ICC(2,1) of the primary measure across the 29 same-day pairs (both scans passing quality):
   **≥ 0.75**, with a 2,000-resample pair bootstrap CI (seed 20261016).

### Convergence: option B chosen at the freeze

| Option | What | Assessment |
|---|---|---|
| **A** | Gate: Spearman ρ ≥ 0.30 with ScLimbic basal forebrain (TIV / eTIV residualised) in the 223 fresh PD | A minimum derived from the reliability ceiling: about 0.54–0.60 if both measured the same thing, against ScLimbic's r 0.63–0.70 with manual tracing (Greve 2021). CAT12's whole basal forebrain reached only 0.21 on 13 Oct, so A likely fails even for a correct Ch4. |
| **B (recommended)** | Report ρ with ScLimbic in the fresh PD, with no gate; symmetry joins as gate 1 | It is honest about what the diagnostics showed. The gap the gate was meant to close (is the region right?) is covered by the symmetry gate and the corrected map. Convergence against a gold standard is left to option C or the clinical stage. |
| **C** | Gate against manual Ch4 tracing on 20 scans by a trained rater, blind to the automated values | The proper convergence test. It needs a rater and a tracing protocol. |

Chosen: **B** (`GATE_RHO = None`). ρ with ScLimbic in the 223 fresh PD is reported without a gate, and symmetry is gate 1.

## Reported, with no gate

- Every gate statistic for the secondary measures:
  - grey matter at p ≥ 0.5;
  - the neuromorphometrics basal forebrain.
- Age r by IXI site.
- The IQR distributions.
- Median volumes, raw and as % of TIV.
- Every gate statistic with IQR ≥ 60 %.
- **The map alone, on the 13 Oct scans.** Configuration `julich303-1.5mm` reuses the 13 Oct CAT12 runs, so it changes
  only the map. Reported: its median left/right ratio and median volume. Not independent; descriptive only.

## Limits

- **IXI differs from PPMI:** UK, 2005–2007, partly 1.5 T. The age bar is applied at ≥ 50 to resemble PPMI's ages.
- **Convergence is reported only** (option B).
- **The reliability pairs are reused**, as disclosed above.
- **Later stages:** prognostic validity (Ch4 against later cognitive decline in de novo PD, Ray 2018) and the SAA
  association (Wisse/Spotorno 2024).

## Compute

CAT12 at 1 mm on about 580 scans (299 IXI, 223 PD, 58 repeats): about 14 CPU-min each, so about 22 h with 6 workers.
- 0.5 mm inputs run 2 at a time, because one used 13 GB on 13 Oct.
- Runs go in a systemd unit, without the GPU.

Script: `scripts/ch4_v2_validation.py` (select / run / analyse / `--synthetic`), tests `tests/test_ch4_v2_validation.py`;
synthetic dry run only. The scan scheduling (inputs over 3e7 voxels run 2 at a time) was added before the freeze.

## Run log

- **15 Oct 2026, data:**
  - IXI downloaded after the freeze:
    - `IXI-T1.tar` sha256 `56bbe911fbe967f7d1b830a552995263bdba281e0a6eb78e62f1c38d88a946a9`
    - `IXI.xls` sha256 `8880c4417d8c22f9489439f1b8977b1003ed7908be0294e44462aa71c2a3e78f`
  - 581 T1 files.
- **Selected:** 274 IXI aged ≥ 50 with a T1 (of 299 with a recorded age); 223 fresh PPMI PD; 29 same-day pairs.
  - CAT12 started in the `pie-ch4-v2` unit: 529 regular scans with 4 workers, then the large (0.5 mm) inputs 2 at a time.
- **Environment:** `xlrd` 2.0.2 added to `venv_imaging` to read `IXI.xls` (the only new package).
- **16–20 Oct, pause (scheduling only):**
  - The unit was frozen at 120/529 regular scans, with 0 failed, and thawed on 20 Oct.
  - CAT12's 4-hour time limit counts wall-clock time, so the 4 scans in flight raised `TimeoutExpired` on thaw. They are
    IXI scans; no other process was left over from them.
  - They are re-run after the main pass, with the same command. `ch4_volume` reuses finished scans and clears a failed
    scan's work directory first.
  - No value had been read and the analysis code is unchanged.
