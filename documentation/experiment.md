# Experiment layer (`pie.experiment`)

`pie.imaging` produces measurements. `pie.experiment` is what turns measurements into a
result that survives review: who is in the cohort, how a model is selected without seeing
its own test set, and how the result is tied to the code and inputs that produced it.

```
PPMI tables + IDPs ──cohort──► analysis frame ──prediction──► out-of-fold predictions
                                                          └──provenance──► manifest.json
```

Nothing here reads PPMI files or images; pass it a `DataFrame` with one row per
participant, a `PATNO` column and an outcome column.

| Module | What it does |
|---|---|
| `cohort.py` | Carrier status, per-module sex decoding, visit concurrency, complete-case masks, conflict-safe per-participant values. |
| `prediction.py` | Nested candidate-grid selection with partition-owned preprocessing; participant-clustered paired bootstrap. |
| `provenance.py` | Chunked SHA-256, numpy-safe JSON, environment capture, `manifest.json` writing and re-verification. |

## cohort

The PPMI-specific traps, each of which is a whole-cohort error when it goes wrong:

```python
from pie.experiment import cohort

frame["carrier"] = cohort.carrier_status(frame)          # NaN, not 0, for an untested gene
frame["sex_male"] = cohort.decode_sex(demographics)      # SEX is coded per data module
paired = frame[cohort.concurrent_visit(frame, "T1_EVENT_ID", "SAA_EVENT_ID")]
sex = cohort.unique_per_participant(visits, "SEX")        # drops participants whose repeats disagree
```

- `carrier_status` returns 0 only when *every* listed gene is an explicit negative. A
  participant untested for one gene is not a genetic control; coding missing genotype as
  non-carrier inflates every carrier-versus-control contrast.
- `decode_sex` raises on a demographics module whose coding has not been verified rather
  than applying another module's mapping. `SCREEN` codes male as 1 and `PARTICIPANT_PROFILE`
  codes male as 1, female as 2 — decoding one as the other flips the whole cohort's sex, and
  every downstream model quietly absorbs it.
- `concurrent_visit` compares visit *codes*, treating `SC` and `BL` as one occasion. Assay
  dates are not sample dates: a `RUNDATE` is when the laboratory thawed the sample, so date
  arithmetic on it fabricates concurrency.
- `unique_per_participant` drops participants whose repeated records disagree instead of
  taking whichever row came first.

## prediction

```python
from pie.experiment import prediction as pred

families = {"t1": t1_columns, "nm": nm_columns}
predictions, audit = pred.nested_fold(
    frame, train_rows, test_rows, families,
    batch_cols=["scanner_batch"], icv="MaskVol", adjust_baseline=False,
    seed=20260909, outcome="saa_prodromal")
```

`predictions` holds one probability vector per family plus `baseline` (covariates only)
and `selected` (the winner of the whole grid) — which is exactly what an
increment-over-covariates comparison needs. `audit` records the winning candidate and its
inner score for each, every candidate's inner-fold AUC and log loss, the participants in
each inner split, and any candidate that failed to fit.

What the module guarantees:

- **No preprocessing object is fitted before splitting.** Imputation medians, scaling,
  feature eligibility, nuisance residualisation and the PCA basis are all learned on one
  partition and applied to the others. A feature present only in the partition being
  predicted is dropped, not imputed from it.
- **The outer test outcome is never read.** `nested_fold` returns predictions; the caller
  joins labels afterwards. The test suite asserts this by flipping every test label and
  checking the predictions are byte-identical.
- **Participants cannot straddle a split.** A `PATNO` on both sides raises.
- **Selection is deterministic**: largest mean inner AUC, then log loss, then declared grid
  order. The same data give the same choice.
- **Non-convergence is a failure, not a model.** A `ConvergenceWarning` disqualifies the
  candidate rather than contributing an unconverged fit to the comparison.

The grid is small and declared up front: covariates alone, or covariates plus one feature
block, each optionally residualised and (for blocks of 64+ features) PCA-compressed, under
logistic regression or gradient boosting. `candidate_grid` is a module-level function, so a
study can narrow it for a cheap sensitivity run.

Measures of the nigra itself are not proportional to head size, so columns prefixed
`nm_`, `dwi_`, `new_nm_` or `new_dti_` are residualised without the intracranial-volume
covariate that cortical and subcortical morphometry gets (`NON_ICV_PREFIXES`).

### Paired increments

```python
table = pred.paired_metrics(out_of_fold)   # columns: PATNO, repeat, y, p_<model>...
```

Predictions are averaged within participant, then resampled by participant and stratified
on outcome, so a participant appearing in several repeats moves in and out of a resample
together and the interval reflects the number of people rather than the number of rows.
The interval is **conditional on the fitted models** — it covers sampling variation in the
evaluated participants, not the variability of the model search or of external validation.
Every row carries that caveat in its `uncertainty` column so the number cannot be quoted
without it.

## provenance

```python
from pie.experiment import provenance as prov

started = time.time()
...
prov.write_manifest(out_dir, inputs=[cohort_csv], code=Path(__file__).parent,
                    started=started, seed=20260913, selections=winners)

prov.verify_manifest(out_dir)    # [] means nothing has changed since the run
```

`write_manifest` records the UTC time, the SHA-256 of every input, of every source file
defining the analysis and of every output, plus the interpreter, platform and package
versions. `verify_manifest` re-hashes and returns what no longer matches — the check for
whether a result directory still corresponds to the code that produced it.

`save_json` rejects NaN rather than emitting the non-standard `NaN` token, on the view that
a NaN reaching a results file means an unfinished computation. `jsonable` converts numpy
scalars, arrays and `Path`s so that a manifest assembled from pandas output serialises at all.

## Provenance of this layer

These modules were extracted from the PPMI prodromal-synucleinopathy study (Study 1,
virtual biomarkers) after it closed, and the numerics are unchanged from the engine that
produced its published results. `tests/test_experiment_prediction.py` carries that study's
own leakage tests.

## Tests

```bash
venv_imaging/bin/python -m pytest tests/test_experiment_cohort.py \
    tests/test_experiment_prediction.py tests/test_experiment_provenance.py -q
```
