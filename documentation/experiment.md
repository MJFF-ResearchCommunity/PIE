# Experiment layer (`pie.experiment`)

`pie.imaging` produces measurements. `pie.experiment` covers what a result needs to survive
review: who is in the cohort, how a model is chosen without seeing its own test set, and
how the result is tied to the code and inputs that produced it.

```
PPMI tables + IDPs ──cohort──► analysis frame ──prediction──► out-of-fold predictions
                                                          └──provenance──► manifest.json
```

Nothing here reads PPMI files or images. Pass it a `DataFrame` with one row per
participant, a `PATNO` column and an outcome column. For classical tests on the same frame
(t-tests, mixed models, survival, multiple-testing correction), see [`stats.md`](stats.md).

| Module | What it does | Exported |
|---|---|---|
| `cohort.py` | Carrier status, per-module sex decoding, visit concurrency, complete-case masks, conflict-safe per-participant values. | `carrier_status`, `decode_sex`, `concurrent_visit`, `complete_case_mask`, `unique_per_participant` |
| `prediction.py` | Nested candidate-grid selection with partition-owned preprocessing. Participant-clustered paired bootstrap. | `nested_fold`, `paired_metrics`, `candidate_grid`, `Candidate`, `CovariateDesign`, `ImageDesign` |
| `provenance.py` | Chunked SHA-256, numpy-safe JSON, environment capture, `manifest.json` writing and re-verification. | `sha256`, `jsonable`, `save_json`, `read_json`, `code_hashes`, `environment`, `write_manifest`, `verify_manifest` |

Everything is importable from `pie.experiment` directly or through the submodules
(`from pie.experiment import cohort, prediction, provenance`).

## cohort

| Function | Returns |
|---|---|
| `carrier_status(frame, genes=GENES)` | float `Series` on `frame.index`: 1, 0 or NaN |
| `decode_sex(table, module_column="PAG_NAME", sex_column="SEX", encodings=SEX_ENCODINGS)` | float `Series` on `table.index`: 1 = male, 0 = female, NaN |
| `concurrent_visit(frame, left, right, equivalent=("SC", "BL"))` | bool `Series` on `frame.index` |
| `complete_case_mask(frame, features)` | bool `ndarray`, one per row |
| `unique_per_participant(table, value, by="PATNO")` | `Series` indexed by `by` |

`GENES = ["LRRK2_carrier", "GBA_carrier", "SNCA_carrier"]`.
`SEX_ENCODINGS = {"SCREEN": {0: 0.0, 1: 1.0}, "PARTICIPANT_PROFILE": {1: 1.0, 2: 0.0}}`.

The PPMI-specific traps. Getting any one of them wrong corrupts the whole cohort:

```python
import numpy as np, pandas as pd
from pie.experiment import cohort

genetics = pd.DataFrame({"PATNO": [1, 2, 3, 4],
                         "LRRK2_carrier": [0, 1, 0, np.nan],
                         "GBA_carrier":   [0, np.nan, np.nan, np.nan],
                         "SNCA_carrier":  [0, 0, 0, np.nan]})
cohort.carrier_status(genetics).tolist()     # [0.0, 1.0, nan, nan]: PATNO 3 is untested for GBA

demographics = pd.DataFrame({"PATNO":    [1, 1, 2, 3, 3],
                             "PAG_NAME": ["SCREEN", "PARTICIPANT_PROFILE", "SCREEN",
                                          "SCREEN", "PARTICIPANT_PROFILE"],
                             "SEX":      [1, 1, 0, 1, 2]})
demographics["sex_male"] = cohort.decode_sex(demographics)
sex_male = cohort.unique_per_participant(demographics, "sex_male")
sex_male.to_dict()                           # {1: 1.0, 2: 0.0}: PATNO 3's modules disagree, dropped
genetics["sex_male"] = genetics["PATNO"].map(sex_male)

visits = pd.DataFrame({"T1_EVENT_ID":  ["BL", "SC", "V04", "UNK"],
                       "SAA_EVENT_ID": ["BL", "BL", "BL", "UNK"]})
cohort.concurrent_visit(visits, "T1_EVENT_ID", "SAA_EVENT_ID").tolist()   # [True, True, False, False]

idps = pd.DataFrame({"nm_contrast": [1.0, np.nan, np.inf, 2.0], "fw_sn": [0.1, 0.2, 0.3, 0.4]})
cohort.complete_case_mask(idps, ["nm_contrast", "fw_sn"])                 # [ True False False  True]
```

- `carrier_status` returns 0 only when *every* listed gene is an explicit negative. A
  participant untested for one gene is not a genetic control. Coding missing genotype as
  non-carrier inflates every carrier-versus-control contrast. The gene columns must be
  numeric 0/1. A string `"1"` or any other code counts as unknown.
- `decode_sex` decodes each row by its source module. `SCREEN` codes male 1 and female 0,
  and `PARTICIPANT_PROFILE` codes male 1 and female 2. A rule written for one module and applied to
  the other either loses every female or relabels them as male, and every downstream model
  quietly absorbs it. A module not in `encodings` raises `ValueError("Unverified demographic
  module")`. It does not borrow another module's mapping. Codes outside a mapping, and rows with no
  module, give NaN. The result is on the *demographics* index. As above, reduce it per participant and map
  by `PATNO`. Assigning it straight into a frame with a different index aligns on row labels,
  which do not correspond to participants.
- `concurrent_visit` compares visit *codes*, and treats `SC` and `BL` as one occasion. Assay
  dates are not sample dates: a `RUNDATE` is when the laboratory thawed the sample, so date
  arithmetic on it fabricates concurrency. An empty or `UNK` code on the left is never
  concurrent. If either column is absent the mask is all `False`, not an error, so a
  misspelt column name gives an empty cohort. Check the count.
- `complete_case_mask` treats ±inf as missing and raises on an empty feature list.
- `unique_per_participant` drops participants whose repeated records disagree instead of
  taking whichever row came first. It also drops participants whose value is always
  missing.

## prediction

### The analysis frame

`nested_fold` expects one row per participant with:

- `PATNO`: participant IDs, numeric or string. The audit records numeric IDs as `int` and
  others as given.
- The outcome column (default `"saa_prodromal"`), 0/1 for every training row. Test-row
  outcomes are never read and may be missing.
- The covariates. The default is `COVARIATES = ["age_at_scan", "sex_male", "LRRK2_carrier",
  "GBA_carrier", "SNCA_carrier", "APOE_e4"]`, or pass `covariates=[...]`. **The order matters.** The
  first covariate gets a spline, so it must be age. The first two are the nuisance set (age
  and sex) that feature blocks are residualised on.
- The feature columns named in `families`, and the batch and ICV columns if used.

A covariate or feature column **missing from the frame is not an error**. It is read as all
missing, imputed to a constant and contributes nothing. Check
`set(covariates) - set(frame.columns)` before running.

### `nested_fold`

```python
predictions, audit = pred.nested_fold(frame, train_ids, test_ids, families,
                                      batch_cols, icv, adjust_baseline, seed,
                                      outcome="saa_prodromal", covariates=None)
```

| Argument | Meaning |
|---|---|
| `train_ids`, `test_ids` | **Positional** row indices (`frame.iloc`), e.g. from `StratifiedKFold.split`. Overlap raises `ValueError("Outer partition overlap")`. A `PATNO` on both sides raises `ValueError("Participant leakage")`. |
| `families` | `{name: [feature columns]}`, one block per imaging/assay modality. `{}` fits the baseline only. |
| `batch_cols` | Columns one-hot encoded as nuisance (scanner, site). Levels seen fewer than 8 times in the fit partition are pooled, so a rare scanner cannot become a participant-identifying feature. Unseen levels encode as zeros. |
| `icv` | Intracranial-volume column, or `None`. It is a residualisation covariate for every feature *not* prefixed with `NON_ICV_PREFIXES`. |
| `adjust_baseline` | `True` also puts `batch_cols` and `icv` into the covariates-only baseline. `False` keeps the baseline to demographics and genetics. |
| `seed` | Inner `StratifiedKFold(3, shuffle=True)` uses `seed + 1000`, inner fits use `seed + k`, the final refit uses `seed`. |

`predictions` is `{name: ndarray}` of test-set probabilities, in `test_ids` order, with one
entry per family plus `baseline` (best covariates-only candidate) and `selected` (the winner of the
whole grid). That set is what an increment-over-covariates comparison needs. `audit` has four keys:

| Key | Contents |
|---|---|
| `selections` | `{name: {candidate, mean_inner_auc, mean_inner_logloss, inner_rank}}` for `baseline`, `selected` and each family. `candidate` is the `Candidate` as a dict. `inner_rank` is its position in the inner ranking, 0 unless the winner failed its final refit. |
| `inner_audit` | `[{inner_fold, fit_patnos, validation_patnos}]`, the participants in every inner split. |
| `candidate_scores` | One `{candidate, auc, logloss}` per grid point, with per-inner-fold lists of 3. |
| `failures` | `[{candidate, inner_fold, stage, error}]`. `candidate` is the grid index. `stage` is `"inner"` or `"final_refit"`, and `inner_fold` is `None` for a final refit. |

What the module guarantees:

- **No preprocessing object is fitted before splitting.** Imputation medians, scaling,
  feature eligibility, nuisance residualisation and the PCA basis are all learned on one
  partition and applied to the others. A feature present only in the partition being
  predicted is dropped, not imputed from it.
- **The outer test outcome is never read.** `nested_fold` returns predictions and the caller
  joins labels afterwards. The test suite asserts this by flipping every test label and
  checking the predictions are byte-identical.
- **Participants cannot straddle a split.** A `PATNO` on both sides raises.
- **Selection is deterministic**: largest mean inner AUC, then lowest log loss, then declared
  grid order. The same data give the same choice.
- **Non-convergence is a failure, not a model.** A `ConvergenceWarning` (or a
  `ValueError`/`FloatingPointError`) disqualifies the candidate rather than letting an
  unconverged fit into the comparison. A candidate that fails in *any* inner fold is
  ineligible. If every candidate for a family fails, the result is
  `ValueError("All candidates failed")`. The refit on the full training partition is treated
  the same way. A winner that fails there is recorded in `failures` with
  `stage="final_refit"`, and the next-ranked eligible candidate is fitted instead, as
  `inner_rank` shows. A candidate that has failed is never retried.

### Preprocessing and the grid

`ImageDesign` (one feature block) keeps a feature only if it is finite in ≥ 70 % of the fit
partition and not constant. It then median-imputes and standardises, optionally
residualises on nuisance with a ridge regression (`alpha=1`), re-standardises, and
optionally PCA-compresses to `min(32, n_fit − 2, n_features)` unit-variance components.
`CovariateDesign` median-imputes with missing-indicator columns, standardises, adds a
degree-2 spline of the first covariate, and one-hot encodes `batch_cols`. `nuisance=True`
restricts it to the first two covariates. Both are `fit(frame)` / `transform(frame)` pairs,
and both record `fit_ids`, the participants they were fitted on.

Measures of the nigra itself are not proportional to head size, so columns prefixed
`nm_`, `dwi_`, `new_nm_` or `new_dti_` (`NON_ICV_PREFIXES`) are residualised without the
intracranial-volume covariate that cortical and subcortical morphometry gets. Functional
connectivity isn't proportional to head size either. `fmri_` edges are correlations between
regional time series, so `fmri_` is in the list too. Name features with these prefixes
on purpose. Any other column, including ratios like a DaT SBR or an assay value, is
treated as morphometry and residualised on ICV whenever `icv` is given.

The grid is small and declared up front by `candidate_grid(families)`, in a fixed order that
is also the tie-breaker. A `Candidate(family, residualize=False, pca=False, model="logistic",
strength=0.1, image_weight=1.0)` is one point. `strength` is `C` for logistic regression
(`C_VALUES = [0.01, 0.1, 1.0, 10.0]`) and `max_depth` for gradient boosting (2 or 3).
`image_weight` 0.2 scales the feature block down relative to the covariates, which penalises
it more heavily under the same `C`.

| Block | Candidates |
|---|---|
| `baseline` | 4 logistic (`C_VALUES`) + 2 boosted = 6 |
| family with < 64 features | × residualise {no, yes} × (8 logistic + 2 boosted) = 20 |
| family with ≥ 64 features | × residualise {no, yes} × PCA {no, yes}: 8 logistic each, boosted only on the PCA version = 36 |

`nested_fold` looks up `candidate_grid` at call time, so replacing it narrows the search for
a cheap sensitivity run or a smoke test. The tests do this.

```python
from unittest import mock
from pie.experiment import prediction as pred

narrow = lambda families: [pred.Candidate("baseline"), *(pred.Candidate(f) for f in families)]
with mock.patch.object(pred, "candidate_grid", narrow):
    ...   # nested_fold calls here search 1 + len(families) candidates
```

### End-to-end: repeated outer cross-validation

```python
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits          # installed with scikit-learn
from pie.experiment import prediction as pred

threadpool_limits(1)      # one OpenMP thread for the boosted candidates, see below
rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({"PATNO": np.arange(1, n + 1),
                      "age_at_scan": rng.uniform(55, 80, n),
                      "sex_male": rng.integers(0, 2, n),
                      "scanner_batch": rng.choice(["A", "B", "C"], n),
                      "MaskVol": rng.normal(1.5e6, 1e5, n),
                      "saa_prodromal": rng.integers(0, 2, n)})
for gene in ["LRRK2_carrier", "GBA_carrier", "SNCA_carrier", "APOE_e4"]:
    frame[gene] = rng.integers(0, 2, n)
frame["nm_sn_contrast"] = 0.8 * frame["saa_prodromal"] + rng.normal(0, 1, n)   # carries signal
frame["putamen_vol"] = rng.normal(0, 1, n)                                     # noise
frame["caudate_vol"] = rng.normal(0, 1, n)
families = {"nm": ["nm_sn_contrast"], "t1": ["putamen_vol", "caudate_vol"]}

folds = []
for repeat, seed in enumerate(pred.SEEDS):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train_ids, test_ids in outer.split(frame, frame["saa_prodromal"]):
        p, audit = pred.nested_fold(frame, train_ids, test_ids, families,
                                    batch_cols=["scanner_batch"], icv="MaskVol",
                                    adjust_baseline=False, seed=seed)
        fold = pd.DataFrame({"PATNO": frame["PATNO"].iloc[test_ids].to_numpy(), "repeat": repeat,
                             "y": frame["saa_prodromal"].iloc[test_ids].to_numpy()})  # labels joined after
        for name, probs in p.items():
            fold[f"p_{name}"] = probs
        folds.append(fold)
oof = pd.concat(folds, ignore_index=True)

audit["selections"]["selected"]["candidate"]     # last fold's winner
table = pred.paired_metrics(oof, n_boot=500)
table[["model", "auc_averaged_oof", "delta_auc", "delta_ci95_low", "delta_ci95_high"]]
```

Keep every fold's `audit` (with `provenance.save_json`) if the selections are going into a
report. The example above keeps only the last one.

**Cost.** Each call fits every candidate in each of 3 inner folds, then refits the winners.
With the two small families above that is 46 candidates, about 140 fits per outer fold.
The gradient-boosted candidates use OpenMP. On small partitions, one thread per fit is
far faster than letting every fit claim every core. Uncapped, a boosted fit on a busy
machine can take seconds instead of a fraction of one. Cap threads with
`threadpool_limits(1)` or `OMP_NUM_THREADS=1`, and always when running outer folds in
parallel.

### Paired increments

```python
table = pred.paired_metrics(oof, n_boot=2000, seed=pred.SEEDS[0], baseline="p_baseline")
```

`oof` has one row per participant per repeat, with columns `PATNO`, `repeat`, `y` and
one `p_<model>` per model. Every column starting with `p_` is treated as a model. The
result has one row per model, including the baseline itself with a delta of 0:

| Column | Meaning |
|---|---|
| `model`, `n`, `n_positive` | model name (the `p_` prefix removed), participants, positives |
| `auc_averaged_oof`, `baseline_auc_averaged_oof`, `delta_auc` | AUC of the repeat-averaged predictions, and its increment over the baseline |
| `delta_ci95_low/high`, `delta_ci98_75_low/high` | bootstrap interval for `delta_auc`. 98.75 % = 1 − 0.05/4, for four comparisons sharing α = 0.05 |
| `auc_ci95_low/high` | bootstrap interval for the model's own AUC |
| `mean_repeat_auc`, `mean_repeat_delta`, `repeat_deltas`, `positive_in_all_repeats` | the same contrast computed within each repeat, without averaging |
| `logloss`, `brier`, `delta_logloss`, `delta_brier` | calibration-sensitive scores. A negative delta is better than baseline |
| `uncertainty` | the caveat below, as text |

Predictions are averaged within participant, then resampled by participant and stratified
on outcome. A participant appearing in several repeats therefore moves in and out of a resample
together, and the interval reflects the number of people rather than the number of rows.
A participant whose `y` differs between rows raises
`ValueError("Participant outcome changed between folds")`. The interval is
**conditional on the fitted models**. It covers sampling variation in the evaluated
participants, not the variability of the model search or of external validation. Every
row carries that caveat in its `uncertainty` column so the number cannot be quoted without
it.

## provenance

| Function | Returns / does |
|---|---|
| `sha256(path)` | hex digest, read in 1 MiB chunks, so a multi-GB NIfTI costs no memory |
| `jsonable(value)` | plain-Python copy: numpy scalars and arrays → int/float/bool/list, `Path` → str, dict keys → str, NaN/±inf → `None` |
| `save_json(path, value, indent=2)` | writes `jsonable(value)` as strict JSON (`allow_nan=False`). Unrecognised types are written as `str(value)` |
| `read_json(path)` | parsed JSON |
| `code_hashes(source, pattern="*.py")` | a directory gives `{file name: sha256}` for its `pattern` matches (not recursive). A file or list of files gives `{path: sha256}` |
| `environment(packages=("numpy", "scipy", "pandas", "sklearn", "joblib", "nibabel", "torch"), binaries=())` | `{python, platform, packages: {name: version}}`, plus `binaries: {path: sha256}` if given. Uninstalled packages are left out rather than recorded as `null` |
| `write_manifest(out_dir, inputs=(), code=None, outputs=None, started=None, name="manifest.json", include_environment=True, **extra)` | writes and returns the record below |
| `verify_manifest(out_dir, name="manifest.json")` | list of input and output paths whose hash no longer matches. `[]` means unchanged |

The manifest has these keys: `created_utc` (timezone-explicit ISO time), `inputs_sha256` (`{path as given: hash}`),
`code_sha256` (if `code`), `environment` (if `include_environment`), `seconds` (if
`started`), every `**extra` keyword (seed, counts, selections), and finally `outputs_sha256`
(`{path relative to out_dir: hash}`, such as `figures/roc.csv` or `../shared.csv`).
`outputs=None` means every file under `out_dir`, subdirectories included, except the
manifest itself. Every input, and every explicit output, must be an existing file.
Otherwise `write_manifest` raises `FileNotFoundError` before writing anything.

```python
import tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from pie.experiment import provenance as prov

work = Path(tempfile.mkdtemp())
(work / "analysis.py").write_text("print('analysis')\n")
cohort_csv = work / "cohort.csv"
pd.DataFrame({"PATNO": [1, 2, 3], "y": [0, 1, 0]}).to_csv(cohort_csv, index=False)

started = time.time()
out_dir = work / "run"
out_dir.mkdir()
prov.save_json(out_dir / "metrics.json", {"auc": np.float64(0.71), "n": np.int64(3),
                                          "brier": float("nan")})         # NaN is written as null
record = prov.write_manifest(out_dir, inputs=[cohort_csv], code=work, started=started,
                             seed=20260913, selections={"nm": "logistic"})
sorted(record)
prov.verify_manifest(out_dir)                        # []

(out_dir / "metrics.json").write_text("{}")          # a result edited after the fact
prov.verify_manifest(out_dir)                        # ['<work>/run/metrics.json']

# verify_manifest does not re-check code. Compare the hashes yourself.
prov.code_hashes(work) == prov.read_json(out_dir / "manifest.json")["code_sha256"]   # True
```

- `verify_manifest` re-hashes **inputs and outputs only**. It tells you whether the files
  are the ones the run read and wrote. Whether the code is still the same is a separate
  check, `code_hashes(code) == record["code_sha256"]` as above.
- Input paths are stored as given. Relative paths are re-resolved against the working
  directory at verification time, so pass absolute paths. A missing input raises instead
  of silently dropping out of the record. Directories aren't hashable inputs, so list their
  files.
- Outputs are keyed relative to `out_dir` and re-checked there, so outputs in subdirectories
  and explicit outputs outside `out_dir` verify correctly. Older manifests keyed top-level
  outputs by bare file name, which is the same key, so they still verify.
- `save_json` turns NaN and ±inf into `null` (via `jsonable`) instead of emitting the
  non-standard `NaN` token that strict readers reject. `allow_nan=False` is the backstop for
  anything that slips past. A `null` metric in a results file usually means an unfinished
  computation, so look for it. Convert DataFrames yourself (`.to_dict("records")`), because
  `default=str` would otherwise write their printed repr.
- `include_environment=True` imports each listed package to read its version, including
  `torch` if it is installed, which takes a few seconds.
- `code_hashes` on a directory is not recursive, and its keys are bare file names.

## Provenance of this layer

These modules were extracted from the PPMI prodromal-synucleinopathy study (Study 1,
virtual biomarkers) after it closed. The numerics are unchanged from the engine that
produced its published results, and `tests/test_experiment_prediction.py` carries its
leakage tests on synthetic data.

Three later changes leave that engine's numbers alone. String `PATNO`s are accepted. A failed
final refit falls back instead of raising. `fmri_` joined `NON_ICV_PREFIXES`. The first two
change behaviour only on inputs where the frozen engine raised, and the third only affects
columns named `fmri_*`. Against the frozen code on synthetic t1/nm/dwi inputs, both
`adjust_baseline` settings give byte-identical predictions, selections, inner audit and
`paired_metrics`, apart from the new `stage` and `inner_rank` fields.

## Tests

```bash
venv_imaging/bin/python -m pytest tests/test_experiment_cohort.py \
    tests/test_experiment_prediction.py tests/test_experiment_provenance.py -q
```

`venv_py311` runs them too.
