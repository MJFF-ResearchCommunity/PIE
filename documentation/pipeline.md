# Tabular pipeline (`pie/pipeline.py`)

One command from a PPMI study-data download to a cross-validated classifier, with an HTML report
per stage. Each stage writes its result into the run directory and the next stage reads it back
from there, which is what lets `--skip-to` resume a run at any stage.

```
./PPMI ─1 reduce─► final_reduced_consolidated_data.csv ─2 engineer─► final_engineered_dataset.csv
       ─3 split + select─► selected_train_data.csv + selected_test_data.csv ─4 classify─► classification/
       ─5────────► pipeline_report.html
```

| Stage | Code | Doc |
|---|---|---|
| 1 Reduction | `pie_clean.DataLoader`, `pie.data_reducer.DataReducer` | [data_loader.md](data_loader.md), [data_reducer.md](data_reducer.md) |
| 2 Engineering | `pie.feature_engineer.FeatureEngineer` | [feature_engineer.md](feature_engineer.md) |
| 3 Split and selection | `pie.classifier.split_train_test`, `pie.feature_selector.FeatureSelector` | [feature_selector.md](feature_selector.md) |
| 4 Classification | `pie.classification_report.generate_report` → `pie.classifier.Classifier` | [classifier.md](classifier.md) |
| 5 Summary, per-stage HTML | `generate_main_report`, `pie.reporting` | [Reports](#reports) |

Two rules hold throughout, because PPMI has several visits per participant:

- **Participants never straddle a split.** The train/test split (stage 3) and every CV fold
  (stage 4) are grouped by `PATNO`. A model scored on another visit of a person it trained on is
  scored on recognising the person.
- **Nothing fitted sees the test rows.** Scaling, imputation and feature selection are fitted on
  the training participants only.

## CLI

```bash
python pie/pipeline.py --data-dir ./PPMI --output-dir output/pd_vs_hc --target-column COHORT \
    --modalities "subject_characteristics motor_assessments non_motor_assessments" \
    --fs-method fdr --fs-param 0.05 --n-models 3 --tune --budget 60
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `./PPMI` | PPMI study-data download. Read only by stage 1. |
| `--output-dir` | `output/pipeline_run` | Run directory; created if missing. |
| `--target-column` | `COHORT` | Column to predict. Must hold class labels, as text or integer codes (0/1, 1/2/3); a continuous column stops stage 3 with `ValueError` (regression is out of scope). Only when it is `COHORT` does stage 1 drop rows without a valid cohort. |
| `--leakage-features-path` | `config/leakage_features.txt` | Columns removed before selection and again before classification ([Leakage list](#leakage-list)). A path that does not exist logs a warning and the run continues with nothing removed. |
| `--modalities` | five core | Comma-, semicolon- or space-separated, case-insensitive, any of `KNOWN_MODALITIES`: the five core modalities (`subject_characteristics`, `medical_history`, `motor_assessments`, `non_motor_assessments`, `biospecimen`) and PIE-clean's extended folders (`study_enrollment`, `imaging`, `ppmi_online`, `remote_screening`, `found`, `roche_app`). Unknown names are dropped with a warning; if none is left, the five core modalities load. |
| `--imaging-features` | none | CSV of imaging-derived phenotypes keyed by `PATNO`/`EVENT_ID` ([below](#imaging-features)). |
| `--fs-method` | `fdr` | Any `FeatureSelector` method (`SUPPORTED_METHODS`); argparse rejects other names. The endgame methods need endgame installed. |
| `--fs-param` | `0.05` | FDR alpha for `fdr`; fraction of features kept for `k_best`, `rfe`, `mrmr` and `relief`; ignored by the others. |
| `--n-models` | `5` | How many top-ranked models are refitted on the full training split (`n_select`). The comparison always runs the whole default model set and the leaderboard shows all of them; only the first refitted model is used afterwards. |
| `--tune` | off | Randomised hyperparameter search on the best model (20 draws, grouped 5-fold, accuracy). The original model is kept if the tuned one does not score higher in the same CV. |
| `--no-plots` | plots on | Skip the matplotlib plots in `classification/plots/`. |
| `--budget` | `30.0` | Minutes for model comparison, checked before each model starts; a model that is already fitting runs to completion. |
| `--skip-to` | none | `reduction`, `engineering`, `selection` or `classification`: start at that stage using the previous stage's file in `--output-dir`. `reduction` is a full run. |

If a stage's input file is missing, `run_pipeline` raises `FileNotFoundError`, which names the file and the stage to run first. Stage 4 raises `ValueError` for unusable input and `RuntimeError` for an engine failure ([classifier.md](classifier.md#generate_report)). The command line logs any of these as one line and exits 1.

## Stages

### 1. Reduction

`run_data_reduction_step(data_dir, output_csv_path, output_html_path, modalities=None, imaging_features=None, target_column="COHORT")`

1. `DataLoader.load(data_path=data_dir, merge_output=False, modalities=...)` (`run_pipeline` passes
   `ALL_MODALITIES` when none are given). Raises `FileNotFoundError` if `data_dir` does not exist.
2. Adds the `--imaging-features` table, if any.
3. `DataReducer` with its default thresholds: `analyze` → `get_drop_suggestions` → `apply_drops` →
   `merge_reduced_data` → `consolidate_cohort_columns`.
4. Writes `final_reduced_consolidated_data.csv` (skipped, with a warning, if empty) and
   `data_reduction_report.html`.

Columns come out prefixed with their source: `subject_characteristics_AGE_AT_VISIT`,
`medical_history_Vital_Signs_SYSSUP`. Rows are one per `PATNO`/`EVENT_ID`. COHORT is always
consolidated into one column, but rows without one of the four valid cohorts are dropped only when
`COHORT` is the target (`keep_only_valid=(target_column == "COHORT")`); for any other target no row
is lost over a label that is not being modelled.

### Imaging features

`--imaging-features` takes any CSV with `PATNO` and `EVENT_ID` columns, typically
`fastsurfer_idps.csv` from `pie/imaging/run.py`.

- Kept: every non-text column except `IMAGEID` and `SCAN_DATE`, plus `EVENT_ID`. Text columns are
  dropped.
- Normally the CSV joins as modality `imaging` and its columns become `imaging_<column>`. When the
  Imaging folder is loaded too (`--modalities imaging`), the CSV is added to that modality as table
  `idps` instead of replacing its tables, and its columns become `imaging_idps_<column>`.
- It passes the same reduction rules as the clinical tables (an IDP missing on > 95 % of rows is
  dropped), and repeated scans at one visit are collapsed to the first non-null value per column.
- `EVENT_ID` must use clinical visit codes. A scan whose `EVENT_ID` matches no clinical visit
  becomes a row of its own, with no cohort.

### 2. Engineering

`run_feature_engineering_step(input_csv_path, output_csv_path, output_html_path, target_column="COHORT")` runs:

```python
FeatureEngineer(df, protected_columns=[target_column]).one_hot_encode(
    auto_identify_threshold=20, max_categories_to_encode=25, min_frequency_for_category=0.01
).scale_numeric_features(scaler_type="standard")
```

The target is protected: it is never one-hot encoded or scaled. Text columns with ≤ 20 distinct
values are one-hot encoded (levels under 1 % pooled into `_OTHER_`); wider text columns stay as
text. Every other numeric column is standardised here on all rows; stage 3 re-fits that scaling on
the training participants. All-NaN numeric columns are dropped. Writes
`final_engineered_dataset.csv` and `feature_engineering_report.html`.

### 3. Split and selection

`run_feature_selection_step(input_csv_path, train_csv_path, test_csv_path, output_html_path, target_column, fs_method, fs_param_value, leakage_features_path=None)`

1. Drop the columns named in the leakage file (exact match). The target, `PATNO` and `EVENT_ID`
   are never dropped by it.
2. Drop rows with a missing target. `check_classification_target` raises `ValueError` if the target
   is continuous. `PATNO` becomes `int`.
3. Features: every column except the target, `PATNO` and `EVENT_ID`.
4. Text columns holding `|`-joined values (PIE-clean writes `"1.0|3.0"` when a visit has several
   values), whose names contain none of `ID`, `DATE`, `TIME`, `PATNO`, `EVENT`: if more than 90 % of
   non-null values parse as numbers, each becomes the mean of its parts.
5. One-hot (`bool`) columns become 0/1; any other non-numeric column is dropped (logged).
6. Split by participant: `split_train_test(y, groups=PATNO, test_size=0.2, random_state=42)` takes
   the first fold of `StratifiedGroupKFold(n_splits=5)`: about 20 % of participants, with the class
   mix kept as close as whole participants allow. Without a `PATNO` column it is a stratified row
   split.
7. Re-standardise every non-one-hot feature on the training rows. z-scoring is affine, so this
   equals fitting stage 2's scaler on the training rows alone. Then fill `NaN` with 0, which is now
   the training mean.
8. Fit `FeatureSelector(fs_method, "classification", ...)` on the training rows and apply it to both.

Writes `selected_train_data.csv` and `selected_test_data.csv` (selected features, `PATNO` for the
grouped CV of stage 4, and the target with its original labels) and `feature_selection_report.html`.

### 4. Classification

Calls `generate_report(train_csv_path=..., test_csv_path=..., use_feature_selection=False,
target_column=..., exclude_features=<leakage list>, output_dir=<run>/classification, ...)` with the
`--n-models`, `--tune`, `--no-plots` and `--budget` values. `generate_report` uses `PATNO` to group
the CV folds and never as a feature. See [classifier.md](classifier.md).

### 5. Summary

`generate_main_report(report_data, output_path)` writes `pipeline_report.html`: a table per stage run
in *this* invocation, with a link to each stage report, then calls `webbrowser.open` on it (set
`BROWSER=true` to suppress that on a headless machine). Stage 4's HTML report also opens itself.

### Output files

| File | Stage |
|---|---|
| `final_reduced_consolidated_data.csv`, `data_reduction_report.html` | 1 |
| `final_engineered_dataset.csv`, `feature_engineering_report.html` | 2 |
| `selected_train_data.csv`, `selected_test_data.csv`, `feature_selection_report.html` | 3 |
| `classification/classification_report.html`, `classification/final_classifier_model.pkl`, `classification/plots/` | 4 |
| `pipeline_report.html` | 5 |

## Python API

```python
from pie.pipeline import run_pipeline

run_pipeline(
    data_dir="./PPMI",
    output_dir="output/pd_vs_hc",
    target_column="COHORT",
    leakage_features_path="config/leakage_features.txt",   # required here, no default
    modalities=None,               # list; None -> ALL_MODALITIES
    fs_method="fdr",
    fs_param_value=0.05,
    n_models_to_compare=5,
    tune_best_model=False,
    generate_plots=True,
    budget_time_minutes=30.0,
    skip_to_step=None,             # "reduction" | "engineering" | "selection" | "classification"
    imaging_features=None,         # path to an IDP CSV
)                                  # returns None
```

The stage functions are public and can run alone; each is wrapped in `timing_decorator`, which logs
its wall time. Each returns a summary dict used by `generate_main_report`:

| Function | Returns |
|---|---|
| `run_data_reduction_step` | `initial_tables`, `reduced_tables`, `initial_size_mb`, `reduced_size_mb`, `output_shape`, `report_path` |
| `run_feature_engineering_step` | `input_shape`, `output_shape`, `new_features`, `report_path` |
| `run_feature_selection_step` | `initial_features`, `final_features`, `train_shape`, `test_shape`, `report_path` |

`parse_modalities(text)` is the `--modalities` parser: it returns the valid names as a list, or
`None` for the core default.

Stages 2–3 on synthetic data, no PPMI access needed:

```python
from pathlib import Path
import numpy as np
import pandas as pd
from pie.pipeline import run_feature_engineering_step, run_feature_selection_step

out = Path("demo_run"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
n = 300
cohort = rng.choice(["Parkinson's Disease", "Healthy Control"], n)
pd.DataFrame({
    "PATNO": np.arange(1, n + 1), "EVENT_ID": "BL", "COHORT": cohort,
    "subject_characteristics_AGE": rng.normal(65, 8, n),
    "motor_assessments_NP3TOT": np.where(cohort == "Parkinson's Disease", 20, 2) + rng.normal(0, 4, n),
    "subject_characteristics_APPRDX": np.where(cohort == "Parkinson's Disease", 1, 2),   # leaks the label
}).to_csv(out / "final_reduced_consolidated_data.csv", index=False)
(out / "leakage.txt").write_text("subject_characteristics_APPRDX\n")

run_feature_engineering_step(str(out / "final_reduced_consolidated_data.csv"),
                             output_csv_path=out / "final_engineered_dataset.csv",
                             output_html_path=out / "feature_engineering_report.html")
info = run_feature_selection_step(str(out / "final_engineered_dataset.csv"),
                                  train_csv_path=out / "selected_train_data.csv",
                                  test_csv_path=out / "selected_test_data.csv",
                                  output_html_path=out / "feature_selection_report.html",
                                  target_column="COHORT", fs_method="fdr", fs_param_value=0.05,
                                  leakage_features_path=str(out / "leakage.txt"))
pd.read_csv(out / "selected_train_data.csv").columns.tolist()
# ['motor_assessments_NP3TOT', 'PATNO', 'COHORT']      APPRDX removed as leakage, AGE not significant
```

`python pie/pipeline.py --output-dir demo_run --skip-to classification` continues from there.

## Leakage list

`config/leakage_features.txt` holds one column name per line, matched exactly against the prefixed
names stage 1 produces (`subject_characteristics_APPRDX`, not `APPRDX`). `config/constants.py`
defines the same names as the Python list `LEAKAGE_FEATURES`. The list's `PATNO` and `EVENT_ID`
entries are harmless: IDs and the target are handled explicitly and never removed by it.

The shipped list is an example for PD-versus-control on `COHORT` and must be rebuilt for each
question. It removes columns that restate the label rather than predict it:

- diagnosis and enrolment fields (`APPRDX`, `NEWDIAG`, `ENRL*`);
- exam findings that define the diagnosis (`FEATBRADY`, `FEATRIGID`, `DXRIGID`, …);
- treatment (`PDTRTMNT`, `PDMEDYN`, `DBSYN`);
- substudy flags only one cohort receives (`AV133STDY`, `SV2ASTDY`, `GAITSTDY`);
- genetic variants that define the genetic cohorts (LRRK2, GBA, SNCA).

For biomarker discovery also remove anything a clinician observed; for prediction at a time point,
anything measured after it.

## Reports

`pie/reporting.py` writes the stage 1–3 HTML reports:

| Function | Writes |
|---|---|
| `generate_data_reduction_html_report(initial_dict_summary, reduced_dict_summary, analysis_report, initial_size_mb, reduced_size_mb, output_html_path, final_consolidated_df_shape=None)` | Totals before/after (tables, memory, rows, columns, null %), then per table: shape before/after and dropped columns with reasons. Biospecimen column lists are omitted because they run to thousands of names. |
| `generate_feature_engineering_report_html(report_data, output_html_path)` | Input/output shape and path, and per-operation counts with up to 10 example names. Reads `input_csv_path`, `input_data_shape`, `output_csv_path`, `output_data_shape`, `feature_engineering_summary` (from `FeatureEngineer.get_engineered_feature_summary()`). |
| `generate_feature_selection_report_html(report_data, output_html_path)` | Target, rows dropped, split shapes, selected feature names, output paths. Its preprocessing, VarianceThreshold and SelectFdr sections are filled only by `tests/test_feature_selector.py`; from the pipeline they show N/A. |

Other modules in `pie/` that the pipeline does not use:

| Module | Contents |
|---|---|
| `pie/utils.py` | `convert_xlsx_to_arff(xlsx_file, arff_file)`: first sheet of an Excel file to Weka ARFF (numeric → `NUMERIC`, datetime → `DATE`, else `STRING`; missing → `?`). |
| `pie/visualizer.py` | `Visualizer.plot_distribution(data, column)`: placeholder that only logs. |

## Caveats

- **Stage 2 still sees every row for the one-hot vocabulary.** Which levels exist and which are
  pooled into `_OTHER_` is decided on all rows. It reads no labels, but test participants' category
  frequencies do shape it.
- **Pipe-averaging reaches only wide text columns.** A `|`-joined column with ≤ 20 distinct values
  is one-hot encoded in stage 2 (`"1.0|3.0"` becomes a level of its own) before stage 3 could
  average it.
- **`--budget` cannot stop a model in progress.** Every model is capped at 2 threads, so the
  gradient-boosting models finish: on a 240 × 8 synthetic table, 2,000 rounds took about 3 s for
  `xgb` and `lgbm` and 22 s for `catboost`. On wide PPMI frames expect the budget to overrun by a
  model's full fitting time.
- **COHORT consolidation uses every column with `COHORT` in its name** ([data_reducer.md](data_reducer.md#consolidate_cohort_columns)).

## Tests

Tests in `tests/` that need the PPMI download, or files produced from it, are marked `ppmi`. Run
the synthetic suite with `-m "not ppmi"` and the real-data tests with `-m ppmi`. pytest warns that
`ppmi` is an unregistered marker until it is declared in a pytest config.

| Test (`tests/test_pipeline.py`) | Checks |
|---|---|
| `test_pipeline_end_to_end_synthetic` | Stages 1–5 with a stubbed `DataLoader` (fake participants, two visits each): every output exists, `COHORT` rows filtered, no `PATNO` in both splits, training features have mean 0, one-hot columns selected, leakage column gone. |
| `test_pipeline_numeric_binary_target` | A 0/1 target: no `COHORT` filtering, target unscaled, classification report written. |
| `test_fraction_based_fs_methods` | `rfe` and `k_best` keep `--fs-param` × features. |
| `test_parse_modalities_accepts_extended_modalities` | `--modalities` parsing. |
| `test_imaging_features_do_not_replace_imaging_tables` | `imaging` modality plus `--imaging-features` keeps both. |
| `test_full_pipeline_with_real_data` (`ppmi`) | Full run on `./PPMI` into `output/test_pipeline_run`, leakage file under `tmp_path`; all outputs exist, `APPRDX` gone, participant-level split. `tests/test_from_fs.py` (`ppmi`) restarts that run's data at stage 3. |

```bash
pytest tests/test_pipeline.py -m "not ppmi" -q
pytest tests/test_pipeline.py tests/test_from_fs.py -m ppmi -q
```
