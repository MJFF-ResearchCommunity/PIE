# DataReducer (`pie.data_reducer`)

A full PPMI load is dozens of tables and thousands of columns, many empty, constant or operational.
`DataReducer` profiles each table *before* the merge, drops the columns that cannot carry signal, and
then merges the survivors into one row per `PATNO`/`EVENT_ID`. Reducing first is what keeps the
merge in memory: the wide frame is built only from columns that survived.

```
DataLoader dict ──analyze──► report ──get_drop_suggestions──► {table: [cols]} ──apply_drops──► reduced dict
reduced dict ──merge_reduced_data──► wide frame ──consolidate_cohort_columns──► analysis frame (one COHORT column)
```

This is step 1 of the [pipeline](pipeline.md), run there with the default configuration.

## Constructor

```python
DataReducer(data_dict: dict, config: dict | None = None)
```

`data_dict` is the output of `DataLoader.load(merge_output=False)`: values are DataFrames or dicts of
DataFrames (e.g. `medical_history`). `config` overrides individual keys of `DataReducer.DEFAULT_CONFIG`:

| Key | Default | Rule |
|---|---|---|
| `missing_threshold` | `0.95` | Drop if the missing fraction is **>** this. |
| `single_value_threshold` | `1.0` | Drop if there is one distinct non-null value covering ≥ this share of non-null rows. |
| `low_variance_threshold` | `0.01` | Numeric columns: drop if standard deviation < this. |
| `check_low_variance_numeric` | `True` | Enable the low-variance rule. |
| `high_cardinality_ratio` | `0.9` | Drop if unique/rows > this **and** the column looks like an ID (numeric, or strings averaging > 10 characters). |
| `check_high_cardinality` | `False` | Enable the high-cardinality rule. Off by default: on one-row-per-visit tables every continuous measurement is high-cardinality. |
| `common_metadata_cols` | `["REC_ID", "ORIG_ENTRY", "LAST_UPDATE", "PAG_NAME", "QUERY_ID", "QUERY_TEXT"]` | Always dropped: PPMI record-keeping, not measurements. |

`PATNO` and `EVENT_ID` are never suggested. Rules run in the order missing → single value → low
variance → high cardinality → metadata, and each column is reported under the first rule it hits.
The thresholds are absolute: `missing_threshold=0.95` drops a biomarker measured in only 4 % of
visits even if it is the best predictor in that 4 %.

## Methods

| Method | Returns |
|---|---|
| `analyze()` | `{key: {"summary_stats": ..., "drop_suggestions": ...}}`. `key` is the modality, or `"modality.table"` for nested dicts. |
| `get_drop_suggestions(analysis_report=None)` | `{key: [columns]}`. Calls `analyze()` when no report is passed. |
| `apply_drops(drop_suggestions)` | A deep copy of `data_dict` with those columns removed; the original is untouched. Unknown keys are logged and skipped. |
| `generate_report_str(analysis_report=None)` | Plain-text summary: shape and up to five dropped columns with reasons per table. |
| `merge_reduced_data(reduced_data_dict, output_filename="merged_reduced_data.csv")` | One wide DataFrame. Writes it to `output_filename`; pass `None` to skip writing. |
| `consolidate_cohort_columns(dataframe, target_cohort_col_name="COHORT", keep_only_valid=True)` | The frame with a single cleaned cohort column; filtered to valid cohorts when `keep_only_valid`. |

### `analyze()` report

```python
report["medical_history.Vital_Signs"] = {
    "summary_stats": {
        "shape": (rows, cols),
        "column_info": {col: {"Dtype", "Non-Null Count", "Null Count", "Null Pct"}},
        "numeric_summary": {col: describe() stats},
        "categorical_summary": {col: {top-5 value counts..., "_unique_count": n}},
    },
    "drop_suggestions": {"columns": [...], "reasons": {col: "High Missing % (97.50%)"}, "count": n},
}
```

An empty table gets `summary_stats = {"shape": (0, 0), "info": "Empty DataFrame"}` and
`drop_suggestions = {"reason": "Empty DataFrame", "columns": []}`.

### `merge_reduced_data`

- Every non-key column is renamed `<modality>_<column>`, or `<modality>_<table>_<column>` for nested
  tables (`medical_history_Vital_Signs_SYSSUP`). Prefixing everything means the same PPMI variable
  name in two forms can never collide, and a column's origin is visible in every downstream report.
  Leakage lists (`config/leakage_features.txt`) use these prefixed names.
- `PATNO` is cast to `str`.
- Tables without both `PATNO` and `EVENT_ID` are skipped with a warning.
- Duplicate `PATNO`/`EVENT_ID` rows within one table are collapsed to the first non-null value per
  column, so each table contributes at most one row per visit and the joins cannot multiply rows.
- The base frame is the union of key pairs over all tables; each table is left-joined onto it. Row
  order is arbitrary.

### `consolidate_cohort_columns`

PPMI records cohort in more than one table, so after the merge it appears as several prefixed
columns (`subject_characteristics_COHORT`, ...). This method:

1. Takes every column whose name contains `COHORT` (case-insensitive) and sets
   `target_cohort_col_name` to the first non-empty value across them, in column order.
2. Drops the source columns.
3. Maps `PD` → `Parkinson's Disease` and `Control` → `Healthy Control` (case-insensitive).
4. With `keep_only_valid=True` (the default), keeps only rows whose cohort is
   `Parkinson's Disease`, `Prodromal`, `Healthy Control` or `SWEDD`; rows with a missing or other
   cohort are **removed**, not relabelled. The pipeline passes `keep_only_valid=False` unless COHORT
   is the target, so rows are not lost over a label that is not being modelled. Missing cohorts stay
   missing (`NaN`).

Any column with `COHORT` in its name counts as a source, so rename unrelated ones first.

## Example

Runnable with synthetic data:

```python
import numpy as np
import pandas as pd
from pie.data_reducer import DataReducer

rng = np.random.default_rng(0)
patno = np.repeat(np.arange(1, 21), 2)                 # 20 fake participants x 2 visits
event = np.tile(["BL", "V04"], 20)
n = len(patno)
data_dict = {
    "subject_characteristics": pd.DataFrame({
        "PATNO": patno, "EVENT_ID": event,
        "COHORT": np.repeat(rng.choice(["PD", "Control", "Prodromal", "Other"], 20), 2),
        "AGE": rng.normal(65, 8, n),
        "REC_ID": np.arange(n),                        # metadata
        "SITE": ["S1"] * n,                            # single value
        "MOSTLY_EMPTY": [1.0] + [np.nan] * (n - 1),    # 97.5 % missing
    }),
    "medical_history": {
        "Vital_Signs": pd.DataFrame({
            "PATNO": np.concatenate([patno, [1]]),     # one duplicate visit row
            "EVENT_ID": np.concatenate([event, ["BL"]]),
            "SYSSUP": rng.normal(125, 10, n + 1),
            "TINY_VAR": 1 + rng.normal(0, 1e-4, n + 1),
        }),
    },
}

reducer = DataReducer(data_dict, config={"missing_threshold": 0.9})
report = reducer.analyze()
print(reducer.generate_report_str(report))
drops = reducer.get_drop_suggestions(report)
# {'subject_characteristics': ['MOSTLY_EMPTY', 'REC_ID', 'SITE'],
#  'medical_history.Vital_Signs': ['TINY_VAR']}

reduced = reducer.apply_drops(drops)
merged = reducer.merge_reduced_data(reduced, output_filename=None)
merged.columns.tolist()
# ['PATNO', 'EVENT_ID', 'subject_characteristics_COHORT', 'subject_characteristics_AGE',
#  'medical_history_Vital_Signs_SYSSUP']            40 rows: the duplicate visit was collapsed

final = reducer.consolidate_cohort_columns(merged)
sorted(final["COHORT"].unique())
# ['Healthy Control', "Parkinson's Disease", 'Prodromal']      the "Other" rows are gone
```

On a real download:

```python
from pie_clean import DataLoader
from pie.data_reducer import DataReducer

data = DataLoader.load("./PPMI", merge_output=False,
                       biospec_exclude=["project_9000", "project_222", "project_196"])
reducer = DataReducer(data)
reduced = reducer.apply_drops(reducer.get_drop_suggestions())
final = reducer.consolidate_cohort_columns(reducer.merge_reduced_data(reduced, output_filename=None))
```

The HTML version of the report is written by `pie.reporting.generate_data_reduction_html_report`
(see [pipeline.md](pipeline.md#reports)).

## Tests

| Test (`tests/test_data_reducer.py`) | Checks |
|---|---|
| `test_consolidate_keep_only_valid_false_keeps_rows` | Synthetic: `keep_only_valid=False` keeps other and missing cohorts, still maps `PD`. |
| `test_data_reduction_workflow` (`ppmi`) | Full workflow on `./PPMI` (skipped if missing); writes `output/final_reduced_consolidated_data.csv` and `output/data_reduction_report.html`, which the feature engineering test reads. |

```bash
pytest tests/test_data_reducer.py -m "not ppmi" -q
```
