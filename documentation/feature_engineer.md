# FeatureEngineer (`pie.feature_engineer`)

Turns the merged analysis frame into model-ready columns. A chainable wrapper: every method edits
an internal copy of the frame and returns `self`; `get_dataframe()` returns the result.

Stage 2 of the [pipeline](pipeline.md) runs exactly:

```python
FeatureEngineer(df, protected_columns=[target_column]).one_hot_encode(
    auto_identify_threshold=20, max_categories_to_encode=25, min_frequency_for_category=0.01
).scale_numeric_features(scaler_type="standard")
```

Every fitted step (scalers, imputers, target encoders, samplers, power transforms) learns from all
rows in the frame it is given. Pass it the training rows only when the result will be evaluated.

## Constructor

`FeatureEngineer(dataframe, protected_columns=None)` works on a copy of `dataframe`. `PATNO`,
`EVENT_ID`, `COHORT` and any `protected_columns` (matched case-insensitively) are never picked by
automatic column selection: not one-hot encoded, scaled, or used for polynomial or interaction
features. Protect the prediction target. Unprotected, a text target would be one-hot encoded away
and a numeric one standardised.

## Methods

| Method | Adds / changes | Needs endgame |
|---|---|---|
| `one_hot_encode(...)` | `<col>_<level>` bool columns; drops the source column | no |
| `handle_pipe_separated_column(column_name, strategy="multi_hot", max_unique_values_for_multi_hot=30, prefix=None)` | `<prefix>_first`, `<prefix>_count` or one 0/1 column per item | no |
| `scale_numeric_features(columns=None, scaler_type="standard", **scaler_params)` | scales in place; drops all-NaN numeric columns | no |
| `engineer_polynomial_features(columns=None, degree=2, interaction_only=False, include_bias=False)` | `a b`, `a^2`, … | no |
| `transform_numeric_distribution(column_name, new_column_name=None, transform_type="log", add_constant_for_log_sqrt=None)` | log / sqrt / Box-Cox / Yeo-Johnson | no |
| `apply_custom_transformation(column_name, func, new_column_name=None)` | `func(series)` | no |
| `impute(method="auto", columns=None, **kwargs)` | fills numeric `NaN` | `auto`, `knn`, `mice`, `missforest` |
| `encode(method="safe_target", columns=None, target=None, **kwargs)` | replaces categorical columns with numbers | `safe_target`, `catboost`, `leave_one_out` |
| `balance_classes(target_column, method="auto", **kwargs)` | resamples rows | `auto` (else imbalanced-learn) |
| `detect_noise(target_column, method="confident_learning", **kwargs)` | returns a mask, frame unchanged | yes |
| `create_interactions(method="auto", columns=None, **kwargs)` | interaction columns | `auto`, `interactions` |
| `get_dataframe()` | copy of the current frame | |
| `get_engineered_feature_summary()` | dict of what was added | |

`ENDGAME_PREPROCESS_AVAILABLE` says whether endgame's preprocessing imported. Methods that need it
fall back as described below when it is missing.

### `one_hot_encode`

```python
one_hot_encode(columns=None, prefix_sep="_", dummy_na=False, drop_first=False,
               max_categories_to_encode=20, min_frequency_for_category=None,
               ignore_for_ohe=None, auto_identify_threshold=50)
```

- `columns=None`: encode text/category columns with at most `auto_identify_threshold` distinct
  values (missing counted as a value). Numeric-coded categories are left alone unless listed.
- Never encoded, case-insensitively, in addition to `ignore_for_ohe`: the protected columns,
  `PAG_NAME`, `REC_ID`, `INFODT`, `ORIG_ENTRY`, `LAST_UPDATE`, `SITE_APRV`, `GUID`, `APPRDX`,
  `PRIMDIAG`. Those are keys, the target, record-keeping or timestamps, or diagnosis fields that
  would restate the target.
- Columns with more than `max_categories_to_encode` levels are skipped with a warning.
- `min_frequency_for_category=0.01` pools levels under 1 % into `_OTHER_` before encoding (only when
  some, not all, levels are rare), so a handful of rare levels does not become a handful of nearly
  empty columns.
- New columns are `bool` (from `pd.get_dummies`).

### `handle_pipe_separated_column`

PIE-clean stores several values for one visit as `"a|b"`. The original column is kept, so drop it
or add it to `ignore_for_ohe` before one-hot encoding.

| `strategy` | Result |
|---|---|
| `first` | `<prefix>_first`: first item, as text |
| `count` | `<prefix>_count`: number of items; a missing value counts as 1 |
| `multi_hot` | one 0/1 column per distinct item, `<prefix>_<item>` lower-cased with spaces and hyphens → `_`; skipped with a warning above `max_unique_values_for_multi_hot` items |

`prefix` defaults to the column name.

### `scale_numeric_features`

`scaler_type="standard"` (`StandardScaler`) or `"minmax"` (`MinMaxScaler`); `**scaler_params` go to the
scaler, and an unknown type logs an error and does nothing. With `columns=None` every numeric column
except the protected columns is scaled; `bool` columns are not numeric and are left alone.
Each column is scaled on its own non-missing values, `NaN` stays `NaN`.

All-NaN numeric columns are dropped here: they cannot be scaled, carry no signal, and would
otherwise surface later as constant-feature warnings and meaningless zero-filled features.

### `engineer_polynomial_features`

`PolynomialFeatures` over the given (or all numeric, minus protected) columns. `NaN` is median-filled for
the computation only. New columns use sklearn's names (`AGE UPSIT`, `AGE^2`; the bias term is
`poly_bias`), and existing columns are never overwritten. The column count grows quadratically, so
name the columns on a wide frame.

### `transform_numeric_distribution`

`transform_type` is `log`, `sqrt`, `box-cox` or `yeo-johnson`. `add_constant_for_log_sqrt` is added
first (to any type). `log` of non-positive values gives `NaN` with a warning; `box-cox` on
non-positive data is skipped. Power transforms use `PowerTransformer(standardize=False)` fitted on
the non-missing values. Without `new_column_name` the column is replaced in place (and not counted
as a new feature).

### `apply_custom_transformation`

`func` takes and returns a Series. Exceptions are logged, not raised, and the frame is left as it
was. New columns are tracked as `custom_transform_<func.__name__>_<column>`.

### `impute`

`columns` defaults to every column with missing values; only the numeric ones are imputed.

| `method` | Imputer |
|---|---|
| `simple_mean`, `simple_median` | sklearn `SimpleImputer` |
| `auto`, `knn`, `mice`, `missforest` | endgame `AutoImputer`, `KNNImputer`, `MICEImputer`, `MissForestImputer`; median without endgame |

An unknown method raises `ValueError`. All-NaN columns are left `NaN`, with a warning: there is
nothing to impute them from.

### `encode`

`columns` defaults to all text/category columns; they are replaced in place.

| `method` | Encoding |
|---|---|
| `ordinal` | sklearn `OrdinalEncoder` (unseen → −1) |
| `label` | sklearn `LabelEncoder` per column, on string values |
| `frequency` | share of rows with that level |
| `safe_target`, `catboost`, `leave_one_out` | endgame target encoders fitted with `target`; frequency encoding without endgame |

Target encoders read the labels, so fit them on training rows only.

### `balance_classes`

Resamples the whole frame and replaces it. `auto` uses endgame's `AutoBalancer`; otherwise
imbalanced-learn: `smote`, `adasyn`, `random_over`, `random_under`, `smoteenn`, `smotetomek`
(`random_state` from kwargs, default 42). All non-target columns must be numeric. Balance the
training split only: resampling before a split puts synthetic neighbours of test rows into training.
With neither library installed it logs a warning and does nothing.

### `detect_noise`

Returns a boolean Series named `is_noisy`, aligned with the frame, `True` for rows whose label
looks wrong. `method` is `confident_learning`, `consensus` or `crossval` (endgame
`ConfidentLearningFilter`, `ConsensusFilter`, `CrossValNoiseDetector`, each run through
`fit_detect`). All non-target columns must be numeric. Raises `ImportError` without endgame. The
frame is not changed; inspect or drop the flagged rows yourself.

### `create_interactions`

`auto` and `interactions` are endgame's `InteractionFeatures`: pairwise products and ratios named
`a*b`, `a/b` (`**kwargs` such as `operations=` or `max_interactions=` go to it). `polynomial` is
`engineer_polynomial_features(columns, degree=2, interaction_only=True)` and is also the fallback
without endgame. `columns` defaults to the numeric columns minus the protected ones; fewer than two
is a no-op.

### `get_engineered_feature_summary`

```python
{"total_original_columns": 9, "total_current_columns": 15, "newly_engineered_features_count": 8,
 "engineered_operations": {"one_hot_encoded": {"count": 2, "features": ["SEX_F", "SEX_M"]}, ...}}
```

`features` lists at most 10 names followed by `"..."`. This dict is what the feature engineering HTML
report renders.

## Example

Runnable with synthetic data:

```python
import numpy as np
import pandas as pd
from pie.feature_engineer import FeatureEngineer

rng = np.random.default_rng(0)
n = 100
df = pd.DataFrame({
    "PATNO": np.arange(1, n + 1), "EVENT_ID": "BL",
    "COHORT": rng.choice(["Parkinson's Disease", "Healthy Control"], n),
    "SEX": rng.choice(["M", "F"], n),
    "SITE": rng.choice([f"S{i:02d}" for i in range(30)], n),   # 30 levels: over the threshold, stays text
    "AGE": rng.normal(65, 8, n),
    "UPSIT": rng.integers(10, 40, n).astype(float),
    "MEDS": rng.choice(["A|B", "B", "A|C"], n),
    "EMPTY": np.nan,                                          # all-NaN: dropped by the scaler
})

fe = (FeatureEngineer(df)
      .handle_pipe_separated_column("MEDS", strategy="multi_hot")
      .one_hot_encode(ignore_for_ohe=["MEDS"], auto_identify_threshold=20,
                      max_categories_to_encode=25, min_frequency_for_category=0.01)
      .transform_numeric_distribution("UPSIT", new_column_name="UPSIT_log", transform_type="log")
      .engineer_polynomial_features(columns=["AGE", "UPSIT"], degree=2, interaction_only=True)
      .apply_custom_transformation("AGE", lambda s: (s >= 65).astype(int), new_column_name="AGE_65PLUS")
      .scale_numeric_features(scaler_type="standard"))

fe.get_dataframe().columns.tolist()
# ['PATNO', 'EVENT_ID', 'COHORT', 'SITE', 'AGE', 'UPSIT', 'MEDS', 'MEDS_a', 'MEDS_b', 'MEDS_c',
#  'SEX_F', 'SEX_M', 'UPSIT_log', 'AGE UPSIT', 'AGE_65PLUS']
fe.get_engineered_feature_summary()["newly_engineered_features_count"]     # 8

# imputation and encoding on the raw frame
FeatureEngineer(df.drop(columns=["MEDS", "EMPTY"])) \
    .impute(method="simple_median").encode(method="frequency", columns=["SITE"])
```

## Tests

`tests/test_feature_engineer.py`:

| Test | Checks |
|---|---|
| `test_impute_leaves_all_nan_columns` | Median imputation with an all-NaN column present. |
| `test_target_is_protected_from_encoding_and_scaling` | Protected text and 0/1 columns come out unchanged. |
| `test_create_interactions_default_method`, `test_detect_noise_default_method` | The endgame defaults run (skipped without endgame). |
| `test_feature_engineering_pipeline` (`ppmi`) | On `output/final_reduced_consolidated_data.csv` from a PPMI run (skipped if absent): one-hot, scaling, polynomial and log steps; writes `output/final_engineered_dataset.csv` and `output/feature_engineering_report.html`. |

```bash
pytest tests/test_feature_engineer.py -m "not ppmi" -q
```
