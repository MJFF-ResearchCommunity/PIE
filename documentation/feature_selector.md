# FeatureSelector (`pie.feature_selector`)

A fit/transform wrapper over sklearn and endgame feature selectors. It is fitted on the training
split only and then applied unchanged to the test split, so the choice of features never sees test
labels. Input must already be numeric and free of `NaN`; the selectors do not impute.

Stage 3 of the [pipeline](pipeline.md) fits it on the training participants; `--fs-param` becomes
`alpha_fdr` for `fdr` and `k_or_frac` for every other method.

## Constructor

```python
FeatureSelector(
    method: str,
    task_type: str,                        # "classification" or "regression"
    k_or_frac: float | None = 0.5,
    alpha_fdr: float = 0.05,
    estimator=None,
    scoring_univariate: str | Callable | None = None,
    random_state: int = 123,
    **kwargs,                              # forwarded to endgame selectors only
)
```

| Argument | Used by | Meaning |
|---|---|---|
| `k_or_frac` | `k_best`, `rfe`, `mrmr`, `relief` | Fraction of the (non-constant) features to keep; at least 1. Must not be `None` for these methods. |
| `alpha_fdr` | `fdr` | Benjamini–Hochberg false-discovery rate. |
| `estimator` | `select_from_model`, `rfe`, `permutation` | Defaults to `LogisticRegression` (classification) or `Lasso` (regression). |
| `scoring_univariate` | `k_best`, `fdr` | Callable, or one of `f_classif`, `f_regression`, `mutual_info_classif`, `mutual_info_regression`. Defaults to `f_classif` / `f_regression` by `task_type`. |
| `random_state` | model-based and endgame selectors | Seed. |

## Methods

`SUPPORTED_METHODS` maps each name to whether it needs endgame (`pip install endgame-ml[tabular]`).
An unknown name raises `ValueError` at `fit`, not at construction.

| `method` | Selector | Keeps |
|---|---|---|
| `k_best` | `SelectKBest` | top `k_or_frac` by univariate score |
| `fdr` | `SelectFdr` | features whose univariate p-value passes FDR at `alpha_fdr` |
| `select_from_model` | `SelectFromModel(threshold="median")` | features with above-median importance (about half) |
| `rfe` | `RFE` | `k_or_frac` by recursive elimination |
| `mrmr`, `relief` | endgame `MRMRSelector`, `ReliefFSelector` | `k_or_frac` |
| `boruta`, `shap`, `adversarial`, `permutation`, `genetic`, `stability`, `knockoff`, `null_importance`, `tree_importance`, `correlation` | endgame selectors of the same name | decided by the selector; tune through `**kwargs` |

| Member | Meaning |
|---|---|
| `fit(X, y)` | Drops constant columns, fits the selector, sets `selected_feature_names_`. Returns `self`. |
| `transform(X)` | `X[selected_feature_names_]`. Raises `RuntimeError` before `fit`; the columns must exist in `X`. |
| `selected_feature_names_` | List of kept column names. |
| `FeatureSelector.available_methods(endgame_only=False)` | Method names usable in this environment. |
| `ENDGAME_FS_AVAILABLE` | Module flag: endgame selectors importable. |

`fit` removes zero-variance columns before the selector sees them. Univariate scorers such as
`f_classif` divide by within-class variance, so a constant column yields a `NaN` F-statistic and a
warning; under FDR a `NaN` also disturbs the ranking of its neighbours. Constant columns are never
in `selected_feature_names_`.

## Example

Runnable with synthetic data:

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pie.feature_selector import FeatureSelector

X, y = make_classification(n_samples=200, n_features=20, n_informative=4, random_state=0)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
X["const"] = 1.0                                         # dropped before selection
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

sel = FeatureSelector(method="fdr", task_type="classification", alpha_fdr=0.05).fit(X_tr, y_tr)
sel.selected_feature_names_                              # e.g. ['f8', 'f9']
X_te_sel = sel.transform(X_te)                           # same columns, fitted on train only

top = FeatureSelector("k_best", "classification", k_or_frac=0.25).fit(X_tr, y_tr)
len(top.selected_feature_names_)                         # 5 = 25 % of the 20 non-constant features

FeatureSelector.available_methods(endgame_only=True)     # endgame methods installed here
```

## Tests

Both tests read files produced from PPMI data; they are marked `ppmi` and skipped when the files are
missing. The pipeline's use of `FeatureSelector` is covered synthetically by
`tests/test_pipeline.py::test_fraction_based_fs_methods`.

| Test | Input | What it does |
|---|---|---|
| `test_feature_selector_class_with_real_data` | `output/test_pipeline_run/final_engineered_dataset.csv` (from `tests/test_pipeline.py`) | Fits and transforms with `FeatureSelector`, asserts selected names match the transformed columns. |
| `test_feature_selection_workflow` | `output/final_engineered_dataset.csv` | Its own `VarianceThreshold(0.01)` → `SelectFdr(0.05)` sequence, not `FeatureSelector`. Writes `output/selected_train_data.csv`, `output/selected_test_data.csv` and `output/feature_selection_report.html`. |

```bash
pytest tests/test_feature_selector.py -m ppmi -q
```
