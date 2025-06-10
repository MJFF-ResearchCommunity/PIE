# PIE FeatureSelector Documentation

## Overview

The `FeatureSelector` class, located in `pie/feature_selector.py`, provides a standardized interface for applying various feature selection techniques to a dataset. This is typically the last data preparation step before training a machine learning model. Its purpose is to reduce the number of input features to only those that are most useful, which can lead to simpler models, reduced training times, and sometimes better performance by eliminating noise.

The class is designed to follow the common `scikit-learn` `fit`/`transform` paradigm. This stateful approach is crucial: the selector is "fitted" on the training data to learn which features are important, and then this same fitted selector is used to "transform" (i.e., select the same columns from) both the training and any subsequent test sets, ensuring consistency.

## Key Features

- **Stateful `fit`/`transform` API**: Ensures that feature selection logic learned from the training set is correctly applied to the test set.
- **Wrapper for `scikit-learn`**: Provides a simplified interface for several powerful `scikit-learn` feature selection methods.
- **Supported Methods**:
    - **Univariate Selection**: `k_best` (`SelectKBest`), `fdr` (`SelectFdr`).
    - **Model-Based Selection**: `select_from_model` (`SelectFromModel`), `rfe` (`RFE` - Recursive Feature Elimination).
- **Task-Aware Defaults**: Automatically selects appropriate scoring functions (e.g., `f_classif` for classification) and estimators (e.g., `LogisticRegression`) based on the specified `task_type`, while still allowing for customization.

## API Reference

### `FeatureSelector(method, task_type, ...)`

The constructor for the class. You initialize it by choosing a selection strategy and its parameters.

```python
def __init__(
    self,
    method: str,
    task_type: str,
    k_or_frac: Optional[float] = 0.5,
    alpha_fdr: float = 0.05,
    estimator: Optional[Any] = None,
    scoring_univariate: Optional[Union[str, Callable]] = None,
    random_state: int = 123
):
```

#### Parameters

- **`method`** `(str)`: The core feature selection strategy to use. Options are:
    - `'k_best'`: Selects a fixed number (`k`) of the highest-scoring features.
    - `'fdr'`: Selects features while controlling the False Discovery Rate.
    - `'select_from_model'`: Selects features based on importance weights from an estimator.
    - `'rfe'`: Recursively removes features and builds a model on the remaining ones.

- **`task_type`** `(str)`: The type of machine learning task. This determines the default scoring functions and estimators. Options:
    - `'classification'`
    - `'regression'`

- **`k_or_frac`** `(float, default=0.5)`: The fraction of features to select. Used by `'k_best'` and `'rfe'`. For example, `0.5` will select the top 50% of features.

- **`alpha_fdr`** `(float, default=0.05)`: The p-value threshold for the False Discovery Rate test. Used only by the `'fdr'` method.

- **`estimator`** `(scikit-learn estimator, optional)`: A `scikit-learn` model instance (e.g., `RandomForestClassifier()`) to use for model-based selection (`'select_from_model'`, `'rfe'`). If not provided, a default (`LogisticRegression` or `Lasso`) is used.

- **`scoring_univariate`** `(str or Callable, optional)`: The scoring function for univariate methods (`'k_best'`, `'fdr'`). Can be a callable function or a string like `'f_classif'` or `'mutual_info_regression'`. If not provided, a default (`f_classif` or `f_regression`) is used.

- **`random_state`** `(int, default=123)`: A seed for reproducibility in estimators that have a random component.

---

### `fit(X, y)`

Fits the feature selector on the training data `X` and `y`. This step learns which features to keep.

- **Parameters**:
    - **`X`** `(pd.DataFrame)`: The training input samples (features).
    - **`y`** `(pd.Series)`: The target values for the training samples.
- **Returns** `(self)`: The fitted `FeatureSelector` instance.

### `transform(X)`

Reduces the input DataFrame `X` to only the features that were selected during the `fit` step.

- **Parameters**:
    - **`X`** `(pd.DataFrame)`: The DataFrame to transform (can be training or test data).
- **Returns** `(pd.DataFrame)`: A new DataFrame containing only the selected feature columns.

---

## Practical Usage Example

This example demonstrates the standard workflow for using the `FeatureSelector`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from pie.feature_selector import FeatureSelector

# Assume 'df_engineered' is your final, engineered dataset
# df_engineered = pd.read_csv("output/final_engineered_dataset.csv")
# For this example, let's create a dummy DataFrame
data = pd.DataFrame({
    'feature1': range(20),
    'feature2': [0]*10 + [1]*10,
    'useless_feature': [5]*20,
    'feature4': np.random.rand(20) * 10,
    'target': ['A']*10 + ['B']*10
})
X = data.drop('target', axis=1)
y = data['target']

# 1. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Original shape of X_train: {X_train.shape}")

# 2. Initialize the FeatureSelector
# We want to select the top 50% of features for a classification task.
selector = FeatureSelector(
    method='k_best',
    task_type='classification',
    k_or_frac=0.5
)

# 3. Fit the selector on the training data ONLY
selector.fit(X_train, y_train)

# You can inspect the selected features
print(f"Selected features: {selector.selected_feature_names_}")

# 4. Transform both the training and test sets
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"\nShape of X_train after selection: {X_train_selected.shape}")
print(f"Shape of X_test after selection: {X_test_selected.shape}")

# The columns of the transformed sets will be identical
assert all(X_train_selected.columns == X_test_selected.columns)
```

---

## How to Run the Tests

The test script `tests/test_feature_selector.py` provides a working example of a feature selection pipeline and generates a summary report.

### Prerequisites

- The test script **requires the output from the feature engineering workflow**. You must first run `tests/test_feature_engineer.py` to generate the input file:
  `output/final_engineered_dataset.csv`

### Running the Script

The test demonstrates a sequential, two-step feature selection process. From the root of the PIE project, run:

```bash
python3 tests/test_feature_selector.py
```

### What the Test Does and What to Expect

The test script in `tests/test_feature_selector.py` actually contains two distinct tests:

1.  **`test_feature_selector_class_with_real_data()`**: This is a direct unit test of the `FeatureSelector` class. It loads the engineered data, splits it, and confirms that the `FeatureSelector` class can be fitted and used to transform train/test sets correctly.
2.  **`test_feature_selection_workflow()`** (The main function run by the script): This function demonstrates a more complex, custom feature selection pipeline. It does the following:
    *   Loads the engineered dataset.
    *   Splits the data into training and test sets.
    *   Applies a `VarianceThreshold` to remove low-variance features.
    *   Applies `SelectFdr` to the remaining features to select the most statistically significant ones.
    *   **Saves** the final selected training and test datasets to `output/selected_train_data.csv` and `output/selected_test_data.csv`.
    *   **Generates and opens** a detailed HTML report at `output/feature_selection_report.html`, which documents the entire process, showing how many features were removed at each step.

Running this script is a great way to see an example of a complete feature selection pipeline in action.
