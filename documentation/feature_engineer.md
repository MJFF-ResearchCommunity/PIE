# PIE FeatureEngineer Documentation

## Overview

The `FeatureEngineer` class, located in `pie/feature_engineer.py`, is the final step in the data preparation pipeline before machine learning. It takes a clean, merged DataFrame (typically the output of the `DataReducer` workflow) and applies a series of transformations to create features suitable for modeling.

This class provides a flexible, chainable interface for common feature engineering tasks such as one-hot encoding categorical variables, scaling numeric features, generating polynomial interactions, and handling special data formats.

## Key Features

- **Chainable Methods**: Each transformation method returns the `FeatureEngineer` instance, allowing for a clean, sequential application of steps.
- **Advanced One-Hot Encoding**: Includes auto-identification of categorical columns, handling for rare categories, and explicit column exclusion.
- **Numeric Transformations**: Supports standard and min-max scaling, polynomial/interaction feature generation, and distribution transformations (log, Box-Cox, etc.).
- **Special Format Handling**: Provides strategies for converting pipe-separated (`|`) string columns into usable features.
- **Extensibility**: Allows for the application of any custom user-defined function to a column.
- **Automated Reporting**: The test script generates a detailed HTML report summarizing the entire feature engineering process, including initial and final data shapes and a breakdown of all newly created features.

## The Feature Engineering Workflow

The `FeatureEngineer` is designed to be the last step in data preparation:

1.  **Get Clean Data**: Start with a single, clean DataFrame, ideally the `final_reduced_consolidated_data.csv` produced by running `tests/test_data_reducer.py`.
2.  **Instantiate**: Create an instance of the `FeatureEngineer` with this DataFrame.
3.  **Chain Transformations**: Call the desired feature engineering methods sequentially.
4.  **Get Final DataFrame**: Use `get_dataframe()` to retrieve the final, model-ready DataFrame.

## API Reference

### `FeatureEngineer(dataframe)`

The constructor for the class.

```python
def __init__(self, dataframe: pd.DataFrame):
```

- **Parameters**:
    - **`dataframe`** `(pd.DataFrame)`: The input DataFrame to be transformed.

---

### Transformation Methods

These methods modify the internal DataFrame and return `self` to allow for chaining.

#### `one_hot_encode(...)`

Performs one-hot encoding on categorical columns.

- **Parameters**:
    - **`columns`** `(List[str], optional)`: A specific list of columns to encode. If `None`, columns are auto-identified based on `dtype` and `auto_identify_threshold`.
    - **`dummy_na`** `(bool, default=False)`: If `True`, creates a separate column for `NaN` values.
    - **`drop_first`** `(bool, default=False)`: Whether to get `k-1` dummies out of `k` categorical levels by removing the first level. Useful for reducing multicollinearity.
    - **`max_categories_to_encode`** `(int, default=20)`: A safety threshold to prevent encoding columns with too many unique values.
    - **`min_frequency_for_category`** `(float, optional)`: If set (e.g., `0.01`), categories that appear less frequently than this proportion are grouped into a single `_OTHER_` category before encoding.
    - **`ignore_for_ohe`** `(List[str], optional)`: A list of columns to explicitly ignore. `PATNO`, `EVENT_ID`, and `COHORT` are automatically ignored.
    - **`auto_identify_threshold`** `(int, default=50)`: When `columns` is `None`, only categorical columns with fewer than this many unique values will be encoded.

#### `handle_pipe_separated_column(column_name, strategy, ...)`

Processes a column containing strings with `|` delimiters.

- **Parameters**:
    - **`column_name`** `(str)`: The name of the column to process.
    - **`strategy`** `(str, default='multi_hot')`: The method to use:
        - `'first'`: Takes only the first value before the first `|`.
        - `'count'`: Creates a new feature with the count of items.
        - `'multi_hot'`: Creates a binary column for each unique item found in the strings (multi-hot encoding).
    - **`max_unique_values_for_multi_hot`** `(int, default=30)`: A safety limit for the `'multi_hot'` strategy.

#### `scale_numeric_features(columns, scaler_type, ...)`

Applies numeric scaling to specified or auto-identified columns.

- **Parameters**:
    - **`columns`** `(List[str], optional)`: List of columns to scale. If `None`, all numeric columns (except IDs) are scaled.
    - **`scaler_type`** `(str, default='standard')`: The type of scaler to use.
        - `'standard'`: `StandardScaler` (zero mean, unit variance).
        - `'minmax'`: `MinMaxScaler` (scales to a range, typically [0, 1]).

#### `engineer_polynomial_features(columns, degree, ...)`

Generates polynomial and interaction features from numeric columns.

- **Parameters**:
    - **`columns`** `(List[str], optional)`: List of columns to use. If `None`, all numeric columns (except IDs) are used.
    - **`degree`** `(int, default=2)`: The degree of the polynomial. `degree=2` will create features like `a*b`, `a^2`, `b^2`.
    - **`interaction_only`** `(bool, default=False)`: If `True`, only interaction features (e.g., `a*b`) are created, not polynomial features (e.g., `a^2`).

#### `transform_numeric_distribution(column_name, transform_type, ...)`

Applies a mathematical function to a column to transform its statistical distribution, which can help with skewed data.

- **Parameters**:
    - **`column_name`** `(str)`: The column to transform.
    - **`new_column_name`** `(str, optional)`: Name for the new column. If `None`, the transformation is done in-place.
    - **`transform_type`** `(str, default='log')`: The type of transformation: `'log'`, `'sqrt'`, `'box-cox'`, or `'yeo-johnson'`.

#### `apply_custom_transformation(column_name, func, new_column_name)`

Applies a user-provided function to a column.

- **Parameters**:
    - **`column_name`** `(str)`: The name of the column to transform.
    - **`func`** `(Callable)`: A function that accepts a pandas Series and returns a pandas Series.
    - **`new_column_name`** `(str, optional)`: Name for the new column. If `None`, the transformation is done in-place.

---

## Practical Usage Example

This example demonstrates a typical feature engineering pipeline using a chain of commands.

```python
import pandas as pd
from pie.feature_engineer import FeatureEngineer

# Assume 'df_clean' is the DataFrame loaded from the output of the DataReducer workflow
# df_clean = pd.read_csv("output/final_reduced_consolidated_data.csv")

# 1. Initialize the engineer
engineer = FeatureEngineer(df_clean)

# 2. Chain feature engineering steps
engineer.one_hot_encode(
    min_frequency_for_category=0.02, # Group very rare categories
    drop_first=True
).scale_numeric_features(
    scaler_type='standard'
).engineer_polynomial_features(
    columns=['AGE_AT_VISIT', 'some_other_numeric_feature'], # Hypothetical columns
    degree=2,
    interaction_only=True
)

# 3. Get the final, engineered DataFrame
final_df = engineer.get_dataframe()

print(f"Original shape: {df_clean.shape}")
print(f"Final engineered shape: {final_df.shape}")

# 4. Get a summary of what was done
summary = engineer.get_engineered_feature_summary()
import json
print(json.dumps(summary, indent=2))
```

---

## How to Run the Test Script

The test script `tests/test_feature_engineer.py` provides a full, working example of the feature engineering pipeline and generates a summary report.

### Prerequisites

-   The test script **requires the output file from the `DataReducer` workflow**. You must first run `tests/test_data_reducer.py` successfully. The feature engineering test specifically looks for its input file at:
    `output/final_reduced_consolidated_data.csv`

### Running the Script

No special parameters are needed. Navigate to the root of the PIE project in your terminal and run:

```bash
python3 tests/test_feature_engineer.py
```

### What the Test Does and What to Expect

The test script performs a demonstration pipeline on the clean, reduced data:
1.  **Loads** the `final_reduced_consolidated_data.csv` file.
2.  **Applies** a series of feature engineering steps, including one-hot encoding, numeric scaling, and creating polynomial features.
3.  **Logs** the DataFrame shape after each major step to show how the data is transformed.
4.  **Saves** the final, model-ready dataset to `output/final_engineered_dataset.csv`.
5.  **Generates and opens** a detailed HTML report at `output/feature_engineering_report.html`, which summarizes the entire process, including the number of features created by each operation.
