# PIE DataReducer Documentation

## Overview

The `DataReducer` class, located in `pie/data_reducer.py`, is a powerful tool designed to analyze and simplify the complex, multi-modal datasets loaded by the `DataLoader`. Its primary purpose is to inspect each data table *before* a final merge, identify columns that are likely to be uninformative or problematic, and apply these reductions to create a smaller, cleaner, and more memory-efficient dataset.

This pre-merge reduction strategy is crucial for managing the massive feature space that results from combining dozens of PPMI files, making subsequent analysis and feature engineering more tractable.

## Key Features

- **Automated Data Profiling**: Analyzes each DataFrame for key quality metrics, including missing value percentages, constant-value columns, low-variance numeric columns, and high-cardinality ID-like columns.
- **Configurable Reduction Logic**: Provides sensible defaults for column-dropping criteria (e.g., drop if >95% missing) but allows for full customization of these thresholds.
- **Intelligent Merging**: Includes a robust `merge_reduced_data` method that correctly handles nested data dictionaries, prefixes column names to avoid collisions, and ensures the final merged DataFrame has unique patient-visit (`PATNO`, `EVENT_ID`) keys.
- **COHORT Consolidation**: Provides a dedicated function to find all cohort-related columns spread across different files, consolidate them into a single standardized `COHORT` column, and filter for valid participant groups.
- **Comprehensive Reporting**: Generates both a console summary and a detailed HTML report summarizing the reduction process, showing what was dropped and why, and quantifying the reduction in size and complexity.

## The Data Reduction Workflow

The `DataReducer` is designed to be used in a sequential workflow after loading data with the `DataLoader`:

1.  **Load Data**: Use `DataLoader.load(merge_output=False)` to get a dictionary of DataFrames.
2.  **Analyze**: Instantiate `DataReducer` with this dictionary and call `analyze()` to profile the data and get drop suggestions.
3.  **Reduce**: Call `apply_drops()` to create a new, smaller dictionary of DataFrames.
4.  **Merge**: Call `merge_reduced_data()` on the reduced dictionary to create a single, wide-format DataFrame.
5.  **Consolidate**: Call `consolidate_cohort_columns()` on the merged DataFrame to clean up the cohort information.

## API Reference

### `DataReducer(data_dict, config)`

The constructor for the class.

```python
def __init__(self, data_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
```

#### Parameters

- **`data_dict`** `(Dict)`: The dictionary of data returned by `DataLoader.load(merge_output=False)`.
- **`config`** `(Dict, optional)`: A dictionary to override the default analysis configuration. See default config below.

#### Default Configuration

```python
DEFAULT_CONFIG = {
    "missing_threshold": 0.95,        # Drop if > 95% of values are missing.
    "single_value_threshold": 1.0,    # Drop if 100% of non-null values are the same.
    "low_variance_threshold": 0.01,   # For numeric columns, drop if standard deviation is less than this.
    "high_cardinality_ratio": 0.9,    # Drop if (unique values / rows) > 0.9 (potential ID columns).
    "common_metadata_cols": [...],    # A list of known metadata columns to drop.
    "check_low_variance_numeric": True,
    "check_high_cardinality": False   # Disabled by default as it can be risky.
}
```

---

### Main Methods

#### `analyze()`

Performs a full analysis of all DataFrames in the data dictionary.

- **Returns** `(Dict)`: A detailed report dictionary containing summary statistics and drop suggestions for every data table.

#### `get_drop_suggestions()`

A convenience method to extract only the drop suggestions from a full analysis report.

- **Returns** `(Dict[str, List[str]])`: A dictionary where keys are the table names (e.g., `'motor_assessments'`, `'medical_history.Concomitant_Medication'`) and values are lists of column names suggested for dropping.

#### `apply_drops(drop_suggestions)`

Applies the suggested column drops to the original data, returning a new, reduced data dictionary.

- **Parameters**:
    - **`drop_suggestions`** `(Dict)`: The dictionary of columns to drop, typically from `get_drop_suggestions()`.
- **Returns** `(Dict)`: A new, deep-copied dictionary of DataFrames with the specified columns removed.

#### `merge_reduced_data(reduced_data_dict, output_filename)`

Merges the DataFrames from a (typically reduced) dictionary into a single, wide-format DataFrame.

- **Parameters**:
    - **`reduced_data_dict`** `(Dict)`: The dictionary of DataFrames to merge.
    - **`output_filename`** `(str, optional)`: If a path is provided, the final merged DataFrame will be saved as a CSV.
- **Returns** `(pd.DataFrame)`: The final, merged DataFrame.

#### `consolidate_cohort_columns(dataframe)`

Finds all columns related to cohort information in a DataFrame, consolidates them into a single `COHORT` column, standardizes the values (e.g., "PD" -> "Parkinson's Disease"), and filters the DataFrame to include only valid, specified cohorts.

- **Parameters**:
    - **`dataframe`** `(pd.DataFrame)`: The DataFrame to process (typically the output of `merge_reduced_data`).
- **Returns** `(pd.DataFrame)`: The processed DataFrame with a clean `COHORT` column.

---

## Practical Usage Example

This example shows the complete, end-to-end workflow from loading to a final, reduced, and consolidated DataFrame.

```python
from pie.data_loader import DataLoader
from pie.data_reducer import DataReducer

# --- 1. Load Data ---
# Always start by loading data as a dictionary
print("Step 1: Loading data...")
data_dict = DataLoader.load(merge_output=False)
print(f"Loaded {len(data_dict)} modalities.")

# --- 2. Initialize and Analyze ---
# Optionally, define a custom configuration
custom_config = {
    "missing_threshold": 0.90, # Be more aggressive with missing values
    "check_high_cardinality": True # Also check for ID-like columns
}
reducer = DataReducer(data_dict, config=custom_config)
print("\nStep 2: Analyzing data for potential reduction...")
analysis_report = reducer.analyze()

# (Optional) Print a simple text summary to the console
print(reducer.generate_report_str(analysis_report))

# --- 3. Get Suggestions and Reduce ---
print("\nStep 3: Applying drop suggestions...")
drop_suggestions = reducer.get_drop_suggestions(analysis_report)
reduced_data_dict = reducer.apply_drops(drop_suggestions)
print("Data reduction complete.")

# --- 4. Merge the Reduced Data ---
print("\nStep 4: Merging the reduced data dictionary...")
merged_df = reducer.merge_reduced_data(
    reduced_data_dict,
    output_filename="./output/merged_after_reduction.csv"
)
print(f"Merged DataFrame shape: {merged_df.shape}")

# --- 5. Consolidate COHORT Information ---
print("\nStep 5: Consolidating and cleaning the COHORT column...")
final_df = reducer.consolidate_cohort_columns(merged_df)
print(f"Final DataFrame shape after COHORT consolidation: {final_df.shape}")

# Save the final, cleaned DataFrame
final_df.to_csv("./output/final_clean_data.csv", index=False)
print("\nWorkflow complete. Final clean data saved to './output/final_clean_data.csv'")

```

---

## How to Run the Test Script

A comprehensive test script, `tests/test_data_reducer.py`, is provided to demonstrate the entire reduction workflow and generate a detailed HTML report.

### Prerequisites

1.  Ensure you have a local copy of the PPMI dataset.
2.  The test script expects the data to be in a folder named `PPMI` in the root of the project directory. If your data is located elsewhere, you must edit the `data_dir` variable in the test script.

    ```python
    # Inside tests/test_data_reducer.py
    data_dir = "./PPMI"  # <-- CHANGE THIS PATH IF YOUR DATA IS ELSEWHERE
    ```

### Running the Script

No special parameters are needed. Simply navigate to the root directory of the PIE project in your terminal and run the script:

```bash
python3 tests/test_data_reducer.py
```

### What the Test Does and What to Expect

The test script executes the full workflow described above:

1.  **Loads** all data modalities using `DataLoader`.
2.  **Summarizes** the initial state (number of tables, columns, size in MB).
3.  **Analyzes** the data using `DataReducer` with default settings.
4.  **Reduces** the data by applying the drop suggestions.
5.  **Summarizes** the reduced state, showing the percentage decrease in size and complexity.
6.  **Merges** the reduced data into a single DataFrame.
7.  **Consolidates** the `COHORT` column in the final merged DataFrame.
8.  **Saves** two key outputs to the `output/` directory:
    *   `final_reduced_consolidated_data.csv`: The final, analysis-ready dataset.
    *   `data_reduction_report.html`: A detailed HTML report.
9.  **Opens the Report**: The script will attempt to automatically open `data_reduction_report.html` in your default web browser for immediate inspection.

The console output will provide a running commentary on each step, including size comparisons, making it easy to see the impact of the reduction process.
