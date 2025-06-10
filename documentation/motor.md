# Motor Assessment Loader Documentation

## Overview

The `motor_loader.py` module provides a streamlined way to load and consolidate motor assessment data from the Parkinson's Progression Markers Initiative (PPMI) dataset. Its primary purpose is to find all relevant motor assessment CSV files within a specified directory, merge them into a single, comprehensive pandas DataFrame, and clean the data to ensure data integrity.

## Key Features

- Loads motor assessment data from a predefined list of file prefixes.
- Recursively searches for files within the target directory.
- Merges individual CSVs into a single DataFrame based on patient (`PATNO`) and visit (`EVENT_ID`) identifiers.
- Intelligently resolves column name conflicts that occur during merging (e.g., `COLUMN_x`, `COLUMN_y`).
- Ensures unique rows per patient-visit pair by aggregating data from duplicate entries. Conflicting values are preserved and pipe-separated (`|`).
- Provides detailed logging for transparency on which files are loaded, how data is merged, and how duplicates are resolved.

## Supported Motor Assessments

The loader is configured to search for files that begin with the following prefixes:

- `Gait_Data___Arm_swing`
- `Gait_Substudy_Gait_Mobility_Assessment`
- `MDS-UPDRS_Part_I`
- `MDS-UPDRS_Part_II`
- `MDS-UPDRS_Part_III`
- `MDS-UPDRS_Part_IV`
- `Modified_Schwab`
- `Neuro_QoL`
- `Participant_Motor_Function`

## Function

The module exposes a single primary function for loading data.

### `load_ppmi_motor_assessments(folder_path: str)`

Loads, merges, and cleans all motor assessment CSV files from a given folder.

**Parameters:**
- `folder_path (str)`: The path to the directory containing motor assessment files. Typically, this would be a path like `./PPMI/Motor___MDS-UPDRS`. The function will search this directory and all its subdirectories for matching CSV files.

**Returns:**
- `pandas.DataFrame`: A single DataFrame containing the merged and cleaned data from all found motor assessment files. If no files are found or an error occurs, it returns an empty DataFrame.

## Basic Usage Example

```python
import pandas as pd
from pie.motor_loader import load_ppmi_motor_assessments

# Define the path to your PPMI motor assessments folder
# This folder should contain the various CSV files for motor assessments.
motor_data_path = "./PPMI/Motor___MDS-UPDRS"

# Load the data
print(f"Loading motor assessments from: {motor_data_path}")
df_motor = load_ppmi_motor_assessments(motor_data_path)

if not df_motor.empty:
    # Display the shape and first few rows of the loaded data
    print(f"Successfully loaded data. Shape: {df_motor.shape}")
    print("First 5 rows of the merged motor assessments data:")
    print(df_motor.head())

    # Example: Inspect a specific patient's data
    # PATNO is loaded as a string to preserve formatting.
    if 'PATNO' in df_motor.columns and '3001' in df_motor['PATNO'].values:
        print("\nData for PATNO 3001:")
        print(df_motor[df_motor['PATNO'] == '3001'].head())

    # Save the merged data to a new CSV file for further analysis
    output_path = "merged_motor_assessments.csv"
    df_motor.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
else:
    print("Loading failed or no data was found. Please check logs and folder path.")
```

## Post-Loading Analysis Example

Once you have loaded the data into a DataFrame, you can perform standard analysis using pandas. For example, you can calculate summary statistics for a specific assessment.

**Note:** The following example assumes that columns like `NP3TOT` (MDS-UPDRS Part III Total Score) and `EVENT_ID` exist in your merged data. Column names will depend on the specific CSV files present in your data folder.

```python
import pandas as pd
from pie.motor_loader import load_ppmi_motor_assessments

# Load the data
motor_data_path = "./PPMI/Motor___MDS-UPDRS"
df_motor = load_ppmi_motor_assessments(motor_data_path)

if not df_motor.empty and 'NP3TOT' in df_motor.columns and 'EVENT_ID' in df_motor.columns:
    print("Performing a simple analysis on the loaded data...")

    # The data in 'NP3TOT' might be strings if aggregation occurred.
    # We need to convert it to a numeric type, handling errors.
    # For this simple analysis, we take the first value if data is pipe-separated.
    def clean_and_convert_to_numeric(series):
        series_str = series.astype(str).str.split('|').str[0]
        return pd.to_numeric(series_str, errors='coerce')

    df_motor['NP3TOT_numeric'] = clean_and_convert_to_numeric(df_motor['NP3TOT'])

    # Remove rows where the score could not be converted
    df_analysis = df_motor.dropna(subset=['NP3TOT_numeric', 'EVENT_ID'])

    # Calculate the average motor score by visit
    avg_score_by_visit = df_analysis.groupby('EVENT_ID')['NP3TOT_numeric'].mean().reset_index()

    # Sort by a typical visit order (simple alphabetical sort here)
    avg_score_by_visit = avg_score_by_visit.sort_values('EVENT_ID')
    
    print("\nAverage MDS-UPDRS Part III Score by Visit:")
    print(avg_score_by_visit)

else:
    print("\nSkipping analysis example: 'NP3TOT' or 'EVENT_ID' not found in the DataFrame.")
```

## Implementation Details

The loading process follows several key steps to ensure data quality and consistency:

1.  **File Discovery**
    The loader scans the provided `folder_path` recursively for any CSV files. It then filters this list to include only those files whose names start with one of the prefixes defined in its internal `FILE_PREFIXES` list. This ensures only relevant motor assessments are loaded.

2.  **Iterative Merging**
    The function initializes a DataFrame with the first valid CSV file it finds. It then iteratively merges subsequent files into this main DataFrame. The merge is performed as an `outer` join on `["PATNO", "EVENT_ID"]`. If a file lacks `EVENT_ID`, the merge is done on `PATNO` alone.

3.  **Handling Merge Suffixes**
    When pandas merges two DataFrames that share column names (other than the merge keys), it appends suffixes (`_x` and `_y`) to distinguish them. A utility function post-processes the merged DataFrame to handle these cases:
    - If a column exists with a suffix (e.g., `COL_x`) but its counterpart (`COL_y`) does not, it is renamed to its base name (`COL`).
    - If both `COL_x` and `COL_y` exist, their values are combined into a single base column (`COL`). The logic prioritizes non-empty values. If both columns have different, non-empty values, they are concatenated with a pipe separator (e.g., `"value1|value2"`).

4.  **Ensuring Unique Patient-Visit Rows**
    After all files are merged, some (`PATNO`, `EVENT_ID`) pairs might appear in multiple rows. This can happen if different source files contain data for the same patient visit. A utility function consolidates these duplicates:
    - It groups the DataFrame by `["PATNO", "EVENT_ID"]`.
    - Within each group, for every other column, it joins multiple unique non-null values with a pipe separator (`|`).
    - If only one unique value exists, it is used for the consolidated row.
    This process guarantees that the final DataFrame has at most one row for each unique (`PATNO`, `EVENT_ID`) pair.

## Data Dictionary

Common fields found in the motor assessment source files include:

| Field | Description | Example Values |
|-------|-------------|----------------|
| PATNO | Patient identifier | 1001, 1002, ... |
| EVENT_ID | Visit identifier | BL (baseline), V01, V02, ... |
| NP1TOT | Total score for MDS-UPDRS Part I (non-motor aspects) | 0-52 |
| NP2TOT | Total score for MDS-UPDRS Part II (motor aspects of daily living) | 0-52 |
| NP3TOT | Total score for MDS-UPDRS Part III (motor examination) | 0-132 |
| NP4TOT | Total score for MDS-UPDRS Part IV (motor complications) | 0-24 |
| HY_STAGE | Modified Hoehn and Yahr Stage | 0-5 |
| MSEADLG | Schwab & England ADL Score | 0-100% |

*Note: Column names like `NP1TOT` may vary slightly between different versions of the PPMI dataset.*

## Troubleshooting

- **Empty DataFrame is Returned:** If the function returns an empty DataFrame, check the following:
    - Verify that the `folder_path` you provided is correct and that the directory exists.
    - Ensure the directory contains CSV files whose names begin with one of the expected prefixes (e.g., `MDS-UPDRS_Part_III...`).
    - Check the application logs. The loader will log warnings if it cannot find any matching files or if files fail to load.

- **Pipe-Separated Values (`|`):** If you see values like `"1|2"` in your data, it means that during the merge and aggregation process, different source files provided different values for the same column, patient, and visit. The loader preserves both values to prevent data loss. The logs may provide more context on which columns were affected.

- **Missing PATNO or EVENT_ID Columns:** The loader expects a `PATNO` column in every file and an `EVENT_ID` column in most. It will log a warning and skip any file that is missing `PATNO`. Merging behavior may change if `EVENT_ID` is missing, which is also logged.