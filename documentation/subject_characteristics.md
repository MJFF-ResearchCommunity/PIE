# Subject Characteristics Loader Documentation

## Overview

The `sub_char_loader.py` module provides a robust mechanism for loading and consolidating subject characteristic data from the PPMI dataset. This includes demographics, cohort information, family history, and genetic data. The loader is designed to find all relevant CSV files, merge them into a single, clean pandas DataFrame, and resolve data conflicts that arise from combining multiple sources.

## Key Features

- Loads data from a predefined list of subject characteristic file prefixes.
- Recursively searches for files within the target directory.
- Merges individual CSVs into a single DataFrame based on patient (`PATNO`) and, where available, visit (`EVENT_ID`) identifiers.
- Intelligently resolves column name conflicts from merges (e.g., `COLUMN_x`, `COLUMN_y`).
- Ensures unique rows per patient-visit pair by aggregating data from duplicate entries. Conflicting values are preserved and pipe-separated (`|`).
- Provides detailed logging for transparency on file loading, merging, and data cleaning operations.

## Supported Subject Characteristics Files

The loader is configured to search for files that begin with the following prefixes:

- `Age_at_visit`
- `Demographics`
- `Family_History`
- `iu_genetic_consensus`
- `Participant_Status`
- `PPMI_PD_Variants`
- `PPMI_Project_9001`
- `Socio-Economics`
- `Subject_Cohort_History`

## Function

The module exposes a single primary function for loading data.

### `load_ppmi_subject_characteristics(folder_path: str)`

Loads, merges, and cleans all subject characteristic CSV files from a given folder.

**Parameters:**
- `folder_path (str)`: The path to the directory containing subject characteristic files. Typically, this would be `./PPMI/_Subject_Characteristics`. The function will search this directory and its subdirectories.

**Returns:**
- `pandas.DataFrame`: A single DataFrame containing the merged and cleaned data from all found files. If no files are found or an error occurs, it returns an empty DataFrame.

## Practical Usage Examples

### Basic Loading and Inspection

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Define the path to your PPMI subject characteristics folder
sub_char_path = "./PPMI/_Subject_Characteristics"

# Load the data
print(f"Loading subject characteristics from: {sub_char_path}")
df_subjects = load_ppmi_subject_characteristics(sub_char_path)

if not df_subjects.empty:
    # Display the shape and first few rows of the loaded data
    print(f"Successfully loaded data. Shape: {df_subjects.shape}")
    print("First 5 rows of the merged subject data:")
    print(df_subjects.head())

    # Example: Check for key demographic columns
    demographic_cols = ['PATNO', 'SEX', 'HISPLAT', 'RACE', 'COHORT']
    available_cols = [col for col in demographic_cols if col in df_subjects.columns]
    if available_cols:
        print("\nSample demographic data:")
        print(df_subjects[available_cols].head())

    # Save the merged data to a new CSV file for further analysis
    output_path = "merged_subject_characteristics.csv"
    df_subjects.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
else:
    print("Loading failed or no data was found. Please check logs and folder path.")
```

### Analysis Example: Demographic Breakdown

Once you have loaded the data, you can analyze it. This example demonstrates how to get a demographic breakdown of the study participants by cohort.

**Note:** This example assumes columns like `COHORT`, `SEX`, and `AGE_AT_VISIT` exist in your merged data. The exact names can vary.

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Load the data
sub_char_path = "./PPMI/_Subject_Characteristics"
df_subjects = load_ppmi_subject_characteristics(sub_char_path)

if not df_subjects.empty and 'COHORT' in df_subjects.columns and 'AGE_AT_VISIT' in df_subjects.columns and 'SEX' in df_subjects.columns:
    print("Performing a demographic analysis by cohort...")

    # For a per-patient analysis, we can drop duplicates based on PATNO
    # This gives us one row per patient, taking the first available record
    df_unique_patients = df_subjects.drop_duplicates(subset=['PATNO'])

    # Group by cohort and get statistics
    cohort_analysis = df_unique_patients.groupby('COHORT').agg(
        num_patients=('PATNO', 'count'),
        sex_distribution=('SEX', lambda s: s.value_counts().to_dict()),
        avg_age=('AGE_AT_VISIT', 'mean')
    ).reset_index()

    print("\nDemographic breakdown by cohort:")
    print(cohort_analysis)

else:
    print("\nSkipping analysis: 'COHORT', 'AGE_AT_VISIT', or 'SEX' column not found in the DataFrame.")
```

### Integration with Other PIE Loaders

The data loaded by this module is foundational and is often merged with other datasets for more comprehensive analysis.

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics
from pie.motor_loader import load_ppmi_motor_assessments

# Load subject characteristics
data_path = "./PPMI"
df_subjects = load_ppmi_subject_characteristics(f"{data_path}/_Subject_Characteristics")

# Load motor assessment data
df_motor = load_ppmi_motor_assessments(f"{data_path}/Motor___MDS-UPDRS")

# Merge the two datasets on patient and visit identifiers
if not df_subjects.empty and not df_motor.empty:
    merge_keys = ['PATNO', 'EVENT_ID']
    if all(key in df_subjects.columns and key in df_motor.columns for key in merge_keys):
        df_combined = pd.merge(df_subjects, df_motor, on=merge_keys, how="inner")
        
        print(f"Combined data shape: {df_combined.shape}")
        print("A few columns in combined data:", df_combined.columns.tolist()[:10])
    else:
        print("Could not merge dataframes, 'PATNO' and/or 'EVENT_ID' missing from one.")

```

## Implementation Details

The loading process follows several key steps to ensure data quality and consistency:

1.  **File Discovery**
    The loader recursively scans the `folder_path` for CSV files, filtering them based on the `FILE_PREFIXES` list.

2.  **Iterative Merging**
    The function loads the first matching file and then iteratively merges subsequent files. The merge logic is adaptive:
    - If both DataFrames have `PATNO` and `EVENT_ID`, it performs an outer join on both keys.
    - If one DataFrame lacks `EVENT_ID` (e.g., it contains static demographic data), the join is performed on `PATNO` only. This effectively broadcasts the static information across all visits for that patient.

3.  **Handling Merge Suffixes**
    Pandas adds `_x` and `_y` suffixes to overlapping columns during merges. A utility function resolves these:
    - If only one suffixed column exists (e.g., `COL_x`), it's renamed to its base name (`COL`).
    - If both `COL_x` and `COL_y` exist, their values are combined. If values are different and non-empty, they are joined with a pipe (`|`) to prevent data loss.

4.  **Ensuring Unique Rows**
    After all files are merged, a final aggregation step ensures each row is unique for its identifying keys.
    - If `EVENT_ID` is present, it groups by `("PATNO", "EVENT_ID")` to consolidate duplicate rows.
    - If `EVENT_ID` is not present, it checks for duplicate `PATNO`s and consolidates them.
    - In both cases, conflicting values in other columns are joined with a pipe separator (`|`).

## Data Dictionary

Common fields found in the subject characteristic source files include:

| Field | Description | Example Values |
|-------|-------------|----------------|
| PATNO | Patient identifier | 1001, 1002, ... |
| EVENT_ID | Visit identifier | BL (baseline), V01, ... |
| COHORT | Study cohort | PD, HC (Healthy Control), Prodromal |
| SEX | Biological sex | Male, Female |
| RACE | Participant's race | White, Black or African American, ... |
| HISPLAT | Hispanic/Latino ethnicity | Yes, No |
| EDUCYRS | Years of education | 16, 18, ... |
| HANDED | Dominant hand | Right, Left |
| AGE_AT_VISIT| Age of participant at the time of visit | 65.4, 72.1, ... |

*Note: Exact column names can vary. Always inspect `df.columns` on your loaded data.*

## Best Practices

- **Verify Paths:** Always double-check that the `folder_path` points to the correct directory containing the subject characteristic CSV files.
- **Check for Emptiness:** After loading, verify that the returned DataFrame is not empty before attempting analysis.
- **Inspect Columns:** Since the final columns depend on the source files, always inspect `df.columns` to see what data is available.
- **Handle Pipe-Separated Data:** Be aware that some columns may contain pipe-separated values. You may need to parse these strings for specific analyses.

## Troubleshooting

- **Empty DataFrame Returned:** If the function returns an empty DataFrame, check that:
    - The `folder_path` is correct.
    - The directory contains CSV files starting with one of the recognized prefixes.
    - The application logs for any warnings about files that could not be loaded.

- **Pipe-Separated Values (`|`):** If you find values like `"value1|value2"`, it means that different source files contained different information for the same patient, visit, and data field. The loader preserves both values.

- **Missing PATNO Column:** The loader expects a `PATNO` column in every file and will log a warning and skip files that lack it.
