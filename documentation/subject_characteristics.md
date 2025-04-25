# Subject Characteristics Loader Documentation

## Overview

The `sub_char_loader.py` module provides functionality for loading and processing PPMI Subject Characteristics data. This module specializes in loading various demographic, genetic, socioeconomic, and cohort information files, handling their integration, and resolving duplicate columns that may arise during merging.

## Key Features

- Load subject characteristic data from multiple file types
- Merge data based on available join keys (PATNO and/or EVENT_ID)
- Handle duplicate columns through value combination logic
- Preserve data integrity when merging from different source files
- Consistent logging for debugging and monitoring

## Module Components

### Constants

#### `FILE_PREFIXES`

A list of filename prefixes to identify relevant subject characteristic files:

```python
FILE_PREFIXES = [
    "Age_at_visit",
    "Demographics",
    "Family_History",
    "iu_genetic_consensus",
    "Participant_Status",
    "PPMI_PD_Variants",
    "PPMI_Project_9001",
    "Socio-Economics",
    "Subject_Cohort_History"
]
```

### Functions

#### `deduplicate_columns(df, duplicate_columns)`

Handles column duplication that may result from merging multiple dataframes.

**Parameters:**
- `df`: The merged DataFrame containing possible duplicates
- `duplicate_columns`: List of column base names that may have duplicates

**Returns:**
- Updated DataFrame with deduplicated columns

**Behavior:**
- For each column name in `duplicate_columns`, checks for `<col>_x` and `<col>_y` variants
- Combines these columns using the following logic:
  1. If both values are NaN/empty, result is empty
  2. If one is empty, use the non-empty value
  3. If both are non-empty and identical, use that value
  4. If both are non-empty and different, combine with a pipe separator (`|`)
- Removes the `<col>_x` and `<col>_y` columns after combining

**Example:**

```python
import pandas as pd
from pie.sub_char_loader import deduplicate_columns

# Create a DataFrame with duplicate columns
df = pd.DataFrame({
    'PATNO': [1, 2, 3],
    'COHORT_x': ['PD', None, 'HC'],
    'COHORT_y': ['PD', 'HC', None]
})

# Deduplicate the COHORT column
df = deduplicate_columns(df, ['COHORT'])

print(df)
# Output:
#    PATNO COHORT
# 0      1    PD
# 1      2    HC
# 2      3    HC
```

#### `load_ppmi_subject_characteristics(folder_path)`

Loads and merges CSV files containing subject characteristic data.

**Parameters:**
- `folder_path`: Path to the `_Subject_Characteristics` folder containing CSV files

**Returns:**
- A merged DataFrame with all subject characteristics data

**Behavior:**
- Searches for CSV files matching any of the prefixes in `FILE_PREFIXES`
- Determines how to merge based on available columns:
  - If both DataFrames have `PATNO` and `EVENT_ID`, merges on both
  - Otherwise, merges only on `PATNO`, replicating static data across all events
- Uses outer joins to preserve all data from both sides
- Deduplicates specified columns after merging
- Returns an empty DataFrame if no files are successfully loaded

**Example:**

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Load all subject characteristics data
data_path = "./PPMI/_Subject_Characteristics"
subject_data = load_ppmi_subject_characteristics(data_path)

# Display the resulting DataFrame
print(f"Loaded {len(subject_data)} rows with {len(subject_data.columns)} columns")
print(subject_data.head())

# Save to CSV if needed
subject_data.to_csv("subject_characteristics.csv", index=False)
```

## Practical Usage Examples

### Basic Loading and Inspection

```python
import pandas as pd
import logging
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# Load subject characteristics 
data_path = "./PPMI/_Subject_Characteristics"
subject_data = load_ppmi_subject_characteristics(data_path)

# Check the dimensions of the loaded data
print(f"Data shape: {subject_data.shape}")

# Examine available columns
print("Available columns:")
for column in sorted(subject_data.columns):
    print(f"- {column}")

# Get a summary of the data
print(subject_data.describe(include='all'))
```

### Filtering and Analyzing Data

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Load the data
data_path = "./PPMI/_Subject_Characteristics"
subject_data = load_ppmi_subject_characteristics(data_path)

# Filter to only include PD patients (if COHORT column exists)
if 'COHORT' in subject_data.columns:
    pd_patients = subject_data[subject_data['COHORT'] == 'PD']
    print(f"Found {len(pd_patients)} rows for PD patients")

# Extract demographic information
if 'SEX' in subject_data.columns and 'AGE' in subject_data.columns:
    demographics = subject_data[['PATNO', 'SEX', 'AGE']].drop_duplicates()
    print(f"Unique patients: {len(demographics)}")
    print(f"Sex distribution: \n{demographics['SEX'].value_counts()}")
    print(f"Age statistics: \n{demographics['AGE'].describe()}")
```

### Joining with Other Data Sources

```python
import pandas as pd
from pie.sub_char_loader import load_ppmi_subject_characteristics
from pie.biospecimen_loader import load_biospecimen_data, merge_biospecimen_data

# Load subject characteristics
data_path = "./PPMI"
subject_data = load_ppmi_subject_characteristics(f"{data_path}/_Subject_Characteristics")

# Load biospecimen data
biospecimen_data = load_biospecimen_data(data_path, "PPMI")
biospecimen_merged = merge_biospecimen_data(
    biospecimen_data, 
    merge_all=True,
    include=["blood_chemistry_hematology"]
)

# Join the datasets
combined_data = pd.merge(
    subject_data,
    biospecimen_merged,
    on=["PATNO", "EVENT_ID"],
    how="inner"
)

print(f"Combined data shape: {combined_data.shape}")
print(f"Number of patients: {combined_data['PATNO'].nunique()}")
```

### Handling Specific Column Deduplication

```python
import pandas as pd
from pie.sub_char_loader import deduplicate_columns

# Create a sample DataFrame with duplicate columns
df = pd.DataFrame({
    'PATNO': [1001, 1002, 1003, 1004],
    'EVENT_ID': ['BL', 'BL', 'V04', 'V04'],
    'INFODT_x': ['2020-01-01', '2020-02-01', None, '2020-04-01'],
    'INFODT_y': ['2020-01-01', None, '2020-03-01', '2020-05-01']
})

print("Before deduplication:")
print(df)

# Deduplicate the INFODT column
df = deduplicate_columns(df, ['INFODT'])

print("\nAfter deduplication:")
print(df)
```

## Data Dictionary

Subject characteristics data typically includes the following categories of information:

- **Demographics**: Age, sex, race, ethnicity
- **Genetic Information**: PD variants, genetic consensus
- **Family History**: Family history of Parkinson's disease
- **Socioeconomic Status**: Education, employment, income
- **Cohort Information**: Patient grouping, study arm
- **Participant Status**: Active/inactive, completion status

The exact columns available depend on which source files are present in the directory.

## Loading Process Details

1. The module scans the specified directory for CSV files matching any of the defined prefixes
2. For each matching file, it attempts to load the CSV into a pandas DataFrame
3. If successful, it merges the DataFrame with the accumulated results:
   - If both DataFrames have PATNO and EVENT_ID, it merges on both columns
   - If one lacks EVENT_ID, it merges on PATNO only, replicating static data across events
4. After all files are merged, it deduplicates specified columns that may have variants
5. The final DataFrame contains all patient characteristics data from the source files

## Best Practices

- **Path Specification**: Ensure the path to the `_Subject_Characteristics` folder is correct
- **Error Handling**: Check if the returned DataFrame is empty before proceeding
- **Column Verification**: Use `df.columns` to check available columns before analysis
- **Memory Efficiency**: For large datasets, consider filtering columns after loading
- **Deduplication**: For custom merges, use the `deduplicate_columns` function to resolve duplicates

## Troubleshooting

- **Empty DataFrame Returned**: Check if the folder path is correct and contains CSV files with the expected prefixes
- **Missing Data**: Verify that the expected source files exist in the directory
- **Duplicate Columns**: Additional column deduplication may be needed for columns not in the default list
- **Memory Issues**: Consider loading fewer files or filtering columns after loading

## Integration with Other PIE Modules

The subject characteristics data is often combined with other data types:

- **Biospecimen Data**: Merge to analyze biomarkers in relation to patient demographics
- **Imaging Data**: Connect imaging findings with genetic or demographic factors
- **Clinical Assessments**: Correlate clinical outcomes with patient characteristics
