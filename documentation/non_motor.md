# Non-Motor Assessment Loader Documentation

## Overview

The `non_motor_loader.py` module provides functions for loading, merging, and processing non-motor assessment data from the Parkinson's Progression Markers Initiative (PPMI) dataset. This module handles various cognitive, psychiatric, autonomic, and other non-motor assessments commonly used in Parkinson's disease research.

## Key Features

- Load multiple non-motor assessment files based on standardized naming conventions
- Intelligently merge related files based on patient ID and visit information
- Handle duplicate columns through a sophisticated combination approach
- Preserve data integrity when integrating different assessment types
- Support both longitudinal (visit-specific) and static patient data

## Supported Assessments

The module supports numerous non-motor assessments, including:

- **Cognitive Tests**: Montreal Cognitive Assessment (MoCA), Hopkins Verbal Learning Test, Symbol Digit Modalities, etc.
- **Psychiatric Scales**: Geriatric Depression Scale, State-Trait Anxiety Inventory, etc.
- **Sleep Assessments**: REM Sleep Behavior Disorder Questionnaire, Epworth Sleepiness Scale
- **Autonomic Function**: SCOPA-AUT
- **Quality of Life**: Neuro QoL
- **Sensory Assessments**: University of Pennsylvania Smell Identification Test
- And many others listed in the `FILE_PREFIXES` constant

## Functions

### `deduplicate_columns(df, duplicate_columns)`

Resolves duplicate columns that arise during merging operations by intelligently combining values.

**Parameters:**
- `df`: The merged DataFrame containing possible duplicates
- `duplicate_columns`: List of column base names that may have duplicates

**Returns:**
- Updated DataFrame with deduplicated columns

**Behavior:**
- For each column in `duplicate_columns`, resolves duplicates created during merging (`column_x` and `column_y`)
- Applies intelligent combination logic:
  1. If both values are empty/NaN, the result is empty
  2. If one is empty, uses the non-empty value
  3. If both are non-empty and identical, uses that value
  4. If both are non-empty and different, combines with a pipe separator (|)

**Example:**

```python
import pandas as pd
from pie.non_motor_loader import deduplicate_columns

# Create sample DataFrame with duplicated columns
data = {
    'PATNO': [1001, 1002, 1003],
    'INFODT_x': ['2020-01-01', None, '2020-03-01'],
    'INFODT_y': [None, '2020-02-01', '2020-03-01']
}
df = pd.DataFrame(data)

# Deduplicate columns
df = deduplicate_columns(df, ['INFODT'])

print(df)
# Output:
#    PATNO      INFODT
# 0   1001  2020-01-01
# 1   1002  2020-02-01
# 2   1003  2020-03-01
```

### `sanitize_suffixes_in_df(df)`

Prepares a DataFrame for merging by handling columns that already have `_x` or `_y` suffixes to prevent merge conflicts.

**Parameters:**
- `df`: The DataFrame to sanitize

**Behavior:**
- Modifies the DataFrame in-place
- Renames columns ending with `_x` or `_y` to avoid conflicts during merge operations
- Appends `_col` or `_col{n}` to create unique column names

**Example:**

```python
import pandas as pd
from pie.non_motor_loader import sanitize_suffixes_in_df

# Create DataFrame with columns already ending in _x or _y
data = {
    'PATNO': [1001, 1002],
    'EVENT_ID': ['BL', 'V01'],
    'SCORE_x': [25, 30],  # Column already has _x suffix
    'STATUS_y': ['Complete', 'Incomplete']  # Column already has _y suffix
}
df = pd.DataFrame(data)

# Sanitize the column names
sanitize_suffixes_in_df(df)

print(df.columns.tolist())
# Output something like: ['PATNO', 'EVENT_ID', 'SCORE_col', 'STATUS_col']
```

### `load_ppmi_non_motor_assessments(folder_path)`

Main function to load and merge all non-motor assessment files from the PPMI dataset.

**Parameters:**
- `folder_path`: Path to the 'Non-motor_Assessments' folder containing CSV files

**Returns:**
- A merged DataFrame containing data from all successfully loaded non-motor assessment files

**Behavior:**
- Searches for CSV files matching the prefixes defined in `FILE_PREFIXES`
- Merges files intelligently based on available columns:
  - If both DataFrames have PATNO and EVENT_ID, merges on both
  - If only PATNO is available, merges on PATNO only (replicating static data across visits)
- Handles duplicate columns through deduplication
- Returns an empty DataFrame if no files are successfully loaded

**Example:**

```python
import pandas as pd
from pie.non_motor_loader import load_ppmi_non_motor_assessments

# Load all non-motor assessment data
data_path = "./PPMI/Non-motor_Assessments"
non_motor_data = load_ppmi_non_motor_assessments(data_path)

# Check the dimensions of the loaded data
print(f"Loaded {len(non_motor_data)} rows with {len(non_motor_data.columns)} columns")

# View the first few rows
print(non_motor_data.head())

# Save the merged data
non_motor_data.to_csv("merged_non_motor_assessments.csv", index=False)
```

## Detailed Usage Examples

### Example 1: Basic Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
from pie.non_motor_loader import load_ppmi_non_motor_assessments

# Load non-motor assessment data
data_path = "./PPMI/Non-motor_Assessments"
non_motor_data = load_ppmi_non_motor_assessments(data_path)

# Display summary information
print(f"Loaded data shape: {non_motor_data.shape}")
print(f"Number of unique patients: {non_motor_data['PATNO'].nunique()}")

# Check available columns for specific assessments
moca_cols = [col for col in non_motor_data.columns if 'MOCA' in col]
print(f"MoCA assessment columns: {moca_cols}")

# Basic summary statistics for MoCA total score (if available)
if 'MCATOT' in non_motor_data.columns:
    print("\nMoCA Total Score Statistics:")
    print(non_motor_data['MCATOT'].describe())
    
    # Distribution of MoCA scores
    plt.figure(figsize=(10, 6))
    plt.hist(non_motor_data['MCATOT'].dropna(), bins=15)
    plt.title('Distribution of MoCA Total Scores')
    plt.xlabel('MoCA Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('moca_distribution.png')
```

### Example 2: Longitudinal Analysis of Cognitive Scores

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pie.non_motor_loader import load_ppmi_non_motor_assessments

# Load non-motor assessment data
data_path = "./PPMI/Non-motor_Assessments"
non_motor_data = load_ppmi_non_motor_assessments(data_path)

# Define the visit order for proper plotting
visit_order = ['BL', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12']

# Focus on cognitive scores across time
cognitive_data = non_motor_data[['PATNO', 'EVENT_ID', 'MCATOT', 'SDMTOTAL', 'HVLTRT1', 'HVLTRT2', 'HVLTRT3']]

# Convert EVENT_ID to categorical with proper order for plotting
cognitive_data['EVENT_ID'] = pd.Categorical(
    cognitive_data['EVENT_ID'], 
    categories=visit_order,
    ordered=True
)

# Sort by PATNO and EVENT_ID
cognitive_data = cognitive_data.sort_values(['PATNO', 'EVENT_ID'])

# Plot MoCA scores across visits
plt.figure(figsize=(12, 7))
sns.boxplot(x='EVENT_ID', y='MCATOT', data=cognitive_data)
plt.title('Montreal Cognitive Assessment (MoCA) Scores Across Visits')
plt.xlabel('Visit')
plt.ylabel('MoCA Total Score')
plt.grid(True, alpha=0.3)
plt.savefig('moca_longitudinal.png')

# Track cognitive change for individual patients (example with first 10 patients)
selected_patients = cognitive_data['PATNO'].unique()[:10]
patient_data = cognitive_data[cognitive_data['PATNO'].isin(selected_patients)]

plt.figure(figsize=(14, 8))
sns.lineplot(
    data=patient_data,
    x='EVENT_ID',
    y='MCATOT',
    hue='PATNO',
    marker='o',
    markersize=8
)
plt.title('Longitudinal MoCA Scores for Selected Patients')
plt.xlabel('Visit')
plt.ylabel('MoCA Total Score')
plt.grid(True, alpha=0.3)
plt.legend(title='Patient ID')
plt.savefig('patient_cognitive_trajectories.png')
```

### Example 3: Combining Non-Motor Data with Subject Characteristics

```python
import pandas as pd
from pie.non_motor_loader import load_ppmi_non_motor_assessments
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Load non-motor assessment data
data_path = "./PPMI"
non_motor_data = load_ppmi_non_motor_assessments(f"{data_path}/Non-motor_Assessments")

# Load subject characteristics
subject_data = load_ppmi_subject_characteristics(f"{data_path}/_Subject_Characteristics")

# Merge datasets
combined_data = pd.merge(
    non_motor_data,
    subject_data[['PATNO', 'EVENT_ID', 'GENDER', 'COHORT', 'AGE']],
    on=['PATNO', 'EVENT_ID'],
    how='left'
)

# Analyze cognitive performance by cohort (PD vs. Control)
if 'MCATOT' in combined_data.columns and 'COHORT' in combined_data.columns:
    # Filter to baseline visits
    baseline_data = combined_data[combined_data['EVENT_ID'] == 'BL']
    
    # Group by cohort
    cohort_stats = baseline_data.groupby('COHORT').agg({
        'MCATOT': ['mean', 'std', 'count'],
        'AGE': 'mean'
    })
    
    print("Cognitive Performance by Cohort (Baseline):")
    print(cohort_stats)
    
    # Save the analysis
    cohort_stats.to_csv('cohort_cognitive_analysis.csv')
    
    # Create comparison visualizations
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='COHORT', y='MCATOT', data=baseline_data)
    plt.title('MoCA Scores by Cohort (Baseline)')
    plt.xlabel('Cohort')
    plt.ylabel('MoCA Total Score')
    plt.grid(True, alpha=0.3)
    plt.savefig('cohort_moca_comparison.png')
```

### Example 4: Custom Deduplication for Specific Columns

```python
import pandas as pd
import numpy as np
from pie.non_motor_loader import deduplicate_columns

# Create a test dataset with duplicate columns
data = {
    'PATNO': [1001, 1002, 1003, 1004],
    'EVENT_ID': ['BL', 'V01', 'BL', 'V01'],
    'PAG_NAME_x': ['MOCA', 'MOCA', None, 'MOCA'],
    'PAG_NAME_y': ['MOCA', None, 'GDS', 'MOCA'],
    'INFODT_x': ['2020-01-01', '2020-02-01', None, '2020-04-01'],
    'INFODT_y': ['2020-01-01', None, '2020-03-01', '2020-05-01'],
    'SCORE_x': [25, 28, None, 22],
    'SCORE_y': [25, None, 5, 23],
}

df = pd.DataFrame(data)
print("Before deduplication:")
print(df)

# Deduplicate the columns
columns_to_deduplicate = ['PAG_NAME', 'INFODT', 'SCORE']
df_deduplicated = deduplicate_columns(df, columns_to_deduplicate)

print("\nAfter deduplication:")
print(df_deduplicated)

# Example showing the pipe separator for conflicting values
print("\nNote how row 3 has conflicting scores (22 vs 23) that were combined with a pipe separator:")
print(df_deduplicated.iloc[3])
```

## Implementation Details

### File Matching and Loading

The module searches for CSV files in the specified directory (and subdirectories) that match the prefixes defined in `FILE_PREFIXES`. This allows for flexibility in file naming and organization while still identifying the relevant assessment files.

### Intelligent Merging Strategy

When merging datasets, the module first determines which columns to use as merge keys:
- If both datasets have `PATNO` and `EVENT_ID`, they are merged on both columns to maintain visit-specific data
- If only `PATNO` is common, the data is merged on patient ID alone, effectively replicating static patient data across all visits

### Column Deduplication Logic

The deduplication process handles several cases:
1. When both columns contain the same value, that value is preserved
2. When one column is empty and the other has a value, the non-empty value is kept
3. When both columns have different values, they are combined with a pipe separator (|)
4. When both columns are empty, the result remains empty

### Merge Suffix Handling

To avoid column naming conflicts during merges, the module:
1. First sanitizes column names that already end with `_x` or `_y` (using `sanitize_suffixes_in_df`)
2. Performs the merge using standard suffixes
3. Deduplicates columns immediately after each merge to maintain clean column names

## Data Dictionary

The specific columns available in the merged dataset depend on which assessment files are successfully loaded. Common column patterns include:

- **Identification Columns**: `PATNO` (patient ID), `EVENT_ID` (visit ID)
- **Administrative Columns**: `PAG_NAME` (assessment name), `INFODT` (assessment date), `LAST_UPDATE`
- **MoCA Assessment**: `MCATOT` (total score), `MCAVFNUM` (verbal fluency score), etc.
- **Hopkins Verbal Learning**: `HVLTRT1`, `HVLTRT2`, `HVLTRT3` (trial scores)
- **Depression Scales**: `GDSTOT` (GDS total score)
- **Anxiety Measures**: `STAIAD` (state anxiety), `STAIT` (trait anxiety)
- **Sleep Assessments**: `REMSLEEP` (REM sleep behavior disorder score), `ESS` (Epworth sleepiness scale)
- **Autonomic Function**: `SCAU` (SCOPA-AUT scores)

## Best Practices

### File Organization

Ensure that all non-motor assessment CSV files are located within the `Non-motor_Assessments` directory (or subdirectories) with filenames that match the expected prefixes.

### Memory Management

When working with large datasets:
- Consider filtering columns after loading to reduce memory usage
- Process the data in chunks if necessary
- Use more efficient data types where appropriate

### Error Handling

The module provides error messages when files cannot be found or loaded. Always check:
- That file paths are correct
- That CSV files are properly formatted
- If any warnings or errors appear during the loading process

### Column Management

Keep track of duplicate columns that may need deduplication beyond the standard set handled by the module. You can add specific columns to the `columns_to_deduplicate` list in the `load_ppmi_non_motor_assessments` function.

## Troubleshooting

### Missing Data

If certain assessments are missing in the output:
- Verify that the CSV files exist in the Non-motor_Assessments directory
- Check that filenames start with one of the prefixes listed in `FILE_PREFIXES`
- Examine any error messages generated during loading

### Duplicate or Conflicting Data

If you see unexpected duplicates or conflicting values:
- Check if the columns should be added to the deduplication list
- Verify if multiple assessments of the same type exist for the same visit
- Consider manual review of the source files for inconsistencies

### Performance Issues

If loading is slow or memory-intensive:
- Consider reducing the number of files to load (modify `FILE_PREFIXES`)
- Filter columns after loading to focus on specific assessments
- Process data in batches rather than all at once

## Integration with Other PIE Modules

This module is designed to work seamlessly with other PIE modules:
- Use alongside `sub_char_loader.py` to merge with demographic and clinical data
- Combine with `motor_loader.py` to analyze relationships between motor and non-motor symptoms
- Integrate with `biospecimen_loader.py` to correlate non-motor symptoms with biomarkers