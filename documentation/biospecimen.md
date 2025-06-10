# Biospecimen Data Loader Documentation

## Overview

The `biospecimen_loader.py` module is a powerful and comprehensive tool for loading, processing, and merging a wide variety of biospecimen data from the PPMI dataset. It is designed to handle the complexity and diversity of biospecimen files, including large-scale proteomics and metabolomics data, by using specialized loaders for each data type and a robust merging strategy.

## Key Features

- **Specialized Loaders**: Contains dedicated functions to correctly parse and reshape numerous, complex biospecimen file types.
- **Master Loading Function**: A single function, `load_biospecimen_data`, orchestrates all individual loaders.
- **Intelligent Merging**: A powerful `merge_biospecimen_data` function combines the disparate datasets into a single, analysis-ready DataFrame.
- **Automatic Column Prefixes**: Prevents column name collisions by automatically adding a prefix to columns based on their source (e.g., `project_9000_...`, `bch_...`).
- **Flexible Filtering**: Allows for selective inclusion or exclusion of specific datasets, which is crucial for managing memory and focusing analysis.
- **Memory Efficiency**: Employs chunking and explicit memory management to handle extremely large datasets.
- **Data Integrity**: Ensures unique patient-visit (`PATNO`, `EVENT_ID`) pairs in the final merged output through a sophisticated aggregation process.

## Core Functions

The module's workflow revolves around two main functions:

### 1. `load_biospecimen_data(data_path, source, exclude=None)`

This is the primary entry point for loading all supported biospecimen data. It scans the specified path and uses the appropriate specialized loader for each file type it finds.

**Parameters:**
- `data_path (str)`: Path to the main data directory (e.g., `./PPMI`). The loader will look for a `Biospecimen` subfolder within this path.
- `source (str)`: The data source (e.g., "PPMI").
- `exclude (list, optional)`: A list of data source keys to *exclude* from loading. This is useful for saving memory by skipping very large datasets.

**Returns:**
- `dict`: A dictionary where keys are the names of the data sources (e.g., `"project_9000"`, `"blood_chemistry_hematology"`) and values are the corresponding pandas DataFrames.

### 2. `merge_biospecimen_data(biospecimen_data, ...)`

This function takes the dictionary of DataFrames produced by `load_biospecimen_data` and merges them.

**Parameters:**
- `biospecimen_data (dict)`: The dictionary of DataFrames to merge.
- `merge_all (bool)`: If `True` (default), merges all DataFrames into one. If `False`, it returns the dictionary of filtered DataFrames.
- `output_dir (str, optional)`: If provided, saves the output file(s) to this directory.
- `output_filename (str)`: The name for the merged CSV file.
- `include (list, optional)`: A list of data source keys to *include* in the merge. All others will be ignored.
- `exclude (list, optional)`: A list of data source keys to *exclude* from the merge. Used only if `include` is not provided.

**Returns:**
- A single merged `pandas.DataFrame` if `merge_all=True`, or a `dict` of DataFrames if `merge_all=False`.

## Example Workflows

### 1. Basic Load and Full Merge

This is the most common workflow: load everything, merge it all, and save the result.

```python
import pandas as pd
from pie import biospecimen_loader

# 1. Load all available biospecimen data sources
data_path = "./PPMI"
print("Loading all biospecimen data...")
biospecimen_dict = biospecimen_loader.load_biospecimen_data(data_path, "PPMI")

# You can inspect the keys to see what was loaded
print(f"Loaded data sources: {list(biospecimen_dict.keys())}")

# 2. Merge all loaded data into a single DataFrame
print("\nMerging all data sources...")
df_merged_all = biospecimen_loader.merge_biospecimen_data(
    biospecimen_dict,
    merge_all=True,
    output_dir="./output",
    output_filename="all_biospecimen_data.csv"
)

if not df_merged_all.empty:
    print(f"\nSuccessfully created a merged DataFrame with shape: {df_merged_all.shape}")
    # Display a few prefixed columns to see the result
    bch_cols = [c for c in df_merged_all.columns if c.startswith('blood_chemistry_hematology_')]
    print("Sample of merged data:")
    print(df_merged_all[['PATNO', 'EVENT_ID'] + bch_cols[:3]].head())
```

### 2. Selective Loading and Merging (Memory-Friendly)

For large datasets, it's often better to exclude or include only specific sources.

```python
from pie import biospecimen_loader

data_path = "./PPMI"

# --- Method 1: Exclude large projects during the initial load ---
print("Loading data, excluding large projects...")
large_projects = ['project_9000', 'project_222', 'project_196']
bio_dict_small = biospecimen_loader.load_biospecimen_data(
    data_path, "PPMI", exclude=large_projects
)

print(f"\nLoaded sources after exclusion: {list(bio_dict_small.keys())}")

df_merged_small = biospecimen_loader.merge_biospecimen_data(
    bio_dict_small,
    output_dir="./output",
    output_filename="biospecimen_data_small.csv"
)
print(f"Created a smaller merged DataFrame with shape: {df_merged_small.shape}")


# --- Method 2: Load everything, but only include specific data in the merge ---
print("\nMerging only proteomics data...")
# Assume biospecimen_dict from the first example is already loaded
proteomics_sources = [
    'project_151_pQTL_CSF', 
    'project_177', 
    'project_214', 
    'urine_proteomics'
]
df_proteomics = biospecimen_loader.merge_biospecimen_data(
    biospecimen_dict, # Using the full dictionary
    include=proteomics_sources,
    output_dir="./output",
    output_filename="proteomics_only.csv"
)
print(f"Created a proteomics-only DataFrame with shape: {df_proteomics.shape}")
```

### 3. Working with Separate DataFrames

If you don't want to merge the data, you can process or save the individual files.

```python
from pie import biospecimen_loader

data_path = "./PPMI"
biospecimen_dict = biospecimen_loader.load_biospecimen_data(data_path, "PPMI")

# Process with merge_all=False and provide an output directory to save individual files
separate_dataframes = biospecimen_loader.merge_biospecimen_data(
    biospecimen_dict,
    merge_all=False,
    output_dir="./output" # This will create an 'individual_biospecimen' subfolder
)

print("\nSaved individual CSV files to ./output/individual_biospecimen")
# The returned object is the dictionary of DataFrames, which you can now use
if 'blood_chemistry_hematology' in separate_dataframes:
    df_bch = separate_dataframes['blood_chemistry_hematology']
    print(f"Blood Chemistry data has shape: {df_bch.shape}")
```

## Implementation Details

- **Pivoting & Reshaping**: Many of the specialized loaders (`load_project_151_pQTL_CSF`, `load_blood_chemistry_hematology`, etc.) transform data from a "long" format (one row per measurement) to a "wide" format (one row per patient-visit, with each measurement as a column). This is essential for merging.
- **Merging Strategy**: The `merge_biospecimen_data` function first creates a base DataFrame containing every unique `(PATNO, EVENT_ID)` pair found across all source files. It then iterates through each source DataFrame, prefixes its columns (e.g., `blood_chemistry_hematology_BCH_ALB_LSIRES`), and performs a `left` merge onto the base DataFrame.
- **Data Aggregation**: Before and after merging, the loader runs an aggregation step to ensure each `(PATNO, EVENT_ID)` pair is unique. If duplicates are found (e.g., from different source files providing conflicting info), the values are combined with a pipe (`|`) separator to prevent data loss.

## Supported Data Sources

The `load_biospecimen_data` function can load the following data types. Use these keys in the `include`/`exclude` parameters:

- `project_151_pQTL_CSF`
- `project_151_pQTL_CSF_batch_corrected`
- `metabolomic_lrrk2`
- `metabolomic_lrrk2_csf`
- `urine_proteomics`
- `project_9000`
- `project_222`
- `project_196`
- `project_177`
- `project_214`
- `current_biospecimen`
- `blood_chemistry_hematology`
- `standard_files` (a combination of simpler files like Clinical Labs, Genetic Testing, etc.)

## Performance & Memory

- **High Memory Usage**: Be aware that loading and merging all biospecimen data, especially the large proteomics projects (`project_9000`, `project_222`, `project_196`), can consume a significant amount of RAM (>32GB).
- **Use Filters**: The `exclude` parameter in `load_biospecimen_data` and the `include`/`exclude` parameters in `merge_biospecimen_data` are the primary tools for managing memory.
- **Chunking**: The loader uses chunking to read and write large files, but the in-memory merge process is still demanding.

## Troubleshooting

- **Memory Errors**: If you encounter a `MemoryError`, the best solution is to use the `exclude` parameter to remove the largest data sources. Load and process data in smaller, more manageable batches.
- **Missing Data**: If a specific data type is not loaded, check that the corresponding CSV files exist in the `PPMI/Biospecimen` directory and that their filenames match the expected patterns. Check the logs for warnings.
- **File Not Found**: Ensure your `data_path` is correct and points to the parent directory of the `Biospecimen` folder.