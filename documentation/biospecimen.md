# Biospecimen Data Loader Documentation

## Overview

The `biospecimen_loader.py` module provides functions for loading, processing, and merging biospecimen data from the Parkinson's Progression Markers Initiative (PPMI) dataset. This module handles various biospecimen data files, including proteomics, metabolomics, blood chemistry, and other biomarker data.

## Key Features

- Load individual biospecimen data files into structured DataFrames
- Handle diverse data formats consistently
- Merge data from multiple sources based on patient ID and event ID
- Memory-efficient processing for large datasets
- Flexible filtering to include or exclude specific data sources
- Save processed data to CSV files

## Functions

### 1. Loading Individual Data Types

The module includes specialized loading functions for different types of biospecimen data:

- `load_project_151_pQTL_CSF()`: Loads proteomics data from Project 151
- `load_metabolomic_lrrk2()`: Loads metabolomic data from LRRK2 studies
- `load_urine_proteomics()`: Loads urine proteomics data
- `load_project_9000()`: Loads Project 9000 data
- `load_project_222()`: Loads Project 222 data
- `load_project_196()`: Loads Project 196 data
- `load_project_177_untargeted_proteomics()`: Loads untargeted proteomics data
- `load_project_214_olink()`: Loads Olink proteomics data
- `load_current_biospecimen_analysis()`: Loads current biospecimen analysis results
- `load_blood_chemistry_hematology()`: Loads blood chemistry and hematology data
- `load_and_join_biospecimen_files()`: Loads standard biospecimen files

### 2. Main Loading Function

#### `load_biospecimen_data(data_path, source)`

Loads all biospecimen data from the specified path.

**Parameters:**
- `data_path`: Path to the data directory
- `source`: The data source (e.g., "PPMI")

**Returns:**
- A dictionary containing loaded biospecimen data with keys for each data type

**Example:**
```python
from pie import biospecimen_loader

# Load all biospecimen data
data_path = "./PPMI"  # Path to your PPMI data folder
biospecimen_data = biospecimen_loader.load_biospecimen_data(data_path, "PPMI")

# Print available data sources
print(biospecimen_data.keys())
```

### 3. Merging Function

#### `merge_biospecimen_data(biospecimen_data, merge_all=True, output_filename="biospecimen.csv", output_dir=None, include=None, exclude=None)`

Merges biospecimen data into a single DataFrame or keeps them as separate DataFrames.

**Parameters:**
- `biospecimen_data`: Dictionary containing loaded biospecimen data
- `merge_all`: If True, merge all DataFrames; if False, return dictionary of DataFrames
- `output_filename`: Name of the output CSV file
- `output_dir`: Directory to save the output file(s); if None, files are not saved
- `include`: List of specific data sources to include (e.g., ['project_151', 'metabolomic_lrrk2'])
- `exclude`: List of specific data sources to exclude (only used if include is None or empty)

**Returns:**
- If `merge_all` is True: A single DataFrame with all biospecimen data
- If `merge_all` is False: Dictionary of DataFrames

**Examples:**

1. Merge all biospecimen data:
```python
from pie import biospecimen_loader

# Load all biospecimen data
data_path = "./PPMI"
biospecimen_data = biospecimen_loader.load_biospecimen_data(data_path, "PPMI")

# Merge all data and save to a CSV file
merged_data = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=True,
    output_filename="all_biospecimen.csv",
    output_dir="./output"
)
```

2. Include only specific data sources:
```python
# Merge only specific data sources
merged_subset = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=True,
    output_filename="blood_and_metabolomics.csv",
    output_dir="./output",
    include=["blood_chemistry_hematology", "metabolomic_lrrk2"]
)
```

3. Exclude specific data sources:
```python
# Merge all data except specific sources
merged_except = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=True,
    output_filename="except_projects.csv",
    output_dir="./output",
    exclude=["project_9000", "project_222"]
)
```

4. Keep data sources separate:
```python
# Keep data sources separate and save individual files
separate_data = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=False,
    output_dir="./output/individual"
)
```

## Memory Optimization

The module includes several memory optimization techniques:

1. **Chunked Processing**: Large files are processed in chunks to reduce memory usage
2. **Targeted Column Selection**: Only necessary columns are kept during merging
3. **Garbage Collection**: Explicit garbage collection after intensive operations
4. **Memory Usage Logging**: Memory consumption is tracked and logged during processing
5. **Chunked CSV Writing**: Large merged datasets are written to disk in chunks

## Data Preprocessing

During loading and merging, the module performs several preprocessing steps:

1. **Standardization**: Column names are standardized across files
2. **Prefix Handling**: "PPMI-" prefixes are removed from patient IDs
3. **Type Conversion**: Patient IDs are converted to strings for consistent joining
4. **Deduplication**: Duplicate records are handled with appropriate strategies
5. **Column Renaming**: Prefix is added to column names to indicate the data source

## Example Workflow

```python
import logging
from pie import biospecimen_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 1. Load all biospecimen data
data_path = "./PPMI"
biospecimen_data = biospecimen_loader.load_biospecimen_data(data_path, "PPMI")

# 2. Print summary of loaded data
for key, df in biospecimen_data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(f"{key}: {len(df)} rows × {len(df.columns)} columns")

# 3. Merge proteomics data only
proteomics_data = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=True,
    output_filename="proteomics.csv",
    output_dir="./output",
    include=["project_151_pQTL_CSF", "project_177", "project_214", "urine_proteomics"]
)

# 4. Create a full merged dataset but exclude large projects
full_except_large = biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=True,
    output_filename="biomarkers_except_large.csv",
    output_dir="./output",
    exclude=["project_9000", "project_222"]
)

# 5. Save individual files for reference
biospecimen_loader.merge_biospecimen_data(
    biospecimen_data,
    merge_all=False,
    output_dir="./output/individual"
)
```

## Performance Considerations

- Processing large biospecimen files (especially Projects 9000, 222, and 196) requires significant memory
- Consider filtering to only necessary data sources when working with limited memory
- The chunked processing approach helps manage memory usage for large datasets
- Memory usage logging can help identify and optimize bottlenecks

## Troubleshooting

- If the merging process crashes due to memory issues:
  - Use the `include` parameter to process fewer data sources at once
  - Increase available system memory or use a machine with more RAM
  - Adjust the chunk size for loading and saving operations
- If certain data sources fail to load:
  - Check that the file structure matches what the loader expects
  - Examine the error logs for specific issues
  - Consider modifying the specific loader function for that data type