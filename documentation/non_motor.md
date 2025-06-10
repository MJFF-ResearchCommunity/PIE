# Non-Motor Assessment Loader Documentation

## Overview

The `non_motor_loader.py` module is designed to load and consolidate a wide range of non-motor assessment data from the PPMI dataset. It automates the process of finding relevant CSV files, merging them into a single pandas DataFrame, and cleaning the result to ensure data integrity and usability for analysis.

## Key Features

- Loads data from a comprehensive list of non-motor assessment file prefixes.
- Recursively searches for CSV files within the specified directory.
- Merges individual assessment files into a single DataFrame based on patient (`PATNO`) and visit (`EVENT_ID`) identifiers.
- Intelligently resolves column name conflicts that arise during merges (e.g., `COLUMN_x`, `COLUMN_y`).
- Ensures unique rows for each patient-visit pair by aggregating data from duplicate entries. Conflicting values are preserved and separated by a pipe (`|`).
- Provides detailed logging for transparency into the loading, merging, and cleaning process.

## Supported Non-Motor Assessments

The loader is configured to search for files that begin with the following prefixes:

- `Benton_Judgement`
- `Clock_Drawing`
- `Cognitive_Categorization`
- `Cognitive_Change`
- `Epworth_Sleepiness_Scale`
- `Geriatric_Depression_Scale`
- `Hopkins_Verbal_Learning_Test`
- `IDEA_Cognitive_Screen`
- `Letter_-_Number_Sequencing`
- `Lexical_Fluency`
- `Modified_Boston_Naming_Test`
- `Modified_Semantic_Fluency`
- `Montreal_Cognitive_Assessment`
- `Neuro_QoL`
- `PDAQ-27`
- `QUIP-Current-Short`
- `REM_Sleep_Behavior_Disorder_Questionnaire`
- `SCOPA-AUT`
- `State-Trait_Anxiety_Inventory`
- `Symbol_Digit_Modalities`
- `Trail_Making`
- `University_of_Pennsylvania_Smell_Identification`

## Function

The module exposes a single primary function for loading data.

### `load_ppmi_non_motor_assessments(folder_path: str)`

Loads, merges, and cleans all non-motor assessment CSV files from a given folder.

**Parameters:**
- `folder_path (str)`: The path to the directory containing non-motor assessment files. A typical path would be `./PPMI/Non-motor_Assessments`. The function will search this directory and its subdirectories.

**Returns:**
- `pandas.DataFrame`: A single DataFrame containing the merged and cleaned data from all found non-motor assessment files. If no files are found or an error occurs, it returns an empty DataFrame.

## Basic Usage Example

```python
import pandas as pd
from pie.non_motor_loader import load_ppmi_non_motor_assessments

# Define the path to your PPMI non-motor assessments folder
# This folder should contain the various CSV files.
non_motor_data_path = "./PPMI/Non-motor_Assessments"

# Load the data
print(f"Loading non-motor assessments from: {non_motor_data_path}")
df_non_motor = load_ppmi_non_motor_assessments(non_motor_data_path)

if not df_non_motor.empty:
    # Display the shape and first few rows of the loaded data
    print(f"Successfully loaded data. Shape: {df_non_motor.shape}")
    print("First 5 rows of the merged non-motor assessments data:")
    print(df_non_motor.head())

    # Example: Inspect columns related to a specific assessment (e.g., MoCA)
    moca_cols = [col for col in df_non_motor.columns if 'MCA' in col.upper()]
    if moca_cols:
        print(f"\nFound MoCA-related columns: {moca_cols}")
        print("Sample MoCA data:")
        print(df_non_motor[['PATNO', 'EVENT_ID'] + moca_cols].head())

    # Save the merged data to a new CSV file for further analysis
    output_path = "merged_non_motor_assessments.csv"
    df_non_motor.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
else:
    print("Loading failed or no data was found. Please check logs and folder path.")
```

## Post-Loading Analysis Example

Once the data is loaded, you can use pandas to analyze it. This example shows how to calculate average cognitive scores across visits.

**Note:** This example assumes that columns like `MCATOT` (Montreal Cognitive Assessment Total Score) and `EVENT_ID` exist in your data. The exact column names depend on the source CSV files.

```python
import pandas as pd
from pie.non_motor_loader import load_ppmi_non_motor_assessments

# Load the data
non_motor_data_path = "./PPMI/Non-motor_Assessments"
df_non_motor = load_ppmi_non_motor_assessments(non_motor_data_path)

if not df_non_motor.empty and 'MCATOT' in df_non_motor.columns and 'EVENT_ID' in df_non_motor.columns:
    print("Performing a simple analysis on MoCA scores...")

    # The data in 'MCATOT' might be strings if aggregation occurred.
    # Convert it to a numeric type, handling errors and pipe-separated values.
    def clean_and_convert_to_numeric(series):
        # Take the first value if data is pipe-separated (e.g., '28|29' -> '28')
        series_str = series.astype(str).str.split('|').str[0]
        return pd.to_numeric(series_str, errors='coerce')

    df_non_motor['MCATOT_numeric'] = clean_and_convert_to_numeric(df_non_motor['MCATOT'])

    # Remove rows where the score could not be converted to a number
    df_analysis = df_non_motor.dropna(subset=['MCATOT_numeric', 'EVENT_ID'])

    # Calculate the average MoCA score by visit
    avg_score_by_visit = df_analysis.groupby('EVENT_ID')['MCATOT_numeric'].mean().reset_index()

    # Sort by a typical visit order (simple alphabetical sort for this example)
    avg_score_by_visit = avg_score_by_visit.sort_values('EVENT_ID')
    
    print("\nAverage MoCA Score by Visit:")
    print(avg_score_by_visit)

else:
    print("\nSkipping analysis example: 'MCATOT' or 'EVENT_ID' not found in the DataFrame.")
```

## Implementation Details

The loading process follows several key steps to ensure data quality and consistency:

1.  **File Discovery**
    The loader recursively scans the provided `folder_path` for CSV files, filtering for files that start with a prefix from its internal `FILE_PREFIXES` list.

2.  **Iterative Merging**
    The function loads the first matching file and then iteratively merges subsequent files into it. The merge is an `outer` join on `["PATNO", "EVENT_ID"]` to include all data. If a file lacks `EVENT_ID`, the merge uses `PATNO` only.

3.  **Handling Merge Suffixes**
    When pandas adds `_x` and `_y` suffixes to overlapping columns, a utility function resolves them:
    - If only `COL_x` or `COL_y` exists, it is renamed to `COL`.
    - If both `COL_x` and `COL_y` exist, they are combined into `COL`. If values differ, they are joined with a pipe (`|`).

4.  **Ensuring Unique Patient-Visit Rows**
    After merging, a final aggregation step ensures each (`PATNO`, `EVENT_ID`) pair is unique. It groups data by these keys and combines information from duplicated rows. Conflicting values in other columns are joined with a pipe (`|`).

## Data Dictionary

Common fields found in the non-motor assessment source files include:

| Field | Description | Example Values |
|-------|-------------|----------------|
| PATNO | Patient identifier | 1001, 1002, ... |
| EVENT_ID | Visit identifier | BL (baseline), V01, V02, ... |
| MCATOT | Montreal Cognitive Assessment (MoCA) total score | 0-30 |
| GDSTOT | Geriatric Depression Scale (GDS) total score | 0-15 or 0-30 |
| HVLT... | Hopkins Verbal Learning Test scores (e.g., `HVLTRT1`) | 0-12 |
| STAI... | State-Trait Anxiety Inventory scores (e.g., `STAIAD`) | 20-80 |
| SCAU... | SCOPA-AUT scores (e.g., `SCAUTOT`) | Various |
| UPSIT... | U-Penn Smell ID Test score (e.g., `UPSIT_TOTAL_SCORE`) | 0-40 |

*Note: Exact column names can vary slightly between different versions or studies in the PPMI dataset.*

## Troubleshooting

- **Empty DataFrame is Returned:** If the function returns an empty DataFrame, check that:
    - The `folder_path` is correct.
    - The directory contains CSV files starting with one of the recognized prefixes.
    - The application logs for any warnings about files that could not be loaded.

- **Pipe-Separated Values (`|`):** If you find values like `"value1|value2"`, it means that different source files contained different information for the same patient, visit, and data field. The loader preserves both values to prevent data loss.

- **Missing PATNO or EVENT_ID Columns:** The loader expects a `PATNO` column in every file and logs a warning before skipping files that lack it. Merging behavior may be less precise if `EVENT_ID` is missing, which is also noted in the logs.