# PIE DataLoader Documentation

## Overview

The `DataLoader` class, located in `pie/data_loader.py`, serves as the main entry point for the PIE library. It provides a high-level, unified interface to load, combine, and process data from various modalities within the PPMI dataset. This class orchestrates the specialized loaders for subject characteristics, medical history, motor/non-motor assessments, and complex biospecimen data, making it easy to get an analysis-ready dataset with a single function call.

## Key Features

- **Unified Interface**: A single static method, `DataLoader.load()`, handles all data loading requests.
- **Multi-Modality Loading**: Load any combination of data types, including subject characteristics, medical history, clinical assessments, and biospecimens.
- **Flexible Output**: Return the loaded data as a dictionary of separate DataFrames or as a single, fully merged DataFrame.
- **Intelligent Merging**: When creating a merged DataFrame, it constructs a comprehensive index of all patient-visit pairs across all modalities, ensuring no data is lost during merges.
- **Memory Management**: Provides crucial options, like `biospec_exclude`, to selectively skip loading extremely large datasets, making it possible to work with the data on standard hardware.
- **Automated Saving**: Easily save the output, whether it's a single merged file or a collection of individual modality files.

## API Reference

The primary interface is the static method `DataLoader.load()`.

### `DataLoader.load(data_path, modalities, source, merge_output, output_file, clean_data, biospec_exclude)`

```python
@staticmethod
def load(
    data_path: str = "./PPMI",
    modalities: Optional[List[str]] = None,
    source: str = "PPMI",
    merge_output: bool = False,
    output_file: str = None,
    clean_data: bool = True,
    biospec_exclude: Optional[List[str]] = None
) -> Union[Dict[str, Any], pd.DataFrame]:
```

#### Parameters

- **`data_path`** `(str, default="./PPMI")`: The root path to your data directory. The loader expects subdirectories like `_Subject_Characteristics`, `Motor___MDS-UPDRS`, etc., to be inside this path.

- **`modalities`** `(List[str], optional)`: A list specifying which data modalities to load. If `None` (default), all available modalities are loaded. It's recommended to use the constants provided by the library:
    - `SUBJECT_CHARACTERISTICS` ("subject_characteristics")
    - `MEDICAL_HISTORY` ("medical_history")
    - `MOTOR_ASSESSMENTS` ("motor_assessments")
    - `NON_MOTOR_ASSESSMENTS` ("non_motor_assessments")
    - `BIOSPECIMEN` ("biospecimen")

- **`source`** `(str, default="PPMI")`: The identifier for the data source. Currently, only "PPMI" is supported.

- **`merge_output`** `(bool, default=False)`: This critical parameter controls the output format.
    - If `False` (default): The function returns a dictionary where keys are the modality names and values are the corresponding data (typically a DataFrame, but a dictionary for Medical History).
    - If `True`: The function returns a single, wide-format pandas DataFrame, where all loaded modalities are merged on `PATNO` and `EVENT_ID`.

- **`output_file`** `(str, optional)`: If a path is provided, the output will be saved to disk.
    - When `merge_output=True`, this is the path for the single merged CSV file (e.g., `./output/merged_data.csv`).
    - When `merge_output=False`, this is used as a base directory to save individual files for each modality. For example, if `output_file="./output/data"`, it will create files like `./output/motor_assessments.csv`, `./output/subject_characteristics.csv`, etc.

- **`clean_data`** `(bool, default=True)`: If `True`, applies relevant cleaning functions. Currently, this primarily affects the `medical_history` modality, where it reshapes and cleans the data tables.

- **`biospec_exclude`** `(List[str], optional)`: A list of biospecimen source keys to **exclude** from loading. This is the most important parameter for managing memory. Use the source keys from the `biospecimen_loader` documentation (e.g., `'project_9000'`, `'project_222'`).

#### Returns

- `Union[Dict[str, Any], pd.DataFrame]`: Either a dictionary of DataFrames or a single merged DataFrame, depending on the `merge_output` parameter.

---

## Practical Usage Examples

### Example 1: Load Specific Modalities as a Dictionary

This is the simplest use case, where you want to get data for a few modalities to work with them separately.

```python
from pie.data_loader import DataLoader
from pie.constants import SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS

print("Loading subject characteristics and motor assessments...")
data_dictionary = DataLoader.load(
    modalities=[SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS],
    merge_output=False
)

# Access the data for each modality
df_subjects = data_dictionary[SUBJECT_CHARACTERISTICS]
df_motor = data_dictionary[MOTOR_ASSESSMENTS]

print(f"Loaded Subject Characteristics: {df_subjects.shape}")
print(f"Loaded Motor Assessments: {df_motor.shape}")
```

### Example 2: Load All Data and Create a Single Merged DataFrame

This example demonstrates the power of the loader to create a single, comprehensive dataset. **Warning:** This can be very memory-intensive if all biospecimen data is included.

```python
from pie.data_loader import DataLoader

print("Loading and merging all modalities (this can take a lot of memory)...")

# For memory safety, we'll exclude the largest biospecimen projects
large_bio_projects = ['project_9000', 'project_222', 'project_196']

df_merged = DataLoader.load(
    merge_output=True,
    biospec_exclude=large_bio_projects,
    output_file="./output/merged_all_modalities.csv"
)

if not df_merged.empty:
    print(f"\nSuccessfully created and saved a merged DataFrame.")
    print(f"Shape: {df_merged.shape}")
    print("Sample columns:", df_merged.columns.tolist()[:5] + df_merged.columns.tolist()[-5:])
```

### Example 3: Saving Individual Modality Files

If you want to load everything but keep the files separate, you can use `merge_output=False` with `output_file`.

```python
from pie.data_loader import DataLoader

print("Loading all modalities and saving them as individual CSV files...")

# When merge_output=False, `output_file` acts as a base path.
# The loader will create files like `output/motor.csv`, `output/biospecimen/project_9000.csv`, etc.
DataLoader.load(
    merge_output=False,
    output_file="./output/individual_files/data" # The filename 'data' will be ignored.
)

print("\nCheck the './output/individual_files/' directory for the saved CSVs.")
```

---

## How to Run the Tests

A test script, `tests/test_data_loader.py`, is provided to verify the functionality of the `DataLoader`.

### Prerequisites

1.  Ensure you have a local copy of the PPMI dataset.
2.  The test script expects the data to be in a folder named `PPMI` in the root of the project directory. If your data is located elsewhere, you will need to edit the `data_dir` variable in the test script.

    ```python
    # Inside tests/test_data_loader.py
    data_dir = "./PPMI"  # <-- CHANGE THIS PATH IF YOUR DATA IS ELSEWHERE
    ```

### Running the Script

Open your terminal, navigate to the root directory of the PIE project, and run the following command:

```bash
python3 tests/test_data_loader.py
```

### What the Test Does

The test script primarily runs `DataLoader.load` with `merge_output=False` and `biospec_exclude` set to `['project_9000', 'project_222', 'project_196']`. It then performs the following checks:
1.  Loads all modalities into a dictionary.
2.  Verifies that the excluded biospecimen projects are **not** present in the loaded biospecimen data dictionary.
3.  Prints a summary of the shapes of all loaded data tables.

This provides a quick and effective way to confirm that your data is structured correctly and that the `DataLoader`'s core functionality, especially the memory-saving `biospec_exclude` feature, is working as intended.
