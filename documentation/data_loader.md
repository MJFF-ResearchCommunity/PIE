# PIE DataLoader Documentation

## Overview

The `DataLoader` class, located in the `pie_clean` package, serves as the main entry point for data into the PIE library. It provides a high-level, unified interface to load, combine, and process data from various modalities within the PPMI dataset. This class orchestrates the specialized loaders for subject characteristics, medical history, motor/non-motor assessments, and complex biospecimen data, making it easy to get an analysis-ready dataset with a single function call.

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

### Extended modalities: every other PPMI study-data folder

The five modalities above are the clinical core and are what `modalities=None` loads. The other folders of a full PPMI
study-data download are loaded generically — every CSV in the folder, `patno`/`event_id` upper-cased (the FOUND tables are
lower-case), files without a `PATNO` column (codebooks, dictionaries) skipped — when you name them explicitly:

| constant | folder | returns |
|---|---|---|
| `STUDY_ENROLLMENT` | `Study_Enrollment` | dict of tables (consent, eligibility incl. `INSAA`, screen fail, visit type, ...) |
| `IMAGING` | `Imaging` | dict of tables (Xing core-lab DaTscan SBR / visual reads, FreeSurfer-7 IDPs, MRIQC, DTI ROIs, PET) |
| `PPMI_ONLINE` | `PPMI_Online` | dict of tables; `EVENT_ID`s are online visits, not clinic visits |
| `REMOTE_SCREENING` | `PPMI_Remote_Screening` | one table merged on `PATNO`/`EVENT_ID` |
| `FOUND` | `Follow_Up_persons_w_Neurologic_Disease` | one table merged on `PATNO` (risk-factor questionnaires) |
| `ROCHE_APP` | `Roche_Smartphone_App` | dict with the long-format app table (one row per test result) |

`EXTENDED_MODALITIES` maps each constant to its merge policy and `KNOWN_MODALITIES` lists everything `DataLoader.load`
accepts. With `merge_output=True` the table-dict modalities are merged table by table on `PATNO`/`EVENT_ID` like medical
history; with `output_file` each gets its own subdirectory. `load_ppmi_folder(folder, name, merge)` is the underlying
helper and `load_data_dictionary(data_path)` returns the annotated data dictionary, code list and deprecated-variable
tables from the `Data___Databases` download (keys `dictionary`, `code_list`, `deprecated`) for decoding any column.

Core-folder tables added in the 2026 data cuts are picked up too: polygenic risk scores, race/ethnicity and ST-Direct
demographics (subject characteristics); `Primary_Research_Diagnosis` and the newer PET/tau/FD4 substudy forms (medical
history); CANTAB cognitive activities, `Smell_and_Genetic_Testing` and the renamed RBD screening questionnaire
(non-motor); the underscore-named MDS-UPDRS Part II file (motor); and, in the biospecimen `standard_files` group, the
CSF alpha-synuclein SAA results (`SAA_Status`, `SAA_Type`), SynOne and whole-blood substudy forms, neuropathology and
pathology-core tables.

```python
from pie_clean import DataLoader, load_data_dictionary
from pie_clean.constants import STUDY_ENROLLMENT, IMAGING, FOUND
d = DataLoader.load("./PPMI", modalities=[STUDY_ENROLLMENT, IMAGING, FOUND])
d[IMAGING]["Xing_Core_Lab_-_Quant_SBR"].head()
codes = load_data_dictionary("./PPMI")["code_list"]
```

## Practical Usage Examples

Please see the [PIE-clean documentation for modality-specific details for loading](https://github.com/MJFF-ResearchCommunity/PIE-clean/tree/main/documentation) the more complex data types, such as biospecimens and non-motor exams. The examples below show the key use cases when loading data for PIE.

### Example 1: Load Specific Modalities as a Dictionary

This is the simplest use case, where you want to get data for a few modalities to work with them separately.

```python
from pie_clean import DataLoader
from pie_clean import SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS

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
from pie_clean import DataLoader

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
from pie_clean import DataLoader

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
