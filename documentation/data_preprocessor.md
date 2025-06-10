# PIE DataPreprocessor Documentation

## Overview

The `DataPreprocessor` class, located in `pie/data_preprocessor.py`, is a collection of static methods designed to perform targeted cleaning, standardization, and feature engineering on specific data tables within the PPMI dataset. Unlike the `DataReducer`, which performs general data reduction based on metrics like missingness, the `DataPreprocessor` applies domain-specific knowledge to fix known data inconsistencies and derive more meaningful variables.

For instance, it can convert raw blood pressure readings into standardized hypertension stages or map free-text medication indications to a consistent set of codes. These methods are typically applied after data is loaded by the `DataLoader` and before it is merged or fed into analysis pipelines.

## Key Features

- **Domain-Specific Cleaning**: Contains specialized functions for cleaning individual data tables like `Vital_Signs`, `Concomitant_Medication`, and more.
- **Value Standardization**: Converts ambiguous or inconsistent values (e.g., text entries, "Uncertain" codes) into a standardized format suitable for analysis.
- **Feature Creation**: Derives new, more informative features from raw data (e.g., creating blood pressure category labels from systolic/diastolic values).
- **Utility Functions**: Provides helpful converters, like mapping `EVENT_ID` strings to a numerical timeline of months.
- **Static Methods**: All methods are static, meaning you can use them directly without creating an instance of the `DataPreprocessor` class.

## API Reference

All methods are static and can be called directly (e.g., `DataPreprocessor.clean_vital_signs(...)`).

### Main Dispatcher Methods

#### `clean(data_dict)`

This is the main entry point that applies all relevant cleaning functions to a full data dictionary loaded by `DataLoader`. It currently dispatches to `clean_medical_history`.

- **Parameters**:
    - **`data_dict`** `(Dict)`: The dictionary of data returned by `DataLoader.load(merge_output=False)`.
- **Returns** `(Dict)`: The same dictionary with the relevant DataFrames cleaned in place.

#### `clean_medical_history(med_hist_dict)`

Orchestrates the cleaning of all tables within the `medical_history` modality.

- **Parameters**:
    - **`med_hist_dict`** `(Dict)`: The dictionary for the `medical_history` modality (i.e., `data_dict['medical_history']`).
- **Returns** `(Dict)`: The medical history dictionary with its DataFrames cleaned.

---

### Specific Cleaning Functions

#### `clean_vital_signs(vs_df)`

Enriches the Vital Signs data by adding blood pressure categories.

- **Action**: Based on `SYSSUP`/`DIASUP` (supine) and `SYSSTND`/`DIASTND` (standing) columns, it adds four new columns:
    - `Sup BP code` & `Sup BP label`: Numeric code (0-4) and text label (e.g., "Normal", "Stage 1 HTN") for supine blood pressure.
    - `Stnd BP code` & `Stnd BP label`: The same for standing blood pressure.
- **Parameters**:
    - **`vs_df`** `(pd.DataFrame)`: The Vital Signs DataFrame.
- **Returns** `(pd.DataFrame)`: The cleaned DataFrame with four new columns.

#### `clean_features_of_parkinsonism(fop_df, uncertain=0.5)`

Standardizes values in the Features of Parkinsonism table.

- **Action**: In columns like `FEATBRADY`, it converts the value `2` (meaning "Uncertain") to a specified numeric value.
- **Parameters**:
    - **`fop_df`** `(pd.DataFrame)`: The Features of Parkinsonism DataFrame.
    - **`uncertain`** `(float, default=0.5)`: The value to replace `2` with.
- **Returns** `(pd.DataFrame)`: The cleaned DataFrame.

#### `clean_gen_physical_exam(gpe_df, uncertain=0.5)`

Standardizes values in the General Physical Exam table.

- **Action**: In the `ABNORM` column, converts the value `2` (meaning "Cannot assess") to a specified numeric value.
- **Parameters**:
    - **`gpe_df`** `(pd.DataFrame)`: The General Physical Exam DataFrame.
    - **`uncertain`** `(float, default=0.5)`: The value to replace `2` with.
- **Returns** `(pd.DataFrame)`: The cleaned DataFrame.

#### `clean_concomitant_meds(concom_meds_df)`

Performs a complex and crucial cleaning of the Concomitant Medications table.

- **Action**:
    1.  Converts `STARTDT` and `STOPDT` columns to proper datetime objects.
    2.  Uses an internal JSON mapping file (`concomitant_meds_indications.json`) to harmonize the `CMINDC` (medication indication code) column. It maps messy free-text entries from `CMINDC_TEXT` to their corresponding numeric codes, ensuring every medication has a standardized indication code.
    3.  Uses the newly harmonized codes to fill in a clean `CMINDC_TEXT` column with standardized labels.
- **Parameters**:
    - **`concom_meds_df`** `(pd.DataFrame)`: The Concomitant Medication DataFrame.
- **Returns** `(pd.DataFrame)`: The harmonized and cleaned DataFrame.

---

### Utility Functions

#### `event_id_to_months(eid)`

Converts a visit `EVENT_ID` string (e.g., "V04") into the corresponding number of months from baseline.

- **Parameters**:
    - **`eid`** `(str)`: The event ID string.
- **Returns** `(int or np.nan)`: The number of months, or `NaN` if the ID is not recognized.

#### `dt_to_datetime(dt_ser)`

Converts a pandas Series of date strings (in "MM/YYYY" format) to a Series of datetime objects.

- **Parameters**:
    - **`dt_ser`** `(pd.Series)`: The Series containing date strings.
- **Returns** `(pd.Series)`: The Series with datetime objects.

---

## Practical Usage Example

The `DataLoader` automatically applies the relevant pre-processing steps when loading `medical_history`. However, you can also apply them manually.

```python
from pie.data_loader import DataLoader
from pie.constants import MEDICAL_HISTORY
from pie.data_preprocessor import DataPreprocessor

# Load data without applying the cleaner via DataLoader first
# (Note: DataLoader's default clean_data=True would normally do this)
data_dict = DataLoader.load(modalities=[MEDICAL_HISTORY], clean_data=False)

# Get the raw Vital Signs table
raw_vitals_df = data_dict[MEDICAL_HISTORY]["Vital_Signs"]
print("Raw Vital Signs columns:", raw_vitals_df.columns.tolist())

# --- Apply a specific cleaner manually ---
print("\nCleaning Vital Signs data manually...")
clean_vitals_df = DataPreprocessor.clean_vital_signs(raw_vitals_df)
print("Cleaned Vital Signs columns:", clean_vitals_df.columns.tolist())
print("\nSample of new blood pressure columns:")
print(clean_vitals_df[['SYSSUP', 'DIASUP', 'Sup BP code', 'Sup BP label']].head())

# --- Apply all medical history cleaners at once ---
print("\nApplying all medical history cleaners...")
clean_med_hist_dict = DataPreprocessor.clean_medical_history(data_dict[MEDICAL_HISTORY])

# Check that the Concomitant Meds table was cleaned
clean_concom_meds_df = clean_med_hist_dict["Concomitant_Medication"]
print("\nSample of cleaned Concomitant Meds data:")
print(f"Original CMINDC nulls: {data_dict[MEDICAL_HISTORY]['Concomitant_Medication']['CMINDC'].isnull().sum()}")
print(f"Cleaned CMINDC nulls: {clean_concom_meds_df['CMINDC'].isnull().sum()}")
```

---

## How to Run the Tests

The test script `tests/test_data_preprocessor.py` contains unit tests for each specific cleaning function to ensure they behave as expected.

### Prerequisites

1.  **Pytest**: You must have `pytest` installed (`pip install pytest`).
2.  **PPMI Data**: The tests require access to the PPMI dataset, as they load data using `DataLoader`. Ensure the data is in a `./PPMI` directory at the project root, or modify the test fixture if needed.

### Running the Script

The tests are designed to be run with `pytest`. From the root directory of the PIE project, run the following command in your terminal:

```bash
pytest tests/test_data_preprocessor.py
```

### What the Tests Do

Pytest will automatically discover and run each test function in the file:
-   **`test_clean_vital_signs`**: Asserts that the new blood pressure code and label columns are correctly added.
-   **`test_clean_features_of_parkinsonism`**: Confirms that the "Uncertain" value `2` is removed and replaced.
-   **`test_clean_concomitant_meds`**: Checks that date columns are converted to datetime objects and, crucially, that all null values in the `CMINDC` column are filled in by the mapping logic.

**Note on Skipped Test**: You will see one test being skipped (`test_create_concomitant_meds`). This is intentional. That function is a one-time utility used to generate the internal JSON mapping file; it is not part of the standard data processing workflow and does not need to be tested during normal operation.
