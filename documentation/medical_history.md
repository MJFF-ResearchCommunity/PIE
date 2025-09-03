# Medical History Loader Documentation

## Overview

The `med_history_loader.py` module provides functionality for loading and processing medical history data from the Parkinson's Progression Markers Initiative (PPMI) dataset. Unlike other data types in PPMI that follow a visit-centric structure, medical history data often captures information collected at irregular intervals or multiple entries per visit (such as adverse events or medications).

This module works in conjunction with `data_preprocessor.py` to load, clean, and prepare medical history data for analysis and integration with other PPMI data modalities.

## Module Architecture

The medical history loader follows a different approach than other loaders because of the unique structure of medical history data:

1. Medical history files are kept as separate DataFrames rather than being merged into a single table
2. Each medical history category (e.g., Adverse Events, Concomitant Medications) is loaded individually
3. The loader returns a dictionary of DataFrames, keyed by the file prefix

## Key Components

### Constants

#### `MEDICAL_HISTORY_PREFIXES`

A list of file prefixes that identify the various medical history files in the PPMI dataset:

```python
MEDICAL_HISTORY_PREFIXES = [
    "Adverse_Event",
    "Clinical_Diagnosis",
    "Concomitant_Medication",
    "Features_of_Parkinsonism",
    "Medical_Conditions",
    "Neurological_Exam",
    "Vital_Signs",
    # ... and many others
]
```

This list ensures that all relevant medical history files are identified and loaded.

## Functions

### `sanitize_suffixes_in_df(df)`

Prepares a DataFrame for potential future merging by handling columns that already end with `_x` or `_y` suffixes.

**Parameters:**
- `df`: The DataFrame to sanitize (modified in-place)

**Behavior:**
- Renames columns that end with `_x` or `_y` to avoid conflicts during any future merge operations
- Uses `_col` or `_col{n}` suffixes to create unique column names

**Example:**

```python
import pandas as pd
from pie.med_hist_loader import sanitize_suffixes_in_df

# Create DataFrame with columns already ending in _x or _y
data = {
    'PATNO': [1001, 1002],
    'EVENT_ID': ['BL', 'V01'],
    'MEDICATION_x': ['Levodopa', 'Ropinirole'],
    'STATUS_y': ['Active', 'Discontinued']
}
df = pd.DataFrame(data)

# Sanitize column names
sanitize_suffixes_in_df(df)

print("Sanitized columns:", df.columns.tolist())
# Output: ['PATNO', 'EVENT_ID', 'MEDICATION_col', 'STATUS_col']
```

### `deduplicate_columns(df, duplicate_columns)`

Resolves duplicate columns in a DataFrame by intelligently combining their values.

**Parameters:**
- `df`: The DataFrame containing possible duplicates
- `duplicate_columns`: List of column base names that may have duplicates

**Returns:**
- Updated DataFrame with deduplicated columns

**Behavior:**
- For each column in `duplicate_columns`, finds and combines the `_x` and `_y` variants
- Uses intelligent logic for combination:
  - If both values are empty/NaN, result is empty
  - If one is empty, uses the non-empty value
  - If both are non-empty and identical, uses that value
  - If both are non-empty and different, combines with a pipe separator (|)

**Example:**

```python
import pandas as pd
from pie.med_hist_loader import deduplicate_columns

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

### `load_ppmi_medical_history(folder_path)`

Main function that loads all medical history files from the PPMI dataset and organizes them into a dictionary.

**Parameters:**
- `folder_path`: Path to the 'Medical_History' folder containing CSV files

**Returns:**
- A dictionary where keys are file prefixes and values are DataFrames containing the loaded data

**Behavior:**
- Searches recursively through the provided folder path for CSV files matching `MEDICAL_HISTORY_PREFIXES`
- Loads each file into a DataFrame and sanitizes column names
- Returns all loaded DataFrames in a dictionary, keyed by their prefix

**Example:**

```python
import pandas as pd
from pie.med_hist_loader import load_ppmi_medical_history

# Load all medical history data
data_path = "./PPMI/Medical_History"
med_history = load_ppmi_medical_history(data_path)

# Check what files were loaded
print("Loaded medical history files:", list(med_history.keys()))

# Look at sample data for concomitant medications
if "Concomitant_Medication" in med_history:
    conmeds = med_history["Concomitant_Medication"]
    print(f"Concomitant medications data: {len(conmeds)} rows, {len(conmeds.columns)} columns")
    print(conmeds[["PATNO", "EVENT_ID", "CMTRT"]].head())
```

## Integration with DataPreprocessor

The `data_preprocessor.py` module provides cleaning functions that work with the data loaded by `med_hist_loader.py`. Here's how they work together:

### Cleaning Medical History Data

The `DataPreprocessor.clean_medical_history()` function takes the dictionary returned by `load_ppmi_medical_history()` and applies specific cleaning functions to each type of medical history data.

Currently implemented cleaners include:
- `clean_ledd_meds`: Processes levodopa-equivalent medication data, including start/stop dates and LEDD calculation
- `clean_concomitant_meds`: Processes medication data, including start/stop dates and indication mapping
- `clean_vital_signs`: Calculates blood pressure categories from raw values
- `clean_features_of_parkinsonism`: Standardizes feature classifications
- `clean_gen_physical_exam`: Standardizes abnormality classifications

**Example of Data Cleaning Workflow:**

```python
import pandas as pd
from pie.data_loader import DataLoader
from pie.data_preprocessor import DataPreprocessor
from pie.med_hist_loader import load_ppmi_medical_history

# Step 1: Load the medical history data
data_path = "./PPMI/Medical_History"
med_history = load_ppmi_medical_history(data_path)

# Step 2: Clean the medical history data
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)

# Step 3: Examine the cleaned concomitant medication data
conmeds = cleaned_med_history["Concomitant_Medication"]
print("Concomitant medications after cleaning:")
print(conmeds[["PATNO", "STARTDT", "STOPDT", "CMINDC", "CMINDC_TEXT"]].head())

# Step 4: Examine processed vital signs
vitals = cleaned_med_history["Vital_Signs"]
print("\nVital Signs with blood pressure categories:")
print(vitals[["PATNO", "EVENT_ID", "SYSSUP", "DIASUP", "Sup BP code", "Sup BP label"]].head())

# Step 5: Use the data within a larger workflow
# Example: Creating a complete data dictionary with all modalities
all_data = {}
all_data[DataLoader.MEDICAL_HISTORY] = cleaned_med_history
# Now you can add other data modalities and process everything together
```

## Detailed Examples

### Example 1: Loading and Exploring Medication Data

```python
import pandas as pd
import matplotlib.pyplot as plt
from pie.med_hist_loader import load_ppmi_medical_history
from pie.data_preprocessor import DataPreprocessor

# Load and clean medical history data
data_path = "./PPMI/Medical_History"
med_history = load_ppmi_medical_history(data_path)
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)

# Focus on concomitant medications
conmeds = cleaned_med_history["Concomitant_Medication"]

# Count medications by indication
indication_counts = conmeds["CMINDC_TEXT"].value_counts()
top_indications = indication_counts.head(10)

# Create visualization
plt.figure(figsize=(12, 6))
top_indications.plot(kind='barh')
plt.title('Top 10 Medication Indications')
plt.xlabel('Number of Medications')
plt.tight_layout()
plt.savefig('medication_indications.png')

# Analyze REM Behavior Disorder-specific medications
rbd_meds = conmeds[conmeds["CMINDC"]==18]
print(f"Found {len(rbd_meds)} RBD medication entries")

# Analyze how many patients are on RBD medications
rbd_patients = rbd_meds["PATNO"].nunique()
total_patients = conmeds["PATNO"].nunique()
print(f"{rbd_patients} out of {total_patients} patients ({rbd_patients/total_patients:.1%}) are on RBD medications")
```

### Example 2: Analyzing Vital Signs Over Time

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pie.med_hist_loader import load_ppmi_medical_history
from pie.data_preprocessor import DataPreprocessor

# Load and clean medical history data
data_path = "./PPMI/Medical_History"
med_history = load_ppmi_medical_history(data_path)
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)

# Get vital signs data
vitals = cleaned_med_history["Vital_Signs"]

# Define visit order for proper plotting
visit_order = ['BL', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12']

# Convert EVENT_ID to categorical with proper order
vitals['EVENT_ID'] = pd.Categorical(vitals['EVENT_ID'], categories=visit_order, ordered=True)

# Sort data
vitals_sorted = vitals.sort_values(['PATNO', 'EVENT_ID'])

# Analyze blood pressure distribution across visits
plt.figure(figsize=(14, 8))
sns.boxplot(x='EVENT_ID', y='SYSSUP', data=vitals_sorted)
plt.title('Supine Systolic Blood Pressure by Visit')
plt.xlabel('Visit')
plt.ylabel('Systolic BP (mmHg)')
plt.grid(True, alpha=0.3)
plt.savefig('bp_by_visit.png')

# Analyze hypertension prevalence by visit
# Group by visit and calculate percentage in each BP category
bp_by_visit = vitals_sorted.groupby('EVENT_ID')['Sup BP label'].value_counts(normalize=True).unstack() * 100

# Plot as stacked bar chart
plt.figure(figsize=(14, 8))
bp_by_visit.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Blood Pressure Categories by Visit')
plt.xlabel('Visit')
plt.ylabel('Percentage (%)')
plt.legend(title='BP Category')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bp_categories_by_visit.png')
```

### Example 3: Combining Medical History with Subject Characteristics

```python
import pandas as pd
from pie.med_hist_loader import load_ppmi_medical_history
from pie.data_preprocessor import DataPreprocessor
from pie.sub_char_loader import load_ppmi_subject_characteristics

# Load data 
data_path = "./PPMI"
med_history = load_ppmi_medical_history(f"{data_path}/Medical_History")
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)
subject_data = load_ppmi_subject_characteristics(f"{data_path}/_Subject_Characteristics")

# Get features of parkinsonism
parkinsonism = cleaned_med_history["Features_of_Parkinsonism"]

# Merge with subject data
combined_data = pd.merge(
    parkinsonism,
    subject_data[['PATNO', 'EVENT_ID', 'GENDER', 'COHORT', 'AGE']],
    on=['PATNO', 'EVENT_ID'],
    how='left'
)

# Analyze parkinsonian features by cohort at baseline
baseline_data = combined_data[combined_data['EVENT_ID'] == 'BL']

# Calculate mean feature scores by cohort
features = ['FEATBRADY', 'FEATPOSINS', 'FEATRIGID', 'FEATTREMOR']
feature_means = baseline_data.groupby('COHORT')[features].mean()

print("Parkinsonian Features by Cohort (Baseline):")
print(feature_means)

# Save the analysis
feature_means.to_csv('parkinsonian_features_by_cohort.csv')

# Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
feature_means.plot(kind='bar')
plt.title('Parkinsonian Features by Cohort (Baseline)')
plt.xlabel('Cohort')
plt.ylabel('Mean Score (0-1)')
plt.ylim(0, 1)
plt.legend(title='Feature')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('parkinsonian_features.png')
```

### Example 4: Analyzing the Timing of Disease Milestones

```python
import pandas as pd
import matplotlib.pyplot as plt
from pie.med_hist_loader import load_ppmi_medical_history
from pie.data_preprocessor import DataPreprocessor

# Load and clean medical history data
data_path = "./PPMI/Medical_History"
med_history = load_ppmi_medical_history(data_path)
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)

# Get initiation of dopaminergic therapy data
therapy_init = cleaned_med_history.get("Initiation_of_Dopaminergic_Therapy", None)

if therapy_init is not None:
    # Analyze time to dopaminergic therapy
    if "INFODT" in therapy_init.columns and "TREATDT" in therapy_init.columns:
        # Convert dates to datetime
        therapy_init["INFODT"] = DataPreprocessor.dt_to_datetime(therapy_init["INFODT"])
        therapy_init["TREATDT"] = DataPreprocessor.dt_to_datetime(therapy_init["TREATDT"])
        
        # Calculate time difference in months
        therapy_init["TIME_TO_THERAPY"] = (therapy_init["TREATDT"] - therapy_init["INFODT"]).dt.days / 30.44  # Average month length
        
        # Plot distribution of time to therapy
        plt.figure(figsize=(10, 6))
        plt.hist(therapy_init["TIME_TO_THERAPY"].dropna(), bins=20)
        plt.title('Time to Initiation of Dopaminergic Therapy')
        plt.xlabel('Months')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
        plt.savefig('time_to_therapy.png')
        
        # Calculate summary statistics
        print("Time to Dopaminergic Therapy (months):")
        print(therapy_init["TIME_TO_THERAPY"].describe())
```

## Common Workflows

### Loading and Cleaning Medical History Data

```python
from pie.med_hist_loader import load_ppmi_medical_history
from pie.data_preprocessor import DataPreprocessor

# Load raw medical history data
med_history = load_ppmi_medical_history("./PPMI/Medical_History")

# Clean and process the data
cleaned_med_history = DataPreprocessor.clean_medical_history(med_history)
```

### Integrating with the Complete Data Pipeline

```python
from pie.data_loader import DataLoader
from pie.data_preprocessor import DataPreprocessor

# Load all data modalities
data = DataLoader.load("./PPMI", source="PPMI")

# Clean all data
cleaned_data = DataPreprocessor.clean(data)

# Access medical history data
med_history = cleaned_data[DataLoader.MEDICAL_HISTORY]

# Use specific types of medical history
conmeds = med_history["Concomitant_Medication"]
vitals = med_history["Vital_Signs"]
features = med_history["Features_of_Parkinsonism"]
```

## Working with Specific Medical History Files

### Concomitant Medications

The concomitant medications data contains information about medications patients are taking, including:
- Start and stop dates
- Medication names
- Indications (reasons for taking the medication)

After cleaning, this data includes:
- Properly formatted datetime objects for start/stop dates
- Mapped indication codes and text descriptions
- Standardized medication classifications

```python
# Access concomitant medications
conmeds = cleaned_med_history["Concomitant_Medication"]

# Find patients on specific medications. NOTE: this does not take account of typos, of which there are many!
lorazepam_patients = conmeds[conmeds["CMTRT"].str.contains("lorazepam", case=False, na=False)]["PATNO"].unique()
print(f"Found {len(lorazepam_patients)} patients on lorazepam")

# Analyze medication indications. This is much more robust than searching by name.
anxiety_meds = conmeds[conmeds["CMINDC_TEXT"] == "Anxiety"]
print(f"Found {len(anxiety_meds)} medication entries for anxiety")
```

### Vital Signs

The vital signs data contains measurements like:
- Blood pressure (supine and standing)
- Heart rate
- Weight and height

After cleaning, this data includes derived fields like blood pressure categories according to AHA guidelines.

```python
# Access vital signs data
vitals = cleaned_med_history["Vital_Signs"]

# Find patients with hypertensive readings
hypertensive = vitals[vitals["Sup BP code"] >= 2]  # Stage 1 HTN or higher
print(f"Found {len(hypertensive)} hypertensive readings across {hypertensive['PATNO'].nunique()} patients")

# Calculate BMI if height and weight are available
if "HEIGHT" in vitals.columns and "WEIGHT" in vitals.columns:
    vitals["BMI"] = vitals["WEIGHT"] / ((vitals["HEIGHT"]/100) ** 2)
    print("BMI Summary:")
    print(vitals["BMI"].describe())
```

### Features of Parkinsonism

This file contains assessments of core parkinsonian features:
- Bradykinesia (slowness of movement)
- Rigidity
- Tremor
- Postural instability

After cleaning, uncertain assessments are represented as intermediate values for easier analysis.

```python
# Access features of parkinsonism data
parkinsonism = cleaned_med_history["Features_of_Parkinsonism"]

# Calculate a total motor feature score
parkinsonism["TOTAL_FEATURE_SCORE"] = parkinsonism[["FEATBRADY", "FEATPOSINS", "FEATRIGID", "FEATTREMOR"]].sum(axis=1)

# Analyze which feature appears first/most commonly
feature_prevalence = {
    "Bradykinesia": (parkinsonism["FEATBRADY"] > 0).mean(),
    "Postural Instability": (parkinsonism["FEATPOSINS"] > 0).mean(),
    "Rigidity": (parkinsonism["FEATRIGID"] > 0).mean(),
    "Tremor": (parkinsonism["FEATTREMOR"] > 0).mean()
}

print("Feature Prevalence:")
for feature, prevalence in feature_prevalence.items():
    print(f"  {feature}: {prevalence:.1%}")
```

## Best Practices

1. **Keep Medical History Files Separate**: Unlike other data types, medical history data is best kept as separate tables rather than merged into a single table.

2. **Use Appropriate Joins**: When joining medical history data with other data types, consider the temporal nature of the data:
   - Some medical history data is associated with specific visits (use EVENT_ID)
   - Other data spans multiple visits (use date ranges or custom logic)
   - See the `notebooks` for examples of how to work with temporal data.

3. **Handle Duplicates Carefully**: Some medical history files may have multiple entries per patient per visit.

4. **Apply Cleaning Functions**: Always use the DataPreprocessor cleaning functions to ensure data is properly formatted and standardized.

5. **Check for Missing Values**: Medical history data often contains missing values that require special handling.

## Troubleshooting

### File Not Found

If certain medical history files are not found:

```python
# Check available files in the Medical_History directory
import os
files = [f for f in os.listdir("./PPMI/Medical_History") if f.endswith('.csv')]
print("Available files:", files)
```

### Empty DataFrames

If a DataFrame in the dictionary is empty:

```python
# Check all DataFrames in the dictionary
for prefix, df in med_history.items():
    if df.empty:
        print(f"Warning: {prefix} DataFrame is empty")
    else:
        print(f"{prefix}: {len(df)} rows, {len(df.columns)} columns")
```

### Date Parsing Errors

If date parsing fails in concomitant medications:

```python
# Check date formats
conmeds = med_history["Concomitant_Medication"]
print("Start date format examples:")
print(conmeds["STARTDT"].dropna().head())
```
