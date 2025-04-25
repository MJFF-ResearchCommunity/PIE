# Motor Assessment Loader Documentation

## Overview

The `motor_loader.py` module provides functions for loading, processing, and analyzing motor assessment data from the Parkinson's Progression Markers Initiative (PPMI) dataset. This module focuses on handling various motor-related assessments that track movement disorders and disease progression in Parkinson's disease patients.

## Key Features

- Load motor assessment data from multiple standardized tests
- Process raw assessment scores into usable metrics
- Calculate derived scores and summary measures
- Handle longitudinal data across multiple visits
- Standardize data formats across different assessment tools
- Clean and validate motor assessment values

## Supported Motor Assessments

The module handles various standardized motor assessments commonly used in Parkinson's research:

- **MDS-UPDRS**: Movement Disorder Society Unified Parkinson's Disease Rating Scale (Parts I-IV)
- **Modified Hoehn and Yahr Scale**: Assesses disease stage and progression
- **Schwab and England Activities of Daily Living Scale**: Evaluates independence in daily activities
- **Tremor measures**: Quantitative assessments of rest and action tremors
- **Bradykinesia metrics**: Measures of slowness of movement
- **Rigidity scores**: Assessments of muscle stiffness
- **Postural stability**: Evaluations of balance and stability

## Functions

### 1. Main Loading Function

#### `load_motor_data(data_path, source)`

Loads all motor assessment data from the specified path.

**Parameters:**
- `data_path`: Path to the data directory containing the motor assessment files
- `source`: The data source (e.g., "PPMI")

**Returns:**
- A dictionary containing processed motor assessment data with keys for each assessment type

**Example:**
```python
from pie import motor_loader

# Load all motor assessment data
data_path = "./PPMI"  # Path to your PPMI data folder
motor_data = motor_loader.load_motor_data(data_path, "PPMI")

# Print available assessment types
print(motor_data.keys())
```

### 2. Processing Functions

#### `process_updrs(updrs_data)`

Processes MDS-UPDRS assessment data, calculating sub-scores and total scores.

**Parameters:**
- `updrs_data`: Raw MDS-UPDRS data as a DataFrame

**Returns:**
- A processed DataFrame with calculated summary scores

**Example:**
```python
from pie import motor_loader
import pandas as pd

# Load raw UPDRS data
raw_updrs = pd.read_csv("./PPMI/Motor_Assessments/MDS_UPDRS_Part_I.csv")

# Process the data
processed_updrs = motor_loader.process_updrs(raw_updrs)

# View the calculated summary scores
print(processed_updrs[["PATNO", "EVENT_ID", "UPDRS1_TOTAL", "NP1CNST"]].head())
```

#### `calculate_motor_summary(motor_data)`

Calculates summary metrics across multiple motor assessments.

**Parameters:**
- `motor_data`: Dictionary containing different motor assessment data

**Returns:**
- A DataFrame with calculated summary metrics

**Example:**
```python
from pie import motor_loader

# Load all motor assessment data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")

# Calculate summary metrics
summary_metrics = motor_loader.calculate_motor_summary(motor_data)

# View the summary metrics
print(summary_metrics[["PATNO", "EVENT_ID", "TOTAL_MOTOR_SCORE", "TREMOR_SCORE", "PIGD_SCORE"]].head())
```

### 3. Utility Functions

#### `merge_motor_assessments(motor_data)`

Merges different motor assessments into a unified DataFrame.

**Parameters:**
- `motor_data`: Dictionary containing different motor assessment data

**Returns:**
- A unified DataFrame with all motor assessments

**Example:**
```python
from pie import motor_loader

# Load all motor assessment data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")

# Merge all assessments into a single DataFrame
merged_motor = motor_loader.merge_motor_assessments(motor_data)

# Save the merged data
merged_motor.to_csv("merged_motor_assessments.csv", index=False)
```

#### `extract_tremor_scores(updrs_data)`

Extracts tremor-specific scores from the MDS-UPDRS assessment.

**Parameters:**
- `updrs_data`: Processed MDS-UPDRS data as a DataFrame

**Returns:**
- A DataFrame with extracted tremor scores

**Example:**
```python
from pie import motor_loader

# Load all motor assessment data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")

# Extract tremor scores
tremor_data = motor_loader.extract_tremor_scores(motor_data["updrs"])

# View the tremor scores
print(tremor_data[["PATNO", "EVENT_ID", "REST_TREMOR_SCORE", "ACTION_TREMOR_SCORE"]].head())
```

## Example Workflows

### 1. Basic Loading and Analysis

```python
import pandas as pd
from pie import motor_loader

# Load motor assessment data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")

# Examine MDS-UPDRS Part III (motor examination)
updrs3 = motor_data["updrs_part3"]
print(f"UPDRS Part III data: {len(updrs3)} rows")

# Calculate average motor scores by visit
avg_by_visit = updrs3.groupby("EVENT_ID")["UPDRS3_TOTAL"].mean().reset_index()
print("Average motor scores by visit:")
print(avg_by_visit)

# Identify patients with severe motor symptoms (UPDRS Part III > 40)
severe_motor = updrs3[updrs3["UPDRS3_TOTAL"] > 40][["PATNO", "EVENT_ID", "UPDRS3_TOTAL"]]
print(f"Found {len(severe_motor)} assessments with severe motor symptoms")
```

### 2. Longitudinal Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from pie import motor_loader

# Load and merge motor assessment data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")
merged_motor = motor_loader.merge_motor_assessments(motor_data)

# Define visit order for longitudinal analysis
visit_order = ["BL", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12"]

# Convert EVENT_ID to categorical with proper order
merged_motor["EVENT_ID"] = pd.Categorical(merged_motor["EVENT_ID"], categories=visit_order, ordered=True)

# Select patients with complete data for first 5 visits
complete_patients = merged_motor.groupby("PATNO").filter(
    lambda x: set(x["EVENT_ID"].iloc[:5]) == set(visit_order[:5])
)["PATNO"].unique()

# Filter data to these patients and first 5 visits
longitudinal_data = merged_motor[
    (merged_motor["PATNO"].isin(complete_patients)) & 
    (merged_motor["EVENT_ID"].isin(visit_order[:5]))
]

# Plot UPDRS Part III progression
plt.figure(figsize=(10, 6))
for patno in complete_patients[:10]:  # Plot first 10 patients
    patient_data = longitudinal_data[longitudinal_data["PATNO"] == patno]
    plt.plot(patient_data["EVENT_ID"], patient_data["UPDRS3_TOTAL"], marker='o', label=f"Patient {patno}")

plt.title("UPDRS Part III Progression Over Time")
plt.xlabel("Visit")
plt.ylabel("UPDRS Part III Total Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("updrs_progression.png")
```

### 3. Comparing PD Subtypes

```python
import pandas as pd
import numpy as np
from pie import motor_loader
from pie import sub_char_loader

# Load motor data
data_path = "./PPMI"
motor_data = motor_loader.load_motor_data(data_path, "PPMI")
merged_motor = motor_loader.merge_motor_assessments(motor_data)

# Load subject characteristics
subject_data = sub_char_loader.load_ppmi_subject_characteristics(f"{data_path}/_Subject_Characteristics")

# Merge motor data with subject data
combined_data = pd.merge(
    merged_motor,
    subject_data[["PATNO", "COHORT", "SEX", "AGE"]],
    on="PATNO",
    how="inner"
)

# Define tremor-dominant vs. PIGD subtypes based on ratio
combined_data["TREMOR_PIGD_RATIO"] = combined_data["TREMOR_SCORE"] / np.maximum(combined_data["PIGD_SCORE"], 0.01)
combined_data["SUBTYPE"] = np.where(
    combined_data["TREMOR_PIGD_RATIO"] > 1.15, "Tremor-Dominant",
    np.where(combined_data["TREMOR_PIGD_RATIO"] < 0.9, "PIGD", "Mixed")
)

# Analyze motor scores by subtype
subtype_analysis = combined_data.groupby("SUBTYPE").agg({
    "UPDRS3_TOTAL": ["mean", "std", "count"],
    "TREMOR_SCORE": "mean",
    "PIGD_SCORE": "mean",
    "AGE": "mean"
}).reset_index()

print("Motor scores by PD subtype:")
print(subtype_analysis)

# Save subtype data
combined_data[["PATNO", "EVENT_ID", "SUBTYPE", "TREMOR_SCORE", "PIGD_SCORE", "UPDRS3_TOTAL"]].to_csv(
    "pd_subtypes.csv", index=False
)
```

## Data Dictionary

Common fields in motor assessment data include:

| Field | Description | Example Values |
|-------|-------------|----------------|
| PATNO | Patient identifier | 1001, 1002, ... |
| EVENT_ID | Visit identifier | BL (baseline), V01, V02, ... |
| UPDRS1_TOTAL | Total score for UPDRS Part I (non-motor aspects) | 0-52 |
| UPDRS2_TOTAL | Total score for UPDRS Part II (motor aspects of daily living) | 0-52 |
| UPDRS3_TOTAL | Total score for UPDRS Part III (motor examination) | 0-132 |
| UPDRS4_TOTAL | Total score for UPDRS Part IV (motor complications) | 0-24 |
| HYSTA | Modified Hoehn and Yahr Stage | 0-5 |
| SEADLG | Schwab & England ADL Score | 0-100% |
| TREMOR_SCORE | Composite score of tremor-related items | 0-28 |
| PIGD_SCORE | Composite score of postural instability and gait difficulty items | 0-20 |
| BRADYKINESIA_SCORE | Composite score of bradykinesia-related items | 0-48 |
| RIGIDITY_SCORE | Composite score of rigidity-related items | 0-20 |

## Implementation Details

### Data Loading Process

1. Identify motor assessment files in the specified directory
2. Load each assessment type into separate DataFrames
3. Process each assessment to calculate derived scores
4. Standardize column names and formats
5. Return a dictionary containing all processed assessments

### Motor Score Calculations

- **UPDRS Total Scores**: Sum of individual item scores within each part
- **Tremor Score**: Weighted combination of rest and action tremor items
- **PIGD Score**: Sum of postural stability, gait, and freezing items
- **Bradykinesia Score**: Sum of slowness of movement items
- **Rigidity Score**: Sum of muscle stiffness items

### Handling Missing Data

- Missing individual item scores are typically coded as 9 or 99
- Items marked as "not applicable" are excluded from calculations
- Complete assessment sections with all missing values are marked as NA
- When calculating composite scores, a minimum number of valid items is required

## Best Practices

- Always check for data completeness before analysis
- Consider comparing derived scores with official PPMI calculations
- Examine raw item scores when investigating specific motor symptoms
- Account for medication status (ON/OFF) when analyzing motor scores
- Use longitudinal modeling techniques for disease progression analysis
- Consider normalizing scores when comparing across different assessment tools

## Advanced Applications

- **Subtyping algorithms**: Using motor scores to classify PD subtypes
- **Progression prediction**: Building models to predict motor decline
- **Treatment response**: Analyzing changes in motor scores after interventions
- **Correlation with biomarkers**: Relating motor symptoms to biological measures
- **Machine learning**: Using motor assessments as features for predictive models

## Troubleshooting

- **Missing assessments**: Verify file paths and assessment availability
- **Inconsistent scores**: Check for version changes in assessment tools
- **Outlier values**: Validate extreme scores against raw assessment items
- **Longitudinal discrepancies**: Consider rater variability between visits
- **Medication effects**: Account for medication timing relative to assessments