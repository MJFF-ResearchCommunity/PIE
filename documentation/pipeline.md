# PIE Main Pipeline Documentation

## Overview

The `pie/pipeline.py` script is the central orchestrator for the **P**arkinson's **I**nformatics **E**nvironment (PIE) library. It provides a single, powerful command-line interface to run the entire end-to-end machine learning workflow. This automated pipeline handles everything from loading raw, multi-modal data to generating a final classification report, ensuring a reproducible and standardized process.

The pipeline executes the following sequence of steps, generating intermediate data files and detailed HTML reports at each stage:

1.  **Data Reduction**: Loads all data, analyzes it for low-value features (e.g., high missingness, zero variance), reduces the feature space, and merges everything into a single consolidated CSV.
2.  **Feature Engineering**: Takes the reduced dataset and applies common feature engineering techniques, such as one-hot encoding and numeric scaling, to prepare the data for machine learning.
3.  **Feature Selection**: Applies statistical methods to select the most relevant features from the engineered dataset, further reducing dimensionality and noise.
4.  **Classification**: Compares multiple machine learning models on the final feature set, optionally tunes the best performer, and generates a comprehensive report with performance metrics and visualizations.

This streamlined approach allows users to go from raw data to actionable insights with a single command, while the modular reports provide full transparency into each step of the process.

## API Reference: `run_pipeline()`

The entire workflow is controlled by the `run_pipeline()` function, which is exposed via a command-line interface.

### Command-Line Usage

You can run the pipeline from your terminal using `python3 pie/pipeline.py` followed by the desired arguments.

```bash
python3 pie/pipeline.py --data-dir ./PPMI --output-dir ./output/my_run --target-column COHORT
```

### Command-Line Arguments

-   **`--data-dir`** `(str, default='./PPMI')`
    Path to the root directory containing the raw PPMI data folders (e.g., `_Subject_Characteristics`, `Motor___MDS-UPDRS`).

-   **`--output-dir`** `(str, default='output/pipeline_run')`
    The main directory where all outputs—including intermediate CSV files, plots, and HTML reports—will be saved.

-   **`--target-column`** `(str, default='COHORT')`
    The name of the target variable in the dataset that you want to predict.

-   **`--leakage-features-path`** `(str, default='config/leakage_features.txt')`
    A crucial parameter pointing to a text file that lists features to be excluded from the analysis. Each feature name should be on a new line. This is used to prevent **data leakage** by removing columns that are direct proxies for the target or contain information not available at prediction time.

-   **`--fs-method`** `(str, default='fdr')`
    The feature selection method to use.
    Options: `'fdr'` (False Discovery Rate), `'k_best'`.

-   **`--fs-param`** `(float, default=0.05)`
    The parameter for the chosen feature selection method.
    -   For `'fdr'`, this is the `alpha` level (p-value threshold).
    -   For `'k_best'`, this is the fraction of features to keep (e.g., `0.2` for top 20%).

-   **`--n-models`** `(int, default=5)`
    The number of top-performing models to compare and display in the classification leaderboard.

-   **`--tune`** `(action, default=False)`
    If this flag is present, the pipeline will perform hyperparameter tuning on the best model found during comparison.

-   **`--no-plots`** `(action, default=True)`
    If this flag is present, the classification step will be disabled from generating performance plots (Confusion Matrix, AUC, etc.). By default, plots are generated.

-   **`--budget`** `(float, default=30.0)`
    A time limit in minutes for the model comparison step in the classification stage. This prevents the pipeline from hanging on computationally expensive models.

-   **`--skip-to`** `(str, optional)`
    Allows you to skip to a specific step, assuming the outputs from previous steps already exist.
    Options: `'reduction'`, `'engineering'`, `'selection'`, `'classification'`.

---

## Practical Usage Example

### A Standard End-to-End Run

This is the most common use case. You have your raw data in `./PPMI` and want to run the full pipeline, saving the results to a new directory.

1.  **Review Leakage Features**: First, open `config/leakage_features.txt` and ensure the list of features to exclude is appropriate for your analysis. For example, if you are predicting `COHORT`, you should exclude columns like `APPRDX` (physician's diagnosis). The leakage features provided in this repo are just an example. Please adjust them to your needs. You want to ensure you do not include features that are nearly equivalent to the Class (e.g. COHORT). If you are aiming to identify biomarkers, you want to exclude features that were clinical observed by the physician. Furthermore, you will want to exclude features that were part of a substudy performed on a single cohort if COHORT is your target class.

2.  **Run the Pipeline**: Open your terminal and execute the following command:

    ```bash
    python3 pie/pipeline.py \
        --data-dir ./PPMI \
        --output-dir ./output/pd_vs_control_run \
        --target-column COHORT \
        --leakage-features-path config/leakage_features.txt \
        --fs-method fdr \
        --fs-param 0.05 \
        --n-models 5 \
        --tune \
        --budget 60.0
    ```

### What Happens When You Run This?

-   A new directory, `./output/pd_vs_control_run`, will be created.
-   The console will show detailed logs for each of the four pipeline steps, including timing information.
-   Four key intermediate files will be created in the output directory:
    1.  `final_reduced_consolidated_data.csv`
    2.  `final_engineered_dataset.csv`
    3.  `selected_train_data.csv`
    4.  `selected_test_data.csv`
-   Four HTML reports will be generated, one for each step.
-   A final, top-level report, `pipeline_report.html`, will be created and automatically opened in your web browser. This report links to all the sub-reports, providing a complete, browsable summary of the entire run.

---

## How to Run the Test Script

A full integration test is available in `tests/test_pipeline.py`. This test is the best way to verify that your environment is set up correctly and that all components of the PIE library are working together.

### Prerequisites

-   You must have the raw PPMI data available locally.
-   The test script expects this data to be in a directory named `PPMI` located at the root of the project. If your data is elsewhere, you will need to edit this line in `tests/test_pipeline.py`:

    ```python
    # In tests/test_pipeline.py
    PPMI_DATA_PATH = PROJECT_ROOT / "PPMI" # <-- CHANGE THIS
    ```

### Running the Test

From the root directory of the PIE project, run `pytest`:

```bash
pytest tests/test_pipeline.py
```

### What the Test Does

The test function `test_full_pipeline_with_real_data()` performs a complete, albeit expedited, run of the main pipeline.
1.  **Sets up a clean output directory** at `output/test_pipeline_run`.
2.  **Writes a default `leakage_features.txt`** to the `config/` directory.
3.  **Calls `run_pipeline()`** with parameters set for a quick run (e.g., comparing only 2 models, no tuning, 5-minute budget).
4.  **Asserts Outputs**: After the pipeline completes, it checks that all expected output files and reports have been created in the correct locations.
5.  **Performs a Sanity Check**: It reads the final training data and asserts that a known data leakage feature (`subject_characteristics_APPRDX`) was successfully removed, confirming that the pipeline's data leakage prevention mechanism is working.

Running this test successfully provides high confidence that the entire PIE ecosystem is functioning correctly.
