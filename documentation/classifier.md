# PIE Classifier and Reporting Documentation

## Overview

The PIE classification module is a powerful, automated pipeline for training, evaluating, and comparing a suite of machine learning models. It is designed to take a feature-engineered dataset and produce a comprehensive report detailing model performance, feature importance, and visualizations.

The module consists of two main components:
1.  **`classification_report.py`**: A high-level script that orchestrates the entire classification pipeline. This is the **primary entry point** for most users. It handles data loading, optional feature selection, model comparison, tuning, and report generation.
2.  **`classifier.py`**:  It provides a stateful `Classifier` class with fine-grained control over model setup, training, and evaluation. Advanced users can interact with this class directly.

This document will first cover the main pipeline script, which is the recommended way to use the module, and then detail the underlying `Classifier` class for advanced use cases.

---

## 1. The Main Pipeline: `generate_report()`

The `generate_report()` function in `classification_report.py` runs the end-to-end classification workflow. It's the simplest way to get from a prepared dataset to a full model comparison report.

### API Reference

```python
def generate_report(
    input_csv_path: str = None,
    train_csv_path: str = None,
    test_csv_path: str = None,
    use_feature_selection: bool = True,
    feature_selection_method: str = 'univariate_kbest',
    target_column: str = "COHORT",
    exclude_features: List[str] = None,
    output_dir: str = "output",
    n_models_to_compare: int = 5,
    tune_best_model: bool = True,
    generate_plots: bool = True,
    budget_time_minutes: float = 30.0
):
```

#### Parameters

-   **`input_csv_path`** `(str, optional)`: Path to a single, engineered CSV file. The script will split this into training and testing sets. This is used if `train_csv_path` and `test_csv_path` are not provided.
-   **`train_csv_path`** `(str, optional)`: Path to a pre-split training data CSV. Must be used with `test_csv_path`.
-   **`test_csv_path`** `(str, optional)`: Path to a pre-split testing data CSV. Must be used with `train_csv_path`.
-   **`use_feature_selection`** `(bool, default=True)`: If `True`, runs a feature selection step before model training. Set this to `False` if your data has already been feature-selected.
-   **`feature_selection_method`** `(str, default='univariate_kbest')`: The method to use for feature selection. Corresponds to the methods in the `FeatureSelector` class (e.g., `'k_best'`, `'fdr'`).
-   **`target_column`** `(str, default="COHORT")`: The name of the column to be predicted.
-   **`exclude_features`** `(List[str], optional)`: A crucial list of features to remove *before* any training or evaluation. This is used to prevent **data leakage** by removing columns that are too closely related to the target or would not be available at the time of prediction.
-   **`output_dir`** `(str, default="output")`: The directory where all outputs (reports, plots, saved models) will be stored.
-   **`n_models_to_compare`** `(int, default=5)`: The number of top models to include in the comparison leaderboard.
-   **`tune_best_model`** `(bool, default=True)`: If `True`, the script will take the best-performing model from the comparison and perform hyperparameter tuning to further optimize it.
-   **`generate_plots`** `(bool, default=True)`: If `True`, generates and saves performance plots (e.g., Confusion Matrix, ROC-AUC, Feature Importance, t-SNE) to the `plots` subdirectory.
-   **`budget_time_minutes`** `(float, default=30.0)`: A time limit in minutes for the `compare_models` step. This prevents the process from hanging on slow models.

### How to Run the Pipeline (`test_classifier.py`)

The easiest way to run the classification pipeline is via the provided test script, `tests/test_classifier.py`.

#### Prerequisites

-   The test script **requires the outputs from the feature selection workflow**. You must first run `tests/test_feature_selector.py` successfully. The script specifically looks for:
    -   `output/selected_train_data.csv`
    -   `output/selected_test_data.csv`

#### Running the Script

No special parameters are needed. From the root directory of the PIE project, run:

```bash
python3 tests/test_classifier.py
```

#### What the Test Script Does

1.  **Defines `leakage_features`**: It specifies a list of features that should be excluded to prevent data leakage. This is a critical step for building a realistic model. **You should review and customize this list for your specific research question.**
2.  **Calls `generate_report`**: It invokes the main pipeline function with the pre-split, feature-selected data.
3.  **Produces Output**: It creates a new `classification_report.html` in the `output` directory and attempts to open it in your browser. It also saves the final trained model as `output/final_classifier_model.pkl`.

### The HTML Report

The pipeline generates a comprehensive HTML report summarizing the entire experiment, including:
-   An overview of the dataset and parameters.
-   A list of any features excluded to prevent data leakage.
-   A leaderboard comparing the performance of all evaluated models.
-   Detailed metrics and hyperparameters for the best-performing model.
-   A gallery of performance plots (Confusion Matrix, ROC Curve, Feature Importance, t-SNE, etc.).
-   Model interpretation plots (e.g., SHAP summary).

---

## 2. The Engine: `Classifier` Class

For advanced users who need more granular control, the `Classifier` class in `pie/classifier.py` can be used directly. 

### Key Methods

-   **`setup_experiment(...)`**: Initializes the PyCaret environment. This **must** be called before any other method. It handles data splitting, preprocessing, and experiment logging setup.
-   **`compare_models(...)`**: Trains and evaluates a pool of models, returning a leaderboard of the results.
-   **`create_model(...)`**: Creates and trains a single specified model (e.g., `'rf'` for Random Forest).
-   **`tune_model(...)`**: Performs hyperparameter tuning on a given model.
-   **`plot_model(...)`**: Generates a specific performance plot for a model.
-   **`interpret_model(...)`**: Generates SHAP-based interpretation plots.
-   **`predict_model(...)`**: Makes predictions on new data.
-   **`save_model(...)` / `load_model(...)`**: Saves and loads the trained model pipeline.

### Advanced Usage Example

This example shows how to use the `Classifier` class directly to build, tune, and evaluate a Random Forest model.

```python
import pandas as pd
from pie.classifier import Classifier

# Assume train_df and test_df are loaded
# train_df = pd.read_csv("output/selected_train_data.csv")
# test_df = pd.read_csv("output/selected_test_data.csv")

# 1. Initialize the Classifier
clf = Classifier()

# 2. Set up the experiment
# We provide both train and test data to PyCaret
clf.setup_experiment(
    data=train_df,
    test_data=test_df,
    target='COHORT',
    session_id=42,
    verbose=False # Keep console output clean for this example
)

# 3. Create a specific model instead of comparing all
print("Creating Random Forest model...")
rf_model = clf.create_model('rf') # 'rf' is the ID for Random Forest

# 4. Tune the created model
print("Tuning Random Forest model...")
tuned_rf = clf.tune_model(rf_model, n_iter=25, optimize='F1') # Tune for F1-score

# 5. Plot results for the tuned model
print("Generating plots...")
clf.plot_model(tuned_rf, plot='confusion_matrix', save=True)
clf.plot_model(tuned_rf, plot='feature', save=True)

# 6. Make predictions on the test set
print("Making predictions...")
predictions = clf.predict_model(tuned_rf)
print("Sample predictions:")
print(predictions.head())

# 7. Save the final model
clf.save_model(tuned_rf, model_name='my_tuned_rf_model')
print("\nWorkflow complete. Tuned model saved as 'my_tuned_rf_model.pkl'.")
```
```