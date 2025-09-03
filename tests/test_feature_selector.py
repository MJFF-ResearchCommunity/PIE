import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFdr, f_classif
import webbrowser # For opening the report
import pytest

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.feature_selector import FeatureSelector
from pie.reporting import generate_feature_selection_report_html

# FeatureSelector class is not directly used here as we are applying a custom sequence
# from pie.feature_selector import FeatureSelector 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.test_feature_selector")

# --- Test setup for real data ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# This test depends on the output of test_pipeline.py
INPUT_CSV_PATH = PROJECT_ROOT / "output" / "test_pipeline_run" / "final_engineered_dataset.csv"
TARGET_COLUMN = "COHORT"

# Mark to skip if the required input file doesn't exist.
requires_engineered_data = pytest.mark.skipif(
    not INPUT_CSV_PATH.exists(),
    reason=f"Engineered data not found at {INPUT_CSV_PATH}. Run the full pipeline test first."
)

@requires_engineered_data
def test_feature_selector_class_with_real_data():
    """
    Tests the FeatureSelector class using a real engineered dataset.
    This test verifies that the class can be instantiated, fitted, and used
    to transform data correctly, resolving the SimpleImputer error.
    """
    logger.info(f"--- Testing FeatureSelector with real data from: {INPUT_CSV_PATH} ---")
    
    # 1. Load the data
    df = pd.read_csv(INPUT_CSV_PATH)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # 2. Separate features and target, excluding identifiers
    id_cols = ['PATNO', 'EVENT_ID']
    # Filter out id_cols that might not be present
    existing_id_cols = [col for col in id_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in [TARGET_COLUMN] + existing_id_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # 3. Split data to mimic a real scenario
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Train/Test split created. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 4. Initialize and use the FeatureSelector
    selector = FeatureSelector(
        method='k_best',
        task_type='classification',
        k_or_frac=0.25  # Select top 25% of features
    )
    logger.info(f"FeatureSelector initialized with method='k_best', k=0.25")

    # 5. Fit on training data
    logger.info("Fitting the selector...")
    selector.fit(X_train, y_train)
    logger.info("Selector fitting complete.")

    # 6. Transform both train and test data
    logger.info("Transforming train and test sets...")
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    logger.info(f"Data transformed. New X_train shape: {X_train_selected.shape}")

    # 7. Assertions
    assert isinstance(X_train_selected, pd.DataFrame), "Transformed train set should be a DataFrame"
    assert isinstance(X_test_selected, pd.DataFrame), "Transformed test set should be a DataFrame"
    
    # Assert columns are the same in train and test transformed sets
    pd.testing.assert_index_equal(X_train_selected.columns, X_test_selected.columns)

    # Assert that features were actually selected (less than original)
    assert X_train_selected.shape[1] < X_train.shape[1]
    
    # Assert that some features were selected
    assert X_train_selected.shape[1] > 0
    
    # Assert that the number of selected features is stored correctly
    assert len(selector.selected_feature_names_) == X_train_selected.shape[1]
    
    logger.info(f"Original feature count: {X_train.shape[1]}")
    logger.info(f"Selected feature count: {X_train_selected.shape[1]}")
    logger.info(f"Top 5 selected features: {selector.selected_feature_names_[:5]}")
    logger.info("--- FeatureSelector test passed successfully! ---")

def test_feature_selection_workflow(
    input_csv_path: str = "output/final_engineered_dataset.csv",
    output_train_csv_path: str = "output/selected_train_data.csv",
    output_test_csv_path: str = "output/selected_test_data.csv",
    output_report_html_path: str = "output/feature_selection_report.html", # New parameter
    target_column: str = "COHORT",
    variance_threshold_val: float = 0.01, 
    fdr_alpha: float = 0.05 
):
    """
    Tests a feature selection workflow including VarianceThreshold and SelectFdr
    on engineered data, ensuring proper train/test split handling and generates a report.
    """
    logger.info("Starting feature selection workflow test (VarianceThreshold -> SelectFdr)...")
    report_data_collector = {
        'input_csv_path': input_csv_path,
        'target_column': target_column,
        'variance_threshold_val': variance_threshold_val,
        'fdr_alpha': fdr_alpha,
        'output_train_csv_path': output_train_csv_path,
        'output_test_csv_path': output_test_csv_path,
    }

    # --- 1. Load Data ---
    input_file = Path(input_csv_path)
    if not input_file.exists():
        logger.error(f"Input data file not found: {input_csv_path}. Test cannot proceed.")
        logger.error("Please ensure 'test_data_reducer.py' (or the main workflow) has run successfully "
                     "to generate 'output/final_engineered_dataset.csv'.")
        return

    try:
        df = pd.read_csv(input_file)
        report_data_collector['raw_data_shape'] = df.shape
        logger.info(f"Successfully loaded data from: {input_csv_path}. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data from {input_csv_path}: {e}", exc_info=True)
        return

    # --- 2. Define Target and Handle Missing Target Values ---
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the DataFrame. Test cannot proceed.")
        return

    initial_rows = len(df)
    df.dropna(subset=[target_column], inplace=True)
    rows_dropped_missing_target = initial_rows - len(df)
    report_data_collector['rows_dropped_missing_target'] = rows_dropped_missing_target
    report_data_collector['clean_data_shape'] = df.shape
    if rows_dropped_missing_target > 0:
        logger.info(f"Dropped {rows_dropped_missing_target} rows due to missing target values in '{target_column}'.")
    
    if df.empty:
        logger.error(f"DataFrame is empty after handling missing target values. Test cannot proceed.")
        return
    
    logger.info(f"Target column: '{target_column}'. Unique values before encoding: {df[target_column].nunique()}")

    # --- 3. Prepare X and y ---
    X = df.drop(columns=[target_column])
    y_original = df[target_column] 

    if X.empty:
        logger.error("Feature set X is empty. Cannot proceed.")
        return

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y_original), index=y_original.index, name=target_column)
    logger.info(f"Target column '{target_column}' label encoded. Unique encoded values: {y_encoded.nunique()}")
    logger.info(f"Initial features shape (X): {X.shape}, Target shape (y_encoded): {y_encoded.shape}")

    # --- 4. Train-Test Split ---
    try:
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        _, _, y_train_original, y_test_original = train_test_split(
            X, y_original, test_size=0.2, random_state=42, stratify=y_original 
        )
    except ValueError as e:
        logger.warning(f"Could not stratify during train-test split: {e}. Splitting without stratify.")
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        _, _, y_train_original, y_test_original = train_test_split(
            X, y_original, test_size=0.2, random_state=42
        )
    
    report_data_collector['X_train_shape'] = X_train.shape
    report_data_collector['y_train_shape'] = y_train_encoded.shape # or y_train_original
    report_data_collector['X_test_shape'] = X_test.shape
    report_data_collector['y_test_shape'] = y_test_encoded.shape # or y_test_original
    logger.info(f"X_train shape: {X_train.shape}, y_train_encoded shape: {y_train_encoded.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test_encoded shape: {y_test_encoded.shape}")

    # --- 5. Preprocessing ---
    # Since the data is already engineered (scaled, one-hot encoded, etc.) from test_feature_engineer.py,
    # we should NOT re-apply these transformations. We only need to handle any remaining missing values.
    
    # First, identify numeric and non-numeric columns
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    non_numeric_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
    
    if non_numeric_features:
        logger.info(f"Found {len(non_numeric_features)} non-numeric columns. These will be dropped for variance-based selection.")
        logger.info(f"Non-numeric columns: {non_numeric_features[:10]}...")  # Show first 10
        X_train = X_train[numeric_features]
        X_test = X_test[numeric_features]
    
    # Check for missing values in numeric columns
    missing_counts = X_train.isnull().sum()
    if missing_counts.any():
        logger.info(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns. Applying simple imputation.")
        
        # First, identify columns that have at least one non-missing value
        cols_with_values = []
        cols_all_missing = []
        for col in X_train.columns:
            if X_train[col].notna().any():
                cols_with_values.append(col)
            else:
                cols_all_missing.append(col)
        
        if cols_all_missing:
            logger.info(f"Dropping {len(cols_all_missing)} columns with all missing values: {cols_all_missing[:5]}...")
            X_train = X_train.drop(columns=cols_all_missing)
            X_test = X_test.drop(columns=cols_all_missing)
        
        if cols_with_values:
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train[cols_with_values])
            X_test_imputed = imputer.transform(X_test[cols_with_values])
            
            # Update the columns with imputed values
            X_train[cols_with_values] = X_train_imputed
            X_test[cols_with_values] = X_test_imputed
    
    # No need for scaling or one-hot encoding - already done in feature engineering
    X_train_processed = X_train
    X_test_processed = X_test
    
    report_data_collector['X_train_processed_shape'] = X_train_processed.shape
    report_data_collector['X_test_processed_shape'] = X_test_processed.shape
    logger.info(f"X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")

    if X_train_processed.empty:
        logger.error("X_train became empty after preprocessing. Cannot proceed.")
        generate_feature_selection_report_html(report_data_collector, output_report_html_path)
        return

    # --- 6. Step 1: Variance Threshold ---
    logger.info(f"Applying VarianceThreshold (threshold={variance_threshold_val}) on processed training data...")
    var_thresh_selector = VarianceThreshold(threshold=variance_threshold_val)
    
    X_train_var_thresh_np = var_thresh_selector.fit_transform(X_train_processed)
    var_thresh_selected_mask = var_thresh_selector.get_support()
    X_train_var_thresh = X_train_processed.loc[:, var_thresh_selected_mask]
    report_data_collector['X_train_var_thresh_shape'] = X_train_var_thresh.shape
    report_data_collector['features_dropped_vt_train'] = X_train_processed.shape[1] - X_train_var_thresh.shape[1]
    logger.info(f"Shape after VarianceThreshold on X_train: {X_train_var_thresh.shape}. Selected {X_train_var_thresh.shape[1]} features.")

    if X_train_var_thresh.empty:
        logger.warning("No features remaining after VarianceThreshold on training data.")
        selected_train_df = y_train_original.to_frame()
        final_test_df = y_test_original.to_frame()
        report_data_collector['num_final_selected_features'] = 0
        report_data_collector['final_selected_feature_names'] = []
    else:
        X_test_var_thresh = X_test_processed.loc[:, var_thresh_selected_mask]
        report_data_collector['X_test_var_thresh_shape'] = X_test_var_thresh.shape
        logger.info(f"Shape after VarianceThreshold on X_test: {X_test_var_thresh.shape}")

        # --- 7. Step 2: SelectFdr ---
        logger.info(f"Applying SelectFdr (alpha={fdr_alpha}, score_func=f_classif) on variance-thresholded training data...")
        fdr_selector = SelectFdr(score_func=f_classif, alpha=fdr_alpha)
        
        X_train_fdr_np = fdr_selector.fit_transform(X_train_var_thresh, y_train_encoded)
        fdr_selected_mask = fdr_selector.get_support()
        X_train_fdr = X_train_var_thresh.loc[:, fdr_selected_mask]
        report_data_collector['X_train_fdr_shape'] = X_train_fdr.shape
        report_data_collector['features_dropped_fdr_train'] = X_train_var_thresh.shape[1] - X_train_fdr.shape[1]
        logger.info(f"Shape after SelectFdr on X_train: {X_train_fdr.shape}. Selected {X_train_fdr.shape[1]} features.")
        
        final_selected_feature_names = X_train_fdr.columns.tolist()
        report_data_collector['final_selected_feature_names'] = final_selected_feature_names
        report_data_collector['num_final_selected_features'] = len(final_selected_feature_names)

        if not final_selected_feature_names:
             logger.warning("No features remaining after SelectFdr on training data.")
             selected_train_df = y_train_original.to_frame()
             final_test_df = y_test_original.to_frame()
        else:
            logger.info(f"Final selected features: {final_selected_feature_names}")
            X_test_fdr = X_test_var_thresh.loc[:, fdr_selected_mask]
            report_data_collector['X_test_fdr_shape'] = X_test_fdr.shape
            logger.info(f"Shape after SelectFdr on X_test: {X_test_fdr.shape}")

            selected_train_df = pd.concat([X_train_fdr.reset_index(drop=True), y_train_original.reset_index(drop=True)], axis=1)
            final_test_df = pd.concat([X_test_fdr.reset_index(drop=True), y_test_original.reset_index(drop=True)], axis=1)

    report_data_collector['final_train_data_shape'] = selected_train_df.shape
    report_data_collector['final_test_data_shape'] = final_test_df.shape if 'final_test_df' in locals() and not final_test_df.empty else (0,0)


    # --- 8. Save Output ---
    output_dir = Path(output_train_csv_path).parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring output directory exists: {output_dir}")
    except Exception as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        generate_feature_selection_report_html(report_data_collector, output_report_html_path)
        return

    try:
        selected_train_df.to_csv(output_train_csv_path, index=False)
        logger.info(f"Selected training data saved to: {output_train_csv_path}")
        
        if 'final_test_df' in locals() and not final_test_df.empty:
            final_test_df.to_csv(output_test_csv_path, index=False)
            logger.info(f"Selected test data saved to: {output_test_csv_path}")
        else:
            logger.info(f"Final test DataFrame is empty or not created, not saving to {output_test_csv_path}.")
            # Ensure the key exists even if empty
            if 'final_test_data_shape' not in report_data_collector or report_data_collector['final_test_data_shape'] == (0,0) :
                 report_data_collector['final_test_data_shape'] = final_test_df.shape if 'final_test_df' in locals() else "N/A (Empty/Not created)"


    except Exception as e:
        logger.error(f"Failed to save output CSVs: {e}", exc_info=True)

    logger.info("Feature selection workflow test completed.")
    generate_feature_selection_report_html(report_data_collector, output_report_html_path)
    
    try: # Try to open the report
        report_abs_path = Path(output_report_html_path).resolve()
        webbrowser.open(f"file://{report_abs_path}")
        logger.info(f"Attempted to open report in browser: file://{report_abs_path}")
    except Exception as e:
        logger.info(f"Could not automatically open the report in browser: {e}")


if __name__ == "__main__":
    input_path = "output/final_engineered_dataset.csv"
    report_path = "output/feature_selection_report.html"

    test_feature_selection_workflow(
        input_csv_path=input_path,
        output_report_html_path=report_path,
        variance_threshold_val=0.001, # Adjusted for dummy data
        fdr_alpha=0.1 # Adjusted for dummy data
    )
