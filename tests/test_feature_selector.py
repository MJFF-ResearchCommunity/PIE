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

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FeatureSelector class is not directly used here as we are applying a custom sequence
# from pie.feature_selector import FeatureSelector 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.test_feature_selector")

def generate_feature_selection_report_html(report_data: dict, output_html_path: str):
    """Generates an HTML report summarizing the feature selection process."""
    logger.info(f"Generating HTML report at: {output_html_path}")

    html_style = """
    <style>
        body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h1 { text-align: center; color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #ecf0f1; }
        .summary-box { border: 1px solid #bdc3c7; padding: 15px; margin-bottom: 20px; background-color: #f8f9f9; border-radius: 5px; }
        .code { background-color: #e8e8e8; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', Courier, monospace; }
        .highlight { color: #e67e22; font-weight: bold; }
        ul { list-style-type: square; padding-left: 20px; }
        li { margin-bottom: 5px; }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Feature Selection Report</title>
        {html_style}
    </head>
    <body>
        <div class="container">
            <h1>PIE Feature Selection Report</h1>

            <div class="summary-box">
                <h2>1. Initial Data Summary</h2>
                <p><strong>Input Dataset:</strong> <span class="code">{report_data.get('input_csv_path', 'N/A')}</span></p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Raw Loaded Data Shape</td><td>{report_data.get('raw_data_shape', 'N/A')}</td></tr>
                    <tr><td>Rows Dropped (Missing Target)</td><td>{report_data.get('rows_dropped_missing_target', 'N/A')}</td></tr>
                    <tr><td>Data Shape (After Handling Missing Target)</td><td>{report_data.get('clean_data_shape', 'N/A')}</td></tr>
                    <tr><td>Target Column</td><td><span class="code">{report_data.get('target_column', 'N/A')}</span></td></tr>
                </table>
            </div>

            <div class="summary-box">
                <h2>2. Train-Test Split</h2>
                <table>
                    <tr><th>Dataset</th><th>Shape</th></tr>
                    <tr><td>X_train</td><td>{report_data.get('X_train_shape', 'N/A')}</td></tr>
                    <tr><td>y_train</td><td>{report_data.get('y_train_shape', 'N/A')}</td></tr>
                    <tr><td>X_test</td><td>{report_data.get('X_test_shape', 'N/A')}</td></tr>
                    <tr><td>y_test</td><td>{report_data.get('y_test_shape', 'N/A')}</td></tr>
                </table>
            </div>

            <div class="summary-box">
                <h2>3. Preprocessing (Imputation, Scaling, One-Hot Encoding)</h2>
                <p>Applied to X_train (fit and transform) and X_test (transform).</p>
                <table>
                    <tr><th>Dataset</th><th>Shape Before</th><th>Shape After</th><th>Features Before</th><th>Features After</th></tr>
                    <tr>
                        <td>X_train</td>
                        <td>{report_data.get('X_train_shape', 'N/A')}</td>
                        <td>{report_data.get('X_train_processed_shape', 'N/A')}</td>
                        <td>{report_data.get('X_train_shape', ('N/A', 'N/A'))[1]}</td>
                        <td>{report_data.get('X_train_processed_shape', ('N/A', 'N/A'))[1]}</td>
                    </tr>
                    <tr>
                        <td>X_test</td>
                        <td>{report_data.get('X_test_shape', 'N/A')}</td>
                        <td>{report_data.get('X_test_processed_shape', 'N/A')}</td>
                        <td>{report_data.get('X_test_shape', ('N/A', 'N/A'))[1]}</td>
                        <td>{report_data.get('X_test_processed_shape', ('N/A', 'N/A'))[1]}</td>
                    </tr>
                </table>
            </div>

            <div class="summary-box">
                <h2>4. Feature Selection Steps</h2>
                
                <h3>4.1 Variance Threshold</h3>
                <p><strong>Threshold Value:</strong> <span class="highlight">{report_data.get('variance_threshold_val', 'N/A')}</span></p>
                <table>
                    <tr><th>Dataset</th><th>Shape Before VT</th><th>Shape After VT</th><th>Features Dropped by VT (Train)</th></tr>
                    <tr>
                        <td>X_train_processed</td>
                        <td>{report_data.get('X_train_processed_shape', 'N/A')}</td>
                        <td>{report_data.get('X_train_var_thresh_shape', 'N/A')}</td>
                        <td rowspan="2" style="vertical-align:middle;">{report_data.get('features_dropped_vt_train', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>X_test_processed</td>
                        <td>{report_data.get('X_test_processed_shape', 'N/A')}</td>
                        <td>{report_data.get('X_test_var_thresh_shape', 'N/A')}</td>
                    </tr>
                </table>

                <h3>4.2 SelectFdr (False Discovery Rate)</h3>
                <p><strong>Alpha (FDR Control):</strong> <span class="highlight">{report_data.get('fdr_alpha', 'N/A')}</span></p>
                <p><strong>Scoring Function:</strong> <span class="code">f_classif</span> (for classification target)</p>
                <table>
                    <tr><th>Dataset</th><th>Shape Before FDR</th><th>Shape After FDR</th><th>Features Dropped by FDR (Train)</th></tr>
                     <tr>
                        <td>X_train_var_thresh</td>
                        <td>{report_data.get('X_train_var_thresh_shape', 'N/A')}</td>
                        <td>{report_data.get('X_train_fdr_shape', 'N/A')}</td>
                        <td rowspan="2" style="vertical-align:middle;">{report_data.get('features_dropped_fdr_train', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>X_test_var_thresh</td>
                        <td>{report_data.get('X_test_var_thresh_shape', 'N/A')}</td>
                        <td>{report_data.get('X_test_fdr_shape', 'N/A')}</td>
                    </tr>
                </table>
            </div>

            <div class="summary-box">
                <h2>5. Final Selected Features</h2>
                <p><strong>Total Selected Features:</strong> <span class="highlight">{report_data.get('num_final_selected_features', 'N/A')}</span></p>
    """
    if report_data.get('final_selected_feature_names'):
        html_content += "<strong>List of Selected Features:</strong><ul>"
        for feature_name in report_data.get('final_selected_feature_names', []):
            html_content += f"<li><span class='code'>{feature_name}</span></li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No features were selected or list is unavailable.</p>"
    html_content += """
            </div>

            <div class="summary-box">
                <h2>6. Final Output Data</h2>
                <table>
                    <tr><th>Dataset</th><th>Final Shape</th><th>Saved To</th></tr>
                    <tr>
                        <td>Selected Training Data</td>
                        <td>{report_data.get('final_train_data_shape', 'N/A')}</td>
                        <td><span class="code">{report_data.get('output_train_csv_path', 'N/A')}</span></td>
                    </tr>
                    <tr>
                        <td>Selected Test Data</td>
                        <td>{report_data.get('final_test_data_shape', 'N/A')}</td>
                        <td><span class="code">{report_data.get('output_test_csv_path', 'N/A')}</span></td>
                    </tr>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_html_path}: {e}")


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