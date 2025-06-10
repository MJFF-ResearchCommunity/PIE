#!/usr/bin/env python3
"""
Master pipeline script for the PIE project.

This script orchestrates the entire workflow from data loading to classification,
generating reports at each stage and a final summary report.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
from datetime import datetime
from typing import List, Dict, Optional
import time

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import PIE modules
from pie.data_loader import DataLoader
from pie.data_reducer import DataReducer
from pie.feature_engineer import FeatureEngineer
from pie.classification_report import generate_report as run_classification_step
from pie.feature_selector import FeatureSelector
from pie.reporting import (
    generate_data_reduction_html_report,
    generate_feature_engineering_report_html,
    generate_feature_selection_report_html
)

# Imports for feature selection step
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFdr, f_classif

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PIE.pipeline")


# --- TIMING DECORATOR ---
def timing_decorator(func):
    """A simple decorator to log the execution time of a function."""
    def wrapper(*args, **kwargs):
        logger.info(f"--- Timing: Starting '{func.__name__}' ---")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"--- Timing: Finished '{func.__name__}' in {elapsed_time:.2f} seconds ---")
        return result
    return wrapper


# --- REPORTING HELPER FUNCTIONS (ADAPTED FROM TEST SCRIPTS) ---

def _calculate_dict_size(data_dict: dict) -> float:
    """Calculates the total memory usage of DataFrames in a dictionary (in MB)."""
    total_size_bytes = 0
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            try:
                total_size_bytes += value.memory_usage(deep=True, index=True).sum()
            except Exception as e:
                 logging.warning(f"Could not calculate size for DataFrame '{key}': {e}")
        elif isinstance(value, dict):
            total_size_bytes += _calculate_dict_size(value)
    return total_size_bytes / (1024 * 1024)

def _get_dict_summary(data_dict: dict) -> dict:
    """Calculates shape and basic stats for each DataFrame, returns as dict."""
    summary = {}
    total_rows_sum, total_cols_sum, total_numeric_cols_sum, total_object_cols_sum, total_nulls_sum, total_cells_sum, dataframe_count = 0, 0, 0, 0, 0, 0, 0
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                dataframe_count += 1
                rows, cols = value.shape
                total_rows_sum += rows; total_cols_sum += cols
                null_sum = value.isnull().sum().sum(); df_cells = rows * cols
                total_nulls_sum += null_sum; total_cells_sum += df_cells
                summary[key] = {"shape": (rows, cols), "null_pct": (null_sum / df_cells) * 100 if df_cells > 0 else 0,
                                "numeric_cols": value.select_dtypes(include=np.number).shape[1],
                                "object_cols": value.select_dtypes(include='object').shape[1], "is_empty": False}
                total_numeric_cols_sum += summary[key]['numeric_cols']; total_object_cols_sum += summary[key]['object_cols']
            else:
                 summary[key] = {"shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True}
        elif isinstance(value, dict):
            if not value: summary[key] = {"shape": "Empty Dict", "is_empty": True}; continue
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, pd.DataFrame) and not sub_value.empty:
                     dataframe_count += 1; rows, cols = sub_value.shape
                     total_rows_sum += rows; total_cols_sum += cols
                     null_sum = sub_value.isnull().sum().sum(); df_cells = rows * cols
                     total_nulls_sum += null_sum; total_cells_sum += df_cells
                     summary[f"{key}.{sub_key}"] = {"shape": (rows, cols), "null_pct": (null_sum / df_cells) * 100 if df_cells > 0 else 0,
                                                   "numeric_cols": sub_value.select_dtypes(include=np.number).shape[1],
                                                   "object_cols": sub_value.select_dtypes(include='object').shape[1], "is_empty": False}
                     total_numeric_cols_sum += summary[f"{key}.{sub_key}"]['numeric_cols']; total_object_cols_sum += summary[f"{key}.{sub_key}"]['object_cols']
                 else: summary[f"{key}.{sub_key}"] = {"shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True}
    summary['totals'] = {"dataframe_count": dataframe_count, "total_rows_sum": total_rows_sum, "total_columns_sum": total_cols_sum,
                         "total_numeric_cols_sum": total_numeric_cols_sum, "total_object_cols_sum": total_object_cols_sum,
                         "overall_null_pct": (total_nulls_sum / total_cells_sum) * 100 if total_cells_sum > 0 else 0}
    return summary

def _generate_reduction_report_html(initial_dict_summary, reduced_dict_summary, analysis_report, initial_size_mb, reduced_size_mb, output_html_path, final_consolidated_df_shape):
    generate_data_reduction_html_report(initial_dict_summary, reduced_dict_summary, analysis_report, initial_size_mb, reduced_size_mb, output_html_path, final_consolidated_df_shape)

def _generate_feature_engineering_report_html(report_data, output_html_path):
    generate_feature_engineering_report_html(report_data, output_html_path)

def _generate_feature_selection_report_html(report_data, output_html_path):
    generate_feature_selection_report_html(report_data, output_html_path)


# --- PIPELINE STEPS ---

@timing_decorator
def run_data_reduction_step(data_dir: str, output_csv_path: Path, output_html_path: Path) -> dict:
    """Loads, reduces, merges, and consolidates data."""
    logger.info("Starting data loading and reduction step...")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Step cannot proceed.")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_dict = DataLoader.load(data_path=data_dir, merge_output=False)
    initial_size_mb = _calculate_dict_size(data_dict)
    initial_summary = _get_dict_summary(data_dict)

    reducer = DataReducer(data_dict)
    analysis_results = reducer.analyze()
    drop_suggestions = reducer.get_drop_suggestions(analysis_results)
    reduced_dict = reducer.apply_drops(drop_suggestions)
    
    reduced_size_mb = _calculate_dict_size(reduced_dict)
    reduced_summary = _get_dict_summary(reduced_dict)

    merged_df = reducer.merge_reduced_data(reduced_dict, output_filename=None)
    final_df = reducer.consolidate_cohort_columns(merged_df) if not merged_df.empty else pd.DataFrame()

    if not final_df.empty:
        final_df.to_csv(output_csv_path, index=False)
        logger.info(f"Final reduced and consolidated data saved to: {output_csv_path}")
    else:
        logger.warning("Final DataFrame is empty. Not saving CSV.")

    _generate_reduction_report_html(
        initial_dict_summary=initial_summary,
        reduced_dict_summary=reduced_summary,
        analysis_report=analysis_results,
        initial_size_mb=initial_size_mb,
        reduced_size_mb=reduced_size_mb,
        output_html_path=str(output_html_path),
        final_consolidated_df_shape=final_df.shape if not final_df.empty else (0, 0)
    )
    
    # Get the relative path for the main report
    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "initial_tables": initial_summary.get('totals', {}).get('dataframe_count', 0),
        "reduced_tables": reduced_summary.get('totals', {}).get('dataframe_count', 0),
        "initial_size_mb": initial_size_mb,
        "reduced_size_mb": reduced_size_mb,
        "output_shape": final_df.shape if not final_df.empty else (0,0),
        "report_path": report_path
    }

@timing_decorator
def run_feature_engineering_step(input_csv_path: str, output_csv_path: Path, output_html_path: Path) -> dict:
    """Applies feature engineering to the reduced data."""
    logger.info("Starting feature engineering step...")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}. Step cannot proceed.")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    report_data = {'input_csv_path': input_csv_path, 'output_csv_path': str(output_csv_path), 'input_data_shape': df.shape}

    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping feature engineering.")
        final_engineered_df = pd.DataFrame()
    else:
        engineer = FeatureEngineer(df.copy())
        engineer.one_hot_encode(auto_identify_threshold=20, max_categories_to_encode=25, min_frequency_for_category=0.01)
        engineer.scale_numeric_features(scaler_type='standard')
        final_engineered_df = engineer.get_dataframe()
        report_data.update(engineer.get_engineered_feature_summary())
    
    if not final_engineered_df.empty:
        final_engineered_df.to_csv(output_csv_path, index=False)
        logger.info(f"Engineered data saved to: {output_csv_path}")
    
    report_data['output_data_shape'] = final_engineered_df.shape if not final_engineered_df.empty else (0,0)
    _generate_feature_engineering_report_html(report_data, str(output_html_path))

    # Get the relative path for the main report
    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "input_shape": report_data['input_data_shape'],
        "output_shape": report_data['output_data_shape'],
        "new_features": report_data.get('feature_engineering_summary', {}).get('newly_engineered_features_count', 0),
        "report_path": report_path
    }

@timing_decorator
def run_feature_selection_step(
    input_csv_path: str,
    train_csv_path: Path,
    test_csv_path: Path,
    output_html_path: Path,
    target_column: str,
    fs_method: str,
    fs_param_value: float
) -> dict:
    """Performs feature selection on the engineered data."""
    logger.info("Starting feature selection step...")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}. Step cannot proceed.")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
        
    report_data = {'input_csv_path': input_csv_path, 'target_column': target_column}
    df = pd.read_csv(input_csv_path)
    
    initial_rows = len(df)
    df.dropna(subset=[target_column], inplace=True)
    report_data['rows_dropped_missing_target'] = initial_rows - len(df)
    report_data['clean_data_shape'] = df.shape

    if 'PATNO' in df.columns:
        df['PATNO'] = df['PATNO'].astype(int)

    id_cols = ['PATNO', 'EVENT_ID']
    feature_cols = [col for col in df.columns if col not in [target_column] + id_cols]
    
    X = df[feature_cols]
    y = df[target_column]

    # --- Start of new code ---
    # Handle pipe-separated values in object columns that are likely numeric
    X = X.copy()  # Avoid SettingWithCopyWarning

    # Define patterns for columns to skip. PATNO and EVENT_ID are already excluded
    # from X, but this handles other potential ID/date-like columns.
    ID_DATE_PATTERNS = ['ID', 'DATE', 'TIME', 'PATNO', 'EVENT']

    for col in X.select_dtypes(include=['object']).columns:
        # Skip if it looks like an ID or date column based on name patterns
        if any(pattern in col.upper() for pattern in ID_DATE_PATTERNS):
            logger.info(f"Skipping pipe-averaging for potential ID/date column: '{col}'")
            continue

        if not X[col].astype(str).str.contains('\|', na=False).any():
            continue

        # This column has pipes. Let's see if it's mostly numeric.
        logger.info(f"Column '{col}' contains pipe-separated values. Analyzing...")

        def average_pipe_values(val):
            if isinstance(val, str) and '|' in val:
                try:
                    # Split, convert to float, and average
                    return np.mean([float(x) for x in val.split('|')])
                except (ValueError, TypeError):
                    # If any part isn't a number, this value is not numeric
                    return np.nan
            return val

        # Apply the averaging function to a temporary series
        converted_series = X[col].apply(average_pipe_values)
        
        # Now, try to convert the entire series to a numeric type
        numeric_series = pd.to_numeric(converted_series, errors='coerce')

        # Heuristic: If over 90% of the original non-null values can be
        # converted to a number, we'll treat the column as numeric.
        original_non_null_count = X[col].notna().sum()
        numeric_count = numeric_series.notna().sum()

        if original_non_null_count > 0 and (numeric_count / original_non_null_count) > 0.9:
            logger.info(f"Converting column '{col}' to numeric by averaging pipe-separated values.")
            X[col] = numeric_series
        else:
            logger.warning(
                f"Column '{col}' has pipe-separated values but is not consistently numeric. "
                f"({numeric_count}/{original_non_null_count} values converted). "
                "Leaving as is."
            )
    # --- End of new code ---

    # Impute any remaining NaNs from feature engineering before selection
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(
            f"Dropping non-numeric columns before feature selection: {list(non_numeric_cols)}"
        )
        X = X.drop(columns=non_numeric_cols)

    # NOTE: Temporarily skipping SimpleImputer due to a shape mismatch error.
    # This is a workaround and the imputation strategy should be revisited.
    logger.warning("Temporarily using fillna(0) instead of SimpleImputer.")
    X_imputed = X.fillna(0)
    
    # Label encode the target for the selector
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test, y_train_encoded, _ = train_test_split(
        X_imputed, y, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    selector = FeatureSelector(
        method=fs_method,
        task_type='classification',
        k_or_frac=fs_param_value if fs_method == 'k_best' else None,
        alpha_fdr=fs_param_value if fs_method == 'fdr' else 0.05
    )
    
    selector.fit(X_train, y_train_encoded)
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)

    # Combine selected features with the original target y
    train_df = pd.concat([X_train_final.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_final.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    logger.info(f"Selected train/test data saved to {train_csv_path} and {test_csv_path}")

    report_data.update({
        'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape, 'y_test_shape': y_test.shape,
        'num_final_selected_features': X_train_final.shape[1],
        'final_train_data_shape': train_df.shape,
        'final_test_data_shape': test_df.shape,
        'output_train_csv_path': str(train_csv_path),
        'output_test_csv_path': str(test_csv_path),
        'final_selected_feature_names': selector.selected_feature_names_
    })
    _generate_feature_selection_report_html(report_data, str(output_html_path))

    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "initial_features": X.shape[1],
        "final_features": X_train_final.shape[1],
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "report_path": report_path
    }

def generate_main_report(report_data: dict, output_path: Path):
    """Generates the main pipeline report that links to sub-reports."""
    logger.info(f"Generating main pipeline report at: {output_path}")

    html_style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #212529; }
        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.05); max-width: 900px; margin: 40px auto; }
        h1, h2 { color: #343a40; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h1 { text-align: center; color: #007bff; font-size: 2.2em; }
        table { width: 100%; border-collapse: collapse; margin: 25px 0; }
        th, td { border: 1px solid #dee2e6; padding: 12px; text-align: left; }
        th { background-color: #f2f3f5; font-weight: 600; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .stage-box { border: 1px solid #e9ecef; padding: 20px; margin-bottom: 20px; background-color: #fff; border-radius: 5px; }
        .stage-title { font-size: 1.5em; color: #495057; margin-bottom: 15px; }
        .report-link { display: inline-block; background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-weight: 500; transition: background-color 0.2s; }
        .report-link:hover { background-color: #0056b3; }
        .metric { font-weight: bold; color: #28a745; }
        .timestamp { color: #6c757d; font-style: italic; text-align: right; margin-top: 20px; font-size: 0.9em; }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PIE Pipeline Overall Report</title>{html_style}</head>
    <body><div class="container">
        <h1>PIE Pipeline Run Summary</h1>
        <p class="timestamp">Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    # --- Data Reduction ---
    if 'reduction' in report_data:
        r = report_data['reduction']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">1. Data Reduction</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Size / Tables</td><td><span class="metric">{r.get('initial_size_mb', 0.0):.2f} MB</span> / {r.get('initial_tables', 'N/A')} tables</td></tr>
                <tr><td>Reduced Size / Tables</td><td><span class="metric">{r.get('reduced_size_mb', 0.0):.2f} MB</span> / {r.get('reduced_tables', 'N/A')} tables</td></tr>
                <tr><td>Final Consolidated Shape</td><td>{r.get('output_shape', 'N/A')}</td></tr>
            </table>
            <a href="{r['report_path']}" target="_blank" class="report-link">View Full Reduction Report</a>
        </div>
        """

    # --- Feature Engineering ---
    if 'feature_engineering' in report_data:
        fe = report_data['feature_engineering']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">2. Feature Engineering</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape Before Engineering</td><td>{fe.get('input_shape', 'N/A')}</td></tr>
                <tr><td>Shape After Engineering</td><td><span class="metric">{fe.get('output_shape', 'N/A')}</span></td></tr>
                <tr><td>New Features Created</td><td><span class="metric">{fe.get('new_features', 'N/A')}</span></td></tr>
            </table>
            <a href="{fe['report_path']}" target="_blank" class="report-link">View Full Engineering Report</a>
        </div>
        """

    # --- Feature Selection ---
    if 'feature_selection' in report_data:
        fs = report_data['feature_selection']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">3. Feature Selection</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Features Before Selection</td><td>{fs.get('initial_features', 'N/A')}</td></tr>
                <tr><td>Features After Selection</td><td><span class="metric">{fs.get('final_features', 'N/A')}</span></td></tr>
                <tr><td>Final Train / Test Shape</td><td>{fs.get('train_shape', 'N/A')} / {fs.get('test_shape', 'N/A')}</td></tr>
            </table>
            <a href="{fs['report_path']}" target="_blank" class="report-link">View Full Selection Report</a>
        </div>
        """
    
    # --- Classification ---
    if 'classification' in report_data:
        c = report_data['classification']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">4. Classification</h2>
            <p>The classification step compares multiple models, tunes the best one (optional), and evaluates its performance on a held-out test set.</p>
             <a href="{c['report_path']}" target="_blank" class="report-link">View Full Classification Report</a>
        </div>
        """

    html_content += "</div></body></html>"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    webbrowser.open(f"file://{os.path.realpath(output_path)}")

def run_pipeline(
    data_dir: str,
    output_dir: str,
    target_column: str,
    leakage_features_path: str,
    fs_method: str = 'fdr',
    fs_param_value: float = 0.05,
    n_models_to_compare: int = 5,
    tune_best_model: bool = False,
    generate_plots: bool = True,
    budget_time_minutes: float = 30.0,
    skip_to_step: Optional[str] = None
):
    """Executes the full PIE pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline_report_data = {}
    
    # Define file paths
    reduced_csv = output_path / "final_reduced_consolidated_data.csv"
    engineered_csv = output_path / "final_engineered_dataset.csv"
    train_csv = output_path / "selected_train_data.csv"
    test_csv = output_path / "selected_test_data.csv"

    # --- 1. Data Reduction ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 1: DATA REDUCTION ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step == 'reduction':
        pipeline_report_data['reduction'] = run_data_reduction_step(
            data_dir,
            output_csv_path=reduced_csv,
            output_html_path=output_path / "data_reduction_report.html"
        )
    
    # --- 2. Feature Engineering ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 2: FEATURE ENGINEERING ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step in ['reduction', 'engineering']:
        if not reduced_csv.exists():
            logger.error(f"{reduced_csv} not found. Cannot run feature engineering. Please run the reduction step first.")
            return
        pipeline_report_data['feature_engineering'] = run_feature_engineering_step(
            str(reduced_csv),
            output_csv_path=engineered_csv,
            output_html_path=output_path / "feature_engineering_report.html"
        )

    # --- 3. Feature Selection ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 3: FEATURE SELECTION ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step in ['reduction', 'engineering', 'selection']:
        if not engineered_csv.exists():
            logger.error(f"{engineered_csv} not found. Cannot run feature selection. Please run the engineering step first.")
            return
        pipeline_report_data['feature_selection'] = run_feature_selection_step(
            str(engineered_csv),
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            output_html_path=output_path / "feature_selection_report.html",
            target_column=target_column,
            fs_method=fs_method,
            fs_param_value=fs_param_value
        )

    # --- 4. Classification ---
    if not train_csv.exists() or not test_csv.exists():
        logger.error(f"Train/Test CSVs not found. Cannot run classification. Please run the full pipeline.")
        return
        
    logger.info("\n" + "="*80)
    logger.info("--- STEP 4: CLASSIFICATION ---")
    logger.info("="*80)
    classification_output_dir = output_path / "classification"
    exclude_features = []
    if leakage_features_path and Path(leakage_features_path).exists():
        with open(leakage_features_path, 'r') as f:
            exclude_features = [line.strip() for line in f if line.strip()]

    # Manually time this step as it's not a single decorated function
    logger.info("--- Timing: Starting 'run_classification_step' ---")
    start_time_class = time.time()
    run_classification_step(
        train_csv_path=str(train_csv),
        test_csv_path=str(test_csv),
        use_feature_selection=False,
        target_column=target_column,
        exclude_features=exclude_features,
        output_dir=str(classification_output_dir),
        n_models_to_compare=n_models_to_compare,
        tune_best_model=tune_best_model,
        generate_plots=generate_plots,
        budget_time_minutes=budget_time_minutes
    )
    end_time_class = time.time()
    logger.info(f"--- Timing: Finished 'run_classification_step' in {end_time_class - start_time_class:.2f} seconds ---")
    
    classification_report_path = classification_output_dir / "classification_report.html"
    relative_classification_report_path = Path(os.path.relpath(classification_report_path, output_path))
    pipeline_report_data['classification'] = {
        "report_path": relative_classification_report_path
    }

    # --- 5. Final Report ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 5: GENERATING FINAL REPORT ---")
    logger.info("="*80)
    generate_main_report(pipeline_report_data, output_path / "pipeline_report.html")
    logger.info("--- PIE Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PIE Automated ML Pipeline.")
    parser.add_argument('--data-dir', type=str, default='./PPMI', help='Path to raw PPMI data directory.')
    parser.add_argument('--output-dir', type=str, default='output/pipeline_run', help='Directory to save all pipeline outputs and reports.')
    parser.add_argument('--target-column', type=str, default='COHORT', help='Name of the target variable.')
    parser.add_argument('--leakage-features-path', type=str, default='config/leakage_features.txt', help='Path to a file containing features to exclude to prevent data leakage.')
    
    # Feature Selection Params
    parser.add_argument('--fs-method', type=str, default='fdr', help="Feature selection method ('fdr' or 'k_best').")
    parser.add_argument('--fs-param', type=float, default=0.05, help="Parameter for the FS method (alpha for 'fdr', k-fraction for 'k_best').")

    # Classification Params
    parser.add_argument('--n-models', type=int, default=5, help='Number of models to compare in classification.')
    parser.add_argument('--tune', action='store_true', help='Tune the best model.')
    parser.add_argument('--no-plots', action='store_false', dest='plots', help='Disable plot generation in classification.')
    parser.add_argument('--budget', type=float, default=30.0, help='Time budget in minutes for model comparison.')

    # Pipeline Control
    parser.add_argument('--skip-to', type=str, choices=['reduction', 'engineering', 'selection', 'classification'], help='Skip to a specific step of the pipeline.')

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_column=args.target_column,
        leakage_features_path=args.leakage_features_path,
        fs_method=args.fs_method,
        fs_param_value=args.fs_param,
        n_models_to_compare=args.n_models,
        tune_best_model=args.tune,
        generate_plots=args.plots,
        budget_time_minutes=args.budget,
        skip_to_step=args.skip_to
    )
