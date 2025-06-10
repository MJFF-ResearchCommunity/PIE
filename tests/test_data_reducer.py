import os
import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np # Import numpy for size calculation
from typing import Optional

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the necessary classes
from pie.data_loader import DataLoader
from pie.data_reducer import DataReducer
# FeatureEngineer will be tested separately
from pie.reporting import generate_data_reduction_html_report as generate_html_report

# Configure logging globally first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Define logger at the module level so helper functions can access it
logger = logging.getLogger("PIE.test_data_reducer")

def calculate_dict_size(data_dict: dict) -> float:
    """Calculates the total memory usage of DataFrames in a dictionary (in MB)."""
    total_size_bytes = 0
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            try:
                # Use index=True to include index size
                total_size_bytes += value.memory_usage(deep=True, index=True).sum()
            except Exception as e:
                 logging.warning(f"Could not calculate size for DataFrame '{key}': {e}") # Use logging directly here too
        elif isinstance(value, dict):
            # Recursively calculate size for nested dictionaries (like medical_history)
            total_size_bytes += calculate_dict_size(value)
    return total_size_bytes / (1024 * 1024) # Convert bytes to MB

def print_dict_summary(data_dict: dict, title: str):
    """Prints shape and basic stats for each DataFrame in the dictionary."""
    logger.info(f"\n--- {title} ---")
    # Use get_dict_summary to avoid duplicating logic for console output
    summary_data = get_dict_summary(data_dict)
    for key, stats in summary_data.items():
        if key == 'totals': # Handle totals separately
            continue
        if stats.get("is_empty"):
            logger.info(f"{key}: Shape=(0, 0) - Empty DataFrame")
        elif "shape" in stats : # Check if shape exists, for non-empty DFs
            logger.info(f"{key}: Shape={stats['shape']}, Nulls={stats['null_pct']:.1f}%, "
                        f"NumericCols={stats['numeric_cols']}, ObjectCols={stats['object_cols']}")
        elif "shape" in stats and stats["shape"] == "Empty Dict": # Handle empty dicts from get_dict_summary
             logger.info(f"{key}: (Empty Dictionary)")


    totals = summary_data.get('totals', {})
    logger.info(f"Overall Totals: DataFrames/Tables={totals.get('dataframe_count', 'N/A')}, "
                f"Total Columns={totals.get('total_columns_sum', 'N/A')}, "
                f"Numeric Cols={totals.get('total_numeric_cols_sum', 'N/A')}, "
                f"Object Cols={totals.get('total_object_cols_sum', 'N/A')}, "
                f"OverallNull%={totals.get('overall_null_pct', 0.0):.1f}%")
    logger.info("-" * (len(title) + 6))


def get_dict_summary(data_dict: dict) -> dict:
    """Calculates shape and basic stats for each DataFrame, returns as dict."""
    summary = {}
    total_rows_sum = 0 
    total_cols_sum = 0 
    total_numeric_cols_sum = 0
    total_object_cols_sum = 0
    total_nulls_sum = 0
    total_cells_sum = 0
    dataframe_count = 0

    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            if not value.empty: 
                dataframe_count += 1
                rows, cols = value.shape
                total_rows_sum += rows
                total_cols_sum += cols
                null_sum = value.isnull().sum().sum()
                df_cells = rows * cols
                total_nulls_sum += null_sum
                total_cells_sum += df_cells
                null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
                numeric_cols = value.select_dtypes(include=np.number).shape[1]
                object_cols = value.select_dtypes(include='object').shape[1]
                total_numeric_cols_sum += numeric_cols
                total_object_cols_sum += object_cols
                summary[key] = {
                    "shape": (rows, cols),
                    "null_pct": null_pct,
                    "numeric_cols": numeric_cols,
                    "object_cols": object_cols,
                    "is_empty": False
                }
            else: 
                 summary[key] = {
                    "shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True
                 }
        elif isinstance(value, dict):
            nested_summary_part = {}
            has_data_in_nested = False
            if not value: # If the dictionary itself is empty
                summary[key] = {"shape": "Empty Dict", "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True}
                continue

            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, pd.DataFrame):
                     if not sub_value.empty:
                         dataframe_count +=1
                         has_data_in_nested = True
                         rows, cols = sub_value.shape
                         total_rows_sum += rows 
                         total_cols_sum += cols
                         null_sum = sub_value.isnull().sum().sum()
                         df_cells = rows * cols
                         total_nulls_sum += null_sum
                         total_cells_sum += df_cells
                         null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
                         numeric_cols = sub_value.select_dtypes(include=np.number).shape[1]
                         object_cols = sub_value.select_dtypes(include='object').shape[1]
                         total_numeric_cols_sum += numeric_cols
                         total_object_cols_sum += object_cols
                         summary[f"{key}.{sub_key}"] = { # Use combined key for easy lookup
                            "shape": (rows, cols),
                            "null_pct": null_pct,
                            "numeric_cols": numeric_cols,
                            "object_cols": object_cols,
                            "is_empty": False
                         }
                     else:
                         summary[f"{key}.{sub_key}"] = {
                            "shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True
                         }
            if not has_data_in_nested and value: 
                 pass


    overall_null_pct = (total_nulls_sum / total_cells_sum) * 100 if total_cells_sum > 0 else 0
    summary['totals'] = {
        "dataframe_count": dataframe_count,
        "total_rows_sum": total_rows_sum,
        "total_columns_sum": total_cols_sum,
        "total_numeric_cols_sum": total_numeric_cols_sum,
        "total_object_cols_sum": total_object_cols_sum,
        "overall_null_pct": overall_null_pct
    }
    return summary

def test_data_reduction_workflow(
        output_html_path: str = "output/data_reduction_report.html",
        final_reduced_consolidated_csv_path: str = "output/final_reduced_consolidated_data.csv"
):
    """
    Tests DataLoader and DataReducer: load, analyze, reduce, merge, consolidate COHORT, and report.
    """
    logger.info("Starting data loading, reduction, and consolidation workflow test...")
    data_dir = "./PPMI" # Ensure this path is correct for your environment
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Test cannot proceed.")
        return

    # --- 1. Load Data ---
    logger.info(f"Loading all modalities from: {data_dir}")
    try:
        data_dict = DataLoader.load(data_path=data_dir, merge_output=False)
        if not data_dict:
            logger.error("DataLoader returned an empty dictionary. Stopping.")
            return
        logger.info(f"Successfully loaded {len(data_dict)} modalities.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return

    # --- 2. Initial Summary & Size ---
    initial_size_mb = calculate_dict_size(data_dict)
    initial_summary = get_dict_summary(data_dict)
    print_dict_summary(data_dict, "Initial Data Summary & Shapes")

    # --- 3. Data Reduction ---
    reducer = DataReducer(data_dict)
    logger.info("DataReducer initialized.")
    analysis_results = reducer.analyze()
    logger.info("--- Reduction Analysis Report Summary (Console) ---")
    print(reducer.generate_report_str(analysis_results))
    drop_suggestions = reducer.get_drop_suggestions(analysis_results)
    
    total_suggested_drops = sum(len(v) for v in drop_suggestions.values())
    logger.info(f"\n{total_suggested_drops} total columns suggested for dropping (Reduction Step).")

    logger.info("\nApplying drop suggestions for reduction...")
    reduced_dict = reducer.apply_drops(drop_suggestions)
    logger.info("Finished applying drops.")

    reduced_size_mb = calculate_dict_size(reduced_dict)
    reduced_summary = get_dict_summary(reduced_dict)
    print_dict_summary(reduced_dict, "Reduced Data Summary & Shapes")

    logger.info("\n--- Reduction Size Comparison (Console) ---")
    logger.info(f"Initial Size (pre-reduction): {initial_size_mb:.2f} MB")
    logger.info(f"Reduced Size (post-reduction): {reduced_size_mb:.2f} MB")
    if initial_size_mb > 0 and initial_size_mb > reduced_size_mb:
        reduction_pct = abs(1 - (reduced_size_mb / initial_size_mb)) * 100
        logger.info(f"Reduction in Size: {reduction_pct:.2f}%")

    # --- 4. Merge the Reduced Data ---
    logger.info("\nMerging the reduced DataFrames...")
    merged_df_after_reduction = pd.DataFrame() 
    if reduced_dict: 
        merged_df_after_reduction = reducer.merge_reduced_data(reduced_dict, output_filename=None) 
        if not merged_df_after_reduction.empty:
            logger.info(f"DataFrame merged post-reduction, shape: {merged_df_after_reduction.shape}")
            if not merged_df_after_reduction.duplicated(subset=['PATNO', 'EVENT_ID']).any():
                logger.info("Verified: (PATNO, EVENT_ID) pairs are unique in merged DataFrame.")
            else:
                logger.error("CRITICAL: Merged DataFrame contains duplicate (PATNO, EVENT_ID) pairs!")
        else:
            logger.warning("Merging (post-reduction) resulted in an empty DataFrame.")
    else:
        logger.warning("Reduced dictionary is empty, skipping merge. Final DataFrame will be empty.")

    # --- 5. COHORT Consolidation ---
    final_consolidated_df = merged_df_after_reduction 
    if not merged_df_after_reduction.empty:
        logger.info("\nConsolidating COHORT columns...")
        final_consolidated_df = reducer.consolidate_cohort_columns(merged_df_after_reduction)
        logger.info(f"DataFrame shape after COHORT consolidation: {final_consolidated_df.shape}")
        if "COHORT" in final_consolidated_df.columns:
            logger.info(f"COHORT column value counts after consolidation:\n{final_consolidated_df['COHORT'].value_counts(dropna=False).to_string()}")
        else:
            logger.warning("COHORT column not found after consolidation attempt.")
    else:
        logger.info("Skipping COHORT consolidation as merged DataFrame is empty.")

    # --- 6. Save Final Reduced and Consolidated Data ---
    if final_reduced_consolidated_csv_path and not final_consolidated_df.empty:
        try:
            Path(final_reduced_consolidated_csv_path).parent.mkdir(parents=True, exist_ok=True)
            final_consolidated_df.to_csv(final_reduced_consolidated_csv_path, index=False)
            logger.info(f"Final reduced and consolidated DataFrame saved to: {final_reduced_consolidated_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save final reduced and consolidated DataFrame: {e}")
    elif final_consolidated_df.empty:
        logger.warning(f"Final consolidated DataFrame is empty. Not saving to {final_reduced_consolidated_csv_path}")

    # --- 7. Generate HTML Report ---
    final_df_shape_for_report = final_consolidated_df.shape if not final_consolidated_df.empty else (0,0)

    if output_html_path:
         generate_html_report(
            initial_dict_summary=initial_summary,
            reduced_dict_summary=reduced_summary,
            analysis_report=analysis_results, 
            initial_size_mb=initial_size_mb,
            reduced_size_mb=reduced_size_mb,
            output_html_path=output_html_path,
            final_consolidated_df_shape=final_df_shape_for_report
        )


if __name__ == "__main__":
    report_file = "output/data_reduction_report.html" 
    final_csv = "output/final_reduced_consolidated_data.csv" # Updated output name
    
    output_dir = Path(report_file).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")

    test_data_reduction_workflow(
        output_html_path=report_file, 
        final_reduced_consolidated_csv_path=final_csv
    )

    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.realpath(report_file)}")
    except Exception as e:
        logger.info(f"Could not automatically open the report: {e}")