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

def generate_html_report(
    initial_dict_summary: dict,
    reduced_dict_summary: dict,
    analysis_report: dict,
    initial_size_mb: float,
    reduced_size_mb: float,
    output_html_path: str,
    final_consolidated_df_shape: Optional[tuple] = None # Shape after merge and COHORT consolidation
):
    """Generates an HTML report summarizing the data reduction process."""
    logger.info(f"Generating HTML report at: {output_html_path}")

    html_style = """
    <style>
        body { font-family: sans-serif; line-height: 1.6; padding: 20px; }
        h1, h2, h3 { color: #333; }
        h2 { border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px;}
        .summary-box { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; border-radius: 5px; }
        .modality-section { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
        table { border-collapse: collapse; width: auto; margin-bottom: 15px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .details { margin-left: 20px; }
        .dropped-columns { background-color: #fff8f8; border: 1px dashed #fcc; padding: 10px; margin-top: 10px; border-radius: 4px;}
        .dropped-columns ul { margin: 0; padding-left: 20px; max-height: 200px; overflow-y: auto; }
        .reason { color: #888; font-style: italic; font-size: 0.9em; }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Data Reduction Report</title>
        {html_style}
    </head>
    <body>
        <h1>PIE Data Reduction Report</h1>

        <div class="summary-box">
            <h2>Overall Summary</h2>
            <table>
                <tr><th>Metric</th><th>Initial (Pre-Reduction)</th><th>Reduced (Post-Reduction)</th></tr>
                <tr><td>Total DataFrames/Tables (non-empty)</td><td>{initial_dict_summary.get('totals', {}).get('dataframe_count', 'N/A')}</td><td>{reduced_dict_summary.get('totals', {}).get('dataframe_count', 'N/A')}</td></tr>
                <tr><td>Memory Size</td><td>{initial_size_mb:.2f} MB</td><td>{reduced_size_mb:.2f} MB</td></tr>
    """
    if initial_size_mb > 0 and initial_size_mb > reduced_size_mb:
        reduction_pct = abs(1 - (reduced_size_mb / initial_size_mb)) * 100
        html_content += f"<tr><td>Reduction in Memory Size %</td><td colspan='2' style='text-align:center;'>{reduction_pct:.2f}%</td></tr>"
    elif initial_size_mb == reduced_size_mb and initial_size_mb > 0 : 
        html_content += "<tr><td>Reduction in Memory Size %</td><td colspan='2' style='text-align:center;'>0.00% (No change)</td></tr>"
    else:
        html_content += "<tr><td>Reduction in Memory Size %</td><td colspan='2' style='text-align:center;'>N/A</td></tr>"

    html_content += f"""
                <tr><td>Total Rows (Sum across non-empty tables)</td><td>{initial_dict_summary.get('totals', {}).get('total_rows_sum', 'N/A')}</td><td>{reduced_dict_summary.get('totals', {}).get('total_rows_sum', 'N/A')}</td></tr>
                <tr><td>Total Columns (Sum across non-empty tables)</td><td>{initial_dict_summary.get('totals', {}).get('total_columns_sum', 'N/A')}</td><td>{reduced_dict_summary.get('totals', {}).get('total_columns_sum', 'N/A')}</td></tr>
                <tr><td>Overall Null % (across all non-empty cells)</td><td>{initial_dict_summary.get('totals', {}).get('overall_null_pct', 0.0):.1f}%</td><td>{reduced_dict_summary.get('totals', {}).get('overall_null_pct', 0.0):.1f}%</td></tr>
                <tr><td>Total Numeric Columns (Sum across non-empty tables)</td><td>{initial_dict_summary.get('totals', {}).get('total_numeric_cols_sum', 'N/A')}</td><td>{reduced_dict_summary.get('totals', {}).get('total_numeric_cols_sum', 'N/A')}</td></tr>
                <tr><td>Total Object Columns (Sum across non-empty tables)</td><td>{initial_dict_summary.get('totals', {}).get('total_object_cols_sum', 'N/A')}</td><td>{reduced_dict_summary.get('totals', {}).get('total_object_cols_sum', 'N/A')}</td></tr>
    """
    total_cols_dropped = 0
    if isinstance(analysis_report, dict):
        for key, item_report in analysis_report.items():
            if isinstance(item_report, dict) and 'drop_suggestions' in item_report:
                total_cols_dropped += item_report['drop_suggestions'].get('count', 0)
    html_content += f"<tr><td>Total Columns Dropped (Reduction Step)</td><td colspan='2' style='text-align:center;'>{total_cols_dropped}</td></tr>"
    
    if final_consolidated_df_shape:
        html_content += f"<tr><td>Shape after Merging & COHORT Consolidation</td><td colspan='2' style='text-align:center;'>({final_consolidated_df_shape[0]}, {final_consolidated_df_shape[1]})</td></tr>"
    html_content += """
            </table>
        </div>
    """

    html_content += "<h2>Data Reduction Details (Per Modality/Table)</h2>"
    all_reduction_keys = sorted(list(set(initial_dict_summary.keys()) | set(reduced_dict_summary.keys()) - {'totals'}))

    for key in all_reduction_keys:
        if key == 'totals': continue 

        html_content += f"""
        <div class="modality-section">
            <h3>Reduction Analysis for: {key}</h3>
        """
        html_content += "<h4>Before Reduction</h4>"
        stats_before = initial_dict_summary.get(key)
        if stats_before and not stats_before.get("is_empty"):
            html_content += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape</td><td>({stats_before['shape'][0]}, {stats_before['shape'][1]})</td></tr>
                <tr><td>Null Percentage</td><td>{stats_before['null_pct']:.1f}%</td></tr>
                <tr><td>Numeric Columns</td><td>{stats_before['numeric_cols']}</td></tr>
                <tr><td>Object Columns</td><td>{stats_before['object_cols']}</td></tr>
            </table>
            """
        elif stats_before and stats_before.get("is_empty"):
            html_content += "<p><em>DataFrame was empty.</em></p>"
        else:
            html_content += "<p>Not present in initial data.</p>"

        html_content += "<h4>After Reduction</h4>"
        stats_after = reduced_dict_summary.get(key)
        if stats_after and not stats_after.get("is_empty"):
            html_content += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape</td><td>({stats_after['shape'][0]}, {stats_after['shape'][1]})</td></tr>
                <tr><td>Null Percentage</td><td>{stats_after['null_pct']:.1f}%</td></tr>
                <tr><td>Numeric Columns</td><td>{stats_after['numeric_cols']}</td></tr>
                <tr><td>Object Columns</td><td>{stats_after['object_cols']}</td></tr>
            </table>
            """
        elif stats_after and stats_after.get("is_empty"):
            html_content += "<p><em>DataFrame is empty after reduction.</em></p>"
        else:
            html_content += "<p>Not present or empty after reduction.</p>"


        current_analysis_report_item = analysis_report.get(key)
        if isinstance(current_analysis_report_item, dict) and 'drop_suggestions' in current_analysis_report_item:
            suggestions = current_analysis_report_item['drop_suggestions']
            dropped_cols = suggestions.get('columns', [])
            reasons = suggestions.get('reasons', {})
            count = suggestions.get('count', 0)

            if count > 0:
                html_content += f"""
                <div class="dropped-columns">
                    <strong>Columns Dropped ({count}):</strong>
                """
                is_biospecimen_related = key == 'biospecimen' or key.startswith('biospecimen.')
                if is_biospecimen_related:
                     html_content += "<p><em style='font-size:0.9em;'>(Details omitted for biospecimen-related data)</em></p>"
                else:
                    html_content += "<ul>"
                    for col in dropped_cols:
                        reason = reasons.get(col, "N/A")
                        html_content += f"<li>{col} <span class='reason'>({reason})</span></li>"
                    html_content += "</ul>"
                html_content += "</div>" 
            elif (stats_before and not stats_before.get("is_empty")) or \
                 (stats_after and not stats_after.get("is_empty")): 
                html_content += "<p><em>No columns suggested for dropping based on criteria.</em></p>"
        elif (stats_before and not stats_before.get("is_empty")) or \
             (stats_after and not stats_after.get("is_empty")): 
             html_content += "<p><em>No columns suggested for dropping based on criteria.</em></p>"
        html_content += "</div>" 

    html_content += """
    </body>
    </html>
    """

    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("HTML report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_html_path}: {e}")


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