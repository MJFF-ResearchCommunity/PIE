import os
import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np # Import numpy for size calculation

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the necessary classes
from pie.data_loader import DataLoader
from pie.data_reducer import DataReducer

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
    total_rows = 0 # Note: this sums rows across *potentially unrelated* DFs
    total_cols = 0
    total_numeric_cols = 0
    total_object_cols = 0
    total_nulls = 0
    total_cells = 0

    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            rows, cols = value.shape
            total_rows += rows
            total_cols += cols
            null_sum = value.isnull().sum().sum()
            df_cells = rows * cols
            total_nulls += null_sum
            total_cells += df_cells
            null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
            numeric_cols = value.select_dtypes(include=np.number).shape[1]
            object_cols = value.select_dtypes(include='object').shape[1]
            total_numeric_cols += numeric_cols
            total_object_cols += object_cols
            logger.info(f"{key}: Shape=({rows}, {cols}), Nulls={null_pct:.1f}%, NumericCols={numeric_cols}, ObjectCols={object_cols}")
        elif isinstance(value, dict):
            logger.info(f"{key}: (Nested Dictionary)")
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, pd.DataFrame):
                     rows, cols = sub_value.shape
                     total_rows += rows
                     total_cols += cols
                     null_sum = sub_value.isnull().sum().sum()
                     df_cells = rows * cols
                     total_nulls += null_sum
                     total_cells += df_cells
                     null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
                     numeric_cols = sub_value.select_dtypes(include=np.number).shape[1]
                     object_cols = sub_value.select_dtypes(include='object').shape[1]
                     total_numeric_cols += numeric_cols
                     total_object_cols += object_cols
                     logger.info(f"  {sub_key}: Shape=({rows}, {cols}), Nulls={null_pct:.1f}%, NumericCols={numeric_cols}, ObjectCols={object_cols}")

    overall_null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
    logger.info(f"Totals: Columns={total_cols}, Numeric={total_numeric_cols}, Object={total_object_cols}, OverallNull%={overall_null_pct:.1f}%")
    logger.info("-" * (len(title) + 6))

def get_dict_summary(data_dict: dict) -> dict:
    """Calculates shape and basic stats for each DataFrame, returns as dict."""
    summary = {}
    total_rows = 0
    total_cols = 0
    total_numeric_cols = 0
    total_object_cols = 0
    total_nulls = 0
    total_cells = 0

    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            rows, cols = value.shape
            total_rows += rows
            total_cols += cols
            null_sum = value.isnull().sum().sum()
            df_cells = rows * cols
            total_nulls += null_sum
            total_cells += df_cells
            null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
            numeric_cols = value.select_dtypes(include=np.number).shape[1]
            object_cols = value.select_dtypes(include='object').shape[1]
            total_numeric_cols += numeric_cols
            total_object_cols += object_cols
            summary[key] = {
                "shape": (rows, cols),
                "null_pct": null_pct,
                "numeric_cols": numeric_cols,
                "object_cols": object_cols
            }
        elif isinstance(value, dict):
            # Handle nested dicts - create a nested summary structure
            nested_summary = {}
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, pd.DataFrame):
                     rows, cols = sub_value.shape
                     total_rows += rows # Still contribute to overall totals
                     total_cols += cols
                     null_sum = sub_value.isnull().sum().sum()
                     df_cells = rows * cols
                     total_nulls += null_sum
                     total_cells += df_cells
                     null_pct = (null_sum / df_cells) * 100 if df_cells > 0 else 0
                     numeric_cols = sub_value.select_dtypes(include=np.number).shape[1]
                     object_cols = sub_value.select_dtypes(include='object').shape[1]
                     total_numeric_cols += numeric_cols
                     total_object_cols += object_cols
                     nested_summary[f"{key}.{sub_key}"] = { # Use combined key for easy lookup later
                        "shape": (rows, cols),
                        "null_pct": null_pct,
                        "numeric_cols": numeric_cols,
                        "object_cols": object_cols
                     }
            summary.update(nested_summary) # Add nested summaries to the main summary dict

    overall_null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
    summary['totals'] = {
        "columns": total_cols,
        "numeric": total_numeric_cols,
        "object": total_object_cols,
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
    # verbose_drops: bool = False <-- Removed
):
    """Generates an HTML report summarizing the data reduction process."""
    logger.info(f"Generating HTML report at: {output_html_path}")

    # --- CSS Styling ---
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

    # --- HTML Content ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Data Reduction Report</title>
        {html_style}
    </head>
    <body>
        <h1>PIE Data Reduction Analysis Report</h1>

        <div class="summary-box">
            <h2>Overall Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Size</td><td>{initial_size_mb:.2f} MB</td></tr>
                <tr><td>Reduced Size</td><td>{reduced_size_mb:.2f} MB</td></tr>
    """
    if initial_size_mb > 0 and initial_size_mb > reduced_size_mb:
        reduction_pct = abs(1 - (reduced_size_mb / initial_size_mb)) * 100
        html_content += f"<tr><td>Size Reduction</td><td>{reduction_pct:.2f}%</td></tr>"
    elif initial_size_mb == reduced_size_mb:
        html_content += "<tr><td>Size Reduction</td><td>0.00% (No change)</td></tr>"
    else:
        html_content += "<tr><td>Size Reduction</td><td>N/A</td></tr>"

    html_content += """
            </table>
        </div>
    """

    # --- Per Modality/Table Sections ---
    all_keys = sorted(list(set(initial_dict_summary.keys()) | set(reduced_dict_summary.keys()) - {'totals'}))

    for key in all_keys:
        if key == 'totals': continue # Skip the totals key here

        html_content += f"""
        <div class="modality-section">
            <h3>Analysis for: {key}</h3>
        """

        # --- Before Reduction Table ---
        html_content += "<h4>Before Reduction</h4>"
        if key in initial_dict_summary:
            stats_before = initial_dict_summary[key]
            html_content += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape</td><td>({stats_before['shape'][0]}, {stats_before['shape'][1]})</td></tr>
                <tr><td>Null Percentage</td><td>{stats_before['null_pct']:.1f}%</td></tr>
                <tr><td>Numeric Columns</td><td>{stats_before['numeric_cols']}</td></tr>
                <tr><td>Object Columns</td><td>{stats_before['object_cols']}</td></tr>
            </table>
            """
        else:
            html_content += "<p>Not present in initial data.</p>"

        # --- After Reduction Table ---
        html_content += "<h4>After Reduction</h4>"
        if key in reduced_dict_summary:
            stats_after = reduced_dict_summary[key]
            html_content += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape</td><td>({stats_after['shape'][0]}, {stats_after['shape'][1]})</td></tr>
                <tr><td>Null Percentage</td><td>{stats_after['null_pct']:.1f}%</td></tr>
                <tr><td>Numeric Columns</td><td>{stats_after['numeric_cols']}</td></tr>
                <tr><td>Object Columns</td><td>{stats_after['object_cols']}</td></tr>
            </table>
            """
        else:
            html_content += "<p>Not present or empty after reduction.</p>"

        # --- Dropped Columns Details ---
        if key in analysis_report and 'drop_suggestions' in analysis_report[key]:
            suggestions = analysis_report[key]['drop_suggestions']
            dropped_cols = suggestions.get('columns', [])
            reasons = suggestions.get('reasons', {})
            count = suggestions.get('count', 0)

            if count > 0:
                html_content += f"""
                <div class="dropped-columns">
                    <strong>Columns Dropped ({count}):</strong>
                """
                # Check if it's the biospecimen key
                if key == 'biospecimen':
                     html_content += "<p><em style='font-size:0.9em;'>(Details omitted for biospecimen)</em></p>"
                else:
                    # Show details for all other keys
                    html_content += "<ul>"
                    for col in dropped_cols:
                        reason = reasons.get(col, "N/A")
                        html_content += f"<li>{col} <span class='reason'>({reason})</span></li>"
                    html_content += "</ul>"
                html_content += "</div>" # end dropped-columns
            else:
                html_content += "<p><em>No columns suggested for dropping based on criteria.</em></p>"


        html_content += "</div>" # end modality-section

    html_content += """
    </body>
    </html>
    """

    # --- Write to File ---
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("HTML report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_html_path}: {e}")

def test_data_reducer_workflow(output_html_path: str = "data_reduction_report.html",
                               final_merged_csv_path: str = "output/final_merged_reduced_data.csv"):
    """
    Tests the DataReducer workflow: load, analyze, reduce, compare size,
    and generate HTML report. Shows dropped column details except for biospecimen.

    Args:
        output_html_path: Path to save the generated HTML report.
        final_merged_csv_path: Path to save the final merged CSV file.
    """
    # Logging is configured globally now, logger is defined globally

    # Define data path (Adjust if necessary)
    data_dir = "./PPMI"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Please ensure PPMI data is present.")
        return

    # 1. Load Data Dictionary
    logger.info(f"Loading all modalities from: {data_dir}")
    try:
        data_dict = DataLoader.load(data_path=data_dir, merge_output=False)
        logger.info(f"Successfully loaded {len(data_dict)} modalities.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True) # Add traceback
        return

    if not data_dict:
        logger.error("Loaded data dictionary is empty. Cannot proceed.")
        return

    # 2. Calculate Initial Size & Get Initial Summary
    initial_size_mb = calculate_dict_size(data_dict)
    initial_summary = get_dict_summary(data_dict)
    logger.info(f"\n--- Initial Data Summary ---")
    for key, stats in initial_summary.items():
        if key != 'totals':
            logger.info(f"{key}: Shape={stats['shape']}, Nulls={stats['null_pct']:.1f}%, "
                        f"NumericCols={stats['numeric_cols']}, ObjectCols={stats['object_cols']}")
    logger.info(f"Totals: Columns={initial_summary['totals']['columns']}, "
                f"Numeric={initial_summary['totals']['numeric']}, "
                f"Object={initial_summary['totals']['object']}, "
                f"OverallNull%={initial_summary['totals']['overall_null_pct']:.1f}%")
    logger.info("-" * 30)


    # 3. Instantiate DataReducer
    analyzer = DataReducer(data_dict) # Use default config
    logger.info("DataReducer initialized.")

    # 4. Get the detailed analysis and drop suggestions
    logger.info("\nRunning analysis...")
    analysis_results = analyzer.analyze()
    logger.info("--- Analysis Report Summary (Console) ---")
    print(analyzer.generate_report_str(analysis_results)) # Keep console summary

    drop_suggestions = analyzer.get_drop_suggestions(analysis_results)

    # Print drop suggestions to console (excluding biospecimen details)
    logger.info("\n--- Drop Suggestions (Console Summary) ---")
    total_suggested_drops = 0
    for key, cols_to_drop in drop_suggestions.items():
        count = len(cols_to_drop)
        total_suggested_drops += count
        if count > 0:
            if key == 'biospecimen':
                logger.info(f"{key}: {count} columns suggested (details omitted).")
            else:
                logger.info(f"{key}: {count} columns suggested:")
                # Try pretty printing the list for non-biospecimen
                try:
                    print(json.dumps({key: cols_to_drop}, indent=2))
                except TypeError:
                    print(f"  {cols_to_drop}") # Fallback plain print
    logger.info(f"Total columns suggested for dropping across all modalities: {total_suggested_drops}")


    # 5. Apply the drops
    logger.info("\nApplying drop suggestions...")
    reduced_dict = analyzer.apply_drops(drop_suggestions)
    logger.info("Finished applying drops.")

    # 6. Calculate Reduced Size & Get Final Summary
    reduced_size_mb = calculate_dict_size(reduced_dict)
    reduced_summary = get_dict_summary(reduced_dict)
    logger.info(f"\n--- Reduced Data Summary ---")
    for key, stats in reduced_summary.items():
         if key != 'totals':
            logger.info(f"{key}: Shape={stats['shape']}, Nulls={stats['null_pct']:.1f}%, "
                        f"NumericCols={stats['numeric_cols']}, ObjectCols={stats['object_cols']}")
    logger.info(f"Totals: Columns={reduced_summary['totals']['columns']}, "
                f"Numeric={reduced_summary['totals']['numeric']}, "
                f"Object={reduced_summary['totals']['object']}, "
                f"OverallNull%={reduced_summary['totals']['overall_null_pct']:.1f}%")
    logger.info("-" * 30)


    # 7. Print Size Comparison (Console)
    logger.info("\n--- Size Comparison (Console) ---")
    logger.info(f"Initial Size: {initial_size_mb:.2f} MB")
    logger.info(f"Reduced Size: {reduced_size_mb:.2f} MB")
    if initial_size_mb > 0 and initial_size_mb > reduced_size_mb:
        reduction_pct = abs(1 - (reduced_size_mb / initial_size_mb)) * 100
        logger.info(f"Size Reduction: {reduction_pct:.2f}%")
    elif initial_size_mb == reduced_size_mb:
         logger.info("Size Reduction: 0.00% (No change)")
    else:
        logger.info("Size Reduction: N/A")

    # 8. Merge the Reduced Data
    logger.info("\nMerging the reduced DataFrames...")
    if reduced_dict: # Ensure there's something to merge
        final_merged_df = analyzer.merge_reduced_data(reduced_dict, output_filename=final_merged_csv_path)
        if not final_merged_df.empty:
            logger.info(f"Final merged DataFrame created with shape: {final_merged_df.shape}")
            
            # Add assertion for uniqueness
            are_patno_event_id_unique = not final_merged_df.duplicated(subset=['PATNO', 'EVENT_ID']).any()
            if not are_patno_event_id_unique:
                num_duplicates = final_merged_df.duplicated(subset=['PATNO', 'EVENT_ID']).sum()
                logger.error(f"Assertion Failed: Final merged DataFrame contains {num_duplicates} duplicate (PATNO, EVENT_ID) pairs!")
                # Optional: print some of the duplicates for debugging
                # duplicated_rows = final_merged_df[final_merged_df.duplicated(subset=['PATNO', 'EVENT_ID'], keep=False)]
                # logger.error(f"Example duplicates:\n{duplicated_rows.head()}")
            assert are_patno_event_id_unique, "Final merged DataFrame contains duplicate (PATNO, EVENT_ID) pairs!"
            logger.info("Verified: (PATNO, EVENT_ID) pairs are unique in the final merged DataFrame.")

            if os.path.exists(final_merged_csv_path):
                logger.info(f"Final merged data successfully saved to: {final_merged_csv_path}")
            else:
                logger.error(f"Failed to save final merged data to: {final_merged_csv_path}")
        else:
            logger.warning("Merging resulted in an empty DataFrame.")
    else:
        logger.warning("Reduced dictionary is empty, skipping final merge.")
        final_merged_df = pd.DataFrame() # Define for HTML report generation even if empty


    # 9. Generate HTML Report
    if output_html_path:
         # Ensure analysis_report keys match summary keys (handle nested dict format)
         analysis_report_adjusted = {}
         for key, report in analysis_results.items():
              # Convert 'modality.table' keys if needed, or just use them as is
              # if get_dict_summary already uses combined keys
              analysis_report_adjusted[key] = report

         generate_html_report(
            initial_dict_summary=initial_summary,
            reduced_dict_summary=reduced_summary,
            analysis_report=analysis_report_adjusted, # Use the potentially adjusted keys
            initial_size_mb=initial_size_mb,
            reduced_size_mb=reduced_size_mb,
            output_html_path=output_html_path,
            # verbose_drops is removed here, logic is inside generate_html_report
        )


if __name__ == "__main__":
    # Define the output file name
    report_file = "output/data_reduction_report.html" # Ensure output dir exists or is created
    final_csv = "output/final_full_reduced_dataset.csv"
    
    # Create output directory if it doesn't exist
    output_dir = Path(report_file).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")
            # Optionally, exit or handle if directory creation is critical

    # Run the workflow
    test_data_reducer_workflow(output_html_path=report_file, final_merged_csv_path=final_csv)


    # Optional: Automatically open the report (platform dependent)
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.realpath(report_file)}")
    except Exception as e:
        logger.info(f"Could not automatically open the report: {e}")