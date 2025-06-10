"""
Reporting utilities for the PIE pipeline.
Contains functions to generate various HTML reports.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict

logger = logging.getLogger("PIE.reporting")

def generate_data_reduction_html_report(
    initial_dict_summary: dict,
    reduced_dict_summary: dict,
    analysis_report: dict,
    initial_size_mb: float,
    reduced_size_mb: float,
    output_html_path: str,
    final_consolidated_df_shape: Optional[tuple] = None
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

def generate_feature_engineering_report_html(report_data: dict, output_html_path: str):
    """Generates an HTML report summarizing the feature engineering process."""
    logger.info(f"Generating Feature Engineering HTML report at: {output_html_path}")

    html_style = """
    <style>
        body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 10px; }
        h1 { text-align: center; color: #27ae60; }
        table { width: auto; border-collapse: collapse; margin-bottom: 20px; margin-left: auto; margin-right: auto; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #27ae60; color: white; }
        tr:nth-child(even) { background-color: #ecf0f1; }
        .summary-box { border: 1px solid #bdc3c7; padding: 15px; margin-bottom: 20px; background-color: #f8f9f9; border-radius: 5px; }
        .code { background-color: #e8e8e8; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', Courier, monospace; }
        .highlight { color: #e67e22; font-weight: bold; }
        ul { list-style-type: square; padding-left: 20px; }
        li { margin-bottom: 5px; }
        .operation-details { margin-left: 20px; }
    </style>
    """

    fe_summary = report_data.get('feature_engineering_summary', {})
    operations = fe_summary.get('engineered_operations', {})

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Feature Engineering Report</title>
        {html_style}
    </head>
    <body>
        <div class="container">
            <h1>PIE Feature Engineering Report</h1>

            <div class="summary-box">
                <h2>1. Input Data</h2>
                <p><strong>Input Dataset:</strong> <span class="code">{report_data.get('input_csv_path', 'N/A')}</span></p>
                <p><strong>Shape of Input Data:</strong> <span class="highlight">{report_data.get('input_data_shape', 'N/A')}</span></p>
            </div>

            <div class="summary-box">
                <h2>2. Feature Engineering Summary</h2>
                <p><strong>Original Columns in DataFrame (before FE):</strong> {fe_summary.get('total_original_columns', 'N/A')}</p>
                <p><strong>Total Columns After Feature Engineering:</strong> <span class="highlight">{fe_summary.get('total_current_columns', 'N/A')}</span></p>
                <p><strong>Total New Features Created/Modified:</strong> <span class="highlight">{fe_summary.get('newly_engineered_features_count', 'N/A')}</span></p>
                
                <h3>Operations Performed:</h3>
    """
    if operations:
        html_content += "<ul>"
        for op_type, op_details in operations.items():
            op_name_pretty = op_type.replace('_', ' ').title()
            html_content += f"<li><strong>{op_name_pretty}:</strong> {op_details.get('count', 0)} features created/modified."
            if op_details.get('features'):
                examples = op_details['features']
                if len(examples) > 10: # Truncate long lists
                    example_str = ", ".join(map(str, examples[:10])) + ", ..."
                else:
                    example_str = ", ".join(map(str, examples))
                html_content += f"<div class='operation-details'>Examples: <span class='code'>{example_str}</span></div>"
            html_content += "</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No specific feature engineering operations were tracked or performed.</p>"
    html_content += """
            </div>

            <div class="summary-box">
                <h2>3. Output Data</h2>
                <p><strong>Shape of Data After Feature Engineering:</strong> <span class="highlight">{report_data.get('output_data_shape', 'N/A')}</span></p>
                <p><strong>Engineered Dataset Saved To:</strong> <span class="code">{report_data.get('output_csv_path', 'N/A')}</span></p>
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