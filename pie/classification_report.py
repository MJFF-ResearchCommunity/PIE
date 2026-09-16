import os
import sys
import logging
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Union, Optional, Any, Dict, Tuple

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.classifier import Classifier, check_classification_target, split_train_test
from pie.feature_selector import FeatureSelector
from sklearn.preprocessing import LabelEncoder

# Endgame visualization / explain imports
try:
    from endgame.visualization import ClassificationReport as EndgameClassificationReport
    from endgame.explain import explain as eg_explain
    ENDGAME_VIS_AVAILABLE = True
except ImportError:
    ENDGAME_VIS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.classification_report")

def generate_classification_report_html(
    report_data: dict,
    output_html_path: str,
    plots_dir: str
):
    """Generates a comprehensive HTML report for the classification pipeline."""
    logger.info(f"Generating Classification HTML report at: {output_html_path}")

    html_style = """
    <style>
        body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); max-width: 1400px; margin: 0 auto; }
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }
        h1 { text-align: center; color: #e74c3c; font-size: 2.5em; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #e74c3c; color: white; font-weight: bold; }
        tr:nth-child(even) { background-color: #ecf0f1; }
        tr:hover { background-color: #d5dbdb; }
        .summary-box { border: 2px solid #bdc3c7; padding: 20px; margin-bottom: 25px; background-color: #f8f9f9; border-radius: 8px; }
        .code { background-color: #2c3e50; color: #ecf0f1; padding: 3px 8px; border-radius: 4px; font-family: 'Courier New', Courier, monospace; }
        .highlight { color: #e74c3c; font-weight: bold; font-size: 1.1em; }
        .metric-value { font-weight: bold; color: #27ae60; font-size: 1.1em; }
        .warning { color: #f39c12; font-weight: bold; }
        .plot-container { margin: 25px 0; text-align: center; background-color: #ecf0f1; padding: 20px; border-radius: 8px; }
        .plot-container img { max-width: 90%; height: auto; border: 2px solid #34495e; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .plot-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 1.2em; }
        .feature-importance { background-color: #e8f8f5; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .model-card { background-color: #fef9e7; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #f9e79f; }
        .best-model { background-color: #d5f4e6; padding: 20px; border-radius: 8px; margin: 20px 0; border: 2px solid #27ae60; }
        .hyperparameter-tuning { background-color: #fadbd8; padding: 15px; border-radius: 8px; margin: 15px 0; }
        ul { list-style-type: square; padding-left: 30px; }
        li { margin-bottom: 8px; }
        .timestamp { color: #7f8c8d; font-style: italic; text-align: right; margin-top: 20px; }
        .leaderboard-table { font-size: 0.95em; }
        .leaderboard-table td, .leaderboard-table th { padding: 8px; }
        .top-model { background-color: #d5f4e6; font-weight: bold; }
    </style>
    """

    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Classification Pipeline Report</title>
        {html_style}
    </head>
    <body>
        <div class="container">
            <h1>PIE Classification Pipeline Report</h1>

            <div class="summary-box">
                <h2>1. Pipeline Overview</h2>
                <table>
                    <tr><th>Component</th><th>Details</th></tr>
                    <tr><td>Input Data</td><td><span class="code">{report_data.get('input_data_path', 'N/A')}</span></td></tr>
                    <tr><td>Data Shape</td><td>{report_data.get('input_data_shape', 'N/A')}</td></tr>
                    <tr><td>Target Variable</td><td><span class="highlight">{report_data.get('target_column', 'N/A')}</span></td></tr>
                    <tr><td>Target Classes</td><td>{report_data.get('n_classes', 'N/A')} classes</td></tr>
                    <tr><td>Train/Test Split</td><td>{report_data.get('train_size', 0.8) * 100:.0f}% / {(1 - report_data.get('train_size', 0.8)) * 100:.0f}%</td></tr>
                    <tr><td>Excluded Features</td><td>{report_data.get('excluded_features_count', 0)} features excluded</td></tr>
                    <tr><td>Feature Selection Applied</td><td>{'Yes' if report_data.get('feature_selection_applied', False) else 'No'}</td></tr>
                    <tr><td>Final Feature Count</td><td><span class="metric-value">{report_data.get('n_features', 'N/A')}</span></td></tr>
                    <tr><td>ML Engine</td><td><span class="code">endgame</span></td></tr>
                </table>
            </div>
    """

    # Add excluded features details if any were excluded
    if report_data.get('excluded_features'):
        html_content += f"""
                <h3>Excluded Features (Data Leakage Prevention)</h3>
                <p>The following features were excluded to prevent data leakage:</p>
                <ul>
        """
        for feature in report_data.get('excluded_features', []):
            html_content += f"<li><span class='code'>{feature}</span></li>"
        html_content += """
                </ul>
                <p><em>These features were excluded because they may be too closely related to the target variable
                or contain information that would not be available at prediction time.</em></p>
        """

    html_content += """
            </div>
    """

    # Add feature selection summary if applied
    if report_data.get('feature_selection_applied', False):
        html_content += f"""
            <div class="summary-box">
                <h2>2. Feature Selection Summary</h2>
                <table>
                    <tr><th>Method</th><th>Details</th></tr>
                    <tr><td>Selection Method</td><td>{report_data.get('feature_selection_method', 'N/A')}</td></tr>
                    <tr><td>Original Features</td><td>{report_data.get('original_features', 'N/A')}</td></tr>
                    <tr><td>Selected Features</td><td><span class="metric-value">{report_data.get('selected_features', 'N/A')}</span></td></tr>
                    <tr><td>Reduction</td><td>{report_data.get('feature_reduction_pct', 'N/A')}%</td></tr>
                </table>
            </div>
        """

    # Model comparison leaderboard
    if report_data.get('leaderboard') is not None:
        html_content += """
            <div class="summary-box">
                <h2>3. Model Comparison Leaderboard</h2>
                <p>Cross-validation results for all evaluated models:</p>
                <div style="overflow-x: auto;">
        """
        # Convert leaderboard to HTML with custom styling for top model
        leaderboard_html = report_data['leaderboard'].to_html(
            classes='leaderboard-table',
            index=True,
            float_format=lambda x: f'{x:.4f}'
        )
        # Highlight the top row
        leaderboard_html = leaderboard_html.replace('<tr>', '<tr class="top-model">', 1)
        html_content += leaderboard_html
        html_content += """
                </div>
                <p><em>Note: The top model (highlighted in green) was selected based on the optimization metric.</em></p>
            </div>
        """

    # Best model details
    if report_data.get('best_model_name'):
        html_content += f"""
            <div class="best-model">
                <h2>4. Best Model Details</h2>
                <h3>Selected Model: <span class="highlight">{report_data.get('best_model_name', 'N/A')}</span></h3>
                <p><em>Cross-validated on the training split, before tuning. Held-out performance is in section 9.</em></p>
        """

        # Add metrics table if available
        best_metrics = report_data.get('best_model_metrics', {})
        if best_metrics:
            html_content += """
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
            """
            for metric, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    html_content += f"<tr><td>{metric}</td><td class='metric-value'>{value:.4f}</td></tr>"
            html_content += """
                </table>
            """
        else:
            html_content += "<p><em>Metrics data not available</em></p>"

        html_content += """
            </div>
        """

    # Hyperparameter tuning results
    if report_data.get('tuning_results'):
        html_content += f"""
            <div class="hyperparameter-tuning">
                <h2>5. Hyperparameter Tuning</h2>
                <p><strong>Optimization Metric:</strong> {report_data.get('tuning_metric', 'N/A')}</p>
                <p><strong>Number of Iterations:</strong> {report_data.get('tuning_iterations', 'N/A')}</p>
                <p><strong>Performance Improvement:</strong>
                    <span class="metric-value">{report_data.get('tuning_improvement', 'N/A')}</span>
                </p>
                <h3>Best Hyperparameters:</h3>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
        """
        for param, value in report_data.get('best_hyperparameters', {}).items():
            html_content += f"<tr><td>{param}</td><td><span class='code'>{value}</span></td></tr>"
        html_content += """
                </table>
            </div>
        """

    # Model plots
    html_content += """
            <div class="summary-box">
                <h2>6. Model Performance Visualizations</h2>
    """

    # Check for all possible plots
    all_possible_plots = [
        ('confusion_matrix', 'Confusion Matrix'),
        ('auc', 'ROC Curve'),
        ('pr', 'Precision-Recall Curve'),
        ('feature', 'Feature Importance'),
        ('learning', 'Learning Curve'),
        ('multiclass_roc', 'Multiclass ROC Curves'),
        ('class_distribution', 'Class Distribution'),
        ('feature_correlation', 'Feature Correlation Heatmap'),
        ('high_correlation_pairs', 'High Correlation Feature Pairs'),
        ('pca_visualization', 'PCA Visualization (2D)'),
        ('pca_3d_visualization', '3D PCA Visualization'),
        ('tsne_visualization_perp30', 't-SNE Visualization (Perplexity 30)'),
        ('tsne_visualization_perp50', 't-SNE Visualization (Perplexity 50)'),
        ('pca_vs_tsne_comparison', 'PCA vs t-SNE Comparison'),
        ('umap_visualization', 'UMAP Visualization'),
        ('prediction_confidence', 'Prediction Confidence Distribution')
    ]

    # Also check for endgame report
    endgame_report_path = Path(plots_dir) / "endgame_report.html"
    if endgame_report_path.exists():
        html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">ENDGAME COMPREHENSIVE REPORT</div>
                    <p><a href="plots/endgame_report.html" target="_blank">Open full endgame classification report (42 charts)</a></p>
                </div>
        """

    plots_found = False
    for plot_file, plot_title in all_possible_plots:
        plot_path = Path(plots_dir) / f"{plot_file}.png"
        if plot_path.exists():
            plots_found = True
            relative_plot_path = f"plots/{plot_file}.png"
            html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_title.upper()}</div>
                    <img src="{relative_plot_path}" alt="{plot_title}">
                </div>
            """

    if not plots_found and not endgame_report_path.exists():
        html_content += "<p><em>No visualization plots were generated successfully.</em></p>"

    html_content += """
            </div>
    """

    # Feature importance details
    if report_data.get('feature_importance'):
        html_content += """
            <div class="feature-importance">
                <h2>7. Top Discriminative Features</h2>
                <p>The following features are most important for distinguishing between classes:</p>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Importance Score</th></tr>
        """
        for i, (feature, importance) in enumerate(report_data['feature_importance'][:20], 1):
            html_content += f"<tr><td>{i}</td><td><span class='code'>{feature}</span></td><td class='metric-value'>{importance:.4f}</td></tr>"
        html_content += """
                </table>
            </div>
        """

    # Model interpretation
    html_content += """
            <div class="summary-box">
                <h2>8. Model Interpretation</h2>
    """

    # Check for different types of interpretation plots
    interpretation_plots = [
        ('shap_summary.png', 'SHAP Summary', 'SHAP (SHapley Additive exPlanations) shows feature contributions to predictions'),
        ('feature_importance_plot.png', 'Permutation Feature Importance', 'Shows the decrease in model performance when each feature is randomly shuffled'),
        ('interpretation_plot.png', 'Feature Importance', 'Shows the relative importance of features in the model'),
        ('feature.png', 'Feature Importance', 'Basic feature importance from the model')
    ]

    interpretation_found = False
    for plot_file, plot_title, plot_description in interpretation_plots:
        plot_path = Path(plots_dir) / plot_file
        if plot_path.exists():
            html_content += f"""
                <p>{plot_description}:</p>
                <div class="plot-container">
                    <div class="plot-title">{plot_title.upper()}</div>
                    <img src="plots/{plot_file}" alt="{plot_title}">
                </div>
            """
            interpretation_found = True
            break

    if not interpretation_found:
        html_content += f"""
            <p><em>Model interpretation plots could not be generated. This can happen when:</em></p>
            <ul>
                <li>The selected model type ({report_data.get('best_model_name', 'Unknown')}) doesn't support SHAP interpretation</li>
                <li>The model is not tree-based (SHAP summary plots only work with tree-based models)</li>
                <li>Technical limitations with the interpretation libraries</li>
            </ul>
            <p><em>Consider using tree-based models (Random Forest, XGBoost, etc.) for better interpretability.</em></p>
        """

    html_content += """
            </div>
    """

    # Test set performance
    html_content += """
            <div class="summary-box">
                <h2>9. Final Test Set Performance</h2>
    """

    test_metrics = report_data.get('test_metrics', {})
    if test_metrics:
        html_content += """
                <p>The final model (tuned, if tuning ran) scored once on the held-out test split:</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric}</td><td class='metric-value'>{value:.4f}</td></tr>"
        html_content += """
                </table>
        """
    else:
        html_content += "<p><em>Test set performance metrics are not available. This may be due to the model evaluation approach used.</em></p>"

    html_content += """
            </div>
    """

    # Recommendations
    html_content += """
            <div class="summary-box">
                <h2>10. Recommendations</h2>
                <ul>
    """

    # Add dynamic recommendations based on results
    recommendations = []

    if report_data.get('best_model_metrics', {}).get('Accuracy', 0) < 0.7:
        recommendations.append("Consider collecting more data or engineering additional features to improve model performance.")

    if report_data.get('feature_reduction_pct', 0) > 80:
        recommendations.append("Significant feature reduction was applied. Consider reviewing if important features were dropped.")

    if report_data.get('n_classes', 2) > 5:
        recommendations.append("For multi-class problems, consider using ensemble methods or one-vs-rest strategies.")

    if not recommendations:
        recommendations.append("Model performance appears satisfactory. Consider deploying with appropriate monitoring.")
        recommendations.append("Regularly retrain the model with new data to maintain performance.")

    for rec in recommendations:
        html_content += f"<li>{rec}</li>"

    html_content += f"""
                </ul>
            </div>

            <div class="timestamp">
                Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    try:
        # Ensure output directory exists
        Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_html_path}: {e}")


def generate_report(
    input_csv_path: str = None,
    train_csv_path: str = None,
    test_csv_path: str = None,
    use_feature_selection: bool = True,
    feature_selection_method: str = 'k_best',
    target_column: str = "COHORT",
    exclude_features: List[str] = None,
    output_dir: str = "output",
    n_models_to_compare: int = 5,
    tune_best_model: bool = True,
    generate_plots: bool = True,
    budget_time_minutes: float = 30.0
):
    """
    Runs the complete classification pipeline including optional feature selection,
    model comparison, hyperparameter tuning, and comprehensive reporting.

    A ``PATNO`` column, if present, groups the train/test split and the CV folds and is
    never a feature. Feature selection is fitted on the training split only. The target
    must hold class labels (text or integer codes); a continuous target raises ValueError.

    Returns ``(classifier, best_model, report_data)``. There is no failure path that
    returns None: bad input raises ``ValueError`` (no input given, unreadable file,
    missing target column, no rows left, continuous target, feature selection keeping
    nothing) and a failure inside the engine raises ``RuntimeError`` (experiment setup,
    model comparison), both naming the cause.
    """
    logger.info("Starting classification report generation...")
    exclude_features = exclude_features or []

    # Validate input parameters
    if input_csv_path is None and (train_csv_path is None or test_csv_path is None):
        # Try to auto-detect data files
        if Path("output/selected_train_data.csv").exists() and Path("output/selected_test_data.csv").exists():
            train_csv_path = "output/selected_train_data.csv"
            test_csv_path = "output/selected_test_data.csv"
            logger.info("Using pre-split feature-selected data from previous pipeline step")
            use_feature_selection = False  # Already selected
        elif Path("output/final_engineered_dataset.csv").exists():
            input_csv_path = "output/final_engineered_dataset.csv"
            logger.info("Using feature-engineered data from previous pipeline step")
        else:
            raise ValueError(
                "No input data given and none found to auto-detect. Pass input_csv_path, or "
                "both train_csv_path and test_csv_path, or run from a directory holding "
                "output/selected_train_data.csv and output/selected_test_data.csv."
            )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        'target_column': target_column,
        'feature_selection_applied': use_feature_selection,
        'plots_dir': str(plots_dir)
    }

    # Load data: a pre-split pair, or one frame that is split below
    try:
        if train_csv_path and test_csv_path:
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            report_data['input_data_path'] = f"Train: {train_csv_path}, Test: {test_csv_path}"
            report_data['input_data_shape'] = f"Train: {train_df.shape}, Test: {test_df.shape}"
        else:
            train_df = pd.read_csv(input_csv_path)
            test_df = None
            report_data['input_data_path'] = input_csv_path
            report_data['input_data_shape'] = train_df.shape
    except Exception as e:
        paths = f"{train_csv_path} and {test_csv_path}" if train_csv_path and test_csv_path else input_csv_path
        raise ValueError(f"Could not read the input data ({paths}): {e}") from e
    report_data['pre_split_data'] = test_df is not None
    frames = lambda: [f for f in (train_df, test_df) if f is not None]

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' is not in the input data, which has "
            f"{len(train_df.columns)} columns, e.g. {list(train_df.columns[:5])}. "
            "Pass target_column=<the label column>."
        )

    # Exclude leakage features. The target and PATNO are never features, so they cannot
    # be excluded; PATNO is kept to group the split and the CV folds.
    existing_excluded = [f for f in exclude_features if f in train_df.columns and f not in (target_column, "PATNO")]
    missing_excluded = [f for f in exclude_features if f not in train_df.columns]
    if existing_excluded:
        logger.info(f"Excluding features: {existing_excluded}")
        train_df = train_df.drop(columns=existing_excluded)
        test_df = test_df.drop(columns=existing_excluded) if test_df is not None else None
    if missing_excluded:
        logger.warning(f"Specified features not found in data (will be ignored): {missing_excluded}")
    report_data['excluded_features'] = existing_excluded
    report_data['excluded_features_count'] = len(existing_excluded)

    # Handle missing target values
    n_before = sum(len(f) for f in frames())
    train_df = train_df.dropna(subset=[target_column])
    test_df = test_df.dropna(subset=[target_column]) if test_df is not None else None
    if sum(len(f) for f in frames()) < n_before:
        logger.info(f"Dropped {n_before - sum(len(f) for f in frames())} rows with missing target values")
    if any(f.empty for f in frames()):
        raise ValueError(
            f"No rows left after dropping rows with a missing '{target_column}'. "
            "Check that the target column is populated in the input data."
        )
    check_classification_target(pd.concat([f[target_column] for f in frames()]))

    group_col = "PATNO" if "PATNO" in train_df.columns else None
    id_cols = [c for c in ("PATNO", "EVENT_ID") if c in train_df.columns]
    if test_df is None:
        # Split here rather than inside setup_experiment so that feature selection is
        # fitted on the training rows only. Participants never straddle the split.
        tr, te = split_train_test(train_df[target_column], train_df[group_col] if group_col else None,
                                  test_size=0.2, random_state=123)
        train_df, test_df = train_df.iloc[tr], train_df.iloc[te]

    df = pd.concat([train_df, test_df], ignore_index=True)  # all rows, for labels and counts
    report_data['n_classes'] = df[target_column].nunique()
    report_data['class_distribution'] = df[target_column].value_counts().to_dict()
    report_data['train_size'] = len(train_df) / len(df)

    feature_cols = [c for c in train_df.columns if c != target_column and c not in id_cols]
    report_data['original_features'] = len(feature_cols)
    if use_feature_selection:
        logger.info(f"Applying feature selection ({feature_selection_method}) on the training split...")
        selector = FeatureSelector(method=feature_selection_method, task_type='classification', random_state=123)
        selector.fit(train_df[feature_cols], LabelEncoder().fit_transform(train_df[target_column]))
        keep = selector.selected_feature_names_
        if not keep:
            raise ValueError(f"Feature selection '{feature_selection_method}' kept no features.")
        dropped = [c for c in feature_cols if c not in keep]
        train_df, test_df = train_df.drop(columns=dropped), test_df.drop(columns=dropped)
        feature_cols = keep
        report_data['feature_selection_method'] = feature_selection_method
        report_data['selected_features'] = len(keep)
        report_data['feature_reduction_pct'] = round((1 - len(keep) / report_data['original_features']) * 100, 2)
        logger.info(f"Feature selection complete. Reduced from {report_data['original_features']} to {len(keep)} features")

    n_features = len(feature_cols)
    report_data['n_features'] = n_features

    classifier = Classifier()
    logger.info("Setting up endgame classification experiment...")
    try:
        classifier.setup_experiment(
            data=train_df, test_data=test_df, target=target_column, session_id=123,
            verbose=False, fold=5, fold_groups=group_col,
            ignore_features=[c for c in id_cols if c != group_col],
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not set up the experiment on {len(train_df)} training and "
            f"{len(test_df)} test rows with {n_features} features: {e}"
        ) from e

    # Compare models
    logger.info(f"Comparing top {n_models_to_compare} models...")

    exclude_models = []

    n_samples = len(df)
    n_features_check = n_features

    if n_samples > 5000 or n_features_check > 100:
        logger.info(f"Large dataset detected ({n_samples} samples, {n_features_check} features). Excluding slow models...")
        exclude_models.extend(['svm', 'qda'])

    if n_samples > 10000:
        logger.info("Very large dataset. Also excluding KNN...")
        exclude_models.append('knn')

    logger.info(f"Excluded models: {exclude_models}")

    try:
        import time
        start_time = time.time()

        best_models = classifier.compare_models(
            fold=5,
            round=4,
            cross_validation=True,
            sort='Accuracy',
            n_select=n_models_to_compare,
            turbo=True,
            verbose=True,
            exclude=exclude_models,
            budget_time=budget_time_minutes,
            errors='ignore'
        )

        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Model comparison completed in {elapsed_time:.2f} minutes")

        # Get leaderboard
        report_data['leaderboard'] = classifier.comparison_results

        # Get best model info
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        report_data['best_model_name'] = type(best_model).__name__

        logger.info(f"Best model: {report_data['best_model_name']}")

        tree_based_patterns = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGB', 'xgb', 'XGBoost', 'LGBM', 'CatBoost', 'DecisionTree', 'AdaBoost']
        is_tree_based = any(pattern.lower() in report_data['best_model_name'].lower() for pattern in tree_based_patterns)
        logger.info(f"Model supports SHAP interpretation: {is_tree_based}")

    except Exception as e:
        raise RuntimeError(
            f"Could not compare models: {e}. Check that the features are numeric and free of "
            "NaN, that every class has enough participants for the CV folds, and that "
            "budget_time_minutes leaves time for at least one model."
        ) from e

    # Tune the best model
    if tune_best_model:
        logger.info("Tuning hyperparameters for the best model...")
        try:
            predictions_before = classifier.predict_model(estimator=best_model, verbose=False)

            tuned_model = classifier.tune_model(
                estimator=best_model,
                fold=5,
                n_iter=20,
                optimize='Accuracy',
                choose_better=True,
                verbose=True
            )

            predictions_after = classifier.predict_model(estimator=tuned_model, verbose=False)

            report_data['tuning_results'] = True
            report_data['tuning_metric'] = 'Accuracy'
            report_data['tuning_iterations'] = 20

            try:
                if hasattr(tuned_model, 'get_params'):
                    report_data['best_hyperparameters'] = tuned_model.get_params()
                else:
                    report_data['best_hyperparameters'] = {}
            except:
                report_data['best_hyperparameters'] = {}

            best_model = tuned_model
            logger.info("Hyperparameter tuning completed")

        except Exception as e:
            logger.warning(f"Failed to tune model: {e}. Using untuned model.")
            report_data['tuning_results'] = False

    # Two different numbers: the selected model's cross-validated scores on the training
    # split (leaderboard row, before tuning), and the final model scored once on the
    # held-out test split.
    cv_row = classifier.comparison_results.iloc[0].to_dict()
    report_data['best_model_metrics'] = {k: v for k, v in cv_row.items()
                                         if isinstance(v, (int, float)) and not pd.isna(v)}
    report_data['test_metrics'] = {k: v for k, v in classifier.evaluate_model(best_model).items()
                                   if not pd.isna(v)}
    logger.info(f"Held-out test metrics of the final model: {report_data['test_metrics']}")

    # Try to generate endgame's comprehensive report
    if ENDGAME_VIS_AVAILABLE:
        try:
            logger.info("Generating endgame comprehensive classification report...")
            endgame_report_path = plots_dir / "endgame_report.html"
            classifier.generate_report(
                estimator=best_model,
                output_path=str(endgame_report_path),
            )
            logger.info(f"Endgame report saved to {endgame_report_path}")
        except Exception as e:
            logger.warning(f"Endgame report generation failed: {e}. Falling back to custom plots.")

    # Generate plots
    if generate_plots:
        logger.info("Generating model visualizations...")

        is_multiclass = report_data.get('n_classes', 2) > 2

        # Get the actual class labels
        try:
            original_class_labels = sorted(df[target_column].unique())
            logger.info(f"Original class labels: {original_class_labels}")
        except Exception as e:
            logger.warning(f"Could not extract class labels: {e}")
            original_class_labels = None

        import matplotlib
        matplotlib.use('Agg')

        # Generate confusion matrix
        try:
            logger.info("Generating confusion matrix plot...")
            from sklearn.metrics import confusion_matrix

            holdout_pred = classifier.predict_model(estimator=best_model, verbose=False)
            y_true = holdout_pred[target_column]
            y_pred = holdout_pred['prediction_label']

            cm = confusion_matrix(y_true, y_pred, labels=original_class_labels)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm,
                      annot=True,
                      fmt='d',
                      cmap='Blues',
                      xticklabels=original_class_labels,
                      yticklabels=original_class_labels,
                      cbar_kws={'label': 'Count'})

            plt.title(f'{type(best_model).__name__} Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Class', fontsize=12)
            plt.ylabel('True Class', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            confusion_matrix_path = plots_dir / 'confusion_matrix.png'
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to {confusion_matrix_path}")

        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix: {e}")

        # Try to get feature importance
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = classifier.get_config('X_train').columns.tolist()
                importances = best_model.feature_importances_
                feature_importance = sorted(zip(feature_names, importances),
                                         key=lambda x: x[1], reverse=True)
                report_data['feature_importance'] = feature_importance

                # Plot feature importance
                plt.figure(figsize=(12, 8))
                top_n = min(20, len(feature_importance))
                top_features = feature_importance[:top_n]
                names = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                plt.barh(range(top_n), values[::-1])
                plt.yticks(range(top_n), names[::-1])
                plt.xlabel('Importance')
                plt.title('Top Feature Importances', fontsize=16, fontweight='bold')
                plt.tight_layout()
                feat_path = plots_dir / 'feature.png'
                plt.savefig(feat_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Feature importance plot saved to {feat_path}")

            elif hasattr(best_model, 'coef_'):
                feature_names = classifier.get_config('X_train').columns.tolist()
                importances = np.abs(best_model.coef_).mean(axis=0) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
                feature_importance = sorted(zip(feature_names, importances),
                                         key=lambda x: x[1], reverse=True)
                report_data['feature_importance'] = feature_importance
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        # Generate SHAP interpretation
        try:
            logger.info("Generating model interpretation plots...")

            model_type = type(best_model).__name__

            tree_based_patterns = [
                'RandomForest', 'ExtraTrees', 'GradientBoosting', 'AdaBoost', 'DecisionTree',
                'XGB', 'xgb', 'XGBoost', 'xgboost',
                'LGBM', 'lgb', 'LightGBM', 'lightgbm',
                'CatBoost', 'catboost', 'Cat',
            ]

            is_tree_based = any(pattern.lower() in model_type.lower() for pattern in tree_based_patterns)

            interpretation_generated = False

            # Try endgame explain first
            if ENDGAME_VIS_AVAILABLE:
                try:
                    logger.info("Using endgame.explain for model interpretation...")
                    X_train = classifier.get_config('X_train')
                    sample_size = min(100, len(X_train))
                    X_sample = X_train.sample(n=sample_size, random_state=42)

                    explanation = eg_explain(best_model, X_sample, method="shap")

                    # If endgame returns a plot, save it
                    if hasattr(explanation, 'plot'):
                        explanation.plot()
                        shap_path = plots_dir / 'shap_summary.png'
                        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        interpretation_generated = True
                        logger.info(f"Endgame SHAP plot saved to {shap_path}")
                except Exception as eg_err:
                    logger.info(f"Endgame explain failed ({eg_err}), falling back to direct SHAP")

            # Fallback: direct SHAP
            if not interpretation_generated and (is_tree_based or 'boost' in model_type.lower()):
                try:
                    import shap

                    X_train = classifier.get_config('X_train')
                    sample_size = min(100, len(X_train))
                    X_sample = X_train.sample(n=sample_size, random_state=42)

                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)

                    class_labels = original_class_labels if original_class_labels else [f"Class {i}" for i in range(len(shap_values) if isinstance(shap_values, list) else 1)]

                    plt.figure(figsize=(12, 8))

                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        shap.summary_plot(
                            shap_values, X_sample,
                            class_names=class_labels,
                            show=False, max_display=20
                        )
                    else:
                        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
                        shap.summary_plot(shap_vals, X_sample, show=False, max_display=20)

                    plt.tight_layout()
                    shap_path = plots_dir / 'shap_summary.png'
                    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    interpretation_generated = True
                    logger.info(f"SHAP summary plot saved to {shap_path}")

                except ImportError:
                    logger.warning("SHAP library not available. Install with: pip install shap")
                except Exception as shap_error:
                    logger.warning(f"SHAP interpretation failed: {shap_error}")

            if not interpretation_generated:
                logger.warning("Could not generate any model interpretation plots")

        except Exception as e:
            logger.error(f"Critical error in interpretation generation: {e}", exc_info=True)

        # Generate dimensionality reduction visualizations
        try:
            logger.info("Generating dimensionality reduction visualizations...")

            X_train = classifier.get_config('X_train')
            y_train_original = classifier.get_config('y_train')

            y_labels = y_train_original

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            # PCA
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)

                plt.figure(figsize=(12, 8))
                colors = plt.cm.Set1(np.linspace(0, 1, len(original_class_labels)))

                for i, class_name in enumerate(original_class_labels):
                    mask = y_labels == class_name
                    if mask.any():
                        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                  c=[colors[i]], label=class_name, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)', fontsize=12)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)', fontsize=12)
                plt.title(f'PCA Visualization - {pca.explained_variance_ratio_.sum():.1%} Total Variance Explained',
                         fontsize=16, fontweight='bold')
                plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                plt.grid(alpha=0.3)
                plt.tight_layout()

                pca_path = plots_dir / 'pca_visualization.png'
                plt.savefig(pca_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"PCA visualization saved to {pca_path}")

            except Exception as pca_error:
                logger.warning(f"Could not generate PCA visualization: {pca_error}")

            # t-SNE
            try:
                from sklearn.manifold import TSNE

                max_samples_tsne = 2000
                if len(X_scaled) > max_samples_tsne:
                    from sklearn.model_selection import train_test_split
                    _, X_tsne_sample, _, y_tsne_sample = train_test_split(
                        X_scaled, y_labels,
                        test_size=max_samples_tsne,
                        stratify=y_labels,
                        random_state=42
                    )
                else:
                    X_tsne_sample = X_scaled
                    y_tsne_sample = y_labels

                perplexity = min(30, len(X_tsne_sample)//4)

                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                           n_iter=1000, learning_rate=200)
                X_tsne = tsne.fit_transform(X_tsne_sample)

                plt.figure(figsize=(12, 8))

                for i, class_name in enumerate(original_class_labels):
                    mask = y_tsne_sample == class_name
                    if mask.any():
                        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                  c=[colors[i]], label=class_name, alpha=0.7, s=60,
                                  edgecolors='black', linewidth=0.5)

                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.title(f't-SNE Visualization (perplexity={perplexity})', fontsize=16, fontweight='bold')
                plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                plt.grid(alpha=0.3)
                plt.tight_layout()

                tsne_path = plots_dir / f'tsne_visualization_perp{perplexity}.png'
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"t-SNE visualization saved to {tsne_path}")

            except Exception as tsne_error:
                logger.warning(f"Could not generate t-SNE visualization: {tsne_error}")

            # Class distribution
            try:
                plt.figure(figsize=(10, 6))
                class_counts = df[target_column].value_counts()
                colors_dist = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))

                bars = plt.bar(class_counts.index, class_counts.values, color=colors_dist)
                plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
                plt.xlabel('Class')
                plt.ylabel('Number of Samples')
                plt.xticks(rotation=45, ha='right')

                for bar, count in zip(bars, class_counts.values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom', fontweight='bold')

                plt.tight_layout()
                dist_path = plots_dir / 'class_distribution.png'
                plt.savefig(dist_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Class distribution plot saved to {dist_path}")

            except Exception as dist_error:
                logger.warning(f"Could not generate class distribution plot: {dist_error}")

        except Exception as e:
            logger.warning(f"Failed to generate dimensionality reduction plots: {e}")

    # Save the final model
    try:
        model_path = output_path / "final_classifier_model"
        classifier.save_model(
            model=best_model,
            model_name=str(model_path),
            verbose=True
        )
        logger.info(f"Model saved to {model_path}.pkl")
    except Exception as e:
        logger.warning(f"Failed to save model: {e}")

    # Generate comprehensive HTML report
    report_path = output_path / "classification_report.html"
    generate_classification_report_html(report_data, str(report_path), str(plots_dir))

    # Try to open the report
    try:
        report_abs_path = report_path.resolve()
        webbrowser.open(f"file://{report_abs_path}")
        logger.info(f"Attempted to open report in browser: file://{report_abs_path}")
    except Exception as e:
        logger.info(f"Could not automatically open the report in browser: {e}")

    logger.info("Classification report generation completed successfully!")

    return classifier, best_model, report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a classification report for a given dataset.")
    parser.add_argument('--input-csv-path', type=str, help='Path to a single CSV file for training and testing.')
    parser.add_argument('--train-csv-path', type=str, help='Path to training data CSV.')
    parser.add_argument('--test-csv-path', type=str, help='Path to test data CSV.')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save outputs.')
    parser.add_argument('--exclude-features-file', type=str, help='Path to a text file with features to exclude (one per line).')

    parser.add_argument('--use-feature-selection', action='store_true', help='Enable feature selection. Disabled by default.')
    parser.add_argument('--feature-selection-method', type=str, default='k_best', help='FeatureSelector method, fitted on the training split.')

    parser.add_argument('--n-models-to-compare', type=int, default=5, help='Number of models to compare.')
    parser.add_argument('--tune-best-model', action='store_true', help='Enable hyperparameter tuning of the best model. Disabled by default.')
    parser.add_argument('--generate-plots', action='store_true', help='Enable plot generation. Disabled by default.')
    parser.add_argument('--budget-time-minutes', type=float, default=30.0, help='Time budget for model comparison in minutes.')

    args = parser.parse_args()

    exclude_features_list = []
    if args.exclude_features_file:
        try:
            with open(args.exclude_features_file, 'r') as f:
                exclude_features_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(exclude_features_list)} features to exclude from {args.exclude_features_file}")
        except Exception as e:
            logger.error(f"Could not read exclude features file: {e}")
            sys.exit(1)

    # Run the report generation. Bad input and engine failures are reported as one line
    # with a non-zero exit status, not as a traceback.
    try:
        generate_report(
            input_csv_path=args.input_csv_path,
            train_csv_path=args.train_csv_path,
            test_csv_path=args.test_csv_path,
            use_feature_selection=args.use_feature_selection,
            feature_selection_method=args.feature_selection_method,
            target_column=args.target_column,
            exclude_features=exclude_features_list,
            output_dir=args.output_dir,
            n_models_to_compare=args.n_models_to_compare,
            tune_best_model=args.tune_best_model,
            generate_plots=args.generate_plots,
            budget_time_minutes=args.budget_time_minutes
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
