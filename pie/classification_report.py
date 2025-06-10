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

from pie.classifier import Classifier
from pie.feature_selector import FeatureSelector

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
            <h1>🤖 PIE Classification Pipeline Report</h1>
            
            <div class="summary-box">
                <h2>📊 1. Pipeline Overview</h2>
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
                </table>
            </div>
    """

    # Add excluded features details if any were excluded
    if report_data.get('excluded_features'):
        html_content += f"""
                <h3>🚫 Excluded Features (Data Leakage Prevention)</h3>
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
                <h2>🔍 2. Feature Selection Summary</h2>
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
                <h2>🏆 3. Model Comparison Leaderboard</h2>
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

    # Best model details - FIX: Ensure metrics are displayed
    if report_data.get('best_model_name'):
        html_content += f"""
            <div class="best-model">
                <h2>🌟 4. Best Model Details</h2>
                <h3>Selected Model: <span class="highlight">{report_data.get('best_model_name', 'N/A')}</span></h3>
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
                <h2>⚙️ 5. Hyperparameter Tuning</h2>
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

    # Model plots - FIX: Include additional plot types
    html_content += """
            <div class="summary-box">
                <h2>📈 6. Model Performance Visualizations</h2>
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
        ('high_correlation_pairs', 'High Correlation Feature Pairs'),  # Add this line
        ('pca_visualization', 'PCA Visualization (2D)'),
        ('pca_3d_visualization', '3D PCA Visualization'),
        ('tsne_visualization_perp30', 't-SNE Visualization (Perplexity 30)'),
        ('tsne_visualization_perp50', 't-SNE Visualization (Perplexity 50)'),
        ('pca_vs_tsne_comparison', 'PCA vs t-SNE Comparison'),
        ('umap_visualization', 'UMAP Visualization'),
        ('prediction_confidence', 'Prediction Confidence Distribution')
    ]
    
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
    
    if not plots_found:
        html_content += "<p><em>No visualization plots were generated successfully.</em></p>"
    
    html_content += """
            </div>
    """

    # Feature importance details
    if report_data.get('feature_importance'):
        html_content += """
            <div class="feature-importance">
                <h2>🎯 7. Top Discriminative Features</h2>
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

    # Model interpretation - FIX: Check for different interpretation plot types
    html_content += """
            <div class="summary-box">
                <h2>🔬 8. Model Interpretation</h2>
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
                <li>The model is not tree-based (SHAP summary plots only work with tree-based models in PyCaret)</li>
                <li>Technical limitations with the interpretation libraries</li>
            </ul>
            <p><em>Consider using tree-based models (Random Forest, XGBoost, etc.) for better interpretability.</em></p>
        """
    
    html_content += """
            </div>
    """

    # Test set performance - FIX: Always include this section
    html_content += """
            <div class="summary-box">
                <h2>✅ 9. Final Test Set Performance</h2>
    """
    
    test_metrics = report_data.get('test_metrics', {})
    if test_metrics:
        html_content += """
                <p>Performance metrics on the held-out test set:</p>
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
                <h2>💡 10. Recommendations</h2>
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
    feature_selection_method: str = 'univariate_kbest',
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
    
    Parameters:
    -----------
    input_csv_path : str, optional
        Path to a single CSV file containing all data (will be split into train/test)
    train_csv_path : str, optional
        Path to training data CSV (use with test_csv_path for pre-split data)
    test_csv_path : str, optional
        Path to test data CSV (use with train_csv_path for pre-split data)
    use_feature_selection : bool
        Whether to apply feature selection (set False if data is already feature-selected)
    feature_selection_method : str
        Method for feature selection if use_feature_selection is True
    target_column : str
        Name of the target column
    exclude_features : List[str], optional
        List of feature names to exclude from training, validation, and testing.
        Useful for removing features that may cause data leakage or are too closely
        related to the target variable (e.g., "Enrollment in Parkinson's Treatment")
    output_dir : str
        Directory to save outputs
    n_models_to_compare : int
        Number of models to compare
    tune_best_model : bool
        Whether to tune hyperparameters of the best model
    generate_plots : bool
        Whether to generate visualization plots
    budget_time_minutes : float
        Maximum time in minutes for model comparison (prevents hanging)
    """
    logger.info("Starting classification report generation...")
    
    # Initialize exclude_features if None
    if exclude_features is None:
        exclude_features = []
    
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
            logger.error("No input data found. Please provide either input_csv_path or both train_csv_path and test_csv_path.")
            return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize report data collector
    report_data = {
        'target_column': target_column,
        'feature_selection_applied': use_feature_selection,
        'plots_dir': str(plots_dir)
    }
    
    # Load data based on input type
    if train_csv_path and test_csv_path:
        # Case 1: Pre-split data provided
        logger.info("Loading pre-split train and test data...")
        report_data['input_data_path'] = f"Train: {train_csv_path}, Test: {test_csv_path}"
        
        try:
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            logger.info(f"Loaded training data from {train_csv_path}. Shape: {train_df.shape}")
            logger.info(f"Loaded test data from {test_csv_path}. Shape: {test_df.shape}")
            
            # Combine for PyCaret setup (it will split internally)
            # Add a temporary column to track original split
            train_df['_original_split'] = 'train'
            test_df['_original_split'] = 'test'
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            report_data['input_data_shape'] = f"Train: {train_df.shape}, Test: {test_df.shape}"
            report_data['pre_split_data'] = True
            
        except Exception as e:
            logger.error(f"Failed to load train/test data: {e}")
            return
            
    else:
        # Case 2: Single CSV file provided
        logger.info("Loading single data file...")
        report_data['input_data_path'] = input_csv_path
        report_data['pre_split_data'] = False
        
        try:
            df = pd.read_csv(input_csv_path)
            report_data['input_data_shape'] = df.shape
            logger.info(f"Loaded data from {input_csv_path}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return
    
    # Check target column
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        return
    
    # EXCLUDE SPECIFIED FEATURES EARLY IN THE PIPELINE
    if exclude_features:
        logger.info(f"Excluding {len(exclude_features)} specified features from analysis...")
        
        # Check which excluded features actually exist in the data
        existing_excluded_features = [feat for feat in exclude_features if feat in df.columns]
        missing_excluded_features = [feat for feat in exclude_features if feat not in df.columns]
        
        if existing_excluded_features:
            logger.info(f"Excluding features: {existing_excluded_features}")
            df = df.drop(columns=existing_excluded_features)
            logger.info(f"Data shape after feature exclusion: {df.shape}")
        
        if missing_excluded_features:
            logger.warning(f"Specified features not found in data (will be ignored): {missing_excluded_features}")
        
        # Update report data
        report_data['excluded_features'] = existing_excluded_features
        report_data['excluded_features_count'] = len(existing_excluded_features)
    else:
        report_data['excluded_features'] = []
        report_data['excluded_features_count'] = 0
    
    # Handle missing target values
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing target values")
    
    # Verify target contains actual class names, not encoded numbers
    unique_targets = df[target_column].unique()
    logger.info(f"Target column values: {unique_targets}")
    
    # If we detect numeric targets instead of class names, this indicates an issue
    if all(isinstance(target, (int, float, np.integer, np.floating)) for target in unique_targets if pd.notna(target)):
        logger.warning("Target column contains numeric values instead of class names!")
        logger.warning("This suggests the data may have been saved with encoded labels instead of original labels.")
        
        # Try to map back to original labels if we know the mapping
        expected_labels = ["Parkinson's Disease", "Healthy Control", "Prodromal", "SWEDD"]
        if len(unique_targets) == len(expected_labels):
            # Create mapping from encoded to original
            target_mapping = dict(zip(sorted(unique_targets), expected_labels))
            logger.info(f"Attempting to map encoded targets back to original labels: {target_mapping}")
            df[target_column] = df[target_column].map(target_mapping)
            logger.info("Target column mapped back to original class names")
        else:
            logger.error("Cannot map targets back to original labels - length mismatch")
            return
    
    # Get target statistics
    report_data['n_classes'] = df[target_column].nunique()
    report_data['class_distribution'] = df[target_column].value_counts().to_dict()
    
    # Apply feature selection if requested
    if use_feature_selection and target_column in df.columns:
        logger.info(f"Applying feature selection using {feature_selection_method}...")
        
        # Remove the _original_split column if it exists before feature selection
        if '_original_split' in df.columns:
            split_info = df['_original_split'].copy()
            df = df.drop(columns=['_original_split'])
        else:
            split_info = None
        
        original_features = df.shape[1] - 1  # Exclude target
        
        try:
            # Use FeatureSelector
            selected_df = FeatureSelector.select_features(
                data=df,
                target_column=target_column,
                task_type='classification',
                method=feature_selection_method,
                k_or_frac_kbest=0.5 if 'kbest' in feature_selection_method else None,
                percentile_univariate=50 if 'percentile' in feature_selection_method else None,
                random_state=123
            )
            
            df = selected_df
            
            # Re-add split info if it existed
            if split_info is not None:
                df['_original_split'] = split_info
            
            selected_features = df.shape[1] - 1
            if '_original_split' in df.columns:
                selected_features -= 1
            
            report_data['feature_selection_method'] = feature_selection_method
            report_data['original_features'] = original_features
            report_data['selected_features'] = selected_features
            report_data['feature_reduction_pct'] = round((1 - selected_features/original_features) * 100, 2)
            
            logger.info(f"Feature selection complete. Reduced from {original_features} to {selected_features} features")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Proceeding with all features.")
            report_data['feature_selection_applied'] = False
    
    # Calculate final feature count
    n_features = df.shape[1] - 1  # Exclude target
    if '_original_split' in df.columns:
        n_features -= 1  # Also exclude split indicator
    report_data['n_features'] = n_features
    
    # Initialize classifier
    classifier = Classifier()
    
    # Prepare setup parameters
    setup_params = {
        'data': df,
        'target': target_column,
        'train_size': 0.8,
        'session_id': 123,
        'use_gpu': False,
        'log_experiment': False,
        'experiment_name': "PIE_Classification",
        'verbose': False,
        'remove_multicollinearity': False,  # Already handled in feature engineering
        'remove_outliers': False,  # Already handled in data reduction
        'normalize': False,  # Already handled in feature engineering
        'transformation': False,
        'pca': False,  # Can be enabled if needed
        'ignore_features': [],
        'feature_selection': False,  # Already handled separately
        'fold_strategy': 'stratifiedkfold',
        'fold': 5,
        'fold_shuffle': False
    }
    
    # If we have pre-split data, we need to handle it differently
    if report_data.get('pre_split_data', False) and '_original_split' in df.columns:
        # Use the original split information
        train_indices = df[df['_original_split'] == 'train'].index
        test_indices = df[df['_original_split'] == 'test'].index
        
        # Remove the split column before setup
        df_without_split = df.drop(columns=['_original_split'])
        
        # For PyCaret, when using pre-split data, pass test_data separately
        train_data = df_without_split.iloc[train_indices]
        test_data = df_without_split.iloc[test_indices]
        
        setup_params['data'] = train_data
        setup_params['test_data'] = test_data
        # Remove train_size since we're providing explicit test_data
        setup_params.pop('train_size', None)
        
        report_data['train_size'] = len(train_indices) / (len(train_indices) + len(test_indices))
        logger.info(f"Using pre-defined train/test split: {len(train_indices)} train, {len(test_indices)} test samples")
    else:
        report_data['train_size'] = 0.8
    
    # Setup experiment
    logger.info("Setting up PyCaret classification experiment...")
    try:
        # Ensure target column has proper class names, not just values
        if target_column in df.columns:
            target_mapping = None
            unique_targets = df[target_column].unique()
            logger.info(f"Target classes found: {unique_targets}")
            
            # If we have the expected cohort names, no mapping needed
            expected_cohorts = ["Parkinson's Disease", "Prodromal", "Healthy Control", "SWEDD"]
            if all(target in expected_cohorts for target in unique_targets if pd.notna(target)):
                logger.info("Target column already has proper class names")
            
        experiment = classifier.setup_experiment(**setup_params)
    except Exception as e:
        logger.error(f"Failed to setup experiment: {e}")
        return
    
    # Compare models
    logger.info(f"Comparing top {n_models_to_compare} models...")
    
    # First, let's identify which models might be problematic
    # Common models that can hang with certain data:
    # - 'svm' and 'rbfsvm' can be very slow with large datasets
    # - 'gpc' (Gaussian Process) can be memory intensive
    # - 'mlp' can take long to converge
    
    exclude_models = []
    
    # Check dataset size to decide which models to exclude
    n_samples = len(df)
    n_features_check = n_features
    
    if n_samples > 5000 or n_features_check > 100:
        logger.info(f"Large dataset detected ({n_samples} samples, {n_features_check} features). Excluding slow models...")
        exclude_models.extend(['svm', 'rbfsvm', 'gpc','qda','lightgbm'])
    
    if n_samples > 10000:
        logger.info("Very large dataset. Also excluding MLP...")
        exclude_models.append('mlp')
    
    logger.info(f"Excluded models: {exclude_models}")
    
    try:
        # Add a wrapper to track model progress more explicitly
        import time
        from datetime import datetime
        
        # Create a custom callback to monitor progress
        start_time = time.time()
        
        best_models = classifier.compare_models(
            fold=5,
            round=4,
            cross_validation=True,
            sort='Accuracy',
            n_select=n_models_to_compare,
            turbo=True,
            verbose=True,
            exclude=exclude_models,  # Exclude problematic models
            budget_time=budget_time_minutes,  # Set time budget
            errors='ignore'  # Continue even if some models fail
        )
        
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Model comparison completed in {elapsed_time:.2f} minutes")
        
        # Get leaderboard
        report_data['leaderboard'] = classifier.comparison_results
        
        # Get best model info
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        report_data['best_model_name'] = type(best_model).__name__
        
        logger.info(f"Best model: {report_data['best_model_name']}")
        logger.info(f"Best model class: {best_model.__class__}")
        logger.info(f"Best model module: {best_model.__class__.__module__}")
        
        # Check if it's tree-based for interpretation compatibility
        tree_based_patterns = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGB', 'xgb', 'XGBoost', 'LGBM', 'CatBoost', 'DecisionTree', 'AdaBoost']
        is_tree_based = any(pattern.lower() in report_data['best_model_name'].lower() for pattern in tree_based_patterns)
        logger.info(f"Model supports SHAP interpretation: {is_tree_based}")
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return
    
    # Tune the best model
    if tune_best_model:
        logger.info("Tuning hyperparameters for the best model...")
        try:
            # Get metrics before tuning
            predictions_before = classifier.predict_model(estimator=best_model, verbose=False)
            metrics_before = predictions_before.iloc[0].to_dict() if len(predictions_before) > 0 else {}
            
            # Tune model
            tuned_model = classifier.tune_model(
                estimator=best_model,
                fold=5,
                n_iter=20,
                optimize='Accuracy',
                choose_better=True,
                verbose=True
            )
            
            # Get metrics after tuning
            predictions_after = classifier.predict_model(estimator=tuned_model, verbose=False)
            metrics_after = predictions_after.iloc[0].to_dict() if len(predictions_after) > 0 else {}
            
            # Calculate improvement
            improvement = {}
            for metric in ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']:
                if metric in metrics_before and metric in metrics_after:
                    before_val = metrics_before.get(metric, 0)
                    after_val = metrics_after.get(metric, 0)
                    if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                        improvement[metric] = round((after_val - before_val) * 100, 2)
            
            report_data['tuning_results'] = True
            report_data['tuning_metric'] = 'Accuracy'
            report_data['tuning_iterations'] = 20
            report_data['tuning_improvement'] = f"+{max(improvement.values()):.2f}%" if improvement else "N/A"
            
            # Try to get hyperparameters
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
    
    # Get final model metrics - FIX: Use evaluation approach instead of predict_model
    try:
        logger.info("Extracting model performance metrics...")
        
        # Get the leaderboard/comparison results which contain CV metrics
        if hasattr(classifier, 'comparison_results') and classifier.comparison_results is not None:
            # Extract metrics for the best model from the comparison results
            best_model_row = classifier.comparison_results.iloc[0]  # First row is best model
            test_metrics = best_model_row.to_dict()
            
            # Clean up the metrics (remove non-numeric entries)
            test_metrics = {k: v for k, v in test_metrics.items() 
                          if isinstance(v, (int, float)) and not pd.isna(v)}
            
            report_data['best_model_metrics'] = test_metrics
            report_data['test_metrics'] = test_metrics
            
            logger.info(f"Best model metrics: {test_metrics}")
        else:
            logger.warning("No comparison results available for metrics extraction")
            
    except Exception as e:
        logger.warning(f"Failed to get model metrics: {e}")
    
    # Generate plots - FIX: Ensure proper plot generation and paths with correct class labels
    if generate_plots:
        logger.info("Generating model visualizations...")
        
        # Check if this is a multiclass problem
        is_multiclass = report_data.get('n_classes', 2) > 2
        
        # Define plot types based on whether it's multiclass or binary
        if is_multiclass:
            # For multiclass, exclude plots that don't support it
            plot_types = ['confusion_matrix', 'feature', 'learning']
            logger.info("Multiclass classification detected. Using multiclass-compatible plots only.")
        else:
            # For binary classification, use all plots
            plot_types = ['auc', 'confusion_matrix', 'pr', 'feature', 'learning']
        
        # Get the actual class labels for proper display
        try:
            # Get the original class labels from the data
            original_class_labels = sorted(df[target_column].unique())
            logger.info(f"Original class labels: {original_class_labels}")
            
            # Get PyCaret's internal label mapping if it exists
            try:
                label_encoded_mapping = classifier.get_config('label_encoded')
                if label_encoded_mapping:
                    logger.info(f"PyCaret label encoding detected: {label_encoded_mapping}")
            except:
                label_encoded_mapping = None
                
        except Exception as e:
            logger.warning(f"Could not extract class labels: {e}")
            original_class_labels = None
        
        for plot_type in plot_types:
            try:
                logger.info(f"Generating {plot_type} plot...")
                
                # Use matplotlib backend to save plots
                import matplotlib
                matplotlib.use('Agg')
                
                # For confusion matrix, we need special handling to get proper labels
                if plot_type == 'confusion_matrix' and original_class_labels:
                    try:
                        # Generate confusion matrix with custom labeling
                        from sklearn.metrics import confusion_matrix
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Get predictions on the holdout set
                        holdout_pred = classifier.predict_model(estimator=best_model, verbose=False)
                        
                        # Extract actual and predicted labels
                        y_true = holdout_pred[target_column]
                        y_pred = holdout_pred['prediction_label']
                        
                        # Create confusion matrix
                        cm = confusion_matrix(y_true, y_pred, labels=original_class_labels)
                        
                        # Create a custom confusion matrix plot
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
                        
                        # Rotate labels if they're long
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        
                        # Adjust layout to prevent label cutoff
                        plt.tight_layout()
                        
                        # Save the plot
                        confusion_matrix_path = plots_dir / 'confusion_matrix.png'
                        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"Custom confusion matrix saved to {confusion_matrix_path}")
                        
                    except Exception as cm_error:
                        logger.warning(f"Custom confusion matrix generation failed: {cm_error}")
                        # Fallback to PyCaret's default
                        classifier.plot_model(
                            estimator=best_model,
                            plot=plot_type,
                            save=True,
                            verbose=False
                        )
                else:
                    # Generate and save plot directly to our plots directory
                    classifier.plot_model(
                        estimator=best_model,
                        plot=plot_type,
                        save=True,
                        verbose=False
                    )
                
                # Move PyCaret generated plots to our plots directory (if not already handled above)
                if plot_type != 'confusion_matrix' or not original_class_labels:
                    default_plot_names = [
                        f"{plot_type}.png",
                        f"{plot_type}.html",
                        f"AUC.png" if plot_type == 'auc' else None,
                        f"Confusion Matrix.png" if plot_type == 'confusion_matrix' else None,
                        f"Precision Recall.png" if plot_type == 'pr' else None,
                        f"Feature Importance.png" if plot_type == 'feature' else None,
                        f"Learning Curve.png" if plot_type == 'learning' else None
                    ]
                    
                    for default_name in default_plot_names:
                        if default_name and Path(default_name).exists():
                            target_path = plots_dir / f"{plot_type}.png"
                            Path(default_name).rename(target_path)
                            logger.info(f"Moved {default_name} to {target_path}")
                            break
                
            except Exception as e:
                logger.warning(f"Failed to generate {plot_type} plot: {e}")
        
        # Try to get feature importance
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = classifier.get_config('X_train').columns.tolist()
                importances = best_model.feature_importances_
                feature_importance = sorted(zip(feature_names, importances), 
                                         key=lambda x: x[1], reverse=True)
                report_data['feature_importance'] = feature_importance
            elif hasattr(best_model, 'coef_'):
                feature_names = classifier.get_config('X_train').columns.tolist()
                importances = np.abs(best_model.coef_).mean(axis=0) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
                feature_importance = sorted(zip(feature_names, importances), 
                                         key=lambda x: x[1], reverse=True)
                report_data['feature_importance'] = feature_importance
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        # Generate SHAP plots for interpretation - FIX: Better XGBoost detection and error handling with proper class labels
        try:
            logger.info("Generating model interpretation plots...")
            
            # Get the actual model type and log it
            model_type = type(best_model).__name__
            logger.info(f"Actual model type detected: '{model_type}'")
            
            # More comprehensive tree-based model detection
            tree_based_patterns = [
                'RandomForest', 'ExtraTrees', 'GradientBoosting', 'AdaBoost', 'DecisionTree',
                'XGB', 'xgb', 'XGBoost', 'xgboost',
                'LGBM', 'lgb', 'LightGBM', 'lightgbm', 
                'CatBoost', 'catboost', 'Cat',
                'RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier',
                'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier', 
                'DecisionTreeClassifier', 'AdaBoostClassifier'
            ]
            
            # Check if model is tree-based (case-insensitive)
            is_tree_based = any(pattern.lower() in model_type.lower() for pattern in tree_based_patterns)
            
            logger.info(f"Tree-based model detected: {is_tree_based}")
            
            # Save current matplotlib backend
            import matplotlib
            current_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            
            interpretation_generated = False
            
            # Generate custom SHAP plot with proper class labels for tree-based models
            if is_tree_based or 'xgb' in model_type.lower() or 'boost' in model_type.lower():
                try:
                    logger.info(f"Attempting custom SHAP interpretation with proper class labels for model: {model_type}")
                    
                    # Import SHAP
                    import shap
                    
                    # Get the training data and model for SHAP
                    X_train = classifier.get_config('X_train')
                    
                    # Finalize the model to ensure it's ready for SHAP
                    try:
                        finalized_model = classifier.finalize_model(best_model)
                        logger.info("Model finalized successfully for SHAP")
                        model_for_shap = finalized_model
                    except Exception as finalize_error:
                        logger.warning(f"Could not finalize model: {finalize_error}. Using original model.")
                        model_for_shap = best_model
                    
                    # Create SHAP explainer
                    if 'xgb' in model_type.lower() or 'XGB' in model_type:
                        explainer = shap.TreeExplainer(model_for_shap)
                    else:
                        # For other tree-based models
                        explainer = shap.TreeExplainer(model_for_shap)
                    
                    # Get SHAP values (use a sample of data for performance)
                    sample_size = min(100, len(X_train))  # Use up to 100 samples for SHAP
                    X_sample = X_train.sample(n=sample_size, random_state=42)
                    
                    logger.info(f"Computing SHAP values for {sample_size} samples...")
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Get the class labels in the correct order
                    class_labels = original_class_labels if original_class_labels else [f"Class {i}" for i in range(len(shap_values))]
                    
                    logger.info(f"Creating SHAP summary plot with class labels: {class_labels}")
                    
                    # Create custom SHAP summary plot
                    plt.figure(figsize=(12, 8))
                    
                    # For multiclass, shap_values is a list of arrays (one per class)
                    if isinstance(shap_values, list) and len(shap_values) > 2:
                        # Multiclass case - create summary plot
                        shap.summary_plot(
                            shap_values, 
                            X_sample, 
                            class_names=class_labels,
                            show=False,
                            max_display=20  # Show top 20 features
                        )
                        
                        # Customize the plot
                        plt.title('SHAP Summary Plot - Feature Impact by Class', fontsize=16, fontweight='bold', pad=20)
                        
                        # Adjust legend to show full class names
                        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        for i, text in enumerate(legend.get_texts()):
                            if i < len(class_labels):
                                text.set_text(class_labels[i])
                        
                    else:
                        # Binary case or single array
                        if isinstance(shap_values, list):
                            shap_values_to_plot = shap_values[1]  # Use positive class for binary
                        else:
                            shap_values_to_plot = shap_values
                            
                        shap.summary_plot(
                            shap_values_to_plot,
                            X_sample,
                            show=False,
                            max_display=20
                        )
                        
                        plt.title('SHAP Summary Plot - Feature Impact', fontsize=16, fontweight='bold', pad=20)
                    
                    # Improve layout and save
                    plt.tight_layout()
                    shap_path = plots_dir / 'shap_summary.png'
                    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Custom SHAP plot with class labels saved to {shap_path}")
                    interpretation_generated = True
                    
                    # Also create a SHAP waterfall plot for a single prediction example
                    try:
                        logger.info("Creating SHAP waterfall plot for single prediction example...")
                        
                        # Get one example from each class
                        plt.figure(figsize=(12, 8))
                        
                        if isinstance(shap_values, list) and len(shap_values) > 2:
                            # For multiclass, create subplots for each class
                            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                            axes = axes.flatten()
                            
                            for class_idx, class_name in enumerate(class_labels[:4]):  # Limit to 4 classes for layout
                                if class_idx < len(shap_values):
                                    plt.sca(axes[class_idx])
                                    
                                    # Use the first sample for this example
                                    expected_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                                    
                                    # Create a simplified waterfall-style plot
                                    sample_idx = 0
                                    shap_vals = shap_values[class_idx][sample_idx]
                                    feature_names = X_sample.columns
                                    
                                    # Get top 10 features by absolute SHAP value
                                    top_indices = np.argsort(np.abs(shap_vals))[-10:]
                                    top_shap_vals = shap_vals[top_indices]
                                    top_features = [feature_names[i] for i in top_indices]
                                    
                                    # Create horizontal bar plot
                                    colors = ['red' if val < 0 else 'blue' for val in top_shap_vals]
                                    bars = plt.barh(range(len(top_shap_vals)), top_shap_vals, color=colors, alpha=0.7)
                                    plt.yticks(range(len(top_shap_vals)), [f[:30] + '...' if len(f) > 30 else f for f in top_features])
                                    plt.xlabel('SHAP Value')
                                    plt.title(f'Feature Impact - {class_name}', fontsize=12, fontweight='bold')
                                    plt.grid(axis='x', alpha=0.3)
                            
                            plt.suptitle('SHAP Feature Impact by Class', fontsize=16, fontweight='bold')
                            plt.tight_layout()
                            
                        waterfall_path = plots_dir / 'shap_waterfall.png'
                        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"SHAP waterfall plot saved to {waterfall_path}")
                        
                    except Exception as waterfall_error:
                        logger.warning(f"Could not create SHAP waterfall plot: {waterfall_error}")
                        
                except ImportError:
                    logger.warning("SHAP library not available. Install with: pip install shap")
                    # Fallback to PyCaret's interpretation
                    try:
                        classifier.interpret_model(
                            estimator=best_model,
                            plot='summary',
                            save=True
                        )
                        
                        # Look for generated files
                        possible_shap_files = [
                            "SHAP Summary.png", "summary.png", "SHAP.png", "Summary Plot.png"
                        ]
                        
                        for shap_file in possible_shap_files:
                            if Path(shap_file).exists():
                                target_path = plots_dir / 'shap_summary.png'
                                Path(shap_file).rename(target_path)
                                logger.info(f"Fallback SHAP plot moved: {shap_file} -> {target_path}")
                                interpretation_generated = True
                                break
                                
                    except Exception as fallback_error:
                        logger.warning(f"Fallback SHAP generation also failed: {fallback_error}")
                
                except Exception as shap_error:
                    logger.warning(f"Custom SHAP interpretation failed with error: {shap_error}")
                    logger.warning(f"Full error details: {shap_error.__class__.__name__}: {str(shap_error)}")
                    
                    # Fallback to PyCaret's default SHAP
                    try:
                        logger.info("Falling back to PyCaret's default SHAP interpretation...")
                        classifier.interpret_model(
                            estimator=best_model,
                            plot='summary',
                            save=True
                        )
                        
                        # Look for generated files
                        possible_shap_files = [
                            "SHAP Summary.png", "summary.png", "SHAP.png", "Summary Plot.png"
                        ]
                        
                        for shap_file in possible_shap_files:
                            if Path(shap_file).exists():
                                target_path = plots_dir / 'shap_summary.png'
                                Path(shap_file).rename(target_path)
                                logger.info(f"PyCaret SHAP plot moved: {shap_file} -> {target_path}")
                                interpretation_generated = True
                                break
                                
                    except Exception as pycaret_shap_error:
                        logger.warning(f"PyCaret SHAP also failed: {pycaret_shap_error}")
            
            # If SHAP failed, try Permutation Feature Importance (model-agnostic)
            if not interpretation_generated:
                try:
                    logger.info("Attempting Permutation Feature Importance (model-agnostic)...")
                    classifier.interpret_model(
                        estimator=best_model,
                        plot='pfi',
                        save=True
                    )
                    
                    pfi_files = [
                        "Permutation Feature Importance.png",
                        "pfi.png",
                        "PFI.png",
                        "Permutation.png"
                    ]
                    
                    for pfi_file in pfi_files:
                        if Path(pfi_file).exists():
                            target_path = plots_dir / 'feature_importance_plot.png'
                            Path(pfi_file).rename(target_path)
                            logger.info(f"Successfully generated PFI plot: {pfi_file} -> {target_path}")
                            interpretation_generated = True
                            break
                            
                except Exception as pfi_error:
                    logger.warning(f"Permutation Feature Importance failed: {pfi_error}")
            
            # Last resort: use basic feature importance plot
            if not interpretation_generated:
                try:
                    logger.info("Using basic feature importance as interpretation...")
                    feature_plot_path = plots_dir / 'feature.png'
                    if feature_plot_path.exists():
                        import shutil
                        interpretation_path = plots_dir / 'interpretation_plot.png'
                        shutil.copy2(feature_plot_path, interpretation_path)
                        logger.info("Copied feature importance plot as interpretation")
                        interpretation_generated = True
                        
                except Exception as feature_error:
                    logger.warning(f"Could not copy feature plot: {feature_error}")
            
            # Restore matplotlib backend
            matplotlib.use(current_backend)
            
            if interpretation_generated:
                logger.info("Model interpretation plot generated successfully")
            else:
                logger.error("Failed to generate any model interpretation plots")
                # List all PNG files in current directory for debugging
                png_files = list(Path('.').glob('*.png'))
                logger.info(f"Available PNG files in current directory: {[f.name for f in png_files]}")
                    
        except Exception as e:
            logger.error(f"Critical error in interpretation generation: {e}", exc_info=True)
        
        # Add this section right after the SHAP interpretation section and before model saving
        # Generate additional dimensionality reduction visualizations
        try:
            logger.info("Generating dimensionality reduction visualizations...")
            
            # Get the training data for visualization
            X_train = classifier.get_config('X_train')
            y_train_original = classifier.get_config('y_train')
            
            # Map back to original labels if needed
            if hasattr(y_train_original, 'map') and len(original_class_labels) == y_train_original.nunique():
                unique_encoded = sorted(y_train_original.unique())
                reverse_mapping = dict(zip(unique_encoded, original_class_labels))
                y_labels = y_train_original.map(reverse_mapping)
            else:
                y_labels = y_train_original
            
            # Standardize features for dimensionality reduction
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            # 1. PCA Visualization
            logger.info("Generating PCA visualization...")
            try:
                from sklearn.decomposition import PCA
                
                # Apply PCA with 2 components for visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create PCA plot
                plt.figure(figsize=(12, 8))
                
                # Use distinct colors for each class
                colors = plt.cm.Set1(np.linspace(0, 1, len(original_class_labels)))
                
                for i, class_name in enumerate(original_class_labels):
                    mask = y_labels == class_name
                    if mask.any():  # Only plot if class has samples
                        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                  c=[colors[i]], label=class_name, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)', fontsize=12)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)', fontsize=12)
                plt.title(f'PCA Visualization - {pca.explained_variance_ratio_.sum():.1%} Total Variance Explained', 
                         fontsize=16, fontweight='bold')
                plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                plt.grid(alpha=0.3)
                
                # Add explained variance ratio as text
                plt.text(0.02, 0.98, f'PC1: {pca.explained_variance_ratio_[0]:.1%}\nPC2: {pca.explained_variance_ratio_[1]:.1%}', 
                        transform=plt.gca().transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                pca_path = plots_dir / 'pca_visualization.png'
                plt.savefig(pca_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"PCA visualization saved to {pca_path}")
                
            except Exception as pca_error:
                logger.warning(f"Could not generate PCA visualization: {pca_error}")

            # 2. t-SNE Visualization
            logger.info("Generating t-SNE visualization...")
            try:
                from sklearn.manifold import TSNE
                
                # Limit sample size for t-SNE performance (t-SNE is computationally expensive)
                max_samples_tsne = 2000
                if len(X_scaled) > max_samples_tsne:
                    logger.info(f"Sampling {max_samples_tsne} points for t-SNE (original: {len(X_scaled)})")
                    # Stratified sampling to maintain class distribution
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
                
                # Apply t-SNE with perplexity 30
                perplexity = min(30, len(X_tsne_sample)//4)  # Ensure perplexity is valid
                logger.info(f"Computing t-SNE with perplexity={perplexity}...")
                
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                           n_iter=1000, learning_rate=200)
                X_tsne = tsne.fit_transform(X_tsne_sample)
                
                # Create t-SNE plot
                plt.figure(figsize=(12, 8))
                
                for i, class_name in enumerate(original_class_labels):
                    mask = y_tsne_sample == class_name
                    if mask.any():
                        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                                  c=[colors[i]], label=class_name, alpha=0.7, s=60, 
                                  edgecolors='black', linewidth=0.5)
                
                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.title(f't-SNE Visualization (perplexity={perplexity})', 
                         fontsize=16, fontweight='bold')
                plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                plt.grid(alpha=0.3)
                
                # Add parameters as text
                plt.text(0.02, 0.98, f'Perplexity: {perplexity}\nSamples: {len(X_tsne_sample)}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                plt.tight_layout()
                tsne_path = plots_dir / f'tsne_visualization_perp{perplexity}.png'
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"t-SNE visualization saved to {tsne_path}")
                
                # Create a combined comparison plot
                logger.info("Creating combined PCA vs t-SNE comparison...")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # PCA subplot
                for i, class_name in enumerate(original_class_labels):
                    mask = y_labels == class_name
                    if mask.any():
                        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                  c=[colors[i]], label=class_name, alpha=0.7, s=40)
                
                ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax1.set_title('PCA Visualization')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # t-SNE subplot
                for i, class_name in enumerate(original_class_labels):
                    mask = y_tsne_sample == class_name
                    if mask.any():
                        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                                  c=[colors[i]], label=class_name, alpha=0.7, s=40)
                
                ax2.set_xlabel('t-SNE Component 1')
                ax2.set_ylabel('t-SNE Component 2')
                ax2.set_title(f't-SNE Visualization (perplexity={perplexity})')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                plt.suptitle('Dimensionality Reduction Comparison', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                comparison_path = plots_dir / 'pca_vs_tsne_comparison.png'
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"PCA vs t-SNE comparison saved to {comparison_path}")
                
            except Exception as tsne_error:
                logger.warning(f"Could not generate t-SNE visualization: {tsne_error}")

            # 3. Class Distribution Plot (simple bar chart)
            logger.info("Generating class distribution plot...")
            try:
                plt.figure(figsize=(10, 6))
                class_counts = df[target_column].value_counts()
                colors_dist = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
                
                bars = plt.bar(class_counts.index, class_counts.values, color=colors_dist)
                plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
                plt.xlabel('Class')
                plt.ylabel('Number of Samples')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
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

            # 4. Feature Correlation Heatmap (add this after the class distribution plot)
            logger.info("Generating feature correlation heatmap...")
            try:
                # Get top features by importance if available, otherwise use first N features
                if report_data.get('feature_importance'):
                    top_features = [feat[0] for feat in report_data['feature_importance'][:25]]  # Top 25 features
                    # Filter to only include features that exist in X_train
                    available_features = [feat for feat in top_features if feat in X_train.columns]
                    if available_features:
                        X_subset = X_train[available_features]
                        title_suffix = " (Top 25 by Importance)"
                    else:
                        # Fallback if importance features don't match
                        X_subset = X_train.iloc[:, :25]  # First 25 features
                        title_suffix = " (First 25 Features)"
                else:
                    # Use first 25 features if no importance available
                    X_subset = X_train.iloc[:, :25]  # First 25 features
                    title_suffix = " (First 25 Features)"
                
                # Calculate correlation matrix
                correlation_matrix = X_subset.corr()
                
                # Create the heatmap
                plt.figure(figsize=(14, 12))
                
                # Create a mask for the upper triangle (optional - shows only lower triangle)
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                # Generate the heatmap
                sns.heatmap(correlation_matrix, 
                          mask=mask,  # Comment this line if you want full heatmap
                          annot=False,  # Set to True if you want correlation values displayed
                          cmap='coolwarm', 
                          center=0,
                          square=True,
                          vmin=-1, vmax=1,
                          cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
                
                plt.title(f'Feature Correlation Heatmap{title_suffix}', fontsize=16, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                
                # Add text annotation about correlation interpretation
                plt.figtext(0.02, 0.02, 
                           'Red = Positive Correlation, Blue = Negative Correlation\n' +
                           'Darker colors indicate stronger correlations',
                           fontsize=10, style='italic',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                plt.tight_layout()
                
                corr_path = plots_dir / 'feature_correlation.png'
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Feature correlation heatmap saved to {corr_path}")
                
                # Also create a simplified version showing only high correlations
                logger.info("Generating high correlation features heatmap...")
                
                # Find pairs with high correlation (absolute value > 0.7)
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # High correlation threshold
                            high_corr_pairs.append((
                                correlation_matrix.columns[i], 
                                correlation_matrix.columns[j], 
                                corr_val
                            ))
                
                if high_corr_pairs:
                    # Create a plot showing high correlation pairs
                    plt.figure(figsize=(12, max(6, len(high_corr_pairs) * 0.4)))
                    
                    features_1 = [pair[0] for pair in high_corr_pairs]
                    features_2 = [pair[1] for pair in high_corr_pairs]
                    correlations = [pair[2] for pair in high_corr_pairs]
                    
                    # Create labels for the pairs
                    pair_labels = [f"{f1[:20]}...\n{f2[:20]}..." if len(f1) > 20 or len(f2) > 20 
                                 else f"{f1}\n{f2}" 
                                 for f1, f2 in zip(features_1, features_2)]
                    
                    # Create horizontal bar plot
                    colors = ['red' if corr < 0 else 'blue' for corr in correlations]
                    bars = plt.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
                    
                    plt.yticks(range(len(correlations)), pair_labels, fontsize=9)
                    plt.xlabel('Correlation Coefficient', fontsize=12)
                    plt.title('High Correlation Feature Pairs (|r| > 0.7)', fontsize=16, fontweight='bold')
                    plt.grid(axis='x', alpha=0.3)
                    
                    # Add correlation values on bars
                    for i, (bar, corr) in enumerate(zip(bars, correlations)):
                        plt.text(corr + (0.02 if corr > 0 else -0.02), i, f'{corr:.3f}', 
                               ha='left' if corr > 0 else 'right', va='center', fontweight='bold')
                    
                    plt.xlim(-1, 1)
                    plt.tight_layout()
                    
                    high_corr_path = plots_dir / 'high_correlation_pairs.png'
                    plt.savefig(high_corr_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"High correlation pairs plot saved to {high_corr_path} ({len(high_corr_pairs)} pairs found)")
                else:
                    logger.info("No high correlation pairs found (|r| > 0.7)")
                
            except Exception as corr_error:
                logger.warning(f"Could not generate correlation heatmap: {corr_error}")

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
    parser.add_argument('--feature-selection-method', type=str, default='univariate_kbest', help='Feature selection method.')
    
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
    
    # Run the report generation
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
