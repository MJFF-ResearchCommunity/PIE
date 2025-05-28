import os
import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.classifier import Classifier
from pie.feature_selector import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.test_classifier")

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
                    <tr><td>Feature Selection Applied</td><td>{'Yes' if report_data.get('feature_selection_applied', False) else 'No'}</td></tr>
                    <tr><td>Final Feature Count</td><td><span class="metric-value">{report_data.get('n_features', 'N/A')}</span></td></tr>
                </table>
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

    # Best model details
    if report_data.get('best_model_name'):
        html_content += f"""
            <div class="best-model">
                <h2>🌟 4. Best Model Details</h2>
                <h3>Selected Model: <span class="highlight">{report_data.get('best_model_name', 'N/A')}</span></h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in report_data.get('best_model_metrics', {}).items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric}</td><td class='metric-value'>{value:.4f}</td></tr>"
        html_content += """
                </table>
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

    # Model plots
    html_content += """
            <div class="summary-box">
                <h2>📈 6. Model Performance Visualizations</h2>
    """
    
    # Add each plot
    plot_descriptions = {
        'auc': 'ROC Curve - Shows the trade-off between true positive rate and false positive rate',
        'confusion_matrix': 'Confusion Matrix - Shows the classification results in detail',
        'pr': 'Precision-Recall Curve - Important for imbalanced datasets',
        'feature': 'Feature Importance - Shows which features contribute most to predictions',
        'learning': 'Learning Curve - Shows if the model is overfitting or underfitting',
        'calibration': 'Calibration Plot - Shows how well the predicted probabilities are calibrated',
        'boundary': 'Decision Boundary - Visualizes the classification boundaries (if applicable)',
        'error': 'Prediction Error Plot - Shows the distribution of prediction errors'
    }
    
    for plot_name in ['auc', 'confusion_matrix', 'pr', 'feature', 'learning', 'calibration']:
        plot_path = Path(plots_dir) / f"{plot_name}.png"
        if plot_path.exists():
            html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_name.upper().replace('_', ' ')}</div>
                    <p><em>{plot_descriptions.get(plot_name, '')}</em></p>
                    <img src="{plot_path.name}" alt="{plot_name} plot">
                </div>
            """
    
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

    # Model interpretation
    html_content += """
            <div class="summary-box">
                <h2>🔬 8. Model Interpretation</h2>
                <p>SHAP (SHapley Additive exPlanations) analysis provides insights into how the model makes predictions:</p>
    """
    
    # Add SHAP plots if they exist
    for shap_plot in ['shap_summary', 'shap_waterfall', 'shap_dependence']:
        plot_path = Path(plots_dir) / f"{shap_plot}.png"
        if plot_path.exists():
            html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{shap_plot.replace('_', ' ').upper()}</div>
                    <img src="{plot_path.name}" alt="{shap_plot}">
                </div>
            """
    
    html_content += """
            </div>
    """

    # Test set performance
    if report_data.get('test_metrics'):
        html_content += """
            <div class="summary-box">
                <h2>✅ 9. Final Test Set Performance</h2>
                <p>Performance metrics on the held-out test set:</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in report_data.get('test_metrics', {}).items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric}</td><td class='metric-value'>{value:.4f}</td></tr>"
        html_content += """
                </table>
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


def test_classification_pipeline(
    input_csv_path: str = None,
    train_csv_path: str = None,
    test_csv_path: str = None,
    use_feature_selection: bool = True,
    feature_selection_method: str = 'univariate_kbest',
    target_column: str = "COHORT",
    output_dir: str = "output",
    n_models_to_compare: int = 5,
    tune_best_model: bool = True,
    generate_plots: bool = True,
    budget_time_minutes: float = 30.0  # Add budget time parameter
):
    """
    Tests the complete classification pipeline including optional feature selection,
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
    logger.info("Starting classification pipeline test...")
    
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
    
    # Handle missing target values
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing target values")
    
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
        exclude_models.extend(['svm', 'rbfsvm', 'gpc'])
    
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
    
    # Get final model metrics
    try:
        # Get predictions on test set
        final_predictions = classifier.predict_model(estimator=best_model, verbose=False)
        if not final_predictions.empty:
            # Extract metrics from the predictions
            test_metrics = {}
            for col in final_predictions.columns:
                if col not in ['Label', 'Score', target_column] and pd.api.types.is_numeric_dtype(final_predictions[col]):
                    # These are likely metric columns
                    test_metrics[col] = final_predictions[col].iloc[0] if len(final_predictions) > 0 else final_predictions[col].mean()
            
            report_data['best_model_metrics'] = test_metrics
            report_data['test_metrics'] = test_metrics
            
            logger.info(f"Test set metrics: {test_metrics}")
    except Exception as e:
        logger.warning(f"Failed to get model metrics: {e}")
    
    # Generate plots
    if generate_plots:
        logger.info("Generating model visualizations...")
        
        plot_types = ['auc', 'confusion_matrix', 'pr', 'feature', 'learning', 'calibration']
        
        for plot_type in plot_types:
            try:
                logger.info(f"Generating {plot_type} plot...")
                plot_path = plots_dir / f"{plot_type}.png"
                
                # Use matplotlib backend to save plots
                import matplotlib
                matplotlib.use('Agg')
                
                # Generate and save plot
                classifier.plot_model(
                    estimator=best_model,
                    plot=plot_type,
                    save=True,
                    verbose=False
                )
                
                # Move the saved plot to our directory
                default_plot_name = f"{plot_type}.png"
                if Path(default_plot_name).exists():
                    Path(default_plot_name).rename(plot_path)
                
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
        
        # Generate SHAP plots for interpretation
        try:
            logger.info("Generating model interpretation plots...")
            
            # Save current plot
            plt.figure(figsize=(10, 6))
            classifier.interpret_model(
                estimator=best_model,
                plot='summary',
                save=True
            )
            plt.savefig(plots_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate interpretation plots: {e}")
    
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
    
    logger.info("Classification pipeline test completed successfully!")
    
    return classifier, best_model, report_data


if __name__ == "__main__":
    # Run the classification pipeline test with pre-split data
    test_classification_pipeline(
        train_csv_path="output/selected_train_data.csv",
        test_csv_path="output/selected_test_data.csv",
        use_feature_selection=False,  # Data is already feature-selected
        target_column="COHORT",
        tune_best_model=True,
        generate_plots=True,
        budget_time_minutes=30.0  # Set 15 minute time limit for model comparison
    )
