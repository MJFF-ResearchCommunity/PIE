"""
classifier.py

Methods for training, evaluating, and comparing classification models using PyCaret.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
import warnings
import json

# PyCaret imports
try:
    from pycaret.classification import (
        setup, create_model, compare_models, tune_model, 
        ensemble_model, blend_models, stack_models,
        plot_model, evaluate_model, interpret_model,
        predict_model, finalize_model, save_model, load_model,
        pull, get_metrics, get_logs, models, get_config
    )
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    warnings.warn("PyCaret is not installed. Please install it using: pip install pycaret")

logger = logging.getLogger(f"PIE.{__name__}")

class Classifier:
    """
    Provides classification model training, evaluation, and comparison using PyCaret.
    """
    
    def __init__(self):
        """Initialize the Classifier."""
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret is required for the Classifier module. Please install it using: pip install pycaret")
        
        self.experiment = None
        self.best_model = None
        self.tuned_model = None
        self.models_dict = {}
        self.comparison_results = None
        self.setup_params = None
        
    def setup_experiment(
        self,
        data: pd.DataFrame,
        target: str,
        train_size: float = 0.8,
        test_data: Optional[pd.DataFrame] = None,
        session_id: int = 123,
        use_gpu: bool = False,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        remove_outliers: bool = False,
        outliers_threshold: float = 0.05,
        normalize: bool = False,
        transformation: bool = False,
        pca: bool = False,
        pca_components: Optional[Union[int, float]] = None,
        ignore_features: Optional[List[str]] = None,
        feature_selection: bool = False,
        feature_selection_method: str = 'classic',
        feature_selection_estimator: Optional[Any] = None,
        n_features_to_select: Union[int, float] = 0.2,
        fold_strategy: str = 'stratifiedkfold',
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Setup the PyCaret classification experiment.
        
        :param data: Input dataframe
        :param target: Name of target column
        :param train_size: Proportion of dataset to use for training
        :param test_data: Optional test dataset
        :param session_id: Random seed for reproducibility
        :param use_gpu: Whether to use GPU for training
        :param log_experiment: Whether to log experiment to MLflow
        :param experiment_name: Name for the experiment
        :param verbose: Whether to show output
        :param remove_multicollinearity: Whether to remove multicollinear features
        :param multicollinearity_threshold: Threshold for multicollinearity removal
        :param remove_outliers: Whether to remove outliers
        :param outliers_threshold: Threshold for outlier removal
        :param normalize: Whether to normalize features
        :param transformation: Whether to apply transformations
        :param pca: Whether to apply PCA
        :param pca_components: Number of PCA components
        :param ignore_features: List of features to ignore
        :param feature_selection: Whether to perform feature selection
        :param feature_selection_method: Method for feature selection
        :param feature_selection_estimator: Estimator for feature selection
        :param n_features_to_select: Number or fraction of features to select
        :param fold_strategy: Cross-validation strategy
        :param fold: Number of CV folds
        :param fold_shuffle: Whether to shuffle folds
        :param fold_groups: Column name for group labels
        :param kwargs: Additional arguments for PyCaret setup
        :return: PyCaret experiment object
        """
        logger.info("Setting up PyCaret classification experiment...")
        
        # Build setup parameters with only valid PyCaret parameters
        self.setup_params = {
            'data': data,
            'target': target,
            'session_id': session_id,
            'use_gpu': use_gpu,
            'log_experiment': log_experiment,
            'experiment_name': experiment_name,
            'verbose': verbose,
            'remove_multicollinearity': remove_multicollinearity,
            'multicollinearity_threshold': multicollinearity_threshold,
            'remove_outliers': remove_outliers,
            'outliers_threshold': outliers_threshold,
            'normalize': normalize,
            'transformation': transformation,
            'pca': pca,
            'pca_components': pca_components,
            'ignore_features': ignore_features,
            'feature_selection': feature_selection,
            'feature_selection_method': feature_selection_method,
            'n_features_to_select': n_features_to_select,
            'fold_strategy': fold_strategy,
            'fold': fold,
            'fold_shuffle': fold_shuffle,
        }
        
        # Add optional parameters only if they're not None
        if test_data is not None:
            self.setup_params['test_data'] = test_data
        else:
            self.setup_params['train_size'] = train_size
            
        if feature_selection_estimator is not None:
            self.setup_params['feature_selection_estimator'] = feature_selection_estimator
            
        if fold_groups is not None:
            self.setup_params['fold_groups'] = fold_groups
            
        # Add any additional kwargs that might be valid
        self.setup_params.update(kwargs)
        
        try:
            self.experiment = setup(**self.setup_params)
            logger.info("PyCaret experiment setup completed successfully.")
            return self.experiment
        except Exception as e:
            logger.error(f"Failed to setup PyCaret experiment: {e}", exc_info=True)
            raise
    
    def compare_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[int] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = 'Accuracy',
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = 'ignore',
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, str]] = None,
        parallel: Optional[Any] = None
    ) -> Union[Any, List[Any]]:
        """
        Compare multiple classification models and return the best one(s).
        
        :param include: List of model IDs to include
        :param exclude: List of model IDs to exclude
        :param fold: Number of CV folds
        :param round: Number of decimal places to round metrics
        :param cross_validation: Whether to use cross-validation
        :param sort: Metric to sort by
        :param n_select: Number of top models to return
        :param budget_time: Time budget in minutes
        :param turbo: Whether to use turbo mode
        :param errors: How to handle errors
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        :param verbose: Whether to print results
        :param probability_threshold: Threshold for binary classification
        :param experiment_custom_tags: Custom tags for MLflow
        :param engine: Engine to use for specific models
        :param parallel: Parallel backend configuration
        :return: Best model(s)
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        logger.info(f"Comparing models, sorting by {sort}...")
        
        # Get list of models to compare
        all_models = models()
        
        if include:
            models_to_compare = include
        else:
            models_to_compare = all_models.index.tolist()
            if exclude:
                models_to_compare = [m for m in models_to_compare if m not in exclude]
        
        # Print models that will be compared
        logger.info(f"Models to compare: {models_to_compare}")
        
        # Create a custom progress callback if verbose
        if verbose:
            print("\n" + "="*60)
            print("COMPARING MODELS")
            print("="*60)
            print(f"Total models to compare: {len(models_to_compare)}")
            print(f"Cross-validation folds: {fold if fold else 'default'}")
            print(f"Optimization metric: {sort}")
            if budget_time:
                print(f"Time budget: {budget_time} minutes")
            if exclude:
                print(f"Excluded models: {exclude}")
            print("="*60 + "\n")
            
            # Add timestamp tracking
            import time
            model_start_times = {}
            
            # Store original print function
            import builtins
            original_print = builtins.print
            
            def custom_print(*args, **kwargs):
                """Custom print function to add model names and timing"""
                output = ' '.join(str(arg) for arg in args)
                
                # Check if this is a model evaluation line
                for model_id in models_to_compare:
                    if model_id in output and 'Fitting' in output:
                        model_name = all_models.loc[model_id, 'Name'] if model_id in all_models.index else model_id
                        model_start_times[model_id] = time.time()
                        original_print(f"\n>>> Starting evaluation of: {model_name} ({model_id}) at {time.strftime('%H:%M:%S')}", **kwargs)
                        break
                    elif model_id in output and any(metric in output for metric in ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']):
                        # Model completed
                        if model_id in model_start_times:
                            elapsed = time.time() - model_start_times[model_id]
                            original_print(f"    Completed in {elapsed:.1f} seconds", **kwargs)
                
                # Print the original output
                original_print(*args, **kwargs)
            
            # Replace print function temporarily
            builtins.print = custom_print
        
        try:
            # Build parameters for compare_models
            compare_params = {
                'include': include,
                'exclude': exclude,
                'fold': fold,
                'round': round,
                'cross_validation': cross_validation,
                'sort': sort,
                'n_select': n_select,
                'budget_time': budget_time,
                'turbo': turbo,
                'errors': errors,
                'fit_kwargs': fit_kwargs or {},
                'groups': groups,
                'verbose': verbose
            }
            
            # Add optional parameters if provided
            if probability_threshold is not None:
                compare_params['probability_threshold'] = probability_threshold
            if experiment_custom_tags is not None:
                compare_params['experiment_custom_tags'] = experiment_custom_tags
            if engine is not None:
                compare_params['engine'] = engine
            if parallel is not None:
                compare_params['parallel'] = parallel
            
            result = compare_models(**compare_params)
            
            # Store comparison results
            self.comparison_results = pull()
            
            if n_select == 1:
                self.best_model = result
                model_name = type(result).__name__
                logger.info(f"Best model selected: {model_name}")
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"BEST MODEL: {model_name}")
                    print(f"{'='*60}\n")
            else:
                self.best_model = result[0] if result else None
                logger.info(f"Top {n_select} models selected.")
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"TOP {n_select} MODELS SELECTED")
                    for i, model in enumerate(result):
                        print(f"{i+1}. {type(model).__name__}")
                    print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}", exc_info=True)
            raise
    
    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[int] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ) -> Any:
        """
        Create and train a specific model.
        
        :param estimator: Model ID or estimator object
        :param fold: Number of CV folds
        :param round: Number of decimal places to round metrics
        :param cross_validation: Whether to use cross-validation
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        :param verbose: Whether to print results
        :param kwargs: Additional model-specific parameters
        :return: Trained model
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        logger.info(f"Creating model: {estimator}")
        
        try:
            model = create_model(
                estimator=estimator,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs or {},
                groups=groups,
                verbose=verbose,
                **kwargs
            )
            
            # Store the model
            model_name = estimator if isinstance(estimator, str) else type(estimator).__name__
            self.models_dict[model_name] = model
            
            logger.info(f"Model {model_name} created successfully.")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}", exc_info=True)
            raise
    
    def tune_model(
        self,
        estimator: Optional[Any] = None,
        fold: Optional[int] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[dict] = None,
        optimize: str = 'Accuracy',
        choose_better: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        tuner_verbose: Union[bool, int] = True,
        return_tuner: bool = False,
        **kwargs
    ) -> Any:
        """
        Tune hyperparameters of a model.
        
        :param estimator: Model to tune (if None, uses best_model)
        :param fold: Number of CV folds
        :param round: Number of decimal places to round metrics
        :param n_iter: Number of iterations for random search
        :param custom_grid: Custom hyperparameter grid
        :param optimize: Metric to optimize
        :param choose_better: Whether to return better model only
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        :param verbose: Whether to print results
        :param tuner_verbose: Verbosity of tuner
        :param return_tuner: Whether to return tuner object
        :param kwargs: Additional tuning parameters
        :return: Tuned model
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.best_model
            if estimator is None:
                raise ValueError("No model to tune. Please create or compare models first.")
        
        logger.info(f"Tuning model: {type(estimator).__name__}")
        
        try:
            tuned = tune_model(
                estimator=estimator,
                fold=fold,
                round=round,
                n_iter=n_iter,
                custom_grid=custom_grid,
                optimize=optimize,
                choose_better=choose_better,
                fit_kwargs=fit_kwargs or {},
                groups=groups,
                verbose=verbose,
                tuner_verbose=tuner_verbose,
                return_tuner=return_tuner,
                **kwargs
            )
            
            self.tuned_model = tuned
            logger.info("Model tuning completed successfully.")
            return tuned
            
        except Exception as e:
            logger.error(f"Failed to tune model: {e}", exc_info=True)
            raise
    
    def ensemble_model(
        self,
        estimator: Optional[Any] = None,
        method: str = 'Bagging',
        fold: Optional[int] = None,
        n_estimators: int = 10,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = 'Accuracy',
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True
    ) -> Any:
        """
        Create an ensemble model.
        
        :param estimator: Base estimator for ensemble
        :param method: Ensemble method ('Bagging' or 'Boosting')
        :param fold: Number of CV folds
        :param n_estimators: Number of estimators in ensemble
        :param round: Number of decimal places to round metrics
        :param choose_better: Whether to return better model only
        :param optimize: Metric to optimize
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        :param verbose: Whether to print results
        :return: Ensemble model
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.best_model or self.tuned_model
            if estimator is None:
                raise ValueError("No model to ensemble. Please create a model first.")
        
        logger.info(f"Creating {method} ensemble for {type(estimator).__name__}")
        
        try:
            ensemble = ensemble_model(
                estimator=estimator,
                method=method,
                fold=fold,
                n_estimators=n_estimators,
                round=round,
                choose_better=choose_better,
                optimize=optimize,
                fit_kwargs=fit_kwargs or {},
                groups=groups,
                verbose=verbose
            )
            
            logger.info("Ensemble model created successfully.")
            return ensemble
            
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {e}", exc_info=True)
            raise
    
    def save_model(
        self,
        model: Optional[Any] = None,
        model_name: str = 'pie_classifier_model',
        model_only: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[Any, str]:
        """
        Save a trained model.
        
        :param model: Model to save (if None, uses best or tuned model)
        :param model_name: Name for the saved model
        :param model_only: Whether to save only the model (not the entire pipeline)
        :param verbose: Whether to print success message
        :param kwargs: Additional arguments for joblib.dump
        :return: Tuple of (model, filename)
        """
        if model is None:
            model = self.tuned_model or self.best_model
            if model is None:
                raise ValueError("No model to save. Please train a model first.")
        
        logger.info(f"Saving model as {model_name}...")
        
        try:
            result = save_model(
                model=model,
                model_name=model_name,
                model_only=model_only,
                verbose=verbose,
                **kwargs
            )
            
            logger.info(f"Model saved successfully as {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise
    
    def load_model(
        self,
        model_name: str,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ) -> Any:
        """
        Load a previously saved model.
        
        :param model_name: Name of the model to load
        :param platform: Cloud platform ('aws', 'gcp', 'azure')
        :param authentication: Authentication credentials for cloud platform
        :param verbose: Whether to print success message
        :return: Loaded model
        """
        logger.info(f"Loading model: {model_name}")
        
        try:
            model = load_model(
                model_name=model_name,
                platform=platform,
                authentication=authentication,
                verbose=verbose
            )
            
            logger.info("Model loaded successfully.")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def predict_model(
        self,
        estimator: Optional[Any] = None,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions using a trained model.
        
        :param estimator: Model to use for predictions
        :param data: Data to predict on (if None, uses test set)
        :param probability_threshold: Threshold for binary classification
        :param encoded_labels: Whether to return encoded labels
        :param raw_score: Whether to return raw scores
        :param round: Number of decimal places to round probabilities
        :param verbose: Whether to print results
        :return: DataFrame with predictions
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.tuned_model or self.best_model
            if estimator is None:
                raise ValueError("No model to use for predictions. Please train a model first.")
        
        logger.info("Making predictions...")
        
        try:
            predictions = predict_model(
                estimator=estimator,
                data=data,
                probability_threshold=probability_threshold,
                encoded_labels=encoded_labels,
                raw_score=raw_score,
                round=round,
                verbose=verbose
            )
            
            logger.info("Predictions completed successfully.")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}", exc_info=True)
            raise
    
    def plot_model(
        self,
        estimator: Optional[Any] = None,
        plot: str = 'auc',
        scale: float = 1,
        save: bool = False,
        fold: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        display_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot model performance visualization.
        
        :param estimator: Model to plot
        :param plot: Type of plot (e.g., 'auc', 'confusion_matrix', 'pr', 'feature')
        :param scale: Scale factor for plot size
        :param save: Whether to save the plot
        :param fold: Fold number to plot
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        :param verbose: Whether to show the plot
        :param display_format: Format for display ('png' or 'svg')
        :return: Path to saved plot if save=True
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.tuned_model or self.best_model
            if estimator is None:
                raise ValueError("No model to plot. Please train a model first.")
        
        logger.info(f"Plotting {plot} for {type(estimator).__name__}")
        
        try:
            result = plot_model(
                estimator=estimator,
                plot=plot,
                scale=scale,
                save=save,
                fold=fold,
                fit_kwargs=fit_kwargs or {},
                groups=groups,
                verbose=verbose,
                display_format=display_format
            )
            
            if save:
                logger.info(f"Plot saved successfully.")
            return result
            
        except Exception as e:
            logger.error(f"Failed to plot model: {e}", exc_info=True)
            raise
    
    def evaluate_model(
        self,
        estimator: Optional[Any] = None,
        fold: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None
    ):
        """
        Evaluate model using interactive plots.
        
        :param estimator: Model to evaluate
        :param fold: Number of CV folds
        :param fit_kwargs: Additional arguments for model fitting
        :param groups: Column name for group labels
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.tuned_model or self.best_model
            if estimator is None:
                raise ValueError("No model to evaluate. Please train a model first.")
        
        logger.info(f"Evaluating {type(estimator).__name__}")
        
        try:
            evaluate_model(
                estimator=estimator,
                fold=fold,
                fit_kwargs=fit_kwargs or {},
                groups=groups
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}", exc_info=True)
            raise
    
    def interpret_model(
        self,
        estimator: Optional[Any] = None,
        plot: str = 'summary',
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        **kwargs
    ):
        """
        Interpret model using SHAP.
        
        :param estimator: Model to interpret
        :param plot: Type of interpretation plot
        :param feature: Feature name for dependence plot
        :param observation: Row index for local interpretation
        :param use_train_data: Whether to use training data
        :param kwargs: Additional arguments for interpretation
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.tuned_model or self.best_model
            if estimator is None:
                raise ValueError("No model to interpret. Please train a model first.")
        
        logger.info(f"Interpreting {type(estimator).__name__} with {plot} plot")
        
        try:
            interpret_model(
                estimator=estimator,
                plot=plot,
                feature=feature,
                observation=observation,
                use_train_data=use_train_data,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Failed to interpret model: {e}", exc_info=True)
            raise
    
    def get_metrics(self) -> pd.DataFrame:
        """
        Get available metrics for the experiment.
        
        :return: DataFrame with available metrics
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        return get_metrics()
    
    def get_logs(
        self,
        experiment_name: Optional[str] = None,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Get experiment logs.
        
        :param experiment_name: Name of experiment to get logs for
        :param save: Whether to save logs to file
        :return: DataFrame with experiment logs
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        return get_logs(experiment_name=experiment_name, save=save)
    
    def get_config(self, variable: Optional[str] = None) -> Any:
        """
        Get experiment configuration.
        
        :param variable: Specific variable to get (if None, returns all)
        :return: Configuration value(s)
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        return get_config(variable=variable)
    
    def generate_report(
        self,
        output_path: str = "classification_report.html",
        include_plots: List[str] = None,
        estimator: Optional[Any] = None
    ) -> str:
        """
        Generate a comprehensive HTML report for the classification experiment.
        
        :param output_path: Path to save the HTML report
        :param include_plots: List of plots to include in report
        :param estimator: Model to report on (if None, uses best/tuned model)
        :return: Path to the generated report
        """
        if self.experiment is None:
            raise ValueError("Experiment not set up. Please run setup_experiment first.")
        
        if estimator is None:
            estimator = self.tuned_model or self.best_model
            if estimator is None:
                raise ValueError("No model to report on. Please train a model first.")
        
        if include_plots is None:
            include_plots = ['auc', 'confusion_matrix', 'pr', 'feature', 'learning']
        
        logger.info(f"Generating classification report at {output_path}")
        
        # Collect report data
        report_data = {
            'model_name': type(estimator).__name__,
            'setup_params': self.setup_params,
            'comparison_results': self.comparison_results,
            'metrics': get_metrics().to_dict() if self.experiment else {},
            'logs': self.get_logs().to_dict('records') if self.experiment else []
        }
        
        # Generate HTML
        html_content = self._generate_html_report(report_data, estimator, include_plots)
        
        # Save report
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Report generated successfully at {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save report: {e}", exc_info=True)
            raise
    
    def _generate_html_report(self, report_data: dict, estimator: Any, include_plots: List[str]) -> str:
        """Generate HTML content for the report."""
        html_style = """
        <style>
            body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
            .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }
            h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }
            h1 { text-align: center; color: #e74c3c; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #e74c3c; color: white; }
            tr:nth-child(even) { background-color: #ecf0f1; }
            .summary-box { border: 1px solid #bdc3c7; padding: 15px; margin-bottom: 20px; background-color: #f8f9f9; border-radius: 5px; }
            .code { background-color: #e8e8e8; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', Courier, monospace; }
            .highlight { color: #e74c3c; font-weight: bold; }
            .metric-value { font-weight: bold; color: #27ae60; }
            .plot-container { margin: 20px 0; text-align: center; }
            .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        </style>
        """
        
        # Get test predictions for metrics
        try:
            test_predictions = self.predict_model(estimator=estimator, verbose=False)
            test_score = test_predictions.iloc[0] if not test_predictions.empty else {}
        except:
            test_score = {}
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>PIE Classification Report</title>
            {html_style}
        </head>
        <body>
            <div class="container">
                <h1>PIE Classification Report</h1>
                
                <div class="summary-box">
                    <h2>1. Model Summary</h2>
                    <p><strong>Best Model:</strong> <span class="highlight">{report_data['model_name']}</span></p>
                    <p><strong>Target Variable:</strong> <span class="code">{report_data['setup_params'].get('target', 'N/A')}</span></p>
                    <p><strong>Training Size:</strong> {report_data['setup_params'].get('train_size', 0.8) * 100:.0f}%</p>
                    <p><strong>Session ID:</strong> {report_data['setup_params'].get('session_id', 'N/A')}</p>
                </div>
                
                <div class="summary-box">
                    <h2>2. Model Performance</h2>
        """
        
        # Add comparison results if available
        if report_data['comparison_results'] is not None and not report_data['comparison_results'].empty:
            html_content += """
                    <h3>Model Comparison Results</h3>
                    <div style="overflow-x: auto;">
            """
            html_content += report_data['comparison_results'].to_html(classes='comparison-table', index=True)
            html_content += """
                    </div>
            """
        
        # Add test set performance if available
        if test_score:
            html_content += """
                    <h3>Test Set Performance</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            for metric, value in test_score.items():
                if metric not in ['Label', 'Score'] and isinstance(value, (int, float)):
                    html_content += f"<tr><td>{metric}</td><td class='metric-value'>{value:.4f}</td></tr>"
            html_content += """
                    </table>
            """
        
        html_content += """
                </div>
                
                <div class="summary-box">
                    <h2>3. Experiment Configuration</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        # Add key setup parameters
        key_params = ['remove_multicollinearity', 'multicollinearity_threshold', 
                      'remove_outliers', 'outliers_threshold', 'normalize', 
                      'transformation', 'pca', 'pca_components']
        
        for param in key_params:
            if param in report_data['setup_params']:
                value = report_data['setup_params'][param]
                html_content += f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html_content += """
                    </table>
                </div>
                
                <div class="summary-box">
                    <h2>4. Available Metrics</h2>
        """
        
        if report_data['metrics']:
            html_content += pd.DataFrame(report_data['metrics']).to_html(classes='metrics-table', index=False)
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def get_available_models(self) -> pd.DataFrame:
        """
        Get list of available models in PyCaret.
        
        :return: DataFrame with available models
        """
        return models()
