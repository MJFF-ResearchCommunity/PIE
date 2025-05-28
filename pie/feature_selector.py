"""
feature_selector.py

Methods for selecting the most relevant features from the dataset.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional, Any, Callable, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    SelectFromModel,
    RFE,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # As potential default estimators

logger = logging.getLogger(f"PIE.{__name__}")

class FeatureSelector:
    """
    Provides feature selection techniques using scikit-learn.
    """

    @staticmethod
    def select_features(
        data: pd.DataFrame,
        target_column: str,
        task_type: str,
        method: str,
        # General parameters for feature count/selection criteria
        k_or_frac_kbest: Optional[Union[int, float]] = None, # For univariate_kbest: int for k, float for fraction
        percentile_univariate: Optional[int] = None, # For univariate_percentile: 1-100
        threshold_from_model: Optional[Union[str, float]] = None, # For select_from_model
        max_features_from_model: Optional[int] = None, # For select_from_model
        n_features_rfe: Optional[Union[int, float]] = None, # For RFE: int for count, float for fraction
        # Estimator for model-based methods
        estimator: Optional[Any] = None,
        # Univariate specific
        scoring_univariate: Optional[Union[str, Callable]] = None,
        # RFE specific
        step_rfe: Union[int, float] = 1,
        # Preprocessing
        impute_strategy_numeric: str = 'mean',
        impute_strategy_categorical: str = 'most_frequent',
        scale_numeric: bool = True,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Selects features using scikit-learn methods.

        :param data: DataFrame containing features and target.
        :param target_column: Name of the target column.
        :param task_type: 'classification' or 'regression'.
        :param method: Feature selection method:
                       'univariate_kbest', 'univariate_percentile',
                       'select_from_model', 'rfe'.
        :param k_or_frac_kbest: For 'univariate_kbest'. Number of top features (int) or fraction (float).
                                Defaults to 10 if method is 'univariate_kbest' and this is None.
        :param percentile_univariate: For 'univariate_percentile'. Percentile of features to keep (1-100).
                                      Defaults to 10 if method is 'univariate_percentile' and this is None.
        :param threshold_from_model: For 'select_from_model'. Threshold for feature importance.
        :param max_features_from_model: For 'select_from_model'. Max number of features to select.
        :param n_features_rfe: For 'rfe'. Number (int) or fraction (float) of features to select.
                               Defaults to 0.5 (half features) if method is 'rfe' and this is None.
        :param estimator: Sklearn estimator for 'select_from_model' or 'rfe'.
                          If None, defaults are used (e.g., LogisticRegression/Lasso).
        :param scoring_univariate: Scoring function for univariate methods.
                                   Can be a string like 'f_classif', 'mutual_info_regression' or a callable.
                                   Defaults to f_classif for classification, f_regression for regression.
        :param step_rfe: For 'rfe'. Number or percentage of features to remove at each iteration.
        :param impute_strategy_numeric: Imputation strategy for numeric features.
        :param impute_strategy_categorical: Imputation strategy for categorical features.
        :param scale_numeric: Whether to scale numeric features.
        :param random_state: Random state for reproducibility.
        :return: DataFrame with selected features and the target column.
                 Returns original data on failure or DataFrame with target only if no features selected.
        """
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found.")
            raise ValueError(f"Target column '{target_column}' not found in data.")

        if data.empty:
            logger.warning("Input DataFrame is empty. Returning an empty DataFrame.")
            return pd.DataFrame()

        X = data.drop(columns=[target_column])
        y = data[target_column]

        if X.empty:
            logger.warning("No feature columns found. Returning DataFrame with only the target column.")
            return y.to_frame()
        
        # Handle y for classification tasks that might require label encoding by some estimators/scorers
        y_processed = y.copy()
        label_encoder = None
        if task_type == 'classification' and y.dtype == 'object':
            logger.debug("Target column is object type for classification, applying LabelEncoder.")
            label_encoder = LabelEncoder()
            y_processed = pd.Series(label_encoder.fit_transform(y_processed), index=y_processed.index, name=y_processed.name)


        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer_steps = [('imputer', SimpleImputer(strategy=impute_strategy_numeric))]
        if scale_numeric:
            numeric_transformer_steps.append(('scaler', StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_transformer_steps)

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy_categorical, fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough' # Keep other columns if any (should not happen if X only contains features)
        )

        try:
            X_processed_np = preprocessor.fit_transform(X)
            # Get feature names after ColumnTransformer
            # Note: get_feature_names_out might produce duplicates if original names were like 'col_x', 'col_y'
            # and onehot makes 'col_x_val1', 'col_y_val1'. This should be fine for selection.
            try:
                processed_feature_names = preprocessor.get_feature_names_out()
            except Exception as e: # Older sklearn might not have fixed get_feature_names_out behavior perfectly.
                 logger.warning(f"Could not reliably get feature names from ColumnTransformer, using generic names: {e}")
                 processed_feature_names = [f"feature_{i}" for i in range(X_processed_np.shape[1])]


            if X_processed_np.shape[1] == 0:
                logger.warning("No features remaining after preprocessing. Returning target column only.")
                return y.to_frame()
            
            X_processed = pd.DataFrame(X_processed_np, columns=processed_feature_names, index=X.index)

            selector = None
            # --- Default parameter handling ---
            if method == 'univariate_kbest' and k_or_frac_kbest is None: k_or_frac_kbest = 10
            if method == 'univariate_percentile' and percentile_univariate is None: percentile_univariate = 10
            if method == 'rfe' and n_features_rfe is None: n_features_rfe = 0.5


            # --- Univariate Selection ---
            if method.startswith('univariate'):
                if scoring_univariate is None:
                    scoring_fn = f_classif if task_type == 'classification' else f_regression
                elif isinstance(scoring_univariate, str):
                    scoring_map = {
                        'f_classif': f_classif, 'f_regression': f_regression,
                        'mutual_info_classif': mutual_info_classif,
                        'mutual_info_regression': mutual_info_regression
                    }
                    scoring_fn = scoring_map.get(scoring_univariate)
                    if scoring_fn is None:
                        raise ValueError(f"Unknown string for scoring_univariate: {scoring_univariate}")
                else: # Callable
                    scoring_fn = scoring_univariate
                
                if method == 'univariate_kbest':
                    k = k_or_frac_kbest
                    if isinstance(k, float):
                        if not (0.0 < k <= 1.0):
                            raise ValueError("k_or_frac_kbest (float) must be between 0 and 1.")
                        k = max(1, int(k * X_processed.shape[1]))
                    elif not isinstance(k, int) or k <=0:
                         raise ValueError("k_or_frac_kbest (int) must be a positive integer.")
                    k = min(k, X_processed.shape[1]) # Cannot select more features than available
                    selector = SelectKBest(score_func=scoring_fn, k=k)
                    logger.info(f"Using SelectKBest with k={k}, scoring={scoring_fn.__name__ if hasattr(scoring_fn, '__name__') else str(scoring_fn)}")

                elif method == 'univariate_percentile':
                    if not (0 < percentile_univariate <= 100):
                        raise ValueError("percentile_univariate must be between 1 and 100.")
                    selector = SelectPercentile(score_func=scoring_fn, percentile=percentile_univariate)
                    logger.info(f"Using SelectPercentile with percentile={percentile_univariate}, scoring={scoring_fn.__name__ if hasattr(scoring_fn, '__name__') else str(scoring_fn)}")
            
            # --- SelectFromModel ---
            elif method == 'select_from_model':
                if estimator is None:
                    if task_type == 'classification':
                        estimator = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=random_state)
                    else: # regression
                        estimator = Lasso(alpha=0.01, random_state=random_state)
                    logger.info(f"No estimator provided for SelectFromModel, using default: {estimator.__class__.__name__}")
                
                # SelectFromModel fits the estimator internally
                selector = SelectFromModel(
                    estimator=estimator, 
                    threshold=threshold_from_model, 
                    max_features=max_features_from_model
                )
                logger.info(f"Using SelectFromModel with estimator={estimator.__class__.__name__}, threshold={threshold_from_model}, max_features={max_features_from_model}")

            # --- RFE (Recursive Feature Elimination) ---
            elif method == 'rfe':
                if estimator is None:
                    if task_type == 'classification':
                        # Using a simple model that has coef_ or feature_importances_
                        estimator = LogisticRegression(C=1.0, solver='liblinear', random_state=random_state) 
                    else: # regression
                        estimator = LinearRegression()
                    logger.info(f"No estimator provided for RFE, using default: {estimator.__class__.__name__}")

                n_features_to_select = n_features_rfe
                if isinstance(n_features_rfe, float):
                    if not (0.0 < n_features_rfe <= 1.0):
                        raise ValueError("n_features_rfe (float) must be between 0 and 1.")
                    n_features_to_select = max(1, int(n_features_rfe * X_processed.shape[1]))
                elif not isinstance(n_features_rfe, int) or n_features_rfe <=0:
                     raise ValueError("n_features_rfe (int) must be a positive integer.")
                n_features_to_select = min(n_features_to_select, X_processed.shape[1])
                
                selector = RFE(
                    estimator=estimator, 
                    n_features_to_select=n_features_to_select, 
                    step=step_rfe
                )
                logger.info(f"Using RFE with estimator={estimator.__class__.__name__}, n_features_to_select={n_features_to_select}, step={step_rfe}")
            
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")

            if selector is None: # Should not happen if logic is correct
                raise RuntimeError("Selector was not initialized.")

            logger.debug(f"Fitting selector: {selector.__class__.__name__} on processed data of shape {X_processed.shape}")
            selector.fit(X_processed, y_processed) # Use processed y for fitting selector
            
            selected_features_mask = selector.get_support()
            X_selected_values = X_processed.loc[:, selected_features_mask]

            if X_selected_values.shape[1] == 0:
                logger.warning("Feature selection resulted in 0 features. Returning target column only.")
                return y.to_frame()

            logger.info(f"Selected {X_selected_values.shape[1]} features out of {X_processed.shape[1]} processed features.")
            logger.info(f"Selected feature names: {X_selected_values.columns.tolist()}")

            # Combine selected features with the original target y
            final_df = pd.concat([X_selected_values.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return final_df

        except Exception as e:
            logger.error(f"Error during scikit-learn feature selection ({method}): {e}", exc_info=True)
            logger.warning("Feature selection failed. Returning original data.")
            return data.copy()
