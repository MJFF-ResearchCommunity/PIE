"""
feature_selector.py

Applies feature selection algorithms to an engineered dataset.
This module assumes that the input data is already numeric and imputed.
"""

import logging
import pandas as pd
from typing import Optional, Any, Callable, List, Union

from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    SelectFdr,
    SelectFromModel,
    RFE,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(f"PIE.{__name__}")

class FeatureSelector:
    """
    A stateful class to apply a feature selection algorithm.
    The selector is fitted on training data and can then be used to transform
    both training and test sets consistently.
    """
    def __init__(
        self,
        method: str,
        task_type: str,
        k_or_frac: Optional[float] = 0.5,
        alpha_fdr: float = 0.05,
        estimator: Optional[Any] = None,
        scoring_univariate: Optional[Union[str, Callable]] = None,
        random_state: int = 123
    ):
        """
        Initializes the FeatureSelector.

        :param method: 'k_best', 'fdr', 'select_from_model', or 'rfe'.
        :param task_type: 'classification' or 'regression'.
        :param k_or_frac: For 'k_best', the fraction of features to keep.
        :param alpha_fdr: For 'fdr', the alpha level to control the false discovery rate.
        :param estimator: An sklearn estimator for model-based selection ('select_from_model', 'rfe').
        :param scoring_univariate: The scoring function for univariate methods.
        :param random_state: Seed for reproducibility.
        """
        self.method = method
        self.task_type = task_type
        self.k_or_frac = k_or_frac
        self.alpha_fdr = alpha_fdr
        self.estimator = estimator
        self.scoring_univariate = scoring_univariate
        self.random_state = random_state

        self.selector: Optional[Any] = None
        self.selected_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the feature selector on the training data."""
        self._initialize_selector(X.shape[1])
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        self.selected_feature_names_ = X.columns[mask].tolist()
        logger.info(f"Selected {len(self.selected_feature_names_)} features using '{self.method}'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using the fitted selector."""
        if self.selector is None:
            raise RuntimeError("Selector has not been fitted. Call 'fit' first.")
        return X[self.selected_feature_names_]

    def _initialize_selector(self, n_features: int):
        """Initializes the scikit-learn selector object."""
        scoring_fn = self._get_univariate_scoring_fn()

        if self.method == 'k_best':
            k = max(1, int(self.k_or_frac * n_features))
            self.selector = SelectKBest(score_func=scoring_fn, k=k)
        elif self.method == 'fdr':
            self.selector = SelectFdr(score_func=scoring_fn, alpha=self.alpha_fdr)
        elif self.method == 'select_from_model':
            estimator = self._get_default_estimator()
            self.selector = SelectFromModel(estimator=estimator, threshold='median')
        elif self.method == 'rfe':
            estimator = self._get_default_estimator()
            n_to_select = max(1, int(self.k_or_frac * n_features))
            self.selector = RFE(estimator=estimator, n_features_to_select=n_to_select)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

    def _get_default_estimator(self) -> Any:
        if self.estimator:
            return self.estimator
        if self.task_type == 'classification':
            return LogisticRegression(random_state=self.random_state)
        return Lasso(random_state=self.random_state)

    def _get_univariate_scoring_fn(self) -> Callable:
        if self.scoring_univariate:
            if callable(self.scoring_univariate):
                return self.scoring_univariate
            scoring_map = {
                'f_classif': f_classif, 'f_regression': f_regression,
                'mutual_info_classif': mutual_info_classif,
                'mutual_info_regression': mutual_info_regression
            }
            if self.scoring_univariate not in scoring_map:
                raise ValueError(f"Unknown string for scoring_univariate: {self.scoring_univariate}")
            return scoring_map[self.scoring_univariate]
        return f_classif if self.task_type == 'classification' else f_regression
