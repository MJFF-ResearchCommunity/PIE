"""
feature_selector.py

Applies feature selection algorithms to an engineered dataset.
This module assumes that the input data is already numeric and imputed.

Supports both sklearn-based methods and endgame's advanced selectors.
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

# Endgame feature selection imports
try:
    from endgame.feature_selection import (
        BorutaSelector,
        SHAPSelector,
        MRMRSelector,
        ReliefFSelector,
        AdversarialFeatureSelector,
        PermutationSelector,
        GeneticSelector,
        StabilitySelector as EndgameStabilitySelector,
        KnockoffSelector,
        NullImportanceSelector,
        TreeImportanceSelector,
        CorrelationSelector,
    )
    ENDGAME_FS_AVAILABLE = True
except ImportError:
    ENDGAME_FS_AVAILABLE = False

logger = logging.getLogger(f"PIE.{__name__}")

# All supported methods and whether they require endgame
SUPPORTED_METHODS = {
    # sklearn methods
    "k_best": False,
    "fdr": False,
    "select_from_model": False,
    "rfe": False,
    # endgame methods
    "boruta": True,
    "shap": True,
    "mrmr": True,
    "relief": True,
    "adversarial": True,
    "permutation": True,
    "genetic": True,
    "stability": True,
    "knockoff": True,
    "null_importance": True,
    "tree_importance": True,
    "correlation": True,
}


class FeatureSelector:
    """
    A stateful class to apply a feature selection algorithm.
    The selector is fitted on training data and can then be used to transform
    both training and test sets consistently.

    Supports sklearn methods (k_best, fdr, select_from_model, rfe) and
    endgame methods (boruta, shap, mrmr, relief, adversarial, permutation,
    genetic, stability, knockoff, null_importance, tree_importance, correlation).
    """
    def __init__(
        self,
        method: str,
        task_type: str,
        k_or_frac: Optional[float] = 0.5,
        alpha_fdr: float = 0.05,
        estimator: Optional[Any] = None,
        scoring_univariate: Optional[Union[str, Callable]] = None,
        random_state: int = 123,
        **kwargs,
    ):
        """
        Initializes the FeatureSelector.

        :param method: Selection method name (see SUPPORTED_METHODS).
        :param task_type: 'classification' or 'regression'.
        :param k_or_frac: For 'k_best', the fraction of features to keep.
        :param alpha_fdr: For 'fdr', the alpha level to control the false discovery rate.
        :param estimator: An sklearn estimator for model-based selection.
        :param scoring_univariate: The scoring function for univariate methods.
        :param random_state: Seed for reproducibility.
        :param kwargs: Additional keyword arguments passed to endgame selectors.
        """
        self.method = method
        self.task_type = task_type
        self.k_or_frac = k_or_frac
        self.alpha_fdr = alpha_fdr
        self.estimator = estimator
        self.scoring_univariate = scoring_univariate
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.selector: Optional[Any] = None
        self.selected_feature_names_: List[str] = []
        # Columns surviving the constant-feature pre-filter; the selector
        # itself only ever sees these.
        self._kept_input_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the feature selector on the training data."""
        # Drop zero-variance columns *before* handing X to the underlying
        # selector. Univariate scorers like f_classif warn ("Features [...]
        # are constant.") and produce NaN F-statistics (divide-by-zero in
        # msb / msw) when constant columns are present — both noisy and they
        # can drop genuinely useful neighbours via FDR correction.
        nunique = X.nunique(dropna=False)
        non_constant_mask = nunique > 1
        n_constant = int((~non_constant_mask).sum())
        if n_constant:
            constant_cols = X.columns[~non_constant_mask].tolist()
            logger.info(
                f"Dropping {n_constant} constant feature(s) before selection: "
                f"{constant_cols[:10]}{'...' if n_constant > 10 else ''}"
            )
            X_fit = X.loc[:, non_constant_mask]
        else:
            X_fit = X
        self._kept_input_features_ = list(X_fit.columns)

        self._initialize_selector(X_fit.shape[1])
        self.selector.fit(X_fit, y)
        mask = self.selector.get_support()
        self.selected_feature_names_ = X_fit.columns[mask].tolist()
        logger.info(f"Selected {len(self.selected_feature_names_)} features using '{self.method}'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using the fitted selector."""
        if self.selector is None:
            raise RuntimeError("Selector has not been fitted. Call 'fit' first.")
        return X[self.selected_feature_names_]

    def _initialize_selector(self, n_features: int):
        """Initializes the selector object based on the chosen method."""

        # --- sklearn methods ---
        if self.method == "k_best":
            scoring_fn = self._get_univariate_scoring_fn()
            k = max(1, int(self.k_or_frac * n_features))
            self.selector = SelectKBest(score_func=scoring_fn, k=k)

        elif self.method == "fdr":
            scoring_fn = self._get_univariate_scoring_fn()
            self.selector = SelectFdr(score_func=scoring_fn, alpha=self.alpha_fdr)

        elif self.method == "select_from_model":
            estimator = self._get_default_estimator()
            self.selector = SelectFromModel(estimator=estimator, threshold="median")

        elif self.method == "rfe":
            estimator = self._get_default_estimator()
            n_to_select = max(1, int(self.k_or_frac * n_features))
            self.selector = RFE(estimator=estimator, n_features_to_select=n_to_select)

        # --- endgame methods ---
        elif self.method == "boruta":
            self._require_endgame()
            self.selector = BorutaSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "shap":
            self._require_endgame()
            self.selector = SHAPSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "mrmr":
            self._require_endgame()
            n_to_select = max(1, int(self.k_or_frac * n_features))
            self.selector = MRMRSelector(
                n_features=n_to_select,
                **self.extra_kwargs,
            )

        elif self.method == "relief":
            self._require_endgame()
            n_to_select = max(1, int(self.k_or_frac * n_features))
            self.selector = ReliefFSelector(
                n_features=n_to_select,
                **self.extra_kwargs,
            )

        elif self.method == "adversarial":
            self._require_endgame()
            self.selector = AdversarialFeatureSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "permutation":
            self._require_endgame()
            estimator = self._get_default_estimator()
            self.selector = PermutationSelector(
                estimator=estimator,
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "genetic":
            self._require_endgame()
            self.selector = GeneticSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "stability":
            self._require_endgame()
            self.selector = EndgameStabilitySelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "knockoff":
            self._require_endgame()
            self.selector = KnockoffSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "null_importance":
            self._require_endgame()
            self.selector = NullImportanceSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "tree_importance":
            self._require_endgame()
            self.selector = TreeImportanceSelector(
                random_state=self.random_state,
                **self.extra_kwargs,
            )

        elif self.method == "correlation":
            self._require_endgame()
            self.selector = CorrelationSelector(
                **self.extra_kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported feature selection method: {self.method}. "
                f"Available: {list(SUPPORTED_METHODS.keys())}"
            )

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

    @staticmethod
    def _require_endgame():
        if not ENDGAME_FS_AVAILABLE:
            raise ImportError(
                "endgame-ml is required for this feature selection method. "
                "Install with: pip install endgame-ml[tabular]"
            )

    @staticmethod
    def available_methods(endgame_only: bool = False) -> List[str]:
        """Return a list of available feature selection method names."""
        if endgame_only:
            return [m for m, needs_eg in SUPPORTED_METHODS.items() if needs_eg and ENDGAME_FS_AVAILABLE]
        return [
            m for m, needs_eg in SUPPORTED_METHODS.items()
            if not needs_eg or ENDGAME_FS_AVAILABLE
        ]
