"""
classifier.py

Thin orchestration layer over endgame's full ML ecosystem.
Provides backward-compatible methods (setup_experiment, compare_models, tune_model,
predict_model) alongside new endgame-native APIs for AutoML, calibration,
ensembles, explainability, and more.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score,
    f1_score, log_loss, matthews_corrcoef, cohen_kappa_score,
)

# ---------------------------------------------------------------------------
# Endgame imports
# ---------------------------------------------------------------------------
try:
    import endgame as eg
    from endgame.automl import TabularPredictor
    from endgame.quick import classify as quick_classify, compare as quick_compare
    from endgame.tune import OptunaOptimizer
    from endgame.explain import explain as eg_explain
    from endgame.ensemble import SuperLearner
    from endgame.calibration import ConformalClassifier
    from endgame.validation import AdversarialValidator, NestedCV, cross_validate_oof
    from endgame.visualization import ClassificationReport as EndgameClassificationReport
    from endgame.preprocessing import AutoImputer, AutoBalancer
    from endgame.feature_selection import BorutaSelector, SHAPSelector
    ENDGAME_AVAILABLE = True
except ImportError:
    ENDGAME_AVAILABLE = False
    warnings.warn(
        "endgame-ml is not installed. Install it with: pip install endgame-ml[tabular]"
    )

logger = logging.getLogger(f"PIE.{__name__}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

_METRIC_MAP = {
    "Accuracy": accuracy_score,
    "AUC": lambda y, yp, **kw: roc_auc_score(y, yp, multi_class="ovr", **kw),
    "Recall": lambda y, yp, **kw: recall_score(y, yp, average="weighted", **kw),
    "Prec.": lambda y, yp, **kw: precision_score(y, yp, average="weighted", **kw),
    "F1": lambda y, yp, **kw: f1_score(y, yp, average="weighted", **kw),
    "MCC": matthews_corrcoef,
    "Kappa": cohen_kappa_score,
}


def _score(y_true, y_pred, y_proba=None, metric_name="Accuracy"):
    """Compute a single classification metric by name."""
    fn = _METRIC_MAP.get(metric_name)
    if fn is None:
        raise ValueError(f"Unknown metric: {metric_name}")
    if metric_name == "AUC" and y_proba is not None:
        try:
            return fn(y_true, y_proba)
        except Exception:
            return np.nan
    return fn(y_true, y_pred)


def _compute_metrics(y_true, y_pred, y_proba=None):
    """Return a dict of all standard classification metrics."""
    results = {}
    for name in _METRIC_MAP:
        try:
            results[name] = _score(y_true, y_pred, y_proba, name)
        except Exception:
            results[name] = np.nan
    return results


# ---------------------------------------------------------------------------
# Dynamic model catalog
# ---------------------------------------------------------------------------

def _can_import(module: str, cls_name: str) -> bool:
    """Return True if *module*.*cls_name* is importable."""
    import importlib
    try:
        mod = importlib.import_module(module)
        getattr(mod, cls_name)
        return True
    except Exception:
        return False


def get_model_catalog(task_type: str = "classification") -> Dict[str, Dict[str, Any]]:
    """
    Return ALL available models from endgame + sklearn + third-party.

    Each entry is ``{model_id: {"name": ..., "module": ..., "class": ...}}``.
    Models that cannot be imported are excluded from the returned catalog.
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    # 1. Pull from endgame's automl registry when available
    if ENDGAME_AVAILABLE:
        try:
            from endgame.automl import list_models, get_model_info
            for model_id in list_models(task_type=task_type):
                info = get_model_info(model_id)
                catalog[model_id] = info
            logger.info(f"Loaded {len(catalog)} models from endgame registry for {task_type}")
        except Exception as exc:
            logger.warning(f"Could not load endgame model registry: {exc}")

    # 2. Build the static catalog (sklearn + third-party + endgame-specific)
    if task_type == "classification":
        _sklearn_models = {
            "lr": {"name": "Logistic Regression", "module": "sklearn.linear_model", "class": "LogisticRegression"},
            "rf": {"name": "Random Forest", "module": "sklearn.ensemble", "class": "RandomForestClassifier"},
            "et": {"name": "Extra Trees", "module": "sklearn.ensemble", "class": "ExtraTreesClassifier"},
            "gbc": {"name": "Gradient Boosting", "module": "sklearn.ensemble", "class": "GradientBoostingClassifier"},
            "ada": {"name": "AdaBoost", "module": "sklearn.ensemble", "class": "AdaBoostClassifier"},
            "dt": {"name": "Decision Tree", "module": "sklearn.tree", "class": "DecisionTreeClassifier"},
            "knn": {"name": "K-Nearest Neighbors", "module": "sklearn.neighbors", "class": "KNeighborsClassifier"},
            "nb": {"name": "Naive Bayes", "module": "sklearn.naive_bayes", "class": "GaussianNB"},
            "svm": {"name": "Support Vector Machine", "module": "sklearn.svm", "class": "SVC"},
            "ridge": {"name": "Ridge Classifier", "module": "sklearn.linear_model", "class": "RidgeClassifier"},
            "lda": {"name": "Linear Discriminant Analysis", "module": "sklearn.discriminant_analysis", "class": "LinearDiscriminantAnalysis"},
            "qda": {"name": "Quadratic Discriminant Analysis", "module": "sklearn.discriminant_analysis", "class": "QuadraticDiscriminantAnalysis"},
        }
        _third_party = {
            "xgboost": {"name": "XGBoost", "module": "xgboost", "class": "XGBClassifier"},
            "lightgbm": {"name": "LightGBM", "module": "lightgbm", "class": "LGBMClassifier"},
            "catboost": {"name": "CatBoost", "module": "catboost", "class": "CatBoostClassifier"},
        }
        _endgame_models = {
            "ebm": {"name": "Explainable Boosting Machine", "module": "endgame.models", "class": "EBMClassifier"},
            "tabnet": {"name": "TabNet", "module": "endgame.models", "class": "TabNetClassifier"},
            "saint": {"name": "SAINT", "module": "endgame.models", "class": "SAINTClassifier"},
            "ft_transformer": {"name": "FT-Transformer", "module": "endgame.models", "class": "FTTransformerClassifier"},
            "node": {"name": "NODE", "module": "endgame.models", "class": "NODEClassifier"},
            "rule_fit": {"name": "RuleFit", "module": "endgame.models", "class": "RuleFitClassifier"},
        }
        static = {**_sklearn_models, **_third_party, **_endgame_models}
    else:
        _sklearn_models = {
            "lr": {"name": "Linear Regression", "module": "sklearn.linear_model", "class": "LinearRegression"},
            "rf": {"name": "Random Forest", "module": "sklearn.ensemble", "class": "RandomForestRegressor"},
            "et": {"name": "Extra Trees", "module": "sklearn.ensemble", "class": "ExtraTreesRegressor"},
            "gbc": {"name": "Gradient Boosting", "module": "sklearn.ensemble", "class": "GradientBoostingRegressor"},
            "ada": {"name": "AdaBoost", "module": "sklearn.ensemble", "class": "AdaBoostRegressor"},
            "dt": {"name": "Decision Tree", "module": "sklearn.tree", "class": "DecisionTreeRegressor"},
            "knn": {"name": "K-Nearest Neighbors", "module": "sklearn.neighbors", "class": "KNeighborsRegressor"},
            "svm": {"name": "Support Vector Regression", "module": "sklearn.svm", "class": "SVR"},
            "ridge": {"name": "Ridge Regression", "module": "sklearn.linear_model", "class": "Ridge"},
            "lasso": {"name": "Lasso Regression", "module": "sklearn.linear_model", "class": "Lasso"},
            "elastic_net": {"name": "Elastic Net", "module": "sklearn.linear_model", "class": "ElasticNet"},
        }
        _third_party = {
            "xgboost": {"name": "XGBoost", "module": "xgboost", "class": "XGBRegressor"},
            "lightgbm": {"name": "LightGBM", "module": "lightgbm", "class": "LGBMRegressor"},
            "catboost": {"name": "CatBoost", "module": "catboost", "class": "CatBoostRegressor"},
        }
        _endgame_models = {
            "ebm": {"name": "Explainable Boosting Machine", "module": "endgame.models", "class": "EBMRegressor"},
            "tabnet": {"name": "TabNet", "module": "endgame.models", "class": "TabNetRegressor"},
            "saint": {"name": "SAINT", "module": "endgame.models", "class": "SAINTRegressor"},
            "ft_transformer": {"name": "FT-Transformer", "module": "endgame.models", "class": "FTTransformerRegressor"},
            "node": {"name": "NODE", "module": "endgame.models", "class": "NODERegressor"},
            "rule_fit": {"name": "RuleFit", "module": "endgame.models", "class": "RuleFitRegressor"},
        }
        static = {**_sklearn_models, **_third_party, **_endgame_models}

    # 3. Merge static into catalog (don't override entries already populated
    #    by the dynamic registry, but add any that are missing)
    for mid, info in static.items():
        if mid not in catalog:
            catalog[mid] = info

    # 4. Validate: only keep entries whose module+class are actually importable
    validated: Dict[str, Dict[str, Any]] = {}
    for mid, info in catalog.items():
        mod_name = info.get("module", "")
        cls_name = info.get("class", "")
        if not mod_name or not cls_name:
            # Entries from the dynamic registry may use a different schema
            validated[mid] = info
            continue
        if _can_import(mod_name, cls_name):
            validated[mid] = info
        else:
            logger.debug(f"Model '{mid}' excluded from catalog: cannot import {mod_name}.{cls_name}")

    return validated


def _instantiate_model(model_id: str, task_type: str = "classification", **kwargs):
    """Instantiate a model from the catalog by *model_id*."""
    import importlib

    catalog = get_model_catalog(task_type)
    if model_id not in catalog:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(catalog.keys())}")
    info = catalog[model_id]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class Classifier:
    """
    Provides classification model training, evaluation, and comparison
    using endgame's full ecosystem.

    Backward-compatible surface:
        setup_experiment, compare_models, tune_model, predict_model,
        get_config, finalize_model, comparison_results attribute.

    New endgame-native surface:
        auto_ml, quick_classify, quick_compare, create_model,
        create_ensemble, calibrate_model, explain_model, validate_drift,
        generate_report, get_available_models, nested_cv, cross_validate.
    """

    def __init__(self):
        """Initialize the Classifier."""
        self.experiment = True  # Sentinel so old code can check `if self.experiment`
        self.best_model = None
        self.tuned_model = None
        self.models_dict: Dict[str, Any] = {}
        self.comparison_results: Optional[pd.DataFrame] = None
        self.setup_params: Optional[dict] = None

        # Internal state
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._y_test: Optional[pd.Series] = None
        self._target_name: Optional[str] = None
        self._feature_names: Optional[List[str]] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._fold: int = 5
        self._random_state: int = 123
        self._predictor: Optional[Any] = None  # TabularPredictor when using auto_ml
        self._task_type: str = "classification"

    # ------------------------------------------------------------------
    # Backward-compatible API
    # ------------------------------------------------------------------

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
        feature_selection_method: str = "classic",
        feature_selection_estimator: Optional[Any] = None,
        n_features_to_select: Union[int, float] = 0.2,
        fold_strategy: str = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Setup the classification experiment.

        Stores data, splits train/test, records configuration.
        """
        logger.info("Setting up endgame classification experiment...")

        self.setup_params = {
            "data": data,
            "target": target,
            "session_id": session_id,
            "use_gpu": use_gpu,
            "log_experiment": log_experiment,
            "experiment_name": experiment_name,
            "verbose": verbose,
            "remove_multicollinearity": remove_multicollinearity,
            "multicollinearity_threshold": multicollinearity_threshold,
            "remove_outliers": remove_outliers,
            "outliers_threshold": outliers_threshold,
            "normalize": normalize,
            "transformation": transformation,
            "pca": pca,
            "pca_components": pca_components,
            "ignore_features": ignore_features,
            "feature_selection": feature_selection,
            "feature_selection_method": feature_selection_method,
            "n_features_to_select": n_features_to_select,
            "fold_strategy": fold_strategy,
            "fold": fold,
            "fold_shuffle": fold_shuffle,
        }
        if test_data is not None:
            self.setup_params["test_data"] = test_data
        else:
            self.setup_params["train_size"] = train_size

        self._target_name = target
        self._fold = fold
        self._random_state = session_id

        ignore = set(ignore_features or [])

        if test_data is not None:
            train_df = data.copy()
            test_df = test_data.copy()
        else:
            train_df, test_df = train_test_split(
                data, train_size=train_size, random_state=session_id,
                stratify=data[target],
            )

        feature_cols = [c for c in train_df.columns if c != target and c not in ignore]
        self._feature_names = feature_cols
        self._X_train = train_df[feature_cols].copy()
        self._y_train = train_df[target].copy()
        self._X_test = test_df[feature_cols].copy()
        self._y_test = test_df[target].copy()

        # Encode target if needed for metric computation
        if not np.issubdtype(self._y_train.dtype, np.number):
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(pd.concat([self._y_train, self._y_test]).unique().astype(str))

        logger.info(
            f"Experiment setup complete. "
            f"Train: {self._X_train.shape}, Test: {self._X_test.shape}, "
            f"Target: {target} ({self._y_train.nunique()} classes)"
        )
        return self.experiment

    def compare_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[int] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, str]] = None,
        parallel: Optional[Any] = None,
    ) -> Union[Any, List[Any]]:
        """
        Compare multiple classification models and return the best one(s).

        Uses endgame's ``quick.compare()`` when available, otherwise loops
        over the model catalog with cross-validated scoring.
        """
        self._require_setup()

        n_folds = fold or self._fold
        logger.info(f"Comparing models, sorting by {sort}...")

        catalog = get_model_catalog(self._task_type)
        if include:
            model_ids = [m for m in include if m in catalog]
        else:
            model_ids = list(catalog.keys())
            if exclude:
                model_ids = [m for m in model_ids if m not in exclude]

        if verbose:
            print("\n" + "=" * 60)
            print("COMPARING MODELS")
            print("=" * 60)
            print(f"Total models to compare: {len(model_ids)}")
            print(f"Cross-validation folds: {n_folds}")
            print(f"Optimization metric: {sort}")
            if budget_time:
                print(f"Time budget: {budget_time} minutes")
            print("=" * 60 + "\n")

        # Try endgame quick.compare first
        if ENDGAME_AVAILABLE:
            try:
                result = quick_compare(
                    self._X_train, self._y_train,
                    preset="competition",
                    cv=n_folds,
                    metric=sort.lower(),
                    time_limit=int(budget_time * 60) if budget_time else None,
                )
                self.comparison_results = pd.DataFrame(result.leaderboard).round(round)
                self.best_model = result.best_model
                if n_select > 1 and hasattr(result, "top_models"):
                    top = result.top_models(n_select)
                    self.best_model = top[0]
                    return top
                return self.best_model
            except Exception as exc:
                logger.info(f"endgame quick.compare unavailable ({exc}), falling back to manual loop")

        # Manual fallback: loop over catalog, cross-validate each
        import time as _time
        deadline = _time.time() + budget_time * 60 if budget_time else float("inf")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._random_state)

        y_enc = self._encode_target(self._y_train)
        rows: List[Dict[str, Any]] = []

        for model_id in model_ids:
            if _time.time() > deadline:
                logger.info("Time budget exhausted, stopping model comparison.")
                break
            try:
                model = _instantiate_model(model_id, self._task_type)
                if verbose:
                    print(f"  Evaluating: {catalog[model_id]['name']} ({model_id})")

                fold_metrics: List[Dict[str, float]] = []
                for train_idx, val_idx in skf.split(self._X_train, y_enc):
                    Xtr = self._X_train.iloc[train_idx]
                    ytr = self._y_train.iloc[train_idx]
                    Xval = self._X_train.iloc[val_idx]
                    yval = self._y_train.iloc[val_idx]

                    ytr_enc = self._encode_target(ytr)
                    yval_enc = self._encode_target(yval)

                    model_clone = _instantiate_model(model_id, self._task_type)
                    model_clone.fit(Xtr, ytr_enc)
                    preds = model_clone.predict(Xval)

                    proba = None
                    if hasattr(model_clone, "predict_proba"):
                        try:
                            proba = model_clone.predict_proba(Xval)
                        except Exception:
                            pass

                    fold_metrics.append(_compute_metrics(yval_enc, preds, proba))

                mean_metrics = {
                    k: np.round(np.mean([fm[k] for fm in fold_metrics]), round)
                    for k in fold_metrics[0]
                }
                mean_metrics["Model"] = catalog[model_id]["name"]
                rows.append(mean_metrics)

                # Train a full model on all training data for later use
                full_model = _instantiate_model(model_id, self._task_type)
                full_model.fit(self._X_train, self._encode_target(self._y_train))
                self.models_dict[model_id] = full_model

            except Exception as exc:
                if errors == "ignore":
                    logger.warning(f"Model {model_id} failed: {exc}")
                else:
                    raise

        if not rows:
            raise RuntimeError("No models were successfully evaluated")

        leaderboard = pd.DataFrame(rows)
        # Reorder columns: Model first, then metrics
        metric_cols = [c for c in leaderboard.columns if c != "Model"]
        leaderboard = leaderboard[["Model"] + metric_cols]
        leaderboard = leaderboard.sort_values(sort, ascending=False).reset_index(drop=True)
        self.comparison_results = leaderboard

        # Select top model(s)
        top_model_name = leaderboard.iloc[0]["Model"]
        top_model_id = next(
            (mid for mid, info in catalog.items() if info["name"] == top_model_name),
            None,
        )
        self.best_model = self.models_dict.get(top_model_id)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"BEST MODEL: {top_model_name}")
            print(f"{'=' * 60}\n")

        if n_select > 1:
            selected = []
            for i in range(min(n_select, len(leaderboard))):
                name = leaderboard.iloc[i]["Model"]
                mid = next((m for m, info in catalog.items() if info["name"] == name), None)
                if mid and mid in self.models_dict:
                    selected.append(self.models_dict[mid])
            self.best_model = selected[0] if selected else self.best_model
            return selected

        return self.best_model

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[int] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """
        Create and train a specific model.

        *estimator* can be a model ID string (looked up in the catalog) or
        any sklearn-compatible estimator instance.
        """
        self._require_setup()

        if isinstance(estimator, str):
            model = _instantiate_model(estimator, self._task_type, **kwargs)
            model_name = estimator
        else:
            model = estimator
            model_name = type(estimator).__name__

        logger.info(f"Creating model: {model_name}")
        model.fit(self._X_train, self._encode_target(self._y_train), **(fit_kwargs or {}))
        self.models_dict[model_name] = model
        logger.info(f"Model {model_name} created successfully.")
        return model

    def tune_model(
        self,
        estimator: Optional[Any] = None,
        fold: Optional[int] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[dict] = None,
        optimize: str = "Accuracy",
        choose_better: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[str] = None,
        verbose: bool = True,
        tuner_verbose: Union[bool, int] = True,
        return_tuner: bool = False,
        **kwargs,
    ) -> Any:
        """
        Tune hyperparameters of a model using Optuna (via endgame) or
        sklearn's RandomizedSearchCV as fallback.
        """
        self._require_setup()

        if estimator is None:
            estimator = self.best_model
        if estimator is None:
            raise ValueError("No model to tune. Create or compare models first.")

        logger.info(f"Tuning model: {type(estimator).__name__}")

        # Try endgame OptunaOptimizer
        if ENDGAME_AVAILABLE:
            try:
                optimizer = OptunaOptimizer(
                    estimator=estimator,
                    X=self._X_train,
                    y=self._encode_target(self._y_train),
                    cv=fold or self._fold,
                    metric=optimize.lower(),
                    n_trials=n_iter,
                    random_state=self._random_state,
                )
                tuned = optimizer.optimize()
                self.tuned_model = tuned
                logger.info("Endgame Optuna tuning completed.")
                return tuned
            except Exception as exc:
                logger.info(f"Endgame tuning unavailable ({exc}), falling back to RandomizedSearchCV")

        # Fallback: sklearn RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV

        param_distributions = custom_grid or self._default_param_grid(estimator)
        if not param_distributions:
            logger.warning("No parameter grid available for this model. Returning original.")
            self.tuned_model = estimator
            return estimator

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=fold or self._fold, shuffle=True, random_state=self._random_state),
            scoring="accuracy" if optimize == "Accuracy" else optimize.lower(),
            random_state=self._random_state,
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )
        search.fit(self._X_train, self._encode_target(self._y_train))
        self.tuned_model = search.best_estimator_
        logger.info(f"Tuning complete. Best params: {search.best_params_}")

        if return_tuner:
            return self.tuned_model, search
        return self.tuned_model

    def predict_model(
        self,
        estimator: Optional[Any] = None,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Make predictions using a trained model.

        Returns a DataFrame with the original columns plus
        ``prediction_label`` and per-class score columns.
        """
        self._require_setup()

        if estimator is None:
            estimator = self.tuned_model or self.best_model
        if estimator is None:
            raise ValueError("No model for predictions. Train a model first.")

        if data is not None:
            X = data[[c for c in self._feature_names if c in data.columns]].copy()
            result_df = data.copy()
        else:
            X = self._X_test.copy()
            result_df = self._X_test.copy()
            result_df[self._target_name] = self._y_test.values

        preds_enc = estimator.predict(X)

        # Decode labels
        if self._label_encoder is not None and not encoded_labels:
            preds = self._label_encoder.inverse_transform(preds_enc)
        else:
            preds = preds_enc

        result_df["prediction_label"] = preds

        # Probabilities
        if hasattr(estimator, "predict_proba"):
            try:
                proba = estimator.predict_proba(X)
                if self._label_encoder is not None:
                    classes = self._label_encoder.classes_
                else:
                    classes = [f"Class_{i}" for i in range(proba.shape[1])]
                for i, cls in enumerate(classes):
                    result_df[f"prediction_score_{cls}"] = np.round(proba[:, i], round)
            except Exception:
                pass

        if verbose:
            logger.info(f"Predictions generated: {result_df.shape}")
        return result_df

    def finalize_model(self, estimator: Optional[Any] = None) -> Any:
        """Refit the model on the full dataset (train + test)."""
        self._require_setup()

        if estimator is None:
            estimator = self.tuned_model or self.best_model
        if estimator is None:
            raise ValueError("No model to finalize.")

        X_full = pd.concat([self._X_train, self._X_test], ignore_index=True)
        y_full = pd.concat([self._y_train, self._y_test], ignore_index=True)
        y_enc = self._encode_target(y_full)

        import copy
        final = copy.deepcopy(estimator)
        final.fit(X_full, y_enc)
        logger.info("Model finalized on full dataset.")
        return final

    def get_config(self, variable: Optional[str] = None) -> Any:
        """
        Get experiment configuration.

        Supported variables: ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``target_name``, ``feature_names``, ``label_encoder``.
        """
        self._require_setup()

        config = {
            "X_train": self._X_train,
            "X_test": self._X_test,
            "y_train": self._y_train,
            "y_test": self._y_test,
            "target_name": self._target_name,
            "feature_names": self._feature_names,
            "label_encoder": self._label_encoder,
        }
        if variable is not None:
            return config.get(variable)
        return config

    def get_available_models(self) -> pd.DataFrame:
        """Return a DataFrame of all available models."""
        catalog = get_model_catalog(self._task_type)
        rows = [{"ID": mid, "Name": info.get("name", mid)} for mid, info in catalog.items()]
        return pd.DataFrame(rows).set_index("ID")

    # ------------------------------------------------------------------
    # Endgame-native API
    # ------------------------------------------------------------------

    def auto_ml(
        self,
        time_limit: int = 3600,
        presets: str = "good_quality",
        constraints: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Run endgame's full AutoML pipeline via ``TabularPredictor``.
        """
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required for auto_ml(). pip install endgame-ml[tabular]")
        self._require_setup()

        train_df = self._X_train.copy()
        train_df[self._target_name] = self._y_train.values

        self._predictor = TabularPredictor(
            label=self._target_name,
            presets=presets,
            time_limit=time_limit,
            **({"constraints": constraints} if constraints else {}),
            **kwargs,
        )
        self._predictor.fit(train_df)
        self.best_model = self._predictor
        try:
            self.comparison_results = self._predictor.leaderboard()
        except Exception:
            pass
        logger.info("AutoML training complete.")
        return self._predictor

    def quick_classify(self, X=None, y=None, preset: str = "competition"):
        """Convenience wrapper around ``endgame.quick.classify()``."""
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required. pip install endgame-ml[tabular]")
        X = X if X is not None else self._X_train
        y = y if y is not None else self._y_train
        return quick_classify(X, y, preset=preset)

    def quick_compare_models(self, X=None, y=None, preset: str = "competition"):
        """Convenience wrapper around ``endgame.quick.compare()``."""
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required. pip install endgame-ml[tabular]")
        X = X if X is not None else self._X_train
        y = y if y is not None else self._y_train
        return quick_compare(X, y, preset=preset)

    def create_ensemble(
        self,
        base_models: Optional[List[Any]] = None,
        method: str = "super_learner",
        **kwargs,
    ) -> Any:
        """
        Create an ensemble from base models.

        Methods: ``super_learner``, ``bma`` (Bayesian Model Averaging),
        ``blending``, ``bagging``, ``boosting``.
        """
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required. pip install endgame-ml[tabular]")
        self._require_setup()

        if base_models is None:
            base_models = list(self.models_dict.values())
        if not base_models:
            raise ValueError("No base models provided or available.")

        method_lower = method.lower().replace(" ", "_")

        if method_lower == "super_learner":
            ensemble = SuperLearner(base_models=base_models, **kwargs)
        elif method_lower == "bma":
            from endgame.ensemble import BayesianModelAveraging
            ensemble = BayesianModelAveraging(base_models=base_models, **kwargs)
        elif method_lower == "blending":
            from endgame.ensemble import BlendingEnsemble
            ensemble = BlendingEnsemble(base_models=base_models, **kwargs)
        else:
            from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
            if method_lower == "bagging":
                ensemble = BaggingClassifier(estimator=base_models[0], **kwargs)
            elif method_lower == "boosting":
                ensemble = AdaBoostClassifier(estimator=base_models[0], **kwargs)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")

        ensemble.fit(self._X_train, self._encode_target(self._y_train))
        logger.info(f"Ensemble ({method}) created and trained.")
        return ensemble

    def calibrate_model(
        self,
        estimator: Optional[Any] = None,
        method: str = "conformal",
        **kwargs,
    ) -> Any:
        """
        Calibrate a model's probability estimates.

        Methods: ``conformal``, ``temperature_scaling``, ``venn_abers``,
        ``platt``, ``isotonic``.
        """
        if estimator is None:
            estimator = self.tuned_model or self.best_model
        if estimator is None:
            raise ValueError("No model to calibrate.")

        if ENDGAME_AVAILABLE and method in ("conformal", "temperature_scaling", "venn_abers"):
            if method == "conformal":
                calibrated = ConformalClassifier(estimator=estimator, **kwargs)
            elif method == "temperature_scaling":
                from endgame.calibration import TemperatureScaling
                calibrated = TemperatureScaling(estimator=estimator, **kwargs)
            else:
                from endgame.calibration import VennABERS
                calibrated = VennABERS(estimator=estimator, **kwargs)
            calibrated.fit(self._X_train, self._encode_target(self._y_train))
            return calibrated

        # Fallback: sklearn CalibratedClassifierCV
        from sklearn.calibration import CalibratedClassifierCV
        cal_method = "isotonic" if method == "isotonic" else "sigmoid"
        calibrated = CalibratedClassifierCV(estimator=estimator, method=cal_method, cv=self._fold)
        calibrated.fit(self._X_train, self._encode_target(self._y_train))
        logger.info(f"Model calibrated with {method}.")
        return calibrated

    def explain_model(
        self,
        estimator: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        method: str = "shap",
        **kwargs,
    ) -> Any:
        """
        Explain model predictions using SHAP, LIME, PDP, or counterfactual.
        """
        if estimator is None:
            estimator = self.tuned_model or self.best_model
        if estimator is None:
            raise ValueError("No model to explain.")
        X = X if X is not None else self._X_test

        if ENDGAME_AVAILABLE:
            try:
                return eg_explain(estimator, X, method=method, **kwargs)
            except Exception as exc:
                logger.info(f"endgame explain failed ({exc}), falling back to SHAP directly")

        # Direct SHAP fallback
        import shap
        if method == "shap":
            explainer = shap.Explainer(estimator, X)
            return explainer(X, **kwargs)
        raise ValueError(f"Fallback only supports 'shap', not '{method}'")

    def validate_drift(
        self,
        train_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Run adversarial validation to detect dataset drift."""
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required. pip install endgame-ml[tabular]")
        train_data = train_data if train_data is not None else self._X_train
        test_data = test_data if test_data is not None else self._X_test
        validator = AdversarialValidator()
        return validator.validate(train_data, test_data)

    def generate_report(
        self,
        estimator: Optional[Any] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        output_path: str = "classification_report.html",
        **kwargs,
    ) -> str:
        """
        Generate endgame's comprehensive 42-chart HTML classification report.

        Falls back to a simple HTML summary when endgame is unavailable.
        """
        if estimator is None:
            estimator = self.tuned_model or self.best_model
        if estimator is None:
            raise ValueError("No model to report on.")

        X_test = X_test if X_test is not None else self._X_test
        y_test = y_test if y_test is not None else self._y_test

        if ENDGAME_AVAILABLE:
            try:
                report = EndgameClassificationReport(
                    estimator, X_test, self._encode_target(y_test), **kwargs
                )
                report.save(output_path)
                logger.info(f"Endgame report saved to {output_path}")
                return output_path
            except Exception as exc:
                logger.warning(f"Endgame report generation failed ({exc}), using fallback.")

        # Fallback: simple HTML
        return self._generate_fallback_report(estimator, X_test, y_test, output_path)

    def nested_cv(
        self,
        estimator: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        outer_cv: int = 5,
        inner_cv: int = 3,
        **kwargs,
    ) -> Any:
        """Proper nested cross-validation via endgame's ``NestedCV``."""
        if not ENDGAME_AVAILABLE:
            raise ImportError("endgame-ml is required. pip install endgame-ml[tabular]")
        if estimator is None:
            estimator = self.tuned_model or self.best_model
        X = X if X is not None else self._X_train
        y = y if y is not None else self._y_train
        ncv = NestedCV(estimator=estimator, outer_cv=outer_cv, inner_cv=inner_cv, **kwargs)
        return ncv.evaluate(X, self._encode_target(y))

    def cross_validate(
        self,
        estimator: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        cv_method: str = "stratifiedkfold",
        n_splits: int = 5,
        **kwargs,
    ) -> Any:
        """
        Cross-validate with endgame's rich CV splitters.
        """
        if estimator is None:
            estimator = self.tuned_model or self.best_model
        X = X if X is not None else self._X_train
        y = y if y is not None else self._y_train

        if ENDGAME_AVAILABLE:
            try:
                return cross_validate_oof(
                    estimator, X, self._encode_target(y),
                    cv=cv_method, n_splits=n_splits, **kwargs,
                )
            except Exception as exc:
                logger.info(f"endgame cross_validate_oof failed ({exc}), using sklearn")

        from sklearn.model_selection import cross_val_score
        return cross_val_score(
            estimator, X, self._encode_target(y),
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self._random_state),
        )

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Optional[Any] = None,
        model_name: str = "pie_classifier_model",
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[Any, str]:
        """Save a trained model to disk using joblib."""
        if model is None:
            model = self.tuned_model or self.best_model
        if model is None:
            raise ValueError("No model to save.")

        filepath = f"{model_name}.pkl"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        payload = model if model_only else {
            "model": model,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "target_name": self._target_name,
        }
        joblib.dump(payload, filepath)
        if verbose:
            logger.info(f"Model saved to {filepath}")
        return model, filepath

    def load_model(
        self,
        model_name: str,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """Load a previously saved model."""
        filepath = f"{model_name}.pkl" if not model_name.endswith(".pkl") else model_name
        payload = joblib.load(filepath)
        if isinstance(payload, dict):
            self._label_encoder = payload.get("label_encoder")
            self._feature_names = payload.get("feature_names")
            self._target_name = payload.get("target_name")
            model = payload["model"]
        else:
            model = payload
        if verbose:
            logger.info(f"Model loaded from {filepath}")
        return model

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_model_catalog(task_type: str = "classification") -> Dict[str, Dict[str, Any]]:
        """Class-level access to the dynamic model catalog."""
        return get_model_catalog(task_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_setup(self):
        if self._X_train is None:
            raise ValueError("Experiment not set up. Run setup_experiment first.")

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels to integers if a LabelEncoder is fitted."""
        if self._label_encoder is not None:
            return self._label_encoder.transform(y.astype(str))
        return y.values

    @staticmethod
    def _default_param_grid(estimator) -> dict:
        """Return a reasonable random-search grid for common estimators."""
        name = type(estimator).__name__.lower()
        if "randomforest" in name or "extratrees" in name:
            return {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        if "gradientboosting" in name:
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            }
        if "xgb" in name:
            return {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            }
        if "lgbm" in name or "lightgbm" in name:
            return {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [-1, 5, 10, 20],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 100],
                "subsample": [0.7, 0.8, 0.9],
            }
        if "catboost" in name:
            return {
                "iterations": [100, 200, 500],
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
            }
        if "logistic" in name:
            return {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "saga"],
                "max_iter": [200, 500, 1000],
            }
        if "svc" in name or "svm" in name:
            return {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            }
        if "kneighbors" in name:
            return {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            }
        if "decisiontree" in name:
            return {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        return {}

    def _generate_fallback_report(self, estimator, X_test, y_test, output_path):
        """Generate a simple HTML report as fallback."""
        preds = estimator.predict(X_test)
        y_enc = self._encode_target(y_test)
        metrics = _compute_metrics(y_enc, preds)

        html = f"""<!DOCTYPE html>
<html><head><title>PIE Classification Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }}
.container {{ max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; }}
h1 {{ color: #e74c3c; text-align: center; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background-color: #e74c3c; color: white; }}
</style></head><body><div class="container">
<h1>PIE Classification Report</h1>
<h2>Model: {type(estimator).__name__}</h2>
<table><tr><th>Metric</th><th>Value</th></tr>"""
        for k, v in metrics.items():
            if isinstance(v, float):
                html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        html += "</table></div></body></html>"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Fallback report saved to {output_path}")
        return output_path
