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
from typing import Union, Optional, Any, Callable, Dict, List, Tuple
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
    """Return True if *module*.*cls_name* is importable AND the class can be
    instantiated with no required runtime deps missing.

    Classes like ``EBMClassifier`` / ``RuleFitClassifier`` import successfully
    even when their optional backend (e.g. ``interpret``) isn't installed; the
    ImportError only fires on first ``__init__``.  Catching that here keeps
    such models out of the user-facing catalog instead of failing later inside
    ``compare_models``.
    """
    import importlib
    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
    except Exception:
        return False

    # Try a no-arg instantiation.  Classes that legitimately *require* args
    # raise TypeError, which we treat as "importable" — only ImportError /
    # ModuleNotFoundError signal a missing optional backend.
    try:
        cls()
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception:
        # Other failures (TypeError for required args, deprecation warnings
        # raised as errors, etc.) are not our concern at catalog time.
        pass
    return True


_catalog_cache: Dict[str, Dict[str, Any]] = {}


def get_model_catalog(task_type: str = "classification") -> Dict[str, Dict[str, Any]]:
    """
    Return ALL available models from endgame + sklearn + third-party.

    Each entry is ``{model_id: {"name": ..., "module": ..., "class": ...}}``.
    Models that cannot be imported are excluded from the returned catalog.
    Results are cached after the first call per task_type.
    """
    if task_type in _catalog_cache:
        return _catalog_cache[task_type]
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

    # 3. Start with the static catalog, then overlay with validated endgame
    #    entries.  This ensures sklearn/xgboost/etc. are always available
    #    even if the endgame registry entry for the same id lacks module/class.
    validated: Dict[str, Dict[str, Any]] = {}

    for mid, info in static.items():
        mod_name = info.get("module", "")
        cls_name = info.get("class", "")
        if mod_name and cls_name and _can_import(mod_name, cls_name):
            validated[mid] = info

    # Overlay with endgame entries that are actually importable
    for mid, info in catalog.items():
        mod_name = _info_get(info, "module", "")
        cls_name = _info_get(info, "class", "")
        if mod_name and cls_name and _can_import(mod_name, cls_name):
            validated[mid] = info

    _catalog_cache[task_type] = validated
    logger.info(f"Model catalog for {task_type}: {len(validated)} models available")
    return validated


def _info_get(info, key: str, default=""):
    """Read a field from a catalog entry (dict or endgame ModelInfo).

    ModelInfo uses ``class_path`` ('module.ClassName') instead of separate
    ``module`` / ``class`` fields.  This helper transparently resolves both.
    """
    if isinstance(info, dict):
        return info.get(key, default)

    # Endgame ModelInfo: derive module/class from class_path
    class_path = getattr(info, "class_path", "") or ""
    if key == "module" and class_path:
        return class_path.rsplit(".", 1)[0] if "." in class_path else default
    if key == "class" and class_path:
        return class_path.rsplit(".", 1)[-1] if "." in class_path else default
    if key == "name":
        return getattr(info, "display_name", "") or getattr(info, "name", default)
    return getattr(info, key, default)


# Conservative thread cap for fitted models. We deliberately do NOT use -1
# (all cores) because joblib's loky backend copies the training data into
# every worker — on wide PPMI feature matrices that quickly exhausts RAM.
_MAX_MODEL_THREADS = 2


class _SkipEndgame(Exception):
    """Internal control-flow signal: skip the endgame fast-path and fall
    through to the budgeted manual CV loop. Distinct from a real failure so
    we don't log it as 'endgame unavailable'."""


def _inject_thread_cap(cls, defaults: dict) -> dict:
    """Inject conservative defaults that keep models honest on a desktop:

    * Cap parallelism (``n_jobs`` / ``thread_count``) — joblib's loky backend
      copies X into every worker, so unbounded parallelism is the dominant
      RAM driver on wide PPMI matrices.
    * Bump ``max_iter`` for solvers whose default (100) is too low for ~30k
      rows × hundreds of features — otherwise every CV fold logs a noisy
      ConvergenceWarning and ships a half-trained model.
    * Quiet CatBoost, which is otherwise extremely chatty.

    Caller kwargs always win — we only fill values the caller didn't set.
    """
    import inspect

    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return defaults

    if "n_jobs" in params and "n_jobs" not in defaults:
        defaults["n_jobs"] = _MAX_MODEL_THREADS
    if "thread_count" in params and "thread_count" not in defaults:
        defaults["thread_count"] = _MAX_MODEL_THREADS

    # Bump iteration budgets for iterative linear solvers. sklearn's defaults
    # (100 for LR, 1000 for Ridge/SGD) are tuned for tiny demos.
    _IGHIGH_ITER_CLASSES = {
        "LogisticRegression", "LogisticRegressionCV",
        "RidgeClassifier", "RidgeClassifierCV",
        "Ridge", "Lasso", "ElasticNet",
        "LinearRegression",  # noop — has no max_iter
        "SGDClassifier", "SGDRegressor",
    }
    if (
        "max_iter" in params
        and "max_iter" not in defaults
        and cls.__name__ in _IGHIGH_ITER_CLASSES
    ):
        defaults["max_iter"] = 2000

    # CatBoost's verbosity defaults to chatty; quiet it when we own the call.
    if "verbose" in params and "verbose" not in defaults and cls.__name__.startswith("CatBoost"):
        defaults["verbose"] = False
    return defaults


def _instantiate_model(model_id: str, task_type: str = "classification", **kwargs):
    """Instantiate a model from the catalog by *model_id*."""
    import importlib

    catalog = get_model_catalog(task_type)
    if model_id not in catalog:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(catalog.keys())}")
    info = catalog[model_id]
    mod = importlib.import_module(_info_get(info, "module"))
    cls = getattr(mod, _info_get(info, "class"))

    # Use endgame default_params as base, let caller kwargs override
    defaults = {}
    if not isinstance(info, dict):
        dp = getattr(info, "default_params", None)
        if isinstance(dp, dict):
            defaults = dp.copy()
    defaults.update(kwargs)
    defaults = _inject_thread_cap(cls, defaults)
    return cls(**defaults)


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

        # Metadata-only setup_params. The raw `data` / `test_data` frames are
        # NOT stored here — they would otherwise pin the full dataset in RAM
        # for the lifetime of the Classifier (in addition to _X_train/_X_test).
        self.setup_params = {
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
        if test_data is None:
            self.setup_params["train_size"] = train_size

        self._target_name = target
        self._fold = fold
        self._random_state = session_id

        ignore = set(ignore_features or [])

        if test_data is not None:
            train_df = data
            test_df = test_data
        else:
            train_df, test_df = train_test_split(
                data, train_size=train_size, random_state=session_id,
                stratify=data[target],
            )

        feature_cols = [c for c in train_df.columns if c != target and c not in ignore]
        self._feature_names = feature_cols

        # Slice once. df[list] returns a new DataFrame (not a view), so the
        # caller's source can be GC'd as soon as they release their reference.
        self._X_train = train_df[feature_cols].copy()
        self._y_train = train_df[target].copy()
        self._X_test = test_df[feature_cols].copy()
        self._y_test = test_df[target].copy()
        del train_df, test_df

        # Strip pandas Categorical dtypes on the slim slices only — numpy
        # cannot interpret them. Doing it here (rather than on the input)
        # avoids mutating the caller's DataFrame.
        for df in (self._X_train, self._X_test):
            for col in df.columns:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(df[col].cat.categories.dtype)

        # Encode target if needed for metric computation
        is_numeric = np.issubdtype(self._y_train.dtype, np.number)
        if not is_numeric:
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
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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
            # Use a fast, representative default set rather than all 70+ models.
            # Users can pass include=list(catalog.keys()) for the full catalog.
            _DEFAULT_COMPARE = [
                "lr", "rf", "et", "gbc", "dt", "knn", "nb",
                "ridge", "lda", "ada",
                "xgb", "lgbm", "catboost",
            ]
            if turbo:
                model_ids = [m for m in _DEFAULT_COMPARE if m in catalog]
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

        # Try endgame quick.compare first — but ONLY if (a) endgame is
        # installed, AND (b) we either have no time budget or the installed
        # signature exposes a way to honour one.  Older endgame releases run
        # `compare()` to completion regardless of how long it takes; on a wide
        # PPMI frame that means hours of CPU and gigabytes of resident worker
        # copies even when the caller asked for a 30-minute budget.  In that
        # case we skip straight to the manual loop, which respects the
        # deadline check at the top of every model iteration.
        if ENDGAME_AVAILABLE:
            try:
                import inspect as _inspect
                try:
                    qc_params = set(_inspect.signature(quick_compare).parameters)
                except (TypeError, ValueError):
                    qc_params = set()

                budget_supported = bool(qc_params & {"time_limit", "timeout", "max_time"})

                if budget_time and not budget_supported:
                    logger.info(
                        "endgame quick.compare doesn't support a time budget in this release; "
                        "using the budgeted manual loop instead."
                    )
                    raise _SkipEndgame()

                candidate_kwargs = {
                    "preset": "default",
                    "cv_folds": n_folds,
                    "cv": n_folds,  # alias for older releases
                    "metric": sort.lower(),
                    "time_limit": int(budget_time * 60) if budget_time else None,
                    "timeout": int(budget_time * 60) if budget_time else None,
                    "max_time": int(budget_time * 60) if budget_time else None,
                    "verbose": verbose,
                }
                accepted = {
                    k: v for k, v in candidate_kwargs.items()
                    if (not qc_params or k in qc_params) and v is not None
                }

                result = quick_compare(self._X_train, self._y_train, **accepted)
                self.comparison_results = pd.DataFrame(result.leaderboard).round(round)
                self.best_model = result.best_model
                if n_select > 1 and hasattr(result, "top_models"):
                    top = result.top_models(n_select)
                    self.best_model = top[0]
                    return top
                return self.best_model
            except _SkipEndgame:
                pass
            except Exception as exc:
                logger.info(f"endgame quick.compare unavailable ({exc}), falling back to manual loop")

        # Manual fallback: loop over catalog, cross-validate each.
        #
        # Memory note: only CV metrics are retained per model. Full-data refits
        # happen exactly once at the end, for the winner(s) — not per model.
        import gc as _gc
        import time as _time
        deadline = _time.time() + budget_time * 60 if budget_time else float("inf")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._random_state)

        y_enc = self._encode_target(self._y_train)
        rows: List[Dict[str, Any]] = []
        ranked_ids: List[str] = []  # successful models, in evaluation order

        # Detect how many classes we're dealing with so we can auto-wrap
        # binary-only estimators (SLIM, FasterRisk, GOSDT, GAM, CORELS) with
        # OneVsRestClassifier. Without this, compare_models would silently
        # skip every binary-only model the user selected.
        try:
            n_classes = int(pd.Series(y_enc).nunique())
        except Exception:
            n_classes = 2

        # Known binary-only models. Source of truth is the `binary_only` flag
        # on endgame's ModelInfo, but the workbench's vendored endgame copy
        # may be older; this fallback set keeps multiclass working until the
        # registry is resynced.
        _BINARY_ONLY_MODELS = {"slim", "fasterrisk", "gosdt", "gam", "corels"}

        def _wrap_if_binary_only(estimator, model_id):
            if n_classes <= 2:
                return estimator
            info = catalog.get(model_id, {})
            binary_only = _info_get(info, "binary_only", False) or model_id in _BINARY_ONLY_MODELS
            if not binary_only:
                return estimator
            from sklearn.multiclass import OneVsRestClassifier
            return OneVsRestClassifier(estimator)

        logger.info(f"Evaluating {len(model_ids)} models with {n_folds}-fold CV on "
                     f"{self._X_train.shape[0]:,} rows x {self._X_train.shape[1]} features...")

        # Swallow callback failures so a broken UI hook can never take down CV.
        def _emit(event: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event)
            except Exception as cb_exc:
                logger.debug("progress_callback raised: %s", cb_exc)

        _emit({
            "phase": "compare_start",
            "n_models": len(model_ids),
            "n_folds": n_folds,
            "n_rows": int(self._X_train.shape[0]),
            "n_features": int(self._X_train.shape[1]),
        })

        for i, model_id in enumerate(model_ids):
            if _time.time() > deadline:
                logger.info("Time budget exhausted, stopping model comparison.")
                _emit({"phase": "budget_exhausted", "completed_models": i})
                break
            model_name = _info_get(catalog.get(model_id, {}), "name", model_id)
            try:
                t0 = _time.time()
                wrap_note = ""
                _info_for_log = catalog.get(model_id, {})
                if n_classes > 2 and (
                    _info_get(_info_for_log, "binary_only", False)
                    or model_id in _BINARY_ONLY_MODELS
                ):
                    wrap_note = f" [wrapped in OneVsRest for {n_classes}-class target]"
                logger.info(f"  [{i+1}/{len(model_ids)}] {model_name} ({model_id}){wrap_note}...")
                _emit({
                    "phase": "model_start",
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_idx": i,
                    "n_models": len(model_ids),
                })

                fold_metrics: List[Dict[str, float]] = []
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self._X_train, y_enc)):
                    _emit({
                        "phase": "fold_start",
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_idx": i,
                        "n_models": len(model_ids),
                        "fold_idx": fold_idx,
                        "n_folds": n_folds,
                    })
                    Xtr = self._X_train.iloc[train_idx]
                    ytr = self._y_train.iloc[train_idx]
                    Xval = self._X_train.iloc[val_idx]
                    yval = self._y_train.iloc[val_idx]

                    ytr_enc = self._encode_target(ytr)
                    yval_enc = self._encode_target(yval)

                    model_clone = _instantiate_model(model_id, self._task_type)
                    model_clone = _wrap_if_binary_only(model_clone, model_id)
                    model_clone.fit(Xtr, ytr_enc)
                    preds = model_clone.predict(Xval)

                    proba = None
                    if hasattr(model_clone, "predict_proba"):
                        try:
                            proba = model_clone.predict_proba(Xval)
                        except Exception:
                            pass

                    fold_metrics.append(_compute_metrics(yval_enc, preds, proba))

                    # Free the fold model + slices before the next fold
                    del model_clone, Xtr, ytr, Xval, yval, ytr_enc, yval_enc, preds, proba
                _gc.collect()

                elapsed = _time.time() - t0
                mean_metrics = {
                    k: np.round(np.mean([fm[k] for fm in fold_metrics]), round)
                    for k in fold_metrics[0]
                }
                mean_metrics["Model"] = model_name
                rows.append(mean_metrics)
                ranked_ids.append(model_id)
                logger.info(f"  [{i+1}/{len(model_ids)}] {model_name}: "
                            f"Accuracy={mean_metrics.get('Accuracy', '?')} ({elapsed:.1f}s)")
                _emit({
                    "phase": "model_done",
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_idx": i,
                    "n_models": len(model_ids),
                    "elapsed_seconds": float(elapsed),
                    "metrics": {k: float(v) for k, v in mean_metrics.items() if k != "Model"},
                })

            except Exception as exc:
                if errors == "ignore":
                    logger.warning(f"Model {model_id} failed: {exc}")
                    _emit({
                        "phase": "model_failed",
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_idx": i,
                        "n_models": len(model_ids),
                        "error": str(exc),
                    })
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

        # Map ranked names back to model_ids using their leaderboard order
        name_to_id = {
            _info_get(catalog.get(mid, {}), "name", mid): mid
            for mid in ranked_ids
        }
        ordered_ids = [name_to_id[row_name] for row_name in leaderboard["Model"].tolist()
                       if row_name in name_to_id]

        top_model_name = leaderboard.iloc[0]["Model"]
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"BEST MODEL: {top_model_name}")
            print(f"{'=' * 60}\n")

        # Refit ONLY the selected model(s) on the full training set.
        # Previously every compared model was refit and retained in models_dict,
        # which kept dozens of fully-trained estimators in RAM at once.
        n_to_refit = max(1, min(n_select, len(ordered_ids)))
        y_train_enc = self._encode_target(self._y_train)
        selected: List[Any] = []
        for mid in ordered_ids[:n_to_refit]:
            try:
                full_model = _instantiate_model(mid, self._task_type)
                full_model = _wrap_if_binary_only(full_model, mid)
                full_model.fit(self._X_train, y_train_enc)
                self.models_dict[mid] = full_model
                selected.append(full_model)
            except Exception as exc:
                logger.warning(f"Final refit of {mid} failed: {exc}")

        if not selected:
            raise RuntimeError("No models could be refit on the full training set")

        self.best_model = selected[0]
        _gc.collect()

        if n_select > 1:
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
            # Capped (not -1): loky workers each receive a copy of X_train, so
            # n_jobs * sizeof(X_train) is the real RAM ceiling here.
            n_jobs=_MAX_MODEL_THREADS,
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
        rows = [{"ID": mid, "Name": _info_get(info, "name", mid)} for mid, info in catalog.items()]
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

        # Preserve (name, estimator) pairs — SuperLearner requires them, and
        # they make logs/introspection far more useful for the other methods too.
        if base_models is None:
            named_models: List[tuple] = list(self.models_dict.items())
        else:
            named_models = [
                (getattr(m, "__class__").__name__ + f"_{i}", m)
                for i, m in enumerate(base_models)
            ]
        if not named_models:
            raise ValueError("No base models provided or available.")
        bare_models = [m for _, m in named_models]

        method_lower = method.lower().replace(" ", "_")
        y_train_enc = self._encode_target(self._y_train)

        if method_lower == "super_learner":
            ensemble = SuperLearner(base_estimators=named_models, **kwargs)
            ensemble.fit(self._X_train, y_train_enc)
        elif method_lower == "bma":
            # BMA uses information-criterion weights over *already fitted*
            # estimators, scored on a held-out validation set. We re-use the
            # test split as the validation set here since no separate val
            # split is wired through.
            from endgame.ensemble import BayesianModelAveraging
            ensemble = BayesianModelAveraging(**kwargs)
            X_val = self._X_test if self._X_test is not None else self._X_train
            y_val_src = self._y_test if self._y_test is not None else self._y_train
            ensemble.fit(bare_models, X_val, self._encode_target(y_val_src))
        elif method_lower == "blending":
            from endgame.ensemble import BlendingEnsemble
            ensemble = BlendingEnsemble(base_estimators=bare_models, **kwargs)
            ensemble.fit(self._X_train, y_train_enc)
        else:
            from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
            if method_lower == "bagging":
                ensemble = BaggingClassifier(estimator=bare_models[0], **kwargs)
            elif method_lower == "boosting":
                ensemble = AdaBoostClassifier(estimator=bare_models[0], **kwargs)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            ensemble.fit(self._X_train, y_train_enc)

        logger.info(f"Ensemble ({method}) created and trained over {len(named_models)} base models.")
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
        return validator.check_drift(train_data, test_data)

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
