# Classification (`pie.classification_report`, `pie.classifier`)

Stage 4 of the [pipeline](pipeline.md). Two layers:

| Module | Use it for |
|---|---|
| `classification_report.py` | `generate_report()`: CSVs in; model comparison, optional tuning, plots, saved model and an HTML report out. What the pipeline calls; also a CLI. |
| `classifier.py` | `Classifier`: the engine. Model catalog, cross-validated comparison, tuning, held-out evaluation, prediction, persistence, and wrappers over endgame-ml (AutoML, ensembles, calibration, explanation). |

The engine is [endgame-ml](https://pypi.org/project/endgame-ml/) (`endgame-ml[tabular]>=1.1.0` in
`requirements.txt`) with scikit-learn fallbacks. PyCaret is no longer used. Method names and many
arguments keep PyCaret's signatures so older scripts still run; arguments listed as *ignored* below
are accepted and have no effect.

Participant-level evaluation is built in: a `PATNO` column (or any column passed as
`fold_groups`) groups the train/test split and every CV fold, so no participant is scored on a
model that trained on another of their visits.

## `generate_report`

```python
generate_report(
    input_csv_path=None, train_csv_path=None, test_csv_path=None,
    use_feature_selection=True, feature_selection_method="k_best",
    target_column="COHORT", exclude_features=None, output_dir="output",
    n_models_to_compare=5, tune_best_model=True, generate_plots=True, budget_time_minutes=30.0,
) -> tuple[Classifier, model, dict] | None
```

| Argument | Meaning |
|---|---|
| `train_csv_path`, `test_csv_path` | Pre-split data, used as given. Take precedence over `input_csv_path`. |
| `input_csv_path` | One CSV, split here 80/20 with `split_train_test` (grouped by `PATNO` when present, stratified, seed 123). The split happens before feature selection so selection only sees training rows. |
| `use_feature_selection` | Fit `FeatureSelector(feature_selection_method)` on the training split's feature columns and keep the selected ones in both splits. Features must be numeric and `NaN`-free. |
| `feature_selection_method` | Any [FeatureSelector](feature_selector.md) method; `k_best` keeps half the features. |
| `target_column` | Class labels ([below](#target-labels)). |
| `exclude_features` | Dropped before anything else; names not in the data are logged and ignored. The target and `PATNO` cannot be excluded. The pipeline passes the leakage list here. |
| `output_dir` | Created if missing; see [Outputs](#outputs). |
| `n_models_to_compare` | `n_select` for `compare_models`: how many top models are refitted on the training split. The first is the one used. |
| `tune_best_model` | `tune_model(n_iter=20, fold=5, optimize="Accuracy")` on the best model; the original is kept unless tuning beats it in CV. |
| `generate_plots` | Write the matplotlib plots. |
| `budget_time_minutes` | `compare_models(budget_time=...)`. |

`PATNO`, if present, is the grouping key and never a feature; `EVENT_ID` is ignored as a feature.
With no input paths it looks, relative to the working directory, for
`output/selected_train_data.csv` plus `output/selected_test_data.csv` (and then skips feature
selection), then `output/final_engineered_dataset.csv`.

Steps:

1. Load; drop `exclude_features`; drop rows with a missing target; check the target.
2. Split a single CSV; apply feature selection on the training split, if asked.
3. `Classifier.setup_experiment(fold=5, session_id=123, fold_groups="PATNO" if present)`.
4. `compare_models(fold=5, sort="Accuracy", n_select=n_models_to_compare, turbo=True, budget_time=budget_time_minutes, errors="ignore", exclude=...)`.
   `svm` and `qda` are excluded above 5,000 rows or 100 features and `knn` above 10,000 rows; only
   the `knn` rule changes anything, since the other two are not in the default set.
5. Tune, if asked. Score the final model once on the held-out split (`Classifier.evaluate_model`).
6. `Classifier.generate_report` writes endgame's HTML report on the test split, when
   `endgame.visualization` imports.
7. Plots, if asked; `Classifier.save_model`; `classification_report.html`, then `webbrowser.open` on it.

Returns `(classifier, best_model, report_data)` — always a tuple, so
`classifier, best_model, info = generate_report(...)` is safe. Failures raise, naming the cause:

| Raised | When |
|---|---|
| `ValueError` | No input given and none to auto-detect; the input cannot be read; `target_column` is not in the data; no rows left once rows with a missing target are dropped; the target is continuous; feature selection keeps nothing. |
| `RuntimeError` | The experiment cannot be set up, or model comparison fails (no model succeeded, features not numeric, a class too small for the CV folds, a budget too short for one model). |

Steps that are genuinely optional stay optional: a failure while tuning, plotting, writing the
endgame report or saving the model is logged as a warning and the run continues.

### Target labels

Any class labels work: text, or integer codes such as 0/1 or 1/2/3. They are label-encoded
internally, since XGBoost requires classes 0..k-1, and decoded back in predictions, plots and the
report. A continuous target (non-integer values) raises `ValueError`: regression is out of scope,
so bin or map it into classes first.

### Outputs

| File (under `output_dir`) | Content |
|---|---|
| `classification_report.html` | Overview, excluded features, feature selection, leaderboard, best-model CV metrics, tuned hyperparameters, plots, top-20 feature importances, held-out test metrics, recommendations. |
| `final_classifier_model.pkl` | joblib dict `{model, label_encoder, feature_names, target_name}`; load with `Classifier().load_model(path)`. |
| `plots/endgame_report.html` | endgame's classification report for the final model on the test split. |
| `plots/confusion_matrix.png` | Test split. |
| `plots/feature.png` | Only for models with `feature_importances_`. |
| `plots/shap_summary.png` | endgame explain, or `shap.TreeExplainer` for tree models, on 100 training rows. |
| `plots/pca_visualization.png` | Standardised training features, first two components, coloured by class. |
| `plots/tsne_visualization_perp<p>.png` | t-SNE of up to 2,000 training rows, `p = min(30, rows // 4)`. |
| `plots/class_distribution.png` | Class counts over all rows. |

`generate_classification_report_html(report_data, output_html_path, plots_dir)` writes the HTML. It
embeds any of a fixed list of PNG names found in `plots_dir`, so plots from other tools placed there
with those names (`auc.png`, `pr.png`, `learning.png`, `umap_visualization.png`, …) are shown too.

### What the numbers mean

| Report section | `report_data` key | Computed on |
|---|---|---|
| 3. Leaderboard | `leaderboard` | k-fold CV on the training split, grouped by participant when `PATNO` is present. Used to pick the model. |
| 4. Best Model Details | `best_model_metrics` | The selected model's leaderboard row: CV on the training split, before tuning. |
| 9. Final Test Set Performance | `test_metrics` | The final model (tuned, if tuning ran) scored once on the held-out split. The number to quote. |

### CLI

```bash
python pie/classification_report.py \
    --train-csv-path output/run/selected_train_data.csv --test-csv-path output/run/selected_test_data.csv \
    --target-column COHORT --exclude-features-file config/leakage_features.txt \
    --output-dir output/run/classification --generate-plots --tune-best-model
```

| Flag | Default | Maps to |
|---|---|---|
| `--input-csv-path` | none | `input_csv_path` |
| `--train-csv-path`, `--test-csv-path` | none | `train_csv_path`, `test_csv_path` |
| `--target-column` | required | `target_column` |
| `--output-dir` | `output` | `output_dir` |
| `--exclude-features-file` | none | one name per line → `exclude_features` |
| `--use-feature-selection` | off | `use_feature_selection` |
| `--feature-selection-method` | `k_best` | `feature_selection_method` |
| `--n-models-to-compare` | `5` | `n_models_to_compare` |
| `--tune-best-model` | off | `tune_best_model` |
| `--generate-plots` | off | `generate_plots` |
| `--budget-time-minutes` | `30.0` | `budget_time_minutes` |

Plots, tuning and feature selection are **off** on the CLI and **on** by default in Python. On a
`ValueError`, `RuntimeError` or `FileNotFoundError` the CLI logs one line and exits 1.

## `Classifier`

```
setup_experiment ─► compare_models / create_model ─► tune_model ─► evaluate_model, predict_model, save_model
```

State lives on the instance: `best_model`, `tuned_model`, `models_dict` (fitted models by id or
class name), `comparison_results` (leaderboard) and `setup_params` (recorded arguments). Every method
that needs data raises `ValueError` before `setup_experiment`.

### `setup_experiment`

```python
setup_experiment(data, target, train_size=0.8, test_data=None, session_id=123,
                 ignore_features=None, fold=10, fold_groups=None, ...)       # returns True
```

| Argument | Effect |
|---|---|
| `data`, `target` | Training frame (or the whole frame when `test_data` is `None`) and the label column. Rows with a missing target are dropped. |
| `test_data` | Held-out frame. Without it, `data` is split with `split_train_test(test_size=1 - train_size, random_state=session_id)`. |
| `fold_groups` | Column name, e.g. `"PATNO"`. Its values never straddle the split (when `test_data` is `None`) or a CV fold (`StratifiedGroupKFold`), and it is not a feature. |
| `session_id` | Seed for the split, CV shuffling and tuning. |
| `ignore_features` | Columns left out of the feature set. |
| `fold` | Default fold count for `compare_models`, `tune_model`, `calibrate_model`. |
| *ignored* | `use_gpu`, `log_experiment`, `experiment_name`, `verbose`, `remove_multicollinearity`, `multicollinearity_threshold`, `remove_outliers`, `outliers_threshold`, `normalize`, `transformation`, `pca`, `pca_components`, `feature_selection`, `feature_selection_method`, `feature_selection_estimator`, `n_features_to_select`, `fold_strategy` (always stratified k-fold), `fold_shuffle` (always shuffled), `**kwargs`. |

No imputation, scaling or encoding of features happens here: they must already be numeric. Column
names with characters other than letters, digits and `_` are rewritten with `_` because LightGBM and
JSON serialisation reject them; `get_config("feature_names")` returns the rewritten names. The target
is checked with `check_classification_target` and always label-encoded.

### `compare_models`

```python
compare_models(include=None, exclude=None, fold=None, round=4, sort="Accuracy", n_select=1,
               budget_time=None, turbo=True, errors="ignore", verbose=True,
               progress_callback=None, ...)
```

Model set: `include` (ids missing from the catalog are dropped silently); otherwise with `turbo=True`
the default set `lr, rf, et, gbc, dt, knn, nb, ridge, lda, ada, xgb, lgbm, catboost` (those present),
or the whole catalog with `turbo=False`, minus `exclude`. `exclude` applies only when `include` is not
given.

Each model is cross-validated with stratified k-fold (`fold` or the setup default, shuffled, seeded,
grouped when `fold_groups` was set). Per fold: `Accuracy`, `AUC` (from `predict_proba`: the
positive-class column for binary targets, one-vs-rest for multiclass, `NaN` without probabilities),
`Recall`, `Prec.`, `F1` (weighted), `MCC`, `Kappa`. The fold means, rounded to `round` and sorted by
`sort` descending, become `comparison_results`. Binary-only models are wrapped in
`OneVsRestClassifier` for multiclass targets. Only the top `n_select` are refitted on the full
training split, to keep memory flat. `budget_time` (minutes) is checked before each model starts; a
model already fitting runs to completion. endgame's own comparison, with its own model pool, is
`quick_compare_models`.

Returns the best model, or a list of `n_select` models when `n_select > 1`. `errors="ignore"` logs a
failing model and moves on; any other value re-raises. Raises `RuntimeError` if no model succeeds.

`progress_callback(event)` receives dicts with `phase` = `compare_start`, `model_start`,
`fold_start`, `model_done` (with `metrics`, `elapsed_seconds`), `model_failed` (with `error`) or
`budget_exhausted`; exceptions raised by the callback are swallowed so a broken UI hook cannot stop
the comparison. *Ignored:* `cross_validation`, `fit_kwargs`, `groups`, `probability_threshold`,
`experiment_custom_tags`, `engine`, `parallel`.

### Other core methods

| Method | Behaviour |
|---|---|
| `create_model(estimator, fit_kwargs=None, **kwargs)` | `estimator` is a catalog id (`**kwargs` go to its constructor) or an estimator instance. Fits on the whole training split, no CV; stored in `models_dict`. *Ignored:* `fold`, `round`, `cross_validation`, `groups`, `verbose`. |
| `tune_model(estimator=None, fold=None, n_iter=10, custom_grid=None, optimize="Accuracy", choose_better=True, return_tuner=False, verbose=True, ...)` | Defaults to `best_model`. `RandomizedSearchCV` over `custom_grid` or a small built-in grid (RF/ET, GBC, XGB, LightGBM, CatBoost, LR, SVC, KNN, DT), on the experiment's (grouped) folds. Scoring is `"accuracy"` for `"Accuracy"`, otherwise `optimize.lower()`, which must be an sklearn scorer name (`"f1"`, `"roc_auc"`, `"f1_weighted"`, `"balanced_accuracy"`). With `choose_better` the original model is cross-validated on the same folds and kept unless the tuned model scores higher. No grid for the model → returned unchanged. Sets `tuned_model`; `return_tuner=True` returns `(model, search)`. *Ignored:* `round`, `fit_kwargs`, `groups`, `tuner_verbose`. |
| `evaluate_model(estimator=None)` | The leaderboard metrics for a fitted model, scored once on the held-out test split. Defaults to `tuned_model`, then `best_model`. |
| `predict_model(estimator=None, data=None, encoded_labels=False, round=4, verbose=True, ...)` | Defaults to `tuned_model`, then `best_model`. `data=None` predicts the test split and includes the true target. Adds `prediction_label` (original labels unless `encoded_labels`) and `prediction_score_<class>` per class when the model has `predict_proba`. New data must use the rewritten column names. *Ignored:* `probability_threshold`, `raw_score`. |
| `finalize_model(estimator=None)` | Deep copy refitted on train + test. For deployment only: nothing is left to evaluate it on. |
| `get_config(variable=None)` | `X_train`, `X_test`, `y_train`, `y_test`, `target_name`, `feature_names`, `label_encoder`; all of them as a dict when `variable` is `None`. |
| `get_available_models()` | DataFrame indexed by model id with a `Name` column. |
| `save_model(model=None, model_name="pie_classifier_model", model_only=False, verbose=True)` | joblib to `<model_name>.pkl`: `{model, label_encoder, feature_names, target_name}`, or the bare model with `model_only=True`. Returns `(model, path)`. |
| `load_model(model_name)` | Loads `<model_name>` or `<model_name>.pkl`, restores the label encoder, feature names and target name onto the instance, returns the model. `predict_model` still needs `setup_experiment`; call the model's `predict` directly otherwise. |

Module-level helpers:

| Function | Does |
|---|---|
| `check_classification_target(y)` | Raises `ValueError` if `sklearn`'s `type_of_target` calls `y` continuous. |
| `split_train_test(y, groups=None, test_size=0.2, random_state=123)` | Positional `(train, test)` indices. With `groups`: first fold of `StratifiedGroupKFold(n_splits=round(1 / test_size))`, so no group is on both sides. Without: stratified `train_test_split`. |

### endgame-native methods

Default estimator is `tuned_model`, then `best_model`; default data is the experiment's splits.

| Method | Does | Without endgame |
|---|---|---|
| `auto_ml(time_limit=3600, presets="good_quality", constraints=None, **kwargs)` | endgame `TabularPredictor` on the training split; sets `best_model` and `comparison_results` (its leaderboard). | `ImportError` |
| `quick_classify(X=None, y=None, preset="competition")` | `endgame.quick.classify`. | `ImportError` |
| `quick_compare_models(X=None, y=None, preset="competition")` | `endgame.quick.compare`, with endgame's own model pool. | `ImportError` |
| `create_ensemble(base_models=None, method="super_learner", **kwargs)` | `super_learner`, `blending` (endgame, fitted on the training split), `bma` (endgame Bayesian model averaging, **weighted on the test split**, which is then no longer held out), `bagging`/`boosting` (sklearn, first base model only). `base_models=None` uses everything in `models_dict`. | `ImportError` |
| `calibrate_model(estimator=None, method="conformal", **kwargs)` | `conformal`, `temperature_scaling`, `venn_abers` (endgame); `platt`, `isotonic` → sklearn `CalibratedClassifierCV(cv=fold)`. | sklearn sigmoid/isotonic |
| `explain_model(estimator=None, X=None, method="shap", **kwargs)` | `endgame.explain`. | `shap.Explainer`, `method="shap"` only |
| `validate_drift(train_data=None, test_data=None)` | Adversarial validation (`AdversarialValidator.check_drift`): can a model tell train from test rows? Slow (minutes even on small data). | `ImportError` |
| `generate_report(estimator=None, X_test=None, y_test=None, output_path="classification_report.html", **kwargs)` | endgame `ClassificationReport` HTML, with class names taken from the label encoder so charts show labels rather than codes. Returns the path. | Plain metrics table |
| `nested_cv(estimator=None, X=None, y=None, outer_cv=5, inner_cv=3, **kwargs)` | endgame `NestedCV.evaluate` (not grouped). | `ImportError` |
| `cross_validate(estimator=None, X=None, y=None, cv_method="stratifiedkfold", n_splits=5, **kwargs)` | endgame `cross_validate_oof` (not grouped). | sklearn `cross_val_score` |

### Model catalog

`get_model_catalog(task_type="classification")` (also `Classifier.get_model_catalog`) returns
`{model_id: info}`. It starts from a static list — `lr`, `rf`, `et`, `gbc`, `ada`, `dt`, `knn`, `nb`,
`svm`, `ridge`, `lda`, `qda` (sklearn); `xgboost`, `lightgbm`, `catboost`; `ebm`, `tabnet`, `saint`,
`ft_transformer`, `node`, `rule_fit` (endgame) — and overlays endgame's model registry (which adds ids
such as `xgb` and `lgbm`; about 60 in total with endgame 1.1.0). An entry is kept only if its class
imports **and** instantiates without a missing optional backend, so the catalog never offers a model
that would fail later inside `compare_models`. The result is cached per `task_type`; the first call
instantiates every class and takes seconds. `task_type="regression"` has its own list, but
`Classifier` always uses `"classification"`.

When a model is built from an id, defaults are injected unless the caller sets them:

- **2 threads** (`n_jobs`, or `thread_count` for CatBoost), also when the value is `None` ("all
  cores" to the libraries) or when endgame's registry sets `-1` (it does for `rf` and
  `extra_trees`). An explicit value passed by the caller is kept. joblib's loky backend copies the training matrix into every worker,
  so on wide PPMI matrices memory, not CPU, is the limit. endgame's `XGBWrapper`, `LGBMWrapper` and
  `CatBoostWrapper` take the setting through `**kwargs`, so they are listed explicitly
  (`_KWARGS_THREAD_PARAM`). Uncapped, the `xgb` wrapper did not finish one fit on a 240-row table in
  15 minutes.
- `max_iter=2000` for logistic/ridge/lasso/elastic-net/SGD, whose defaults stop before convergence
  on thousands of rows × hundreds of features.
- `verbose=False` for CatBoost.

## Examples

Runnable with synthetic data:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pie.classifier import Classifier

X, y = make_classification(n_samples=300, n_features=8, n_informative=4, n_classes=3,
                           n_clusters_per_class=1, random_state=0)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
df["COHORT"] = np.array(["Healthy Control", "Parkinson's Disease", "Prodromal"])[y]
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["COHORT"], random_state=0)

clf = Classifier()
clf.setup_experiment(data=train_df, test_data=test_df, target="COHORT", session_id=42, fold=5)
best = clf.compare_models(include=["lr", "rf", "dt"], sort="AUC")
clf.comparison_results          # Model, Accuracy, AUC, Recall, Prec., F1, MCC, Kappa (5-fold CV, train split)

rf = clf.create_model("rf", n_estimators=200)
tuned = clf.tune_model(rf, n_iter=5, verbose=False)
clf.evaluate_model(tuned)       # the same metrics on the held-out split
pred = clf.predict_model(tuned)  # test split + prediction_label + prediction_score_<class>
_, path = clf.save_model(tuned, model_name="demo_rf")     # writes demo_rf.pkl
model = Classifier().load_model(path)
```

With several rows per participant, keep the ID column and pass it as `fold_groups`:
`clf.setup_experiment(data=frame_with_patno, target="COHORT", fold_groups="PATNO")`.

The full report from the same frames:

```python
from pie.classification_report import generate_report

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
clf, best, info = generate_report(
    train_csv_path="train.csv", test_csv_path="test.csv", use_feature_selection=False,
    target_column="COHORT", exclude_features=["feat_7"], output_dir="demo_classification",
    n_models_to_compare=1, tune_best_model=True, budget_time_minutes=5)
info["test_metrics"]             # held-out metrics of the final model
```

## Tests

`tests/test_classifier.py` runs on fake participants (`participants_frame`: each participant keeps
one label across visits).

| Test | Checks |
|---|---|
| `test_binary_auc_is_computed` | Binary AUC from two-column probabilities. |
| `test_thread_cap_applies_to_none_and_kwargs_wrappers` | 2-thread cap for `None` and for endgame's `**kwargs` wrappers; explicit values win. |
| `test_gbdt_comparison_finishes_and_honours_include` | `xgb`, `lgbm`, `catboost` compared without a budget, in a subprocess with a 300 s timeout; exactly those three on the leaderboard. |
| `test_generate_report_feature_selection_is_applied` | `k_best` keeps 10 of 20 features. |
| `test_generate_report_numeric_binary_target`, `test_labels_decode_to_original_values`, `test_continuous_target_raises` | 0/1 and 1/2 targets work and decode; a continuous target raises. |
| `test_report_test_metrics_are_held_out` | `test_metrics` equals the tuned model's held-out accuracy. |
| `test_tune_model_choose_better_keeps_original` | A worse tuned model does not replace the original. |
| `test_grouped_split_and_folds` | `fold_groups` keeps participants on one side and out of the features. |
| `test_classification_pipeline` (`ppmi`) | `generate_report` on `output/selected_*.csv` from a real-data run. |

```bash
pytest tests/test_classifier.py -m "not ppmi" -q
```
