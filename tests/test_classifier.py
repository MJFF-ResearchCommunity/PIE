"""Classifier and classification-report tests on synthetic data (fake participants).

The legacy run on pipeline outputs derived from PPMI data is marked ``ppmi``.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.classification_report import generate_report
from pie.classifier import Classifier, ENDGAME_AVAILABLE, _compute_metrics, _inject_thread_cap, get_model_catalog

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)


def participants_frame(n=60, visits=2, n_features=6, n_informative=3, seed=0,
                       labels=("Healthy Control", "Parkinson's Disease")):
    """n fake participants x `visits` rows; each participant keeps one label."""
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=n_informative,
                               n_redundant=0, random_state=seed)
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), visits)
    df = pd.DataFrame(X[rows] + rng.normal(0, 0.1, (n * visits, n_features)),
                      columns=[f"feat_{i}" for i in range(n_features)])
    df.insert(0, "PATNO", rows + 1)
    df["COHORT"] = np.array(labels)[y[rows]]
    return df


def _report(tmp_path, df, **kw):
    df.to_csv(tmp_path / "data.csv", index=False)
    args = dict(input_csv_path=str(tmp_path / "data.csv"), use_feature_selection=False,
                target_column="COHORT", output_dir=str(tmp_path / "out"), n_models_to_compare=1,
                tune_best_model=False, generate_plots=False, budget_time_minutes=0.05)
    args.update(kw)
    return generate_report(**args)


def test_binary_auc_is_computed():
    """AUC for a binary target used to be NaN: both probability columns went to roc_auc_score."""
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([[.8, .2], [.3, .7], [.6, .4], [.2, .8], [.4, .6], [.9, .1]])
    assert _compute_metrics(y, proba.argmax(1), proba)["AUC"] == pytest.approx(1.0)


@pytest.mark.skipif(not ENDGAME_AVAILABLE, reason="endgame not installed")
def test_thread_cap_applies_to_none_and_kwargs_wrappers():
    """Bug 2: the cap must reach endgame's **kwargs wrappers and override n_jobs=None."""
    from endgame.models.wrappers import XGBWrapper, CatBoostWrapper
    from sklearn.ensemble import RandomForestClassifier
    assert _inject_thread_cap(XGBWrapper, {"n_jobs": None})["n_jobs"] == 2
    assert _inject_thread_cap(CatBoostWrapper, {})["thread_count"] == 2
    assert _inject_thread_cap(RandomForestClassifier, {"n_jobs": None})["n_jobs"] == 2
    assert _inject_thread_cap(RandomForestClassifier, {"n_jobs": 1})["n_jobs"] == 1
    # endgame's registry sets n_jobs=-1 for rf; that is capped, a caller's explicit value is not.
    from pie.classifier import _instantiate_model
    assert _instantiate_model("rf").get_params()["n_jobs"] == 2
    assert _instantiate_model("rf", n_jobs=1).get_params()["n_jobs"] == 1


@pytest.mark.skipif(not ENDGAME_AVAILABLE, reason="endgame not installed")
def test_gbdt_comparison_finishes_and_honours_include():
    """Bug 2: endgame's GBDT models finish on a small frame, and the no-budget path
    compares exactly the models in `include`. Runs in a subprocess so a hang fails the
    test instead of stalling the suite."""
    code = textwrap.dedent("""
        import logging; logging.disable(logging.CRITICAL)
        import pandas as pd
        from sklearn.datasets import make_classification
        from pie.classifier import Classifier
        X, y = make_classification(n_samples=150, n_features=6, n_informative=4, n_classes=3,
                                   n_clusters_per_class=1, random_state=0)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]); df["y"] = y
        c = Classifier(); c.setup_experiment(data=df, target="y", fold=3)
        c.compare_models(include=["xgb", "lgbm", "catboost"], verbose=False)
        print(len(c.comparison_results))
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "3"


def test_generate_report_feature_selection_is_applied(tmp_path):
    """Bug 3: use_feature_selection used to call a non-existent method and silently keep everything."""
    clf, best, info = _report(tmp_path, participants_frame(n_features=20, n_informative=4),
                              use_feature_selection=True, feature_selection_method="k_best")
    assert info["feature_selection_applied"] and info["selected_features"] == 10
    assert len(clf.get_config("feature_names")) == 10


def test_generate_report_numeric_binary_target(tmp_path):
    """Bug 4: a 0/1 target used to make generate_report return None."""
    result = _report(tmp_path, participants_frame(labels=(0, 1)))
    assert result is not None
    clf, best, info = result
    assert info["n_classes"] == 2
    assert set(clf.predict_model(best)["prediction_label"]) <= {0, 1}


def test_continuous_target_raises(tmp_path):
    df = participants_frame()
    df["COHORT"] = np.random.default_rng(0).normal(size=len(df))
    with pytest.raises(ValueError, match="continuous"):
        _report(tmp_path, df)


@pytest.mark.skipif("xgboost" not in get_model_catalog(), reason="xgboost not installed")
def test_labels_decode_to_original_values():
    """Bug 4: every target is label-encoded, so a 1/2-coded target trains XGBoost and
    predictions come back as 1/2."""
    df = participants_frame(labels=(1, 2)).drop(columns="PATNO")
    c = Classifier()
    c.setup_experiment(data=df, target="COHORT", fold=3)
    model = c.create_model("xgboost", n_estimators=20)
    assert set(c.predict_model(model)["prediction_label"]) <= {1, 2}


def test_report_test_metrics_are_held_out(tmp_path):
    """Bug 5: 'Final Test Set Performance' is the final (tuned) model on the test split."""
    clf, best, info = _report(tmp_path, participants_frame(), tune_best_model=True)
    pred = clf.predict_model(best)
    held_out_accuracy = (pred["prediction_label"] == pred["COHORT"]).mean()
    assert info["test_metrics"]["Accuracy"] == pytest.approx(held_out_accuracy)


def test_tune_model_choose_better_keeps_original():
    """Bug 6: choose_better was ignored; a worse tuned model replaced the original."""
    c = Classifier()
    c.setup_experiment(data=participants_frame(), target="COHORT", fold=3, fold_groups="PATNO")
    rf = c.create_model("rf", n_estimators=50)
    tuned = c.tune_model(rf, custom_grid={"max_depth": [1], "n_estimators": [1]}, n_iter=1,
                         choose_better=True, verbose=False)
    assert tuned is rf


def test_grouped_split_and_folds():
    """Bug 9: with fold_groups, no participant is on both sides of the split or of a CV fold."""
    from pie.classifier import split_train_test
    df = participants_frame()
    c = Classifier()
    c.setup_experiment(data=df, target="COHORT", fold=3, fold_groups="PATNO")
    train_ids = set(df.loc[c.get_config("X_train").index, "PATNO"])
    test_ids = set(df.loc[c.get_config("X_test").index, "PATNO"])
    assert train_ids and test_ids and not train_ids & test_ids
    assert "PATNO" not in c.get_config("feature_names")
    assert isinstance(c._cv(3), StratifiedGroupKFold)

    tr, te = split_train_test(df["COHORT"], df["PATNO"])
    assert not set(df["PATNO"].iloc[tr]) & set(df["PATNO"].iloc[te])


def test_generate_report_no_input_raises(tmp_path, monkeypatch):
    """Every failure path raises instead of returning None, which used to surface as
    'cannot unpack non-sequence NoneType' at the caller."""
    monkeypatch.chdir(tmp_path)  # nothing to auto-detect
    with pytest.raises(ValueError, match="No input data"):
        generate_report(target_column="COHORT", output_dir=str(tmp_path / "out"))


def test_generate_report_unreadable_input_raises(tmp_path):
    with pytest.raises(ValueError, match="missing.csv"):
        generate_report(input_csv_path=str(tmp_path / "missing.csv"), target_column="COHORT",
                        output_dir=str(tmp_path / "out"))


def test_generate_report_missing_target_column_raises(tmp_path):
    with pytest.raises(ValueError, match="NOPE"):
        _report(tmp_path, participants_frame(), target_column="NOPE")


def test_generate_report_empty_after_dropping_target_raises(tmp_path):
    df = participants_frame()
    df["COHORT"] = np.nan
    with pytest.raises(ValueError, match="missing 'COHORT'"):
        _report(tmp_path, df)


def test_generate_report_setup_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Classifier, "setup_experiment",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("engine exploded")))
    with pytest.raises(RuntimeError, match="set up the experiment"):
        _report(tmp_path, participants_frame())


def test_generate_report_comparison_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Classifier, "compare_models",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("no models evaluated")))
    with pytest.raises(RuntimeError, match="compare models"):
        _report(tmp_path, participants_frame())


def test_cli_exits_non_zero_with_one_line_message(tmp_path):
    participants_frame().to_csv(tmp_path / "data.csv", index=False)
    r = subprocess.run([sys.executable, "pie/classification_report.py",
                        "--input-csv-path", str(tmp_path / "data.csv"),
                        "--target-column", "NOPE", "--output-dir", str(tmp_path / "out")],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    output = r.stdout + r.stderr
    assert r.returncode == 1
    assert "NOPE" in output
    assert "Traceback" not in output


SELECTED = PROJECT_ROOT / "output" / "selected_train_data.csv"


@pytest.mark.ppmi
@pytest.mark.skipif(not SELECTED.exists(), reason="needs output/selected_*.csv from a PPMI pipeline run")
def test_classification_pipeline():
    """Runs generate_report on the selected CSVs of an earlier real-data pipeline run."""
    leakage = [l.strip() for l in (PROJECT_ROOT / "config" / "leakage_features.txt").read_text().splitlines() if l.strip()]
    result = generate_report(train_csv_path=str(SELECTED), test_csv_path=str(PROJECT_ROOT / "output" / "selected_test_data.csv"),
                             use_feature_selection=False, target_column="COHORT", exclude_features=leakage,
                             tune_best_model=False, generate_plots=True, budget_time_minutes=30.0)
    assert result is not None


if __name__ == "__main__":
    test_classification_pipeline()
