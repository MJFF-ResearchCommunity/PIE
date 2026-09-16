"""Pipeline tests.

The synthetic tests need no PPMI data: a stub replaces ``DataLoader.load``. The real-data
integration test is marked ``ppmi`` (``pytest -m ppmi``; deselect with ``-m "not ppmi"``)
and is skipped when ./PPMI is missing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pie
import pie.pipeline as pp
from pie.pipeline import run_pipeline, run_data_reduction_step, run_feature_selection_step
from config.constants import LEAKAGE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PPMI_DATA_PATH = PROJECT_ROOT / "PPMI"
LEAKAGE = "PATNO\nEVENT_ID\nsubject_characteristics_APPRDX\n"


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)


def fake_modalities(n=60, seed=0):
    """DataLoader-shaped dict for n fake participants x 2 visits.

    COHORT drives a numeric motor score and a text column (SMELL_STATUS), so selection
    sees signal of both kinds; APPRDX restates the label (a leakage column); RBD is a 0/1
    target for the non-COHORT test; participants 1-2 have a cohort outside the valid four.
    """
    rng = np.random.default_rng(seed)
    m = 2 * n
    patno = np.repeat(np.arange(1, n + 1), 2)
    event = np.tile(["BL", "V04"], n)
    is_pd = np.repeat(rng.random(n) < 0.5, 2)
    cohort = np.where(is_pd, "PD", "Control").astype(object)
    cohort[:4] = "Other"
    sc = pd.DataFrame({
        "PATNO": patno, "EVENT_ID": event, "COHORT": cohort,
        "AGE": rng.normal(65, 8, m),
        "SMELL_STATUS": np.where(is_pd ^ (rng.random(m) < 0.1), "LOSS", "NORMAL"),
        "APPRDX": np.where(is_pd, 1, 2),
        "RBD": ((is_pd & (rng.random(m) < 0.8)) | (rng.random(m) < 0.1)).astype(int),
    })
    motor = pd.DataFrame({"PATNO": patno, "EVENT_ID": event,
                          "NP3TOT": np.where(is_pd, 20, 3) + rng.normal(0, 4, m)})
    return {"subject_characteristics": sc, "motor_assessments": motor}


def _run(tmp_path, monkeypatch, target):
    out = tmp_path / "run"
    leak = tmp_path / "leakage.txt"
    leak.write_text(LEAKAGE)
    monkeypatch.setattr(pp.DataLoader, "load", lambda **_: fake_modalities())
    run_pipeline(data_dir=str(tmp_path), output_dir=str(out), target_column=target,
                 leakage_features_path=str(leak), n_models_to_compare=1, tune_best_model=False,
                 generate_plots=False, budget_time_minutes=0.1)
    return out


def test_missing_leakage_file_warns(tmp_path, caplog):
    # a typo in the path used to cost the leakage protection with no error at all
    cohort = np.array(["PD", "Control"] * 10)
    frame = pd.DataFrame({"PATNO": np.repeat(np.arange(1, 11), 2), "COHORT": cohort,
                          "signal": np.where(cohort == "PD", 1.0, 0.0) + np.linspace(0, 0.1, 20),
                          "noise": np.linspace(0, 1, 20)})
    csv = tmp_path / "engineered.csv"
    frame.to_csv(csv, index=False)
    with caplog.at_level("WARNING"):
        run_feature_selection_step(str(csv), train_csv_path=tmp_path / "train.csv",
                                   test_csv_path=tmp_path / "test.csv",
                                   output_html_path=tmp_path / "fs.html", target_column="COHORT",
                                   fs_method="fdr", fs_param_value=0.05,
                                   leakage_features_path=str(tmp_path / "typo.txt"))
    assert any("Leakage features file not found" in r.message for r in caplog.records)


@pytest.mark.parametrize("step", ["engineering", "selection", "classification"])
def test_missing_stage_input_is_an_error(tmp_path, step):
    # used to log and return None (CLI exit 0), indistinguishable from success
    with pytest.raises(FileNotFoundError):
        run_pipeline(data_dir=str(tmp_path), output_dir=str(tmp_path / "run"), target_column="COHORT",
                     leakage_features_path=None, skip_to_step=step)


def test_pipeline_end_to_end_synthetic(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, "COHORT")
    for name in ["final_reduced_consolidated_data.csv", "data_reduction_report.html",
                 "final_engineered_dataset.csv", "feature_engineering_report.html",
                 "selected_train_data.csv", "selected_test_data.csv", "feature_selection_report.html",
                 "classification/classification_report.html", "classification/final_classifier_model.pkl",
                 "pipeline_report.html"]:
        assert (out / name).exists(), name

    reduced = pd.read_csv(out / "final_reduced_consolidated_data.csv")
    assert set(reduced["COHORT"]) == {"Parkinson's Disease", "Healthy Control"}  # COHORT is the target

    train = pd.read_csv(out / "selected_train_data.csv")
    test = pd.read_csv(out / "selected_test_data.csv")
    assert "subject_characteristics_APPRDX" not in train.columns
    # Bug 9: split by participant, and continuous features standardised on training rows only.
    assert not set(train["PATNO"]) & set(test["PATNO"])
    assert abs(train["motor_assessments_NP3TOT"].mean()) < 1e-9
    # Bug 1: one-hot (bool) columns reach feature selection instead of being dropped.
    assert any(c.startswith("subject_characteristics_SMELL_STATUS_") for c in train.columns)


def test_pipeline_numeric_binary_target(tmp_path, monkeypatch):
    """Bug 8: a 0/1 target other than COHORT runs through every stage."""
    out = _run(tmp_path, monkeypatch, "subject_characteristics_RBD")
    reduced = pd.read_csv(out / "final_reduced_consolidated_data.csv")
    assert len(reduced) == 120 and "Other" in set(reduced["COHORT"])  # no COHORT filtering
    engineered = pd.read_csv(out / "final_engineered_dataset.csv")
    assert set(engineered["subject_characteristics_RBD"]) == {0, 1}  # neither scaled nor encoded
    assert (out / "classification" / "classification_report.html").exists()


@pytest.mark.parametrize("method", ["rfe", "k_best"])
def test_fraction_based_fs_methods(tmp_path, method):
    """Bug 7: --fs-param reaches every fraction-based method (rfe used to get k_or_frac=None)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(80, 10)), columns=[f"f{i}" for i in range(10)])
    df.insert(0, "PATNO", np.repeat(np.arange(1, 41), 2))
    df.insert(1, "EVENT_ID", np.tile(["BL", "V04"], 40))
    df["COHORT"] = np.where(df["f0"] + df["f1"] > 0, "PD", "HC")
    df.to_csv(tmp_path / "engineered.csv", index=False)
    info = run_feature_selection_step(str(tmp_path / "engineered.csv"), tmp_path / "tr.csv", tmp_path / "te.csv",
                                      tmp_path / "fs.html", "COHORT", method, 0.5)
    assert info["final_features"] == 5


def test_parse_modalities_accepts_extended_modalities():
    from pie.pipeline import parse_modalities
    assert parse_modalities("Imaging, subject_characteristics; bogus") == ["imaging", "subject_characteristics"]
    assert parse_modalities("") is None
    assert parse_modalities("bogus") is None


def test_imaging_features_do_not_replace_imaging_tables(tmp_path, monkeypatch):
    """Bug 10: with the Imaging folder loaded, the IDP CSV is added, not swapped in."""
    data = fake_modalities(n=20)
    base = data["subject_characteristics"][["PATNO", "EVENT_ID"]]
    data["imaging"] = {"Scan_Table": base.assign(SBR=np.linspace(1, 3, len(base)))}
    base.assign(IMAGEID="X", Left_Hippocampus=np.linspace(3000, 4000, len(base))).to_csv(tmp_path / "idps.csv", index=False)
    monkeypatch.setattr(pp.DataLoader, "load", lambda **_: data)
    run_data_reduction_step(str(tmp_path), tmp_path / "reduced.csv", tmp_path / "reduction.html",
                            imaging_features=str(tmp_path / "idps.csv"))
    cols = pd.read_csv(tmp_path / "reduced.csv").columns
    assert "imaging_Scan_Table_SBR" in cols and "imaging_idps_Left_Hippocampus" in cols


def test_cli_exits_non_zero_when_stage_input_missing(tmp_path):
    """A missing stage input is one line and exit 1, not a traceback."""
    import subprocess
    r = subprocess.run([sys.executable, "pie/pipeline.py", "--output-dir", str(tmp_path / "run"),
                        "--skip-to", "classification"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    output = r.stdout + r.stderr
    assert r.returncode == 1
    assert "not found" in output
    assert "Traceback" not in output


def test_no_shadowed_utils_package():
    """Bug 12: pie/utils/ (an unimportable copy of pie/reporting.py) is gone."""
    import pie.reporting  # noqa: F401
    assert not (Path(pie.__file__).parent / "utils").is_dir()


@pytest.mark.ppmi
@pytest.mark.skipif(not PPMI_DATA_PATH.exists(), reason=f"PPMI data not found at {PPMI_DATA_PATH}")
def test_full_pipeline_with_real_data(tmp_path):
    """Full run on the local PPMI download. Outputs go to output/test_pipeline_run, which
    tests/test_feature_selector.py and tests/test_from_fs.py read."""
    import shutil
    output_dir = PROJECT_ROOT / "output" / "test_pipeline_run"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    leakage_path = tmp_path / "leakage_features.txt"
    leakage_path.write_text("\n".join(sorted(set(LEAKAGE_FEATURES))))

    run_pipeline(data_dir=str(PPMI_DATA_PATH), output_dir=str(output_dir), target_column="COHORT",
                 leakage_features_path=str(leakage_path), fs_method="fdr", fs_param_value=0.05,
                 n_models_to_compare=2, tune_best_model=False, generate_plots=True, budget_time_minutes=5.0)

    for name in ["data_reduction_report.html", "final_reduced_consolidated_data.csv",
                 "feature_engineering_report.html", "final_engineered_dataset.csv",
                 "feature_selection_report.html", "selected_train_data.csv", "selected_test_data.csv",
                 "classification/classification_report.html", "classification/final_classifier_model.pkl",
                 "pipeline_report.html"]:
        assert (output_dir / name).exists(), name
    train = pd.read_csv(output_dir / "selected_train_data.csv")
    test = pd.read_csv(output_dir / "selected_test_data.csv")
    assert len(train) > 0 and "COHORT" in train.columns
    assert "subject_characteristics_APPRDX" not in train.columns
    assert not set(train["PATNO"]) & set(test["PATNO"])
