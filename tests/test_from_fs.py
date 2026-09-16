import sys
from pathlib import Path
import pytest
import shutil
import pandas as pd

# Add parent dir to path to import pie
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.pipeline import run_pipeline
from config.constants import LEAKAGE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# This test starts from the engineered dataset of the real-data run in tests/test_pipeline.py.
ENGINEERED_DATA_PATH = PROJECT_ROOT / "output" / "test_pipeline_run" / "final_engineered_dataset.csv"


@pytest.mark.ppmi
@pytest.mark.skipif(not ENGINEERED_DATA_PATH.exists(),
                    reason=f"Engineered data not found at {ENGINEERED_DATA_PATH}. Run the full pipeline test first.")
def test_pipeline_from_feature_selection(tmp_path):
    """Runs the pipeline from the feature selection step on an earlier run's engineered data."""
    output_path = tmp_path / "from_fs_run"
    output_path.mkdir()
    shutil.copy(ENGINEERED_DATA_PATH, output_path / "final_engineered_dataset.csv")
    leakage_path = tmp_path / "leakage_features.txt"
    leakage_path.write_text("\n".join(sorted(set(LEAKAGE_FEATURES))))

    run_pipeline(
        data_dir=str(PROJECT_ROOT / "PPMI"),  # not read when skipping to selection
        output_dir=str(output_path),
        target_column='COHORT',
        leakage_features_path=str(leakage_path),
        skip_to_step='selection',
        fs_method='fdr',
        fs_param_value=0.05
    )

    # Steps before feature selection did not run
    assert not (output_path / "data_reduction_report.html").exists()
    assert not (output_path / "feature_engineering_report.html").exists()

    # Feature selection and classification outputs were created
    assert (output_path / "feature_selection_report.html").exists()
    train_df = pd.read_csv(output_path / "selected_train_data.csv")
    test_df = pd.read_csv(output_path / "selected_test_data.csv")
    assert train_df.shape[0] > 0
    assert (output_path / "classification" / "classification_report.html").exists()
    assert (output_path / "classification" / "final_classifier_model.pkl").exists()
    assert (output_path / "pipeline_report.html").exists()

    # Leakage exclusion and participant-level split
    assert 'COHORT' in train_df.columns
    assert 'subject_characteristics_APPRDX' not in train_df.columns
    assert not set(train_df["PATNO"]) & set(test_df["PATNO"])
