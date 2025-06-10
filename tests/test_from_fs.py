import os
import sys
from pathlib import Path
import pytest
import shutil
import pandas as pd

# Add parent dir to path to import pie
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.pipeline import run_pipeline
from config.constants import LEAKAGE_FEATURES # Import leakage features from the main test

# --- Test setup for real data ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# This test depends on the output of test_pipeline.py (the full run)
# It uses the engineered dataset as its starting point.
ENGINEERED_DATA_PATH = PROJECT_ROOT / "output" / "test_pipeline_run" / "final_engineered_dataset.csv"
DATA_DIR_FOR_CLASSIFICATION = str(PROJECT_ROOT / "PPMI") # Still need this path for context, though it's not directly loaded

# Mark to skip if the required input file doesn't exist.
requires_engineered_data = pytest.mark.skipif(
    not ENGINEERED_DATA_PATH.exists(),
    reason=f"Engineered data not found at {ENGINEERED_DATA_PATH}. Run the full pipeline test first."
)

@requires_engineered_data
def test_pipeline_from_feature_selection():
    """
    Tests the pipeline starting from the feature selection step, using the
    output from a previous feature engineering run.
    """
    output_dir = str(PROJECT_ROOT / "output" / "test_from_fs_run")
    config_dir = PROJECT_ROOT / "config"
    leakage_path = config_dir / "leakage_features.txt"

    # --- Setup ---
    # The input for this run is the engineered data from the full pipeline test.
    # We need to copy it to the new output directory so the pipeline can find it.
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True)
    
    # The pipeline expects the engineered file to be in its own output directory
    shutil.copy(ENGINEERED_DATA_PATH, Path(output_dir) / "final_engineered_dataset.csv")

    # The classification step still needs the leakage file
    config_dir.mkdir(exist_ok=True)
    with open(leakage_path, 'w') as f:
        f.write("\n".join(sorted(list(set(LEAKAGE_FEATURES)))))

    # --- Run the pipeline, skipping to the 'selection' step ---
    run_pipeline(
        data_dir=DATA_DIR_FOR_CLASSIFICATION, # Provide the original data dir context
        output_dir=output_dir,
        target_column='COHORT',
        leakage_features_path=str(leakage_path),
        skip_to_step='selection', # This is the key parameter for this test
        fs_method='fdr',
        fs_param_value=0.05
    )

    # --- Assertions ---
    output_path = Path(output_dir)
    
    # Assert that steps BEFORE feature selection did NOT run
    assert not (output_path / "data_reduction_report.html").exists()
    
    # Assert that feature engineering report was not re-generated in this run
    # (though the file we copied exists)
    # To check this, we'd need to compare timestamps, which is complex.
    # For now, we confirm the file exists from our copy operation.
    assert (output_path / "final_engineered_dataset.csv").exists()

    # Assert that outputs FROM feature selection and classification WERE created
    assert (output_path / "feature_selection_report.html").exists()
    train_df_path = output_path / "selected_train_data.csv"
    test_df_path = output_path / "selected_test_data.csv"
    assert train_df_path.exists()
    assert test_df_path.exists()
    assert pd.read_csv(train_df_path).shape[0] > 0
    
    classification_dir = output_path / "classification"
    assert classification_dir.is_dir()
    assert (classification_dir / "classification_report.html").exists()
    assert (classification_dir / "final_classifier_model.pkl").exists()

    assert (output_path / "pipeline_report.html").exists()

    # --- Sanity check ---
    # Ensure a known leakage feature is not in the final data, proving
    # that the classification step's exclusion logic still works.
    train_df = pd.read_csv(train_df_path)
    assert 'COHORT' in train_df.columns
    assert 'subject_characteristics_APPRDX' not in train_df.columns

if __name__ == "__main__":
    # This block allows the script to be run directly for profiling purposes,
    # bypassing the pytest runner.
    print("--- Running test_from_fs.py as a script for profiling ---")
    
    # Manually check for data, since the pytest mark is not active here
    if not ENGINEERED_DATA_PATH.exists():
        print(f"ERROR: Engineered data not found at {ENGINEERED_DATA_PATH}.")
        print("Please run the full pipeline test first via: pytest tests/test_pipeline.py")
    else:
        test_pipeline_from_feature_selection()

    print("--- Script execution finished ---")
