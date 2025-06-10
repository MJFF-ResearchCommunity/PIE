import os
import sys
from pathlib import Path
import pytest
import shutil
import pandas as pd

# Add parent dir to path to import pie
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.pipeline import run_pipeline
from config.constants import LEAKAGE_FEATURES

# Make path relative to the test file location for robustness
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PPMI_DATA_PATH = PROJECT_ROOT / "PPMI"

# Mark to skip if real data directory doesn't exist
requires_ppmi_data = pytest.mark.skipif(not PPMI_DATA_PATH.exists(), reason=f"PPMI data not found at {PPMI_DATA_PATH}")

@requires_ppmi_data
def test_full_pipeline_with_real_data():
    """
    Tests the full pipeline from data loading to classification report generation
    using the actual PPMI data. This is an integration test.
    """
    data_dir = str(PPMI_DATA_PATH)
    output_dir = str(PROJECT_ROOT / "output" / "test_pipeline_run")
    config_dir = PROJECT_ROOT / "config"
    leakage_path = config_dir / "leakage_features.txt"

    # Setup: Clean output directory and create config file
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    
    config_dir.mkdir(exist_ok=True)
    with open(leakage_path, 'w') as f:
        # Use a set to handle duplicates from the original list
        f.write("\n".join(sorted(list(set(LEAKAGE_FEATURES)))))

    # Run the pipeline with parameters for a quick integration test
    run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        target_column='COHORT',
        leakage_features_path=str(leakage_path),
        fs_method='fdr',
        fs_param_value=0.05,
        n_models_to_compare=2,
        tune_best_model=False,
        generate_plots=True,
        budget_time_minutes=5.0
    )

    # Assert that all expected output files and directories are created
    output_path = Path(output_dir)
    
    # Step 1: Data Reduction outputs
    assert (output_path / "data_reduction_report.html").exists()
    reduced_df_path = output_path / "final_reduced_consolidated_data.csv"
    assert reduced_df_path.exists()
    assert pd.read_csv(reduced_df_path).shape[0] > 0

    # Step 2: Feature Engineering outputs
    assert (output_path / "feature_engineering_report.html").exists()
    engineered_df_path = output_path / "final_engineered_dataset.csv"
    assert engineered_df_path.exists()
    assert pd.read_csv(engineered_df_path).shape[0] > 0
    
    # Step 3: Feature Selection outputs
    assert (output_path / "feature_selection_report.html").exists()
    train_df_path = output_path / "selected_train_data.csv"
    test_df_path = output_path / "selected_test_data.csv"
    assert train_df_path.exists()
    assert test_df_path.exists()
    assert pd.read_csv(train_df_path).shape[0] > 0
    
    # Step 4: Classification outputs
    classification_dir = output_path / "classification"
    assert classification_dir.is_dir()
    assert (classification_dir / "classification_report.html").exists()
    assert (classification_dir / "final_classifier_model.pkl").exists()

    # Step 5: Final pipeline summary report
    assert (output_path / "pipeline_report.html").exists()

    # --- Sanity check on the data ---
    # Check that a key leakage feature was dropped from the final training data
    train_df = pd.read_csv(train_df_path)
    assert 'COHORT' in train_df.columns
    # This is a strong leakage feature that should have been removed by the pipeline.
    # The reducer prefixes the original column 'APPRDX' with its modality.
    assert 'subject_characteristics_APPRDX' not in train_df.columns


if __name__ == "__main__":
    # This block allows the script to be run directly for profiling purposes,
    # bypassing the pytest runner.
    print("--- Running test_pipeline.py as a script for profiling ---")
    
    # Manually check for data, since the pytest mark is not active here
    if not PPMI_DATA_PATH.exists():
        print(f"ERROR: PPMI data not found at {PPMI_DATA_PATH}. Cannot run profiling script.")
    else:
        test_full_pipeline_with_real_data()

    print("--- Script execution finished ---")
