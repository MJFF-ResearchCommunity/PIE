"""
Tests for link to PIE-clean
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the PIE-clean classes and constants
from pie_clean import DataLoader, DataPreprocessor, ALL_MODALITIES, BIOSPECIMEN

def test_data_loader():
    # Data location
    data_dir = "./PPMI"
    # Define biospecimen projects to exclude - MUST use exact project names
    biospec_exclude = ['project_9000', 'project_222', 'project_196']

    # Load all data as a dictionary
    all_data_dict = DataLoader.load(
        data_path=data_dir,
        merge_output=False,
        biospec_exclude=biospec_exclude
    )

    assert BIOSPECIMEN in all_data_dict
    # Check if biospecimen data is loaded and verify exclusions
    biospec_data = all_data_dict[BIOSPECIMEN]
    assert isinstance(biospec_data, pd.DataFrame)
    if 'PROJ_ID' in biospec_data.columns:
        # Verify excluded projects are not present
        for project in biospec_exclude:
            project_id = project.replace('project_', '')
            assert project_id not in biospec_data['PROJ_ID'].values

def test_data_preprocessor():
    # Break out these steps manually
    data_dict = DataLoader.load(clean_data=False)
    clean_dict = DataPreprocessor.clean(data_dict)
    # Test cleaning runs on full dict and returns another dict
    assert isinstance(clean_dict, dict), "Expected a dictionary from cleaning."
    for modality in ALL_MODALITIES:
        assert modality in clean_dict

if __name__ == "__main__":
    test_data_loader()
