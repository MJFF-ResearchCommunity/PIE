"""
test_data_preprocessor.py

Tests for the data preprocessor module.
"""

import pytest
import numpy as np
import pandas as pd
from pie.data_loader import DataLoader
from pie.data_preprocessor import DataPreprocessor

@pytest.fixture()
def data_dict():
    return DataLoader.load("./PPMI", "PPMI")

def test_clean(data_dict):
    clean_dict = DataPreprocessor.clean(data_dict)
    # Test cleaning runs on full dict and returns another dict
    assert isinstance(clean_dict, dict), "Expected a dictionary from cleaning."
    for modality in ["clinical", "biologics", "imaging", "wearables", "exams"]:
        assert modality in clean_dict

    # Add assertions for individual components of the clean_dict
    assert "med_hist" in clean_dict["clinical"]

# Now test the actual cleaning code
def test_clean_concomitant_meds(data_dict):
    clean_df = DataPreprocessor.clean_concomitant_meds(data_dict["clinical"]["med_hist"]["Concomitant_Medication"])
    assert "CMTRT" in clean_df.columns

    assert clean_df["CMTRT"].notnull().all() # All should have names
    assert np.issubdtype(clean_df["STARTDT"], np.datetime64) # Dates should be converted from string
    assert np.issubdtype(clean_df["STOPDT"], np.datetime64) # Dates should be converted from string

    assert clean_df["CMINDC"].notnull().all() # After cleaning, all TEXT is mapped to indication code
    counts = clean_df["CMINDC"].value_counts()
    assert counts.index[0] == 25 # The most frequent mapping is 25: Other
    assert counts.index[-1] == 21 # The least frequent mapping is 21: Drooling


@pytest.mark.skip(reason="Don't recreate every time")
def test_create_concomitant_meds(data_dict):
    DataPreprocessor.create_concomitant_meds(data_dict["clinical"]["med_hist"])
