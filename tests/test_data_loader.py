"""
test_data_loader.py

Placeholder tests for the data loader module.
"""

import pytest
import pandas as pd
from pie.data_loader import DataLoader

from pie.clinical_loader import load_clinical_data
from pie.med_hist_loader import load_ppmi_medical_history

def test_data_loader():
    data = DataLoader.load("./PPMI", "PPMI")
    assert isinstance(data, dict), "Expected a dictionary from the loader."
    for modality in ["clinical", "biologics", "imaging", "wearables", "exams"]:
        assert modality in data

    # TODO: Fill out this list as specific files and sub-modalities are added
    assert "med_hist" in data["clinical"]

    # Medical history-specific data
    assert isinstance(data["clinical"]["med_hist"], dict)
    assert "Concomitant_Medication" in data["clinical"]["med_hist"]
    assert isinstance(data["clinical"]["med_hist"]["Concomitant_Medication"], pd.DataFrame)
    assert "CMTRT" in data["clinical"]["med_hist"]["Concomitant_Medication"].columns

# Test specific modality loaders
def test_load_clinical():
    data = load_clinical_data("./PPMI", "PPMI")
    assert isinstance(data, dict)
    assert "med_hist" in data
    assert isinstance(data["med_hist"], dict)
    assert "Concomitant_Medication" in data["med_hist"]
    assert "CMTRT" in data["med_hist"]["Concomitant_Medication"].columns

    # Load directly as medical history
    data = load_ppmi_medical_history("./PPMI/Medical History")
    assert "CMTRT" in data["Concomitant_Medication"].columns
    assert "CMINDC" in data["Concomitant_Medication"].columns
    assert "CMDOSE" in data["Concomitant_Medication"].columns
