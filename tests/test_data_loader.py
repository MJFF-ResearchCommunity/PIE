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
    data = DataLoader.load() # All modalities in the standard location
    assert isinstance(data, dict), "Expected a dictionary from the loader."
    for modality in DataLoader.ALL_MODALITIES:
        assert modality in data

    # Medical history-specific data
    mh = DataLoader.MEDICAL_HISTORY
    assert isinstance(data[mh], dict)
    assert "Concomitant_Medication" in data[mh]
    assert isinstance(data[mh]["Concomitant_Medication"], pd.DataFrame)
    assert "CMTRT" in data[mh]["Concomitant_Medication"].columns

    assert "Features_of_Parkinsonism" in data[mh]
    assert isinstance(data[mh]["Features_of_Parkinsonism"], pd.DataFrame)
    assert "FEATBRADY" in data[mh]["Features_of_Parkinsonism"].columns

    assert "Vital_Signs" in data[mh]
    assert isinstance(data[mh]["Vital_Signs"], pd.DataFrame)
    assert "TEMPC" in data[mh]["Vital_Signs"].columns

    assert "General_Physical_Exam" in data[mh]
    assert isinstance(data[mh]["General_Physical_Exam"], pd.DataFrame)
    assert "PECAT" in data[mh]["General_Physical_Exam"].columns

# Test specific modality loaders
def test_load_ppmi_medical_history():
    data = load_ppmi_medical_history("./PPMI/Medical_History")
    assert "CMTRT" in data["Concomitant_Medication"].columns
    assert "CMINDC" in data["Concomitant_Medication"].columns
    assert "CMDOSE" in data["Concomitant_Medication"].columns

    assert "FEATBRADY" in data["Features_of_Parkinsonism"].columns
    assert "TEMPC" in data["Vital_Signs"].columns
    assert "PECAT" in data["General_Physical_Exam"].columns
