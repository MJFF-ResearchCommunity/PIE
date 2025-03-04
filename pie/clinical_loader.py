"""
clinical_loader.py

Contains functions for loading and processing clinical data.
"""
from pie.med_hist_loader import load_ppmi_medical_history

def load_clinical_data(data_path: str, source: str):
    """
    Load clinical data from the specified path. Currently a placeholder.
    
    :param data_path: Path to the clinical data.
    :param source: The data source (e.g., PPMI).
    :return: A placeholder dictionary or DataFrame representing clinical data.
    """
    # TODO: Add other parts of clinical data as added and tested
    data_dict = {}
    data_dict["med_hist"] = load_ppmi_medical_history(data_path)
    return data_dict
