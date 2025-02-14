"""
data_loader.py

High-level data loading interface that relies on specialized loaders.
"""

from .clinical_loader import load_clinical_data
from .biologics_loader import load_biologics_data
from .imaging_loader import load_imaging_data
from .wearables_loader import load_wearables_data
from .exams_loader import load_exams_data


class DataLoader:
    """
    Main DataLoader class that coordinates the loading of different data types.
    """

    @staticmethod
    def load(data_path: str, source: str = "PPMI"):
        """
        Load data using specialized loaders. For now, only stubs are provided.

        :param data_path: Path to data directory.
        :param source: The data source (e.g., PPMI).
        :return: Dictionary of loaded data keyed by data type.
        """
        data_dict = {}
        data_dict['clinical'] = load_clinical_data(data_path, source)
        data_dict['biologics'] = load_biologics_data(data_path, source)
        data_dict['imaging'] = load_imaging_data(data_path, source)
        data_dict['wearables'] = load_wearables_data(data_path, source)
        data_dict['exams'] = load_exams_data(data_path, source)
        return data_dict 