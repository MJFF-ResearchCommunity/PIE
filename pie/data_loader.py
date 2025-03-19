"""
data_loader.py

High-level data loading interface that provides a unified way to load data
from different modalities and sources.
"""

# Fix imports to work both as a module and when run directly
try:
    # When imported as a module
    from .clinical_loader import load_clinical_data
    from .biologics_loader import load_biologics_data
    from .imaging_loader import load_imaging_data
    from .wearables_loader import load_wearables_data
    from .exams_loader import load_exams_data
    from .sub_char_loader import load_ppmi_subject_characteristics
    from .med_hist_loader import load_ppmi_medical_history
    from .motor_loader import load_ppmi_motor_assessments
    from .non_motor_loader import load_ppmi_non_motor_assessments
except ImportError:
    # When run directly as a script
    from pie.clinical_loader import load_clinical_data
    from pie.biologics_loader import load_biologics_data
    from pie.imaging_loader import load_imaging_data
    from pie.wearables_loader import load_wearables_data
    from pie.exams_loader import load_exams_data
    from pie.sub_char_loader import load_ppmi_subject_characteristics
    from pie.med_hist_loader import load_ppmi_medical_history
    from pie.motor_loader import load_ppmi_motor_assessments
    from pie.non_motor_loader import load_ppmi_non_motor_assessments

import logging
import os
from typing import Dict, Any, List, Union, Optional

import pandas as pd

logger = logging.getLogger(f"PIE.{__name__}")

class DataLoader:
    """
    Main DataLoader class that coordinates the loading of different data types.
    
    This class provides a unified interface to load data from different modalities
    and sources, with options to specify which modalities to load.
    """
    
    # Define constants for modality names to ensure consistency
    SUBJECT_CHARACTERISTICS = "subject_characteristics"
    MEDICAL_HISTORY = "medical_history"
    MOTOR_ASSESSMENTS = "motor_assessments"
    NON_MOTOR_ASSESSMENTS = "non_motor_assessments"
    CLINICAL = "clinical"
    BIOLOGICS = "biologics"
    IMAGING = "imaging"
    WEARABLES = "wearables"
    EXAMS = "exams"
    
    # Define all available modalities
    ALL_MODALITIES = [
        SUBJECT_CHARACTERISTICS,
        MEDICAL_HISTORY,
        MOTOR_ASSESSMENTS,
        NON_MOTOR_ASSESSMENTS,
        CLINICAL,
        BIOLOGICS,
        IMAGING,
        WEARABLES,
        EXAMS
    ]
    
    # Define folder paths for each modality
    FOLDER_PATHS = {
        SUBJECT_CHARACTERISTICS: "_Subject_Characteristics",
        MEDICAL_HISTORY: "Medical_History",
        MOTOR_ASSESSMENTS: "Motor___MDS-UPDRS",
        NON_MOTOR_ASSESSMENTS: "Non-motor_Assessments",
        # Other modalities might have different folder structures
    }
    
    def __init__(self):
        """Initialize the DataLoader."""
        pass
    
    @staticmethod
    def load(
        data_path: str = "./PPMI",
        modalities: Optional[List[str]] = None,
        source: str = "PPMI"
    ) -> Dict[str, Any]:
        """
        Load data from specified modalities.
        
        Args:
            data_path: Path to the data directory
            modalities: List of modalities to load. If None, loads all available modalities.
                        Valid options are:
                        - "subject_characteristics"
                        - "medical_history"
                        - "motor_assessments"
                        - "non_motor_assessments"
                        - "clinical"
                        - "biologics"
                        - "imaging"
                        - "wearables"
                        - "exams"
            source: Data source identifier (e.g., "PPMI")
            
        Returns:
            Dictionary containing loaded data for each requested modality
        """
        # If no modalities specified, load all
        if modalities is None:
            modalities = DataLoader.ALL_MODALITIES
        
        # Validate modalities
        for modality in modalities:
            if modality not in DataLoader.ALL_MODALITIES:
                logger.warning(f"Unknown modality: {modality}. Will be skipped.")
        
        # Filter to valid modalities
        valid_modalities = [m for m in modalities if m in DataLoader.ALL_MODALITIES]
        
        # Initialize results dictionary
        data_dict = {}
        
        # Load each requested modality
        for modality in valid_modalities:
            logger.info(f"Loading {modality} data...")
            
            if modality == DataLoader.SUBJECT_CHARACTERISTICS:
                folder = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
                if os.path.exists(folder):
                    data_dict[modality] = load_ppmi_subject_characteristics(folder)
                    logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                else:
                    logger.warning(f"Directory not found: {folder}")
                    data_dict[modality] = pd.DataFrame()
            
            elif modality == DataLoader.MEDICAL_HISTORY:
                folder = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
                if os.path.exists(folder):
                    data_dict[modality] = load_ppmi_medical_history(folder)
                    logger.info(f"Loaded {len(data_dict[modality])} {modality} tables")
                else:
                    logger.warning(f"Directory not found: {folder}")
                    data_dict[modality] = {}
            
            elif modality == DataLoader.MOTOR_ASSESSMENTS:
                folder = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
                if os.path.exists(folder):
                    data_dict[modality] = load_ppmi_motor_assessments(folder)
                    logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                else:
                    logger.warning(f"Directory not found: {folder}")
                    data_dict[modality] = pd.DataFrame()
            
            elif modality == DataLoader.NON_MOTOR_ASSESSMENTS:
                folder = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
                if os.path.exists(folder):
                    data_dict[modality] = load_ppmi_non_motor_assessments(folder)
                    logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                else:
                    logger.warning(f"Directory not found: {folder}")
                    data_dict[modality] = pd.DataFrame()
            
            elif modality == DataLoader.CLINICAL:
                data_dict[modality] = load_clinical_data(data_path, source)
                logger.info(f"Loaded {modality} data")
            
            elif modality == DataLoader.BIOLOGICS:
                data_dict[modality] = load_biologics_data(data_path, source)
                logger.info(f"Loaded {modality} data")
            
            elif modality == DataLoader.IMAGING:
                data_dict[modality] = load_imaging_data(data_path, source)
                logger.info(f"Loaded {modality} data")
            
            elif modality == DataLoader.WEARABLES:
                data_dict[modality] = load_wearables_data(data_path, source)
                logger.info(f"Loaded {modality} data")
            
            elif modality == DataLoader.EXAMS:
                data_dict[modality] = load_exams_data(data_path, source)
                logger.info(f"Loaded {modality} data")
        
        return data_dict

def load_ppmi_data(data_dir: str = "./PPMI") -> Dict[str, Any]:
    """
    Legacy function to load all PPMI data from the specified directory.
    
    This function is maintained for backward compatibility.
    It calls the DataLoader.load() method with all modalities.
    
    Args:
        data_dir: Path to the directory containing PPMI data folders
        
    Returns:
        A dictionary with loaded data for all modalities
    """
    return DataLoader.load(data_dir)


# def main():
#     """
#     Example usage of the DataLoader class.
#     """
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # Example 1: Load all data
#     logger.info("Example 1: Loading all data")
#     all_data = DataLoader.load()
    
#     # Example 2: Load only subject characteristics and motor assessments
#     logger.info("\nExample 2: Loading specific modalities")
#     specific_data = DataLoader.load(
#         modalities=[
#             DataLoader.SUBJECT_CHARACTERISTICS,
#             DataLoader.MOTOR_ASSESSMENTS
#         ]
#     )
    
#     # Example 3: Load from a different path
#     logger.info("\nExample 3: Loading from a custom path")
#     custom_path_data = DataLoader.load(
#         data_path="./custom/data/path",
#         modalities=[DataLoader.MEDICAL_HISTORY]
#     )
    
#     # Print summary of loaded data from Example 2
#     logger.info("\nSummary of loaded data (Example 2):")
    
#     if DataLoader.SUBJECT_CHARACTERISTICS in specific_data:
#         df = specific_data[DataLoader.SUBJECT_CHARACTERISTICS]
#         if not df.empty:
#             logger.info(f"Subject characteristics: {len(df)} rows")
#         else:
#             logger.info("Subject characteristics: No data loaded")
    
#     if DataLoader.MOTOR_ASSESSMENTS in specific_data:
#         df = specific_data[DataLoader.MOTOR_ASSESSMENTS]
#         if not df.empty:
#             logger.info(f"Motor assessments: {len(df)} rows")
#         else:
#             logger.info("Motor assessments: No data loaded")

# if __name__ == "__main__":
#     main()