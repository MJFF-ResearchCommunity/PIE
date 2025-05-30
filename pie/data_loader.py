"""
data_loader.py

High-level data loading interface that provides a unified way to load data
from different modalities and sources.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import gc # Make sure gc is imported at the top

# Import constants
from pie.constants import (
    SUBJECT_CHARACTERISTICS, MEDICAL_HISTORY, MOTOR_ASSESSMENTS, 
    NON_MOTOR_ASSESSMENTS, BIOSPECIMEN, ALL_MODALITIES, FOLDER_PATHS
)

try:
    # When imported as a module
    from .biospecimen_loader import load_biospecimen_data, merge_biospecimen_data
    from .sub_char_loader import load_ppmi_subject_characteristics
    from .med_hist_loader import load_ppmi_medical_history
    from .motor_loader import load_ppmi_motor_assessments
    from .non_motor_loader import load_ppmi_non_motor_assessments
    from .data_preprocessor import DataPreprocessor
except ImportError:
    # When run directly as a script
    from pie.biospecimen_loader import load_biospecimen_data, merge_biospecimen_data
    from pie.sub_char_loader import load_ppmi_subject_characteristics
    from pie.med_hist_loader import load_ppmi_medical_history
    from pie.motor_loader import load_ppmi_motor_assessments
    from pie.non_motor_loader import load_ppmi_non_motor_assessments
    from pie.data_preprocessor import DataPreprocessor

logger = logging.getLogger(f"PIE.{__name__}")

class DataLoader:
    """
    Main DataLoader class that coordinates the loading of different data types.
    
    This class provides a unified interface to load data from different modalities
    and sources, with options to specify which modalities to load.
    """
    
    # Use the constants directly
    SUBJECT_CHARACTERISTICS = SUBJECT_CHARACTERISTICS
    MEDICAL_HISTORY = MEDICAL_HISTORY
    MOTOR_ASSESSMENTS = MOTOR_ASSESSMENTS
    NON_MOTOR_ASSESSMENTS = NON_MOTOR_ASSESSMENTS
    BIOSPECIMEN = BIOSPECIMEN
    
    # Use the ALL_MODALITIES constant
    ALL_MODALITIES = ALL_MODALITIES
    
    # Use the FOLDER_PATHS constant
    FOLDER_PATHS = FOLDER_PATHS
    
    def __init__(self):
        """Initialize the DataLoader."""
        pass
    
    @staticmethod
    def load(
        data_path: str = "./PPMI",
        modalities: Optional[List[str]] = None,
        source: str = "PPMI",
        merge_output: bool = False,
        output_file: str = None,
        clean_data: bool = True,
        biospec_exclude: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], pd.DataFrame]:
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
                        - "biospecimen"
            source: Data source identifier (e.g., "PPMI")
            merge_output: If True, returns a single DataFrame with all modalities merged on PATNO and EVENT_ID.
                         If False, returns a dictionary of modalities.
            output_file: Path to save the output (merged DataFrame or individual files). If None, no file is saved.
            clean_data: If True, applies appropriate data cleaning functions to each modality
            biospec_exclude: List of biospecimen data sources to exclude (e.g., ['project_9000', 'project_222'])
                            Passed to merge_biospecimen_data() as the exclude parameter
            
        Returns:
            If merge_output is False: Dictionary containing loaded data for each requested modality
            If merge_output is True: A single DataFrame with all modalities merged on PATNO and EVENT_ID
        """
        # If no modalities specified, load all
        if modalities is None:
            modalities = DataLoader.ALL_MODALITIES
        
        # Initialize biospec_exclude if None
        if biospec_exclude is None:
            biospec_exclude = []
        
        # Validate modalities
        for modality in modalities:
            if modality not in DataLoader.ALL_MODALITIES:
                logger.warning(f"Unknown modality: {modality}. Will be skipped.")
        
        # Filter to valid modalities
        valid_modalities = [m for m in modalities if m in DataLoader.ALL_MODALITIES]
        
        # Initialize results dictionary
        data_dict = {}
        
        # Track all pairs of PATNO/EVENT_ID for potential merging
        all_pairs = set()
        
        # *** ADDED LOGGING FOR EXCLUSION ***
        if BIOSPECIMEN in valid_modalities:
            logger.info(f"Biospecimen modality requested. Exclusion list: {biospec_exclude}")
        # *** END ADDED LOGGING ***
        
        # Load each requested modality
        for modality in valid_modalities:
            logger.info(f"Loading {modality} data...")
            folder_path = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
            
            if not os.path.exists(folder_path):
                logger.warning(f"Directory not found: {folder_path}")
                if modality == DataLoader.MEDICAL_HISTORY:
                    data_dict[modality] = {}
                else:
                    data_dict[modality] = pd.DataFrame()
                continue
            
            if modality == DataLoader.SUBJECT_CHARACTERISTICS:
                data_dict[modality] = load_ppmi_subject_characteristics(folder_path)
                logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                
                # Collect PATNO/EVENT_ID pairs for potential merging
                if not data_dict[modality].empty and "PATNO" in data_dict[modality].columns and "EVENT_ID" in data_dict[modality].columns:
                    for _, row in data_dict[modality][["PATNO", "EVENT_ID"]].iterrows():
                        all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
            
            elif modality == DataLoader.MEDICAL_HISTORY:
                med_hist_data = load_ppmi_medical_history(folder_path)
                
                # Clean the medical history data if requested
                if clean_data and med_hist_data:
                    med_hist_data = DataPreprocessor.clean_medical_history(med_hist_data)
                
                data_dict[modality] = med_hist_data
                logger.info(f"Loaded {len(data_dict[modality])} {modality} tables")
                
                # For medical history, collect PATNO/EVENT_ID pairs from each table
                for table_name, df in med_hist_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty and "PATNO" in df.columns and "EVENT_ID" in df.columns:
                        for _, row in df[["PATNO", "EVENT_ID"]].iterrows():
                            all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
            
            elif modality == DataLoader.MOTOR_ASSESSMENTS:
                data_dict[modality] = load_ppmi_motor_assessments(folder_path)
                logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                
                # Collect PATNO/EVENT_ID pairs for potential merging
                if not data_dict[modality].empty and "PATNO" in data_dict[modality].columns and "EVENT_ID" in data_dict[modality].columns:
                    for _, row in data_dict[modality][["PATNO", "EVENT_ID"]].iterrows():
                        all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
            
            elif modality == DataLoader.NON_MOTOR_ASSESSMENTS:
                data_dict[modality] = load_ppmi_non_motor_assessments(folder_path)
                logger.info(f"Loaded {modality} with {len(data_dict[modality])} rows")
                
                # Collect PATNO/EVENT_ID pairs for potential merging
                if not data_dict[modality].empty and "PATNO" in data_dict[modality].columns and "EVENT_ID" in data_dict[modality].columns:
                    for _, row in data_dict[modality][["PATNO", "EVENT_ID"]].iterrows():
                        all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
            
            elif modality == DataLoader.BIOSPECIMEN:
                # Pass the biospec_exclude parameter to load_biospecimen_data
                # Log the call to be sure
                logger.debug(f"Calling load_biospecimen_data with exclude={biospec_exclude}")
                biospec_data = load_biospecimen_data(data_path, source, exclude=biospec_exclude)
                
                if merge_output:
                    # If we will merge everything later, just store the dictionary
                    data_dict[modality] = biospec_data
                else:
                    # If we're not merging everything, merge just the biospecimen data
                    # The exclude parameter is now redundant here since we already excluded during loading,
                    # but keeping it for completeness and backward compatibility
                    data_dict[modality] = merge_biospecimen_data(
                        biospec_data, 
                        merge_all=True,
                        output_filename=None,
                        exclude=biospec_exclude
                    )
                    
                    # Collect PATNO/EVENT_ID pairs for potential merging
                    if isinstance(data_dict[modality], pd.DataFrame) and not data_dict[modality].empty and "PATNO" in data_dict[modality].columns and "EVENT_ID" in data_dict[modality].columns:
                        for _, row in data_dict[modality][["PATNO", "EVENT_ID"]].iterrows():
                            all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
                
                logger.info(f"Loaded {modality} data")
            
            # After loading each modality into data_dict[modality] or handling medical history:
            
            current_data = data_dict.get(modality)
            
            # --- OPTIMIZED all_pairs COLLECTION ---
            if isinstance(current_data, pd.DataFrame):
                if not current_data.empty and "PATNO" in current_data.columns and "EVENT_ID" in current_data.columns:
                    unique_pairs_df = current_data[["PATNO", "EVENT_ID"]].drop_duplicates()
                    # Convert to tuples of strings for consistent hashing
                    new_pairs = set(map(lambda x: (str(x[0]), x[1]), unique_pairs_df.itertuples(index=False, name=None)))
                    all_pairs.update(new_pairs)
                    del unique_pairs_df # Free memory
                    del new_pairs
            elif isinstance(current_data, dict) and modality == DataLoader.MEDICAL_HISTORY:
                for table_name, df in current_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty and "PATNO" in df.columns and "EVENT_ID" in df.columns:
                        unique_pairs_df = df[["PATNO", "EVENT_ID"]].drop_duplicates()
                        new_pairs = set(map(lambda x: (str(x[0]), x[1]), unique_pairs_df.itertuples(index=False, name=None)))
                        all_pairs.update(new_pairs)
                        del unique_pairs_df # Free memory
                        del new_pairs
            # --- END OPTIMIZED all_pairs COLLECTION ---
        
        # Handle output for dictionary or merged DataFrame
        if merge_output:
            logger.info("Loading and incrementally merging modalities into a single DataFrame")
            
            # Use a temporary dictionary to hold data just before merging
            loaded_data = {}
            all_pairs = set() # Still collect pairs to build the initial base_df

            # First pass: Load data and collect all_pairs
            for modality in valid_modalities:
                logger.info(f"Loading {modality} data...")
                folder_path = os.path.join(data_path, DataLoader.FOLDER_PATHS[modality])
                
                if not os.path.exists(folder_path):
                    logger.warning(f"Directory not found: {folder_path}")
                    if modality == DataLoader.MEDICAL_HISTORY:
                        loaded_data[modality] = {}
                    # Store empty DF for non-found non-medical history to avoid key errors later
                    elif modality != DataLoader.BIOSPECIMEN: 
                         loaded_data[modality] = pd.DataFrame()
                    continue

                # Load data based on modality
                data_to_process = None
                if modality == DataLoader.SUBJECT_CHARACTERISTICS:
                    data_to_process = load_ppmi_subject_characteristics(folder_path)
                    loaded_data[modality] = data_to_process
                elif modality == DataLoader.MEDICAL_HISTORY:
                    med_hist_data = load_ppmi_medical_history(folder_path)
                    if clean_data and med_hist_data:
                        med_hist_data = DataPreprocessor.clean_medical_history(med_hist_data)
                    loaded_data[modality] = med_hist_data
                    data_to_process = med_hist_data # Process this dict below
                elif modality == DataLoader.MOTOR_ASSESSMENTS:
                    data_to_process = load_ppmi_motor_assessments(folder_path)
                    loaded_data[modality] = data_to_process
                elif modality == DataLoader.NON_MOTOR_ASSESSMENTS:
                    data_to_process = load_ppmi_non_motor_assessments(folder_path)
                    loaded_data[modality] = data_to_process
                elif modality == DataLoader.BIOSPECIMEN:
                    logger.debug(f"Calling load_biospecimen_data with exclude={biospec_exclude}")
                    data_to_process = load_biospecimen_data(data_path, source, exclude=biospec_exclude)
                    loaded_data[modality] = data_to_process
                    # Biospecimen often returns a dict, handle below

                logger.info(f"Loaded {modality}")

                # Collect PATNO/EVENT_ID pairs
                if isinstance(data_to_process, pd.DataFrame):
                    if not data_to_process.empty and "PATNO" in data_to_process.columns and "EVENT_ID" in data_to_process.columns:
                        unique_pairs_df = data_to_process[["PATNO", "EVENT_ID"]].drop_duplicates()
                        new_pairs = set(map(lambda x: (str(x[0]), x[1]), unique_pairs_df.itertuples(index=False, name=None)))
                        all_pairs.update(new_pairs)
                        del unique_pairs_df, new_pairs
                        gc.collect()
                elif isinstance(data_to_process, dict): # Handles Medical History and Biospecimen (if not merged yet)
                     for key, df_or_dict in data_to_process.items():
                         # Handle nested dicts for biospecimen if needed
                         if isinstance(df_or_dict, dict): 
                              for sub_key, df in df_or_dict.items():
                                   if isinstance(df, pd.DataFrame) and not df.empty and "PATNO" in df.columns and "EVENT_ID" in df.columns:
                                        unique_pairs_df = df[["PATNO", "EVENT_ID"]].drop_duplicates()
                                        new_pairs = set(map(lambda x: (str(x[0]), x[1]), unique_pairs_df.itertuples(index=False, name=None)))
                                        all_pairs.update(new_pairs)
                                        del unique_pairs_df, new_pairs
                                        gc.collect()
                         elif isinstance(df_or_dict, pd.DataFrame):
                             df = df_or_dict
                             if not df.empty and "PATNO" in df.columns and "EVENT_ID" in df.columns:
                                 unique_pairs_df = df[["PATNO", "EVENT_ID"]].drop_duplicates()
                                 new_pairs = set(map(lambda x: (str(x[0]), x[1]), unique_pairs_df.itertuples(index=False, name=None)))
                                 all_pairs.update(new_pairs)
                                 del unique_pairs_df, new_pairs
                                 gc.collect()
            
            # Create the base DataFrame
            if not all_pairs:
                 logger.warning("No PATNO/EVENT_ID pairs found across any modalities. Returning empty DataFrame.")
                 return pd.DataFrame()
                 
            base_df = pd.DataFrame(list(all_pairs), columns=["PATNO", "EVENT_ID"])
            # Optional: Convert types for efficiency before merging
            # base_df['PATNO'] = base_df['PATNO'].astype('category') 
            # base_df['EVENT_ID'] = base_df['EVENT_ID'].astype('category')
            logger.info(f"Created base DataFrame with {len(base_df)} unique PATNO/EVENT_ID pairs")
            merged_df = base_df.copy()
            del base_df, all_pairs # Free memory
            gc.collect()

            # Second pass: Merge data incrementally
            for modality in valid_modalities:
                data = loaded_data.get(modality)
                if data is None: # Should not happen if we store empty DFs above
                     logger.warning(f"No data loaded for modality {modality}, skipping merge.")
                     continue

                logger.info(f"Merging {modality} data...")
                
                if modality == DataLoader.MEDICAL_HISTORY:
                    for table_name, df in data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty and "PATNO" in df.columns and "EVENT_ID" in df.columns:
                            df = df.copy()
                            df["PATNO"] = df["PATNO"].astype(str)
                            duplicate_cols = [col for col in df.columns if col in merged_df.columns and col not in ["PATNO", "EVENT_ID"]]
                            if duplicate_cols:
                                df.rename(columns={col: f"{table_name}_{col}" for col in duplicate_cols}, inplace=True)
                            
                            merged_df = pd.merge(merged_df, df, on=["PATNO", "EVENT_ID"], how="left")
                            logger.info(f"Merged {table_name} table from {modality}")
                            # del df # Free memory for the specific table
                            gc.collect()

                elif modality == DataLoader.BIOSPECIMEN:
                     # If biospec data is a dict, merge it first
                     if isinstance(data, dict):
                          biospec_merged = merge_biospecimen_data(
                              data, 
                              merge_all=True,
                              output_filename=None,
                              exclude=biospec_exclude # Redundant but harmless
                          )
                     elif isinstance(data, pd.DataFrame):
                          # This case might happen if load_biospecimen_data is modified to return a single DF
                          biospec_merged = data 
                     else:
                          logger.warning(f"Unexpected data type for biospecimen: {type(data)}. Skipping merge.")
                          biospec_merged = pd.DataFrame()

                     if not biospec_merged.empty and "PATNO" in biospec_merged.columns and "EVENT_ID" in biospec_merged.columns:
                         biospec_merged = biospec_merged.copy()
                         biospec_merged["PATNO"] = biospec_merged["PATNO"].astype(str)
                         duplicate_cols = [col for col in biospec_merged.columns if col in merged_df.columns and col not in ["PATNO", "EVENT_ID"]]
                         if duplicate_cols:
                             biospec_merged.rename(columns={col: f"{modality}_{col}" for col in duplicate_cols}, inplace=True)
                             
                         merged_df = pd.merge(merged_df, biospec_merged, on=["PATNO", "EVENT_ID"], how="left")
                         logger.info(f"Merged {modality} data")
                         del biospec_merged # Free memory
                         gc.collect()

                elif isinstance(data, pd.DataFrame): # Handle other modalities
                    if not data.empty and "PATNO" in data.columns and "EVENT_ID" in data.columns:
                        data_copy = data.copy()
                        data_copy["PATNO"] = data_copy["PATNO"].astype(str)
                        duplicate_cols = [col for col in data_copy.columns if col in merged_df.columns and col not in ["PATNO", "EVENT_ID"]]
                        if duplicate_cols:
                            data_copy.rename(columns={col: f"{modality}_{col}" for col in duplicate_cols}, inplace=True)
                        
                        merged_df = pd.merge(merged_df, data_copy, on=["PATNO", "EVENT_ID"], how="left")
                        logger.info(f"Merged {modality} data")
                        del data_copy # Free memory
                        gc.collect()
                
                # Remove the modality from the temporary dictionary after merging
                if modality in loaded_data:
                     del loaded_data[modality]
                     gc.collect()


            # Save to output file if specified
            if output_file:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                logger.info(f"Saving merged data to {output_file}")
                merged_df.to_csv(output_file, index=False)
            
            logger.info(f"Final merged DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns")
            return merged_df
        
        else:
            # For dictionary output, save individual files if output_file is specified
            if output_file:
                output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save each modality to its own file
                for modality, data in data_dict.items():
                    if modality == DataLoader.MEDICAL_HISTORY:
                        # Create subdirectory for medical history files
                        med_hist_dir = os.path.join(output_dir, "medical_history")
                        if not os.path.exists(med_hist_dir):
                            os.makedirs(med_hist_dir)
                        
                        # Save each table to its own file
                        for table_name, df in data.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                file_path = os.path.join(med_hist_dir, f"{table_name}.csv")
                                df.to_csv(file_path, index=False)
                                logger.info(f"Saved {table_name} to {file_path}")
                    
                    elif isinstance(data, pd.DataFrame) and not data.empty:
                        file_path = os.path.join(output_dir, f"{modality}.csv")
                        data.to_csv(file_path, index=False)
                        logger.info(f"Saved {modality} to {file_path}")
                    
                    elif modality == DataLoader.BIOSPECIMEN:
                        # Create subdirectory for biospecimen files
                        biospec_dir = os.path.join(output_dir, "biospecimen")
                        if not os.path.exists(biospec_dir):
                            os.makedirs(biospec_dir)
                        
                        # Save each source to its own file
                        merge_biospecimen_data(
                            data,
                            merge_all=False,
                            output_dir=biospec_dir,
                            exclude=biospec_exclude  # Pass the biospec_exclude parameter
                        )
                        logger.info(f"Saved biospecimen data to {biospec_dir}")
            
            return data_dict

    def load_ppmi_data(
        self,
        data_dir: str = "./PPMI", 
        merge_output: bool = False, 
        output_file: str = None
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Legacy function to load all PPMI data from the specified directory.
        
        This function is maintained for backward compatibility.
        It calls the DataLoader.load() method with all modalities.
        
        Args:
            data_dir: Path to the directory containing PPMI data folders
            merge_output: If True, returns a single DataFrame with all modalities merged
            output_file: Path to save the output (if None, no file is saved)
            
        Returns:
            If merge_output is False: A dictionary with loaded data for all modalities
            If merge_output is True: A single DataFrame with all modalities merged
        """
        return self.load(
            data_path=data_dir,
            modalities=self.ALL_MODALITIES,
            merge_output=merge_output,
            output_file=output_file
        )


# def main():
#     """
#     Example usage of the DataLoader class.
#     """
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # Example 1: Load all data as a merged DataFrame
#     logger.info("\nExample 2: Loading all data as a merged DataFrame")
#     all_data_merged = DataLoader.load(merge_output=True)

#     # Example 2: Load all data as a dictionary
#     logger.info("Example 1: Loading all data as a dictionary")
#     all_data_dict = DataLoader.load(merge_output=False)
    

#     # Example 3: Load only subject characteristics and motor assessments
#     logger.info("\nExample 3: Loading specific modalities")
#     specific_data = DataLoader.load(
#         modalities=[
#             DataLoader.SUBJECT_CHARACTERISTICS,
#             DataLoader.MOTOR_ASSESSMENTS
#         ]
#     )
    
#     # Example 4: Load and save merged data to a file
#     logger.info("\nExample 4: Loading and saving merged data to a file")
#     DataLoader.load(
#         modalities=[
#             DataLoader.SUBJECT_CHARACTERISTICS,
#             DataLoader.MOTOR_ASSESSMENTS,
#             DataLoader.NON_MOTOR_ASSESSMENTS
#         ],
#         merge_output=True,
#         output_file="./output/merged_data.csv"
#     )
    
#     # Print summary of loaded data from Example 1
#     logger.info("\nSummary of loaded data (Example 1):")
    
#     for modality, data in all_data_dict.items():
#         if modality == DataLoader.MEDICAL_HISTORY:
#             logger.info(f"Medical history tables: {len(data)}")
#             for table_name, df in data.items():
#                 if isinstance(df, pd.DataFrame) and not df.empty:
#                     logger.info(f"  - {table_name}: {len(df)} rows")
#         elif isinstance(data, pd.DataFrame) and not data.empty:
#             logger.info(f"{modality}: {len(data)} rows, {len(data.columns)} columns")
#         else:
#             logger.info(f"{modality}: No data loaded or empty")
    
#     # Print summary of merged data from Example 2
#     if isinstance(all_data_merged, pd.DataFrame) and not all_data_merged.empty:
#         logger.info(f"\nMerged data: {len(all_data_merged)} rows, {len(all_data_merged.columns)} columns")
#     else:
#         logger.info("\nMerged data: No data loaded or empty")

# if __name__ == "__main__":
#     main()