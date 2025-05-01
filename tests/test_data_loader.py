#!/usr/bin/env python3
"""
Test script for data_loader.py functionality.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the DataLoader class and constants
from pie.data_loader import DataLoader
from pie.constants import (
    SUBJECT_CHARACTERISTICS, MEDICAL_HISTORY, MOTOR_ASSESSMENTS,
    NON_MOTOR_ASSESSMENTS, BIOSPECIMEN
)

# Import the biospecimen loader directly to verify exclusions
from pie.biospecimen_loader import merge_biospecimen_data

def test_data_loader():
    """
    Test basic functionality of the data loading functions.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("PIE.test_data_loader")
    
    # Example paths - modify as needed
    data_dir = "./PPMI"  # Change this to your actual PPMI data path
    output_dir = "./output"
    
    # Define biospecimen projects to exclude - MUST use exact project names
    biospec_exclude = ['project_9000', 'project_222', 'project_196']
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # # VERIFICATION: Check if biospecimen exclusion works directly with merge_biospecimen_data
    # logger.info("\nVerification: Testing biospecimen exclusion directly")
    # try:
    #     from pie.biospecimen_loader import load_biospecimen_data
        
    #     # Load biospecimen data directly
    #     biospec_data = load_biospecimen_data(data_dir, "PPMI")
        
    #     # Log loaded biospecimen sources before exclusion
    #     logger.info(f"Available biospecimen sources before exclusion: {list(biospec_data.keys())}")
        
    #     # Check if excluded projects are in the loaded data
    #     for project in biospec_exclude:
    #         if project in biospec_data:
    #             logger.info(f"Project {project} is available in the loaded data")
    #         else:
    #             logger.warning(f"Project {project} not found in the loaded data - check project name")
        
    #     # Merge with explicit exclusion
    #     merged_biospec = merge_biospecimen_data(
    #         biospec_data,
    #         merge_all=True,
    #         output_filename=None,
    #         exclude=biospec_exclude
    #     )
        
    #     logger.info(f"After explicit exclusion, biospecimen data has {len(merged_biospec)} rows")
        
    #     # Verify excluded projects are not in the merged data
    #     # This would require examining the merged data for project identifiers
    # except Exception as e:
    #     logger.error(f"Error in verification: {str(e)}")
    
    # # Example 1: Load all data as a merged DataFrame
    # logger.info("\nExample 1: Loading all data as a merged DataFrame")
    # try:
    #     all_data_merged = DataLoader.load(
    #         data_path=data_dir,
    #         merge_output=True,
    #         biospec_exclude=biospec_exclude
    #     )
        
    #     # Print summary of merged data from Example 1
    #     if isinstance(all_data_merged, pd.DataFrame) and not all_data_merged.empty:
    #         logger.info(f"Merged data: {len(all_data_merged)} rows, {len(all_data_merged.columns)} columns")
            
    #         # Check if PROJ_ID column exists in merged data
    #         if 'PROJ_ID' in all_data_merged.columns:
    #             # Verify excluded projects are not present
    #             for project in biospec_exclude:
    #                 project_id = project.replace('project_', '')
    #                 if project_id in all_data_merged['PROJ_ID'].values:
    #                     logger.error(f"ERROR: {project} data found in merged output despite exclusion!")
    #                 else:
    #                     logger.info(f"Verified: {project} data excluded from merged output")
    #         else:
    #             logger.info("PROJ_ID column not found in merged data, cannot verify exclusions")
    #     else:
    #         logger.info("Merged data: No data loaded or empty")
    # except Exception as e:
    #     logger.error(f"Error in Example 1: {str(e)}")
    
    # Example 2: Load all data as a dictionary
    logger.info("\nExample 2: Loading all data as a dictionary")
    try:
        all_data_dict = DataLoader.load(
            data_path=data_dir,
            merge_output=False,
            biospec_exclude=biospec_exclude
        )
        
        # Check if biospecimen data is loaded and verify exclusions
        if BIOSPECIMEN in all_data_dict:
            biospec_data = all_data_dict[BIOSPECIMEN]
            
            if isinstance(biospec_data, pd.DataFrame):
                logger.info(f"Biospecimen data loaded as DataFrame with {len(biospec_data)} rows")
                
                # Check if PROJ_ID column exists in biospecimen data
                if 'PROJ_ID' in biospec_data.columns:
                    # Verify excluded projects are not present
                    for project in biospec_exclude:
                        project_id = project.replace('project_', '')
                        if project_id in biospec_data['PROJ_ID'].values:
                            logger.error(f"ERROR: {project} data found in biospecimen DataFrame despite exclusion!")
                        else:
                            logger.info(f"Verified: {project} data excluded from biospecimen DataFrame")
                else:
                    logger.info("PROJ_ID column not found in biospecimen data, cannot verify exclusions")
            
            elif isinstance(biospec_data, dict):
                # This shouldn't happen with merge_output=False, but check anyway
                logger.info(f"Biospecimen data loaded as dictionary with {len(biospec_data)} sources")
                
                # Check if excluded projects are in the dictionary
                for project in biospec_exclude:
                    if project in biospec_data:
                        logger.error(f"ERROR: {project} found in biospecimen dict despite exclusion!")
                    else:
                        logger.info(f"Verified: {project} excluded from biospecimen dict")
        
        # Print summary of loaded data
        logger.info("\nSummary of loaded data (Example 2):")
        for modality, data in all_data_dict.items():
            if modality == MEDICAL_HISTORY:
                logger.info(f"Medical history tables: {len(data)}")
                for table_name, df in data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        logger.info(f"  - {table_name}: {len(df)} rows")
            elif isinstance(data, pd.DataFrame) and not data.empty:
                logger.info(f"{modality}: {len(data)} rows, {len(data.columns)} columns")
            else:
                logger.info(f"{modality}: No data loaded or empty")
    
    except Exception as e:
        logger.error(f"Error in Example 2: {str(e)}")

if __name__ == "__main__":
    test_data_loader()
