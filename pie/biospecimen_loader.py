"""
biospecimen_loader.py

Contains functions for loading and processing biospecimen data.

Completed Data Files:
- Project_151_pQTL_in_CSF_1_of_6
- Project_151_pQTL_in_CSF_2_of_6
- Project_151_pQTL_in_CSF_3_of_6
- Project_151_pQTL_in_CSF_4_of_6
- Project_151_pQTL_in_CSF_5_of_6
- Project_151_pQTL_in_CSF_6_of_6
- Project_151_pQTL_in_CSF_1_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_2_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_3_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_4_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_5_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_6_of_7_Batch_Corrected
- Project_151_pQTL_in_CSF_7_of_7_Batch_Corrected
- Metabolomic_Analysis_of_LRRK2_PD_1_of_5
- Metabolomic_Analysis_of_LRRK2_PD_2_of_5
- Metabolomic_Analysis_of_LRRK2_PD_3_of_5
- Metabolomic_Analysis_of_LRRK2_PD_4_of_5
- Metabolomic_Analysis_of_LRRK2_PD_5_of_5
- Metabolomic_Analysis_of_LRRK2_PD__CSF
- Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_1_of_5
- Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_2_of_5
- Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_3_of_5
- Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_4_of_5
- Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_5_of_5
- PPMI_Project_9000_CSF_Cardio_NPX
- PPMI_Project_9000_CSF_INF_NPX
- PPMI_Project_9000_CSF_NEU_NPX
- PPMI_Project_9000_CSF_ONC_NPX
- PPMI_Project_9000_Plasma_Cardio_NPX
- PPMI_Project_9000_Plasma_INF_NPX
- PPMI_Project_9000_Plasma_NEURO_NPX
- PPMI_Project_9000_Plasma_ONC_NPX
- PPMI_Project_222_CSF_Cardio_NPX
- PPMI_Project_222_CSF_INF_NPX
- PPMI_Project_222_CSF_NEU_NPX
- PPMI_Project_222_CSF_ONC_NPX
- PPMI_Project_222_Plasma_Cardio_NPX
- PPMI_Project_222_Plasma_INF_NPX
- PPMI_Project_222_Plasma_NEURO_NPX
- PPMI_Project_222_Plasma_ONC_NPX
- PPMI_Project_196_CSF_Cardio_Counts
- PPMI_Project_196_CSF_INF_Counts
- PPMI_Project_196_CSF_NEURO_Counts
- PPMI_Project_196_CSF_ONC_Counts
- PPMI_Project_196_Plasma_CARDIO_Counts
- PPMI_Project_196_Plasma_INF_Counts
- PPMI_Project_196_Plasma_Neuro_Counts
- PPMI_Project_196_Plasma_ONC_Counts
- PPMI_Project_196_CSF_INF_NPX
- PPMI_Project_196_CSF_NEU_NPX
- PPMI_Project_196_CSF_ONC_NPX
- PPMI_Project_196_CSF_Cardio_NPX
- PPMI_Project_196_Plasma_INF_NPX
- PPMI_Project_196_Plasma_ONC_NPX
- PPMI_Project_196_Plasma_NEURO_NPX
- PPMI_Project_196_Plasma_Cardio_NPX
- PPMI_Project_177_Untargeted_Proteomics
- Project_214_Olink
- Current_Biospecimen_Analysis_Results
- Blood_Chemistry___Hematology


Data Files Not Requiring Individual Loading Functions:
- Clinical_Labs
- Genetic_Testing_Results
- Skin_Biopsy
- Research_Biospecimens
- Lumbar_Puncture
- Laboratory_Procedures_with_Elapsed_Times


"""

import os
import glob
import pandas as pd
import logging
import numpy as np
from typing import Union

logger = logging.getLogger(f"PIE.{__name__}")


def load_project_151_pQTL_CSF(folder_path: str, batch_corrected: bool = False) -> pd.DataFrame:
    """
    Load and process Project_151_pQTL_in_CSF data files.
    
    This function:
    1. Finds all files with the prefix "Project_151_pQTL_in_CSF"
    2. Filters based on whether we want batch-corrected files or not
    3. Renames CLINICAL_EVENT to EVENT_ID
    4. Pivots the data to create columns for each unique TESTNAME
    5. Adds "151_" prefix to each TESTNAME column
    6. Keeps only PATNO, SEX, COHORT, EVENT_ID, and the new TESTNAME columns
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
        batch_corrected: If True, use only batch-corrected files; if False, use non-batch-corrected files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix and batch_corrected parameter
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        is_project_151 = filename.startswith("Project_151_pQTL_in_CSF")
        is_batch_corrected = "Batch_Corrected" in filename
        
        if is_project_151 and is_batch_corrected == batch_corrected:
            matching_files.append(file_path)
    
    if not matching_files:
        batch_type = "batch-corrected" if batch_corrected else "non-batch-corrected"
        logger.warning(f"No {batch_type} Project_151_pQTL_in_CSF files found in {folder_path}")
        return pd.DataFrame()
    
    # Load and combine all matching files
    dfs = []
    for file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dfs:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if required columns exist
    required_columns = ["PATNO", "TESTNAME", "TESTVALUE"]
    for col in required_columns:
        if col not in combined_df.columns:
            logger.error(f"Required column {col} not found in the data")
            return pd.DataFrame()
    
    # Keep only the columns we need
    keep_columns = ["PATNO", "EVENT_ID", "SEX", "COHORT", "TESTNAME", "TESTVALUE"]
    keep_columns = [col for col in keep_columns if col in combined_df.columns]
    combined_df = combined_df[keep_columns]
    
    # Pivot the data to create columns for each TESTNAME
    try:
        # First, make sure we have no duplicates for the same PATNO, EVENT_ID, and TESTNAME
        # If there are duplicates, keep the first occurrence
        combined_df = combined_df.drop_duplicates(subset=["PATNO", "EVENT_ID", "TESTNAME"], keep="first")
        
        # Pivot the data
        pivot_columns = ["PATNO", "EVENT_ID"]
        if "SEX" in combined_df.columns:
            pivot_columns.append("SEX")
        if "COHORT" in combined_df.columns:
            pivot_columns.append("COHORT")
        
        pivoted_df = combined_df.pivot_table(
            index=pivot_columns,
            columns="TESTNAME",
            values="TESTVALUE",
            aggfunc="first"  # In case there are still duplicates
        ).reset_index()
        
        # Rename columns to add "151_" prefix to TESTNAME columns
        # First, get the names of columns that were created from TESTNAME
        testname_columns = [col for col in pivoted_df.columns if col not in pivot_columns]
        
        # Create a dictionary for renaming
        rename_dict = {col: f"151_{col}" for col in testname_columns}
        
        # Rename the columns
        pivoted_df = pivoted_df.rename(columns=rename_dict)
        
        logger.info(f"Successfully processed Project_151_pQTL_in_CSF data: {len(pivoted_df)} rows, {len(pivoted_df.columns)} columns")
        return pivoted_df
        
    except Exception as e:
        logger.error(f"Error pivoting data: {e}")
        return pd.DataFrame()


def load_metabolomic_lrrk2(folder_path: str, include_csf: bool = True) -> pd.DataFrame:
    """
    Load and process Metabolomic_Analysis_of_LRRK2_PD data files.
    
    This function:
    1. Finds all files with the prefix "Metabolomic_Analysis_of_LRRK2"
    2. Optionally includes or excludes CSF-specific files
    3. Renames CLINICAL_EVENT to EVENT_ID
    4. Pivots the data to create columns for each unique TESTNAME
    5. Adds "LRRK2_" prefix to each TESTNAME column
    6. Keeps only PATNO, SEX, COHORT, EVENT_ID, and the new TESTNAME columns
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
        include_csf: Whether to include CSF-specific files (default: True)
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        is_metabolomic_lrrk2 = filename.startswith("Metabolomic_Analysis_of_LRRK2")
        is_csf = "_CSF" in filename
        
        # Include the file if:
        # 1. It's a regular LRRK2 file (not CSF) OR
        # 2. It's a CSF file and include_csf is True
        if is_metabolomic_lrrk2 and (not is_csf or include_csf):
            matching_files.append(file_path)
    
    if not matching_files:
        csf_status = "including CSF files" if include_csf else "excluding CSF files"
        logger.warning(f"No Metabolomic_Analysis_of_LRRK2 files found in {folder_path} ({csf_status})")
        return pd.DataFrame()
    
    # Load and combine all matching files
    dfs = []
    for file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dfs:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if required columns exist
    required_columns = ["PATNO", "TESTNAME", "TESTVALUE"]
    for col in required_columns:
        if col not in combined_df.columns:
            logger.error(f"Required column {col} not found in the data")
            return pd.DataFrame()
    
    # Keep only the columns we need
    keep_columns = ["PATNO", "EVENT_ID", "SEX", "COHORT", "TESTNAME", "TESTVALUE"]
    keep_columns = [col for col in keep_columns if col in combined_df.columns]
    combined_df = combined_df[keep_columns]
    
    # Pivot the data to create columns for each TESTNAME
    try:
        # First, make sure we have no duplicates for the same PATNO, EVENT_ID, and TESTNAME
        # If there are duplicates, keep the first occurrence
        combined_df = combined_df.drop_duplicates(subset=["PATNO", "EVENT_ID", "TESTNAME"], keep="first")
        
        # Pivot the data
        pivot_columns = ["PATNO", "EVENT_ID"]
        if "SEX" in combined_df.columns:
            pivot_columns.append("SEX")
        if "COHORT" in combined_df.columns:
            pivot_columns.append("COHORT")
        
        pivoted_df = combined_df.pivot_table(
            index=pivot_columns,
            columns="TESTNAME",
            values="TESTVALUE",
            aggfunc="first"  # In case there are still duplicates
        ).reset_index()
        
        # Rename columns to add "LRRK2_" prefix to TESTNAME columns
        # First, get the names of columns that were created from TESTNAME
        testname_columns = [col for col in pivoted_df.columns if col not in pivot_columns]
        
        # Create a dictionary for renaming
        rename_dict = {col: f"LRRK2_{col}" for col in testname_columns}
        
        # Rename the columns
        pivoted_df = pivoted_df.rename(columns=rename_dict)
        
        logger.info(f"Successfully processed Metabolomic_Analysis_of_LRRK2 data: {len(pivoted_df)} rows, {len(pivoted_df.columns)} columns")
        return pivoted_df
        
    except Exception as e:
        logger.error(f"Error pivoting data: {e}")
        return pd.DataFrame()


def load_urine_proteomics(folder_path: str) -> pd.DataFrame:
    """
    Load and process Targeted___untargeted_MS-based_proteomics_of_urine_in_PD data files.
    
    This function:
    1. Finds all files with the prefix "Targeted___untargeted_MS-based_proteomics_of_urine_in_PD"
    2. Renames CLINICAL_EVENT to EVENT_ID
    3. Pivots the data to create columns for each unique TESTNAME
    4. Adds "URINE_" prefix to each TESTNAME column
    5. Keeps only PATNO, SEX, COHORT, EVENT_ID, and the new TESTNAME columns
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("Targeted___untargeted_MS-based_proteomics_of_urine_in_PD"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No Targeted___untargeted_MS-based_proteomics_of_urine_in_PD files found in {folder_path}")
        return pd.DataFrame()
    
    # Load and combine all matching files
    dfs = []
    for file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dfs:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if required columns exist
    required_columns = ["PATNO", "TESTNAME", "TESTVALUE"]
    for col in required_columns:
        if col not in combined_df.columns:
            logger.error(f"Required column {col} not found in the data")
            return pd.DataFrame()
    
    # Keep only the columns we need
    keep_columns = ["PATNO", "EVENT_ID", "SEX", "COHORT", "TESTNAME", "TESTVALUE"]
    keep_columns = [col for col in keep_columns if col in combined_df.columns]
    combined_df = combined_df[keep_columns]
    
    # Pivot the data to create columns for each TESTNAME
    try:
        # First, make sure we have no duplicates for the same PATNO, EVENT_ID, and TESTNAME
        # If there are duplicates, keep the first occurrence
        combined_df = combined_df.drop_duplicates(subset=["PATNO", "EVENT_ID", "TESTNAME"], keep="first")
        
        # Pivot the data
        pivot_columns = ["PATNO", "EVENT_ID"]
        if "SEX" in combined_df.columns:
            pivot_columns.append("SEX")
        if "COHORT" in combined_df.columns:
            pivot_columns.append("COHORT")
        
        pivoted_df = combined_df.pivot_table(
            index=pivot_columns,
            columns="TESTNAME",
            values="TESTVALUE",
            aggfunc="first"  # In case there are still duplicates
        ).reset_index()
        
        # Rename columns to add "URINE_" prefix to TESTNAME columns
        # First, get the names of columns that were created from TESTNAME
        testname_columns = [col for col in pivoted_df.columns if col not in pivot_columns]
        
        # Create a dictionary for renaming
        rename_dict = {col: f"URINE_{col}" for col in testname_columns}
        
        # Rename the columns
        pivoted_df = pivoted_df.rename(columns=rename_dict)
        
        logger.info(f"Successfully processed urine proteomics data: {len(pivoted_df)} rows, {len(pivoted_df.columns)} columns")
        return pivoted_df
        
    except Exception as e:
        logger.error(f"Error pivoting data: {e}")
        return pd.DataFrame()


def load_project_9000(folder_path: str) -> pd.DataFrame:
    """
    Load and process PPMI_Project_9000 data files.
    
    This function:
    1. Finds all files with the prefix "PPMI_Project_9000"
    2. For each unique UNIPROT-ASSAY combination, creates three columns:
       - UNIPROT_ASSAY_MISSINGFREQ
       - UNIPROT_ASSAY_LOD
       - UNIPROT_ASSAY_NPX
    3. Adds "9000_" prefix to each created column
    4. Keeps only PATNO, EVENT_ID, and the newly created columns
    5. Removes "PPMI-" prefix from PATNO values
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each UNIPROT-ASSAY metric
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("PPMI_Project_9000"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No PPMI_Project_9000 files found in {folder_path}")
        return pd.DataFrame()
    
    # First, get unique PATNO/EVENT_ID combinations to create the base dataframe
    logger.info("Creating base dataframe with unique PATNO/EVENT_ID combinations")
    patno_event_pairs = set()
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only PATNO and EVENT_ID columns to get unique combinations
            df_ids = pd.read_csv(file_path, usecols=["PATNO", "EVENT_ID"])
            
            for _, row in df_ids.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                patno_event_pairs.add((patno, row["EVENT_ID"]))
        except Exception as e:
            logger.error(f"Error reading PATNO/EVENT_ID from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {pair: {} for pair in patno_event_pairs}
    
    # Process each file separately to reduce memory usage
    for file_path in matching_files:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000  # Increased chunk size for better performance
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "UNIPROT", "ASSAY", "MISSINGFREQ", "LOD", "NPX"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Create a combined key for UNIPROT and ASSAY
                chunk["UNIPROT_ASSAY"] = chunk["UNIPROT"] + "_" + chunk["ASSAY"]
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    ua = row["UNIPROT_ASSAY"]
                    
                    # Add each metric to the dictionary
                    for metric in ["MISSINGFREQ", "LOD", "NPX"]:
                        col_name = f"9000_{ua}_{metric}"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[metric]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[metric]
                
                logger.info(f"Processed chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Convert the dictionary to a DataFrame efficiently
    logger.info("Converting collected data to DataFrame")
    
    # Create a list of dictionaries, each representing a row
    rows = []
    for (patno, event_id), values in data_dict.items():
        row_dict = {"PATNO": patno, "EVENT_ID": event_id}
        row_dict.update(values)
        rows.append(row_dict)
    
    # Create DataFrame from the list of dictionaries (much more efficient than adding columns one by one)
    result_df = pd.DataFrame(rows)
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info(f"Successfully processed Project 9000 data: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


def load_project_222(folder_path: str) -> pd.DataFrame:
    """
    Load and process PPMI_Project_222 data files.
    
    This function:
    1. Finds all files with the prefix "PPMI_Project_222"
    2. For each unique UNIPROT-ASSAY combination, creates three columns:
       - UNIPROT_ASSAY_MISSINGFREQ
       - UNIPROT_ASSAY_LOD
       - UNIPROT_ASSAY_NPX
    3. Adds "222_" prefix to each created column
    4. Keeps only PATNO, EVENT_ID, and the newly created columns
    5. Removes "PPMI-" prefix from PATNO values
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each UNIPROT-ASSAY metric
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("PPMI_Project_222"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No PPMI_Project_222 files found in {folder_path}")
        return pd.DataFrame()
    
    # First, get unique PATNO/EVENT_ID combinations to create the base dataframe
    logger.info("Creating base dataframe with unique PATNO/EVENT_ID combinations")
    patno_event_pairs = set()
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only PATNO and EVENT_ID columns to get unique combinations
            df_ids = pd.read_csv(file_path, usecols=["PATNO", "EVENT_ID"])
            
            for _, row in df_ids.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                patno_event_pairs.add((patno, row["EVENT_ID"]))
        except Exception as e:
            logger.error(f"Error reading PATNO/EVENT_ID from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {pair: {} for pair in patno_event_pairs}
    
    # Process each file separately to reduce memory usage
    for file_path in matching_files:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000  # Increased chunk size for better performance
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "UNIPROT", "ASSAY", "MISSINGFREQ", "LOD", "NPX"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Create a combined key for UNIPROT and ASSAY
                chunk["UNIPROT_ASSAY"] = chunk["UNIPROT"] + "_" + chunk["ASSAY"]
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    ua = row["UNIPROT_ASSAY"]
                    
                    # Add each metric to the dictionary
                    for metric in ["MISSINGFREQ", "LOD", "NPX"]:
                        col_name = f"222_{ua}_{metric}"  # Using "222_" prefix instead of "9000_"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[metric]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[metric]
                
                logger.info(f"Processed chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Convert the dictionary to a DataFrame efficiently
    logger.info("Converting collected data to DataFrame")
    
    # Create a list of dictionaries, each representing a row
    rows = []
    for (patno, event_id), values in data_dict.items():
        row_dict = {"PATNO": patno, "EVENT_ID": event_id}
        row_dict.update(values)
        rows.append(row_dict)
    
    # Create DataFrame from the list of dictionaries (much more efficient than adding columns one by one)
    result_df = pd.DataFrame(rows)
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info(f"Successfully processed Project 222 data: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


def load_project_196(folder_path: str) -> pd.DataFrame:
    """
    Load and process PPMI_Project_196 data files.
    
    This function handles two types of Project 196 files:
    1. Files with "NPX" in the name - processed like Project 222 files with MISSINGFREQ, LOD, NPX
    2. Files with "Counts" in the name - processed with COUNT, INCUB, AMP, EXT columns
    
    For each unique UNIPROT-ASSAY combination, creates columns:
    - For NPX files: UNIPROT_ASSAY_MISSINGFREQ, UNIPROT_ASSAY_LOD, UNIPROT_ASSAY_NPX
    - For Counts files: UNIPROT_ASSAY_COUNT, UNIPROT_ASSAY_INCUB, UNIPROT_ASSAY_AMP, UNIPROT_ASSAY_EXT
    
    Adds "196_" prefix to each created column and removes "PPMI-" prefix from PATNO values.
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each UNIPROT-ASSAY metric
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("PPMI_Project_196"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No PPMI_Project_196 files found in {folder_path}")
        return pd.DataFrame()
    
    # Separate files into NPX and Counts categories
    npx_files = [f for f in matching_files if "NPX" in os.path.basename(f)]
    counts_files = [f for f in matching_files if "Counts" in os.path.basename(f)]
    
    logger.info(f"Found {len(npx_files)} NPX files and {len(counts_files)} Counts files")
    
    # First, get unique PATNO/EVENT_ID combinations to create the base dataframe
    logger.info("Creating base dataframe with unique PATNO/EVENT_ID combinations")
    patno_event_pairs = set()
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only PATNO and EVENT_ID columns to get unique combinations
            df_ids = pd.read_csv(file_path, usecols=["PATNO", "EVENT_ID"])
            
            for _, row in df_ids.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                patno_event_pairs.add((patno, row["EVENT_ID"]))
        except Exception as e:
            logger.error(f"Error reading PATNO/EVENT_ID from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {pair: {} for pair in patno_event_pairs}
    
    # Process NPX files
    for file_path in npx_files:
        try:
            logger.info(f"Processing NPX file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "UNIPROT", "ASSAY", "MISSINGFREQ", "LOD", "NPX"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Create a combined key for UNIPROT and ASSAY
                chunk["UNIPROT_ASSAY"] = chunk["UNIPROT"] + "_" + chunk["ASSAY"]
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    ua = row["UNIPROT_ASSAY"]
                    
                    # Add each metric to the dictionary
                    for metric in ["MISSINGFREQ", "LOD", "NPX"]:
                        col_name = f"196_{ua}_{metric}"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[metric]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[metric]
                
                logger.info(f"Processed NPX chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing NPX file {file_path}: {e}")
    
    # Process Counts files
    for file_path in counts_files:
        try:
            logger.info(f"Processing Counts file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "UNIPROT", "ASSAY", "COUNT", 
                                   "INCUBATIONCONTROLCOUNT", "AMPLIFICATIONCONTROLCOUNT", 
                                   "EXTENSIONCONTROLCOUNT"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Create a combined key for UNIPROT and ASSAY
                chunk["UNIPROT_ASSAY"] = chunk["UNIPROT"] + "_" + chunk["ASSAY"]
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    ua = row["UNIPROT_ASSAY"]
                    
                    # Map the long column names to abbreviated versions
                    metric_mapping = {
                        "COUNT": "COUNT",
                        "INCUBATIONCONTROLCOUNT": "INCUB",
                        "AMPLIFICATIONCONTROLCOUNT": "AMP",
                        "EXTENSIONCONTROLCOUNT": "EXT"
                    }
                    
                    # Add each metric to the dictionary
                    for long_name, short_name in metric_mapping.items():
                        col_name = f"196_{ua}_{short_name}"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[long_name]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[long_name]
                
                logger.info(f"Processed Counts chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing Counts file {file_path}: {e}")
    
    # Convert the dictionary to a DataFrame efficiently
    logger.info("Converting collected data to DataFrame")
    
    # Create a list of dictionaries, each representing a row
    rows = []
    for (patno, event_id), values in data_dict.items():
        row_dict = {"PATNO": patno, "EVENT_ID": event_id}
        row_dict.update(values)
        rows.append(row_dict)
    
    # Create DataFrame from the list of dictionaries
    result_df = pd.DataFrame(rows)
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info(f"Successfully processed Project 196 data: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


def load_project_177_untargeted_proteomics(folder_path: str) -> pd.DataFrame:
    """
    Load and process PPMI_Project_177_Untargeted_Proteomics data files.
    
    This function:
    1. Finds all files with the prefix "PPMI_Project_177"
    2. Renames CLINICAL_EVENT to EVENT_ID if present
    3. Pivots the data to create columns for each unique TESTNAME
    4. Adds "177_" prefix to each TESTNAME column
    5. Keeps only PATNO, SEX, COHORT, EVENT_ID, and the new TESTNAME columns
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("PPMI_Project_177"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No PPMI_Project_177 files found in {folder_path}")
        return pd.DataFrame()
    
    # Load and combine all matching files
    dfs = []
    for file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dfs:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if required columns exist
    required_columns = ["PATNO", "EVENT_ID", "TESTNAME", "TESTVALUE"]
    for col in required_columns:
        if col not in combined_df.columns:
            logger.error(f"Required column {col} not found in the data")
            return pd.DataFrame()
    
    try:
        # Determine which columns to keep for the pivot
        pivot_columns = ["PATNO", "EVENT_ID"]
        if "SEX" in combined_df.columns:
            pivot_columns.append("SEX")
        if "COHORT" in combined_df.columns:
            pivot_columns.append("COHORT")
        
        # Pivot the data to create columns for each TESTNAME
        pivoted_df = pd.pivot_table(
            combined_df,
            index=pivot_columns,
            columns="TESTNAME",
            values="TESTVALUE",
            aggfunc="first"  # In case there are duplicates
        ).reset_index()
        
        # Rename columns to add "177_" prefix to TESTNAME columns
        # First, get the names of columns that were created from TESTNAME
        testname_columns = [col for col in pivoted_df.columns if col not in pivot_columns]
        
        # Create a dictionary for renaming
        rename_dict = {col: f"177_{col}" for col in testname_columns}
        
        # Rename the columns
        pivoted_df = pivoted_df.rename(columns=rename_dict)
        
        logger.info(f"Successfully processed Project 177 data: {len(pivoted_df)} rows, {len(pivoted_df.columns)} columns")
        return pivoted_df
        
    except Exception as e:
        logger.error(f"Error pivoting data: {e}")
        return pd.DataFrame()


def load_project_214_olink(folder_path: str) -> pd.DataFrame:
    """
    Load and process Project_214_Olink data files.
    
    This function:
    1. Finds all files with the prefix "Project_214_Olink"
    2. Renames CLINICAL_EVENT to EVENT_ID if present
    3. Renames MISSING_FREQ to MISSINGFREQ if present
    4. For each unique UNIPROT-ASSAY combination, creates three columns:
       - UNIPROT_ASSAY_MISSINGFREQ
       - UNIPROT_ASSAY_LOD
       - UNIPROT_ASSAY_NPX
    5. Adds "214_" prefix to each created column
    6. Keeps PATNO, EVENT_ID, SEX, COHORT, and the newly created columns
    7. Removes "PPMI-" prefix from PATNO values
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each UNIPROT-ASSAY metric
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("Project_214_Olink"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No Project_214_Olink files found in {folder_path}")
        return pd.DataFrame()
    
    # First, get unique PATNO/EVENT_ID combinations and their SEX and COHORT values
    logger.info("Creating base dataframe with unique PATNO/EVENT_ID combinations")
    patno_event_pairs = set()  # Initialize the set here
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only the necessary columns for the base dataframe
            df_base = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df_base.columns and "EVENT_ID" not in df_base.columns:
                df_base = df_base.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            # Ensure we have the required columns for the base dataframe
            if not all(col in df_base.columns for col in ["PATNO", "EVENT_ID"]):
                logger.warning(f"Missing required base columns in {file_path}")
                continue
            
            for _, row in df_base.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                patno_event_pairs.add((patno, row["EVENT_ID"]))
        except Exception as e:
            logger.error(f"Error reading PATNO/EVENT_ID from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {}
    for pair in patno_event_pairs:
        data_dict[pair] = {}
    
    # Process each file separately to reduce memory usage
    for file_path in matching_files:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Rename CLINICAL_EVENT to EVENT_ID if it exists
                if "CLINICAL_EVENT" in chunk.columns and "EVENT_ID" not in chunk.columns:
                    chunk = chunk.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
                
                # Rename MISSING_FREQ to MISSINGFREQ if it exists
                if "MISSING_FREQ" in chunk.columns and "MISSINGFREQ" not in chunk.columns:
                    chunk = chunk.rename(columns={"MISSING_FREQ": "MISSINGFREQ"})
                
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "UNIPROT", "ASSAY", "MISSINGFREQ", "LOD", "NPX"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Create a combined key for UNIPROT and ASSAY
                chunk["UNIPROT_ASSAY"] = chunk["UNIPROT"] + "_" + chunk["ASSAY"]
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    ua = row["UNIPROT_ASSAY"]
                    
                    # Add each metric to the dictionary
                    for metric in ["MISSINGFREQ", "LOD", "NPX"]:
                        col_name = f"214_{ua}_{metric}"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[metric]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[metric]
                
                logger.info(f"Processed chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Convert the dictionary to a DataFrame efficiently
    logger.info("Converting collected data to DataFrame")
    
    # Create a list of dictionaries, each representing a row
    rows = []
    for (patno, event_id), values in data_dict.items():
        row_dict = {"PATNO": patno, "EVENT_ID": event_id}
        row_dict.update(values)
        rows.append(row_dict)
    
    # Create DataFrame from the list of dictionaries
    result_df = pd.DataFrame(rows)
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info(f"Successfully processed Project 214 data: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


def load_current_biospecimen_analysis(folder_path: str) -> pd.DataFrame:
    """
    Load and process Current_Biospecimen_Analysis_Results data files.
    
    This function:
    1. Finds all files with the prefix "Current_Biospecimen_Analysis_Results"
    2. Renames CLINICAL_EVENT to EVENT_ID if present
    3. Pivots the data to create columns for each unique TESTNAME
    4. Adds "BIO_" prefix to each TESTNAME column
    5. Keeps only PATNO, SEX, COHORT, EVENT_ID, and the new TESTNAME columns
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("Current_Biospecimen_Analysis_Results"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No Current_Biospecimen_Analysis_Results files found in {folder_path}")
        return pd.DataFrame()
    
    # Load and combine all matching files
    dfs = []
    for file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dfs:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if required columns exist
    required_columns = ["PATNO", "EVENT_ID", "TESTNAME", "TESTVALUE"]
    for col in required_columns:
        if col not in combined_df.columns:
            logger.error(f"Required column {col} not found in the data")
            return pd.DataFrame()
    
    try:
        # Determine which columns to keep for the pivot
        pivot_columns = ["PATNO", "EVENT_ID"]
        if "SEX" in combined_df.columns:
            pivot_columns.append("SEX")
        if "COHORT" in combined_df.columns:
            pivot_columns.append("COHORT")
        
        # Pivot the data to create columns for each TESTNAME
        pivoted_df = pd.pivot_table(
            combined_df,
            index=pivot_columns,
            columns="TESTNAME",
            values="TESTVALUE",
            aggfunc="first"  # In case there are duplicates
        ).reset_index()
        
        # Rename columns to add "BIO_" prefix to TESTNAME columns
        # First, get the names of columns that were created from TESTNAME
        testname_columns = [col for col in pivoted_df.columns if col not in pivot_columns]
        
        # Create a dictionary for renaming
        rename_dict = {col: f"BIO_{col}" for col in testname_columns}
        
        # Rename the columns
        pivoted_df = pivoted_df.rename(columns=rename_dict)
        
        logger.info(f"Successfully processed Current Biospecimen Analysis data: {len(pivoted_df)} rows, {len(pivoted_df.columns)} columns")
        return pivoted_df
        
    except Exception as e:
        logger.error(f"Error pivoting data: {e}")
        return pd.DataFrame()


def load_blood_chemistry_hematology(folder_path: str) -> pd.DataFrame:
    """
    Load and process Blood_Chemistry___Hematology data files.
    
    This function:
    1. Finds all files with the prefix "Blood_Chemistry___Hematology"
    2. For each unique LTSTCODE-LTSTNAME combination, creates three columns:
       - LTSTCODE_LTSTNAME_LSIRES (result value)
       - LTSTCODE_LTSTNAME_LSILORNG (lower range)
       - LTSTCODE_LTSTNAME_LSIHIRNG (higher range)
    3. Adds "BCH_" prefix to each created column
    4. Replaces spaces with underscores in the column names
    5. Keeps only PATNO, EVENT_ID, and the newly created columns
    6. Removes "PPMI-" prefix from PATNO values if present
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns for each test metric
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefix
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if filename.startswith("Blood_Chemistry___Hematology"):
            matching_files.append(file_path)
    
    if not matching_files:
        logger.warning(f"No Blood_Chemistry___Hematology files found in {folder_path}")
        return pd.DataFrame()
    
    # First, get unique PATNO/EVENT_ID combinations to create the base dataframe
    logger.info("Creating base dataframe with unique PATNO/EVENT_ID combinations")
    patno_event_pairs = set()
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only PATNO and EVENT_ID columns to get unique combinations
            df_ids = pd.read_csv(file_path, usecols=["PATNO", "EVENT_ID"])
            
            for _, row in df_ids.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                patno_event_pairs.add((patno, row["EVENT_ID"]))
        except Exception as e:
            logger.error(f"Error reading PATNO/EVENT_ID from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {pair: {} for pair in patno_event_pairs}
    
    # Process each file separately to reduce memory usage
    for file_path in matching_files:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the file in chunks to reduce memory usage
            chunk_size = 50000  # Increased chunk size for better performance
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Check if required columns exist
                required_columns = ["PATNO", "EVENT_ID", "LTSTCODE", "LTSTNAME", "LSIRES", "LSILORNG", "LSIHIRNG"]
                if not all(col in chunk.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in chunk.columns]
                    logger.error(f"Required columns {missing} not found in {file_path}")
                    continue
                
                # Process each row efficiently
                for _, row in chunk.iterrows():
                    # Remove "PPMI-" prefix from PATNO if it exists
                    patno = row["PATNO"]
                    if isinstance(patno, str) and patno.startswith("PPMI-"):
                        patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                    
                    event_id = row["EVENT_ID"]
                    key = (patno, event_id)
                    
                    # Skip if this PATNO/EVENT_ID combination wasn't in our original set
                    if key not in data_dict:
                        continue
                    
                    # Create a combined key for LTSTCODE and LTSTNAME
                    # Replace spaces with underscores
                    test_code = str(row["LTSTCODE"]).strip()
                    test_name = str(row["LTSTNAME"]).strip().replace(" ", "_")
                    combined_name = f"{test_code}_{test_name}"
                    
                    # Add each metric to the dictionary
                    for metric, column in [
                        ("LSIRES", "LSIRES"), 
                        ("LSILORNG", "LSILORNG"), 
                        ("LSIHIRNG", "LSIHIRNG")
                    ]:
                        col_name = f"BCH_{combined_name}_{metric}"
                        
                        # Only update if we don't have a value yet or if the current value is not NaN
                        # and the existing one is NaN
                        if (col_name not in data_dict[key] or 
                            (pd.notna(row[column]) and pd.isna(data_dict[key].get(col_name)))):
                            data_dict[key][col_name] = row[column]
                
                logger.info(f"Processed chunk with {len(chunk)} rows")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Convert the dictionary to a DataFrame efficiently
    logger.info("Converting collected data to DataFrame")
    
    # Create a list of dictionaries, each representing a row
    rows = []
    for (patno, event_id), values in data_dict.items():
        row_dict = {"PATNO": patno, "EVENT_ID": event_id}
        row_dict.update(values)
        rows.append(row_dict)
    
    # Create DataFrame from the list of dictionaries
    result_df = pd.DataFrame(rows)
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info(f"Successfully processed Blood Chemistry & Hematology data: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


def load_and_join_biospecimen_files(folder_path: str, file_prefixes: list, combine_duplicates: bool = True) -> pd.DataFrame:
    """
    Load and join multiple biospecimen data files based on PATNO and EVENT_ID.
    
    This function:
    1. Finds all CSV files matching the provided prefixes
    2. Loads each file and ensures it has PATNO and EVENT_ID columns
    3. Joins all dataframes on PATNO and EVENT_ID
    4. For duplicate columns, either:
       - Combines values with a pipe separator (|) if combine_duplicates=True
       - Adds numeric suffixes if combine_duplicates=False
    5. Logs any duplicate column names that are encountered during merging
    
    Args:
        folder_path: Path to the Biospecimen folder containing the CSV files
        file_prefixes: List of file prefixes to include (e.g., ["Clinical_Labs", "Genetic_Testing_Results"])
        combine_duplicates: If True, combine duplicate column values with a pipe separator;
                           if False, add numeric suffixes to duplicate columns
    
    Returns:
        A DataFrame with one row per PATNO/EVENT_ID and columns from all matching files
    """
    # Find all CSV files in the folder and its subdirectories
    all_csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    
    # Filter files based on prefixes
    matching_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        for prefix in file_prefixes:
            if filename.startswith(prefix):
                matching_files.append((prefix, file_path))
                break
    
    if not matching_files:
        logger.warning(f"No files matching the provided prefixes found in {folder_path}")
        return pd.DataFrame()
    
    # Group files by prefix for logging purposes
    files_by_prefix = {}
    for prefix, file_path in matching_files:
        if prefix not in files_by_prefix:
            files_by_prefix[prefix] = []
        files_by_prefix[prefix].append(file_path)
    
    # Log the files found for each prefix
    for prefix, files in files_by_prefix.items():
        logger.info(f"Found {len(files)} files for prefix '{prefix}'")
    
    # Load each file and prepare for merging
    dataframes = []
    column_sources = {}  # Track which file each column came from
    
    for prefix, file_path in matching_files:
        try:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df.columns and "EVENT_ID" not in df.columns:
                df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            # Check if required columns exist
            if "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
                logger.warning(f"File {file_path} is missing PATNO or EVENT_ID columns, skipping")
                continue
            
            # Remove "PPMI-" prefix from PATNO if it exists
            if df["PATNO"].dtype == object:  # Only process if PATNO is a string type
                df["PATNO"] = df["PATNO"].apply(
                    lambda x: x[5:] if isinstance(x, str) and x.startswith("PPMI-") else x
                )
            
            # Track the source of each column
            filename = os.path.basename(file_path)
            for col in df.columns:
                if col not in ["PATNO", "EVENT_ID"]:  # Don't track join columns
                    if col in column_sources:
                        column_sources[col].append(filename)
                    else:
                        column_sources[col] = [filename]
            
            dataframes.append(df)
            logger.info(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    if not dataframes:
        logger.warning("No files were successfully loaded")
        return pd.DataFrame()
    
    # Check for duplicate columns across dataframes
    duplicate_columns = {col: sources for col, sources in column_sources.items() if len(sources) > 1}
    if duplicate_columns:
        logger.warning(f"Found {len(duplicate_columns)} duplicate column names across files:")
        for col, sources in duplicate_columns.items():
            logger.warning(f"  Column '{col}' appears in: {', '.join(sources)}")
    
    if combine_duplicates and duplicate_columns:
        logger.info("Will combine duplicate column values with pipe separator (|)")
        
        # Create a dictionary to store the combined data
        # Structure: {(patno, event_id): {column_name: [values]}}
        combined_data = {}
        
        # Process each dataframe to collect all values
        for df_idx, df in enumerate(dataframes):
            for _, row in df.iterrows():
                patno = row["PATNO"]
                event_id = row["EVENT_ID"]
                key = (patno, event_id)
                
                if key not in combined_data:
                    combined_data[key] = {"PATNO": patno, "EVENT_ID": event_id}
                
                # Add all other columns
                for col in df.columns:
                    if col not in ["PATNO", "EVENT_ID"]:
                        value = row[col]
                        
                        # Skip NaN values
                        if pd.isna(value):
                            continue
                            
                        # Convert to string
                        value_str = str(value)
                        
                        # If column already exists, append the value
                        if col in combined_data[key]:
                            # Only append if it's a new value
                            if value_str not in combined_data[key][col].split("|"):
                                combined_data[key][col] += f"|{value_str}"
                        else:
                            combined_data[key][col] = value_str
        
        # Convert the combined data to a DataFrame
        result_rows = []
        for key, row_data in combined_data.items():
            result_rows.append(row_data)
        
        result_df = pd.DataFrame(result_rows)
        
        # Log the final result
        logger.info(f"Successfully merged {len(dataframes)} dataframes with combined duplicate values")
        logger.info(f"Final dataframe has {len(result_df)} rows and {len(result_df.columns)} columns")
        
        return result_df
    
    else:
        # Merge all dataframes using the original method with suffixes
        logger.info("Merging dataframes on PATNO and EVENT_ID")
        
        # Start with the first dataframe
        result_df = dataframes[0]
        
        # Merge with each subsequent dataframe
        for i, df in enumerate(dataframes[1:], 1):
            # Use outer join to keep all PATNO/EVENT_ID combinations
            result_df = pd.merge(
                result_df, 
                df, 
                on=["PATNO", "EVENT_ID"], 
                how="outer",
                suffixes=("", f"_{i}")  # Add suffix only to duplicate columns from right dataframe
            )
            
            # Check if any columns were renamed due to duplicates
            renamed_columns = [col for col in result_df.columns if col.endswith(f"_{i}")]
            if renamed_columns:
                logger.warning(f"After merging dataframe {i+1}, {len(renamed_columns)} columns were renamed:")
                for col in renamed_columns[:5]:  # Show first 5 as examples
                    logger.warning(f"  '{col[:-len(f'_{i}')]}' renamed to '{col}'")
                if len(renamed_columns) > 5:
                    logger.warning(f"  ... and {len(renamed_columns) - 5} more")
        
        # Log the final result
        logger.info(f"Successfully merged {len(dataframes)} dataframes with suffixed duplicate columns")
        logger.info(f"Final dataframe has {len(result_df)} rows and {len(result_df.columns)} columns")
        
        return result_df


def load_biospecimen_data(data_path: str, source: str, exclude: list = None) -> dict:
    """
    Load biospecimen data from the specified path.
    
    Args:
        data_path: Path to the data directory
        source: The data source (e.g., PPMI)
        exclude: List of biospecimen data sources to exclude (e.g., ['project_9000', 'project_222'])
        
    Returns:
        A dictionary containing loaded biospecimen data
    """
    # Initialize exclude if None
    if exclude is None:
        exclude = []
    
    if exclude:
        logger.info(f"Will exclude the following projects: {exclude}")
    
    biospecimen_data = {}
    
    # Path to the Biospecimen folder
    biospecimen_path = os.path.join(data_path, "Biospecimen")
    
    if not os.path.exists(biospecimen_path):
        logger.warning(f"Biospecimen directory not found: {biospecimen_path}")
        return biospecimen_data
    
    # Load Project_151_pQTL_in_CSF data (non-batch-corrected)
    if "project_151_pQTL_CSF" not in exclude:
        try:
            biospecimen_data["project_151_pQTL_CSF"] = load_project_151_pQTL_CSF(
                biospecimen_path, 
                batch_corrected=False
            )
            logger.info(f"Loaded Project_151_pQTL_in_CSF data: {len(biospecimen_data['project_151_pQTL_CSF'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project_151_pQTL_in_CSF data: {e}")
            biospecimen_data["project_151_pQTL_CSF"] = pd.DataFrame()
    else:
        logger.info("Skipping Project_151_pQTL_in_CSF (excluded)")
    
    # Load Project_151_pQTL_in_CSF data (batch-corrected)
    if "project_151_pQTL_CSF_batch_corrected" not in exclude:
        try:
            biospecimen_data["project_151_pQTL_CSF_batch_corrected"] = load_project_151_pQTL_CSF(
                biospecimen_path, 
                batch_corrected=True
            )
            logger.info(f"Loaded batch-corrected Project_151_pQTL_in_CSF data: {len(biospecimen_data['project_151_pQTL_CSF_batch_corrected'])} rows")
        except Exception as e:
            logger.error(f"Error loading batch-corrected Project_151_pQTL_in_CSF data: {e}")
            biospecimen_data["project_151_pQTL_CSF_batch_corrected"] = pd.DataFrame()
    else:
        logger.info("Skipping batch-corrected Project_151_pQTL_in_CSF (excluded)")
    
    # Load Metabolomic_Analysis_of_LRRK2 data (excluding CSF)
    if "metabolomic_lrrk2" not in exclude:
        try:
            biospecimen_data["metabolomic_lrrk2"] = load_metabolomic_lrrk2(
                biospecimen_path, 
                include_csf=False
            )
            logger.info(f"Loaded Metabolomic_Analysis_of_LRRK2 data: {len(biospecimen_data['metabolomic_lrrk2'])} rows")
        except Exception as e:
            logger.error(f"Error loading Metabolomic_Analysis_of_LRRK2 data: {e}")
            biospecimen_data["metabolomic_lrrk2"] = pd.DataFrame()
    else:
        logger.info("Skipping Metabolomic_Analysis_of_LRRK2 (excluded)")
    
    # Load Metabolomic_Analysis_of_LRRK2_CSF data
    if "metabolomic_lrrk2_csf" not in exclude:
        try:
            biospecimen_data["metabolomic_lrrk2_csf"] = load_metabolomic_lrrk2(
                biospecimen_path, 
                include_csf=True
            )
            # Filter to only include CSF files
            if not biospecimen_data["metabolomic_lrrk2_csf"].empty:
                csf_columns = [col for col in biospecimen_data["metabolomic_lrrk2_csf"].columns if col.startswith("LRRK2_") and "_CSF_" in col]
                if csf_columns:
                    keep_cols = ["PATNO", "EVENT_ID"]
                    if "SEX" in biospecimen_data["metabolomic_lrrk2_csf"].columns:
                        keep_cols.append("SEX")
                    if "COHORT" in biospecimen_data["metabolomic_lrrk2_csf"].columns:
                        keep_cols.append("COHORT")
                    keep_cols.extend(csf_columns)
                    biospecimen_data["metabolomic_lrrk2_csf"] = biospecimen_data["metabolomic_lrrk2_csf"][keep_cols]
            
            logger.info(f"Loaded Metabolomic_Analysis_of_LRRK2_CSF data: {len(biospecimen_data['metabolomic_lrrk2_csf'])} rows")
        except Exception as e:
            logger.error(f"Error loading Metabolomic_Analysis_of_LRRK2_CSF data: {e}")
            biospecimen_data["metabolomic_lrrk2_csf"] = pd.DataFrame()
    else:
        logger.info("Skipping Metabolomic_Analysis_of_LRRK2_CSF (excluded)")
    
    # Load Targeted___untargeted_MS-based_proteomics_of_urine_in_PD data
    if "urine_proteomics" not in exclude:
        try:
            biospecimen_data["urine_proteomics"] = load_urine_proteomics(biospecimen_path)
            logger.info(f"Loaded urine proteomics data: {len(biospecimen_data['urine_proteomics'])} rows")
        except Exception as e:
            logger.error(f"Error loading urine proteomics data: {e}")
            biospecimen_data["urine_proteomics"] = pd.DataFrame()
    else:
        logger.info("Skipping urine proteomics (excluded)")
    
    # Load PPMI_Project_9000 data
    if "project_9000" not in exclude:
        try:
            biospecimen_data["project_9000"] = load_project_9000(biospecimen_path)
            logger.info(f"Loaded Project 9000 data: {len(biospecimen_data['project_9000'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project 9000 data: {e}")
            biospecimen_data["project_9000"] = pd.DataFrame()
    else:
        logger.info("Skipping Project 9000 (excluded)")
    
    # Load PPMI_Project_222 data
    if "project_222" not in exclude:
        try:
            biospecimen_data["project_222"] = load_project_222(biospecimen_path)
            logger.info(f"Loaded Project 222 data: {len(biospecimen_data['project_222'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project 222 data: {e}")
            biospecimen_data["project_222"] = pd.DataFrame()
    else:
        logger.info("Skipping Project 222 (excluded)")
    
    # Load PPMI_Project_196 data
    if "project_196" not in exclude:
        try:
            biospecimen_data["project_196"] = load_project_196(biospecimen_path)
            logger.info(f"Loaded Project 196 data: {len(biospecimen_data['project_196'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project 196 data: {e}")
            biospecimen_data["project_196"] = pd.DataFrame()
    else:
        logger.info("Skipping Project 196 (excluded)")
    
    # Load PPMI_Project_177 data
    if "project_177" not in exclude:
        try:
            biospecimen_data["project_177"] = load_project_177_untargeted_proteomics(biospecimen_path)
            logger.info(f"Loaded Project 177 data: {len(biospecimen_data['project_177'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project 177 data: {e}")
            biospecimen_data["project_177"] = pd.DataFrame()
    else:
        logger.info("Skipping Project 177 (excluded)")
    
    # Load Project_214_Olink data
    if "project_214" not in exclude:
        try:
            biospecimen_data["project_214"] = load_project_214_olink(biospecimen_path)
            logger.info(f"Loaded Project 214 data: {len(biospecimen_data['project_214'])} rows")
        except Exception as e:
            logger.error(f"Error loading Project 214 data: {e}")
            biospecimen_data["project_214"] = pd.DataFrame()
    else:
        logger.info("Skipping Project 214 (excluded)")
    
    # Load Current_Biospecimen_Analysis_Results data
    if "current_biospecimen" not in exclude:
        try:
            biospecimen_data["current_biospecimen"] = load_current_biospecimen_analysis(biospecimen_path)
            logger.info(f"Loaded Current Biospecimen Analysis data: {len(biospecimen_data['current_biospecimen'])} rows")
        except Exception as e:
            logger.error(f"Error loading Current Biospecimen Analysis data: {e}")
            biospecimen_data["current_biospecimen"] = pd.DataFrame()
    else:
        logger.info("Skipping Current Biospecimen Analysis (excluded)")
    
    # Load Blood_Chemistry___Hematology data
    if "blood_chemistry_hematology" not in exclude:
        try:
            biospecimen_data["blood_chemistry_hematology"] = load_blood_chemistry_hematology(biospecimen_path)
            logger.info(f"Loaded Blood Chemistry & Hematology data: {len(biospecimen_data['blood_chemistry_hematology'])} rows")
        except Exception as e:
            logger.error(f"Error loading Blood Chemistry & Hematology data: {e}")
            biospecimen_data["blood_chemistry_hematology"] = pd.DataFrame()
    else:
        logger.info("Skipping Blood Chemistry & Hematology (excluded)")
    
    # Load files that don't require individual processing
    if "standard_files" not in exclude:
        try:
            standard_file_prefixes = [
                "Clinical_Labs",
                "Genetic_Testing_Results",
                "Skin_Biopsy",
                "Research_Biospecimens",
                "Lumbar_Puncture",
                "Laboratory_Procedures_with_Elapsed_Times"
            ]
            
            biospecimen_data["standard_files"] = load_and_join_biospecimen_files(
                biospecimen_path,
                standard_file_prefixes,
                combine_duplicates=True  # Use the new parameter to combine duplicate values
            )
            logger.info(f"Loaded standard biospecimen files: {len(biospecimen_data['standard_files'])} rows")
        except Exception as e:
            logger.error(f"Error loading standard biospecimen files: {e}")
            biospecimen_data["standard_files"] = pd.DataFrame()
    else:
        logger.info("Skipping standard files (excluded)")
    
    return biospecimen_data


def merge_biospecimen_data(biospecimen_data: dict, merge_all: bool = True, 
                          output_filename: str = "biospecimen.csv", output_dir: str = None,
                          include: list = None, exclude: list = None) -> Union[pd.DataFrame, dict]:
    """
    Merge all biospecimen data into either a single DataFrame or keep as separate DataFrames.
    
    Args:
        biospecimen_data: Dictionary containing loaded biospecimen data from load_biospecimen_data()
        merge_all: If True, merge all DataFrames on PATNO and EVENT_ID; if False, return dictionary of DataFrames
        output_filename: Name of the output CSV file (default: "biospecimen.csv")
        output_dir: Directory to save the output file(s); if None, files are not saved
        include: List of specific data sources to include (e.g., ['project_151', 'metabolomic_lrrk2'])
                 If provided, only these sources will be used
        exclude: List of specific data sources to exclude
                 Only used if include is None or empty
    
    Returns:
        If merge_all is True: A single DataFrame with all biospecimen data merged on PATNO and EVENT_ID
        If merge_all is False: The original dictionary of DataFrames with potential file saving
    """
    # Import only when needed for memory tracking
    import psutil
    import gc
    
    # Initialize include/exclude to empty lists if None
    if include is None:
        include = []
    if exclude is None:
        exclude = []
    
    # Filter the biospecimen_data dictionary based on include/exclude
    filtered_data = {}
    
    if include:
        # If include is provided, only use those sources
        logger.info(f"Including only specified sources: {include}")
        for source_name in include:
            if source_name in biospecimen_data:
                filtered_data[source_name] = biospecimen_data[source_name]
            else:
                logger.warning(f"Requested source '{source_name}' not found in biospecimen_data")
    elif exclude:
        # If exclude is provided but include is not, use all sources except those excluded
        logger.info(f"Excluding specified sources: {exclude}")
        for source_name, df in biospecimen_data.items():
            if source_name not in exclude:
                filtered_data[source_name] = df
            else:
                logger.info(f"Excluding source: {source_name}")
    else:
        # If neither include nor exclude are provided, use all sources
        logger.info("Using all available data sources")
        filtered_data = biospecimen_data.copy()
    
    if not filtered_data:
        logger.warning("No data sources remain after applying include/exclude filters")
        if merge_all:
            return pd.DataFrame()
        else:
            return {}
    
    # Log the sources that will be processed
    logger.info(f"Processing {len(filtered_data)} data sources: {list(filtered_data.keys())}")
    
    process = psutil.Process()
    
    def log_memory_usage(label):
        """Log current memory usage with a label"""
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        logger.info(f"Memory usage ({label}): {memory_mb:.2f} MB")
    
    log_memory_usage("start")
    
    if merge_all:
        logger.info("Merging filtered biospecimen data into a single DataFrame")
        
        # Get unique PATNO/EVENT_ID combinations first to create base DataFrame
        all_pairs = set()
        for source_name, df in filtered_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
                continue
            
            # Collect unique pairs
            for _, row in df[["PATNO", "EVENT_ID"]].iterrows():
                all_pairs.add((str(row["PATNO"]), row["EVENT_ID"]))
        
        # Create base DataFrame from unique pairs
        merged_df = pd.DataFrame(all_pairs, columns=["PATNO", "EVENT_ID"])
        logger.info(f"Created base DataFrame with {len(merged_df)} unique PATNO/EVENT_ID pairs")
        log_memory_usage("after creating base DataFrame")
        
        # Track the number of rows and columns from each source
        source_stats = {}
        
        # Merge each DataFrame one by one
        for source_name, df in filtered_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning(f"Skipping {source_name}: No data available")
                continue
                
            if "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
                logger.warning(f"Skipping {source_name}: Missing PATNO or EVENT_ID columns")
                continue
            
            # Record stats
            source_stats[source_name] = {
                "rows": len(df),
                "columns": len(df.columns) - 2  # Subtract PATNO and EVENT_ID
            }
            
            logger.info(f"Merging {source_name} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Convert PATNO to string type for consistent merging
            df = df.copy()
            df["PATNO"] = df["PATNO"].astype(str)
            
            # Handle duplicate columns by renaming in the df being merged
            duplicate_cols = [col for col in df.columns if col in merged_df.columns and col not in ["PATNO", "EVENT_ID"]]
            if duplicate_cols:
                logger.warning(f"Found {len(duplicate_cols)} duplicate columns when merging {source_name}")
                # Add source prefix to duplicate columns
                rename_dict = {col: f"{source_name}_{col}" for col in duplicate_cols}
                df = df.rename(columns=rename_dict)
                
                # Log a few examples of renamed columns
                examples = list(rename_dict.items())[:3]
                logger.info(f"Renamed columns (examples): {examples}")
                if len(rename_dict) > 3:
                    logger.info(f"... and {len(rename_dict) - 3} more")
            
            # Keep only columns not already in merged_df to minimize memory
            new_cols = [col for col in df.columns if col not in merged_df.columns or col in ["PATNO", "EVENT_ID"]]
            df_subset = df[new_cols]
            
            # Merge with left join to maintain the rows we already have
            # This is more memory efficient than outer join in this case
            merged_df = pd.merge(
                merged_df, 
                df_subset, 
                on=["PATNO", "EVENT_ID"], 
                how="left"
            )
            
            # Force garbage collection
            del df_subset
            gc.collect()
            
            logger.info(f"Merged {source_name}: Added {len(new_cols) - 2} columns")
            log_memory_usage(f"after merging {source_name}")
        
        # Log final stats
        logger.info(f"Final merged DataFrame: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        
        # Log contribution from each source
        logger.info("Data contribution from each source:")
        for source, stats in source_stats.items():
            logger.info(f"  {source}: {stats['rows']} rows, {stats['columns']} columns")
        
        # Save to CSV if output_dir is provided
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, output_filename)
            logger.info(f"Saving merged data to {output_path}")
            
            # Save in chunks to reduce memory usage
            chunk_size = 10000
            for i in range(0, len(merged_df), chunk_size):
                logger.info(f"Saving chunk {i//chunk_size + 1} of {(len(merged_df) + chunk_size - 1)//chunk_size}")
                chunk = merged_df.iloc[i:i+chunk_size]
                if i == 0:
                    # First chunk, write with header
                    chunk.to_csv(output_path, index=False, mode='w')
                else:
                    # Append without header
                    chunk.to_csv(output_path, index=False, mode='a', header=False)
                del chunk
                gc.collect()
        
        log_memory_usage("end")
        return merged_df
    
    else:
        logger.info("Keeping biospecimen data as separate DataFrames")
        
        # Save individual DataFrames if output_dir is provided
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create a subdirectory for individual files
            individual_dir = os.path.join(output_dir, "individual_biospecimen")
            if not os.path.exists(individual_dir):
                os.makedirs(individual_dir)
            
            # Save each DataFrame to a separate CSV file
            for source_name, df in filtered_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"Skipping {source_name}: No data available")
                    continue
                
                file_path = os.path.join(individual_dir, f"{source_name}.csv")
                logger.info(f"Saving {source_name} data to {file_path}")
                df.to_csv(file_path, index=False)
                
                # Force garbage collection after saving each file
                gc.collect()
        
        # Return filtered dictionary instead of original
        return filtered_data


def main():
    """
    Test the biospecimen data loading functions with exclusion.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Path to the data directory
    data_path = "./PPMI"
    
    # Define projects to exclude
    exclude_projects = ['project_9000', 'project_222', 'project_196']
    
    # Step 1: Load with exclusion directly in load_biospecimen_data
    logger.info("STEP 1: Testing direct exclusion in load_biospecimen_data")
    biospecimen_data_excluded = load_biospecimen_data(
        data_path=data_path, 
        source="PPMI", 
        exclude=exclude_projects
    )
    
    # Verify exclusion
    for project in exclude_projects:
        if project in biospecimen_data_excluded:
            logger.error(f"FAILURE: {project} was loaded despite being explicitly excluded")
        else:
            logger.info(f"SUCCESS: {project} was properly excluded during loading")
    
    logger.info(f"Loaded {len(biospecimen_data_excluded)} biospecimen data sources with exclusion")
    
    # Step 2: Compare with the traditional approach (exclude during merge)
    logger.info("\nSTEP 2: Comparing with traditional approach (exclude during merge)")
    
    # Load without exclusion first
    biospecimen_data_all = load_biospecimen_data(
        data_path=data_path, 
        source="PPMI",
        exclude=None  # No exclusion during loading
    )
    
    logger.info(f"Loaded {len(biospecimen_data_all)} total biospecimen data sources without exclusion")
    
    # Verify the excluded projects are present in the unfiltered data
    for project in exclude_projects:
        if project in biospecimen_data_all:
            logger.info(f"Confirmed: {project} is present in the unfiltered data")
            if isinstance(biospecimen_data_all[project], pd.DataFrame):
                logger.info(f"  - Contains {len(biospecimen_data_all[project])} rows")
        else:
            logger.warning(f"NOTE: {project} not found in the unfiltered data - may not exist in dataset")
    
    # Now merge with exclusion
    merged_with_exclusion = merge_biospecimen_data(
        biospecimen_data_all,
        merge_all=True,
        output_filename=None,
        exclude=exclude_projects
    )
    
    logger.info(f"Merged data with exclusion during merge: {len(merged_with_exclusion)} rows, {len(merged_with_exclusion.columns)} columns")
    
    # Step 3: Compare results between the two approaches
    logger.info("\nSTEP 3: Comparing results between the two approaches")
    
    # Merge the pre-filtered data
    merged_pre_filtered = merge_biospecimen_data(
        biospecimen_data_excluded,
        merge_all=True,
        output_filename=None
    )
    
    logger.info(f"Merged data with pre-filtered approach: {len(merged_pre_filtered)} rows, {len(merged_pre_filtered.columns)} columns")
    
    # Compare the results
    logger.info("\nCOMPARISON RESULTS:")
    logger.info(f"1. Traditional approach (exclude during merge): {len(merged_with_exclusion)} rows, {len(merged_with_exclusion.columns)} columns")
    logger.info(f"2. New approach (exclude during loading): {len(merged_pre_filtered)} rows, {len(merged_pre_filtered.columns)} columns")
    
    # Check if the results are the same
    rows_match = len(merged_with_exclusion) == len(merged_pre_filtered)
    cols_match = len(merged_with_exclusion.columns) == len(merged_pre_filtered.columns)
    
    if rows_match and cols_match:
        logger.info("SUCCESS: Both approaches produced the same size result")
    else:
        if not rows_match:
            logger.error(f"DIFFERENCE: Row counts differ: {len(merged_with_exclusion)} vs {len(merged_pre_filtered)}")
        if not cols_match:
            logger.error(f"DIFFERENCE: Column counts differ: {len(merged_with_exclusion.columns)} vs {len(merged_pre_filtered.columns)}")
    
    # Step 4: Test passing exclude to both functions (should be idempotent)
    logger.info("\nSTEP 4: Testing exclude parameter passed to both functions")
    
    # Load with exclusion and then merge with the same exclusion
    biospecimen_data_double_exclude = load_biospecimen_data(
        data_path=data_path, 
        source="PPMI", 
        exclude=exclude_projects
    )
    
    merged_double_exclude = merge_biospecimen_data(
        biospecimen_data_double_exclude,
        merge_all=True,
        output_filename=None,
        exclude=exclude_projects  # Same exclusion list
    )
    
    logger.info(f"Merged data with double exclusion: {len(merged_double_exclude)} rows, {len(merged_double_exclude.columns)} columns")
    
    # Compare with the pre-filtered approach
    if len(merged_double_exclude) == len(merged_pre_filtered) and len(merged_double_exclude.columns) == len(merged_pre_filtered.columns):
        logger.info("SUCCESS: Double exclusion is idempotent (same as excluding once)")
    else:
        logger.error(f"ERROR: Double exclusion produced different results: {len(merged_double_exclude)}x{len(merged_double_exclude.columns)} vs {len(merged_pre_filtered)}x{len(merged_pre_filtered.columns)}")
    
    # Step 5: Demonstrate the fix in DataLoader.py
    logger.info("\nSTEP 5: Demonstrating how this fix works with DataLoader.load()")
    logger.info("To apply this fix properly in DataLoader.py, modify the BIOSPECIMEN section:")
    logger.info("""
    elif modality == DataLoader.BIOSPECIMEN:
        # Old code:
        # biospec_data = load_biospecimen_data(data_path, source)
        
        # New code with exclude parameter:
        biospec_data = load_biospecimen_data(data_path, source, exclude=biospec_exclude)
        
        if merge_output:
            # If we will merge everything later, just store the dictionary
            data_dict[modality] = biospec_data
        else:
            # If we're not merging everything, merge just the biospecimen data
            data_dict[modality] = merge_biospecimen_data(
                biospec_data, 
                merge_all=True,
                output_filename=None,
                exclude=biospec_exclude  # This is now redundant but harmless
            )
    """)
    
    logger.info("\nBiospecimen exclusion test complete")

if __name__ == "__main__":
    main()