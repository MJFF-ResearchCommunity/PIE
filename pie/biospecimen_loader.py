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


Data Files Requiring Individual Loading Functions:
- Blood_Chemistry___Hematology
- Current_Biospecimen_Analysis_Results
- IUSM_ASSAY_DEV_CATALOG
- IUSM_CATALOG



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
    patno_event_data = {}  # Will store {(patno, event_id): {'SEX': sex, 'COHORT': cohort}}
    
    # Process files one by one to avoid loading all data at once
    for file_path in matching_files:
        try:
            # Read only the necessary columns for the base dataframe
            df_base = pd.read_csv(file_path)
            
            # Rename CLINICAL_EVENT to EVENT_ID if it exists
            if "CLINICAL_EVENT" in df_base.columns and "EVENT_ID" not in df_base.columns:
                df_base = df_base.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
            
            # Ensure we have the required columns for the base dataframe
            if not all(col in df_base.columns for col in ["PATNO", "EVENT_ID", "SEX", "COHORT"]):
                logger.warning(f"Missing required base columns in {file_path}")
                continue
            
            for _, row in df_base.iterrows():
                # Remove "PPMI-" prefix from PATNO if it exists
                patno = row["PATNO"]
                if isinstance(patno, str) and patno.startswith("PPMI-"):
                    patno = patno[5:]  # Remove first 5 characters ("PPMI-")
                
                event_id = row["EVENT_ID"]
                key = (patno, event_id)
                
                # Store SEX and COHORT values
                if key not in patno_event_data:
                    patno_event_data[key] = {
                        'SEX': row["SEX"],
                        'COHORT': row["COHORT"]
                    }
        except Exception as e:
            logger.error(f"Error reading base data from {file_path}: {e}")
    
    # Create a dictionary to collect all data
    # Structure: {(patno, event_id): {column_name: value}}
    data_dict = {}
    for key, base_data in patno_event_data.items():
        data_dict[key] = {
            'SEX': base_data['SEX'],
            'COHORT': base_data['COHORT']
        }
    
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


def load_biospecimen_data(data_path: str, source: str):
    """
    Load biospecimen data from the specified path.
    
    Args:
        data_path: Path to the data directory
        source: The data source (e.g., PPMI)
        
    Returns:
        A dictionary containing loaded biospecimen data
    """
    biospecimen_data = {}
    
    # Path to the Biospecimen folder
    biospecimen_path = os.path.join(data_path, "Biospecimen")
    
    if not os.path.exists(biospecimen_path):
        logger.warning(f"Biospecimen directory not found: {biospecimen_path}")
        return biospecimen_data
    
    # Load Project_151_pQTL_in_CSF data (non-batch-corrected)
    try:
        biospecimen_data["project_151_pQTL_CSF"] = load_project_151_pQTL_CSF(
            biospecimen_path, 
            batch_corrected=False
        )
        logger.info(f"Loaded Project_151_pQTL_in_CSF data: {len(biospecimen_data['project_151_pQTL_CSF'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project_151_pQTL_in_CSF data: {e}")
        biospecimen_data["project_151_pQTL_CSF"] = pd.DataFrame()
    
    # Load Project_151_pQTL_in_CSF data (batch-corrected)
    try:
        biospecimen_data["project_151_pQTL_CSF_batch_corrected"] = load_project_151_pQTL_CSF(
            biospecimen_path, 
            batch_corrected=True
        )
        logger.info(f"Loaded batch-corrected Project_151_pQTL_in_CSF data: {len(biospecimen_data['project_151_pQTL_CSF_batch_corrected'])} rows")
    except Exception as e:
        logger.error(f"Error loading batch-corrected Project_151_pQTL_in_CSF data: {e}")
        biospecimen_data["project_151_pQTL_CSF_batch_corrected"] = pd.DataFrame()
    
    # Load Metabolomic_Analysis_of_LRRK2 data (excluding CSF)
    try:
        biospecimen_data["metabolomic_lrrk2"] = load_metabolomic_lrrk2(
            biospecimen_path, 
            include_csf=False
        )
        logger.info(f"Loaded Metabolomic_Analysis_of_LRRK2 data: {len(biospecimen_data['metabolomic_lrrk2'])} rows")
    except Exception as e:
        logger.error(f"Error loading Metabolomic_Analysis_of_LRRK2 data: {e}")
        biospecimen_data["metabolomic_lrrk2"] = pd.DataFrame()
    
    # Load Metabolomic_Analysis_of_LRRK2_CSF data
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
    
    # Load Targeted___untargeted_MS-based_proteomics_of_urine_in_PD data
    try:
        biospecimen_data["urine_proteomics"] = load_urine_proteomics(biospecimen_path)
        logger.info(f"Loaded urine proteomics data: {len(biospecimen_data['urine_proteomics'])} rows")
    except Exception as e:
        logger.error(f"Error loading urine proteomics data: {e}")
        biospecimen_data["urine_proteomics"] = pd.DataFrame()
    
    # Load PPMI_Project_9000 data
    try:
        biospecimen_data["project_9000"] = load_project_9000(biospecimen_path)
        logger.info(f"Loaded Project 9000 data: {len(biospecimen_data['project_9000'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project 9000 data: {e}")
        biospecimen_data["project_9000"] = pd.DataFrame()
    
    # Load PPMI_Project_222 data
    try:
        biospecimen_data["project_222"] = load_project_222(biospecimen_path)
        logger.info(f"Loaded Project 222 data: {len(biospecimen_data['project_222'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project 222 data: {e}")
        biospecimen_data["project_222"] = pd.DataFrame()
    
    # Load PPMI_Project_196 data
    try:
        biospecimen_data["project_196"] = load_project_196(biospecimen_path)
        logger.info(f"Loaded Project 196 data: {len(biospecimen_data['project_196'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project 196 data: {e}")
        biospecimen_data["project_196"] = pd.DataFrame()
    
    # Load PPMI_Project_177 data
    try:
        biospecimen_data["project_177"] = load_project_177_untargeted_proteomics(biospecimen_path)
        logger.info(f"Loaded Project 177 data: {len(biospecimen_data['project_177'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project 177 data: {e}")
        biospecimen_data["project_177"] = pd.DataFrame()
    
    # Load Project_214_Olink data
    try:
        biospecimen_data["project_214"] = load_project_214_olink(biospecimen_path)
        logger.info(f"Loaded Project 214 data: {len(biospecimen_data['project_214'])} rows")
    except Exception as e:
        logger.error(f"Error loading Project 214 data: {e}")
        biospecimen_data["project_214"] = pd.DataFrame()
    
    # Load Current_Biospecimen_Analysis_Results data
    try:
        biospecimen_data["current_biospecimen"] = load_current_biospecimen_analysis(biospecimen_path)
        logger.info(f"Loaded Current Biospecimen Analysis data: {len(biospecimen_data['current_biospecimen'])} rows")
    except Exception as e:
        logger.error(f"Error loading Current Biospecimen Analysis data: {e}")
        biospecimen_data["current_biospecimen"] = pd.DataFrame()
    
    # TODO: Add loaders for other biospecimen data types
    
    return biospecimen_data

def main():
    """
    Example usage of biospecimen data loading functions.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Path to the data directory
    data_path = "./PPMI"
    biospecimen_path = os.path.join(data_path, "Biospecimen")
    
    # Test the load_current_biospecimen_analysis function directly
    print("\n" + "="*80)
    print("TESTING CURRENT BIOSPECIMEN ANALYSIS LOADER")
    print("="*80 + "\n")
    
    biospecimen_df = load_current_biospecimen_analysis(biospecimen_path)
    
    if not biospecimen_df.empty:
        rows, cols = biospecimen_df.shape
        print(f"Shape: {rows} rows × {cols} columns")
        
        # Print first few column names as a sample
        sample_columns = list(biospecimen_df.columns)[:10]
        print(f"Sample columns (first 10): {sample_columns}")
        
        # Count the number of columns with the BIO_ prefix
        bio_cols = [col for col in biospecimen_df.columns if col.startswith("BIO_")]
        print(f"\nNumber of columns with BIO_ prefix: {len(bio_cols)}")
        
        # Print first 10 rows
        print("\nFirst 10 rows:")
        print(biospecimen_df.head(10))
    else:
        print("No Current Biospecimen Analysis data found or loaded.")
    
    print("\n" + "="*80)
    
    # Optional: Also run the full biospecimen data loader
    print("\nRUNNING FULL BIOSPECIMEN DATA LOADER:")
    biospecimen_data = load_biospecimen_data(data_path, "PPMI")
    
    # Print summary of loaded data
    for key, df in biospecimen_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            rows, cols = df.shape
            print(f"{key}: Shape = {rows} rows × {cols} columns")
        else:
            print(f"{key}: No data loaded")

if __name__ == "__main__":
    main() 