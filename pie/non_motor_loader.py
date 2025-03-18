import glob
import os
import pandas as pd
import numpy as np

# List of file prefixes to search for in the Non-motor_Assessments folder
FILE_PREFIXES = [
    "Benton_Judgement",
    "Clock_Drawing",
    "Cognitive_Categorization",
    "Cognitive_Change",
    "Epworth_Sleepiness_Scale",
    "Geriatric_Depression_Scale",
    "Hopkins_Verbal_Learning_Test",
    "IDEA_Cognitive_Screen",
    "Letter_-_Number_Sequencing",
    "Lexical_Fluency",
    "Modified_Boston_Naming_Test",
    "Modified_Semantic_Fluency",
    "Montreal_Cognitive_Assessment",
    "Neuro_QoL",
    "PDAQ-27",
    "QUIP-Current-Short",
    "REM_Sleep_Behavior_Disorder_Questionnaire",
    "SCOPA-AUT",
    "State-Trait_Anxiety_Inventory",
    "Symbol_Digit_Modalities",
    "Trail_Making",
    "University_of_Pennsylvania_Smell_Identification"
]

def deduplicate_columns(df: pd.DataFrame, duplicate_columns: list[str]) -> pd.DataFrame:
    """
    Given a list of column *base names*, deduplicate them by combining the columns
    that may have appeared as colName_x and colName_y in the merged DataFrame.

    For each base name in duplicate_columns:
      - Look for <col>_x and <col>_y in df.columns.
      - Create or overwrite <col> based on row-by-row logic:
          1. If both values are NaN/empty, result is empty.
          2. If one is empty, use the non-empty value.
          3. If both are non-empty and same => use that one.
          4. If both are non-empty and differ => combine with '|' => "val1|val2"
      - Drop the <col>_x and <col>_y columns if they exist.

    :param df: The merged DataFrame containing possible duplicates.
    :param duplicate_columns: List of column base names that may have duplicates.
    :return: Updated DataFrame with deduplicated columns.
    """
    for col in duplicate_columns:
        col_x = f"{col}_x"
        col_y = f"{col}_y"

        # Only proceed if both variants actually exist in the DataFrame
        if col_x in df.columns and col_y in df.columns:
            # Create or overwrite col if it doesn't already exist
            if col not in df.columns:
                df[col] = np.nan

            # Perform row-by-row logic
            def combine_values(row):
                v1 = row[col_x]
                v2 = row[col_y]

                # Normalize "empty" or NaN
                # E.g., treat None / np.nan / empty string as empty
                is_empty_1 = pd.isna(v1) or v1 == ""
                is_empty_2 = pd.isna(v2) or v2 == ""

                if is_empty_1 and is_empty_2:
                    return np.nan
                elif is_empty_1 and not is_empty_2:
                    return v2  # v1 empty, v2 has data
                elif not is_empty_1 and is_empty_2:
                    return v1  # v1 has data, v2 empty
                else:
                    # Both are non-empty
                    # Check if they're the same
                    if str(v1) == str(v2):
                        return v1
                    else:
                        return f"{v1}|{v2}"

            df[col] = df.apply(combine_values, axis=1)
            # Drop the now-unneeded columns
            df.drop(columns=[col_x, col_y], inplace=True)
    return df


def sanitize_suffixes_in_df(df: pd.DataFrame) -> None:
    """
    Rename columns in df if they already end with '_x' or '_y',
    so that Pandas won't clash when merging with suffixes=('_x', '_y').
    Example: If a CSV already has a column named 'SUB_EVENT_ID_x', Pandas
    merging would try to rename it again to 'SUB_EVENT_ID_x_x', causing
    a MergeError. We'll rename 'SUB_EVENT_ID_x' to something like 'SUB_EVENT_ID_x_col'.
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            # Remove the last two chars and append _col to make it unique
            base = col[:-2]    # e.g. 'SUB_EVENT_ID'
            new_col = base + "_col"
            # If that new_col also exists, keep incrementing
            count = 1
            while new_col in df.columns or new_col in rename_map.values():
                new_col = f"{base}_col{count}"
                count += 1
                
            rename_map[col] = new_col

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def load_ppmi_non_motor_assessments(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files in the specified folder, searching for any file name that
    starts with one of the known FILE_PREFIXES. If a DataFrame has both PATNO and EVENT_ID,
    it will merge on [PATNO, EVENT_ID]. Otherwise, if the DataFrame lacks EVENT_ID,
    it will merge on PATNO only, effectively replicating any static data across all events
    for a patient.

    After merging, we call deduplicate_columns() to handle any known duplicate columns.

    :param folder_path: Path to 'Non-motor_Assessments' folder containing CSV files.
    :return: A merged pd.DataFrame with columns combined accordingly. Returns an
             empty DataFrame if no files are successfully loaded.
    """
    df_merged = None
    found_any_file = False
    
    # Get all CSV files in the target folder and its subdirectories
    all_csv_files = list(glob.iglob("**/*.csv", root_dir=folder_path, recursive=True))
    
    for prefix in FILE_PREFIXES:
        # Gather all files that start with the prefix (filename only; strip directory path)
        matching_files = [f for f in all_csv_files if f.split("/")[-1].startswith(prefix)]
        
        if not matching_files:
            # Include the full path in the error message
            expected_path = os.path.join(folder_path, f"{prefix}.csv")
            print(f"[ERROR] No CSV file found for prefix: {prefix} (expected at path like: {expected_path})")
            continue
        
        for filename in matching_files:
            csv_file = os.path.join(folder_path, filename)
            try:
                df_temp = pd.read_csv(csv_file)
                found_any_file = True
            except Exception as e:
                print(f"[ERROR] Could not read file '{csv_file}': {e}")
                continue
            
            # Sanitize any columns already ending in _x or _y
            sanitize_suffixes_in_df(df_temp)
            
            # If df_merged is empty, just set df_merged = df_temp
            if df_merged is None:
                df_merged = df_temp
            else:
                # Also sanitize df_merged before merging
                sanitize_suffixes_in_df(df_merged)
                
                # Determine which columns to merge on:
                # - Always merge on PATNO if present
                # - Merge on EVENT_ID only if BOTH frames have an EVENT_ID column
                merge_keys = ["PATNO"]
                if "EVENT_ID" in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    merge_keys = ["PATNO", "EVENT_ID"]
                
                # Merge with suffixes and immediately deduplicate
                df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer", suffixes=('_x', '_y'))
                
                # Columns that might appear in multiple DataFrames
                columns_to_deduplicate = [
                    "PAG_NAME", "INFODT", "ORIG_ENTRY", "LAST_UPDATE", "COHORT", "REC_ID"
                    # Add more columns as needed
                ]
                
                # Deduplicate immediately after each merge
                df_merged = deduplicate_columns(df_merged, columns_to_deduplicate)
    
    # If nothing loaded successfully, return empty DataFrame
    if not found_any_file or df_merged is None:
        print("[WARNING] No matching non-motor assessment CSV files were successfully loaded. Returning empty DataFrame.")
        return pd.DataFrame()
    
    return df_merged


def main():
    """
    Example usage of load_ppmi_non_motor_assessments:
    Loads and merges all non-motor assessment files from the PPMI/Non-motor_Assessments folder.
    """
    path_to_non_motor_assessments = "./PPMI/Non-motor_Assessments"
    
    # Print all CSV files in the PPMI directory to help debug
    print("[INFO] Listing all CSV files in the Non-motor_Assessments directory:")
    if os.path.exists(path_to_non_motor_assessments):
        for root, dirs, files in os.walk(path_to_non_motor_assessments):
            for file in files:
                if file.lower().endswith('.csv'):
                    print(f"  - {os.path.join(root, file)}")
    else:
        print(f"[WARNING] Directory not found: {path_to_non_motor_assessments}")
    
    df_non_motor = load_ppmi_non_motor_assessments(path_to_non_motor_assessments)
    print(df_non_motor.head(25))  # Show first rows to see merge results
    df_non_motor.to_csv("ppmi_non_motor_assessments.csv", index=False)

if __name__ == "__main__":
    main() 