import glob
import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(f"PIE.{__name__}")

FILE_PREFIXES = [
    "Age_at_visit",
    "Demographics",
    "Family_History",
    "iu_genetic_consensus",
    "Participant_Status",
    "PPMI_PD_Variants",
    "PPMI_Project_9001",
    "Socio-Economics",
    "Subject_Cohort_History"
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


def load_ppmi_subject_characteristics(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files in the specified folder, searching for any file name that
    starts with one of the known FILE_PREFIXES. If a DataFrame has both PATNO and EVENT_ID,
    it will merge on [PATNO, EVENT_ID]. Otherwise, if the DataFrame lacks EVENT_ID,
    it will merge on PATNO only, effectively replicating any static data across all events
    for a patient.

    After merging, we call deduplicate_columns() to handle any known duplicate columns.

    :param folder_path: Path to '_Subject_Characteristics' folder containing CSV files.
    :return: A merged pd.DataFrame with columns combined accordingly. Returns an
             empty DataFrame if no files are successfully loaded.
    """
    df_merged = None
    found_any_file = False
    # Get all CSV files in the target folder
    all_csv_files = list(glob.iglob("**/*.csv", root_dir=folder_path, recursive=True))
    for prefix in FILE_PREFIXES:
        # Gather all files that start with the prefix (filename only; strip directory path)
        matching_files = [f for f in all_csv_files if f.split("/")[-1].startswith(prefix)]
        if not matching_files:
            logger.warning(f"No CSV file found for prefix: {prefix}")
            continue
        for filename in matching_files:
            csv_file = os.path.join(folder_path, filename)
            try:
                df_temp = pd.read_csv(csv_file)
                found_any_file = True
            except Exception as e:
                logger.error(f"Could not read file '{csv_file}': {e}")
                continue
            # If df_merged is empty, just set df_merged = df_temp
            if df_merged is None:
                df_merged = df_temp
            else:
                # Determine which columns to merge on:
                # - Always merge on PATNO if present
                # - Merge on EVENT_ID only if BOTH frames have an EVENT_ID column
                merge_keys = ["PATNO"]
                if "EVENT_ID" in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    merge_keys = ["PATNO", "EVENT_ID"]
                df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer")
    # If nothing loaded successfully, return empty DataFrame
    if not found_any_file or df_merged is None:
        logger.warning("No matching CSV files were successfully loaded. Returning empty DataFrame.")
        return pd.DataFrame()
    columns_to_deduplicate = ["PAG_NAME","INFODT","ORIG_ENTRY","LAST_UPDATE","COHORT","REC_ID"]
    # Deduplicate the columns
    df_merged = deduplicate_columns(df_merged, columns_to_deduplicate)
    return df_merged


# def main():
#     """
#     Example usage of load_ppmi_subject_characteristics:
#     If some CSVs have only PATNO (no EVENT_ID),
#     those columns will be replicated across all event rows for that PATNO.
#     """
#     path_to_subject_characteristics = "./PPMI/_Subject_Characteristics"
#     df_subjects = load_ppmi_subject_characteristics(path_to_subject_characteristics)
#     logger.info(df_subjects.head(25))  # Show first rows to see merge results
#     df_subjects.to_csv("subject_characteristics.csv", index=False)

# if __name__ == "__main__":
#     main()
