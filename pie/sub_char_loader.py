import os
import pandas as pd

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

def load_ppmi_subject_characteristics(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files in the specified folder, searching for any file name that
    starts with one of the known FILE_PREFIXES. If a DataFrame has both PATNO and EVENT_ID,
    it will merge on [PATNO, EVENT_ID]. Otherwise, if the DataFrame lacks EVENT_ID,
    it will merge on PATNO only, effectively replicating any static data across all events
    for a patient.

    :param folder_path: Path to '_Subject_Characteristics' folder containing CSV files.
    :return: A merged pd.DataFrame with columns combined accordingly. Returns an
             empty DataFrame if no files are successfully loaded.
    """
    df_merged = None
    found_any_file = False

    # Get all CSV files in the target folder
    all_csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    for prefix in FILE_PREFIXES:
        # Gather all files that start with the prefix
        matching_files = [f for f in all_csv_files if f.startswith(prefix)]

        if not matching_files:
            print(f"[ERROR] No CSV file found for prefix: {prefix}")
            continue

        for filename in matching_files:
            csv_file = os.path.join(folder_path, filename)
            try:
                df_temp = pd.read_csv(csv_file)
                found_any_file = True
            except Exception as e:
                print(f"[ERROR] Could not read file '{csv_file}': {e}")
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
        print("[WARNING] No matching CSV files were successfully loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    return df_merged

# def main():
#     """
#     Example usage of load_ppmi_subject_characteristics:
#     If some CSVs have only PATNO (no EVENT_ID),
#     those columns will be replicated across all event rows for that PATNO.
#     """
#     path_to_subject_characteristics = "./PPMI/_Subject_Characteristics"
#     df_subjects = load_ppmi_subject_characteristics(path_to_subject_characteristics)
#     print(df_subjects.head(25))  # Show first rows to see merge results

# if __name__ == "__main__":
#     main()