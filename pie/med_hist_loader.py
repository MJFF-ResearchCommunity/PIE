import glob
import os
import pandas as pd
import numpy as np

# Many medical history files cannot be merged, because data is recorded multiple times per visit,
# or on a timeline that doesn't correspond to the visits. Examples of the first are Adverse Event
# logs, where at the visit, multiple adverse events over the past period can be recorded. An
# example of the second is Concomitant Meds, which have start and end dates based on when the
# medication was taken. As a result, medical history tables should be kept separate.

MEDICAL_HISTORY_PREFIXES = [
    "Adverse_Event",
    "AV-133_Prodromal",
    "C05-05_PET_Imaging_Substudy",
    "Clinical_Diagnosis",
    "Clinical_Global_Impression",
    "Concomitant_Medication",
    "Determination_of_Freezing_and_Falls",
    "DPA-714_PET_Imaging_Substudy_Adverse_Event",
    "Early_Imaging",
    "Features_of_Parkinsonism",
    "Features_of_REM_Behavior_Disorder",
    "Gait_Substudy_Adverse_Event",
    "General_Physical_Exam",
    "Initiation_of_Dopaminergic_Therapy",
    "LEDD_Concomitant_Medication",
    "Medical_Conditions",
    "Neurological_Exam",
    "Other_Clinical_Features",
    "Participant_Global_Impression",
    "PD_Diagnosis_History",
    "Pregnancy_Test",
    "Primary_Clincial_Diagnosis",
    "Procedure_for_PD_Log",
    "Report_of_Pregnancy",
    "SVA2_PET_Imaging_Substudy",
    "Tau_Substudy",
    "Vital_Signs"
]

def sanitize_suffixes_in_df(df: pd.DataFrame) -> None:
    """
    Rename columns in df if they already end with '_x' or '_y'.
    For example:
      'SUB_EVENT_ID_x' -> 'SUB_EVENT_ID_x_col'
      'SUB_EVENT_ID_y' -> 'SUB_EVENT_ID_y_col'
    If 'SUB_EVENT_ID_x_col' also exists, keep incrementing => 'SUB_EVENT_ID_x_col1', etc.
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            base = col  # e.g. 'SUB_EVENT_ID_x'
            # Chop off the last two chars ('_x' or '_y') and append '_col'
            new_base = base[:-2] + "_col"  # e.g. 'SUB_EVENT_ID_col'
            new_col = new_base

            # If that new_col also exists, keep incrementing
            count = 1
            while new_col in df.columns or new_col in rename_map.values():
                new_col = f"{new_base}{count}"
                count += 1

            rename_map[col] = new_col

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def deduplicate_columns(df: pd.DataFrame, duplicate_columns: list[str]) -> pd.DataFrame:
    """
    For each <col> in duplicate_columns:
      If <col>_x and <col>_y both exist, combine them row-wise:
        - If both empty => NaN
        - If one is empty => use the other
        - If both differ => join with '|'
        - If both match => keep one
      Then drops the original <col>_x / <col>_y.
    """
    for col in duplicate_columns:
        col_x = f"{col}_x"
        col_y = f"{col}_y"

        if col_x in df.columns and col_y in df.columns:
            if col not in df.columns:
                df[col] = np.nan

            def combine_values(row):
                v1 = row[col_x]
                v2 = row[col_y]
                empty1 = pd.isna(v1) or v1 == ""
                empty2 = pd.isna(v2) or v2 == ""

                if empty1 and empty2:
                    return np.nan
                elif empty1 and not empty2:
                    return v2
                elif not empty1 and empty2:
                    return v1
                else:
                    return v1 if str(v1) == str(v2) else f"{v1}|{v2}"

            df[col] = df.apply(combine_values, axis=1)
            df.drop(columns=[col_x, col_y], inplace=True)

    return df


def load_ppmi_medical_history(folder_path: str) -> pd.DataFrame:
    """
    1) Lists all CSV files in 'folder_path' that start with any MEDICAL_HISTORY_PREFIX.
    2) For each CSV, read into df_temp.
         - sanitize_suffixes_in_df(df_temp), in case it is merged in the future
         - Store in a dict of tables, with the prefix as the key
    3) Return the dict or empty if no files found.
    """
    df_dict = {}
    found_any_file = False

    # Some columns that might appear in multiple DataFrames
    columns_to_deduplicate = [
        "PAG_NAME", "INFODT", "ORIG_ENTRY", "LAST_UPDATE", "COHORT", "REC_ID"
        # Add additional column names here if needed
    ]

    # Search all subdirectories too
    all_csv_files = list(glob.iglob("**/*.csv", root_dir=folder_path, recursive=True))

    for prefix in MEDICAL_HISTORY_PREFIXES:
        # Strip directory path, and look only at the filename for the prefix
        matching_files = [f for f in all_csv_files if f.split("/")[-1].startswith(prefix)]
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

            # 1) Rename any leftover _x / _y columns in df_temp
            sanitize_suffixes_in_df(df_temp)

            df_dict[prefix] = df_temp

    # Return empty DataFrame if no data was loaded
    if not found_any_file:
        print("[WARNING] No matching medical history CSV files were loaded - returning empty dict.")

    return df_dict


def main():
    path_to_med_history = "./PPMI/Medical_History"
    med_history = load_ppmi_medical_history(path_to_med_history)
    print(sorted(list(med_history.keys())))
if __name__ == "__main__":
    main()
