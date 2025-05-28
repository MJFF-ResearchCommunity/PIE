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

def _sanitize_suffixes_in_df(df: pd.DataFrame) -> None:
    """
    Rename columns in df if they already end with '_x' or '_y',
    so that Pandas won't clash when merging with suffixes=('_x', '_y').
    Example: 'COL_x' -> 'COL_x_orig'.
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            new_col_candidate = f"{base}_{col[-1]}_orig" # e.g. SOME_COL_x_orig
            # Ensure new_col_candidate is unique
            count = 0
            new_col = new_col_candidate
            while new_col in df.columns or new_col in rename_map.values():
                count += 1
                new_col = f"{new_col_candidate}{count}"
            rename_map[col] = new_col
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.debug(f"Sanitized existing suffixed columns: {rename_map}")


def _general_deduplicate_suffixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies all columns with '_x' and '_y' suffixes, then merges them
    into a base column name.
    - If only one of col_x or col_y exists, it's renamed to base_col.
    - If both exist, their values are combined:
        - If one is NaN, the other is used.
        - If both are non-NaN and equal, one is used.
        - If both are non-NaN and different, they are pipe-separated.
    """
    if df.empty:
        return df

    cols_to_process = set()
    for col_name in df.columns:
        if col_name.endswith('_x'):
            cols_to_process.add(col_name[:-2])
        elif col_name.endswith('_y'):
            cols_to_process.add(col_name[:-2])

    if not cols_to_process:
        return df

    logger.debug(f"Deduplicating suffixed columns for bases: {cols_to_process}")

    for base_col_name in list(cols_to_process): # Iterate over a copy
        col_x = f"{base_col_name}_x"
        col_y = f"{base_col_name}_y"

        has_x = col_x in df.columns
        has_y = col_y in df.columns

        if has_x and has_y:
            logger.debug(f"Combining {col_x} and {col_y} into {base_col_name}")
            # Ensure base_col_name doesn't overwrite an existing non-suffixed column
            # that wasn't part of this _x/_y pair (should be rare if sanitization worked)
            if base_col_name in df.columns and base_col_name != col_x and base_col_name != col_y:
                 logger.warning(f"Base column {base_col_name} already exists. Combining _x/_y may overwrite it.")

            def combine_values(row):
                v1 = row[col_x]
                v2 = row[col_y]
                is_empty_1 = pd.isna(v1) or str(v1).strip() == ""
                is_empty_2 = pd.isna(v2) or str(v2).strip() == ""

                if is_empty_1 and is_empty_2: return np.nan
                elif is_empty_1: return v2
                elif is_empty_2: return v1
                else: # Both are non-empty
                    # Convert to string for comparison to handle mixed types robustly
                    s_v1, s_v2 = str(v1), str(v2)
                    if s_v1 == s_v2:
                        return v1 # Return original type if possible
                    else:
                        # Attempt to convert to a common numeric type if possible before string concatenation
                        try:
                            f_v1 = float(v1)
                            f_v2 = float(v2)
                            if np.isclose(f_v1, f_v2): return v1
                        except (ValueError, TypeError):
                            pass # Not both convertible to float, or one is string etc.
                        return f"{s_v1}|{s_v2}"

            df[base_col_name] = df.apply(combine_values, axis=1)
            df.drop(columns=[col_x, col_y], inplace=True)
        elif has_x: # Only _x exists
            logger.debug(f"Renaming {col_x} to {base_col_name}")
            df.rename(columns={col_x: base_col_name}, inplace=True)
        elif has_y: # Only _y exists
            logger.debug(f"Renaming {col_y} to {base_col_name}")
            df.rename(columns={col_y: base_col_name}, inplace=True)
    return df


def _aggregate_by_patno_eventid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures (PATNO, EVENT_ID) pairs are unique by grouping and aggregating.
    For non-grouping columns, it combines unique non-null string values with a pipe.
    If only one unique non-null value exists, it's used directly (attempting to keep original type).
    """
    if df.empty:
        return df

    group_cols = ["PATNO", "EVENT_ID"]
    if not all(gc in df.columns for gc in group_cols):
        logger.warning(f"Subject Characteristics: Cannot aggregate by {group_cols} as one or more are missing. Returning original DataFrame.")
        return df

    if not df.duplicated(subset=group_cols).any():
        return df

    logger.info(
        "Subject Characteristics: Consolidating rows with duplicate (PATNO, EVENT_ID) pairs. "
        "Non-null values for other columns will be pipe-separated if different."
    )

    def combine_series_values(series):
        # Drop NaN values and convert to string, then get unique values
        unique_non_null_strs = series.dropna().astype(str).unique()
        
        if len(unique_non_null_strs) == 0:
            return np.nan
        elif len(unique_non_null_strs) == 1:
            # Attempt to return the original value if it was unique and non-null
            # This helps preserve original types (e.g. int, float) if all were the same
            original_non_null_values = series.dropna()
            if original_non_null_values.nunique() == 1:
                return original_non_null_values.iloc[0]
            else: # Should be rare if unique_non_null_strs has 1 element
                return unique_non_null_strs[0]
        else:
            return "|".join(sorted(unique_non_null_strs))

    agg_cols = [col for col in df.columns if col not in group_cols]
    if not agg_cols:
        return df.drop_duplicates(subset=group_cols, keep='first')

    agg_dict = {col: combine_series_values for col in agg_cols}
    
    # Ensure PATNO is string before groupby if it's not already
    df_copy = df.copy()
    if 'PATNO' in df_copy.columns:
        df_copy['PATNO'] = df_copy['PATNO'].astype(str)

    df_aggregated = df_copy.groupby(group_cols, as_index=False).agg(agg_dict)
    
    ordered_cols = group_cols + [col for col in df.columns if col in df_aggregated.columns and col not in group_cols]
    # Handle case where a column might have been completely removed if it was only in group_cols before
    final_ordered_cols = [col for col in ordered_cols if col in df_aggregated.columns]

    return df_aggregated[final_ordered_cols]


def load_ppmi_subject_characteristics(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for subject characteristics.
    Ensures unique (PATNO, EVENT_ID) rows in the output by merging information.
    """
    df_merged = None
    found_any_file = False
    
    all_csv_files = list(glob.iglob(os.path.join(folder_path, "**/*.csv"), recursive=True))
    
    for prefix in FILE_PREFIXES:
        matching_files = [f for f in all_csv_files if os.path.basename(f).startswith(prefix)]
        if not matching_files:
            logger.debug(f"No CSV file found for prefix: {prefix} in {folder_path}")
            continue
            
        for csv_file_path in matching_files:
            try:
                logger.debug(f"Loading subject characteristics file: {csv_file_path}")
                df_temp = pd.read_csv(csv_file_path, low_memory=False)
                found_any_file = True
            except Exception as e:
                logger.error(f"Could not read file '{csv_file_path}': {e}")
                continue

            if "PATNO" not in df_temp.columns:
                logger.warning(f"File {csv_file_path} is missing PATNO column, skipping.")
                continue
            
            # Standardize PATNO to string early
            df_temp['PATNO'] = df_temp['PATNO'].astype(str)


            # Sanitize columns in df_temp before merging
            _sanitize_suffixes_in_df(df_temp)
            
            if df_merged is None:
                df_merged = df_temp
            else:
                # Ensure df_merged PATNO is string
                if 'PATNO' in df_merged.columns:
                     df_merged['PATNO'] = df_merged['PATNO'].astype(str)

                # Sanitize columns in df_merged before merging
                _sanitize_suffixes_in_df(df_merged)
                
                merge_keys = ["PATNO"]
                # Merge on EVENT_ID only if both frames have it
                if "EVENT_ID" in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    merge_keys.append("EVENT_ID")
                elif "EVENT_ID" in df_merged.columns and "EVENT_ID" not in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (it lacks EVENT_ID).")
                elif "EVENT_ID" not in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (df_merged lacks EVENT_ID).")


                try:
                    df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer", suffixes=('_x', '_y'))
                except Exception as e:
                    logger.error(f"Error merging {os.path.basename(csv_file_path)} into df_merged: {e}")
                    logger.error(f"df_merged columns: {df_merged.columns.tolist()}")
                    logger.error(f"df_temp columns: {df_temp.columns.tolist()}")
                    logger.error(f"Merge keys: {merge_keys}")
                    continue # Skip this file if merge fails

    if not found_any_file or df_merged is None or df_merged.empty:
        logger.warning("No matching subject characteristics CSV files were successfully loaded or merged. Returning empty DataFrame.")
        return pd.DataFrame()

    # 1. Resolve _x, _y suffixed columns resulting from merges
    logger.debug("Applying general suffixed column deduplication...")
    df_merged = _general_deduplicate_suffixed_columns(df_merged)

    # 2. Ensure (PATNO, EVENT_ID) uniqueness by aggregating rows
    # This step must happen after all files are merged and _x/_y columns are resolved.
    if "EVENT_ID" in df_merged.columns:
        logger.debug("Aggregating rows to ensure unique (PATNO, EVENT_ID) pairs for Subject Characteristics...")
        df_merged = _aggregate_by_patno_eventid(df_merged)
    else:
        logger.warning("EVENT_ID column not found in the final merged subject characteristics DataFrame. "
                       "Ensuring PATNO uniqueness only if duplicates exist.")
        if "PATNO" in df_merged.columns and df_merged.duplicated(subset=["PATNO"]).any():
             # For PATNO-only aggregation, we'll use the same logic
             logger.info(
                "Subject Characteristics: Consolidating rows with duplicate PATNO "
                "by combining unique non-null values for other columns."
             )
             # Temporarily rename PATNO for the groupby function if EVENT_ID is missing
             # This is a bit of a hack to reuse the same _aggregate_by_patno_eventid logic
             # Or better, adapt _aggregate_by_patno_eventid to handle single key
             
             # Simplified aggregation for PATNO only if EVENT_ID is missing
             def combine_patno_only_series(series):
                unique_non_null_strs = series.dropna().astype(str).unique()
                if len(unique_non_null_strs) == 0: return np.nan
                if len(unique_non_null_strs) == 1:
                    original_non_null_values = series.dropna()
                    if original_non_null_values.nunique() == 1:
                        return original_non_null_values.iloc[0]
                    return unique_non_null_strs[0]
                return "|".join(sorted(unique_non_null_strs))

             agg_cols_patno = [col for col in df_merged.columns if col != "PATNO"]
             if agg_cols_patno:
                 agg_dict_patno = {col: combine_patno_only_series for col in agg_cols_patno}
                 df_merged['PATNO'] = df_merged['PATNO'].astype(str)
                 df_merged = df_merged.groupby("PATNO", as_index=False).agg(agg_dict_patno)
             else: # Only PATNO column exists
                 df_merged = df_merged.drop_duplicates(subset=["PATNO"], keep='first')
        
    logger.info(f"Final loaded subject characteristics shape: {df_merged.shape}")
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
