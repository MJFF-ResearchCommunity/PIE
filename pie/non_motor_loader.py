import glob
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(f"PIE.{__name__}")

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
    "Neuro_QoL", # Note: Neuro_QoL also appears in Motor. Prefixes will ensure correct loading.
    "PDAQ-27",
    "QUIP-Current-Short",
    "REM_Sleep_Behavior_Disorder_Questionnaire",
    "SCOPA-AUT",
    "State-Trait_Anxiety_Inventory",
    "Symbol_Digit_Modalities",
    "Trail_Making",
    "University_of_Pennsylvania_Smell_Identification"
]

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
                else: 
                    s_v1, s_v2 = str(v1), str(v2)
                    if s_v1 == s_v2:
                        return v1 
                    else:
                        try:
                            f_v1 = float(v1)
                            f_v2 = float(v2)
                            if np.isclose(f_v1, f_v2): return v1
                        except (ValueError, TypeError):
                            pass 
                        return f"{s_v1}|{s_v2}"

            df[base_col_name] = df.apply(combine_values, axis=1)
            df.drop(columns=[col_x, col_y], inplace=True)
        elif has_x: 
            logger.debug(f"Renaming {col_x} to {base_col_name}")
            df.rename(columns={col_x: base_col_name}, inplace=True)
        elif has_y: 
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
        logger.warning(f"Non-Motor Assessments: Cannot aggregate by {group_cols} as one or more are missing. Returning original DataFrame.")
        return df

    df_copy = df.copy()
    if 'PATNO' in df_copy.columns:
        df_copy['PATNO'] = df_copy['PATNO'].astype(str)

    if not df_copy.duplicated(subset=group_cols).any():
        return df_copy

    logger.info(
        "Non-Motor Assessments: Consolidating rows with duplicate (PATNO, EVENT_ID) pairs. "
        "Non-null values for other columns will be pipe-separated if different."
    )

    agg_cols = [col for col in df.columns if col not in group_cols]
    if not agg_cols:
        return df_copy.drop_duplicates(subset=group_cols, keep='first')

    df_indexed = df_copy.set_index(group_cols)

    grouped = df_indexed.groupby(level=group_cols)
    nunique_df = grouped[agg_cols].nunique()
    result_df = grouped[agg_cols].first()

    pipe_separated_stats = {}

    for col in agg_cols:
        multi_value_groups_mask = nunique_df[col] > 1
        if not multi_value_groups_mask.any():
            continue

        num_affected_groups = multi_value_groups_mask.sum()
        if num_affected_groups > 0:
            pipe_separated_stats[col] = num_affected_groups

        multi_value_group_indices = nunique_df.index[multi_value_groups_mask]
        
        rows_for_col_agg_mask = df_indexed.index.isin(multi_value_group_indices)
        df_subset_for_col = df_indexed.loc[rows_for_col_agg_mask, [col]]

        if df_subset_for_col.empty:
            continue

        def string_agg_slow(series: pd.Series) -> str:
            unique_strings = series.dropna().astype(str).unique()
            return "|".join(sorted(unique_strings))

        slow_agg_results = df_subset_for_col.groupby(level=group_cols)[col].agg(string_agg_slow)

        # If the target column is not already an object/string type, cast it.
        # This prevents FutureWarning about incompatible dtypes.
        if result_df[col].dtype != 'object' and not pd.api.types.is_string_dtype(result_df[col].dtype):
            result_df[col] = result_df[col].astype(object)

        result_df.loc[slow_agg_results.index, col] = slow_agg_results
        
    df_aggregated = result_df.reset_index()

    if pipe_separated_stats:
        logger.info("Summary of pipe-separated columns for Non-Motor Assessments:")
        sorted_stats = sorted(pipe_separated_stats.items(), key=lambda item: item[1], reverse=True)
        
        for i, (col, count) in enumerate(sorted_stats):
            logger.info(f"  - Column '{col}': {count} groups had multiple values.")
            if i < 3: # Log examples for the top 3
                first_offending_group_index = nunique_df.index[nunique_df[col] > 1][0]
                conflicting_values = df_indexed.loc[first_offending_group_index, col].dropna().astype(str).unique()
                logger.info(f"    - Example for group {first_offending_group_index}: values were {list(conflicting_values)}")
    
    ordered_cols = group_cols + [col for col in df.columns if col in df_aggregated.columns and col not in group_cols]
    final_ordered_cols = [col for col in ordered_cols if col in df_aggregated.columns]
    return df_aggregated[final_ordered_cols]


def load_ppmi_non_motor_assessments(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for non-motor assessments.
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
                logger.debug(f"Loading non-motor assessment file: {csv_file_path}")
                df_temp = pd.read_csv(csv_file_path, low_memory=False)
                found_any_file = True
            except Exception as e:
                logger.error(f"Could not read file '{csv_file_path}': {e}")
                continue

            if "PATNO" not in df_temp.columns:
                logger.warning(f"File {csv_file_path} is missing PATNO column, skipping.")
                continue
            
            df_temp['PATNO'] = df_temp['PATNO'].astype(str)
            
            if df_merged is None:
                df_merged = df_temp
            else:
                if 'PATNO' in df_merged.columns:
                     df_merged['PATNO'] = df_merged['PATNO'].astype(str)
                
                merge_keys = ["PATNO"]
                if "EVENT_ID" in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    merge_keys.append("EVENT_ID")
                elif "EVENT_ID" in df_merged.columns and "EVENT_ID" not in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (it lacks EVENT_ID).")
                elif "EVENT_ID" not in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (df_merged lacks EVENT_ID).")

                try:
                    df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer", suffixes=('_x', '_y'))
                    df_merged = _general_deduplicate_suffixed_columns(df_merged)
                except Exception as e:
                    logger.error(f"Error merging {os.path.basename(csv_file_path)} into non-motor df_merged: {e}")
                    logger.error(f"df_merged columns: {df_merged.columns.tolist()}")
                    logger.error(f"df_temp columns: {df_temp.columns.tolist()}")
                    logger.error(f"Merge keys: {merge_keys}")
                    continue
    
    if not found_any_file or df_merged is None or df_merged.empty:
        logger.warning("No matching non-motor assessment CSV files were successfully loaded or merged. Returning empty DataFrame.")
        return pd.DataFrame()
    
    if "EVENT_ID" in df_merged.columns:
        logger.debug("Non-motor assessments: Aggregating rows to ensure unique (PATNO, EVENT_ID) pairs...")
        df_merged = _aggregate_by_patno_eventid(df_merged)
    else:
        logger.warning("EVENT_ID column not found in the final merged non-motor assessments DataFrame. "
                       "Ensuring PATNO uniqueness only if duplicates exist.")
        if "PATNO" in df_merged.columns and df_merged.duplicated(subset=["PATNO"]).any():
             logger.info(
                "Non-Motor Assessments: Consolidating rows with duplicate PATNO "
                "by combining unique non-null values for other columns."
             )
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
             else: 
                 df_merged = df_merged.drop_duplicates(subset=["PATNO"], keep='first')
        
    logger.info(f"Final loaded non-motor assessments shape: {df_merged.shape}")
    return df_merged


# def main():
#     """
#     Example usage of load_ppmi_non_motor_assessments:
#     Loads and merges all non-motor assessment files from the PPMI/Non-motor_Assessments folder.
#     """
#     path_to_non_motor_assessments = "./PPMI/Non-motor_Assessments"
    
#     # Print all CSV files in the PPMI directory to help debug
#     print("[INFO] Listing all CSV files in the Non-motor_Assessments directory:")
#     if os.path.exists(path_to_non_motor_assessments):
#         for root, dirs, files in os.walk(path_to_non_motor_assessments):
#             for file in files:
#                 if file.lower().endswith('.csv'):
#                     print(f"  - {os.path.join(root, file)}")
#     else:
#         print(f"[WARNING] Directory not found: {path_to_non_motor_assessments}")
    
#     df_non_motor = load_ppmi_non_motor_assessments(path_to_non_motor_assessments)
#     print(df_non_motor.head(25))  # Show first rows to see merge results
#     df_non_motor.to_csv("ppmi_non_motor_assessments.csv", index=False)

# if __name__ == "__main__":
#     main() 