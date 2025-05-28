# pie/data_analyzer.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import gc
from pathlib import Path

logger = logging.getLogger(f"PIE.{__name__}")

class DataReducer:
    """
    Analyzes a dictionary of DataFrames (as produced by DataLoader)
    to provide summary statistics and identify columns potentially safe to drop
    before merging, aiming to reduce memory usage.
    """

    DEFAULT_CONFIG = {
        "missing_threshold": 0.95,  # Drop if > 95% missing
        "single_value_threshold": 1.0, # Drop if 100% of non-NaN values are the same
        "low_variance_threshold": 0.01, # Threshold for std dev near zero (for numeric) - ADJUST AS NEEDED
        "high_cardinality_ratio": 0.9, # Consider dropping if unique values / rows > 0.9 (potential IDs) - USE WITH CAUTION
        "common_metadata_cols": ["REC_ID", "ORIG_ENTRY", "LAST_UPDATE", "PAG_NAME", "QUERY_ID", "QUERY_TEXT"], # Known metadata/operational columns
        "check_low_variance_numeric": True,
        "check_high_cardinality": False # Default to False as it can be risky
    }

    def __init__(self, data_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initializes the analyzer with the data dictionary.

        Args:
            data_dict: The dictionary of DataFrames (or nested dicts) from DataLoader.
            config: Optional dictionary to override default analysis thresholds.
        """
        self.data_dict = data_dict
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        logger.info(f"DataReducer initialized with config: {self.config}")

    def analyze(self) -> Dict[str, Any]:
        """
        Performs analysis on all DataFrames in the data_dict.

        Returns:
            A dictionary containing analysis results for each modality/table.
            Keys are modality names (or 'modality.table' for nested dicts),
            values are dictionaries with 'summary_stats' and 'drop_suggestions'.
        """
        analysis_report = {}
        for modality, data in self.data_dict.items():
            if isinstance(data, pd.DataFrame):
                logger.info(f"Analyzing DataFrame for modality: {modality}")
                analysis_report[modality] = self._analyze_dataframe(data, modality)
            elif isinstance(data, dict):
                logger.info(f"Analyzing nested dictionary for modality: {modality}")
                for table_name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        logger.info(f"Analyzing DataFrame for: {modality}.{table_name}")
                        report_key = f"{modality}.{table_name}"
                        analysis_report[report_key] = self._analyze_dataframe(df, modality, table_name)
                    else:
                         logger.warning(f"Skipping non-DataFrame item in {modality}: {table_name} ({type(df)})")
            else:
                logger.warning(f"Skipping non-DataFrame/non-dict item: {modality} ({type(data)})")
        return analysis_report

    def _analyze_dataframe(self, df: pd.DataFrame, modality_name: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes a single DataFrame."""
        report = {"summary_stats": {}, "drop_suggestions": {}}
        if df.empty:
            report["summary_stats"]["shape"] = (0, 0)
            report["summary_stats"]["info"] = "Empty DataFrame"
            report["drop_suggestions"]["reason"] = "Empty DataFrame"
            report["drop_suggestions"]["columns"] = []
            return report

        num_rows, num_cols = df.shape
        report["summary_stats"]["shape"] = (num_rows, num_cols)

        # Basic Info (dtypes, non-null counts)
        info_df = pd.DataFrame({
            'Dtype': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        info_df['Null Pct'] = info_df['Null Count'] / num_rows
        report["summary_stats"]["column_info"] = info_df.to_dict('index') # More structured than df.info() output

        # Descriptive Stats for Numeric Columns
        try:
            # Ignore errors for non-numeric types during describe
            numeric_stats = df.describe(include=np.number).transpose()
            report["summary_stats"]["numeric_summary"] = numeric_stats.to_dict('index')
        except Exception as e:
            report["summary_stats"]["numeric_summary"] = f"Error calculating numeric summary: {e}"

        # Value Counts for Object/Categorical (maybe top 5)
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        value_counts_summary = {}
        for col in object_cols:
            counts = df[col].value_counts(dropna=False)
            value_counts_summary[col] = counts.head(5).to_dict() # Top 5 + NaN count if present
            value_counts_summary[col]['_unique_count'] = df[col].nunique(dropna=False)
        report["summary_stats"]["categorical_summary"] = value_counts_summary


        # --- Drop Suggestions ---
        drop_suggestions = []
        reasons = {}

        for col in df.columns:
            # Skip primary keys
            if col in ["PATNO", "EVENT_ID"]:
                 continue

            # 1. High Missing Values
            missing_pct = info_df.loc[col, 'Null Pct']
            if missing_pct > self.config["missing_threshold"]:
                drop_suggestions.append(col)
                reasons[col] = f"High Missing % ({missing_pct:.2%})"
                continue # If dropped for missing, don't check other reasons

            # 2. Single Value (check non-null values)
            unique_vals_non_null = df[col].dropna().unique()
            if len(unique_vals_non_null) == 1:
                 # Check if this single value covers the threshold % of non-null data
                 # (Usually this will be 100% if len==1, but good practice)
                 val_counts = df[col].dropna().value_counts(normalize=True)
                 if not val_counts.empty and val_counts.iloc[0] >= self.config["single_value_threshold"]:
                     drop_suggestions.append(col)
                     reasons[col] = f"Single Value ('{unique_vals_non_null[0]}')"
                     continue

            # 3. Low Variance (Numeric)
            if self.config["check_low_variance_numeric"] and pd.api.types.is_numeric_dtype(df[col]):
                 std_dev = df[col].std()
                 # Check if std_dev is very close to zero (handling potential NaNs)
                 if pd.notna(std_dev) and std_dev < self.config["low_variance_threshold"]:
                      # Avoid dropping if it's just a single value column (already caught above)
                      if col not in reasons:
                           drop_suggestions.append(col)
                           reasons[col] = f"Near-Zero Variance (std={std_dev:.4f})"
                           continue

            # 4. High Cardinality (Potential Identifiers) - Use with caution!
            if self.config["check_high_cardinality"]:
                 unique_count = df[col].nunique()
                 if num_rows > 0 and (unique_count / num_rows) > self.config["high_cardinality_ratio"]:
                      # Check if it looks like an ID (e.g., mostly numbers or long strings)
                      # Simple check: if dtype is int/float or object with high mean string length
                      looks_like_id = False
                      if pd.api.types.is_numeric_dtype(df[col].dtype):
                           looks_like_id = True
                      elif df[col].dtype == 'object':
                           mean_len = df[col].dropna().astype(str).str.len().mean()
                           if pd.notna(mean_len) and mean_len > 10: # Arbitrary length threshold
                                looks_like_id = True

                      if looks_like_id and col not in reasons:
                           drop_suggestions.append(col)
                           reasons[col] = f"High Cardinality ({(unique_count / num_rows):.2%})"
                           continue

            # 5. Common Metadata Columns
            if col in self.config["common_metadata_cols"]:
                 if col not in reasons:
                      drop_suggestions.append(col)
                      reasons[col] = "Common Metadata"
                      continue


        report["drop_suggestions"]["columns"] = sorted(list(set(drop_suggestions))) # Unique & sorted
        report["drop_suggestions"]["reasons"] = reasons
        report["drop_suggestions"]["count"] = len(report["drop_suggestions"]["columns"])

        return report

    def get_drop_suggestions(self, analysis_report: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Extracts all drop suggestions from an analysis report.

        Args:
            analysis_report: The report generated by analyze(). If None, runs analyze().

        Returns:
            A dictionary mapping modality/table name to a list of columns suggested for dropping.
        """
        if analysis_report is None:
            analysis_report = self.analyze()

        all_suggestions = {}
        for key, report in analysis_report.items():
            if "drop_suggestions" in report and "columns" in report["drop_suggestions"]:
                all_suggestions[key] = report["drop_suggestions"]["columns"]
        return all_suggestions

    def apply_drops(self, drop_suggestions: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Applies the suggested column drops to the original data_dict.

        Args:
            drop_suggestions: Dictionary mapping modality/table name to list of columns to drop.

        Returns:
            A new dictionary with the specified columns removed from the DataFrames.
            The original data_dict remains unchanged.
        """
        import copy
        reduced_data_dict = copy.deepcopy(self.data_dict) # Work on a copy

        for key, columns_to_drop in drop_suggestions.items():
            if not columns_to_drop:
                continue

            target_data = None
            try:
                # Handle nested keys like "medical_history.Adverse_Event"
                if '.' in key:
                    modality, table_name = key.split('.', 1)
                    if modality in reduced_data_dict and isinstance(reduced_data_dict[modality], dict) and \
                       table_name in reduced_data_dict[modality] and isinstance(reduced_data_dict[modality][table_name], pd.DataFrame):

                        target_df = reduced_data_dict[modality][table_name]
                        cols_exist = [col for col in columns_to_drop if col in target_df.columns]
                        if cols_exist:
                             logger.info(f"Dropping {len(cols_exist)} columns from {key}: {cols_exist}")
                             reduced_data_dict[modality][table_name] = target_df.drop(columns=cols_exist)
                        else:
                             logger.debug(f"No columns to drop found in {key}")

                # Handle top-level keys like "subject_characteristics"
                elif key in reduced_data_dict and isinstance(reduced_data_dict[key], pd.DataFrame):
                     target_df = reduced_data_dict[key]
                     cols_exist = [col for col in columns_to_drop if col in target_df.columns]
                     if cols_exist:
                          logger.info(f"Dropping {len(cols_exist)} columns from {key}: {cols_exist}")
                          reduced_data_dict[key] = target_df.drop(columns=cols_exist)
                     else:
                          logger.debug(f"No columns to drop found in {key}")
                else:
                    logger.warning(f"Could not find DataFrame for key '{key}' in apply_drops")

            except Exception as e:
                logger.error(f"Error dropping columns for key '{key}': {e}")

        return reduced_data_dict

    def generate_report_str(self, analysis_report: Optional[Dict[str, Any]] = None) -> str:
         """Generates a human-readable summary string from the analysis report."""
         if analysis_report is None:
            analysis_report = self.analyze()

         report_lines = ["Data Analysis Report:"]
         report_lines.append("=" * 20)

         for key, report in analysis_report.items():
             report_lines.append(f"\n--- Analysis for: {key} ---")
             if "summary_stats" in report:
                 stats = report["summary_stats"]
                 shape = stats.get('shape', '(N/A, N/A)')
                 report_lines.append(f"  Shape: {shape[0]} rows, {shape[1]} columns")
                 # Add more summary details here if desired (e.g., total missing %)
             if "drop_suggestions" in report:
                 suggestions = report["drop_suggestions"]
                 count = suggestions.get('count', 0)
                 report_lines.append(f"  Drop Suggestions: {count} columns")
                 if count > 0:
                     reasons = suggestions.get('reasons', {})
                     cols = suggestions.get('columns', [])
                     # Show reason for each dropped column (or first few)
                     for i, col in enumerate(cols[:5]): # Show first 5
                         report_lines.append(f"    - {col} ({reasons.get(col, 'N/A')})")
                     if count > 5:
                         report_lines.append(f"    ... and {count - 5} more.")
             report_lines.append("-" * (len(key) + 20))

         return "\n".join(report_lines)

    def _aggregate_duplicates(self, df: pd.DataFrame, group_cols: List[str], df_name_for_log: str) -> pd.DataFrame:
        """
        Handles duplicate rows based on group_cols by taking the first non-null value
        for other columns. This ensures the DataFrame has unique entries for group_cols.
        """
        if df.empty or not df.duplicated(subset=group_cols).any():
            return df

        logger.warning(
            f"DataFrame '{df_name_for_log}' has duplicate {group_cols} entries. "
            f"Aggregating by taking the first non-null value for other columns to ensure uniqueness."
        )
        
        # Define a custom aggregation function that takes the first non-NaN value
        def first_valid(series):
            if series.notna().any():
                return series.dropna().iloc[0]
            return np.nan

        # Create an aggregation dictionary for all columns not in group_cols
        agg_dict = {
            col: first_valid for col in df.columns if col not in group_cols
        }
        
        # If there are no columns to aggregate (only group_cols exist), just drop duplicates
        if not agg_dict:
            return df.drop_duplicates(subset=group_cols, keep='first')

        # Perform the groupby and aggregation. as_index=False keeps group_cols as columns.
        df_aggregated = df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Ensure the column order is maintained as much as possible
        # Start with group_cols, then other original columns that are still present.
        final_columns = group_cols + [col for col in df.columns if col in df_aggregated.columns and col not in group_cols]
        
        return df_aggregated[final_columns]


    def merge_reduced_data(self,
                           reduced_data_dict: Dict[str, Any],
                           output_filename: str = "merged_reduced_data.csv"
                           ) -> pd.DataFrame:
        """
        Merges all DataFrames in the (reduced) data dictionary into a single DataFrame
        based on PATNO and EVENT_ID. Ensures each source DataFrame contributes at most
        one row per PATNO/EVENT_ID by aggregating duplicates.

        Args:
            reduced_data_dict: The dictionary of DataFrames, potentially reduced by apply_drops.
            output_filename: The name of the CSV file to save the merged DataFrame.

        Returns:
            A single merged DataFrame.
        """
        logger.info("Starting merge of reduced data...")

        prepared_dataframes_for_merge = [] # List of DataFrames ready for final merge
        all_patno_event_pairs = set()

        logger.info("Preparing DataFrames, collecting unique PATNO/EVENT_ID pairs, and handling within-table duplicates...")
        for modality_key, data_item in reduced_data_dict.items():
            dataframes_to_process = []
            if isinstance(data_item, pd.DataFrame):
                dataframes_to_process.append({'df': data_item, 'name_prefix': modality_key, 'original_name': modality_key})
            elif isinstance(data_item, dict): # Nested dictionary (e.g., medical_history)
                for table_name, df_table in data_item.items():
                    if isinstance(df_table, pd.DataFrame):
                        dataframes_to_process.append({'df': df_table, 'name_prefix': f"{modality_key}_{table_name}", 'original_name': f"{modality_key}.{table_name}"})

            for item_info in dataframes_to_process:
                df_original = item_info['df']
                name_prefix = item_info['name_prefix']
                original_df_name = item_info['original_name']

                if df_original.empty:
                    logger.info(f"Skipping empty DataFrame: {original_df_name}")
                    continue
                if "PATNO" not in df_original.columns or "EVENT_ID" not in df_original.columns:
                    logger.warning(f"DataFrame {original_df_name} is missing PATNO or EVENT_ID. Skipping.")
                    continue
                
                df_copy = df_original.copy()
                df_copy["PATNO"] = df_copy["PATNO"].astype(str)

                # --- Crucial step: Handle duplicates within this specific df_copy ---
                df_deduplicated = self._aggregate_duplicates(df_copy, ["PATNO", "EVENT_ID"], original_df_name)

                # Collect unique (PATNO, EVENT_ID) pairs from the now-deduplicated-within-table version
                current_pairs = set(map(lambda x: (x[0], x[1]), # PATNO is already string
                                        df_deduplicated[["PATNO", "EVENT_ID"]].drop_duplicates().itertuples(index=False, name=None)))
                all_patno_event_pairs.update(current_pairs)

                # Rename columns (except PATNO, EVENT_ID) using the name_prefix
                rename_map = {
                    col: f"{name_prefix}_{col}"
                    for col in df_deduplicated.columns if col not in ["PATNO", "EVENT_ID"]
                }
                df_deduplicated.rename(columns=rename_map, inplace=True)
                
                prepared_dataframes_for_merge.append(df_deduplicated)
                logger.debug(f"Prepared {original_df_name} for merging. Shape after dedup: {df_deduplicated.shape}. Columns: {df_deduplicated.columns.tolist()[:5]}...")
        
        if not all_patno_event_pairs:
            logger.warning("No PATNO/EVENT_ID pairs found across any DataFrames. Returning an empty DataFrame.")
            return pd.DataFrame()

        logger.info(f"Creating base DataFrame with {len(all_patno_event_pairs)} unique PATNO/EVENT_ID pairs.")
        merged_df = pd.DataFrame(list(all_patno_event_pairs), columns=["PATNO", "EVENT_ID"])
        merged_df["PATNO"] = merged_df["PATNO"].astype(str)
        # Optional: Ensure EVENT_ID is consistent type if necessary, e.g., merged_df["EVENT_ID"] = merged_df["EVENT_ID"].astype(str)

        del all_patno_event_pairs
        gc.collect()

        logger.info(f"Starting iterative merge of {len(prepared_dataframes_for_merge)} prepared DataFrames...")
        for i, df_to_merge in enumerate(prepared_dataframes_for_merge):
            # Attempt to get a meaningful name for logging from the prefixed columns
            df_name_for_log_parts = [col.split('_')[0] for col in df_to_merge.columns if col not in ["PATNO", "EVENT_ID"]]
            df_name_for_log = df_name_for_log_parts[0] if df_name_for_log_parts else "UnknownSource"
            
            logger.info(f"Merging DataFrame {i+1}/{len(prepared_dataframes_for_merge)} (derived from ~{df_name_for_log}) (Shape: {df_to_merge.shape})")
            
            if df_to_merge.empty or "PATNO" not in df_to_merge.columns or "EVENT_ID" not in df_to_merge.columns:
                logger.warning(f"Skipping a DataFrame as it's empty or missing merge keys post-preparation.")
                continue
            
            df_to_merge["PATNO"] = df_to_merge["PATNO"].astype(str)

            merged_df = pd.merge(merged_df, df_to_merge, on=["PATNO", "EVENT_ID"], how="left", suffixes=('', '_unexpected_dup'))
            
            # Check for _unexpected_dup columns which indicate issues if prefixes weren't unique enough
            unexpected_dup_cols = [col for col in merged_df.columns if col.endswith('_unexpected_dup')]
            if unexpected_dup_cols:
                logger.error(f"Unexpected duplicate columns found after merge step {i+1}: {unexpected_dup_cols}. "
                               "This may indicate an issue with column prefixing strategy.")
                # Example: To resolve, one might drop the _unexpected_dup columns or investigate the source of the clash
                # merged_df.drop(columns=unexpected_dup_cols, inplace=True)

            logger.debug(f"  Merged_df shape after step {i+1}: {merged_df.shape}")
            
            prepared_dataframes_for_merge[i] = None 
            del df_to_merge
            gc.collect()

        final_shape = merged_df.shape
        logger.info(f"Final merged DataFrame shape: {final_shape[0]} rows, {final_shape[1]} columns.")
        
        # Final check for duplicates in PATNO, EVENT_ID in the final merged_df
        if merged_df.duplicated(subset=["PATNO", "EVENT_ID"]).any():
            num_duplicates = merged_df.duplicated(subset=["PATNO", "EVENT_ID"]).sum()
            logger.error(f"CRITICAL: Final merged_df STILL contains {num_duplicates} duplicate (PATNO, EVENT_ID) pairs! Further investigation needed.")
            # As a last resort, one might do:
            # merged_df = merged_df.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep='first')
            # logger.warning("Applied final drop_duplicates on PATNO, EVENT_ID as a last resort.")
        else:
            logger.info("Confirmed: Final merged_df has unique (PATNO, EVENT_ID) pairs.")

        if output_filename:
            try:
                file_path = Path(output_filename)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                merged_df.to_csv(file_path, index=False)
                logger.info(f"Merged DataFrame saved to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save merged DataFrame to {output_filename}: {e}", exc_info=True)

        return merged_df

    def consolidate_cohort_columns(self, dataframe: pd.DataFrame,
                                   target_cohort_col_name: str = "COHORT") -> pd.DataFrame:
        """
        Consolidates multiple columns containing "COHORT" in their name into a single
        target COHORT column. For each row, the value for the new COHORT column
        will be the first non-null/non-empty value found across the identified
        source cohort columns.

        Args:
            dataframe: The input DataFrame (e.g., the output of merge_reduced_data).
            target_cohort_col_name: The name of the final, consolidated COHORT column.

        Returns:
            The DataFrame with a single consolidated COHORT column and original
            cohort-related columns (except the target if it pre-existed) dropped.
        """
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            logger.warning("Input DataFrame is empty or not a DataFrame. Skipping COHORT consolidation.")
            return dataframe

        df_copy = dataframe.copy()

        # Identify columns that contain "COHORT" (case-insensitive)
        cohort_related_cols = [
            col for col in df_copy.columns
            if "COHORT" in col.upper()
        ]

        # Exclude the target column itself if it's already one of the cohort_related_cols
        # and we intend to populate it. If it doesn't exist, it will be created.
        # If it exists and is part of cohort_related_cols, we use it as a source too.
        source_cohort_cols = [col for col in cohort_related_cols if col != target_cohort_col_name]
        if not source_cohort_cols and target_cohort_col_name in cohort_related_cols:
            # This means only the target_cohort_col_name itself matches "COHORT". No consolidation needed from others.
             if target_cohort_col_name in df_copy.columns:
                logger.info(f"Only target COHORT column '{target_cohort_col_name}' found. No other COHORT columns to consolidate.")
                return df_copy # Return as is if only the target column name exists
        elif not source_cohort_cols and target_cohort_col_name not in cohort_related_cols:
            # This means NO columns with "COHORT" in their name were found at all.
            logger.info("No columns containing 'COHORT' found. Skipping consolidation.")
            return df_copy


        # If the target column doesn't exist yet, or if it does and it's also a source,
        # we need to apply the logic.
        # The effective source columns are all 'cohort_related_cols'.
        # The target is 'target_cohort_col_name'.

        logger.info(f"Consolidating COHORT information from columns: {cohort_related_cols} into '{target_cohort_col_name}'.")

        def get_first_valid_cohort(row):
            for col in cohort_related_cols: # Check all identified cohort columns
                val = row[col]
                if pd.notna(val) and str(val).strip() != "":
                    return val
            return np.nan

        df_copy[target_cohort_col_name] = df_copy.apply(get_first_valid_cohort, axis=1)

        # Columns to drop: all identified cohort_related_cols EXCEPT the target_cohort_col_name
        cols_to_drop_after_consolidation = [
            col for col in cohort_related_cols if col != target_cohort_col_name and col in df_copy.columns
        ]

        if cols_to_drop_after_consolidation:
            logger.info(f"Dropping original cohort-related columns: {cols_to_drop_after_consolidation}")
            df_copy.drop(columns=cols_to_drop_after_consolidation, inplace=True)
        
        final_cohort_counts = df_copy[target_cohort_col_name].value_counts(dropna=False)
        logger.info(f"Value counts for the new '{target_cohort_col_name}' column:\n{final_cohort_counts}")

        return df_copy
