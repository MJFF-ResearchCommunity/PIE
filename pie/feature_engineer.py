# pie/feature_engineer.py
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, PowerTransformer

logger = logging.getLogger(f"PIE.{__name__}")

class FeatureEngineer:
    """
    Handles feature engineering tasks on a given DataFrame.
    Operations include one-hot encoding, custom transformations,
    and handling of special column formats (e.g., pipe-separated).
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the FeatureEngineer with a DataFrame.

        Args:
            dataframe: The pandas DataFrame to engineer features on.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = dataframe.copy() # Work on a copy
        self.original_columns = list(dataframe.columns)
        self.engineered_feature_names: Dict[str, List[str]] = {} # To track new features from operations
        logger.info(f"FeatureEngineer initialized with DataFrame of shape: {self.df.shape}")

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame with engineered features."""
        return self.df.copy()

    def one_hot_encode(self,
                       columns: Optional[List[str]] = None,
                       prefix_sep: str = '_',
                       dummy_na: bool = False,
                       drop_first: bool = False,
                       max_categories_to_encode: int = 20,
                       min_frequency_for_category: Optional[float] = None, 
                       ignore_for_ohe: Optional[List[str]] = None, # Columns to explicitly ignore for OHE
                       auto_identify_threshold: int = 50 
                       ) -> 'FeatureEngineer':
        """
        Performs one-hot encoding on specified or auto-identified categorical columns.

        Args:
            columns: List of column names to encode. If None, attempts to auto-identify
                     categorical columns based on dtype ('object', 'category') and
                     `auto_identify_threshold`.
            prefix_sep: Separator for new column names (e.g., COL_Value).
            dummy_na: If True, add a column to indicate NaNs.
            drop_first: If True, drop the first category to reduce multicollinearity.
            max_categories_to_encode: Skip encoding for columns with more unique categories
                                       than this threshold.
            min_frequency_for_category: If set (e.g., 0.01), categories occurring less frequently
                                        than this proportion will be grouped into an 'Other' category
                                        before one-hot encoding. This applies only if >1 category exists.
            ignore_for_ohe: List of column names to explicitly ignore during one-hot encoding,
                               even if they are auto-identified or provided in `columns`.
                               'PATNO', 'EVENT_ID', 'COHORT' are added to this list by default
                               (case-insensitive matching for these defaults).
            auto_identify_threshold: If `columns` is None, only object/category columns with
                                     fewer unique values than this threshold are encoded.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        # Prepare the comprehensive list of columns to ignore for OHE
        default_ignores_upper = {'PATNO', 'EVENT_ID', 'COHORT'}
        
        current_ignore_list = list(ignore_for_ohe) if ignore_for_ohe is not None else []
        
        current_ignore_list_upper = {col.upper() for col in current_ignore_list}
        for default_ignore_upper in default_ignores_upper:
            if default_ignore_upper not in current_ignore_list_upper:
                original_casing_found = False
                # Try to find original casing from DataFrame columns for these defaults
                for df_col_name in self.df.columns:
                    if df_col_name.upper() == default_ignore_upper:
                        current_ignore_list.append(df_col_name)
                        original_casing_found = True
                        break
                if not original_casing_found: # If not found in df, add the uppercase default
                    current_ignore_list.append(default_ignore_upper)

        final_ignore_list_for_ohe_dedup = []
        seen_upper_ignores = set()
        for item in current_ignore_list:
            if item.upper() not in seen_upper_ignores:
                seen_upper_ignores.add(item.upper())
                final_ignore_list_for_ohe_dedup.append(item)
        
        logger.debug(f"Columns to ignore for OHE (case-insensitive matching for defaults): {final_ignore_list_for_ohe_dedup}")

        if columns is None:
            logger.info("Auto-identifying categorical columns for one-hot encoding...")
            candidate_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            columns_to_encode = []
            for col in candidate_cols:
                # Check against the final deduplicated ignore list (case-insensitive)
                if any(col.upper() == ignore_col.upper() for ignore_col in final_ignore_list_for_ohe_dedup):
                    logger.info(f"Auto-OHE: Skipping '{col}' as it is in the OHE ignore list.")
                    continue
                nunique = self.df[col].nunique(dropna=False) 
                if nunique <= auto_identify_threshold:
                    columns_to_encode.append(col)
                else:
                    logger.info(f"Skipping auto-OHE for '{col}': {nunique} unique values > threshold {auto_identify_threshold}.")
            logger.info(f"Auto-identified for OHE (after ignores): {columns_to_encode}")
        else:
            explicit_cols_to_encode = []
            for col_name in columns:
                if col_name not in self.df.columns:
                    logger.warning(f"Column '{col_name}' specified for OHE not found in DataFrame. Skipping.")
                    continue
                # Check against the final deduplicated ignore list (case-insensitive)
                if any(col_name.upper() == ignore_col.upper() for ignore_col in final_ignore_list_for_ohe_dedup):
                    logger.info(f"OHE: Skipping explicitly requested '{col_name}' as it is in the OHE ignore list.")
                    continue
                explicit_cols_to_encode.append(col_name)
            columns_to_encode = explicit_cols_to_encode
            logger.info(f"User-specified columns for OHE (after ignores): {columns_to_encode}")

        encoded_cols_map: Dict[str, List[str]] = {}

        for col in columns_to_encode:
            if col not in self.df.columns: # Should be caught above, but double check
                logger.warning(f"Column '{col}' not found for one-hot encoding. Skipping.")
                continue

            nunique = self.df[col].nunique(dropna=not dummy_na) 
            if nunique > max_categories_to_encode:
                logger.warning(f"Skipping one-hot encoding for '{col}': {nunique} unique categories > max {max_categories_to_encode}.")
                continue
            
            current_df_col = self.df[col]

            if min_frequency_for_category is not None and nunique > 1:
                counts = current_df_col.value_counts(normalize=True, dropna=False) 
                infrequent_categories = counts[counts < min_frequency_for_category].index
                if len(infrequent_categories) > 0 and len(infrequent_categories) < len(counts): 
                    logger.info(f"For column '{col}', grouping {len(infrequent_categories)} infrequent categories into '_OTHER_'.")
                    col_series_modified = current_df_col.copy()
                    col_series_modified[current_df_col.isin(infrequent_categories)] = '_OTHER_'
                    current_df_col_for_dummies = col_series_modified
                else:
                    current_df_col_for_dummies = current_df_col
            else:
                current_df_col_for_dummies = current_df_col

            logger.info(f"One-hot encoding column: '{col}' (Unique values for encoding: {current_df_col_for_dummies.nunique(dropna=False)})")
            dummies = pd.get_dummies(current_df_col_for_dummies, prefix=col, prefix_sep=prefix_sep,
                                     dummy_na=dummy_na, drop_first=drop_first)
            
            encoded_cols_map[col] = dummies.columns.tolist()

            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(columns=[col], inplace=True)
            logger.debug(f"Dropped original column '{col}' after OHE. New shape: {self.df.shape}")

        if encoded_cols_map:
             self.engineered_feature_names['one_hot_encoded'] = self.engineered_feature_names.get('one_hot_encoded', []) + \
                                                               [item for sublist in encoded_cols_map.values() for item in sublist]
        return self

    def handle_pipe_separated_column(self,
                                     column_name: str,
                                     strategy: str = 'multi_hot', # 'first', 'count', 'multi_hot'
                                     max_unique_values_for_multi_hot: int = 30,
                                     prefix: Optional[str] = None
                                     ) -> 'FeatureEngineer':
        """
        Handles columns with pipe-separated values.

        Args:
            column_name: The name of the column to process.
            strategy:
                'first': Take the first value.
                'count': Create a new column with the count of pipe-separated items.
                'multi_hot': Create new binary columns for each unique value found across
                             all rows (multi-hot encoding). Limited by max_unique_values_for_multi_hot.
            max_unique_values_for_multi_hot: Max unique values to create columns for in 'multi_hot'.
            prefix: Prefix for new columns created by 'multi_hot' or 'count'. Defaults to column_name.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for pipe-separation handling. Skipping.")
            return self

        if prefix is None:
            prefix = column_name
        
        new_feature_names = []

        if strategy == 'first':
            new_col_name = f"{prefix}_first"
            self.df[new_col_name] = self.df[column_name].astype(str).apply(lambda x: x.split('|')[0] if pd.notna(x) and x else np.nan)
            new_feature_names.append(new_col_name)
            logger.info(f"Created '{new_col_name}' by taking first value from '{column_name}'.")
        
        elif strategy == 'count':
            new_col_name = f"{prefix}_count"
            self.df[new_col_name] = self.df[column_name].astype(str).apply(lambda x: len(x.split('|')) if pd.notna(x) and x else 0)
            new_feature_names.append(new_col_name)
            logger.info(f"Created '{new_col_name}' with count of items in '{column_name}'.")

        elif strategy == 'multi_hot':
            all_values = set()
            self.df[column_name].dropna().astype(str).apply(lambda x: all_values.update(val.strip() for val in x.split('|') if val.strip()))
            
            unique_values = sorted(list(all_values))
            if len(unique_values) > max_unique_values_for_multi_hot:
                logger.warning(f"Skipping multi-hot for '{column_name}': {len(unique_values)} unique values > max {max_unique_values_for_multi_hot}. Consider increasing threshold or using a different strategy.")
                return self

            logger.info(f"Multi-hot encoding '{column_name}' for values: {unique_values}")
            for val in unique_values:
                new_col_name = f"{prefix}_{val.replace(' ', '_').replace('-', '_').lower()}" 
                self.df[new_col_name] = self.df[column_name].astype(str).apply(lambda x: 1 if pd.notna(x) and val in [v.strip() for v in x.split('|')] else 0)
                new_feature_names.append(new_col_name)
        else:
            logger.warning(f"Unknown strategy '{strategy}' for pipe-separated column '{column_name}'. Skipping.")
            return self
        
        if new_feature_names:
            self.engineered_feature_names[f'pipe_handled_{column_name}'] = new_feature_names
        return self

    def apply_custom_transformation(self,
                                    column_name: str,
                                    func: Callable[[pd.Series], pd.Series],
                                    new_column_name: Optional[str] = None) -> 'FeatureEngineer':
        """
        Applies a custom function to a specific column.

        Args:
            column_name: The column to transform.
            func: A function that takes a pandas Series and returns a pandas Series.
            new_column_name: Name for the new transformed column. If None, modifies in place.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for custom transformation. Skipping.")
            return self
        
        try:
            transformed_series = func(self.df[column_name])
            op_name = f'custom_transform_{func.__name__}_{column_name}'
            if new_column_name:
                self.df[new_column_name] = transformed_series
                logger.info(f"Applied custom function '{func.__name__}' to '{column_name}', new column: '{new_column_name}'.")
                self.engineered_feature_names[op_name] = [new_column_name]
            else:
                self.df[column_name] = transformed_series
                logger.info(f"Applied custom function '{func.__name__}' in-place to '{column_name}'.")
        except Exception as e:
            logger.error(f"Error applying custom function '{func.__name__}' to '{column_name}': {e}", exc_info=True)
        return self

    def scale_numeric_features(self,
                               columns: Optional[List[str]] = None,
                               scaler_type: str = 'standard', 
                               **scaler_params) -> 'FeatureEngineer':
        """
        Scales numeric features using StandardScaler or MinMaxScaler.

        Args:
            columns: List of numeric column names to scale. If None, scales all numeric columns
                     (excluding known ID columns like PATNO, EVENT_ID, and COHORT if numeric).
            scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
            **scaler_params: Additional parameters for the scaler.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            # Exclude common ID/target columns if they are numeric
            columns_to_scale = [
                col for col in numeric_cols 
                if col.upper() not in ['PATNO', 'EVENT_ID', 'COHORT']
            ]
            if not columns_to_scale:
                 logger.info("No numeric columns found to scale (or only ID/target columns).")
                 return self
            logger.info(f"Auto-identifying numeric columns for scaling: {columns_to_scale}")
        else:
            columns_to_scale = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]
            if len(columns_to_scale) != len(columns):
                missing_or_non_numeric = set(columns) - set(columns_to_scale)
                logger.warning(f"Columns not found or non-numeric, skipped for scaling: {missing_or_non_numeric}")

        if not columns_to_scale:
            logger.info("No valid numeric columns provided or identified for scaling.")
            return self

        if scaler_type == 'standard':
            scaler = StandardScaler(**scaler_params)
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler(**scaler_params)
        else:
            logger.error(f"Unknown scaler_type: {scaler_type}. Choose 'standard' or 'minmax'.")
            return self

        logger.info(f"Scaling columns using {scaler_type}: {columns_to_scale}")
        for col in columns_to_scale:
            if self.df[col].notna().any(): 
                 self.df[col] = scaler.fit_transform(self.df[[col]])
            else:
                 logger.warning(f"Column '{col}' contains all NaN values. Skipping scaling for this column.")
        return self

    def engineer_polynomial_features(self,
                                     columns: Optional[List[str]] = None,
                                     degree: int = 2,
                                     interaction_only: bool = False,
                                     include_bias: bool = False) -> 'FeatureEngineer':
        """
        Generates polynomial and interaction features for numeric columns.

        Args:
            columns: List of numeric column names to use for generating features.
                     If None, uses all numeric columns (excluding PATNO, EVENT_ID, COHORT).
            degree: The degree of the polynomial features.
            interaction_only: If True, only interaction features are produced.
            include_bias: If True, include a bias column.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            columns_to_process = [
                col for col in numeric_cols 
                if col.upper() not in ['PATNO', 'EVENT_ID', 'COHORT']
            ]
            if not columns_to_process:
                 logger.info("No numeric columns found for polynomial features (or only ID/target columns).")
                 return self
            logger.info(f"Auto-identifying numeric columns for polynomial features: {columns_to_process}")
        else:
            columns_to_process = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]
            if len(columns_to_process) != len(columns):
                missing_or_non_numeric = set(columns) - set(columns_to_process)
                logger.warning(f"Columns not found or non-numeric, skipped for polynomial features: {missing_or_non_numeric}")
        
        if not columns_to_process:
            logger.info("No valid numeric columns to process for polynomial features.")
            return self

        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        
        data_for_poly = self.df[columns_to_process].copy()
        imputed_values = {}
        for col in columns_to_process:
            if data_for_poly[col].isnull().any():
                median_val = data_for_poly[col].median()
                data_for_poly[col].fillna(median_val, inplace=True)
                imputed_values[col] = median_val
                logger.debug(f"NaNs in '{col}' for polynomial features imputed with median ({median_val}).")

        poly_features = poly.fit_transform(data_for_poly)
        poly_feature_names = poly.get_feature_names_out(input_features=columns_to_process)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=self.df.index)
        
        newly_added_poly_cols = []
        for col_name in poly_df.columns:
            if col_name == '1' and include_bias: 
                self.df[f"poly_bias"] = poly_df[col_name]
                newly_added_poly_cols.append(f"poly_bias")
            elif col_name not in self.df.columns: 
                self.df[col_name] = poly_df[col_name]
                newly_added_poly_cols.append(col_name)
            elif col_name in columns_to_process:
                pass # Original column, already present or handled if degree 1 etc.

        if newly_added_poly_cols:
            logger.info(f"Engineered {len(newly_added_poly_cols)} polynomial/interaction features from {columns_to_process}. Examples: {newly_added_poly_cols[:5]}")
            self.engineered_feature_names['polynomial_features'] = self.engineered_feature_names.get('polynomial_features', []) + newly_added_poly_cols
        return self

    def transform_numeric_distribution(self,
                                 column_name: str,
                                 new_column_name: Optional[str] = None,
                                 transform_type: str = 'log', 
                                 add_constant_for_log_sqrt: Optional[float] = None 
                                 ) -> 'FeatureEngineer':
        """
        Applies a mathematical transformation to change the distribution of a numeric feature.

        Args:
            column_name: The numeric column to transform.
            new_column_name: Name for the new transformed column. If None, modifies in place.
            transform_type: Type of transformation. Options: 'log', 'sqrt', 'box-cox', 'yeo-johnson'.
            add_constant_for_log_sqrt: Constant added before 'log' or 'sqrt'.

        Returns:
            The FeatureEngineer instance for chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for distribution transformation. Skipping.")
            return self
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            logger.warning(f"Column '{column_name}' is not numeric. Skipping distribution transformation.")
            return self

        original_series = self.df[column_name].copy()
        transformed_series = None
        op_name = f'dist_transform_{transform_type}_{column_name}'

        if add_constant_for_log_sqrt is not None:
            original_series = original_series + add_constant_for_log_sqrt
            logger.debug(f"Added constant {add_constant_for_log_sqrt} to '{column_name}' before {transform_type} transform.")

        if transform_type == 'log':
            if (original_series <= 0).any():
                logger.warning(f"Column '{column_name}' (after constant) contains non-positive values. Log transform may result in NaNs/errors. Min: {original_series.min()}")
            transformed_series = np.log(original_series.replace(-np.inf, np.nan).replace(np.inf, np.nan)) 
            transformed_series.replace(-np.inf, np.nan, inplace=True) 
        elif transform_type == 'sqrt':
            if (original_series < 0).any():
                logger.warning(f"Column '{column_name}' (after constant) contains negative values. Sqrt transform will result in NaNs.")
            transformed_series = np.sqrt(original_series)
        elif transform_type in ['box-cox', 'yeo-johnson']:
            pt = PowerTransformer(method=transform_type, standardize=False) 
            col_data = original_series.dropna().to_frame()
            if col_data.empty:
                logger.warning(f"Column '{column_name}' is all NaNs after pre-processing for PowerTransform. Skipping.")
                return self
            if transform_type == 'box-cox' and (col_data[column_name] <= 0).any():
                logger.warning(f"Box-Cox for '{column_name}' requires positive values. Min: {col_data[column_name].min()}. Skipping.")
                return self
            
            try:
                transformed_data = pt.fit_transform(col_data)
                transformed_series = pd.Series(transformed_data.flatten(), index=col_data.index)
            except ValueError as e:
                logger.error(f"Error during {transform_type} for '{column_name}': {e}. Skipping.")
                return self
        else:
            logger.error(f"Unknown transform_type: {transform_type}. Skipping.")
            return self

        if transformed_series is not None:
            target_col = new_column_name if new_column_name else column_name
            self.df[target_col] = transformed_series 
            logger.info(f"Applied '{transform_type}' to '{column_name}'. Result in '{target_col}'.")
            if new_column_name:
                 self.engineered_feature_names[op_name] = [new_column_name]
        return self

    def get_engineered_feature_summary(self) -> Dict[str, Any]:
        """Returns a summary of engineered features."""
        summary = {
            "total_original_columns": len(self.original_columns),
            "total_current_columns": self.df.shape[1],
            "newly_engineered_features_count": sum(len(v) for v in self.engineered_feature_names.values()),
            "engineered_operations": {}
        }
        for op_type, names in self.engineered_feature_names.items():
            summary["engineered_operations"][op_type] = {
                "count": len(names),
                "features": names[:10] + (["..."] if len(names) > 10 else []) 
            }
        return summary