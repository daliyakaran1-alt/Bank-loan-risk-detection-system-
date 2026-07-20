"""
Data Cleaner module.

Provides the DataCleaner class to perform production-ready dataset inspection,
cleaning, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from core.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class DataCleaner:
    """Handles data cleaning operations like missing values, duplicates, and outliers."""

    def __init__(self):
        """Initializes the DataCleaner."""
        pass

    def inspect_dataset(self, df: pd.DataFrame) -> None:
        """
        STEP 1: Dataset Inspection
        Safely inspect the dataset and print key metadata.
        
        Args:
            df (pd.DataFrame): The dataset to inspect.
        """
        logger.info("--- Dataset Inspection ---")
        logger.info(f"Number of rows: {df.shape[0]}")
        logger.info(f"Number of columns: {df.shape[1]}")
        logger.info(f"Data types:\n{df.dtypes.value_counts()}")
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        
        duplicate_rows = df.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicate_rows}")
        
        missing_values = df.isnull().sum().sum()
        logger.info(f"Total missing values: {missing_values}")
        logger.info("--------------------------")

    def generate_quality_report(self, df: pd.DataFrame, output_path: str) -> None:
        """
        STEP 2: Data Quality Report
        Generates a complete data quality report and saves to CSV.
        
        Args:
            df (pd.DataFrame): The dataset.
            output_path (str): Path to save the report CSV.
        """
        logger.info("Generating Data Quality Report...")
        
        report_data = []
        
        for col in df.columns:
            total_missing = df[col].isnull().sum()
            percent_missing = (total_missing / len(df)) * 100
            unique_vals = df[col].nunique()
            dtype = str(df[col].dtype)
            
            # Determine column category
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_vals == 2:
                    col_type = "Boolean"
                else:
                    col_type = "Numerical"
            else:
                col_type = "Categorical"
                
            is_identifier = "Yes" if unique_vals == len(df) else "No"
            is_target = "Yes" if col.upper() == "TARGET" else "No"
            
            report_data.append({
                "Column": col,
                "Data Type": dtype,
                "Column Type": col_type,
                "Total Missing": total_missing,
                "Percentage Missing": f"{percent_missing:.2f}%",
                "Unique Values": unique_vals,
                "Potential Identifier": is_identifier,
                "Target Column": is_target
            })
            
        report_df = pd.DataFrame(report_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report_df.to_csv(output_path, index=False)
        logger.info(f"Data Quality Report saved to {output_path}")

    def handle_missing_values(self, df: pd.DataFrame, report_path: str) -> pd.DataFrame:
        """
        STEP 3: Missing Value Handling
        Imputes missing values (Median for numerical, Mode for categorical).
        Logs and reports columns with > 70% missing values.
        
        Args:
            df (pd.DataFrame): The dataset to clean.
            report_path (str): Path to save the high missing columns report.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        logger.info("Handling missing values...")
        
        high_missing_cols = []
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct > 0.70:
                high_missing_cols.append({
                    "Column": col,
                    "Missing Percentage": f"{missing_pct*100:.2f}%",
                    "Recommendation": "Review for potential removal due to > 70% missing data."
                })
                # Do NOT remove immediately, as requested.
                
            # Imputation
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    
        # Save report for high missing columns
        if high_missing_cols:
            high_missing_df = pd.DataFrame(high_missing_cols)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            high_missing_df.to_csv(report_path, index=False)
            logger.info(f"Found {len(high_missing_cols)} columns with > 70% missing data. Report saved to {report_path}")
            
        logger.info("Missing Values Filled")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 4: Duplicate Handling
        Removes duplicate rows from the dataset safely.

        Args:
            df (pd.DataFrame): The dataset to clean.

        Returns:
            pd.DataFrame: The dataset without duplicates.
        """
        logger.info("Checking for duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        
        logger.info(f"Duplicates Removed: {removed_rows} rows")
        return df

    def handle_outliers(self, df: pd.DataFrame, report_path: str) -> pd.DataFrame:
        """
        STEP 5: Outlier Detection
        Implements IQR and Z-score methods. Marks outliers in boolean flags.
        Does not remove outliers. Generates report.

        Args:
            df (pd.DataFrame): The dataset.
            report_path (str): Path to save the outlier report.

        Returns:
            pd.DataFrame: The dataset with outlier flags.
        """
        logger.info("Detecting outliers...")
        outlier_summary = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip boolean-like or ID columns
            if df[col].nunique() <= 2 or col.upper() in ["SK_ID_CURR", "TARGET"]:
                continue
                
            # IQR Method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            df[f'outlier_iqr_{col}'] = iqr_outliers.astype(int)
            
            # Z-score Method
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = (df[col] - mean) / std
                z_outliers = (z_scores.abs() > 3)
                df[f'outlier_z_{col}'] = z_outliers.astype(int)
            else:
                z_outliers = pd.Series(False, index=df.index)
            
            num_iqr = iqr_outliers.sum()
            num_z = z_outliers.sum()
            
            if num_iqr > 0 or num_z > 0:
                outlier_summary.append({
                    "Column": col,
                    "IQR Outliers Count": num_iqr,
                    "Z-Score Outliers Count": num_z
                })
                
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            outlier_df.to_csv(report_path, index=False)
            logger.info(f"Outliers Detected. Report saved to {report_path}")
            
        return df

    def correct_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 6: Data Type Correction
        Converts columns to optimal data types (Boolean, Category, Numeric) 
        to save memory and enforce correct types.

        Args:
            df (pd.DataFrame): The dataset.

        Returns:
            pd.DataFrame: The dataset with corrected data types.
        """
        logger.info("Correcting data types...")
        
        for col in df.columns:
            if col.startswith('outlier_'):
                continue
                
            # Convert object types with low cardinality to Category
            if df[col].dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:  # Arbitrary threshold for categorical
                    df[col] = df[col].astype('category')
                    
            # Convert numerics to smallest possible types
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                    df[col] = df[col].astype('int8')
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                    
        logger.info("Data types corrected.")
        return df

    def validate_target(self, df: pd.DataFrame, output_path: str, target_col: str = "TARGET") -> None:
        """
        STEP 7: Target Validation
        Validates the TARGET column, shows class distribution, and saves a chart.

        Args:
            df (pd.DataFrame): The dataset.
            output_path (str): Path to save the distribution chart.
            target_col (str): The name of the target column.
        """
        logger.info(f"Validating Target Column: {target_col}")
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset!")
            return
            
        counts = df[target_col].value_counts()
        total = len(df)
        
        # Assuming 1 is default, 0 is non-default
        non_default_count = counts.get(0, 0)
        default_count = counts.get(1, 0)
        
        non_default_pct = (non_default_count / total) * 100
        default_pct = (default_count / total) * 100
        
        logger.info(f"Class Distribution -> Non-Default (0): {non_default_pct:.2f}%, Default (1): {default_pct:.2f}%")
        
        # Plotting
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Non-Default (0)', 'Default (1)'], [non_default_count, default_count], color=['#059669', '#FF6B6B'])
        plt.title('Target Variable Distribution')
        plt.ylabel('Number of Applicants')
        plt.xlabel('Target Class')
        
        # Add labels
        for bar, pct in zip(bars, [non_default_pct, default_pct]):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{pct:.2f}%', va='bottom', ha='center')
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Target distribution chart saved to {output_path}")

