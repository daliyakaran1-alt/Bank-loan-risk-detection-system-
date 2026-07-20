"""Data preprocessor pipeline controller."""
import os
import logging
import pandas as pd
from typing import Any

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class DataPreprocessor:
    """Controls the data preprocessing pipeline."""

    def __init__(self, loader: Any = None, validator: Any = None, cleaner: Any = None, transformer: Any = None) -> None:
        """Initialize the pipeline controller with its components.

        Args:
            loader (Any): Data loader instance.
            validator (Any): Data validator instance.
            cleaner (Any): Data cleaner instance.
            transformer (Any): Data transformer instance.
        """
        self.loader = loader
        self.validator = validator
        self.cleaner = cleaner
        self.transformer = transformer
        logger.info("Initialized DataPreprocessor")

    def run_pipeline(self, file_path: str, output_path: str = "data/processed/application_train_cleaned.csv") -> pd.DataFrame:
        """Run the data preprocessing pipeline.

        Args:
            file_path (str): Path to the input CSV file.
            output_path (str): Path to save the processed CSV file.

        Returns:
            pd.DataFrame: The fully preprocessed data.
        """
        logger.info(f"Starting preprocessing pipeline for {file_path}")
        
        # 1. Load data
        logger.info("[STEP] Dataset Loaded")
        if self.loader:
            data = self.loader.load(file_path)
        else:
            # Fallback to direct pandas loading if loader not provided
            data = pd.read_csv(file_path)
            
        # Optional validation
        if self.validator and not self.validator.validate(data):
            logger.warning("Data validation failed. Pipeline halting.")
            return None
            
        if not self.cleaner:
            from src.data.cleaner import DataCleaner
            self.cleaner = DataCleaner()
            
        # STEP 1: Dataset Inspection
        self.cleaner.inspect_dataset(data)
        
        # STEP 2: Data Quality Report
        self.cleaner.generate_quality_report(data, output_path="reports/data_quality_report.csv")
        
        # STEP 4 (Reordered for efficiency): Duplicate Handling
        data = self.cleaner.remove_duplicates(data)
        
        # STEP 3: Missing Value Handling
        data = self.cleaner.handle_missing_values(data, report_path="reports/high_missing_columns_report.csv")
        
        # STEP 5: Outlier Detection
        data = self.cleaner.handle_outliers(data, report_path="reports/outlier_report.csv")
        
        # STEP 6: Data Type Correction
        data = self.cleaner.correct_datatypes(data)
        
        # STEP 7: Target Validation
        self.cleaner.validate_target(data, output_path="reports/target_distribution.png", target_col="TARGET")
        
        # Optional Transformation
        if self.transformer:
            data = self.transformer.transform(data)
            
        # STEP 10: Output
        logger.info(f"Saving cleaned dataset to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        
        logger.info("[STEP] Cleaning Completed")
        logger.info("Preprocessing pipeline completed successfully.")
        return data

