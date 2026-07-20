import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd

from core.exceptions import RinMitraException
from src.data.schema import DataSchema

logger = logging.getLogger(__name__)

class DataValidationError(RinMitraException):
    """Exception raised for errors during dataset validation."""
    pass

@dataclass
class ValidationReport:
    """Structured report object containing the results of a dataset validation."""
    dataset_name: str
    rows: int
    columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    duplicate_columns: List[str]
    memory_usage_mb: float
    column_types: Dict[str, str]
    validation_status: str
    timestamp: str

class DataValidator:
    """
    Enterprise dataset validation layer for Rin Mitra.
    
    Responsible for validating data structure, identifying missing values,
    detecting duplicates, and generating comprehensive validation reports.
    """
    
    def __init__(self, schema: DataSchema):
        """
        Initialize the DataValidator with a schema.
        
        Args:
            schema (DataSchema): The central schema registry.
        """
        self.schema = schema
        self._report: Optional[ValidationReport] = None

    def validate_file(self, file_path: str | Path) -> bool:
        """
        Validates that a file exists and is accessible.
        
        Args:
            file_path: Path to the dataset file.
            
        Returns:
            bool: True if valid.
            
        Raises:
            DataValidationError: If the file does not exist.
        """
        logger.info("Method Start: validate_file")
        start_time = time.time()
        path = Path(file_path)
        
        if not path.exists() or not path.is_file():
            exec_time = time.time() - start_time
            msg = f"File validation failed: {path}"
            logger.error(f"Method Failure: validate_file failed after {exec_time:.2f}s. {msg}")
            raise DataValidationError(msg)
            
        exec_time = time.time() - start_time
        logger.info(f"Method Success: validate_file completed in {exec_time:.2f}s")
        return True

    def check_empty_dataset(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the dataset is empty.
        
        Args:
            dataframe: The pandas DataFrame.
            
        Returns:
            bool: True if not empty.
            
        Raises:
            DataValidationError: If dataset is empty.
        """
        logger.info("Method Start: check_empty_dataset")
        start_time = time.time()
        
        if dataframe.empty:
            exec_time = time.time() - start_time
            msg = "Dataset validation failed: The dataset is completely empty."
            logger.error(f"Method Failure: check_empty_dataset failed after {exec_time:.2f}s. {msg}")
            raise DataValidationError(msg)
            
        exec_time = time.time() - start_time
        logger.info(f"Method Success: check_empty_dataset completed in {exec_time:.2f}s")
        return True

    def check_duplicate_rows(self, dataframe: pd.DataFrame) -> int:
        """Counts duplicate rows in the dataset."""
        logger.info("Method Start: check_duplicate_rows")
        start_time = time.time()
        
        duplicates = int(dataframe.duplicated().sum())
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: check_duplicate_rows completed in {exec_time:.2f}s. Found {duplicates} duplicate rows.")
        return duplicates

    def check_duplicate_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """Identifies duplicate columns in the dataset."""
        logger.info("Method Start: check_duplicate_columns")
        start_time = time.time()
        
        cols = dataframe.columns.tolist()
        duplicates = list(set([x for x in cols if cols.count(x) > 1]))
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: check_duplicate_columns completed in {exec_time:.2f}s.")
        return duplicates

    def check_missing_values(self, dataframe: pd.DataFrame) -> Dict[str, int]:
        """Calculates missing values per column."""
        logger.info("Method Start: check_missing_values")
        start_time = time.time()
        
        missing = dataframe.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: check_missing_values completed in {exec_time:.2f}s.")
        return missing_dict

    def check_datatypes(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """Retrieves data types for all columns."""
        logger.info("Method Start: check_datatypes")
        start_time = time.time()
        
        dtypes = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: check_datatypes completed in {exec_time:.2f}s.")
        return dtypes

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        """Validates dataframe columns against the schema."""
        logger.info("Method Start: validate_columns")
        start_time = time.time()
        
        try:
            result = self.schema.validate_columns(dataframe.columns.tolist())
            exec_time = time.time() - start_time
            logger.info(f"Method Success: validate_columns completed in {exec_time:.2f}s.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: validate_columns failed after {exec_time:.2f}s. Error: {str(e)}")
            raise DataValidationError(f"Column validation failed: {str(e)}") from e

    def validate_schema(self, dataframe: pd.DataFrame) -> bool:
        """Performs full schema validation."""
        logger.info("Method Start: validate_schema")
        start_time = time.time()
        
        try:
            result = self.schema.validate_schema(dataframe)
            exec_time = time.time() - start_time
            logger.info(f"Method Success: validate_schema completed in {exec_time:.2f}s.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: validate_schema failed after {exec_time:.2f}s. Error: {str(e)}")
            raise DataValidationError(f"Schema validation failed: {str(e)}") from e

    def generate_validation_report(self, dataframe: pd.DataFrame) -> ValidationReport:
        """Generates a comprehensive validation report object."""
        logger.info("Method Start: generate_validation_report")
        start_time = time.time()
        
        if self._report is not None:
            return self._report
            
        rows, cols = dataframe.shape
        missing = self.check_missing_values(dataframe)
        dup_rows = self.check_duplicate_rows(dataframe)
        dup_cols = self.check_duplicate_columns(dataframe)
        dtypes = self.check_datatypes(dataframe)
        mem_usage = dataframe.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        status = "PASSED"
        try:
            self.validate_schema(dataframe)
        except Exception:
            status = "FAILED"
            
        self._report = ValidationReport(
            dataset_name=self.schema.dataset_name,
            rows=rows,
            columns=cols,
            missing_values=missing,
            duplicate_rows=dup_rows,
            duplicate_columns=dup_cols,
            memory_usage_mb=round(mem_usage, 2),
            column_types=dtypes,
            validation_status=status,
            timestamp=datetime.now().isoformat()
        )
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: generate_validation_report completed in {exec_time:.2f}s.")
        return self._report

    def validate_dataset(self, dataframe: pd.DataFrame) -> ValidationReport:
        """
        Executes the full validation pipeline on the dataset.
        
        Args:
            dataframe: The pandas DataFrame.
            
        Returns:
            ValidationReport: Structured report of the validation.
            
        Raises:
            DataValidationError: If critical checks fail.
        """
        logger.info("Method Start: validate_dataset")
        start_time = time.time()
        
        self.check_empty_dataset(dataframe)
        self.validate_schema(dataframe)
        report = self.generate_validation_report(dataframe)
        
        exec_time = time.time() - start_time
        logger.info(f"Method Success: validate_dataset completed in {exec_time:.2f}s.")
        return report
