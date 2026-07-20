import logging
from dataclasses import dataclass, field
from typing import List, Set

from core.exceptions import RinMitraException

logger = logging.getLogger(__name__)

class SchemaValidationError(RinMitraException):
    """Exception raised for schema validation errors."""
    pass

@dataclass
class DataSchema:
    """
    Central schema registry for the Rin Mitra Enterprise Banking Risk Analytics Platform.
    
    This dataclass defines the expected structure, columns, and constraints for the
    Home Credit Default Risk dataset (and future datasets).
    """
    
    dataset_name: str = "Home Credit Default Risk"
    target_column: str = "TARGET"
    primary_identifier: str = "SK_ID_CURR"
    
    # Expected column categorizations
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    ignored_columns: List[str] = field(default_factory=list)
    
    # Minimal required columns for the dataset to be valid
    required_columns: Set[str] = field(default_factory=lambda: {"SK_ID_CURR", "TARGET"})
    
    def validate_columns(self, dataframe_columns: List[str]) -> bool:
        """
        Validates if the provided columns contain all required columns.
        
        Args:
            dataframe_columns (List[str]): List of columns present in the dataset.
            
        Returns:
            bool: True if validation passes, raises SchemaValidationError otherwise.
            
        Raises:
            SchemaValidationError: If required columns are missing.
        """
        logger.info(f"Validating columns for dataset: {self.dataset_name}")
        df_cols_set = set(dataframe_columns)
        missing = self.required_columns - df_cols_set
        
        if missing:
            msg = f"Schema validation failed. Missing required columns: {missing}"
            logger.error(msg)
            raise SchemaValidationError(msg)
            
        logger.info("Schema column validation passed successfully.")
        return True

    def validate_schema(self, dataframe) -> bool:
        """
        Full schema validation against a pandas DataFrame.
        
        Args:
            dataframe: The pandas DataFrame to validate.
            
        Returns:
            bool: True if schema is valid.
        """
        logger.info(f"Performing full schema validation for: {self.dataset_name}")
        self.validate_columns(dataframe.columns.tolist())
        
        logger.info("Full schema validation completed successfully.")
        return True
