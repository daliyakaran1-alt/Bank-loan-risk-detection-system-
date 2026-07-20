"""Main facade for data operations."""
import logging
from typing import Any

from .exporter import DataExporter
from .preprocessor import DataPreprocessor

try:
    from .loader import DataLoader
    from .validator import DataValidator
    from .cleaner import DataCleaner
    from .transformer import DataTransformer
except ImportError:
    # Using simple mock classes if the first half of the foundation is not yet available
    class DataLoader:
        def load(self, path): pass
    class DataValidator:
        def validate(self, data): return True
    class DataCleaner:
        def clean(self, data): return data
    class DataTransformer:
        def transform(self, data): return data

logger = logging.getLogger(__name__)

class DataManager:
    """Facade for data loading, validation, cleaning, transformation, and exporting."""

    def __init__(self) -> None:
        """Initialize the DataManager and its components."""
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.exporter = DataExporter()
        
        self.preprocessor = DataPreprocessor(
            loader=self.loader,
            validator=self.validator,
            cleaner=self.cleaner,
            transformer=self.transformer
        )
        logger.info("Initialized DataManager")

    def load_data(self, file_path: str) -> Any:
        """Load data from the specified path.
        
        Args:
            file_path (str): Path to load data from.
            
        Returns:
            Any: Loaded data.
        """
        logger.info(f"DataManager routing load request for {file_path}")
        return self.loader.load(file_path)

    def validate(self, data: Any) -> bool:
        """Validate the provided data.
        
        Args:
            data (Any): Data to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        logger.info("DataManager routing validate request")
        return self.validator.validate(data)

    def clean(self, data: Any) -> Any:
        """Clean the provided data.
        
        Args:
            data (Any): Data to clean.
            
        Returns:
            Any: Cleaned data.
        """
        logger.info("DataManager routing clean request")
        return self.cleaner.clean(data)

    def transform(self, data: Any) -> Any:
        """Transform the provided data.
        
        Args:
            data (Any): Data to transform.
            
        Returns:
            Any: Transformed data.
        """
        logger.info("DataManager routing transform request")
        return self.transformer.transform(data)

    def preprocess(self, file_path: str) -> Any:
        """Run the full preprocessing pipeline on the specified file.
        
        Args:
            file_path (str): Path to the input file.
            
        Returns:
            Any: Preprocessed data.
        """
        logger.info(f"DataManager routing preprocess request for {file_path}")
        return self.preprocessor.run_pipeline(file_path)

    def save(self, data: Any, file_path: str, format: str = 'csv') -> None:
        """Save the data to the specified path in the specified format.
        
        Args:
            data (Any): Data to save.
            file_path (str): Path to save data to.
            format (str): The format to save the data in ('csv', 'excel', 'parquet', 'json').
        """
        logger.info(f"DataManager routing save request for {file_path} in {format} format")
        
        format = format.lower()
        if format == 'csv':
            self.exporter.export_to_csv(data, file_path)
        elif format == 'excel':
            self.exporter.export_to_excel(data, file_path)
        elif format == 'parquet':
            self.exporter.export_to_parquet(data, file_path)
        elif format == 'json':
            self.exporter.export_to_json(data, file_path)
        else:
            logger.error(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported export format: {format}")
