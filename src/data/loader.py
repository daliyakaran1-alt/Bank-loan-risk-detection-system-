import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.exceptions import RinMitraException

logger = logging.getLogger(__name__)

class DataLoadError(RinMitraException):
    """Exception raised for errors during data loading."""
    pass

class DataLoader:
    """
    Professional data loading layer for Rin Mitra.
    
    Handles safe ingestion of datasets from multiple formats, validates paths,
    measures execution time, and provides dataset introspection methods.
    """
    
    def __init__(self, file_path: str | Path):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str | Path): Path to the dataset file.
        """
        self.file_path = Path(file_path)
        self.dataframe: Optional[pd.DataFrame] = None

    def _validate_path(self) -> None:
        """Validates that the file path exists and is a file."""
        logger.info(f"Validating path: {self.file_path}")
        if not self.file_path.exists():
            msg = f"File not found at path: {self.file_path}"
            logger.error(msg)
            raise DataLoadError(msg)
        if not self.file_path.is_file():
            msg = f"Path exists but is not a file: {self.file_path}"
            logger.error(msg)
            raise DataLoadError(msg)

    def load_csv(self, **kwargs) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        
        Args:
            **kwargs: Additional arguments to pass to pandas.read_csv.
            
        Returns:
            pd.DataFrame: The loaded dataset.
            
        Raises:
            DataLoadError: If loading fails.
        """
        logger.info(f"Method Start: load_csv for {self.file_path}")
        start_time = time.time()
        self._validate_path()
        
        try:
            self.dataframe = pd.read_csv(self.file_path, **kwargs)
            exec_time = time.time() - start_time
            logger.info(f"Method Success: load_csv completed in {exec_time:.2f} seconds. Rows: {len(self.dataframe)}")
            return self.dataframe
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: load_csv failed after {exec_time:.2f} seconds. Error: {str(e)}")
            raise DataLoadError(f"Failed to load CSV: {str(e)}") from e

    def load_excel(self, **kwargs) -> pd.DataFrame:
        """Placeholder for future Excel loading."""
        logger.info("Method Start: load_excel")
        start_time = time.time()
        self._validate_path()
        try:
            # Future implementation: self.dataframe = pd.read_excel(self.file_path, **kwargs)
            raise NotImplementedError("Excel loading not yet implemented.")
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: load_excel failed after {exec_time:.2f} seconds. Error: {str(e)}")
            raise DataLoadError(f"Failed to load Excel: {str(e)}") from e

    def load_parquet(self, **kwargs) -> pd.DataFrame:
        """Placeholder for future Parquet loading."""
        logger.info("Method Start: load_parquet")
        start_time = time.time()
        self._validate_path()
        try:
            # Future implementation: self.dataframe = pd.read_parquet(self.file_path, **kwargs)
            raise NotImplementedError("Parquet loading not yet implemented.")
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: load_parquet failed after {exec_time:.2f} seconds. Error: {str(e)}")
            raise DataLoadError(f"Failed to load Parquet: {str(e)}") from e

    def load_json(self, **kwargs) -> pd.DataFrame:
        """Placeholder for future JSON loading."""
        logger.info("Method Start: load_json")
        start_time = time.time()
        self._validate_path()
        try:
            # Future implementation: self.dataframe = pd.read_json(self.file_path, **kwargs)
            raise NotImplementedError("JSON loading not yet implemented.")
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Method Failure: load_json failed after {exec_time:.2f} seconds. Error: {str(e)}")
            raise DataLoadError(f"Failed to load JSON: {str(e)}") from e

    def load(self, **kwargs) -> pd.DataFrame:
        """
        Auto-detects file format by extension and calls the appropriate loader.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        ext = self.file_path.suffix.lower()
        if ext == '.csv':
            return self.load_csv(**kwargs)
        elif ext in ('.xls', '.xlsx'):
            return self.load_excel(**kwargs)
        elif ext == '.parquet':
            return self.load_parquet(**kwargs)
        elif ext == '.json':
            return self.load_json(**kwargs)
        else:
            raise DataLoadError(f"Unsupported file extension: {ext}")

    def _ensure_loaded(self) -> None:
        """Ensures that the dataframe is loaded before introspection."""
        if self.dataframe is None:
            raise DataLoadError("Data has not been loaded yet. Call load() first.")

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Returns the first n rows of the dataset."""
        self._ensure_loaded()
        return self.dataframe.head(n)

    def shape(self) -> Tuple[int, int]:
        """Returns the shape of the dataset (rows, columns)."""
        self._ensure_loaded()
        return self.dataframe.shape

    def columns(self) -> List[str]:
        """Returns the list of columns in the dataset."""
        self._ensure_loaded()
        return self.dataframe.columns.tolist()

    def data_types(self) -> pd.Series:
        """Returns the data types of the columns."""
        self._ensure_loaded()
        return self.dataframe.dtypes

    def memory_usage(self) -> pd.Series:
        """Returns the memory usage of each column in bytes."""
        self._ensure_loaded()
        return self.dataframe.memory_usage(deep=True)

    def summary(self) -> pd.DataFrame:
        """Returns descriptive statistics for numerical columns."""
        self._ensure_loaded()
        return self.dataframe.describe()
