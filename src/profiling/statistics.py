"""
Module for computing dataset statistics.
"""

import logging
import time
from typing import Any, Dict
from dataclasses import dataclass
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class StatisticsResult:
    """Dataclass to hold statistics results."""
    num_rows: int
    num_columns: int
    memory_usage_bytes: int
    dtypes: Dict[str, str]
    numeric_counts: Dict[str, int]
    unique_values: Dict[str, int]


class DatasetStatistics:
    """
    Class to calculate dataset statistics like rows, columns, memory usage, datatypes, counts, unique values.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DatasetStatistics class.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def calculate(self) -> StatisticsResult:
        """
        Calculate statistics for the dataset.

        Returns:
            StatisticsResult: The calculated statistics.

        Raises:
            ProfilingError: If an error occurs during calculation.
        """
        start_time = time.time()
        logger.info("Starting dataset statistics calculation.")
        
        try:
            num_rows, num_columns = self.df.shape
            memory_usage_bytes = self.df.memory_usage(deep=True).sum()
            dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            
            numeric_counts = {}
            for col in self.df.select_dtypes(include='number').columns:
                numeric_counts[col] = int(self.df[col].count())
                
            unique_values = {col: int(self.df[col].nunique()) for col in self.df.columns}

            result = StatisticsResult(
                num_rows=num_rows,
                num_columns=num_columns,
                memory_usage_bytes=int(memory_usage_bytes),
                dtypes=dtypes,
                numeric_counts=numeric_counts,
                unique_values=unique_values
            )
            
            exec_time = time.time() - start_time
            logger.info(f"Successfully calculated dataset statistics in {exec_time:.4f} seconds.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Failed to calculate dataset statistics after {exec_time:.4f} seconds. Error: {e}")
            raise ProfilingError(f"Failed to calculate statistics: {e}") from e
