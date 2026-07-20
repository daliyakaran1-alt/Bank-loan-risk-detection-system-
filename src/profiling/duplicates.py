"""
Module for computing duplicate analysis.
"""

import logging
import time
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class DuplicateResult:
    """Dataclass to hold duplicate analysis results."""
    total_duplicates: int
    duplicate_percentage: float
    duplicate_rows_indices: List[int]
    potential_duplicate_columns: Dict[str, str]


class DuplicateAnalysis:
    """
    Class to calculate duplicate rows, percentages, and potential duplicate columns.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DuplicateAnalysis class.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def analyze(self) -> DuplicateResult:
        """
        Analyze duplicate data in the dataset.

        Returns:
            DuplicateResult: The duplicate analysis results.

        Raises:
            ProfilingError: If an error occurs during analysis.
        """
        start_time = time.time()
        logger.info("Starting duplicate analysis.")
        
        try:
            num_rows = len(self.df)
            
            duplicates = self.df.duplicated()
            total_duplicates = int(duplicates.sum())
            duplicate_percentage = float(total_duplicates / num_rows * 100) if num_rows > 0 else 0.0
            
            duplicate_rows_indices = duplicates[duplicates].index.tolist()
            
            # Identify potential duplicate columns (columns with identical values)
            potential_duplicate_columns = {}
            columns = self.df.columns.tolist()
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    if self.df[col1].equals(self.df[col2]):
                        potential_duplicate_columns[col1] = col2

            result = DuplicateResult(
                total_duplicates=total_duplicates,
                duplicate_percentage=duplicate_percentage,
                duplicate_rows_indices=duplicate_rows_indices,
                potential_duplicate_columns=potential_duplicate_columns
            )
            
            exec_time = time.time() - start_time
            logger.info(f"Successfully analyzed duplicates in {exec_time:.4f} seconds.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Failed to analyze duplicates after {exec_time:.4f} seconds. Error: {e}")
            raise ProfilingError(f"Failed to analyze duplicates: {e}") from e
