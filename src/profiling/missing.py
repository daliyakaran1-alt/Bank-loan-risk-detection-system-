"""
Module for computing missing value analysis.
"""

import logging
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class MissingResult:
    """Dataclass to hold missing value analysis results."""
    total_missing: int
    missing_by_column: Dict[str, int]
    missing_percentage_by_column: Dict[str, float]
    column_ranking: List[str]
    heatmap_data: List[List[bool]]


class MissingValueAnalysis:
    """
    Class to calculate missing count, percentage, ranking, and heatmap data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the MissingValueAnalysis class.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def analyze(self) -> MissingResult:
        """
        Analyze missing values in the dataset.

        Returns:
            MissingResult: The missing value analysis results.

        Raises:
            ProfilingError: If an error occurs during analysis.
        """
        start_time = time.time()
        logger.info("Starting missing value analysis.")
        
        try:
            num_rows = len(self.df)
            total_missing = int(self.df.isna().sum().sum())
            
            missing_by_col_series = self.df.isna().sum()
            missing_by_column = {col: int(count) for col, count in missing_by_col_series.items()}
            
            missing_percentage_by_column = {
                col: float(count / num_rows * 100) if num_rows > 0 else 0.0
                for col, count in missing_by_column.items()
            }
            
            # Rank columns by missing percentage descending
            column_ranking = sorted(
                missing_percentage_by_column.keys(),
                key=lambda k: missing_percentage_by_column[k],
                reverse=True
            )
            
            # Heatmap data
            heatmap_data = self.df.isna().values.tolist()

            result = MissingResult(
                total_missing=total_missing,
                missing_by_column=missing_by_column,
                missing_percentage_by_column=missing_percentage_by_column,
                column_ranking=column_ranking,
                heatmap_data=heatmap_data
            )
            
            exec_time = time.time() - start_time
            logger.info(f"Successfully analyzed missing values in {exec_time:.4f} seconds.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Failed to analyze missing values after {exec_time:.4f} seconds. Error: {e}")
            raise ProfilingError(f"Failed to analyze missing values: {e}") from e
