"""
Module for computing data quality and health scores.
"""

import logging
import time
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class QualityResult:
    """Dataclass to hold data quality results."""
    health_score: float
    missing_ratio: float
    duplicate_ratio: float


class DataQuality:
    """
    Class to calculate Data Quality metrics and a Dataset Health Score.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataQuality class.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def evaluate(self) -> QualityResult:
        """
        Evaluate the data quality and calculate the health score.

        Returns:
            QualityResult: The calculated data quality metrics.

        Raises:
            ProfilingError: If an error occurs during evaluation.
        """
        start_time = time.time()
        logger.info("Starting data quality evaluation.")
        
        try:
            num_rows = len(self.df)
            if num_rows == 0:
                result = QualityResult(health_score=0.0, missing_ratio=0.0, duplicate_ratio=0.0)
                exec_time = time.time() - start_time
                logger.info(f"Successfully evaluated empty dataset in {exec_time:.4f} seconds.")
                return result

            total_cells = num_rows * len(self.df.columns)
            missing_cells = self.df.isna().sum().sum()
            missing_ratio = float(missing_cells / total_cells) if total_cells > 0 else 0.0

            duplicate_rows = self.df.duplicated().sum()
            duplicate_ratio = float(duplicate_rows / num_rows)
            
            # Simple health score: 100 - (missing_ratio * 50) - (duplicate_ratio * 50)
            health_score = max(0.0, min(100.0, 100.0 * (1.0 - (missing_ratio * 0.5 + duplicate_ratio * 0.5))))

            result = QualityResult(
                health_score=health_score,
                missing_ratio=missing_ratio,
                duplicate_ratio=duplicate_ratio
            )
            
            exec_time = time.time() - start_time
            logger.info(f"Successfully evaluated data quality in {exec_time:.4f} seconds.")
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Failed to evaluate data quality after {exec_time:.4f} seconds. Error: {e}")
            raise ProfilingError(f"Failed to evaluate data quality: {e}") from e
