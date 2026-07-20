"""Target Analysis Module."""

import logging
import time
from typing import Any, Dict
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)


class TargetAnalysis:
    """Analyzes the target column of a DataFrame."""

    def __init__(self, target_column: str = "TARGET") -> None:
        """
        Initialize TargetAnalysis.

        Args:
            target_column (str): The name of the target column to analyze.
        """
        self.target_column = target_column

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the target column for class distribution and imbalance.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing target class distribution,
                balance status, default percentage, and imbalance ratio.

        Raises:
            ProfilingError: If an error occurs during target analysis.
        """
        start_time = time.time()
        logger.info("Starting target analysis for column: %s.", self.target_column)
        
        try:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

            results: Dict[str, Any] = {}
            target_series = df[self.target_column].dropna()
            
            value_counts = target_series.value_counts().to_dict()
            total_count = len(target_series)
            
            if total_count == 0:
                raise ValueError(f"Target column '{self.target_column}' is empty or contains only NaNs.")

            distribution = {k: v for k, v in value_counts.items()}
            percentages = {k: (v / total_count) * 100 for k, v in value_counts.items()}
            
            results["distribution"] = distribution
            results["percentages"] = percentages
            
            # Assuming binary classification where 1 is default/positive class
            # Calculate default percentage and imbalance ratio
            if len(value_counts) == 2:
                classes = list(value_counts.keys())
                # Infer default class (usually 1, or the minority class if not 0/1)
                default_class = 1 if 1 in classes else min(value_counts, key=value_counts.get)
                majority_class = [c for c in classes if c != default_class][0]
                
                default_count = value_counts.get(default_class, 0)
                majority_count = value_counts.get(majority_class, 0)
                
                results["default_percentage"] = (default_count / total_count) * 100
                
                if default_count > 0:
                    imbalance_ratio = majority_count / default_count
                    results["imbalance_ratio"] = f"{imbalance_ratio:.2f}:1"
                else:
                    results["imbalance_ratio"] = "Undefined (no default cases)"
                    
                is_balanced = 0.8 <= (default_count / majority_count) <= 1.25 if majority_count > 0 else False
                results["is_balanced"] = is_balanced
            else:
                results["note"] = "Target is not binary. Default percentage and imbalance ratio are not calculated."

            execution_time = time.time() - start_time
            logger.info("Target analysis completed successfully in %.4f seconds.", execution_time)
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Target analysis failed after %.4f seconds: %s", execution_time, e)
            raise ProfilingError(f"Failed to analyze target column: {e}") from e
