import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any
import pandas as pd

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class ColumnCardinalityResult:
    unique_count: int
    total_count: int
    unique_percentage: float
    category_counts: Dict[str, int]
    rare_categories: List[str]
    is_high_cardinality: bool
    is_low_cardinality: bool

@dataclass
class CardinalityResult:
    high_cardinality_columns: List[str]
    low_cardinality_columns: List[str]
    column_details: Dict[str, ColumnCardinalityResult]
    recommendations: List[str]

class CardinalityAnalysis:
    """Class to perform cardinality analysis on categorical data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_categorical_columns(self) -> List[str]:
        return self.df.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

    def analyze(self, high_cardinality_threshold: float = 0.5, low_cardinality_threshold: int = 10, rare_category_threshold: float = 0.01) -> CardinalityResult:
        logger.info("Starting cardinality analysis.")
        start_time = time.time()

        if self.df is None or self.df.empty:
            logger.error("Empty dataframe provided for cardinality analysis.")
            raise ProfilingError("Cannot perform cardinality analysis on an empty dataframe.")

        try:
            categorical_cols = self._get_categorical_columns()
            if not categorical_cols:
                logger.warning("No standard categorical columns found for cardinality analysis.")
                
            column_details = {}
            high_cardinality_columns = []
            low_cardinality_columns = []
            
            for col in categorical_cols:
                series = self.df[col].dropna()
                total_count = len(series)
                
                if total_count == 0:
                    continue

                value_counts = series.value_counts()
                unique_count = len(value_counts)
                unique_percentage = unique_count / total_count if total_count > 0 else 0.0

                is_high_cardinality = unique_percentage > high_cardinality_threshold and unique_count > low_cardinality_threshold
                is_low_cardinality = unique_count <= low_cardinality_threshold
                
                if is_high_cardinality:
                    high_cardinality_columns.append(col)
                if is_low_cardinality:
                    low_cardinality_columns.append(col)

                category_counts = value_counts.head(100).to_dict()
                category_counts = {str(k): int(v) for k, v in category_counts.items()}

                rare_threshold_count = max(1, int(rare_category_threshold * total_count))
                rare_series = value_counts[value_counts <= rare_threshold_count]
                rare_categories = [str(idx) for idx in rare_series.index.tolist()][:100]

                column_details[col] = ColumnCardinalityResult(
                    unique_count=unique_count,
                    total_count=total_count,
                    unique_percentage=unique_percentage,
                    category_counts=category_counts,
                    rare_categories=rare_categories,
                    is_high_cardinality=is_high_cardinality,
                    is_low_cardinality=is_low_cardinality
                )

            recommendations = []
            if high_cardinality_columns:
                recommendations.append(f"High cardinality detected in: {', '.join(high_cardinality_columns[:5])}. Consider target encoding, hashing, or dropping these columns.")
            if any(len(res.rare_categories) > 0 for res in column_details.values()):
                recommendations.append("Rare categories detected. Consider grouping them into an 'Other' category.")
            if not recommendations:
                recommendations.append("Cardinality is well-balanced across categorical columns.")

            elapsed_time = time.time() - start_time
            logger.info(f"Finished cardinality analysis in {elapsed_time:.4f} seconds.")
            
            return CardinalityResult(
                high_cardinality_columns=high_cardinality_columns,
                low_cardinality_columns=low_cardinality_columns,
                column_details=column_details,
                recommendations=recommendations
            )

        except ProfilingError:
            raise
        except Exception as e:
            logger.error(f"Error during cardinality analysis: {str(e)}")
            raise ProfilingError(f"Cardinality analysis failed: {str(e)}") from e
