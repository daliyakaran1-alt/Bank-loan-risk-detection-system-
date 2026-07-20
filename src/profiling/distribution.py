import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class ColumnDistributionResult:
    skewness: float
    kurtosis: float
    is_normal: bool
    normality_p_value: float
    histogram_bins: List[float]
    histogram_counts: List[int]
    boxplot_metadata: Dict[str, float]
    density_x: List[float]
    density_y: List[float]
    distribution_classification: str

@dataclass
class DistributionResult:
    column_details: Dict[str, ColumnDistributionResult]
    summary: Dict[str, Any]

class DistributionAnalysis:
    """Class to perform distribution analysis on numeric data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def analyze(self) -> DistributionResult:
        logger.info("Starting distribution analysis.")
        start_time = time.time()

        if self.df is None or self.df.empty:
            logger.error("Empty dataframe provided for distribution analysis.")
            raise ProfilingError("Cannot perform distribution analysis on an empty dataframe.")

        try:
            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                logger.warning("No numeric columns found for distribution analysis.")
                return DistributionResult(column_details={}, summary={"message": "No numeric columns available."})

            column_details = {}

            for col in numeric_cols:
                series = self.df[col].dropna()
                if len(series) < 8:
                    continue
                
                skewness = float(series.skew())
                kurtosis = float(series.kurtosis())
                
                try:
                    stat, p_value = stats.normaltest(series)
                    is_normal = p_value > 0.05
                    normality_p_value = float(p_value)
                except Exception:
                    is_normal = False
                    normality_p_value = 0.0

                if is_normal:
                    classification = "Normal"
                elif skewness > 1:
                    classification = "Highly Right-Skewed"
                elif skewness < -1:
                    classification = "Highly Left-Skewed"
                elif skewness > 0.5:
                    classification = "Moderately Right-Skewed"
                elif skewness < -0.5:
                    classification = "Moderately Left-Skewed"
                else:
                    classification = "Approximately Symmetric"

                counts, bins = np.histogram(series, bins='auto')
                histogram_counts = counts.tolist()
                histogram_bins = bins.tolist()

                q1 = float(series.quantile(0.25))
                median = float(series.quantile(0.50))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                
                lower_whisker_data = series[series >= (q1 - 1.5 * iqr)]
                upper_whisker_data = series[series <= (q3 + 1.5 * iqr)]
                
                whisker_low = float(lower_whisker_data.min()) if not lower_whisker_data.empty else q1
                whisker_high = float(upper_whisker_data.max()) if not upper_whisker_data.empty else q3
                
                boxplot_metadata = {
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "whisker_low": whisker_low,
                    "whisker_high": whisker_high
                }

                try:
                    kde = stats.gaussian_kde(series)
                    density_x = np.linspace(series.min(), series.max(), 100)
                    density_y = kde(density_x)
                    density_x_list = density_x.tolist()
                    density_y_list = density_y.tolist()
                except Exception:
                    density_x_list = []
                    density_y_list = []

                column_details[col] = ColumnDistributionResult(
                    skewness=skewness,
                    kurtosis=kurtosis,
                    is_normal=is_normal,
                    normality_p_value=normality_p_value,
                    histogram_bins=histogram_bins,
                    histogram_counts=histogram_counts,
                    boxplot_metadata=boxplot_metadata,
                    density_x=density_x_list,
                    density_y=density_y_list,
                    distribution_classification=classification
                )

            summary = {
                "total_columns_analyzed": len(column_details),
                "normal_columns": sum(1 for v in column_details.values() if v.is_normal),
                "skewed_columns": sum(1 for v in column_details.values() if not v.is_normal)
            }

            elapsed_time = time.time() - start_time
            logger.info(f"Finished distribution analysis in {elapsed_time:.4f} seconds.")
            return DistributionResult(column_details=column_details, summary=summary)

        except ProfilingError:
            raise
        except Exception as e:
            logger.error(f"Error during distribution analysis: {str(e)}")
            raise ProfilingError(f"Distribution analysis failed: {str(e)}") from e
