import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)

@dataclass
class ColumnOutlierResult:
    count: int
    percentage: float
    indices: List[int]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

@dataclass
class OutlierResult:
    outlier_count: int
    outlier_percentage: float
    affected_columns: List[str]
    column_details: Dict[str, ColumnOutlierResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    method_used: str

class OutlierAnalysis:
    """Class to perform outlier analysis using various statistical methods."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def _get_numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def analyze(self, method: str = 'iqr', threshold: float = 1.5) -> OutlierResult:
        logger.info(f"Starting outlier analysis using method: {method}")
        start_time = time.time()
        
        if self.df is None or self.df.empty:
            logger.error("Empty dataframe provided for outlier analysis.")
            raise ProfilingError("Cannot perform outlier analysis on an empty dataframe.")
            
        try:
            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                logger.warning("No numeric columns found for outlier analysis.")
                return OutlierResult(
                    outlier_count=0,
                    outlier_percentage=0.0,
                    affected_columns=[],
                    column_details={},
                    summary={"message": "No numeric columns available."},
                    recommendations=["Ensure data contains numeric features to detect outliers."],
                    method_used=method
                )

            column_details = {}
            total_rows = len(self.df)
            global_outlier_indices = set()

            for col in numeric_cols:
                series = self.df[col].dropna()
                if series.empty:
                    continue
                    
                if method == 'iqr':
                    outliers_series, lower, upper = self._iqr_method(series, threshold)
                elif method == 'zscore':
                    outliers_series, lower, upper = self._zscore_method(series, threshold)
                elif method == 'modified_zscore':
                    outliers_series, lower, upper = self._modified_zscore_method(series, threshold)
                else:
                    raise ProfilingError(f"Unsupported outlier detection method: {method}")

                outlier_idx = outliers_series[outliers_series].index.tolist()
                count = len(outlier_idx)
                
                if count > 0:
                    global_outlier_indices.update(outlier_idx)
                    percentage = (count / total_rows) * 100
                    column_details[col] = ColumnOutlierResult(
                        count=count,
                        percentage=percentage,
                        indices=outlier_idx,
                        lower_bound=lower,
                        upper_bound=upper
                    )

            affected_columns = list(column_details.keys())
            total_outliers_count = len(global_outlier_indices)
            total_outliers_percentage = (total_outliers_count / total_rows) * 100 if total_rows > 0 else 0.0
            
            recommendations = []
            if total_outliers_percentage > 10:
                recommendations.append("High overall percentage of outliers detected. Consider robust scaling or capping.")
            if affected_columns:
                recommendations.append(f"Investigate columns with outliers: {', '.join(affected_columns[:5])}{'...' if len(affected_columns)>5 else ''}")
            if not recommendations:
                recommendations.append("No significant outliers detected requiring immediate action.")

            result = OutlierResult(
                outlier_count=total_outliers_count,
                outlier_percentage=total_outliers_percentage,
                affected_columns=affected_columns,
                column_details=column_details,
                summary={
                    "total_numeric_columns": len(numeric_cols),
                    "columns_with_outliers": len(affected_columns),
                    "total_rows_with_outliers": total_outliers_count
                },
                recommendations=recommendations,
                method_used=method
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Finished outlier analysis in {elapsed_time:.4f} seconds. Detected outliers in {len(affected_columns)} columns.")
            return result
            
        except ProfilingError:
            raise
        except Exception as e:
            logger.error(f"Error during outlier analysis: {str(e)}")
            raise ProfilingError(f"Outlier analysis failed: {str(e)}") from e

    def _iqr_method(self, series: pd.Series, threshold: float = 1.5) -> Tuple[pd.Series, float, float]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = float(q1 - (threshold * iqr))
        upper_bound = float(q3 + (threshold * iqr))
        return (series < lower_bound) | (series > upper_bound), lower_bound, upper_bound

    def _zscore_method(self, series: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, float, float]:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index), float(mean), float(mean)
        z_scores = (series - mean) / std
        lower_bound = float(mean - threshold * std)
        upper_bound = float(mean + threshold * std)
        return z_scores.abs() > threshold, lower_bound, upper_bound

    def _modified_zscore_method(self, series: pd.Series, threshold: float = 3.5) -> Tuple[pd.Series, float, float]:
        median = series.median()
        abs_dev = (series - median).abs()
        mad = abs_dev.median()
        
        if mad == 0:
            mad = abs_dev.mean()
            if mad == 0:
                return pd.Series(False, index=series.index), float(median), float(median)
                
        mod_z_scores = 0.6745 * (series - median) / mad
        
        lower_bound = float(median - (threshold * mad / 0.6745))
        upper_bound = float(median + (threshold * mad / 0.6745))
        
        return mod_z_scores.abs() > threshold, lower_bound, upper_bound
