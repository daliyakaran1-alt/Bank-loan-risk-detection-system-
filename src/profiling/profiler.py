"""
Data Profiler Orchestrator.

Orchestrates the generation of a profiling report by coordinating
various profiling modules (e.g., statistics, quality, missing data).
"""

import logging
from typing import Any, Optional
import pandas as pd
from dataclasses import asdict

from core.exceptions import ProfilingError
from src.profiling.report import ProfilingReport

from src.profiling.statistics import DatasetStatistics
from src.profiling.quality import DataQuality
from src.profiling.missing import MissingValueAnalysis
from src.profiling.duplicates import DuplicateAnalysis
from src.profiling.outliers import OutlierAnalysis
from src.profiling.distribution import DistributionAnalysis
from src.profiling.cardinality import CardinalityAnalysis
from src.profiling.correlation import CorrelationAnalysis
from src.profiling.target_analysis import TargetAnalysis

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Orchestrates the data profiling process.
    """
    
    def __init__(self) -> None:
        """Initialize the DataProfiler."""
        pass

    def profile(self, df: pd.DataFrame, dataset_name: str = "dataset", target_column: Optional[str] = None) -> ProfilingReport:
        """
        Generate a comprehensive profiling report for the given DataFrame.
        
        Args:
            df: The pandas DataFrame to profile.
            dataset_name: A name for the dataset being profiled.
            target_column: Optional column name for target analysis.
            
        Returns:
            A ProfilingReport containing the profiling results.
            
        Raises:
            ProfilingError: If the profiling process fails.
        """
        logger.info(f"Starting profiling for dataset: {dataset_name}")
        
        if df is None:
            raise ProfilingError(f"Failed to profile dataset '{dataset_name}': DataFrame is None")
            
        try:
            # 1. Base Statistics
            stats_module = DatasetStatistics(df)
            stats_result = stats_module.calculate()
            
            # 2. Quality
            quality_module = DataQuality(df)
            quality_result = quality_module.evaluate()
            
            # 3. Missing Values
            missing_module = MissingValueAnalysis(df)
            missing_result = missing_module.analyze()
            
            # 4. Duplicates
            duplicates_module = DuplicateAnalysis(df)
            duplicates_result = duplicates_module.analyze()
            
            # 5. Outliers
            outliers_module = OutlierAnalysis(df)
            try:
                outliers_result = outliers_module.analyze()
                outliers_dict = asdict(outliers_result)
            except ProfilingError:
                outliers_dict = {}

            # 6. Distribution
            dist_module = DistributionAnalysis(df)
            try:
                dist_result = dist_module.analyze()
                dist_dict = asdict(dist_result)
            except ProfilingError:
                dist_dict = {}
                
            # 7. Cardinality
            cardinality_module = CardinalityAnalysis(df)
            try:
                cardinality_result = cardinality_module.analyze()
                cardinality_dict = asdict(cardinality_result)
            except ProfilingError:
                cardinality_dict = {}
                
            # 8. Correlation
            correlation_module = CorrelationAnalysis()
            try:
                correlation_dict = correlation_module.analyze(df)
            except ProfilingError:
                correlation_dict = {}
                
            # 9. Target Analysis
            target_dict = {}
            if target_column and target_column in df.columns:
                target_module = TargetAnalysis(target_column=target_column)
                try:
                    target_dict = target_module.analyze(df)
                except ProfilingError:
                    pass
            
            metrics = {
                "row_count": stats_result.num_rows,
                "column_count": stats_result.num_columns,
                "memory_usage_bytes": stats_result.memory_usage_bytes,
                "dtypes": stats_result.dtypes,
                "numeric_counts": stats_result.numeric_counts,
                "unique_values": stats_result.unique_values
            }
            
            quality_issues = {
                "missing_ratio": quality_result.missing_ratio,
                "duplicate_ratio": quality_result.duplicate_ratio,
                "total_missing": missing_result.total_missing,
                "total_duplicates": duplicates_result.total_duplicates,
                "missing_by_column": missing_result.missing_by_column,
                "potential_duplicate_columns": duplicates_result.potential_duplicate_columns
            }
            
            report = ProfilingReport(
                dataset_name=dataset_name,
                overall_health_score=quality_result.health_score,
                data_quality_score=quality_result.health_score,
                metrics=metrics,
                quality_issues=quality_issues,
                advanced_statistics={},
                outlier_summary=outliers_dict,
                correlation_summary=correlation_dict,
                distribution_summary=dist_dict,
                cardinality_summary=cardinality_dict,
                target_analysis=target_dict,
                business_recommendations=[]
            )
            
            logger.info(f"Profiling completed successfully for dataset: {dataset_name}")
            return report
            
        except Exception as e:
            logger.error(f"Profiling failed for dataset %s: %s", dataset_name, str(e))
            raise ProfilingError(f"Failed to profile dataset '{dataset_name}': {str(e)}") from e
