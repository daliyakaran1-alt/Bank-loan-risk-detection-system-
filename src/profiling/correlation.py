"""
Correlation Analysis Module.

Calculates and analyzes correlations between numerical features.
"""

import logging
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from core.exceptions import ProfilingError

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAnalysis:
    """
    Analyzes correlations between numerical features in a dataset.
    """
    correlation_threshold: float = 0.8
    weak_threshold: float = 0.2

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform correlation analysis on the numerical columns of the DataFrame.

        Args:
            df: The pandas DataFrame to analyze.

        Returns:
            A dictionary containing correlation matrices, highly correlated features,
            weak correlations, positive correlations, negative correlations, and recommendations.
            
        Raises:
            ProfilingError: If the correlation analysis fails.
        """
        start_time = time.time()
        logger.info("Starting correlation analysis.")

        try:
            # Select numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                logger.warning("Not enough numerical columns for correlation analysis.")
                return {
                    "pearson_matrix": {},
                    "spearman_matrix": {},
                    "highly_correlated_features": [],
                    "weak_correlations": [],
                    "positive_correlations": [],
                    "negative_correlations": [],
                    "recommendations": ["Not enough numerical columns to compute correlations."],
                    "plot_metadata": {}
                }

            # Vectorized pandas operations to compute correlation matrices
            pearson_corr = numeric_df.corr(method="pearson")
            spearman_corr = numeric_df.corr(method="spearman")

            highly_correlated = []
            weak_correlations = []
            positive_correlations = []
            negative_correlations = []
            recommendations = []

            # Extract upper triangle without diagonal for analysis
            upper_tri = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
            
            # Stack to create a Series with MultiIndex (feature1, feature2)
            corr_pairs = upper_tri.stack().reset_index()
            corr_pairs.columns = ["feature1", "feature2", "correlation"]
            
            # Categorize correlations
            high_mask = corr_pairs["correlation"].abs() >= self.correlation_threshold
            weak_mask = corr_pairs["correlation"].abs() <= self.weak_threshold
            pos_mask = corr_pairs["correlation"] > 0
            neg_mask = corr_pairs["correlation"] < 0
            
            for _, row in corr_pairs[high_mask].iterrows():
                highly_correlated.append((row["feature1"], row["feature2"], float(row["correlation"])))
            
            for _, row in corr_pairs[weak_mask].iterrows():
                weak_correlations.append((row["feature1"], row["feature2"], float(row["correlation"])))
                
            for _, row in corr_pairs[pos_mask].iterrows():
                positive_correlations.append((row["feature1"], row["feature2"], float(row["correlation"])))
                
            for _, row in corr_pairs[neg_mask].iterrows():
                negative_correlations.append((row["feature1"], row["feature2"], float(row["correlation"])))

            if highly_correlated:
                recommendations.append(
                    f"Found {len(highly_correlated)} highly correlated feature pairs. "
                    "Consider removing one of the features in each pair to reduce multicollinearity."
                )
            else:
                recommendations.append("No highly correlated feature pairs found. Multicollinearity risk is low.")

            # Prepare plot metadata for future Plotly Heatmaps
            plot_metadata = {
                "type": "heatmap",
                "x_labels": pearson_corr.columns.tolist(),
                "y_labels": pearson_corr.index.tolist(),
                "z_values_pearson": pearson_corr.values.tolist()
            }

            results = {
                "pearson_matrix": pearson_corr.to_dict(),
                "spearman_matrix": spearman_corr.to_dict(),
                "highly_correlated_features": highly_correlated,
                "weak_correlations": weak_correlations,
                "positive_correlations": positive_correlations,
                "negative_correlations": negative_correlations,
                "recommendations": recommendations,
                "plot_metadata": plot_metadata
            }

            elapsed_time = time.time() - start_time
            logger.info(f"Finished correlation analysis in {elapsed_time:.4f} seconds.")
            return results

        except Exception as e:
            logger.error(f"Error during correlation analysis: {str(e)}")
            raise ProfilingError(f"Correlation analysis failed: {str(e)}") from e
