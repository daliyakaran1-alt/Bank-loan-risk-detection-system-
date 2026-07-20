"""
EDA Engine Module.

Generates Dataset Overview, Univariate, Bivariate, Multivariate, and Correlation Analytics.
"""

import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from src.analytics.analytics_models import DashboardSection, ChartData
from src.analytics.chart_factory import ChartFactory

logger = logging.getLogger(__name__)


class EDAEngine:
    """Handles core Exploratory Data Analysis."""

    def __init__(self, target_col: str = "TARGET"):
        self.target_col = target_col
        self.chart_factory = ChartFactory()

    def get_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset profile summary."""
        logger.info("Generating Dataset Overview...")
        overview = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024), # in MB
            "target_distribution": df[self.target_col].value_counts(normalize=True).to_dict() if self.target_col in df.columns else {},
            "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "numerical_features": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_features": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "boolean_features": df.select_dtypes(include=['bool']).columns.tolist(),
        }
        
        # Cardinality
        cardinality = df[overview["categorical_features"]].nunique()
        overview["high_cardinality"] = cardinality[cardinality > 20].index.tolist()
        overview["low_cardinality"] = cardinality[cardinality <= 20].index.tolist()
        return overview

    def generate_correlation_analytics(self, df: pd.DataFrame) -> DashboardSection:
        """Generate correlation matrix and heatmaps."""
        logger.info("Generating Correlation Analytics...")
        section = DashboardSection(section_name="Correlation Analytics")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return section

        # Due to performance with large datasets, sample or take top features
        if num_df.shape[1] > 20:
            if self.target_col in num_df.columns:
                corrs = num_df.corrwith(num_df[self.target_col]).abs().sort_values(ascending=False)
                top_cols = corrs.head(20).index.tolist()
                num_df = num_df[top_cols]

        pearson_corr = num_df.corr(method='pearson')
        spearson_corr = num_df.corr(method='spearman')
        
        fig_pearson = go.Figure(data=go.Heatmap(
            z=pearson_corr.values,
            x=pearson_corr.columns,
            y=pearson_corr.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1
        ))
        fig_pearson.update_layout(title="Pearson Correlation Heatmap")
        
        section.charts["pearson_correlation_heatmap"] = ChartData("pearson_correlation_heatmap", "heatmap", "Pearson Correlation Heatmap", fig_pearson)
        
        fig_spearman = go.Figure(data=go.Heatmap(
            z=spearson_corr.values,
            x=spearson_corr.columns,
            y=spearson_corr.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1
        ))
        fig_spearman.update_layout(title="Spearman Correlation Heatmap")
        section.charts["spearman_correlation_heatmap"] = ChartData("spearman_correlation_heatmap", "heatmap", "Spearman Correlation Heatmap", fig_spearman)

        return section

    def generate_univariate_analysis(self, df: pd.DataFrame, max_cols: int = 10) -> DashboardSection:
        """Auto-generate Univariate Analysis for numerical columns."""
        logger.info("Generating Univariate Analysis...")
        section = DashboardSection(section_name="Univariate Analysis")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to most important to avoid massive memory usage
        if self.target_col in df.columns and self.target_col in num_cols:
            num_cols.remove(self.target_col)
            corrs = df[num_cols].corrwith(df[self.target_col]).abs().sort_values(ascending=False)
            num_cols = corrs.head(max_cols).index.tolist()
        else:
            num_cols = num_cols[:max_cols]
            
        for col in num_cols:
            plot_df = df[col].dropna()
            if plot_df.empty: continue
            
            # Histogram
            fig_hist = self.chart_factory.create_histogram(df, x=col, title=f'{col} Distribution')
            section.charts[f"uni_hist_{col}"] = ChartData(f"uni_hist_{col}", "histogram", f"{col} Distribution", fig_hist)
            
            # Boxplot
            fig_box = self.chart_factory.create_box_plot(df, x=None, y=col, title=f'{col} Box Plot')
            section.charts[f"uni_box_{col}"] = ChartData(f"uni_box_{col}", "box", f"{col} Box Plot", fig_box)

        return section

    def generate_bivariate_analysis(self, df: pd.DataFrame, max_cols: int = 10) -> DashboardSection:
        """Auto-generate Bivariate Analysis vs Target."""
        logger.info("Generating Bivariate Analysis...")
        section = DashboardSection(section_name="Bivariate Analysis")
        if self.target_col not in df.columns:
            return section
            
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols.remove(self.target_col)
        
        # Select top correlated
        corrs = df[num_cols].corrwith(df[self.target_col]).abs().sort_values(ascending=False)
        num_cols = corrs.head(max_cols).index.tolist()
        
        df_plot = df.copy()
        df_plot[self.target_col] = df_plot[self.target_col].astype(str)
        
        for col in num_cols:
            fig_box = self.chart_factory.create_box_plot(df_plot, x=self.target_col, y=col, color=self.target_col, title=f'{col} vs Target')
            section.charts[f"bi_box_{col}"] = ChartData(f"bi_box_{col}", "box", f"{col} vs Target", fig_box)

        return section

    def generate_multivariate_analysis(self, df: pd.DataFrame) -> DashboardSection:
        """Generate Multivariate Analysis (Scatter matrix, Parallel Coordinates, 3D)."""
        logger.info("Generating Multivariate Analysis...")
        section = DashboardSection(section_name="Multivariate Analysis")
        if self.target_col not in df.columns:
            return section
            
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols.remove(self.target_col)
        
        # Take top 3 for 3D and 4 for parallel
        corrs = df[num_cols].corrwith(df[self.target_col]).abs().sort_values(ascending=False)
        top_cols = corrs.head(4).index.tolist()
        
        if len(top_cols) >= 3:
            plot_df = df.sample(min(1000, len(df)), random_state=42)
            # 3D Scatter
            fig_3d = px.scatter_3d(plot_df, x=top_cols[0], y=top_cols[1], z=top_cols[2], color=self.target_col)
            section.charts["3d_scatter"] = ChartData("3d_scatter", "scatter3d", "3D Scatter Plot", fig_3d)
            
            # Parallel Coordinates
            fig_pc = px.parallel_coordinates(plot_df, dimensions=top_cols, color=self.target_col)
            section.charts["parallel_coords"] = ChartData("parallel_coords", "parallel_coordinates", "Parallel Coordinates", fig_pc)
            
            # Scatter Matrix
            fig_sm = px.scatter_matrix(plot_df, dimensions=top_cols, color=self.target_col)
            section.charts["scatter_matrix"] = ChartData("scatter_matrix", "scatter_matrix", "Scatter Matrix", fig_sm)

        return section
