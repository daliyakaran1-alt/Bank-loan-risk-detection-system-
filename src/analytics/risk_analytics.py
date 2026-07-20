"""
Risk Analytics Module.

Handles complex risk profiling segmentations and analytics, specifically
focusing on Default Risk by various demographic and socio-economic factors.
Generates risk heatmaps and bar charts using Plotly.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.analytics.analytics_models import DashboardSection, ChartData
from src.analytics.chart_factory import ChartFactory
from src.analytics.analytics_utils import bin_numerical_column

logger = logging.getLogger(__name__)

class RiskAnalytics:
    """Handles comprehensive risk analytics and groupings."""
    
    def __init__(self, target_col: str = "TARGET"):
        """
        Initialize RiskAnalytics with the target column.
        
        Args:
            target_col (str): The column name representing the default target.
        """
        self.target_col = target_col
        self.chart_factory = ChartFactory()

    def _get_col(self, df: pd.DataFrame, possibilities: List[str]) -> str:
        """Helper to find the first matching column from possibilities."""
        for p in possibilities:
            if p in df.columns:
                return p
        return ""

    def generate_section(self, df: pd.DataFrame) -> DashboardSection:
        """
        Generate risk analytics section with default risk by multiple factors.
        
        Args:
            df (pd.DataFrame): The input dataset.
            
        Returns:
            DashboardSection: The generated risk analytics dashboard section.
        """
        section = DashboardSection(section_name="Risk Analytics")
        if df is None or df.empty or self.target_col not in df.columns:
            logger.warning("Dataset is empty or target column is missing for Risk Analytics.")
            return section

        # Univariate Default Risk (Bar Charts)
        self._add_default_by_categorical(df, section, ["CODE_GENDER", "GENDER"], "Gender")
        self._add_default_by_categorical(df, section, ["NAME_EDUCATION_TYPE", "EDUCATION"], "Education")
        self._add_default_by_categorical(df, section, ["NAME_INCOME_TYPE", "OCCUPATION_TYPE", "OCCUPATION"], "Occupation")
        self._add_default_by_categorical(df, section, ["NAME_FAMILY_STATUS", "FAMILY_STATUS"], "Family Status")
        self._add_default_by_categorical(df, section, ["NAME_HOUSING_TYPE", "HOUSING"], "Housing")
        self._add_default_by_categorical(df, section, ["FLAG_OWN_CAR", "OWN_CAR"], "Car Ownership")
        self._add_default_by_categorical(df, section, ["ORGANIZATION_TYPE", "ORGANIZATION"], "Organization")

        # Binned Numerical Default Risk (Bar Charts)
        self._add_default_by_binned_numerical(df, section, ["DAYS_BIRTH", "AGE"], "Age", bins=5, is_days=True)
        self._add_default_by_binned_numerical(df, section, ["AMT_INCOME_TOTAL", "INCOME"], "Income", bins=5, is_days=False)

        # Risk Heatmaps (Bivariate Default Risk)
        self._add_risk_heatmap(
            df, section, 
            col1_possibilities=["CODE_GENDER", "GENDER"], col1_name="Gender",
            col2_possibilities=["NAME_EDUCATION_TYPE", "EDUCATION"], col2_name="Education"
        )
        self._add_risk_heatmap(
            df, section, 
            col1_possibilities=["NAME_FAMILY_STATUS", "FAMILY_STATUS"], col1_name="Family Status",
            col2_possibilities=["NAME_HOUSING_TYPE", "HOUSING"], col2_name="Housing"
        )

        return section

    def _add_default_by_categorical(self, df: pd.DataFrame, section: DashboardSection, 
                                    possibilities: List[str], feature_name: str):
        """Generate a bar chart for default risk by a categorical feature."""
        col = self._get_col(df, possibilities)
        if col:
            grouped = df.groupby(col)[self.target_col].mean().reset_index()
            grouped.columns = [feature_name, 'Default Rate']
            # Sort for better visualization
            grouped = grouped.sort_values(by='Default Rate', ascending=False)
            title = f'Default Rate by {feature_name}'
            
            # Use horizontal bar chart if there are many categories (e.g., Organization)
            if len(grouped) > 5:
                fig = self.chart_factory.create_bar_chart(
                    grouped, x='Default Rate', y=feature_name, title=title, orientation='h'
                )
                chart_type = "horizontal_bar"
            else:
                fig = self.chart_factory.create_bar_chart(
                    grouped, x=feature_name, y='Default Rate', title=title
                )
                chart_type = "bar"
                
            chart_id = f"default_by_{feature_name.lower().replace(' ', '_')}"
            section.charts[chart_id] = ChartData(chart_id, chart_type, title, fig)

    def _add_default_by_binned_numerical(self, df: pd.DataFrame, section: DashboardSection, 
                                         possibilities: List[str], feature_name: str, bins: int, is_days: bool):
        """Generate a bar chart for default risk by a binned numerical feature."""
        col = self._get_col(df, possibilities)
        if col:
            temp_df = df.copy()
            if is_days:
                # Convert negative days to positive years (assuming DAYS_BIRTH style)
                temp_df[col] = (temp_df[col] / -365.25).apply(lambda x: x if x > 0 else temp_df[col].abs() / 365.25)
            
            binned_col = f'{feature_name}_Group'
            temp_df[binned_col] = bin_numerical_column(temp_df, col, bins=bins)
            
            grouped = temp_df.groupby(binned_col)[self.target_col].mean().reset_index()
            grouped.columns = [f'{feature_name} Group', 'Default Rate']
            title = f'Default Rate by {feature_name} Group'
            
            fig = self.chart_factory.create_bar_chart(
                grouped, x=f'{feature_name} Group', y='Default Rate', title=title
            )
            chart_id = f"default_by_{feature_name.lower()}"
            section.charts[chart_id] = ChartData(chart_id, "bar", title, fig)

    def _add_risk_heatmap(self, df: pd.DataFrame, section: DashboardSection, 
                          col1_possibilities: List[str], col1_name: str, 
                          col2_possibilities: List[str], col2_name: str):
        """Generate a heatmap showing default risk across two categorical features."""
        col1 = self._get_col(df, col1_possibilities)
        col2 = self._get_col(df, col2_possibilities)
        
        if col1 and col2:
            # Create a cross-tabulation with the mean of the target variable
            pivot = pd.pivot_table(df, values=self.target_col, index=col2, columns=col1, aggfunc=np.mean)
            
            # Fill NaNs with 0 or the overall mean for visualization purposes
            pivot = pivot.fillna(0)
            
            x_labels = pivot.columns.astype(str).tolist()
            y_labels = pivot.index.astype(str).tolist()
            z_values = pivot.values.tolist()
            
            title = f'Risk Heatmap: {col1_name} vs {col2_name}'
            fig = self.chart_factory.create_heatmap(z=z_values, x=x_labels, y=y_labels, title=title)
            
            chart_id = f"risk_heatmap_{col1_name.lower().replace(' ', '_')}_{col2_name.lower().replace(' ', '_')}"
            section.charts[chart_id] = ChartData(chart_id, "heatmap", title, fig)
