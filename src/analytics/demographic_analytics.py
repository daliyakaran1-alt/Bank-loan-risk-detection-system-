"""
Demographic Analytics Module.

Generates insights and charts for demographic groupings.
"""

import logging
import pandas as pd

from src.analytics.analytics_models import DashboardSection, ChartData
from src.analytics.chart_factory import ChartFactory
from src.analytics.analytics_utils import bin_numerical_column

logger = logging.getLogger(__name__)

class DemographicAnalytics:
    """Handles analytics related to demographics."""
    
    def __init__(self, target_col: str = "TARGET"):
        self.target_col = target_col
        self.chart_factory = ChartFactory()

    def _get_col(self, df: pd.DataFrame, possibilities: list) -> str:
        for p in possibilities:
            if p in df.columns:
                return p
        return None

    def generate_section(self, df: pd.DataFrame) -> DashboardSection:
        """Generate demographic analytics section."""
        section = DashboardSection(section_name="Demographic Analytics")
        if df is None or df.empty:
            return section

        self._add_age_groups(df, section)
        self._add_family_size_groups(df, section)
        self._add_employment_groups(df, section)
        
        return section
        
    def _add_age_groups(self, df: pd.DataFrame, section: DashboardSection):
        col = self._get_col(df, ["DAYS_BIRTH", "AGE"])
        if col:
            temp_df = df.copy()
            if temp_df[col].mean() < 0:
                # likely days_birth in negative days
                temp_df['Age_Years'] = abs(temp_df[col]) / 365.25
            else:
                temp_df['Age_Years'] = temp_df[col]
                
            temp_df['Age_Group'] = bin_numerical_column(temp_df, 'Age_Years', bins=[18, 25, 35, 45, 55, 65, 100], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            counts = temp_df['Age_Group'].value_counts().reset_index()
            counts.columns = ['Age Group', 'Count']
            fig = self.chart_factory.create_bar_chart(counts, x='Age Group', y='Count', title='Age Groups')
            section.charts["age_groups"] = ChartData("age_groups", "bar", "Age Groups", fig)

    def _add_family_size_groups(self, df: pd.DataFrame, section: DashboardSection):
        col = self._get_col(df, ["CNT_FAM_MEMBERS", "FAMILY_MEMBERS"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ['Family Size', 'Count']
            # Limit to top 6 sizes to avoid clutter
            counts = counts.head(6)
            fig = self.chart_factory.create_pie_chart(counts, names='Family Size', values='Count', title='Family Size Groups', hole=0.4)
            section.charts["family_size_groups"] = ChartData("family_size_groups", "donut", "Family Size Groups", fig)

    def _add_employment_groups(self, df: pd.DataFrame, section: DashboardSection):
        col = self._get_col(df, ["NAME_INCOME_TYPE", "INCOME_TYPE"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ['Employment Group', 'Count']
            fig = self.chart_factory.create_bar_chart(counts, x='Count', y='Employment Group', title='Employment Groups', orientation='h')
            section.charts["employment_groups"] = ChartData("employment_groups", "horizontal_bar", "Employment Groups", fig)
