"""
Financial Analytics Module.

Generates financial health indicators and distributions for Business Intelligence & EDA.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.analytics_models import DashboardSection, ChartData
from src.analytics.chart_factory import ChartFactory
from src.analytics.analytics_utils import safe_divide

logger = logging.getLogger(__name__)

class FinancialAnalytics:
    """Handles analytics related to financial health and ratios."""
    
    def __init__(self, target_col: str = "TARGET"):
        """
        Initialize FinancialAnalytics.
        
        Args:
            target_col (str): The column name representing the target/default variable.
        """
        self.target_col = target_col
        self.chart_factory = ChartFactory()

    def _get_col(self, df: pd.DataFrame, possibilities: list) -> str:
        for p in possibilities:
            if p in df.columns:
                return p
        return None

    def generate_section(self, df: pd.DataFrame) -> DashboardSection:
        """
        Generate financial analytics section containing Plotly visualizations.
        
        Args:
            df (pd.DataFrame): The input dataset.
            
        Returns:
            DashboardSection: A dashboard section containing charts.
        """
        section = DashboardSection(section_name="Financial Analytics")
        if df is None or df.empty:
            return section

        self._add_distributions(df, section)
        self._add_bivariate_analysis(df, section)
        self._add_default_analysis(df, section)
        
        return section
        
    def _add_distributions(self, df: pd.DataFrame, section: DashboardSection):
        """Add distributions for Income, Credit, and Annuity."""
        # Income Distribution (Histogram with density proxy)
        inc_col = self._get_col(df, ["AMT_INCOME_TOTAL", "INCOME"])
        if inc_col:
            q_high = df[inc_col].quantile(0.95)
            plot_df = df[df[inc_col] < q_high]
            fig_inc = self.chart_factory.create_histogram(plot_df, x=inc_col, title='Income Distribution (Bottom 95%)', nbins=50)
            section.charts["income_dist"] = ChartData("income_dist", "histogram", "Income Distribution", fig_inc)

        # Credit Amount Distribution (Box Plot)
        cred_col = self._get_col(df, ["AMT_CREDIT", "CREDIT_AMOUNT"])
        if cred_col:
            fig_cred = self.chart_factory.create_box_plot(df, x=None, y=cred_col, title='Credit Amount Distribution')
            section.charts["credit_dist"] = ChartData("credit_dist", "box", "Credit Amount Distribution", fig_cred)

        # Loan Amount Distribution (Histogram)
        # Using AMT_CREDIT if LOAN_AMOUNT is not available, as they often mean the same in some datasets
        loan_col = self._get_col(df, ["LOAN_AMOUNT", "AMT_CREDIT"])
        if loan_col:
            fig_loan = self.chart_factory.create_histogram(df, x=loan_col, title='Loan Amount Distribution', nbins=40)
            section.charts["loan_dist"] = ChartData("loan_dist", "histogram", "Loan Amount Distribution", fig_loan)

        # Annuity Distribution (Histogram)
        annuity_col = self._get_col(df, ["AMT_ANNUITY", "ANNUITY"])
        if annuity_col:
            q_high = df[annuity_col].quantile(0.95)
            plot_df = df[df[annuity_col] < q_high]
            fig_ann = self.chart_factory.create_histogram(plot_df, x=annuity_col, title='Annuity Distribution (Bottom 95%)', nbins=50)
            section.charts["annuity_dist"] = ChartData("annuity_dist", "histogram", "Annuity Distribution", fig_ann)

    def _add_bivariate_analysis(self, df: pd.DataFrame, section: DashboardSection):
        """Add Income vs Credit and Income vs Annuity."""
        inc_col = self._get_col(df, ["AMT_INCOME_TOTAL", "INCOME"])
        cred_col = self._get_col(df, ["AMT_CREDIT", "CREDIT_AMOUNT"])
        annuity_col = self._get_col(df, ["AMT_ANNUITY", "ANNUITY"])
        
        # Sample data to avoid overplotting if large
        plot_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df.copy()
        
        if inc_col and cred_col:
            fig_inc_cred = self.chart_factory.create_scatter_plot(plot_df, x=inc_col, y=cred_col, title='Income vs Credit Amount')
            section.charts["income_vs_credit"] = ChartData("income_vs_credit", "scatter", "Income vs Credit Amount", fig_inc_cred)
            
        if inc_col and annuity_col:
            fig_inc_ann = self.chart_factory.create_scatter_plot(plot_df, x=inc_col, y=annuity_col, title='Income vs Annuity')
            section.charts["income_vs_annuity"] = ChartData("income_vs_annuity", "scatter", "Income vs Annuity", fig_inc_ann)

    def _add_default_analysis(self, df: pd.DataFrame, section: DashboardSection):
        """Add Credit vs Default, Income vs Default, Loan Amount vs Default."""
        if self.target_col not in df.columns:
            return
            
        target = self.target_col
        # Convert target to string for categorical plotting
        df_plot = df.copy()
        df_plot[target] = df_plot[target].astype(str)
        
        inc_col = self._get_col(df_plot, ["AMT_INCOME_TOTAL", "INCOME"])
        cred_col = self._get_col(df_plot, ["AMT_CREDIT", "CREDIT_AMOUNT"])
        loan_col = self._get_col(df_plot, ["LOAN_AMOUNT", "AMT_CREDIT"])
        
        if cred_col:
            fig_cred_def = self.chart_factory.create_box_plot(df_plot, x=target, y=cred_col, color=target, title='Credit Amount vs Default')
            section.charts["credit_vs_default"] = ChartData("credit_vs_default", "box", "Credit Amount vs Default", fig_cred_def)
            
        if inc_col:
            # Filter income for violin plot to look nice
            q_high = df_plot[inc_col].quantile(0.95)
            filtered_df = df_plot[df_plot[inc_col] < q_high]
            fig_inc_def = self.chart_factory.create_violin_plot(filtered_df, x=target, y=inc_col, color=target, title='Income vs Default (Violin Plot)')
            section.charts["income_vs_default"] = ChartData("income_vs_default", "violin", "Income vs Default", fig_inc_def)
            
        if loan_col:
            fig_loan_def = self.chart_factory.create_box_plot(df_plot, x=target, y=loan_col, color=target, title='Loan Amount vs Default')
            section.charts["loan_vs_default"] = ChartData("loan_vs_default", "box", "Loan Amount vs Default", fig_loan_def)
