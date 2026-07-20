"""
Loan Analytics Module.

Generates analytics for loan portfolio characteristics and segmentations.
"""

import logging
import pandas as pd
import numpy as np

from src.analytics.analytics_models import DashboardSection, ChartData
from src.analytics.chart_factory import ChartFactory
from src.analytics.analytics_utils import bin_numerical_column

logger = logging.getLogger(__name__)

class LoanAnalytics:
    """Handles analytics related to loan portfolios."""
    
    def __init__(self, target_col: str = "TARGET"):
        """
        Initialize LoanAnalytics.
        
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
        Generate loan analytics section containing Plotly visualizations.
        
        Args:
            df (pd.DataFrame): The input dataset.
            
        Returns:
            DashboardSection: A dashboard section containing charts.
        """
        section = DashboardSection(section_name="Loan Analytics")
        if df is None or df.empty:
            return section

        self._add_portfolio_composition(df, section)
        self._add_segments(df, section)
        
        return section
        
    def _add_portfolio_composition(self, df: pd.DataFrame, section: DashboardSection):
        """Add Portfolio Composition by Loan Type and Contract Type."""
        # Contract Type (Treemap)
        contract_col = self._get_col(df, ["NAME_CONTRACT_TYPE", "CONTRACT_TYPE"])
        if contract_col:
            counts = df[contract_col].value_counts().reset_index()
            counts.columns = ['Contract Type', 'Count']
            counts['Portfolio'] = 'Loan Portfolio'
            fig_contract = self.chart_factory.create_treemap(
                counts, path=['Portfolio', 'Contract Type'], values='Count', 
                title='Portfolio Composition by Contract Type'
            )
            section.charts["contract_type_composition"] = ChartData(
                "contract_type_composition", "treemap", "Contract Type Composition", fig_contract
            )

        # Loan Type (Sunburst)
        loan_type_col = self._get_col(df, ["NAME_TYPE_SUITE", "LOAN_TYPE", "NAME_HOUSING_TYPE"])
        if loan_type_col:
            counts = df[loan_type_col].value_counts().reset_index()
            counts.columns = ['Loan Type', 'Count']
            counts['Portfolio'] = 'Loan Portfolio'
            fig_loan = self.chart_factory.create_sunburst(
                counts, path=['Portfolio', 'Loan Type'], values='Count', 
                title='Portfolio Composition by Loan Type'
            )
            section.charts["loan_type_composition"] = ChartData(
                "loan_type_composition", "sunburst", "Loan Type Composition", fig_loan
            )

    def _add_segments(self, df: pd.DataFrame, section: DashboardSection):
        """Add Customer, Income, Credit, and Risk segments."""
        # Customer Segment (Treemap/Sunburst depending on data, using Pie/Treemap)
        cust_col = self._get_col(df, ["NAME_FAMILY_STATUS", "CUSTOMER_SEGMENT", "NAME_EDUCATION_TYPE"])
        if cust_col:
            counts = df[cust_col].value_counts().reset_index()
            counts.columns = ['Customer Segment', 'Count']
            counts['Portfolio'] = 'Loan Portfolio'
            fig_cust = self.chart_factory.create_treemap(
                counts, path=['Portfolio', 'Customer Segment'], values='Count', 
                title='Portfolio by Customer Segment'
            )
            section.charts["customer_segment"] = ChartData("customer_segment", "treemap", "Customer Segments", fig_cust)

        # Income Segment (Stacked Bar Chart)
        inc_col = self._get_col(df, ["AMT_INCOME_TOTAL", "INCOME"])
        if inc_col:
            df_plot = df.copy()
            # Calculate quartiles for segmentation if possible, else 4 bins
            try:
                df_plot['Income Segment'] = pd.qcut(df_plot[inc_col], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            except ValueError:
                # Fallback to cut if qcut fails due to duplicate edges
                df_plot['Income Segment'] = pd.cut(df_plot[inc_col], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
                
            if self.target_col in df_plot.columns:
                counts = df_plot.groupby(['Income Segment', self.target_col], observed=True).size().reset_index(name='Count')
                counts[self.target_col] = counts[self.target_col].astype(str)
                fig_inc = self.chart_factory.create_bar_chart(
                    counts, x='Income Segment', y='Count', color=self.target_col, 
                    title='Income Segments by Default Risk'
                )
                fig_inc.update_layout(barmode='stack')
            else:
                counts = df_plot['Income Segment'].value_counts().reset_index()
                counts.columns = ['Income Segment', 'Count']
                fig_inc = self.chart_factory.create_bar_chart(
                    counts, x='Income Segment', y='Count', title='Income Segments'
                )
            
            section.charts["income_segment"] = ChartData("income_segment", "bar", "Income Segments", fig_inc)

        # Credit Segment (Stacked Bar Chart)
        cred_col = self._get_col(df, ["AMT_CREDIT", "CREDIT_AMOUNT"])
        if cred_col:
            df_plot = df.copy()
            try:
                df_plot['Credit Segment'] = pd.qcut(df_plot[cred_col], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'], duplicates='drop')
            except ValueError:
                df_plot['Credit Segment'] = pd.cut(df_plot[cred_col], bins=4, labels=['Small', 'Medium', 'Large', 'Very Large'])
                
            if self.target_col in df_plot.columns:
                counts = df_plot.groupby(['Credit Segment', self.target_col], observed=True).size().reset_index(name='Count')
                counts[self.target_col] = counts[self.target_col].astype(str)
                fig_cred = self.chart_factory.create_bar_chart(
                    counts, x='Credit Segment', y='Count', color=self.target_col, 
                    title='Credit Segments by Default Risk'
                )
                fig_cred.update_layout(barmode='stack')
            else:
                counts = df_plot['Credit Segment'].value_counts().reset_index()
                counts.columns = ['Credit Segment', 'Count']
                fig_cred = self.chart_factory.create_bar_chart(
                    counts, x='Credit Segment', y='Count', title='Credit Segments'
                )
                
            section.charts["credit_segment"] = ChartData("credit_segment", "bar", "Credit Segments", fig_cred)

        # Risk Segment (Treemap)
        risk_col = self._get_col(df, ["REGION_RATING_CLIENT_W_CITY", "RISK_SEGMENT", "RATING", "REGION_RATING_CLIENT"])
        if risk_col:
            counts = df[risk_col].astype(str).value_counts().reset_index()
            counts.columns = ['Risk Segment', 'Count']
            counts['Portfolio'] = 'Loan Portfolio'
            fig_risk = self.chart_factory.create_treemap(
                counts, path=['Portfolio', 'Risk Segment'], values='Count', 
                title='Portfolio by Risk Segments'
            )
            section.charts["risk_segment"] = ChartData("risk_segment", "treemap", "Risk Segments", fig_risk)
