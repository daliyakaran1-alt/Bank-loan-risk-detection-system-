"""
Customer Analytics Module.

Generates Plotly visualizations for Customer Demographics.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class CustomerAnalytics:
    """Handles analytics related to customer demographics and metrics."""

    def __init__(self, target_col: str = "TARGET") -> None:
        """
        Initialize the Customer Analytics.

        Args:
            target_col (str): The name of the target column.
        """
        self.target_col = target_col

    def _get_col(self, df: pd.DataFrame, possibilities: List[str]) -> Optional[str]:
        """
        Find the first matching column in the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            possibilities (List[str]): Potential column names to search for.
            
        Returns:
            Optional[str]: The matched column name, or None if not found.
        """
        for p in possibilities:
            if p in df.columns:
                return p
        return None

    def generate_demographic_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Generate real Plotly visualizations for Customer Demographics.

        Args:
            df (pd.DataFrame): The input DataFrame containing customer data.

        Returns:
            Dict[str, go.Figure]: A dictionary mapping chart names to Plotly Figure objects.
        """
        charts: Dict[str, go.Figure] = {}

        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for demographic charting.")
            return charts

        # 1. Gender Distribution
        col = self._get_col(df, ["CODE_GENDER", "GENDER"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Gender", "Count"]
            fig = px.bar(
                counts, 
                x="Gender", 
                y="Count", 
                title="Gender Distribution", 
                color="Gender",
                template="plotly_white"
            )
            charts["gender_distribution"] = fig

        # 2. Age Distribution
        col = self._get_col(df, ["DAYS_BIRTH", "AGE"])
        if col:
            if col == "DAYS_BIRTH":
                # Convert negative days to years
                age_series = (df[col].abs() / 365.25).dropna()
            else:
                age_series = df[col].dropna()
            
            fig = px.histogram(
                x=age_series, 
                nbins=30, 
                title="Age Distribution",
                labels={"x": "Age (Years)", "y": "Count"},
                template="plotly_white"
            )
            charts["age_distribution"] = fig

        # 3. Family Status
        col = self._get_col(df, ["NAME_FAMILY_STATUS", "FAMILY_STATUS"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Family Status", "Count"]
            fig = px.pie(
                counts, 
                names="Family Status", 
                values="Count", 
                title="Family Status Distribution", 
                hole=0.4,
                template="plotly_white"
            )
            charts["family_status"] = fig

        # 4. Children Distribution
        col = self._get_col(df, ["CNT_CHILDREN", "CHILDREN"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Number of Children", "Count"]
            fig = px.bar(
                counts, 
                x="Number of Children", 
                y="Count", 
                title="Children Distribution",
                template="plotly_white"
            )
            fig.update_xaxes(type="category")
            charts["children_distribution"] = fig

        # 5. Education Level
        col = self._get_col(df, ["NAME_EDUCATION_TYPE", "EDUCATION"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Education Level", "Count"]
            fig = px.bar(
                counts, 
                x="Count", 
                y="Education Level", 
                orientation="h", 
                title="Education Level",
                template="plotly_white"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            charts["education_level"] = fig

        # 6. Occupation
        col = self._get_col(df, ["OCCUPATION_TYPE", "OCCUPATION"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Occupation", "Count"]
            fig = px.bar(
                counts, 
                x="Count", 
                y="Occupation", 
                orientation="h", 
                title="Occupation Distribution",
                template="plotly_white"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            charts["occupation"] = fig

        # 7. Organization Type
        col = self._get_col(df, ["ORGANIZATION_TYPE", "ORGANIZATION"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Organization Type", "Count"]
            fig = px.bar(
                counts, 
                x="Count", 
                y="Organization Type", 
                orientation="h", 
                title="Organization Type",
                template="plotly_white"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            charts["organization_type"] = fig

        # 8. Housing Type
        col = self._get_col(df, ["NAME_HOUSING_TYPE", "HOUSING_TYPE"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Housing Type", "Count"]
            fig = px.pie(
                counts, 
                names="Housing Type", 
                values="Count", 
                title="Housing Type",
                template="plotly_white"
            )
            charts["housing_type"] = fig

        # 9. Car Ownership
        col = self._get_col(df, ["FLAG_OWN_CAR", "OWN_CAR"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Car Ownership", "Count"]
            fig = px.pie(
                counts, 
                names="Car Ownership", 
                values="Count", 
                title="Car Ownership",
                template="plotly_white"
            )
            charts["car_ownership"] = fig

        # 10. Real Estate Ownership
        col = self._get_col(df, ["FLAG_OWN_REALTY", "OWN_REALTY"])
        if col:
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Real Estate Ownership", "Count"]
            fig = px.pie(
                counts, 
                names="Real Estate Ownership", 
                values="Count", 
                title="Real Estate Ownership",
                template="plotly_white"
            )
            charts["real_estate_ownership"] = fig

        return charts
