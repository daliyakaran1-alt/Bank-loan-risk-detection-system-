"""
Executive Dashboard Analytics Module.

Calculates high-level Executive KPIs from the dataset.
"""

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class ExecutiveDashboardAnalytics:
    """Handles analytics for the Executive Dashboard."""

    def __init__(self, target_col: str = "TARGET") -> None:
        """
        Initialize the Executive Dashboard Analytics.

        Args:
            target_col (str): The name of the target column indicating default.
        """
        self.target_col = target_col

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate real Executive KPIs from a given pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing loan application data.

        Returns:
            Dict[str, Any]: A structured dictionary containing calculated KPIs.
        """
        kpis: Dict[str, Any] = {}

        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for KPI calculation.")
            return kpis

        # 1. Total Customers
        if "SK_ID_CURR" in df.columns:
            kpis["total_customers"] = int(df["SK_ID_CURR"].nunique())
        else:
            kpis["total_customers"] = len(df)

        # 2. Total Loan Applications
        kpis["total_loan_applications"] = len(df)

        # 3. Avg Loan Amount & 5. Avg Credit
        if "AMT_CREDIT" in df.columns:
            avg_credit = float(df["AMT_CREDIT"].mean())
            kpis["avg_loan_amount"] = avg_credit
            kpis["avg_credit"] = avg_credit
        else:
            kpis["avg_loan_amount"] = 0.0
            kpis["avg_credit"] = 0.0

        # 4. Avg Income
        if "AMT_INCOME_TOTAL" in df.columns:
            kpis["avg_income"] = float(df["AMT_INCOME_TOTAL"].mean())
        else:
            kpis["avg_income"] = 0.0

        # 6. Avg Annuity
        if "AMT_ANNUITY" in df.columns:
            kpis["avg_annuity"] = float(df["AMT_ANNUITY"].mean())
        else:
            kpis["avg_annuity"] = 0.0

        # 7. Avg Family Size
        if "CNT_FAM_MEMBERS" in df.columns:
            kpis["avg_family_size"] = float(df["CNT_FAM_MEMBERS"].mean())
        else:
            kpis["avg_family_size"] = 0.0

        # 8 & 9. Default Rate and Repayment Rate
        if self.target_col in df.columns:
            default_rate = float(df[self.target_col].mean())
            kpis["default_rate"] = default_rate
            kpis["repayment_rate"] = 1.0 - default_rate
        else:
            kpis["default_rate"] = 0.0
            kpis["repayment_rate"] = 0.0

        # 10. Loan Approval Population
        # Typically the entire dataset represents the applications (approved or otherwise)
        kpis["loan_approval_population"] = len(df)

        return kpis
