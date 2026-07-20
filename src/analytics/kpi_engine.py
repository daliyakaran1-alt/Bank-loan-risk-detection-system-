"""
KPI Engine Module.

Calculates reusable enterprise Key Performance Indicators.
"""

import logging
import pandas as pd
from typing import Dict, Optional

from src.analytics.analytics_models import KPI
from src.analytics.analytics_utils import safe_divide, safe_mean, safe_median, format_currency, format_percentage

logger = logging.getLogger(__name__)


class KPIEngine:
    """Computes various Key Performance Indicators for the dataset."""

    def __init__(self, target_col: str = "TARGET"):
        self.target_col = target_col

    def _get_col_or_none(self, df: pd.DataFrame, possible_names: list) -> Optional[str]:
        """Find the first matching column name from a list of possibilities."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def calculate_all_kpis(self, df: pd.DataFrame) -> Dict[str, KPI]:
        """Calculate all available KPIs and return them in a dictionary."""
        kpis = {}
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to KPIEngine.")
            return kpis

        try:
            kpis["total_applications"] = self.calc_total_applications(df)
            kpis["total_customers"] = self.calc_total_customers(df)
            kpis["default_rate"] = self.calc_default_rate(df)
            
            kpis["avg_credit_amount"] = self.calc_avg_credit_amount(df)
            kpis["median_credit"] = self.calc_median_credit(df)
            kpis["avg_income"] = self.calc_avg_income(df)
            kpis["median_income"] = self.calc_median_income(df)
            kpis["avg_annuity"] = self.calc_avg_annuity(df)
            
            kpis["avg_employment_years"] = self.calc_avg_employment_years(df)
            kpis["avg_family_members"] = self.calc_avg_family_members(df)
            kpis["avg_children"] = self.calc_avg_children(df)
            
            kpis["male_percentage"] = self.calc_male_percentage(df)
            kpis["female_percentage"] = self.calc_female_percentage(df)
            
            kpis["home_ownership_percentage"] = self.calc_home_ownership_percentage(df)
            kpis["car_ownership_percentage"] = self.calc_car_ownership_percentage(df)
            kpis["working_population_percentage"] = self.calc_working_population_percentage(df)
            
            kpis["loan_portfolio_size"] = self.calc_loan_portfolio_size(df)
            kpis["risk_index"] = self.calc_risk_index(df)
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")

        # Filter out None values that might have failed to compute due to missing columns
        return {k: v for k, v in kpis.items() if v is not None}

    def calc_total_applications(self, df: pd.DataFrame) -> KPI:
        val = len(df)
        return KPI(name="Total Applications", value=val, formatted_value=f"{val:,}", is_good=True)

    def calc_total_customers(self, df: pd.DataFrame) -> KPI:
        id_col = self._get_col_or_none(df, ["SK_ID_CURR", "CLIENT_ID", "ID"])
        val = df[id_col].nunique() if id_col else len(df)
        return KPI(name="Total Customers", value=val, formatted_value=f"{val:,}", is_good=True)

    def calc_default_rate(self, df: pd.DataFrame) -> Optional[KPI]:
        if self.target_col not in df.columns:
            return None
        val = safe_mean(df[self.target_col])
        return KPI(name="Default Rate", value=val, formatted_value=format_percentage(val), is_good=False)

    def calc_avg_credit_amount(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_CREDIT", "CREDIT_AMOUNT", "LOAN_AMOUNT"])
        if not col: return None
        val = safe_mean(df[col])
        return KPI(name="Average Credit Amount", value=val, formatted_value=format_currency(val))

    def calc_median_credit(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_CREDIT", "CREDIT_AMOUNT", "LOAN_AMOUNT"])
        if not col: return None
        val = safe_median(df[col])
        return KPI(name="Median Credit Amount", value=val, formatted_value=format_currency(val))

    def calc_avg_income(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_INCOME_TOTAL", "INCOME", "ANNUAL_INCOME"])
        if not col: return None
        val = safe_mean(df[col])
        return KPI(name="Average Income", value=val, formatted_value=format_currency(val), is_good=True)

    def calc_median_income(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_INCOME_TOTAL", "INCOME", "ANNUAL_INCOME"])
        if not col: return None
        val = safe_median(df[col])
        return KPI(name="Median Income", value=val, formatted_value=format_currency(val))

    def calc_avg_annuity(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_ANNUITY", "ANNUITY"])
        if not col: return None
        val = safe_mean(df[col])
        return KPI(name="Average Annuity", value=val, formatted_value=format_currency(val))

    def calc_avg_employment_years(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["DAYS_EMPLOYED", "EMPLOYMENT_DAYS"])
        if not col: return None
        # Convert days to years (assuming days are negative in some datasets like Home Credit)
        mean_days = df[col].replace(365243, pd.NA).mean() # Handle 365243 anomaly if present
        if pd.isna(mean_days):
            return None
        val = abs(mean_days) / 365.25
        return KPI(name="Avg Employment Years", value=val, formatted_value=f"{val:.1f} Yrs")

    def calc_avg_family_members(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["CNT_FAM_MEMBERS", "FAMILY_MEMBERS"])
        if not col: return None
        val = safe_mean(df[col])
        return KPI(name="Avg Family Members", value=val, formatted_value=f"{val:.1f}")

    def calc_avg_children(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["CNT_CHILDREN", "CHILDREN"])
        if not col: return None
        val = safe_mean(df[col])
        return KPI(name="Avg Children", value=val, formatted_value=f"{val:.1f}")

    def calc_male_percentage(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["CODE_GENDER", "GENDER"])
        if not col: return None
        val = (df[col] == 'M').mean() if 'M' in df[col].values else (df[col] == 'Male').mean()
        return KPI(name="Male Percentage", value=val, formatted_value=format_percentage(val))

    def calc_female_percentage(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["CODE_GENDER", "GENDER"])
        if not col: return None
        val = (df[col] == 'F').mean() if 'F' in df[col].values else (df[col] == 'Female').mean()
        return KPI(name="Female Percentage", value=val, formatted_value=format_percentage(val))

    def calc_home_ownership_percentage(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["FLAG_OWN_REALTY", "OWN_HOME", "HOUSING_TYPE"])
        if not col: return None
        if col == "HOUSING_TYPE":
            val = (df[col] == 'House / apartment').mean()
        else:
            val = (df[col] == 'Y').mean() if 'Y' in df[col].values else (df[col] == 1).mean()
        return KPI(name="Home Ownership", value=val, formatted_value=format_percentage(val))

    def calc_car_ownership_percentage(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["FLAG_OWN_CAR", "OWN_CAR"])
        if not col: return None
        val = (df[col] == 'Y').mean() if 'Y' in df[col].values else (df[col] == 1).mean()
        return KPI(name="Car Ownership", value=val, formatted_value=format_percentage(val))

    def calc_working_population_percentage(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["NAME_INCOME_TYPE", "INCOME_TYPE"])
        if not col: return None
        val = df[col].isin(['Working', 'Commercial associate', 'State servant']).mean()
        return KPI(name="Working Population", value=val, formatted_value=format_percentage(val))

    def calc_loan_portfolio_size(self, df: pd.DataFrame) -> Optional[KPI]:
        col = self._get_col_or_none(df, ["AMT_CREDIT", "CREDIT_AMOUNT", "LOAN_AMOUNT"])
        if not col: return None
        val = df[col].sum()
        return KPI(name="Loan Portfolio Size", value=val, formatted_value=format_currency(val))

    def calc_risk_index(self, df: pd.DataFrame) -> Optional[KPI]:
        # A synthetic risk index (0 to 100) based on default rate. (e.g. default_rate * 1000 scaled)
        if self.target_col not in df.columns:
            return None
        default_rate = safe_mean(df[self.target_col])
        # simple heuristic
        val = min(100.0, default_rate * 1000) 
        return KPI(name="Risk Index", value=val, formatted_value=f"{val:.1f}/100", is_good=False)
