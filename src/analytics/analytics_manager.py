"""
Analytics Manager Module.

Main orchestration layer for the Enterprise Banking Analytics Engine.
Coordinates KPI extraction, insight generation, and domain-specific analytics.
"""

import logging
from datetime import datetime
import pandas as pd

from src.analytics.analytics_models import AnalyticsResult
from src.analytics.business_insights import BusinessInsightEngine
from src.analytics.executive_dashboard import ExecutiveDashboardAnalytics
from src.analytics.customer_analytics import CustomerAnalytics
from src.analytics.loan_analytics import LoanAnalytics
from src.analytics.risk_analytics import RiskAnalytics
from src.analytics.financial_analytics import FinancialAnalytics
from src.analytics.demographic_analytics import DemographicAnalytics

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Orchestrates all analytics modules."""

    def __init__(self, target_col: str = "TARGET"):
        self.target_col = target_col
        self.insight_engine = BusinessInsightEngine(target_col)
        self.executive_analytics = ExecutiveDashboardAnalytics(target_col)
        self.customer_analytics = CustomerAnalytics(target_col)
        self.loan_analytics = LoanAnalytics(target_col)
        self.risk_analytics = RiskAnalytics(target_col)
        self.financial_analytics = FinancialAnalytics(target_col)
        self.demographic_analytics = DemographicAnalytics(target_col)

    def run_analytics(self, df: pd.DataFrame, dataset_name: str = "Enterprise Dataset") -> AnalyticsResult:
        """
        Execute every analytics module and return a dashboard-ready object.
        """
        logger.info(f"Starting analytics run for {dataset_name} with {len(df)} rows.")
        start_time = datetime.now()

        result = AnalyticsResult(
            dataset_name=dataset_name,
            timestamp=start_time.isoformat()
        )

        try:
            # 1. Executive Metrics
            logger.info("Generating Executive Metrics...")
            result.executive_metrics = self.executive_analytics.calculate_kpis(df)
            result.sections["Executive Dashboard"] = self.executive_analytics.calculate_kpis(df)

            # 2. Business Insights
            logger.info("Generating Business Insights...")
            result.global_insights = self.insight_engine.generate_all_insights(df)

            # 3. Customer Analytics
            logger.info("Generating Customer Analytics...")
            result.sections["Customer Analytics"] = self.customer_analytics.generate_section(df)

            # 4. Loan Analytics
            logger.info("Generating Loan Analytics...")
            result.sections["Loan Analytics"] = self.loan_analytics.generate_section(df)

            # 5. Risk Analytics
            logger.info("Generating Risk Analytics...")
            result.sections["Risk Analytics"] = self.risk_analytics.generate_section(df)

            # 6. Financial Analytics
            logger.info("Generating Financial Analytics...")
            result.sections["Financial Analytics"] = self.financial_analytics.generate_section(df)

            # 7. Demographic Analytics
            logger.info("Generating Demographic Analytics...")
            result.sections["Demographic Analytics"] = self.demographic_analytics.generate_section(df)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully completed analytics run in {execution_time:.2f} seconds.")
            
        except Exception as e:
            logger.error(f"Analytics run failed: {e}")

        return result
