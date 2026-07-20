"""
Business Insight Engine.

Dynamically generates natural language observations, interpretations, impacts, and recommendations from data.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

from src.analytics.analytics_models import BusinessInsight

logger = logging.getLogger(__name__)

class BusinessInsightEngine:
    """Generates dynamic, rule-based business insights."""

    def __init__(self, target_col: str = "TARGET"):
        self.target_col = target_col

    def generate_all_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate a complete list of insights from the dataset."""
        insights = []
        if df is None or df.empty:
            return insights

        try:
            self._generate_correlation_insights(df, insights)
            self._generate_default_risk_insights(df, insights)
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            
        return insights

    def _create_insight_obj(self, observation: str, interpretation: str, impact: str, recommendation: str, priority: str) -> BusinessInsight:
        content = f"**Observation**: {observation}\n\n**Interpretation**: {interpretation}\n\n**Impact**: {impact}\n\n**Recommendation**: {recommendation}"
        return BusinessInsight(insight_type="strategic", content=content, priority=priority)

    def _generate_correlation_insights(self, df: pd.DataFrame, insights: List[BusinessInsight]):
        """Dynamically find top correlations and generate insights."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or num_df.shape[1] < 2: return
        
        corr = num_df.corr().abs()
        # Get upper triangle
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        # Find top correlation
        stacked = upper.stack().reset_index()
        stacked.columns = ['Feature1', 'Feature2', 'Correlation']
        top_corr = stacked.sort_values(by='Correlation', ascending=False).head(1)
        
        if not top_corr.empty:
            f1, f2, c = top_corr.iloc[0]
            if c > 0.7:
                obs = f"A high correlation ({c:.2f}) exists between {f1} and {f2}."
                interp = f"Changes in {f1} are strongly mirrored by changes in {f2}, indicating multicollinearity."
                impact = "Keeping both features in a predictive model might inflate variance without adding new information."
                rec = f"Consider dropping or combining {f1} and {f2} in feature engineering phases."
                insights.append(self._create_insight_obj(obs, interp, impact, rec, "medium"))

    def _generate_default_risk_insights(self, df: pd.DataFrame, insights: List[BusinessInsight]):
        """Dynamically find highest risk segments."""
        if self.target_col not in df.columns: return
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols[:5]: # limit to top 5 categorical columns to avoid too many insights
            grouped = df.groupby(col)[self.target_col].agg(['mean', 'count'])
            grouped = grouped[grouped['count'] > 100] # statistically significant
            if grouped.empty: continue
            
            highest_risk_cat = grouped['mean'].idxmax()
            highest_risk_val = grouped['mean'].max()
            lowest_risk_cat = grouped['mean'].idxmin()
            
            overall_mean = df[self.target_col].mean()
            
            if highest_risk_val > overall_mean * 1.5:
                obs = f"Customers in the '{highest_risk_cat}' segment of {col} show significantly higher default rates ({highest_risk_val:.2%} vs overall {overall_mean:.2%})."
                interp = f"The {highest_risk_cat} demographic inherently carries higher repayment risk."
                impact = f"High exposure to this segment decreases overall loan portfolio health."
                rec = f"Introduce stricter underwriting and pricing adjustments for {col} = {highest_risk_cat}."
                insights.append(self._create_insight_obj(obs, interp, impact, rec, "high"))
