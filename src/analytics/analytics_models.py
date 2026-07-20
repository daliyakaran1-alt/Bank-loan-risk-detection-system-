"""
Data models for the Enterprise Banking Analytics Engine.

Contains standard output structures for KPIs, Charts, Insights, and complete Analytics Results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class KPI:
    """Represents a single Key Performance Indicator."""
    name: str
    value: Any
    formatted_value: str
    description: Optional[str] = None
    trend: Optional[str] = None  # e.g., "up", "down", "stable"
    is_good: Optional[bool] = None  # None means neutral


@dataclass
class ChartData:
    """Represents chart configuration and data for UI rendering."""
    chart_id: str
    chart_type: str  # e.g., "bar", "line", "pie"
    title: str
    figure: Any  # Plotly Figure object
    description: Optional[str] = None


@dataclass
class BusinessInsight:
    """Represents a dynamically generated business insight or recommendation."""
    insight_type: str  # e.g., "observation", "warning", "recommendation", "opportunity"
    content: str
    priority: str = "medium"  # "low", "medium", "high"


@dataclass
class DashboardSection:
    """A logical grouping of KPIs, charts, and insights for a specific dashboard section."""
    section_name: str
    kpis: Dict[str, KPI] = field(default_factory=dict)
    charts: Dict[str, ChartData] = field(default_factory=dict)
    insights: List[BusinessInsight] = field(default_factory=list)


@dataclass
class ExecutiveMetrics:
    """High-level metrics intended for the executive dashboard."""
    portfolio_health_score: float
    approval_rate: float
    default_rate: float
    total_applications: int
    total_customers: int
    average_credit_amount: float
    average_income: float
    average_loan_amount: float


@dataclass
class AnalyticsResult:
    """The complete result of the analytics engine run."""
    dataset_name: str
    executive_metrics: Optional[ExecutiveMetrics] = None
    sections: Dict[str, DashboardSection] = field(default_factory=dict)
    global_insights: List[BusinessInsight] = field(default_factory=list)
    timestamp: str = ""
    
    def get_section(self, section_name: str) -> DashboardSection:
        """Helper to get or create a section."""
        if section_name not in self.sections:
            self.sections[section_name] = DashboardSection(section_name=section_name)
        return self.sections[section_name]
