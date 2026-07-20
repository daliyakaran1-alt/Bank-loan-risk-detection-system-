"""
Data Profiling Report.

Defines the data structures to hold the results of the profiling process.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class ProfilingReport:
    """Encapsulates the results of a data profiling run."""
    
    dataset_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Base summaries
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_issues: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced summaries
    advanced_statistics: Dict[str, Any] = field(default_factory=dict)
    outlier_summary: Dict[str, Any] = field(default_factory=dict)
    correlation_summary: Dict[str, Any] = field(default_factory=dict)
    distribution_summary: Dict[str, Any] = field(default_factory=dict)
    cardinality_summary: Dict[str, Any] = field(default_factory=dict)
    target_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Actionable insights
    business_recommendations: List[str] = field(default_factory=list)
    
    # Scores
    ml_readiness_score: float = 0.0
    data_quality_score: float = 0.0
    overall_health_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "quality_issues": self.quality_issues,
            "advanced_statistics": self.advanced_statistics,
            "outlier_summary": self.outlier_summary,
            "correlation_summary": self.correlation_summary,
            "distribution_summary": self.distribution_summary,
            "cardinality_summary": self.cardinality_summary,
            "target_analysis": self.target_analysis,
            "business_recommendations": self.business_recommendations,
            "ml_readiness_score": self.ml_readiness_score,
            "data_quality_score": self.data_quality_score,
            "overall_health_score": self.overall_health_score
        }
