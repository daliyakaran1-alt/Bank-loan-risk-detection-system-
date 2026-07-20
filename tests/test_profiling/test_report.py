"""
Tests for Data Profiling Report.
"""

from src.profiling.report import ProfilingReport

def test_profiling_report_creation():
    """Test creating a profiling report with default values."""
    report = ProfilingReport(dataset_name="test_dataset")
    
    assert report.dataset_name == "test_dataset"
    assert report.overall_health_score == 0.0
    assert report.metrics == {}
    assert report.quality_issues == {}
    assert report.timestamp is not None

def test_profiling_report_to_dict():
    """Test converting a profiling report to a dictionary."""
    report = ProfilingReport(
        dataset_name="test_dataset",
        overall_health_score=85.5,
        metrics={"row_count": 100},
        quality_issues={"missing_values": 5}
    )
    
    report_dict = report.to_dict()
    
    assert report_dict["dataset_name"] == "test_dataset"
    assert report_dict["overall_health_score"] == 85.5
    assert report_dict["metrics"] == {"row_count": 100}
    assert report_dict["quality_issues"] == {"missing_values": 5}
    assert "timestamp" in report_dict
