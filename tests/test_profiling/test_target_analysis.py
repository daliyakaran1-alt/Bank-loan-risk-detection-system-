"""Tests for Target Analysis."""

import pytest
import pandas as pd
from src.profiling.target_analysis import TargetAnalysis
from core.exceptions import ProfilingError

@pytest.fixture
def sample_df():
    """Fixture for sample DataFrame."""
    return pd.DataFrame({
        "TARGET": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 20% default rate
        "FEATURE": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

def test_target_analysis_success(sample_df):
    """Test successful target analysis."""
    analyzer = TargetAnalysis(target_column="TARGET")
    results = analyzer.analyze(sample_df)
    
    assert "distribution" in results
    assert "percentages" in results
    assert results["percentages"][1] == 20.0
    assert results["default_percentage"] == 20.0
    assert "is_balanced" in results
    assert results["is_balanced"] is False

def test_target_analysis_missing_column():
    """Test target analysis when column is missing."""
    analyzer = TargetAnalysis(target_column="NOT_THERE")
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ProfilingError, match="not found"):
        analyzer.analyze(df)

def test_target_analysis_error():
    """Test target analysis error handling."""
    analyzer = TargetAnalysis()
    with pytest.raises(ProfilingError):
        analyzer.analyze(None)
