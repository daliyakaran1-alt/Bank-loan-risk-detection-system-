"""Tests for Correlation Analysis."""

import pytest
import pandas as pd
from src.profiling.correlation import CorrelationAnalysis
from core.exceptions import ProfilingError

@pytest.fixture
def sample_df():
    """Fixture for sample DataFrame."""
    # A and B are perfectly correlated
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [1, 0, 1, 0, 1]
    })

def test_correlation_analysis_success(sample_df):
    """Test successful correlation analysis."""
    analyzer = CorrelationAnalysis(correlation_threshold=0.9)
    results = analyzer.analyze(sample_df)
    
    assert "pearson_matrix" in results
    assert "spearman_matrix" in results
    assert "highly_correlated_features" in results
    
    # Check if A and B are marked as highly correlated
    pairs = results["highly_correlated_features"]
    assert len(pairs) >= 1
    found = any(
        (p[0] == "A" and p[1] == "B") or 
        (p[0] == "B" and p[1] == "A") 
        for p in pairs
    )
    assert found

def test_correlation_analysis_error():
    """Test correlation analysis error handling."""
    analyzer = CorrelationAnalysis()
    with pytest.raises(ProfilingError):
        analyzer.analyze(None)
