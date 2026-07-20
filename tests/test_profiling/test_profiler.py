"""
Tests for Data Profiler Orchestrator.
"""

import pytest
import pandas as pd

from src.profiling.profiler import DataProfiler
from src.profiling.report import ProfilingReport
from core.exceptions import ProfilingError

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

def test_data_profiler_initialization():
    """Test that DataProfiler initializes correctly."""
    profiler = DataProfiler()
    assert isinstance(profiler, DataProfiler)

def test_data_profiler_profile_success(sample_dataframe):
    """Test profiling a DataFrame successfully."""
    profiler = DataProfiler()
    report = profiler.profile(df=sample_dataframe, dataset_name="test_dataset")
    
    assert isinstance(report, ProfilingReport)
    assert report.dataset_name == "test_dataset"
    assert report.overall_health_score == 100.0
    assert report.metrics["row_count"] == 3
    assert report.metrics["column_count"] == 2

def test_data_profiler_profile_error():
    """Test profiling failure raises ProfilingError."""
    profiler = DataProfiler()
    
    # Simulate an error by passing None instead of DataFrame
    with pytest.raises(ProfilingError) as exc_info:
        profiler.profile(df=None, dataset_name="error_dataset")  # type: ignore
        
    assert "Failed to profile dataset" in str(exc_info.value)
    assert "error_dataset" in str(exc_info.value)
