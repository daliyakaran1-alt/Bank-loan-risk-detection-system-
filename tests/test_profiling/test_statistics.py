"""
Tests for DatasetStatistics.
"""
import pytest
import pandas as pd
from profiling.statistics import DatasetStatistics, StatisticsResult
from core.exceptions import ProfilingError

def test_dataset_statistics_basic():
    """Test DatasetStatistics calculates metrics correctly on a basic dataset."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    stats = DatasetStatistics(df)
    result = stats.calculate()
    
    assert isinstance(result, StatisticsResult)
    assert result.num_rows == 3
    assert result.num_columns == 2
    assert result.numeric_counts['a'] == 3
    assert result.unique_values['a'] == 3

def test_dataset_statistics_empty():
    """Test DatasetStatistics handles empty datasets."""
    df = pd.DataFrame()
    stats = DatasetStatistics(df)
    result = stats.calculate()
    
    assert result.num_rows == 0
    assert result.num_columns == 0

def test_dataset_statistics_error_handling():
    """Test DatasetStatistics raises ProfilingError on failure."""
    class BadDataFrame:
        @property
        def shape(self):
            raise ValueError("Intentional error")
            
    stats = DatasetStatistics(BadDataFrame()) # type: ignore
    with pytest.raises(ProfilingError):
        stats.calculate()
