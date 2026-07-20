import pytest
import pandas as pd
import numpy as np
from src.profiling.distribution import DistributionAnalysis, DistributionResult
from core.exceptions import ProfilingError

def test_distribution_analysis():
    np.random.seed(42)
    df = pd.DataFrame({
        'normal_col': np.random.normal(0, 1, 1000),
        'skewed_col': np.random.exponential(1, 1000),
        'cat_col': ['A'] * 1000
    })
    
    analyzer = DistributionAnalysis(df)
    result = analyzer.analyze()
    
    assert isinstance(result, DistributionResult)
    assert 'normal_col' in result.column_details
    assert 'skewed_col' in result.column_details
    assert 'cat_col' not in result.column_details
    
    assert result.column_details['skewed_col'].skewness > 0.5

def test_distribution_analysis_empty_df():
    analyzer = DistributionAnalysis(pd.DataFrame())
    with pytest.raises(ProfilingError):
        analyzer.analyze()

def test_distribution_small_df():
    df = pd.DataFrame({'A': [1, 2, 3]})
    analyzer = DistributionAnalysis(df)
    result = analyzer.analyze()
    assert 'A' not in result.column_details
